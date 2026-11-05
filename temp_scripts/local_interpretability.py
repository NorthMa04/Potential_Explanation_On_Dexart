"""Inspectable local follow-up. No policy/SAE training, test loading, or simulation.

New probes are explicitly exploratory train/validation fits. Saved physical
branches are observations, not newly simulated counterfactuals.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import warnings
from datetime import datetime, timezone, timedelta

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.metrics import average_precision_score, log_loss
from threadpoolctl import threadpool_limits

from temp_scripts.project_paths import PROJECT_ROOT, output_path, default_index
from temp_scripts.activation_baseline import sample_weights, coordinate_ranking
from temp_scripts.data_access import DataRepository
from temp_scripts.physical_runtime import NumericSAE, actor_tail
from temp_scripts.state_assisted_controls import transform_supervised_rotation
from temp_scripts.constrained_intervention import project_delta, match_projected_norm, stable_random_directions
from temp_scripts.state_assisted_intervention import weighted_quantile

STUDY = PROJECT_ROOT / 'outputs/state_assisted/20261102_122123_738477'
PHYSICAL = PROJECT_ROOT / 'outputs/combined_physical/20261103_140606_955714'
TARGETS = ['distance_m', 'opening_rad', 'any_contact']
SIZES = [1, 2, 4, 6]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def weighted_r(x, y, w):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x=x-np.average(x,weights=w); y=y-np.average(y,weights=w)
    denominator = np.sqrt(np.average(x*x,weights=w)*np.average(y*y,weights=w))
    return float(np.average(x*y,weights=w)/denominator) if denominator>1e-12 else None


def infer(model, h):
    zs, rs = [], []
    for start in range(0, len(h), 256):
        z,r,_ = model.infer(h[start:start+256]); zs.append(z); rs.append(r)
    return np.concatenate(zs), np.concatenate(rs)


def source_hashes():
    paths = [STUDY/'selected_arrays.npz', STUDY/'selected_frames.csv', PHYSICAL/'protocol_hashes.json',
             PHYSICAL/'bundle/actor.npz', PHYSICAL/'bundle/combined_intervention.json']
    for kind,configuration in [('pure','dense_l0p0'),('assisted','slots_s0.3_a1.0')]:
        for seed in range(3):
            paths += [STUDY/f'{configuration}_seed{seed}'/x for x in ['best_model.pt','coordinate_ranking.csv','fitted_probes.joblib']]
            paths += [PHYSICAL/f'bundle/models/{kind}_seed{seed}'/x for x in ['weights.npz','model.json']]
    paths.append(STUDY/'controls/ridge_rotation_10p0_coordinate/projection.npz')
    return {str(p.relative_to(PROJECT_ROOT)): sha(p) for p in paths}


def load_development(out):
    repo = DataRepository(default_index())
    records = repo.filter(('train','validation'))
    frames=[]; hs=[]; ys=[]; coverage=[]
    with np.load(STUDY/'selected_arrays.npz') as a:
        local = {k:a[k].copy() for k in a.files}
    local_keys = set(zip(local['trajectory_id'].tolist(), local['state_index'].tolist()))
    for meta in records:
        t=repo.load(meta['trajectory_id']); n=t.n
        h=t.arrays['activations']
        opening=t.opening[:n]
        indices=np.flatnonzero(opening >= .05)
        onset=int(indices[0]) if len(indices) else None
        distance=t.state('palm_handle_distance')[:n]
        contact=t.state('finger_palm_contact')[:n].any(axis=1).astype(float)
        hs.append(h); ys.append(np.column_stack([distance,opening,contact]))
        success=meta['trajectory_success']=='True'
        coverage.append(dict(trajectory=t.tid,split=meta['split'],object=int(meta['object_id']),
            natural_success=success,steps=n,onset=onset,post_onset_steps=n-onset if onset is not None else 0))
        for j in range(n):
            frames.append(dict(trajectory=t.tid,split=meta['split'],object=int(meta['object_id']),state_index=j,
                time=j*t.dt,local_window=(t.tid,j) in local_keys,
                post_onset=onset is not None and j>=onset,natural_success=success))
    f=pd.DataFrame(frames); h=np.concatenate(hs).astype(float); y=np.concatenate(ys)
    assert not f.duplicated(['trajectory','state_index']).any()
    assert set(f.split)=={'train','validation'}
    assert len(f)==len(h)==len(y) and np.isfinite(h).all() and np.isfinite(y).all()
    # Verify original window alignment from source, not only manifest claims.
    lookup={(r.trajectory,int(r.state_index)):i for i,r in f.iterrows()}
    positions=np.array([lookup[(str(t),int(s))] for t,s in zip(local['trajectory_id'],local['state_index'])])
    np.testing.assert_array_equal(h[positions],local['h'])
    np.testing.assert_allclose(y[positions],local['y'],atol=1e-12,rtol=0)
    f.to_csv(out/'development_frames.csv',index=False,encoding='utf-8-sig')
    save_csv(out/'development_coverage.csv',coverage)
    np.savez_compressed(out/'development_arrays.npz',h=h,y=y,local_positions=positions)
    return f,h,y,local,positions


def regression_metrics(y,p,w):
    return dict(mae=float(np.average(abs(y-p),weights=w)),rmse=float(np.sqrt(np.average((y-p)**2,weights=w))))


def fit_readout(x,y,train,val,groups,classification):
    # Consistent float64 fitting/standardization across raw, sparse and residual
    # representations; avoids dtype-dependent sample-weight/transform rounding.
    x=np.asarray(x,dtype=np.float64)
    w=sample_weights(groups[train]); vw=sample_weights(groups[val])
    scaler=StandardScaler().fit(x[train],sample_weight=w)
    x=scaler.transform(x)
    ym=float(np.average(y[train],weights=w)); ys=float(np.sqrt(np.average((y[train]-ym)**2,weights=w)))
    if classification: ym=0.;ys=1.
    ys=max(ys,1e-6)
    candidates=[]; notices=[]
    if classification and len(np.unique(y[train]))<2:
        return dict(constant=float(y[train][0]),scaler=scaler,y_mean=ym,y_scale=ys),np.full(len(y),float(y[train][0])),0.,[]
    for value in [.1,1.,10.]:
        model=LogisticRegression(C=value,max_iter=2000) if classification else Ridge(alpha=value)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(x[train],y[train] if classification else (y[train]-ym)/ys,sample_weight=w)
        pred=model.predict_proba(x)[:,list(model.classes_).index(1)] if classification else model.predict(x)*ys+ym
        score=float(log_loss(y[val],np.column_stack([1-pred[val],pred[val]]),labels=[0,1],sample_weight=vw)) if classification else float(np.average(abs(y[val]-pred[val]),weights=vw))
        candidates.append((score,value,model,pred))
        notices.extend(str(c.message) for c in caught)
    score,value,model,pred=min(candidates,key=lambda item:(item[0],item[1]))
    return dict(model=model,scaler=scaler,y_mean=ym,y_scale=ys,classification=classification),pred,value,notices


def probe_information(out,f,h,y,models,local,positions):
    tr=f.split.eq('train').to_numpy(); va=~tr; groups=f.trajectory.to_numpy()
    original_train=np.flatnonzero(tr & f.local_window.to_numpy())
    ow=sample_weights(groups[original_train])
    with np.load(STUDY/'controls/ridge_rotation_10p0_coordinate/projection.npz') as a:
        projection={k:a[k] for k in a.files}
    reps={'raw':h,'ridge':transform_supervised_rotation(h,projection)}
    recons={}; source_checks=[]
    for name,m in models.items():
        z,rec=infer(m,h);reps[name]=z;recons[name]=rec
        cfg='dense_l0p0' if name.startswith('pure') else 'slots_s0.3_a1.0'
        with np.load(STUDY/f'{cfg}_seed{name[-1]}'/'reconstruction.npz') as a:expected=a['z']
        error=np.max(abs(z[positions]-expected),axis=1)
        source_checks.append(dict(model=name,max_code_abs_error=float(error.max()),rows_exceeding_2e_4=int((error>2e-4).sum())))
    save_csv(out/'numeric_reference_checks.csv',source_checks)
    rankings={}; artifacts={}; predictions={}; metrics=[]; trajectory_errors=[]; cards=[];examples=[]
    for method,x in reps.items():
        rankings[method]={}
        for j,target in enumerate(TARGETS):
            rank,_=coordinate_ranking(x[original_train],y[original_train,j],ow)
            rankings[method][target]=rank.tolist()
        if method in models:
            cfg='dense_l0p0' if method.startswith('pure') else 'slots_s0.3_a1.0'
            saved=pd.read_csv(STUDY/f'{cfg}_seed{method[-1]}'/'coordinate_ranking.csv')
            for target in TARGETS:
                expected=saved[(saved.target==target)&saved.representation.eq('z')].sort_values('rank').coordinate.astype(int).tolist()
                # Frozen train rankings, never reselect according to new phase/val results.
                rankings[method][target]=expected
        print('PROBES',method,flush=True)
        for scope,mask in [('local',f.local_window.to_numpy()),('post_onset',f.post_onset.to_numpy())]:
            train=tr&mask;val=va&mask
            if not train.any() or not val.any():raise ValueError('Empty scope')
            for j,target in enumerate(TARGETS):
                rank=np.array(rankings[method][target])
                jobs=[(f'top{n}',x[:,rank[:n]],rank[:n].tolist()) for n in SIZES]+[('full',x,list(range(x.shape[1])))]
                if target=='opening_rad':
                    jobs += [(f'complement{n}',x[:,rank[n:]],rank[n:].tolist()) for n in SIZES]
                    if method in recons:
                        jobs += [('residual',h-recons[method],list(range(64))),
                            ('top4_plus_residual',np.column_stack([x[:,rank[:4]],h-recons[method]]),rank[:4].tolist())]
                for representation,features,coordinates in jobs:
                    ident='__'.join([scope,method,target,representation])
                    artifact,pred,value,notices=fit_readout(features,y[:,j],train,val,groups,j==2)
                    artifacts[ident]={**artifact,'coordinates':coordinates,'method':method,'representation':representation,'scope':scope,'target':target}
                    predictions[ident]=pred.astype(np.float64)
                    for eval_scope,eval_mask in [(scope,val)]+([('post_onset_transfer',va&f.post_onset.to_numpy())] if scope=='local' else []):
                        w=sample_weights(groups[eval_mask])
                        row=dict(fit_scope=scope,eval_scope=eval_scope,method=method,target=target,representation=representation,
                            dimensions=features.shape[1],regularization=value,train_states=int(train.sum()),validation_states=int(eval_mask.sum()),
                            validation_trajectories=len(set(groups[eval_mask])),warning_count=len(notices))
                        if j==2:
                            row.update(average_precision=float(average_precision_score(y[eval_mask,j],pred[eval_mask],sample_weight=w)) if len(np.unique(y[eval_mask,j]))==2 else None,
                                log_loss=float(log_loss(y[eval_mask,j],np.column_stack([1-pred[eval_mask],pred[eval_mask]]),labels=[0,1],sample_weight=w)))
                        else:row.update(regression_metrics(y[eval_mask,j],pred[eval_mask],w))
                        metrics.append(row)
                        if j==1:
                            for tid in sorted(set(groups[eval_mask])):
                                q=eval_mask&(groups==tid)
                                trajectory_errors.append(dict(fit_scope=scope,eval_scope=eval_scope,method=method,representation=representation,trajectory=tid,mae=float(np.mean(abs(y[q,j]-pred[q])))))
        if method in models:
            for rank,coord in enumerate(rankings[method]['opening_rad'][:6],1):
                for scope,scope_mask in [('local',f.local_window.to_numpy()),('post_onset',f.post_onset.to_numpy())]:
                    for split,mask in [('train',tr),('validation',va)]:
                        mask=mask&scope_mask;w=sample_weights(groups[mask]);v=x[mask,coord]
                        cards.append(dict(model=method,rank=rank,coordinate=coord,scope=scope,split=split,active_fraction=float(np.average(v>0,weights=w)),
                            distance_r=weighted_r(v,y[mask,0],w),opening_r=weighted_r(v,y[mask,1],w),contact_r=weighted_r(v,y[mask,2],w),time_r=weighted_r(v,f.time.to_numpy()[mask],w)))
                train_values=x[original_train,coord];active=train_values>0
                if active.any():
                    threshold=weighted_quantile(train_values[active],ow[active],.8)
                    median=weighted_quantile(y[original_train,1],ow,.5)
                    sign=1 if weighted_r(train_values,y[original_train,1],ow)>=0 else -1
                    high=va&f.local_window.to_numpy()&(x[:,coord]>0)&(x[:,coord]>=threshold)
                    expected=(y[:,1]>=median) if sign>0 else (y[:,1]<=median)
                    for label,mask in [('median_rule_consistent',high&expected),('median_rule_counterexample',high&~expected)]:
                        selected=[];seen=set()
                        for i in np.argsort(-x[:,coord],kind='stable'):
                            if not mask[i] or groups[i] in seen:continue
                            selected.append(i);seen.add(groups[i])
                            if len(selected)==3:break
                        for i in selected:
                            examples.append(dict(model=method,rank=rank,coordinate=coord,case_type=label,trajectory=groups[i],state_index=int(f.iloc[i].state_index),
                                activation=float(x[i,coord]),train_active_q80=threshold,train_opening_median=median,train_sign=sign,
                                opening_rad=float(y[i,1]),distance_m=float(y[i,0]),contact=int(y[i,2]),
                                interpretation='train median display rule, not semantic truth'))
    # Time-only is a deliberately simple nuisance reference, fitted per scope.
    for scope,mask in [('local',f.local_window.to_numpy()),('post_onset',f.post_onset.to_numpy())]:
        train=tr&mask;val=va&mask
        spl=SplineTransformer(n_knots=5,degree=3,include_bias=False,extrapolation='linear')
        time=f.time.to_numpy()[:,None];spl.fit(time[train],sample_weight=sample_weights(groups[train]))
        art,pred,value,notices=fit_readout(spl.transform(time),y[:,1],train,val,groups,False)
        artifacts[f'{scope}__time__opening_rad__full']={**art,'spline':spl};predictions[f'{scope}__time__opening_rad__full']=pred
        metrics.append(dict(fit_scope=scope,eval_scope=scope,method='time',target='opening_rad',representation='full',dimensions=spl.n_features_out_,regularization=value,
            train_states=int(train.sum()),validation_states=int(val.sum()),validation_trajectories=len(set(groups[val])),warning_count=len(notices),**regression_metrics(y[val,1],pred[val],sample_weights(groups[val]))))
        constant=float(np.average(y[train,1],weights=sample_weights(groups[train])))
        metrics.append(dict(fit_scope=scope,eval_scope=scope,method='constant',target='opening_rad',representation='full',dimensions=0,regularization=0,
            train_states=int(train.sum()),validation_states=int(val.sum()),validation_trajectories=len(set(groups[val])),warning_count=0,
            **regression_metrics(y[val,1],np.full(val.sum(),constant),sample_weights(groups[val]))))
    save_csv(out/'probe_metrics.csv',metrics);save_csv(out/'opening_by_trajectory.csv',trajectory_errors)
    save_csv(out/'feature_cards.csv',cards);save_csv(out/'feature_examples.csv',examples);dump(out/'frozen_rankings.json',rankings)
    joblib.dump(artifacts,out/'new_development_probes.joblib');np.savez_compressed(out/'probe_predictions.npz',**predictions)
    e=pd.DataFrame(trajectory_errors);e['kind']=e.method.str.replace(r'_seed\d+$','',regex=True)
    boot=[]; rng=np.random.default_rng(20261104)
    for (scope,kind),g in e[e.eval_scope.isin(['local','post_onset'])].groupby(['eval_scope','kind']):
        g=g.groupby(['trajectory','representation']).mae.mean().unstack()
        for a,b in [('top1','top2'),('top2','top4'),('top4','top6'),('top6','full'),('top4','top4_plus_residual')]:
            if a not in g or b not in g:continue
            delta=(g[a]-g[b]).dropna().to_numpy()
            draws=rng.choice(delta,size=(2000,len(delta)),replace=True).mean(axis=1)
            boot.append(dict(scope=scope,kind=kind,comparison=a+'_to_'+b,trajectories=len(delta),improvement=float(delta.mean()),ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975))))
    save_csv(out/'opening_increment_bootstrap.csv',boot)
    # One-SE simplification rule; developmental and conditional on sampled objects.
    chosen={}
    for kind in ['assisted','pure']:
        g=e[(e.eval_scope=='post_onset')&e.method.str.startswith(kind)&e.representation.isin([f'top{n}' for n in SIZES])]
        g=g.groupby(['trajectory','representation']).mae.mean().unstack()
        means=g.mean();best=means.idxmin();se=float(g[best].std(ddof=1)/np.sqrt(len(g)))
        n=next(n for n in SIZES if means[f'top{n}'] <= means[best]+se)
        chosen[kind]=dict(n=n,best=best,best_mae=float(means[best]),best_se=se,rule='smallest N within one trajectory-level SE of best post-onset validation MAE; exploratory')
    dump(out/'candidate_selection.json',chosen)
    return reps,recons,rankings,chosen


def bidirectional_edits(out,local,models,rankings,chosen,actor):
    """New fixed-input protocol: bounded active-code increase/decrease at equal h norm.

    Both directions share min(native decrease norm, native bounded increase norm),
    with doses .5/1 of that budget, capped at .1||h||. Zero joint support is retained.
    These operations are NOT called semantic enhancement or physical rescue.
    """
    h=local['h'];tr=local['split']=='train';groups=local['trajectory_id']
    sizes=sorted(set([1,chosen['assisted']['n']]))
    all_summary=[];quality=[]
    for name,m in models.items():
        # Single-state inference matches the physical runtime calling convention.
        z=np.asarray([m.infer(row)[0][0] for row in h]);decoder=m.w['decoder_weight']*m.w['scale'][:,None]
        rank=rankings[name]['opening_rad'];trainweights=sample_weights(groups[tr])
        std=np.zeros(64);upper=np.zeros(64)
        for c in range(64):
            active=z[tr,c]>0
            if active.any():
                values=z[tr,c][active];w=trainweights[active]
                std[c]=np.sqrt(np.average((values-np.average(values,weights=w))**2,weights=w))
                upper[c]=weighted_quantile(values,w,.99)
        emitted=[];rows=[]
        for i,original in enumerate(h):
            for n in sizes:
                coords=np.array(rank[:n]);before=z[i]
                decrease=decoder[:,coords]@(-before[coords])
                increase_amount=np.where(before[coords]>0,np.maximum(0,np.minimum(.5*std[coords],upper[coords]-before[coords])),0.)
                increase=decoder[:,coords]@increase_amount
                down=project_delta(original,decrease);up=project_delta(original,increase)
                budget=min(np.linalg.norm(down),np.linalg.norm(up),.1*np.linalg.norm(original))
                for dose in [.5,1.]:
                    wanted=dose*budget
                    dirs=[('decrease',decrease),('bounded_increase',increase)]+[(f'random{k}',v) for k,v in enumerate(stable_random_directions(f'{groups[i]}|{local["state_index"][i]}|{name}|{n}',64,3))]
                    for control,direction in dirs:
                        if wanted<=1e-9:delta=np.zeros(64);status='no_bidirectional_common_support'
                        else:delta,status=match_projected_norm(original,direction,wanted)
                        edit=np.maximum(original+delta,0).astype(np.float32) if status=='ok' else original.astype(np.float32)
                        emitted.append(edit)
                        rows.append(dict(source_index=i,split=str(local['split'][i]),trajectory=str(groups[i]),state_index=int(local['state_index'][i]),model=name,n=n,dose=dose,control=control,status=status,requested_norm=wanted,
                            actual_norm=float(np.linalg.norm(edit.astype(float)-original)),target_indices=json.dumps(coords.tolist())))
        edits=np.asarray(emitted);after,_=infer(m,edits);pred_after=m.predict(after);pred_before=m.predict(z)
        mu=actor_tail(actor,edits);base_mu=actor_tail(actor,h)
        table=pd.DataFrame(rows);idx=table.source_index.to_numpy()
        table['norm_error']=abs(table.actual_norm-table.requested_norm)
        remaining=[];other=[];target_change=[]
        for k,r in table.iterrows():
            coords=np.array(rank[:int(r.n)]);mask=np.ones(64,dtype=bool);mask[coords]=False
            b=np.linalg.norm(z[int(r.source_index),coords]);a=np.linalg.norm(after[k,coords])
            remaining.append(float(a/b) if b>1e-12 else np.nan)
            target_change.append(float(np.linalg.norm(after[k,coords]-z[int(r.source_index),coords])))
            other.append(float(np.linalg.norm(after[k,mask]-z[int(r.source_index),mask])))
        table['target_remaining']=remaining;table['target_code_change']=target_change;table['other_code_l2']=other
        table['opening_probe_delta_degrees']=np.rad2deg(pred_after[:,1]-pred_before[idx,1])
        table['distance_probe_delta_m']=pred_after[:,0]-pred_before[idx,0]
        table['contact_probe_delta']=pred_after[:,2]-pred_before[idx,2]
        table['raw_action_l2']=np.linalg.norm(mu-base_mu[idx],axis=1)
        table['clipped_action_l2']=np.linalg.norm(mu.clip(-1,1)-base_mu[idx].clip(-1,1),axis=1)
        assert np.isfinite(edits).all() and (edits>=0).all()
        valid=table.status.eq('ok')
        maxerror=float(table.loc[valid,'norm_error'].max()) if valid.any() else 0.
        if maxerror>2e-6:raise ValueError('New offline matching tolerance failed')
        table.to_csv(out/f'new_edits_{name}.csv',index=False,encoding='utf-8-sig')
        np.savez_compressed(out/f'new_edit_vectors_{name}.npz',source_index=idx,h_applied=edits,z_before=z,z_after=after,mu=mu,base_mu=base_mu)
        quality.append(dict(model=name,rows=len(table),valid=int(valid.sum()),max_norm_error=maxerror,negative_h=0))
        met=['actual_norm','target_remaining','target_code_change','other_code_l2','opening_probe_delta_degrees','distance_probe_delta_m','contact_probe_delta','raw_action_l2','clipped_action_l2']
        for key,g in table.groupby(['split','n','dose','control']):
            good=g[g.status=='ok'];means=good.groupby('trajectory')[met].mean().mean() if len(good) else pd.Series(dtype=float)
            all_summary.append(dict(zip(['split','n','dose','control'],key),model=name,states=len(g),valid=len(good),valid_trajectories=good.trajectory.nunique(),**{k:float(v) for k,v in means.items()}))
        print('EDITED',name,len(table),flush=True)
    save_csv(out/'new_edit_summary.csv',all_summary);save_csv(out/'new_edit_checks.csv',quality)
    dump(out/'new_edit_protocol.json',dict(scope='original 1365 train/validation local-window states; no simulation',sizes=sizes,doses=[.5,1.],
        increase='active coordinates only, request <=0.5 train active-code SD and <=train active q99; coordinate increase is not semantic opening enhancement',
        budget='min projected native up/down norms and 0.1||h||; both targets and random controls use same statewise budget; all zero support retained',
        weights='equal valid trajectories within each model and condition for conditional metrics; coverage reported separately',
        semantic_interpretation='frozen probe diagnostic only, not physical angle or rescue',model_training=False))


def physical_cases(out,actor):
    existing=pd.read_csv(PHYSICAL/'review_20261104/physical_descriptive.csv')
    failure=existing[existing.model.notna() & existing.control.eq('target') & ~existing.success]
    base_cache={};case_rows=[]
    def read(row):
        folder=PHYSICAL/f'remote/campaign/object{int(row.object)}_seed{int(row.environment_seed)}'/row['name']
        with np.load(folder/(row['name']+'.npz'),allow_pickle=False) as a:
            keys=['opening','distance','contact','post_opening','post_distance','post_contact','post_success','h_original','h_applied','action','mu','scheduled','applied','edit_z_before','edit_z_after']
            return {k:a[k].copy() for k in keys}
    def first(mask):
        indices=np.flatnonzero(mask);return int(indices[0]) if len(indices) else None
    for _,r in failure.iterrows():
        origin=(int(r.object),int(r.environment_seed))
        if origin not in base_cache:
            nr=existing[(existing.object==origin[0])&(existing.environment_seed==origin[1])&existing.name.eq('natural')].iloc[0]
            base_cache[origin]=read(nr)
        b=base_cache[origin];a=read(r);n=min(len(a['action']),len(b['action']))
        peers=existing[(existing.object==origin[0])&(existing.environment_seed==origin[1])&existing.model.eq(r.model)&existing.group.eq(r.group)&existing.duration.eq(r.duration)&existing.dose.eq(r.dose)]
        random=peers[peers.control.str.startswith('random',na=False)]
        row=r.to_dict();row.update(random_successes=int(random.success.sum()),random_count=len(random),
            ridge_success=bool(peers[peers.control.eq('ridge')].success.iloc[0]),
            first_action_difference=first(np.max(abs(a['action'][:n]-b['action'][:n]),axis=1)>1e-5),
            first_opening_difference_2deg=first(abs(np.rad2deg(a['post_opening'][:n]-b['post_opening'][:n]))>2),
            first_distance_difference_2cm=first(abs(a['post_distance'][:n]-b['post_distance'][:n])>.02),
            first_contact_loss_vs_natural=first(b['post_contact'][:n] & ~a['post_contact'][:n]),
            natural_success_step=first(b['post_success']),
            post_opening_at_natural_end_degrees=float(np.rad2deg(a['post_opening'][n-1])),
            natural_end_opening_degrees=float(np.rad2deg(b['post_opening'][n-1])))
        case_rows.append(row)
    save_csv(out/'target_failure_mechanism_cases.csv',case_rows)
    # Full paired timeline for the previously highlighted case, including controls.
    chosen=existing[(existing.object==11477)&(existing.environment_seed==2026110301)&((existing.model.eq('assisted_seed1')&existing.dose.eq('nominal')&existing.duration.eq(5))|existing.name.eq('natural'))]
    baseline=base_cache[(11477,2026110301)];timeline=[];arrays={}
    for _,r in chosen.iterrows():
        a=read(r);arrays[r['name']]=a;n=len(a['action']);bn=len(baseline['action'])
        for t in range(n):
            timeline.append(dict(name=r['name'],group=r.group if pd.notna(r.group) else 'natural',control=r.control if pd.notna(r.control) else 'natural',step=t,
                opening_deg=float(np.rad2deg(a['post_opening'][t])),distance=float(a['post_distance'][t]),contact=bool(a['post_contact'][t]),success=bool(a['post_success'][t]),
                scheduled=bool(a['scheduled'][t]),applied=bool(a['applied'][t]),
                action_delta_l2=float(np.linalg.norm(a['action'][t]-baseline['action'][t])) if t<bn else None))
    save_csv(out/'example_full_timelines.csv',timeline)
    equality=[]
    for left,right in [('top1','top2'),('top2','top3'),('top3','top4'),('top4','top6')]:
        a=arrays[f'assisted_seed1_{left}_d5_nominal_target'];b=arrays[f'assisted_seed1_{right}_d5_nominal_target']
        stop=min(len(a['h_applied']),len(b['h_applied']))
        equality.append(dict(left=left,right=right,compared_steps=stop,
            maximum_h_difference=float(np.max(abs(a['h_applied'][:stop]-b['h_applied'][:stop]))),
            maximum_action_difference=float(np.max(abs(a['action'][:stop]-b['action'][:stop]))),
            same_length=len(a['action'])==len(b['action']),
            trigger_h_difference=float(np.max(abs(a['h_applied'][11]-b['h_applied'][11])))))
    save_csv(out/'example_group_equality.csv',equality)
    return case_rows


def success_components(out):
    """Recompute the task's conjunctive success predicate on EVERY saved step."""
    existing=pd.read_csv(PHYSICAL/'review_20261104/physical_descriptive.csv')
    rows=[];bad=0;strict_bad=0;steps=0
    env_source=PROJECT_ROOT.parents[1]/'research/embodied_sae/runs/laptop_resume_bundle/third_party/dexart-release/dexart/env/rl_env/laptop_env.py'
    for _,r in existing.iterrows():
        path=PHYSICAL/f'remote/campaign/object{int(r.object)}_seed{int(r.environment_seed)}'/r['name']/(r['name']+'.npz')
        with np.load(path,allow_pickle=False) as a:
            goal=a['post_progress']>.95;near=a['post_distance']<=.2
            strict=a['post_strict_contact'];actual=a['post_success'];pred=goal&near&strict
            reconstructed_strict=(a['post_contacts'][:,:-1].sum(axis=1)>=2)&a['post_contacts'][:,-1].astype(bool)
            strict_bad+=int(np.count_nonzero(strict!=reconstructed_strict))
            bad+=int(np.count_nonzero(actual!=pred));steps+=len(actual)
            rows.append(dict(object=int(r.object),environment_seed=int(r.environment_seed),name=r['name'],model=r.model,group=r.group,control=r.control,dose=r.dose,duration=r.duration,
                success=bool(actual.any()),progress_goal_reached=bool(goal.any()),goal_and_near=bool((goal&near).any()),goal_and_strict=bool((goal&strict).any()),goal_and_near_and_strict=bool(pred.any()),
                max_progress=float(a['post_progress'].max()),max_opening_degrees=float(np.rad2deg(a['post_opening']).max()),final_opening_degrees=float(np.rad2deg(a['post_opening'][-1])),goal_steps=int(goal.sum()),goal_strict_steps=int((goal&strict).sum())))
    save_csv(out/'physical_success_components.csv',rows)
    if bad or strict_bad:raise ValueError('Saved task success/contact definition mismatch')
    dump(out/'success_predicate_audit.json',dict(conditions=len(rows),post_action_steps=steps,mismatches=bad,strict_contact_mismatches=strict_bad,
        predicate='post_progress > .95 AND post_distance <= .2 AND post_strict_contact',
        strict_contact='at least two finger groups AND palm; this is a task predicate, not a certified stable grasp',
        progress='1 - abs(joint_qpos - target_middle) / total; not monotone opening amount',
        source_file=str(env_source),source_sha256=sha(env_source)))
    timeline=[]
    for name in ['natural','assisted_seed1_top1_d5_nominal_target','assisted_seed1_top4_d5_nominal_target']:
        path=PHYSICAL/'remote/campaign/object11477_seed2026110301'/name/(name+'.npz')
        with np.load(path) as a:
            for t in range(len(a['action'])):
                timeline.append(dict(name=name,step=t,progress=float(a['post_progress'][t]),strict_contact=bool(a['post_strict_contact'][t]),
                    near=bool(a['post_distance'][t]<=.2),success=bool(a['post_success'][t])))
    save_csv(out/'example_success_predicate_timeline.csv',timeline)


def factorial_edits(out,local,models,rankings,actor):
    """Fixed-component A/B/AB diagnostic; actual h input is additive, no new physics."""
    h=local['h'].astype(np.float32);records=[]
    w2=actor['mlp_extractor.policy_net.2.weight'];b2=actor['mlp_extractor.policy_net.2.bias']
    for name,m in models.items():
        z=np.array([m.infer(v)[0][0] for v in h]);r=rankings[name]['opening_rad'][:4]
        decoder=m.w['decoder_weight']*m.w['scale'][:,None]
        variants=[];details=[]
        for i,v in enumerate(h):
            base=v.astype(float)
            a=project_delta(base,decoder[:,r[:2]]@(-z[i,r[:2]]))
            b=project_delta(base,decoder[:,r[2:]]@(-z[i,r[2:]]))
            ab=a+b;negative=ab<0
            nonnegative_scale=float(np.min(base[negative]/(-ab[negative]))) if negative.any() else 1.
            cap=.1*np.linalg.norm(base);maximum=max(np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(ab))
            scale=max(0,min(1.,nonnegative_scale,cap/max(maximum,1e-30)))
            va=(base+scale*a).astype(np.float32);vb=(base+scale*b).astype(np.float32);vab=(base+scale*ab).astype(np.float32)
            if min(va.min(),vb.min(),vab.min()) < -1e-7:raise ValueError('Nonnegative factorial construction failed')
            add_error=float(np.max(abs(vab.astype(float)-va-vb+base)))
            if add_error>2e-6:raise ValueError('Factorial actual-input additivity failed')
            variants.extend([v,va,vb,vab]);details.append((scale,add_error,np.linalg.norm(scale*a)>1e-9 and np.linalg.norm(scale*b)>1e-9))
        variants=np.array(variants);mu=actor_tail(actor,variants).reshape(len(h),4,22)
        h2=np.maximum(variants@w2.T+b2,0).reshape(len(h),4,64)
        interaction=mu[:,3]-mu[:,1]-mu[:,2]+mu[:,0]
        clipping=mu.clip(-1,1);clipped_interaction=clipping[:,3]-clipping[:,1]-clipping[:,2]+clipping[:,0]
        gate_changed=np.any((h2[:,1:]>0)!=(h2[:,0:1]>0),axis=(1,2))
        for i,(scale,error,valid) in enumerate(details):
            records.append(dict(model=name,split=str(local['split'][i]),trajectory=str(local['trajectory_id'][i]),state_index=int(local['state_index'][i]),
                scale=scale,valid_two_components=bool(valid),input_additivity_error=error,gate_changed=bool(gate_changed[i]),
                raw_mu_interaction_l2=float(np.linalg.norm(interaction[i])),clipped_interaction_l2=float(np.linalg.norm(clipped_interaction[i]))))
        np.savez_compressed(out/f'factorial_vectors_{name}.npz',h=variants,mu=mu,h2=h2)
    save_csv(out/'factorial_results.csv',records)
    dump(out/'factorial_protocol.json',dict(groups='A=first 2 opening coordinates; B=next 2; AB=union',
        construction='project individual directions, then jointly scale fixed A/B components so A,B,AB are nonnegative and capped; actual inputs additive within float rounding',
        interaction='mu_AB-mu_A-mu_B+mu_0; clipping interaction separately; neither proves temporal compensation or a single semantic circuit'))


def plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=pd.read_csv(out/'probe_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    colors={'assisted':'#c45d24','pure':'#386cb0','raw':'#666666','ridge':'#298c6d'}
    for ax,scope in zip(axes,['local','post_onset']):
        g=m[(m.target=='opening_rad')&(m.eval_scope==scope)&m.representation.isin([f'top{n}' for n in SIZES])].copy()
        g['kind']=g.method.str.replace(r'_seed\d+$','',regex=True);g['N']=g.representation.str.replace('top','').astype(int)
        for kind,v in g.groupby('kind'):
            means=v.groupby('N').mae.mean();ax.plot(means.index,means.values,marker='o',label=kind,color=colors[kind])
        ax.set(title='Original local window' if scope=='local' else 'Full post-onset phase',xlabel='Fixed coordinate count',ylabel='Validation opening MAE (rad)',xticks=SIZES)
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Development readouts of frozen representations; new probes, no SAE retraining')
    fig.tight_layout();fig.savefig(out/'information_curves.png',dpi=180);plt.close(fig)
    t=pd.read_csv(out/'example_full_timelines.csv')
    fig,axes=plt.subplots(2,2,figsize=(11.5,7))
    for label in ['natural','top1','top2','top3','top4']:
        g=t[(t.group==label)&t.control.isin(['target','natural'])]
        if g.empty:continue
        axes[0,0].plot(g.step,g.opening_deg,label=label)
        early=g[g.step<45];axes[0,1].plot(early.step,early.opening_deg,label=label)
        axes[1,0].plot(early.step,early.distance,label=label)
        axes[1,1].plot(early.step,early.action_delta_l2,label=label)
    for ax,title,ylabel in zip(axes.flat,['Whole saved episode','Opening: early trajectory','Palm-handle distance','Action difference vs natural (common times only)'],['Opening (degrees)','Opening (degrees)','Distance (m)','Clipped action L2']):
        ax.set(title=title,xlabel='Control step',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.axvspan(11,16,color='grey',alpha=.12)
    fig.suptitle('Saved physical example: 11477 / environment seed 2026110301 / assisted model seed 1')
    fig.tight_layout();fig.savefig(out/'physical_example.png',dpi=180);plt.close(fig)
    if (out/'example_success_predicate_timeline.csv').is_file():
        t=pd.read_csv(out/'example_success_predicate_timeline.csv')
        fig,axes=plt.subplots(2,1,figsize=(10,5.6),sharex=True)
        for name,g in t[t.step<40].groupby('name',sort=False):
            label='natural' if name=='natural' else 'Top1 pulse' if 'top1' in name else 'Top4 pulse'
            line=axes[0].plot(g.step,g.progress,label=label)[0]
            # Offset rows so the three binary contact traces remain distinguishable.
            offset={'natural':0,'Top1 pulse':1.5,'Top4 pulse':3}[label]
            axes[1].step(g.step,g.strict_contact.astype(float)+offset,where='post',color=line.get_color(),label=label)
            goal=g[g.progress>.95]
            axes[0].scatter(goal.step,goal.progress,color=line.get_color(),marker='o',s=28)
        axes[0].axhline(.95,color='grey',ls='--',label='progress threshold')
        axes[0].set(ylabel='Task progress',title='Progress threshold alone is insufficient for task success')
        axes[0].legend(fontsize=8);axes[1].set(xlabel='Control step',ylabel='Strict-contact traces (offset)',yticks=[0,1,1.5,2.5,3,4],yticklabels=['N: no','N: yes','T1: no','T1: yes','T4: no','T4: yes'])
        for ax in axes:ax.axvspan(11,16,color='grey',alpha=.12);ax.grid(alpha=.15)
        fig.tight_layout();fig.savefig(out/'success_components.png',dpi=180);plt.close(fig)


def summarize_new_edits(out):
    rows=pd.read_csv(out/'new_edit_summary.csv');rows['kind']=rows.model.str.replace(r'_seed\d+$','',regex=True)
    metrics=['target_remaining','opening_probe_delta_degrees','distance_probe_delta_m','contact_probe_delta','raw_action_l2','clipped_action_l2','other_code_l2']
    summary=rows.groupby(['split','kind','n','dose','control']).agg(states=('states','sum'),valid=('valid','sum'),**{k:(k,'mean') for k in metrics}).reset_index()
    summary['coverage']=summary.valid/summary.states
    summary.to_csv(out/'new_edit_kind_summary.csv',index=False,encoding='utf-8-sig')
    f=pd.read_csv(out/'factorial_results.csv');f['kind']=f.model.str.replace(r'_seed\d+$','',regex=True)
    f.groupby(['split','kind']).agg(cases=('state_index','size'),two_component_cases=('valid_two_components','sum'),gate_change_cases=('gate_changed','sum'),
        raw_interaction_mean=('raw_mu_interaction_l2','mean'),clipped_interaction_mean=('clipped_interaction_l2','mean')).to_csv(out/'factorial_summary.csv',encoding='utf-8-sig')


def run(destination,skip_edits=False):
    out=output_path(destination);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    before=source_hashes()
    dump(out/'protocol.json',dict(status='running',created_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        source_hashes=before,scope='local exploratory train/validation follow-up; frozen SAE and actor; no test or new physics',
        post_onset='first pre-action opening >=0.05 rad through last available action; includes stall/reversal; terminal state excluded',
        ranking='original train local-window frozen target-specific ordering; no val/physics ranking',
        new_probes='train-only weighted fit, val alpha/C selection from .1/1/10; trajectory equal; scope-specific refitting explicitly developmental',
        model_seeds=[0,1,2],bootstrap='2000 whole-val-trajectory pairs, average model seeds first; descriptive object-conditional intervals'))
    shutil.copy2(Path(__file__),out/'source_local_interpretability.py')
    shutil.copy2(PROJECT_ROOT/'14_run_local_interpretability.py',out/'14_run_local_interpretability.py')
    models={p.name:NumericSAE(p) for p in (PHYSICAL/'bundle/models').iterdir()}
    with np.load(PHYSICAL/'bundle/actor.npz') as a:actor={k:a[k] for k in a.files}
    with threadpool_limits(limits=1):
        f,h,y,local,positions=load_development(out)
        print('LOADED',len(f),'states',f.trajectory.nunique(),'trajectories',flush=True)
        reps,recons,rankings,chosen=probe_information(out,f,h,y,models,local,positions)
        case_rows=physical_cases(out,actor)
        success_components(out)
        factorial_edits(out,local,models,rankings,actor)
        if not skip_edits:
            bidirectional_edits(out,local,models,rankings,chosen,actor)
            summarize_new_edits(out)
        plots(out)
    after=source_hashes()
    if before!=after:raise RuntimeError('Frozen source artifact changed')
    dump(out/'completion.json',dict(status='completed',development_states=len(f),development_trajectories=f.trajectory.nunique(),
        local_states=len(local['h']),post_onset_states=int(f.post_onset.sum()),failure_cases=len(case_rows),
        candidate_selection=chosen,edits_completed=not skip_edits,source_artifacts_unchanged=True,
        new_sae_training=False,test_numeric_trajectories_loaded=0,new_physical_rollouts=0))
    print('LOCAL FOLLOW-UP COMPLETE',out,flush=True)
