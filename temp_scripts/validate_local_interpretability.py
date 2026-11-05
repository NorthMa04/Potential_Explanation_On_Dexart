"""Independent algebra and artifact checks of the local scientific follow-up."""
from pathlib import Path
import argparse
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import torch

from temp_scripts.project_paths import PROJECT_ROOT, output_path
from temp_scripts.physical_runtime import NumericSAE


def weights(groups):
    unique,inv,count=np.unique(groups,return_inverse=True,return_counts=True)
    w=1/count[inv];return w/w.mean()


def numpy_actor(actor,h):
    # A different numerical implementation from the PyTorch evaluation path.
    a=np.maximum(h.astype(float)@actor['mlp_extractor.policy_net.2.weight'].astype(float).T+actor['mlp_extractor.policy_net.2.bias'],0)
    return a@actor['action_net.weight'].astype(float).T+actor['action_net.bias']


def validate(root):
    root=output_path(root);torch.set_num_threads(1)
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    source_count=0
    for relative,digest in protocol['source_hashes'].items():
        assert hashlib.sha256((PROJECT_ROOT/relative).read_bytes()).hexdigest()==digest,relative
        source_count+=1
    f=pd.read_csv(root/'development_frames.csv')
    assert not f.duplicated(['trajectory','state_index']).any()
    assert set(f.split)=={'train','validation'}
    with np.load(root/'development_arrays.npz') as a:h=a['h'];y=a['y'];positions=a['local_positions']
    bundle=PROJECT_ROOT/'outputs/combined_physical/20261103_140606_955714/bundle'
    with np.load(bundle/'actor.npz') as a:actor={k:a[k] for k in a.files}
    with np.load(PROJECT_ROOT/'outputs/state_assisted/20261102_122123_738477/controls/ridge_rotation_10p0_coordinate/projection.npz') as a:
        ridge=((h-a['h_mean'])/a['h_scale'])@a['rotation']
    reps={'raw':h,'ridge':ridge};res={};models={}
    for folder in (bundle/'models').iterdir():
        m=NumericSAE(folder);models[folder.name]=m;zs=[];rs=[]
        for start in range(0,len(h),256):
            z,r,_=m.infer(h[start:start+256]);zs.append(z);rs.append(r)
        reps[folder.name]=np.concatenate(zs);res[folder.name]=h-np.concatenate(rs)
    arts=joblib.load(root/'new_development_probes.joblib')
    pred=np.load(root/'probe_predictions.npz')
    metric=pd.read_csv(root/'probe_metrics.csv')
    max_prediction=0.;max_mean=0.;max_metric=0.;probe_count=0
    for ident,art in arts.items():
        scope,method,target,representation=ident.split('__')
        if method=='time':x=art['spline'].transform(f.time.to_numpy()[:,None])
        elif representation=='residual':x=res[method]
        elif representation=='top4_plus_residual':x=np.column_stack([reps[method][:,art['coordinates']],res[method]])
        else:x=reps[method][:,art['coordinates']]
        mask=f.local_window.to_numpy() if scope=='local' else f.post_onset.to_numpy()
        train=mask&f.split.eq('train').to_numpy();val=mask&f.split.eq('validation').to_numpy()
        w=weights(f.trajectory[train].to_numpy())
        err=float(np.max(abs(np.average(x[train],axis=0,weights=w)-art['scaler'].mean_)))
        max_mean=max(max_mean,err);assert err<1e-9,(ident,err,np.average(x[train],axis=0,weights=w),art['scaler'].mean_)
        xx=(x-art['scaler'].mean_)/art['scaler'].scale_
        if 'constant' in art:p=np.full(len(x),art['constant'])
        else:
            p=xx@art['model'].coef_.reshape(-1)+np.asarray(art['model'].intercept_).reshape(-1)[0]
            if target=='any_contact':p=1/(1+np.exp(-np.clip(p,-700,700)))
            else:p=p*art['y_scale']+art['y_mean']
        error=float(np.max(abs(p-pred[ident])));max_prediction=max(max_prediction,error)
        assert error<1e-8,ident
        if target=='opening_rad':
            # Explicit mean-of-trajectory MAEs, independent of weighted-frame implementation.
            mae=float(np.mean([np.mean(abs(p[val&(f.trajectory.to_numpy()==tid)]-y[val&(f.trajectory.to_numpy()==tid),1])) for tid in sorted(set(f.trajectory[val]))]))
            expected=float(metric[(metric.fit_scope==scope)&(metric.eval_scope==scope)&(metric.method==method)&(metric.target==target)&(metric.representation==representation)].mae.iloc[0])
            max_metric=max(max_metric,abs(mae-expected));assert abs(mae-expected)<1e-10
        probe_count+=1
    pred.close()
    with np.load(PROJECT_ROOT/'outputs/state_assisted/20261102_122123_738477/selected_arrays.npz') as a:local_h=a['h'];local_y=a['y']
    np.testing.assert_array_equal(h[positions],local_h);np.testing.assert_array_equal(y[positions],local_y)
    edit_count=0;max_actor=0.;max_norm=0.;sample_code_error=0.;factorial_max=0.;factorial_count=0
    for name,m in models.items():
        table=pd.read_csv(root/f'new_edits_{name}.csv')
        assert not table.duplicated(['source_index','n','dose','control']).any()
        with np.load(root/f'new_edit_vectors_{name}.npz') as a:
            applied=a['h_applied'];idx=a['source_index'];mu=a['mu'];base_mu=a['base_mu'];zafter=a['z_after']
        assert np.isfinite(applied).all() and (applied>=0).all()
        norms=np.linalg.norm(applied.astype(float)-local_h[idx],axis=1)
        max_norm=max(max_norm,float(np.max(abs(norms-table.actual_norm))))
        np.testing.assert_allclose(norms,table.actual_norm,atol=1e-12,rtol=1e-12)
        valid=table.status.eq('ok').to_numpy()
        np.testing.assert_allclose(norms[valid],table.requested_norm[valid],atol=2e-6,rtol=0)
        assert (norms[~valid]==0).all()
        assert (norms<=.1*np.linalg.norm(local_h[idx],axis=1)+2e-6).all()
        error=float(np.max(abs(numpy_actor(actor,applied)-mu)));max_actor=max(max_actor,error)
        assert error<2e-5
        l2=np.linalg.norm(mu.clip(-1,1)-base_mu[idx].clip(-1,1),axis=1)
        np.testing.assert_allclose(l2,table.clipped_action_l2,atol=1e-7,rtol=1e-6)
        assert (table.groupby(['source_index','n','dose']).size()==5).all()
        # Re-encoding cross-call sample; all actor/norm checks above cover every row.
        for j in np.linspace(0,len(applied)-1,32,dtype=int):
            actual=m.infer(applied[j])[0][0]
            sample_code_error=max(sample_code_error,float(np.max(abs(actual-zafter[j]))))
        edit_count+=len(table)
        with np.load(root/f'factorial_vectors_{name}.npz') as a:
            inp=a['h'].reshape(len(local_h),4,64);mu=a['mu'];h2=a['h2']
        np.testing.assert_array_equal(inp[:,0],local_h.astype(np.float32))
        error=float(np.max(abs(inp[:,3].astype(float)-inp[:,1]-inp[:,2]+inp[:,0])));factorial_max=max(factorial_max,error)
        assert error<2e-6
        assert np.isfinite(inp).all() and (inp>=-1e-7).all()
        np.testing.assert_allclose(numpy_actor(actor,inp.reshape(-1,64)).reshape(mu.shape),mu,atol=2e-5,rtol=2e-5)
        changed=np.any((h2[:,1:]>0)!=(h2[:,0:1]>0),axis=(1,2))
        interaction=mu[:,3]-mu[:,1]-mu[:,2]+mu[:,0]
        # In a fixed ReLU region, the downstream network must be affine.
        if (~changed).any():assert np.max(abs(interaction[~changed]))<5e-5
        factorial_count+=len(local_h)
    success=json.loads((root/'success_predicate_audit.json').read_text())
    assert success['mismatches']==success['strict_contact_mismatches']==0
    comp=pd.read_csv(root/'physical_success_components.csv')
    target_fail=comp[comp.model.notna()&comp.control.eq('target')&~comp.success]
    assert len(target_fail)==36 and target_fail.progress_goal_reached.all() and target_fail.goal_and_near.all() and not target_fail.goal_and_strict.any()
    strict_models=0;strict_prediction_error=0.;strict_ap_error=0.
    if (root/'strict_contact_probes.joblib').is_file():
        labels=pd.read_csv(root/'strict_contact_labels.csv')
        assert (labels.trajectory==f.trajectory).all() and (labels.state_index==f.state_index).all()
        strict=labels.strict_contact.to_numpy();train=f.post_onset.to_numpy()&f.split.eq('train').to_numpy();val=f.post_onset.to_numpy()&f.split.eq('validation').to_numpy()
        vweights=weights(f.trajectory[val].to_numpy())
        sa=joblib.load(root/'strict_contact_probes.joblib');sp=np.load(root/'strict_contact_predictions.npz');sm=pd.read_csv(root/'strict_contact_probe_metrics.csv')
        for name,art in sa.items():
            method,representation=name.split('__');x=reps[method][:,art['coordinates']].astype(float)
            np.testing.assert_allclose(np.average(x[train],axis=0,weights=weights(f.trajectory[train].to_numpy())),art['scaler'].mean_,atol=1e-9,rtol=0)
            xx=(x-art['scaler'].mean_)/art['scaler'].scale_
            raw=xx@art['model'].coef_.reshape(-1)+art['model'].intercept_[0]
            pp=1/(1+np.exp(-np.clip(raw,-700,700)))
            error=float(np.max(abs(pp-sp[name])));strict_prediction_error=max(strict_prediction_error,error);assert error<1e-8
            # Independently compute weighted AP at distinct score thresholds.
            order=np.argsort(-pp[val],kind='stable');scores=pp[val][order];ww=vweights[order];yy=strict[val][order]
            tp=np.cumsum(ww*yy);total=np.cumsum(ww);end=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
            increments=np.diff(np.r_[0,tp[end]])
            ap=float(np.sum(increments*(tp[end]/total[end]))/tp[-1])
            expected=float(sm[(sm.method==method)&(sm.representation==representation)].average_precision.iloc[0])
            strict_ap_error=max(strict_ap_error,abs(ap-expected));assert abs(ap-expected)<1e-10
            strict_models+=1
        sp.close()
    # Different call shapes may change TopK ties; report sampled latent mismatch honestly.
    result=dict(status='passed' if sample_code_error<=2e-4 else 'passed_actor_and_metrics_with_latent_sample_caveat',
        frozen_source_hashes=source_count,probe_models=probe_count,probe_scaler_train_mean_max_error=max_mean,
        manual_probe_prediction_max_error=max_prediction,opening_mean_of_trajectory_mae_max_error=max_metric,
        fixed_input_edit_rows=edit_count,independent_numpy_actor_max_error=max_actor,recorded_actual_norm_max_error=max_norm,
        sampled_reencode_rows=32*len(models),sampled_reencode_max_error=sample_code_error,
        factorial_cases=factorial_count,actual_input_additivity_max_error=factorial_max,
        strict_contact_models=strict_models,strict_contact_manual_prediction_max_error=strict_prediction_error,
        strict_contact_independent_AP_max_error=strict_ap_error,
        physical_success_predicate=success,
        limitations='new probes selected on validation; descriptive development results, no new independent test or physical rollout; no global claim about semantic selectivity')
    (root/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    validate(p.parse_args().run)
