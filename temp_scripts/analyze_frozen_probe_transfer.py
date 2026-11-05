"""Apply pre-online frozen readouts to all new natural seeds, without fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from temp_scripts.physical_runtime import NumericSAE
from temp_scripts.project_paths import PROJECT_ROOT


def native(path):
    value = str(Path(path).resolve())
    return Path('\\\\?\\' + value) if os.name == 'nt' and not value.startswith('\\\\?\\') else Path(value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def predict(artifact, features):
    x = (np.asarray(features, float) - artifact['scaler'].mean_) / artifact['scaler'].scale_
    result = x @ artifact['model'].coef_.reshape(-1) + float(np.asarray(artifact['model'].intercept_).reshape(-1)[0])
    result = result * artifact['y_scale'] + artifact['y_mean']
    check = artifact['model'].predict(artifact['scaler'].transform(np.asarray(features, float)))
    check = check * artifact['y_scale'] + artifact['y_mean']
    np.testing.assert_allclose(result, check, atol=1e-12, rtol=1e-12)
    return result, float(np.max(abs(result - check)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=PROJECT_ROOT/'outputs/deeper_analysis_20261105/readout_transfer')
    args = parser.parse_args()
    out = native(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    old = native(PROJECT_ROOT/'outputs/local_interpretability/20261104_mainline_v3')
    online = native(PROJECT_ROOT/'outputs/online_development_20261104')
    source = [old/'new_development_probes.joblib', old/'frozen_rankings.json', online/'remote/screen/campaign.json']
    projection_path = native(PROJECT_ROOT/'outputs/state_assisted/20261102_122123_738477/controls/ridge_rotation_10p0_coordinate/projection.npz')
    source.append(projection_path)
    probes = joblib.load(old/'new_development_probes.joblib')
    campaign = json.loads((online/'remote/screen/campaign.json').read_text())
    assert campaign['status'] == 'completed' and len(campaign['cases']) == 32
    metadata, arrays, coverage = [], [], []
    for case in campaign['cases']:
        record = [r for r in case['runs'] if r['name'] == 'natural' and r['status'] == 'completed'][-1]
        path = online/'remote/screen'/('object%d_seed%d' % (case['object'],case['seed']))/record['relative_dir']/'natural.npz'
        assert digest(path) == record['npz_sha256']
        source.append(path)
        with np.load(path, allow_pickle=False) as archive:
            a = {k: archive[k].copy() for k in archive.files if k in ['h_original','opening','distance','contact','step','post_success'] or k.startswith('z__')}
        hits = np.flatnonzero(a['opening'] >= .05)
        onset = int(hits[0]) if len(hits) else None
        success = bool(a['post_success'].any())
        tid = '%d/%d' % (case['object'],case['seed'])
        coverage.append(dict(trajectory=tid,object=case['object'],seed=case['seed'],success=success,onset=onset,
                             steps=len(a['step']),post_onset_steps=0 if onset is None else len(a['step'])-onset))
        for step in range(len(a['step'])):
            metadata.append(dict(trajectory=tid,object=case['object'],seed=case['seed'],state_index=step,
                                 success=success,post_onset=onset is not None and step>=onset,
                                 onset_first5=onset is not None and onset<=step<onset+5,
                                 onset_later=onset is not None and step>=onset+5))
        arrays.append(a)
    frames = pd.DataFrame(metadata)
    h = np.concatenate([a['h_original'] for a in arrays]).astype(float)
    truth = np.concatenate([a['opening'] for a in arrays])
    with np.load(projection_path,allow_pickle=False) as p:
        ridge = ((h-p['h_mean'])/p['h_scale'])@p['rotation']
    reps = {'raw':h,'ridge':ridge}
    residuals, code_checks = {}, []
    for folder in sorted((online/'package/bundle/models').iterdir()):
        source += [folder/'weights.npz',folder/'model.json']
        model = NumericSAE(folder)
        saved = np.concatenate([a['z__'+folder.name] for a in arrays])
        recomputed = np.asarray([model.infer(row)[0][0] for row in h])
        support_differences = np.any((saved>0)!=(recomputed>0),axis=1)
        code_checks.append(dict(model=folder.name,states=len(h),max_abs=float(np.max(abs(saved-recomputed))),
                                changed_support_states=int(support_differences.sum()),
                                rows_exceeding_2e_4=int((np.max(abs(saved-recomputed),axis=1)>2e-4).sum())))
        # Primary readouts use the codes actually recorded during these natural runs.
        reps[folder.name] = saved
        with torch.no_grad():
            rec = F.linear(torch.as_tensor(saved,dtype=torch.float32),model.t['decoder_weight'],model.t['decoder_bias']).numpy()
        rec = rec.astype(float)*model.w['scale']+model.w['mean']
        residuals[folder.name] = h-rec
    source_hashes = {str(p):digest(p) for p in source}
    requested = ['top1','top2','top4','top6','full','complement4','residual','top4_plus_residual']
    scope_masks = {'post_onset':frames.post_onset.to_numpy(),'onset_first5':frames.onset_first5.to_numpy(),
                   'onset_later':frames.onset_later.to_numpy(),'before_onset':~frames.post_onset.to_numpy()}
    errors, max_prediction_error, prediction_arrays = [], 0., {}
    for method,x in reps.items():
        for fit_scope in ['post_onset','local']:
            for representation in requested:
                ident='__'.join([fit_scope,method,'opening_rad',representation])
                if ident not in probes:
                    continue
                artifact=probes[ident]
                if representation=='residual':features=residuals[method]
                elif representation=='top4_plus_residual':features=np.column_stack([x[:,artifact['coordinates']],residuals[method]])
                else:features=x[:,artifact['coordinates']]
                pred,check=predict(artifact,features)
                max_prediction_error=max(max_prediction_error,check)
                prediction_arrays[ident]=pred
                for eval_scope,mask in scope_masks.items():
                    for tid in frames.loc[mask,'trajectory'].unique():
                        q=mask&frames.trajectory.eq(tid).to_numpy()
                        f=frames.loc[q].iloc[0]
                        errors.append(dict(fit_scope=fit_scope,eval_scope=eval_scope,method=method,
                                           family=method.split('_seed')[0],representation=representation,
                                           trajectory=tid,object=int(f['object']),success=bool(f.success),
                                           states=int(q.sum()),mae=float(np.mean(abs(pred[q]-truth[q]))),
                                           bias=float(np.mean(pred[q]-truth[q]))))
    error_table=pd.DataFrame(errors)
    # First average the three dictionary seeds within a physical trajectory.
    keys=['fit_scope','eval_scope','family','representation','trajectory','object','success']
    nested=error_table.groupby(keys,as_index=False).agg(mae=('mae','mean'),bias=('bias','mean'),dictionary_seeds=('method','size'),states=('states','first'))
    summaries=[]
    for key,g in nested.groupby(['fit_scope','eval_scope','family','representation']):
        for population,sub in [('all',g),('natural_success',g[g.success]),('natural_failure',g[~g.success])]:
            if sub.empty:continue
            summaries.append(dict(zip(['fit_scope','eval_scope','family','representation'],key),population=population,
                                  trajectories=len(sub),objects=sub.object.nunique(),states=int(sub.states.sum()),
                                  equal_trajectory_mae=float(sub.mae.mean()),equal_object_mae=float(sub.groupby('object').mae.mean().mean()),
                                  mean_bias=float(sub.bias.mean())))
    # Paired seed-within-object bootstrap, keeping the four observed objects fixed.
    primary=nested[(nested.fit_scope=='post_onset')&nested.eval_scope.eq('post_onset')]
    pivot=primary.pivot(index=['trajectory','object'],columns=['family','representation'],values='mae')
    comparisons=[(k,'top1',k,'top4') for k in ['assisted','pure','raw','ridge']]
    comparisons += [(k,'top4',k,'top6') for k in ['assisted','pure','raw','ridge']]
    comparisons += [('assisted','top4',k,'top4') for k in ['pure','raw','ridge']]
    comparisons += [('assisted','top4','assisted',r) for r in ['full','complement4','residual','top4_plus_residual']]
    bootstrap=[]
    rng=np.random.default_rng(20261105)
    for fa,ra,fb,rb in comparisons:
        if (fa,ra) not in pivot or (fb,rb) not in pivot:continue
        delta=(pivot[(fa,ra)]-pivot[(fb,rb)]).dropna()
        object_arrays=[v.to_numpy() for _,v in delta.groupby(level='object')]
        draws=np.mean([rng.choice(v,size=(2000,len(v)),replace=True).mean(axis=1) for v in object_arrays],axis=0)
        bootstrap.append(dict(comparison=fa+'_'+ra+'_minus_'+fb+'_'+rb,trajectories=len(delta),objects=len(object_arrays),
                              equal_trajectory_difference=float(delta.mean()),equal_object_difference=float(np.mean([v.mean() for v in object_arrays])),
                              low=float(np.quantile(draws,.025)),high=float(np.quantile(draws,.975)),
                              interpretation='positive means second readout lower MAE; descriptive, observed objects fixed, no re-fitting or confirmatory testing'))
    # Training-mean constant reference derived from the existing frozen artifact.
    constant=float(probes['post_onset__raw__opening_rad__top1']['y_mean'])
    constant_rows=[]
    for tid in frames.loc[frames.post_onset,'trajectory'].unique():
        q=frames.post_onset.to_numpy()&frames.trajectory.eq(tid).to_numpy()
        constant_rows.append(dict(trajectory=tid,mae=float(np.mean(abs(truth[q]-constant)))))
    for name,table in [('coverage',pd.DataFrame(coverage)),('frames',frames),('trajectory_metrics',error_table),('origin_nested_metrics',nested),
                       ('summary',pd.DataFrame(summaries)),('paired_bootstrap',pd.DataFrame(bootstrap)),('code_cross_runtime_checks',pd.DataFrame(code_checks)),
                       ('constant_reference',pd.DataFrame(constant_rows))]:
        table.to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    np.savez_compressed(out/'frozen_predictions.npz',truth=truth,**prediction_arrays)
    assert all(digest(p)==value for p,value in source_hashes.items())
    # Independently reconcile trajectory-equal MAE with frame-weighted arithmetic.
    ident='post_onset__assisted_seed0__opening_rad__top4'
    q=frames.post_onset.to_numpy(); groups=frames.loc[q,'trajectory'].to_numpy()
    counts=pd.Series(groups).value_counts()
    weights=np.array([1./counts[g] for g in groups])
    direct=float(np.average(abs(prediction_arrays[ident][q]-truth[q]),weights=weights))
    from_table=float(error_table[(error_table.fit_scope=='post_onset')&error_table.eval_scope.eq('post_onset')&error_table.method.eq('assisted_seed0')&error_table.representation.eq('top4')].mae.mean())
    assert abs(direct-from_table)<1e-12
    dump(out/'verification.json',dict(passed=True,natural_origins=len(coverage),states=len(h),post_onset_origins=int(sum(r['onset'] is not None for r in coverage)),
         post_onset_states=int(q.sum()),new_fits=0,all_natural_outcomes_retained=True,prediction_formula_max_error=max_prediction_error,
         trajectory_weight_reconciliation_error=abs(direct-from_table),source_hashes=source_hashes,
         current_source_unchanged=True,primary_code_source='recorded during actual natural simulation; local recoding sensitivity reported separately',
         limits=['post hoc new-environment-seed evaluation on the same four development objects; no holdout-object claim',
                 'natural rollouts stop at first task success, so observation ranges differ with outcome; trajectory equality does not remove this conditioning',
                 'post-onset probes are applied without refitting; before-onset and first-five phase cuts are exploratory scope transfers',
                 'complement/residual have more dimensions than Top4; informativeness does not establish causal control']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=pd.DataFrame(summaries)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained')
    for ax,population in zip(axes,['all','natural_success']):
        for family,color,marker in [('assisted','#d55e00','o'),('pure','#0072b2','s'),('raw','#009e73','^'),('ridge','#6f4c9b','D')]:
            rows=s[(s.fit_scope=='post_onset')&s.eval_scope.eq('post_onset')&s.family.eq(family)&s.population.eq(population)].set_index('representation')
            ax.plot([1,2,4,6],[rows.loc['top'+str(n),'equal_trajectory_mae'] for n in [1,2,4,6]],color=color,marker=marker,label=family)
        ax.set_xticks([1,2,4,6]);ax.set_xlabel('Fixed coordinates (N)');ax.set_ylabel('Opening MAE (rad), equal trajectories')
        ax.set_title('All 31 post-onset origins' if population=='all' else '25 naturally successful origins')
        ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Frozen pre-online readouts on new natural environment seeds; no fitting')
    fig.savefig(out/'new_seed_information_curves.png',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(passed=True,origins=len(coverage),states=len(h),post_onset_states=int(q.sum()),new_fits=0),ensure_ascii=False))


if __name__=='__main__':
    main()
