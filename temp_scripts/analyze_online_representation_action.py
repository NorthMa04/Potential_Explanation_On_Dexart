"""Frozen readouts and instantaneous actor transmission in the online pilot.

This is a read-only, post-hoc analysis, not a simulator, new probe fit, or causal
mediation analysis. First assigned steps share baseline inputs; later rows use
their own current branch h. No outcome-based exposure filtering is performed.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from temp_scripts.physical_runtime import NumericSAE


EPS = 1e-8
METRICS = ['actual_h_norm', 'target_norm_change', 'target_remaining',
           'other_code_l2', 'distance_probe_delta_m', 'opening_probe_delta_rad',
           'any_contact_probe_delta', 'strict_contact_probe_delta',
           'h2_delta_l2', 'h2_gate_flips', 'raw_mu_delta_l2',
           'clipped_action_delta_l2', 'raw_action_gain', 'clip_action_gain']


def native(path):
    path = Path(path).resolve()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def display(path):
    value = str(path)
    return value[4:] if value.startswith('\\\\?\\') else value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-16 else np.nan


def sign_label(value):
    return 'positive' if value > EPS else 'negative' if value < -EPS else 'zero'


def direct_predict(model, z):
    """Independent affine coefficients in raw-code units, plus sigmoid if needed."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    results = []
    for i, probe in enumerate(model.meta['probes']):
        coords = np.asarray(probe['coordinates'])
        slope = model.w['probe%d_coef' % i].reshape(-1) / model.w['z_scale'][coords]
        intercept = model.w['probe%d_intercept' % i].reshape(-1)[0] - model.w['z_mean'][coords] @ slope
        value = z[:, coords] @ slope + intercept
        if probe['classification']:
            value = 1. / (1. + np.exp(-np.clip(value, -700, 700)))
        else:
            value = value * probe['target_scale'] + probe['target_mean']
        results.append(value)
    return np.column_stack(results)


def strict_predict(probe, z):
    z = np.atleast_2d(np.asarray(z, float))[:, probe['coordinates']]
    scaler, model = probe['scaler'], probe['model']
    slope = model.coef_.reshape(-1) / scaler.scale_
    offset = model.intercept_[0] - scaler.mean_ @ slope
    logits = z @ slope + offset
    return 1. / (1. + np.exp(-np.clip(logits, -700, 700)))


def actor(weights, h):
    h = np.asarray(h, dtype=np.float64)
    pre = h @ weights['mlp_extractor.policy_net.2.weight'].T + weights['mlp_extractor.policy_net.2.bias']
    middle = np.maximum(pre, 0.)
    mu = middle @ weights['action_net.weight'].T + weights['action_net.bias']
    return pre, middle, mu, np.clip(mu, -1., 1.)


def summarize(frame, out, prefix):
    """Each model/condition first averages its assigned frames, then seeds/origins."""
    keys = ['object', 'environment_seed', 'baseline_stratum', 'family', 'model',
            'group', 'control', 'duration']
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, key))
        row.update(assigned_frames=len(group), nonzero_frames=int(group.nonzero_h.sum()),
                   positive_opening_frames=int(group.opening_direction.eq('positive').sum()),
                   negative_opening_frames=int(group.opening_direction.eq('negative').sum()),
                   zero_opening_frames=int(group.opening_direction.eq('zero').sum()))
        row.update({field: float(group[field].mean()) for field in METRICS})
        rows.append(row)
    condition = pd.DataFrame(rows)
    condition.to_csv(out / (prefix + '_model_condition.csv'), index=False, encoding='utf-8-sig')
    origin_keys = ['object', 'environment_seed', 'baseline_stratum', 'family', 'group', 'control', 'duration']
    origin_rows = []
    for key, group in condition.groupby(origin_keys, dropna=False, sort=True):
        row = dict(zip(origin_keys, key))
        row.update(assigned_model_seeds=len(group), models_with_any_nonzero=int(group.nonzero_frames.gt(0).sum()))
        row.update({field: float(group[field].mean()) for field in METRICS})
        origin_rows.append(row)
    origin = pd.DataFrame(origin_rows)
    origin.to_csv(out / (prefix + '_origin.csv'), index=False, encoding='utf-8-sig')
    summary_keys = ['baseline_stratum', 'family', 'group', 'control', 'duration']
    summary = []
    for key, group in origin.groupby(summary_keys, dropna=False, sort=True):
        row = dict(zip(summary_keys, key))
        row.update(origins=len(group), objects=group.object.nunique())
        row.update({field: float(group[field].mean()) for field in METRICS})
        summary.append(row)
    pd.DataFrame(summary).to_csv(out / (prefix + '_origin_weighted_summary.csv'), index=False, encoding='utf-8-sig')
    return condition, origin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('outputs/online_development_20261104'))
    parser.add_argument('--out', type=Path, default=Path('outputs/deeper_analysis_20261105/representation'))
    args = parser.parse_args()
    root, out = native(args.run), native(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    campaign_path = root / 'remote/pilot/campaign.json'
    campaign = read_json(campaign_path)
    plan_path = root / 'pilot_plan.json'
    plan = read_json(plan_path)
    bundle = root / 'package/bundle'
    meta_path = bundle / 'online_intervention.json'
    meta = read_json(meta_path)
    project = native(Path(__file__).resolve().parent.parent)
    strict_path = project / 'outputs/local_interpretability/20261104_mainline_v3/strict_contact_probes.joblib'
    strict_probes = joblib.load(strict_path)
    numeric = {name: NumericSAE(bundle / 'models' / name) for name in meta['models']}
    with np.load(bundle / 'actor.npz', allow_pickle=False) as data:
        weights = {key: data[key].copy().astype(float) for key in data.files}
    outcome = pd.read_csv(root / 'analysis/assigned_origin_outcomes.csv').set_index(['object', 'environment_seed', 'name'])
    sources = {}
    def track(path):
        sources[display(path)] = digest(path)
    for path in [campaign_path, plan_path, meta_path, strict_path, bundle / 'actor.npz',
                 root / 'analysis/assigned_origin_outcomes.csv']:
        track(path)
    for name in numeric:
        track(bundle / 'models' / name / 'model.json')
        track(bundle / 'models' / name / 'weights.npz')
    rows, vector_data = [], {key: [] for key in ['h_delta', 'h2_delta', 'mu_delta', 'action_delta', 'z_before', 'z_after']}
    maxima = dict(actor_mu=0., actor_action=0., actor_h2=0., code_before=0., code_after=0.,
                  old_probe_manual=0., strict_probe_manual=0., same_input_first_step=0.)
    counter = 0
    for case in campaign['cases']:
        obj, seed = case['object'], case['seed']
        source_folder = root / 'remote/screen' / ('object%d_seed%d' % (obj, seed)) / 'natural'
        baseline_path = source_folder / 'natural.npz'
        track(baseline_path)
        with np.load(baseline_path, allow_pickle=False) as data:
            natural = {key: data[key].copy() for key in ['h_original', 'mu', 'action', 'post_opening', 'post_success']}
        trigger = int(case['trigger_step'])
        latest = {record['name']: record for record in case['runs']}
        for record in latest.values():
            if record['name'] == 'noop_gate':
                continue
            condition = record['condition']
            model_name = condition['model']
            model = numeric[model_name]
            strict = strict_probes[model_name + '__opening_top4']
            assert strict['coordinates'] == meta['models'][model_name]['opening_coordinates'][:4]
            coords = np.asarray(meta['models'][model_name]['opening_coordinates'][:meta['groups'][condition['group']]])
            other = np.ones(64, dtype=bool); other[coords] = False
            path = root / 'remote/pilot' / ('object%d_seed%d' % (obj, seed)) / record['relative_dir'] / (record['name'] + '.npz')
            track(path)
            assert sources[display(path)] == record['npz_sha256']
            result = outcome.loc[obj, seed, record['name']]
            with np.load(path, allow_pickle=False) as data:
                indices = np.flatnonzero(data['scheduled'])
                assert len(indices) and indices[0] == trigger
                assert record['exit_code'] == 0
                same_error = float(np.max(abs(data['h_original'][trigger] - natural['h_original'][trigger])))
                maxima['same_input_first_step'] = max(maxima['same_input_first_step'], same_error)
                assert same_error == 0.
                h0, h1 = data['h_original'][indices].astype(float), data['h_applied'][indices].astype(float)
                z0, z1 = data['edit_z_before'][indices], data['edit_z_after'][indices]
                pred0, pred1 = model.predict(z0), model.predict(z1)
                manual0, manual1 = direct_predict(model, z0), direct_predict(model, z1)
                maxima['old_probe_manual'] = max(maxima['old_probe_manual'], float(np.max(abs(manual0 - pred0))), float(np.max(abs(manual1 - pred1))))
                strict0, strict1 = strict_predict(strict, z0), strict_predict(strict, z1)
                strict_sklearn = strict['model'].predict_proba(strict['scaler'].transform(z1[:, strict['coordinates']]))[:, 1]
                maxima['strict_probe_manual'] = max(maxima['strict_probe_manual'], float(np.max(abs(strict_sklearn - strict1))))
                pre0, mid0, mu0, act0 = actor(weights, h0)
                pre1, mid1, mu1, act1 = actor(weights, h1)
                maxima['actor_mu'] = max(maxima['actor_mu'], float(np.max(abs(mu1 - data['mu'][indices]))))
                maxima['actor_action'] = max(maxima['actor_action'], float(np.max(abs(act1 - data['action'][indices]))))
                maxima['actor_h2'] = max(maxima['actor_h2'], float(np.max(abs(mid1 - data['h2'][indices]))))
                for j, t in enumerate(indices):
                    before = model.infer(h0[j])[0][0]
                    after = model.infer(h1[j])[0][0]
                    maxima['code_before'] = max(maxima['code_before'], float(np.max(abs(before - z0[j]))))
                    maxima['code_after'] = max(maxima['code_after'], float(np.max(abs(after - z1[j]))))
                    dh, dm, da = h1[j] - h0[j], mu1[j] - mu0[j], act1[j] - act0[j]
                    hnorm = float(np.linalg.norm(dh))
                    target_before, target_after = np.linalg.norm(z0[j, coords]), np.linalg.norm(z1[j, coords])
                    expected = -1 if condition['control'] in ['decrease', 'ridge_decrease'] else 1 if condition['control'] in ['bounded_increase', 'ridge_increase'] else 0
                    row = dict(row=counter, object=obj, environment_seed=seed, baseline_stratum=case['selection']['baseline_stratum'],
                               branch=record['name'], archive=str(path.relative_to(root)), frame=int(t), relative_step=int(t-trigger),
                               first_assigned_step=(t == trigger), model=model_name, family=model_name.split('_seed')[0],
                               model_seed=int(model_name[-1]), group=condition['group'], control=condition['control'], duration=condition['duration'],
                               scheduled=True, applied=bool(data['applied'][t]), edit_status=int(data['edit_status_code'][t]), nonzero_h=hnorm>EPS,
                               baseline_success=bool(natural['post_success'].any()), success=bool(result.success), flipped=bool(result.flipped),
                               actual_h_norm=hnorm, recorded_budget=float(data['edit_budget'][t]),
                               target_before_l2=float(target_before), target_after_l2=float(target_after),
                               target_norm_change=float(target_after-target_before), target_remaining=float(target_after/target_before) if target_before>EPS else np.nan,
                               other_code_l2=float(np.linalg.norm((z1[j]-z0[j])[other])),
                               distance_probe_before_m=float(pred0[j,0]), opening_probe_before_rad=float(pred0[j,1]), any_contact_probe_before=float(pred0[j,2]), strict_contact_probe_before=float(strict0[j]),
                               distance_probe_delta_m=float(pred1[j,0]-pred0[j,0]), opening_probe_delta_rad=float(pred1[j,1]-pred0[j,1]),
                               any_contact_probe_delta=float(pred1[j,2]-pred0[j,2]), strict_contact_probe_delta=float(strict1[j]-strict0[j]),
                               actual_current_opening_rad=float(data['opening'][t]), actual_current_distance_m=float(data['distance'][t]),
                               actual_current_any_contact=bool(data['contact'][t]), actual_current_strict_contact=bool(data['strict_contact'][t]),
                               h2_delta_l2=float(np.linalg.norm(mid1[j]-mid0[j])), h2_gate_flips=int(np.sum((pre0[j]>0)!=(pre1[j]>0))),
                               h2_gates_on=int(np.sum((pre0[j]<=0)&(pre1[j]>0))), h2_gates_off=int(np.sum((pre0[j]>0)&(pre1[j]<=0))),
                               raw_mu_delta_l2=float(np.linalg.norm(dm)), clipped_action_delta_l2=float(np.linalg.norm(da)),
                               raw_action_gain=float(np.linalg.norm(dm)/hnorm) if hnorm>EPS else np.nan,
                               clip_action_gain=float(np.linalg.norm(da)/hnorm) if hnorm>EPS else np.nan,
                               original_saturated_actions=int(np.sum(abs(mu0[j])>=1)), edited_saturated_actions=int(np.sum(abs(mu1[j])>=1)),
                               opening_direction=sign_label(pred1[j,1]-pred0[j,1]), expected_readout_sign=expected,
                               opening_expected_sign_ok=bool((pred1[j,1]-pred0[j,1])*expected>EPS) if expected else None)
                    if t == trigger:
                        for offset in [0,4]:
                            key = 'physical_opening_delta_after%d_rad' % (offset+1)
                            row[key] = float(data['post_opening'][t+offset]-natural['post_opening'][t+offset]) if t+offset<len(data['step']) and t+offset<len(natural['post_opening']) else np.nan
                    rows.append(row)
                    for key,value in [('h_delta',dh),('h2_delta',mid1[j]-mid0[j]),('mu_delta',dm),('action_delta',da),('z_before',z0[j]),('z_after',z1[j])]:
                        vector_data[key].append(value)
                    counter += 1
            print('READ', obj, seed, record['name'], len(indices), flush=True)
    table = pd.DataFrame(rows)
    assert len(table.branch.drop_duplicates()) == 33
    assert table[['object','environment_seed','branch','frame']].duplicated().sum() == 0
    assert int(table.first_assigned_step.sum()) == 231
    table.to_csv(out/'scheduled_readout_action_rows.csv',index=False,encoding='utf-8-sig')
    vectors = {key:np.asarray(value) for key,value in vector_data.items()}
    np.savez_compressed(out/'scheduled_readout_action_vectors.npz',**vectors)
    first = table[table.first_assigned_step].copy()
    pulse = table[table.duration.eq(5)].copy()
    primary_first = first[first.group.eq('top4') & first.duration.eq(5)].copy()
    primary_pulse = pulse[pulse.group.eq('top4')].copy()
    first.to_csv(out/'first_assigned_step.csv',index=False,encoding='utf-8-sig')
    summarize(primary_first,out,'primary_first')
    summarize(primary_pulse,out,'primary_pulse')
    # Rows have not been removed because of support. Conditional sign proportions
    # and cosine diagnostics separately disclose their nonzero denominators.
    sign_rows=[]
    for scope,frame in [('first',primary_first),('pulse',primary_pulse)]:
        for key, group in frame.groupby(['family','control']):
            sign_rows.append(dict(scope=scope,family=key[0],control=key[1],assigned_rows=len(group),
                                  physical_origins=group[['object','environment_seed']].drop_duplicates().shape[0],
                                  nonzero_rows=int(group.nonzero_h.sum()),
                                  positive=int(group.opening_direction.eq('positive').sum()),negative=int(group.opening_direction.eq('negative').sum()),zero=int(group.opening_direction.eq('zero').sum()),
                                  expected_sign_positive=(int(group.opening_expected_sign_ok.fillna(False).sum()) if not key[1].startswith('random') else None),
                                  mean_opening_probe_delta_rad=float(group.opening_probe_delta_rad.mean()),
                                  mean_strict_contact_probe_delta=float(group.strict_contact_probe_delta.mean())))
    signs=pd.DataFrame(sign_rows)
    signs.to_csv(out/'opening_direction_counts.csv',index=False,encoding='utf-8-sig')
    pair_rows=[]
    for key,group in primary_first[primary_first.family.eq('assisted')].groupby(['object','environment_seed','model']):
        indexed=group.set_index('control')
        for target,signed_ridge in [('decrease','ridge_decrease'),('bounded_increase','ridge_increase')]:
            a=indexed.loc[target]
            for comparator in ['random0','random1','random2',signed_ridge]:
                b=indexed.loc[comparator]; i,j=int(a.row),int(b.row)
                assert abs(a.actual_h_norm-b.actual_h_norm)<2e-6
                pair_rows.append(dict(object=key[0],environment_seed=key[1],model=key[2],target=target,comparator=comparator,
                                      target_row=i,comparator_row=j,actual_h_norm=float(a.actual_h_norm), both_nonzero=bool(a.nonzero_h and b.nonzero_h),
                                      h_cosine=cosine(vectors['h_delta'][i],vectors['h_delta'][j]),
                                      h2_cosine=cosine(vectors['h2_delta'][i],vectors['h2_delta'][j]),
                                      raw_mu_cosine=cosine(vectors['mu_delta'][i],vectors['mu_delta'][j]),
                                      clipped_action_cosine=cosine(vectors['action_delta'][i],vectors['action_delta'][j]),
                                      raw_gain_ratio=float(a.raw_action_gain/b.raw_action_gain) if b.raw_action_gain>EPS else np.nan,
                                      clipped_gain_ratio=float(a.clip_action_gain/b.clip_action_gain) if b.clip_action_gain>EPS else np.nan,
                                      target_opening_probe_delta=float(a.opening_probe_delta_rad),comparator_opening_probe_delta=float(b.opening_probe_delta_rad),
                                      target_strict_probe_delta=float(a.strict_contact_probe_delta),comparator_strict_probe_delta=float(b.strict_contact_probe_delta),
                                      target_outcome_success=bool(a.success),comparator_outcome_success=bool(b.success)))
    pairs=pd.DataFrame(pair_rows)
    pairs.to_csv(out/'first_step_same_input_control_pairs.csv',index=False,encoding='utf-8-sig')
    pair_origins=pairs.groupby(['object','environment_seed','target','comparator']).agg(
        assigned_seeds=('model','size'),nonzero_seeds=('both_nonzero','sum'),
        h_cosine=('h_cosine','mean'),raw_mu_cosine=('raw_mu_cosine','mean'),clipped_action_cosine=('clipped_action_cosine','mean'),
        raw_gain_ratio=('raw_gain_ratio','mean'),clipped_gain_ratio=('clipped_gain_ratio','mean')).reset_index()
    pair_origins.to_csv(out/'first_step_same_input_control_pair_origins.csv',index=False,encoding='utf-8-sig')
    calibration=[]
    for key,group in primary_first[primary_first.control.eq('decrease')].groupby(['object','environment_seed','family']):
        calibration.append(dict(object=key[0],environment_seed=key[1],family=key[2],model_seeds=len(group),
                                mean_predicted_opening=float(group.opening_probe_before_rad.mean()),actual_opening=float(group.actual_current_opening_rad.iloc[0]),
                                opening_mae=float(abs(group.opening_probe_before_rad-group.actual_current_opening_rad).mean()),
                                mean_predicted_strict_contact=float(group.strict_contact_probe_before.mean()),actual_strict_contact=bool(group.actual_current_strict_contact.iloc[0])))
    pd.DataFrame(calibration).to_csv(out/'trigger_probe_calibration.csv',index=False,encoding='utf-8-sig')
    figures(out,primary_first,pairs)
    for path,old in sources.items():
        assert digest(native(path))==old,'Source changed: '+path
    passed=(maxima['actor_mu']<2e-5 and maxima['actor_action']<2e-5 and maxima['actor_h2']<2e-5
            and maxima['code_before']<2e-4 and maxima['code_after']<2e-4
            and maxima['old_probe_manual']<1e-12 and maxima['strict_probe_manual']<1e-12
            and maxima['same_input_first_step']==0.)
    verification=dict(passed=passed,created_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                      assigned_conditions=231,assigned_first_steps=len(first),primary_first_steps=len(primary_first),
                      scheduled_steps=len(table),pulse_steps=len(pulse),physical_origins=7,source_files=len(sources),
                      source_hashes_before_after_unchanged=True,max_abs=maxima,
                      probes_frozen=True,new_probes_fitted=0,first_step_control_pairs=len(pairs),
                      validation_scope='All source archives hashed; all assigned scheduled codes reencoded, independent affine probe formula and NumPy actor checked; first-step source h exact equal; matched-control norms checked.',
                      inference_limits=['Probe predicts current state, not a future motor instruction',
                                        'Strict-contact probe was fitted previously on development post-onset train and selected on validation; reused without refitting',
                                        'Raw hidden/action perturbations after first step are instantaneous differences at each branch current h, not matched trajectories',
                                        '21 model-origin rows per control come from seven physical origins; no condition-level significance tests',
                                        'Readout directions are post-hoc diagnostics; no causal mediation claim'])
    dump(out/'source_hashes.json',sources)
    dump(out/'verification.json',verification)
    write_report(out,table,primary_first,primary_pulse,signs,pairs,maxima)
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    return 0 if passed else 1


def figures(out,first,pairs):
    fig,axes=plt.subplots(1,2,figsize=(11,5.1),constrained_layout=True)
    palette={'decrease':'#6a3d9a','bounded_increase':'#e66101','random0':'#737373','random1':'#969696','random2':'#bdbdbd','ridge_decrease':'#1b9e77','ridge_increase':'#66a61e'}
    source=first[first.family.eq('assisted')]
    for control,group in source.groupby('control',sort=False):
        axes[0].scatter(np.rad2deg(group.opening_probe_delta_rad),group.clipped_action_delta_l2,
                        label=control,c=palette[control],marker='x' if control.startswith('random') else 'o',alpha=.8,s=35)
    axes[0].axvline(0,c='black',lw=.7)
    axes[0].set(xlabel='Frozen opening-state readout change (degrees)',ylabel='Instantaneous clipped-action change (L2)',title='First treatment step: 7 origins x 3 SAE seeds')
    axes[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3,frameon=False)
    target=pairs[pairs.comparator.str.startswith('ridge') & pairs.both_nonzero]
    for name,group in target.groupby('target'):
        axes[1].scatter(group.h_cosine,group.clipped_action_cosine,label=name,c=palette[name],alpha=.8,s=35)
    axes[1].axhline(0,c='black',lw=.7);axes[1].axvline(0,c='black',lw=.7)
    axes[1].set(xlabel='Target vs signed Ridge: h-change cosine',ylabel='Target vs signed Ridge: action-change cosine',
                xlim=(-1.05,1.05),ylim=(-1.05,1.05),title='Equal current h and budget; nonzero pairs only')
    axes[1].legend(fontsize=8,frameon=False)
    fig.savefig(out/'trigger_readout_action.png',dpi=180)
    plt.close(fig)


def write_report(out,table,first,pulse,signs,pairs,maxima):
    lines=['# 在线编辑的状态读出与动作传递','',
           '2026-11-05，北京时间。仅复用冻结模型、探针和已回传轨迹；没有新训练、服务器连接或仿真。', '',
           '本分析把231个处理的全部实际调度帧保留，主比较限定Top4、5步。首个分配步与自然分支的h完全相同，可比较真实随机/Ridge分支；后续步各分支已分叉，本文所报即时动作差是相对该分支当前h不编辑的actor输出，不是相对自然轨迹的同输入效应。', '',
           '三个旧读出分别为当前距离、当前开启量和任一接触，且各用不同的固定Top4；开启读出恰好用处理组Top4，因此其变化不是独立语义认证。另复用11月4日已冻结的开启Top4严格接触开发探针，未按本批结局挑选或重新拟合。当前状态估计变化与未来物理控制需要是两件事。', '',
           '|范围|表示|处理|全部分配帧|非零h编辑|开启读出正/负/零|预期符号命中|平均开启读出变化(rad)|平均严格接触读出变化|',
           '|---|---|---|---:|---:|---|---:|---:|---:|']
    for _,row in signs.iterrows():
        expected = '—' if pd.isna(row.expected_sign_positive) else str(int(row.expected_sign_positive))
        lines.append('|%s|%s|%s|%d|%d|%d/%d/%d|%s|%.5f|%.5f|' % (row.scope,row.family,row.control,row.assigned_rows,row.nonzero_rows,row.positive,row.negative,row.zero,expected,row.mean_opening_probe_delta_rad,row.mean_strict_contact_probe_delta))
    lines += ['', 'first每个操作的21个模型—起点行来自7个物理起点×3个SAE种子；pulse把每条轨迹的五个分配时刻保留。表内帧均值只作为分布描述，推断用的起点等权表另存于primary_first/pulse_origin_weighted_summary.csv。预期符号只针对码下降/增加及有向Ridge；随机方向不设预期。零编辑和零读出变化都留在分母。', '',
              '同输入首步配对底表为first_step_same_input_control_pairs.csv：目标与三个随机方向/同符号Ridge的实际h范数均匹配；非零向量cosine在分母为零时保留缺失，不把零向量伪造为正交。对动作增益的条件比较不能当成“控制住中介后的处理效应”。', '',
              '全体调度帧的原始码、目标组L2变化、非目标码变化、四个冻结读出、h2门翻转及未裁剪/裁剪动作差，见scheduled_readout_action_rows.csv；对应向量在同名npz。first_assigned_step.csv还保存相对自然分支第1/5动作后的物理开启差（缺失时不补终值），用于检验内部当前开启量读出是否就是物理开盖命令。', '',
              '核验：所有输入文件哈希前后一致；全部已分配调度帧重编码、NumPy actor/h2与记录、旧探针的独立raw-code仿射公式、严格接触探针的手算logit均核对。最大差见verification.json。统计单位仍是7起点/4物体，模型种子和帧不是新增独立样本。', '',
              '![同输入首步：状态读出与动作变化](trigger_readout_action.png)', '',
              '本文件给出可复算结果，不将读出增加称为物理开盖增强、不将h2门翻转称为补偿、不按最终成败筛掉无效编辑。主终点和20个破坏/22个救援仍沿用原复合成功判据；这里不重新定义成功。']
    (out/'REPRESENTATION_ACTION_2026-11-05.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    raise SystemExit(main())
