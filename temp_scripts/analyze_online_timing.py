"""Read-only timing sensitivity analysis of the 2026-11-04 online pilot.

The real success definition stays fixed. Alternative thresholds and nonzero
contact lags are retrospective diagnostics on the observed, terminated prefix.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __package__:
    from .analyze_online_development import long_path, readable_path, sha, read_json, dump, latest_runs
else:
    from analyze_online_development import long_path, readable_path, sha, read_json, dump, latest_runs


THRESHOLDS = [.90, .925, .95, .975]
GAPS = [0, 1, 2, 3]
CONTROL_DT = .04  # create_env frame_skip=10 times LaptopEnv scene timestep=.004
KEYS = ['step', 'opening', 'distance', 'contacts', 'strict_contact', 'success', 'progress',
        'post_step', 'post_opening', 'post_distance', 'post_contacts', 'post_strict_contact',
        'post_success', 'post_progress', 'object_qpos', 'object_qvel', 'post_object_qpos',
        'post_object_qvel', 'done', 'applied']


def first(mask):
    ids = np.flatnonzero(mask)
    return int(ids[0]) if len(ids) else None


def finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in KEYS}


def spans(mask):
    padded = np.r_[False, mask, False].astype(int)
    starts, stops = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    return [(int(start), int(stop-1)) for start, stop in zip(starts, stops)]


def nearest(goal, contact):
    goal_ids, contact_ids = np.flatnonzero(goal), np.flatnonzero(contact)
    if not len(goal_ids) or not len(contact_ids):
        return None, None, None, None, None
    lags = contact_ids[None, :] - goal_ids[:, None]
    smallest = int(np.abs(lags).min())
    gi, ci = np.where(np.abs(lags) == smallest)
    return smallest, int(lags[gi, ci].min()), int(lags[gi, ci].max()), int(goal_ids[gi[0]]), int(contact_ids[ci[0]])


def timing_fields(v):
    p, distance = v['post_progress'], v['post_distance']
    strict = (v['post_contacts'][:, :-1].sum(axis=1) >= 2) & v['post_contacts'][:, -1].astype(bool)
    assert np.array_equal(strict, v['post_strict_contact'])
    genuine = (p > .95) & (distance <= .2) & strict
    assert np.array_equal(genuine, v['post_success'])
    assert np.array_equal(v['done'], np.arange(len(p)) == len(p)-1)
    assert np.array_equal(v['step'], np.arange(len(p)))
    for key, array in v.items():
        assert len(array) == len(p) and np.isfinite(array).all(), key
    near = distance <= .2
    goal = (p > .95) & near
    goal_ids = np.flatnonzero(goal)
    contact = strict & near
    lag, signed_min, signed_max, nearest_goal, nearest_contact = nearest(goal, contact)
    window_spans = spans(goal)
    entry = int(goal_ids[0]) if len(goal_ids) else None
    average_velocity = np.rad2deg(v['post_opening'] - v['opening']) / CONTROL_DT
    slope = np.polyfit(v['post_object_qpos'][:, 0], v['post_opening'], 1)[0]
    orientation = 1 if slope > 0 else -1
    sampled_velocity = orientation * np.rad2deg(v['post_object_qvel'][:, 0])
    field = dict(steps=len(p), genuine_success=bool(genuine.any()), genuine_success_step=first(genuine),
                 onset_step=first(v['opening'] >= .05), first_goal_step=entry,
                 first_goal_seconds=(entry+1)*CONTROL_DT if entry is not None else None,
                 goal_reached=bool((p > .95).any()), goal_near_reached=bool(goal.any()),
                 observed_goal_frames=int(goal.sum()), observed_progress_only_frames=int((p > .95).sum()),
                 goal_spell_count=len(window_spans), first_spell_frames=window_spans[0][1]-window_spans[0][0]+1 if window_spans else 0,
                 first_spell_success_right_censored=bool(window_spans and window_spans[0][1] == len(p)-1 and genuine[-1]),
                 first_goal_action_average_deg_s=float(average_velocity[entry]) if entry is not None else None,
                 first_goal_post_instantaneous_deg_s=float(sampled_velocity[entry]) if entry is not None else None,
                 mean_goal_action_average_deg_s=float(average_velocity[goal].mean()) if goal.any() else None,
                 strict_near_ever=bool(contact.any()), strict_ever=bool(strict.any()),
                 min_abs_contact_gap=lag, nearest_contact_lag_min=signed_min, nearest_contact_lag_max=signed_max,
                 nearest_goal_step=nearest_goal, nearest_contact_step=nearest_contact,
                 max_progress_at_strict_near=float(p[contact].max()) if contact.any() else None,
                 first_strict_near_step=first(contact),
                 first_strict_near_after_goal_step=first(contact & (np.arange(len(p)) >= entry)) if entry is not None else None,
                 max_opening_degrees=float(np.rad2deg(v['post_opening']).max()),
                 final_opening_degrees=float(np.rad2deg(v['post_opening'][-1])),
                 goal_entry_opening_degrees=float(np.rad2deg(v['post_opening'][entry])) if entry is not None else None,
                 continued_opening_after_entry_degrees=float(np.rad2deg(v['post_opening'][entry:]).max()-np.rad2deg(v['post_opening'][entry])) if entry is not None else None,
                 max_recorded_opening_speed_deg_s=float(sampled_velocity.max()),
                 median_absolute_recorded_speed_deg_s=float(np.median(np.abs(sampled_velocity))))
    return field, average_velocity, sampled_velocity, strict, near


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('outputs/online_development_20261104'))
    parser.add_argument('--out', type=Path, default=Path('outputs/deeper_analysis_20261105/timing'))
    args = parser.parse_args()
    run, out = long_path(args.run), long_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    remote = run/'remote'
    campaign_path, plan_path = remote/'pilot/campaign.json', remote/'pilot_plan.json'
    selection_path = remote/'screen/selected_origins.json'
    endpoint_path, endpoint_csv = run/'analysis/verification.json', run/'analysis/assigned_origin_outcomes.csv'
    edit_audit_path = run/'analysis_edit_audit/verification.json'
    campaign, plan, selection = map(read_json, [campaign_path, plan_path, selection_path])
    assert campaign['status'] == 'completed'
    assert read_json(endpoint_path)['passed'] and read_json(edit_audit_path)['passed']
    previous = pd.read_csv(endpoint_csv)
    assert len(previous) == 231 and previous.audit_passed.all()
    original_fields = {(int(row.object), int(row.environment_seed), row['name']): row for _, row in previous.iterrows()}
    source_files = {path:sha(path) for path in [campaign_path, plan_path, selection_path, endpoint_path, endpoint_csv, edit_audit_path]}
    simulation_root = Path(__file__).resolve().parents[3]/'research/embodied_sae/runs/laptop_resume_bundle/third_party/dexart-release'
    env_files = [simulation_root/'dexart/env/create_env.py', simulation_root/'dexart/env/sim_env/laptop_env.py',
                 simulation_root/'dexart/env/rl_env/laptop_env.py']
    for path in env_files:
        source_files[path] = sha(path)
    # Confirm both source facts for the conversion; all primary time measurements
    # remain in recorded control steps regardless of the derived seconds.
    assert 'frame_skip=10' in env_files[0].read_text(encoding='utf-8')
    assert 'set_timestep(0.004)' in env_files[1].read_text(encoding='utf-8')
    protocol = dict(version='timing_sensitivity_v1', date='2026-11-05',
                    genuine_success='same post-action state: progress>.95 AND distance<=.2 AND >=2 finger groups plus palm',
                    scope='231 assigned conditions and seven selected natural baselines; no noop duplicates',
                    progress_thresholds=THRESHOLDS, max_abs_lags=GAPS,
                    diagnostic='exists goal time t and strict-contact time u with abs(u-t)<=k; distance<=.2 at BOTH t and u',
                    secondary_diagnostic='strict-contact distance requirement at u omitted, while distance at t retained',
                    control_dt_seconds=CONTROL_DT, time_conversion='create_env default frame_skip=10 times scene dt=.004; saved indices are primary',
                    no_outcome_relabeling=True, no_simulator_calls=True,
                    missing_future='No posttermination carry forward or inferred future states. Success terminates at original endpoint; observed windows are right censored.',
                    unit='Physical origin/object. Dictionary seeds averaged within origin; condition counts are descriptive only.',
                    filtering='All 231 allocations retained including all-zero and partial edits. Diagnostic populations fixed to ORIGINAL natural outcome strata.',
                    causal_boundary='Endpoint decomposition/sensitivity, not an explanation of why contact changed or a counterfactual result under modified task termination.')
    dump(out/'protocol.json', protocol)
    rows, diagnostic_rows, window_rows, frame_rows = [], [], [], []
    source_rows, arrays_cache = [], {}
    agreement_checks = 0
    for case in campaign['cases']:
        obj, seed = case['object'], case['seed']
        baseline_path = remote/'screen'/case['selection']['natural_relative_path']
        baseline_report_path = baseline_path.with_name('report.json')
        baseline_digest = sha(baseline_path)
        assert baseline_digest == case['reference_sha256'] == case['selection']['natural_npz_sha256']
        assert sha(baseline_report_path) == case['baseline_report_sha256']
        baseline_report = read_json(baseline_report_path)
        schedule = [('natural', None, baseline_path, baseline_digest, case['baseline_report_sha256'])]
        latest = latest_runs(case)
        for index, condition in enumerate(plan['conditions']):
            name = 'c%03d_%s_%s_%s_%s_d%d' % (index, condition['model'], condition['group'], condition['control'], condition['dose'], condition['duration'])
            record = latest[name]
            assert record['exit_code'] == 0 and record['condition'] == condition
            archive = remote/'pilot'/('object%d_seed%d' % (obj, seed))/record['relative_dir']/(name+'.npz')
            schedule.append((name, condition, archive, record['npz_sha256'], record['report_sha256']))
        for name, condition, archive, expected_npz, expected_report in schedule:
            actual_hash, report_path = sha(archive), archive.with_name('report.json')
            assert actual_hash == expected_npz and sha(report_path) == expected_report
            source_files[archive], source_files[report_path] = actual_hash, expected_report
            source_rows.append(dict(object=obj,environment_seed=seed,name=name,
                                    archive=readable_path(archive),npz_sha256=actual_hash,report_sha256=expected_report))
            report = read_json(report_path)
            v = load(archive)
            arrays_cache[(obj, seed, name)] = v
            field, average_velocity, sampled_velocity, strict, near = timing_fields(v)
            baseline_success = bool(baseline_report['summary']['success'])
            context = dict(object=obj,environment_seed=seed,name=name,kind='natural' if condition is None else 'assigned',
                           baseline_success=baseline_success,baseline_stratum='success' if baseline_success else 'post_onset_failure',
                           model=condition['model'] if condition else 'natural',family=condition['model'].split('_seed')[0] if condition else 'natural',
                           model_seed=int(condition['model'].split('_seed')[1]) if condition else -1,
                           group=condition['group'] if condition else 'natural',control=condition['control'] if condition else 'natural',
                           duration=condition['duration'] if condition else 0,dose=condition['dose'] if condition else 'none')
            assert field['genuine_success'] == report['summary']['success']
            if condition:
                prior = original_fields[(obj,seed,name)]
                assert field['genuine_success'] == prior.success
                context.update(exposure_class=prior.exposure_class,
                               original_success_to_failure=bool(prior.success_to_failure),original_failure_to_success=bool(prior.failure_to_success))
            else:
                context.update(exposure_class='natural',original_success_to_failure=False,original_failure_to_success=False)
            rows.append(dict(**context,**field))
            for mask_name, mask in [('progress_only',v['post_progress']>.95),('goal_and_near',(v['post_progress']>.95)&near)]:
                for spell_index,(start,end) in enumerate(spans(mask)):
                    window_rows.append(dict(**context,mask=mask_name,spell_index=spell_index,start_step=start,end_step=end,
                                            observed_frames=end-start+1,observed_sample_span_seconds=(end-start)*CONTROL_DT,
                                            nominal_frame_count_seconds=(end-start+1)*CONTROL_DT,
                                            left_censored=start==0,right_censored=end==len(v['step'])-1,
                                            right_censored_by_original_success=bool(end==len(v['step'])-1 and field['genuine_success']),
                                            mean_action_average_deg_s=float(average_velocity[start:end+1].mean()),
                                            mean_post_instantaneous_deg_s=float(sampled_velocity[start:end+1].mean())))
            for threshold in THRESHOLDS:
                goal = (v['post_progress'] > threshold) & near
                gap, signed_min, signed_max, goal_t, contact_u = nearest(goal,strict & near)
                raw_gap = nearest(goal,strict)[0]
                for k in GAPS:
                    matched = bool(gap is not None and gap<=k)
                    # Independent Boolean dilation check, distinct from all-pairs lags.
                    dilation = np.convolve((strict&near).astype(int),np.ones(2*k+1,dtype=int),mode='same')>0
                    assert matched == bool((goal & dilation).any())
                    agreement_checks += 1
                    if threshold==.95 and k==0:
                        assert matched == field['genuine_success']
                    diagnostic_rows.append(dict(**context,threshold=threshold,max_abs_lag=k,
                                                diagnostic_match=matched,diagnostic_raw_contact_match=bool(raw_gap is not None and raw_gap<=k),
                                                min_abs_gap=gap,nearest_signed_lag_min=signed_min,nearest_signed_lag_max=signed_max,
                                                nearest_goal_step=goal_t,nearest_strict_near_step=contact_u,
                                                observed_goal_frames=int(goal.sum()),
                                                observed_prefix_original_success=field['genuine_success']))
            # Small source table, all relevant physical states, no visual inputs.
            for t in range(len(v['step'])):
                frame_rows.append(dict(object=obj,environment_seed=seed,name=name,step=t,
                                       post_seconds=(t+1)*CONTROL_DT,progress=float(v['post_progress'][t]),
                                       near=bool(near[t]),strict=bool(strict[t]),strict_and_near=bool(strict[t]&near[t]),
                                       fingers=int(v['post_contacts'][t,:-1].sum()),palm=bool(v['post_contacts'][t,-1]),
                                       opening_degrees=float(np.rad2deg(v['post_opening'][t])),distance=float(v['post_distance'][t]),
                                       action_average_deg_s=float(average_velocity[t]),post_instantaneous_deg_s=float(sampled_velocity[t])))
    timing = pd.DataFrame(rows)
    assert len(timing)==238 and timing.kind.eq('assigned').sum()==231
    diagnostics = pd.DataFrame(diagnostic_rows)
    baselines = timing[timing.kind.eq('natural')].set_index(['object','environment_seed'])
    for name in ['first_goal_step','first_goal_action_average_deg_s','first_strict_near_step','max_opening_degrees']:
        timing['natural_'+name] = [baselines.loc[(r.object,r.environment_seed),name] for r in timing.itertuples()]
        timing['delta_vs_natural_'+name] = timing[name]-timing['natural_'+name]
    natural_labels = diagnostics[diagnostics.kind.eq('natural')][['object','environment_seed','threshold','max_abs_lag','diagnostic_match']].rename(columns={'diagnostic_match':'natural_diagnostic_match'})
    diagnostics = diagnostics.merge(natural_labels,on=['object','environment_seed','threshold','max_abs_lag'],validate='many_to_one')
    diagnostics['diagnostic_success_difference']=diagnostics.diagnostic_match.astype(int)-diagnostics.natural_diagnostic_match.astype(int)
    diagnostics['original_flip_remains_discordant']=np.where(diagnostics.original_success_to_failure,
             diagnostics.natural_diagnostic_match & ~diagnostics.diagnostic_match,
             np.where(diagnostics.original_failure_to_success,~diagnostics.natural_diagnostic_match & diagnostics.diagnostic_match,False))
    timing.to_csv(out/'trajectory_timing.csv',index=False,encoding='utf-8-sig')
    diagnostics.to_csv(out/'diagnostic_sensitivity.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(window_rows).to_csv(out/'observed_window_spells.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(frame_rows).to_csv(out/'physical_frame_source.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(source_rows).to_csv(out/'source_archives.csv',index=False,encoding='utf-8-sig')
    geometry=[]
    for obj in sorted(timing.object.unique()):
        natural = next(v for (oid,seed,name),v in arrays_cache.items() if oid==obj and name=='natural')
        stop = first(natural['post_progress']>.8)
        p,x = natural['post_progress'][:stop+1],natural['post_opening'][:stop+1]
        mask = p>.02
        assert mask.sum()>=2
        denominator, offset = np.polyfit(p[mask],x[mask],1)
        center = denominator+offset
        max_error=0.
        for (oid,seed,name),v in arrays_cache.items():
            if oid==obj:
                reconstructed=1-np.abs(v['post_opening']-center)/abs(denominator)
                max_error=max(max_error,float(np.max(np.abs(reconstructed-v['post_progress']))))
        assert max_error < 1e-6
        geometry.append(dict(object=int(obj),inferred_target_opening_rad=float(center),inferred_progress_distance_scale_rad=float(abs(denominator)),
                             center_degrees=float(np.rad2deg(center)),progress95_full_band_degrees=float(np.rad2deg(.1*abs(denominator))),
                             progress95_low_degrees=float(np.rad2deg(center-.05*abs(denominator))),
                             progress95_high_degrees=float(np.rad2deg(center+.05*abs(denominator))),
                             all_saved_progress_reconstruction_max_abs=max_error,
                             method='fit ascending natural opening versus progress before first .8, verify triangular formula on every saved trajectory of object'))
    pd.DataFrame(geometry).to_csv(out/'inferred_goal_geometry.csv',index=False,encoding='utf-8-sig')
    # Preserve every allocation in diagnostic aggregation, including ineffective edits.
    assigned = diagnostics[diagnostics.kind.eq('assigned')].copy()
    keys=['baseline_stratum','family','group','control','duration','threshold','max_abs_lag']
    origin_rows=[]
    for key,g in assigned.groupby(['object','environment_seed']+keys,dropna=False):
        ids=sorted(int(value) for value in g.model_seed.unique())
        origin_rows.append(dict(zip(['object','environment_seed']+keys,key),assigned_conditions=len(g),
                                planned_model_seeds=json.dumps(ids),seed_scope='three_nested_seeds' if ids==[0,1,2] else 'seed0_mechanism',
                                diagnostic_match_fraction=float(g.diagnostic_match.mean()),natural_diagnostic_match=bool(g.natural_diagnostic_match.iloc[0]),
                                paired_success_difference=float(g.diagnostic_success_difference.mean()),
                                original_flip_count=int((g.original_success_to_failure|g.original_failure_to_success).sum()),
                                remaining_original_flip_count=int(g.original_flip_remains_discordant.sum()),
                                all_zero_conditions=int(g.exposure_class.eq('all_zero_actual_exposure').sum()),
                                partial_conditions=int(g.exposure_class.eq('partial_actual_exposure').sum())))
    origins=pd.DataFrame(origin_rows)
    origins.to_csv(out/'origin_nested_sensitivity.csv',index=False,encoding='utf-8-sig')
    summary_rows=[]
    for key,g in origins.groupby(keys,dropna=False):
        summary_rows.append(dict(zip(keys,key),physical_origins=len(g),objects=g.object.nunique(),
                                 assigned_conditions=int(g.assigned_conditions.sum()),
                                 mean_origin_diagnostic_match=float(g.diagnostic_match_fraction.mean()),
                                 mean_origin_paired_success_difference=float(g.paired_success_difference.mean()),
                                 original_flip_conditions=int(g.original_flip_count.sum()),
                                 remaining_original_flip_conditions=int(g.remaining_original_flip_count.sum()),
                                 all_zero_conditions=int(g.all_zero_conditions.sum()),partial_conditions=int(g.partial_conditions.sum())))
    pd.DataFrame(summary_rows).to_csv(out/'origin_weighted_sensitivity.csv',index=False,encoding='utf-8-sig')
    flips=timing[(timing.kind=='assigned')&(timing.original_success_to_failure|timing.original_failure_to_success)].copy()
    flips.to_csv(out/'original_flip_timing.csv',index=False,encoding='utf-8-sig')
    flip_summary=[]
    for threshold in THRESHOLDS:
        for k in GAPS:
            g=assigned[(assigned.threshold==threshold)&(assigned.max_abs_lag==k)]
            damage=g[g.original_success_to_failure]
            rescue=g[g.original_failure_to_success]
            flip_summary.append(dict(threshold=threshold,max_abs_lag=k,assigned_conditions=len(g),
                                     treatment_diagnostic_matches=int(g.diagnostic_match.sum()),
                                     replicated_natural_diagnostic_matches=int(g.natural_diagnostic_match.sum()),
                                     all_positive_paired_diagnostic_conditions=int((g.diagnostic_success_difference>0).sum()),
                                     all_negative_paired_diagnostic_conditions=int((g.diagnostic_success_difference<0).sum()),
                                     original_damage_conditions=len(damage),original_rescue_conditions=len(rescue),
                                     damage_treatment_diagnostic_matches=int(damage.diagnostic_match.sum()),
                                     damage_remaining_discordances=int(damage.original_flip_remains_discordant.sum()),
                                     rescue_baseline_diagnostic_matches=int(rescue.natural_diagnostic_match.sum()),
                                     rescue_remaining_discordances=int(rescue.original_flip_remains_discordant.sum())))
    pd.DataFrame(flip_summary).to_csv(out/'original_flip_sensitivity.csv',index=False,encoding='utf-8-sig')
    # Main five-step Top4 comparisons retain matching model/group budgets and
    # aggregate dictionary seeds within each physical origin. No condition-level CI.
    plot_rows=[]
    method_map={'decrease':'Assisted down','bounded_increase':'Assisted up','ridge_decrease':'Ridge down','ridge_increase':'Ridge up'}
    main=origins[(origins.family=='assisted')&(origins.group=='top4')&(origins.duration==5)&(origins.threshold==.95)]
    for key,g in main.groupby(['object','environment_seed','baseline_stratum','max_abs_lag']):
        for control,label in method_map.items():
            h=g[g.control==control]
            assert len(h)==1
            plot_rows.append(dict(object=key[0],environment_seed=key[1],baseline_stratum=key[2],lag=key[3],method=label,
                                  paired_success_difference=float(h.paired_success_difference.iloc[0]),assigned_conditions=3))
        h=g[g.control.isin(['random0','random1','random2'])]
        assert len(h)==3
        plot_rows.append(dict(object=key[0],environment_seed=key[1],baseline_stratum=key[2],lag=key[3],method='Random mean',
                              paired_success_difference=float(h.paired_success_difference.mean()),assigned_conditions=9))
    plot_table=pd.DataFrame(plot_rows)
    plot_table.to_csv(out/'main_plot_source.csv',index=False,encoding='utf-8-sig')
    make_plots(out,plot_table,pd.DataFrame(frame_rows),timing)
    damage=flips[flips.original_success_to_failure]
    rescue=flips[flips.original_failure_to_success]
    assert len(damage)==20 and len(rescue)==22
    # Numeric diagnostic monotonicity is checked for every trajectory, not just flips.
    for _,g in diagnostics.groupby(['object','environment_seed','name','threshold']):
        assert np.all(np.diff(g.sort_values('max_abs_lag').diagnostic_match.astype(int))>=0)
    for _,g in diagnostics.groupby(['object','environment_seed','name','max_abs_lag']):
        assert np.all(np.diff(g.sort_values('threshold').diagnostic_match.astype(int))<=0)
    changed=[]
    for path,expected in source_files.items():
        if sha(path)!=expected:
            changed.append(readable_path(path))
    assert not changed
    verification=dict(passed=True,date='2026-11-05',trajectories=238,assigned_conditions=231,natural_baselines=7,
                      post_action_states=sum(len(v['step']) for v in arrays_cache.values()),
                      original_damage_conditions=20,original_rescue_conditions=22,
                      damage_nearest_contact_gap_counts={str(int(k)):int(v) for k,v in damage.min_abs_contact_gap.value_counts().sort_index().items()},
                      damage_observed_goal_width_counts={str(int(k)):int(v) for k,v in damage.observed_goal_frames.value_counts().sort_index().items()},
                      original_rescue_origins=int(rescue[['object','environment_seed']].drop_duplicates().shape[0]),
                      genuine_predicate_mismatches=0,lag_dilation_agreements=agreement_checks,
                      all_allocations_per_diagnostic=231,diagnostic_rows=len(diagnostics),assigned_diagnostic_rows=len(assigned),
                      source_files_unchanged=True,source_hash_count=len(source_files),control_dt_seconds=CONTROL_DT,
                      implementation_sha256=sha(Path(__file__)),
                      boundaries=['Retrospective diagnostic labels only; original task outcomes remain fixed',
                                  'Successful episodes terminate early, making target-window width and stricter-threshold labels right censored',
                                  'No claim that reclassifying near misses establishes their causal mechanism',
                                  'All allocations retained; dictionary seeds and conditions are not independent origins',
                                  'Different models/TopN have different native budgets, and branched trajectories have different cumulative dose',
                                  'Nonzero lag permits two different physical states; it is not proof of stable or task-correct grasp',
                                  'No sub-control-step contacts are recorded; a 40ms grid cannot resolve continuous contact timing'])
    dump(out/'verification.json',verification)
    dump(out/'source_hashes.json',{readable_path(path):value for path,value in source_files.items()})
    dump(out/'summary.json',dict(verification=verification,flip_sensitivity=flip_summary,goal_geometry=geometry))
    print(json.dumps(verification,ensure_ascii=False,indent=2))


def make_plots(out,table,frames,timing):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    colors={'Assisted down':'#0072B2','Assisted up':'#D55E00','Random mean':'#444444','Ridge down':'#009E73','Ridge up':'#CC79A7'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharex=True,sharey=True)
    for ax,stratum,title in zip(axes,['success','post_onset_failure'],['Original successful origins (4)','Original failed origins (3)']):
        for method in colors:
            g=table[(table.baseline_stratum==stratum)&(table.method==method)].groupby('lag').paired_success_difference.mean()
            ax.plot(g.index,100*g.values,marker='o',label=method,color=colors[method],linestyle='--' if 'Ridge' in method else '-')
        ax.axhline(0,color='#888888',lw=.8)
        ax.set(title=title,xlabel='Allowed absolute contact lag (control steps)',xticks=GAPS)
        ax.set_ylim(-40,35)
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Diagnostic match difference (pp)')
    axes[1].legend(loc='best',fontsize=8)
    fig.suptitle('Timing-label sensitivity on the same assigned trajectories; progress > .95',fontsize=12)
    fig.text(.5,.005,'Seeds averaged within physical origins; all allocations retained. Nonzero lags are diagnostics, not new task outcomes.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,.93]);fig.savefig(out/'timing_sensitivity.png');plt.close(fig)
    cases=[(11477,2026110400,['natural','c000_assisted_seed0_top4_decrease_full_d5','c003_assisted_seed0_top4_random1_full_d5','c005_assisted_seed0_top4_ridge_decrease_full_d5'],['Natural','Down s0','Random1 s0','Ridge down s0']),
           (11477,2026110402,['natural','c001_assisted_seed0_top4_bounded_increase_full_d5','c010_assisted_seed1_top4_bounded_increase_full_d5','c003_assisted_seed0_top4_random1_full_d5'],['Natural','Up s0','Up s1','Random1 s0']),
           (11406,2026110402,['natural','c001_assisted_seed0_top4_bounded_increase_full_d5','c032_assisted_seed0_top4_bounded_increase_full_d-1','c013_assisted_seed1_top4_random2_full_d5'],['Natural','Up 5-step s0','Up continuous s0','Random2 s1'])]
    fig,axes=plt.subplots(3,2,figsize=(12,9),gridspec_kw={'width_ratios':[1.7,1]})
    palette=['#000000','#D55E00','#0072B2','#009E73']
    for row,(obj,seed,names,labels) in enumerate(cases):
        anchor=int(timing[(timing.object==obj)&(timing.environment_seed==seed)&(timing.name=='natural')].first_goal_step.iloc[0])
        ax,raster=axes[row]
        for i,(name,label) in enumerate(zip(names,labels)):
            g=frames[(frames.object==obj)&(frames.environment_seed==seed)&(frames.name==name)]
            g=g[(g.step>=anchor-6)&(g.step<=anchor+9)]
            x=g.step-anchor
            ax.plot(x,g.progress,marker='o',ms=3,lw=1.5,label=label,color=palette[i])
            # Contact markers are only saved states with both strict contact and proximity.
            h=g[g.strict_and_near]
            raster.scatter(h.step-anchor,np.full(len(h),i),marker='s',s=28,color=palette[i])
        ax.axhspan(.95,1,color='#dddddd',alpha=.5)
        ax.axhline(.95,color='#666666',ls='--',lw=.8)
        ax.axvline(0,color='#999999',lw=.7)
        ax.set(xlim=(-6.5,9.5),ylim=(.70,1.005),ylabel='Progress',title='%d / %d (natural first goal = step %d)'%(obj,seed,anchor))
        ax.legend(loc='lower left',fontsize=7,ncol=2)
        raster.axvline(0,color='#999999',lw=.7)
        raster.set(xlim=(-6.5,9.5),ylim=(-.5,3.5),yticks=range(4),yticklabels=labels,title='Strict contact AND distance <= .2m')
        raster.grid(axis='x',alpha=.15)
    axes[-1,0].set_xlabel('Control steps relative to natural first goal entry')
    axes[-1,1].set_xlabel('Control steps relative to natural first goal entry')
    fig.suptitle('Observed prefixes: target progress and strict contact timing',fontsize=13)
    fig.text(.5,.01,'Curves stop at original episode termination; no terminal-value carry forward. One control step = 40 ms.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(out/'case_timing.png');plt.close(fig)


if __name__=='__main__':
    main()
