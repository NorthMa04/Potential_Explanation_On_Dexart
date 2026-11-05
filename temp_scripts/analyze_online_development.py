"""Independently audit retrieved online-development evidence and describe outcomes.

This entry point never connects to a server, runs a simulator or edits a frozen
package. It deliberately does not reconstruct SAE codes or requested edit vectors.
Run with the MachineLearning Python environment after retrieving screen/pilot.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ACTOR_ATOL = 2e-5
ACTOR_RTOL = 2e-5
NORM_ATOL = 2e-6
STATE_KEYS = ['step', 'opening', 'distance', 'contact', 'strict_contact', 'contacts',
              'robot_qpos', 'robot_qvel', 'object_qpos', 'object_qvel', 'success', 'progress']
COMMON_TRAJECTORY_KEYS = STATE_KEYS + ['post_' + key for key in STATE_KEYS] + [
    'h_original', 'h_applied', 'mu', 'action', 'reward', 'done', 'observation_state']


def long_path(path):
    """Use Windows extended paths internally, including deeply nested run names."""
    path = Path(path)
    if os.name == 'nt' and str(path).startswith('\\\\?\\'):
        return path
    path = path.resolve()
    if os.name == 'nt':
        value = str(path)
        path = Path('\\\\?\\UNC\\'+value[2:]) if value.startswith('\\\\') else Path('\\\\?\\'+value)
    return path


def readable_path(path):
    value = str(path)
    if value.startswith('\\\\?\\UNC\\'):
        return '\\\\'+value[8:]
    return value[4:] if value.startswith('\\\\?\\') else value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def latest_runs(case):
    return {record['name']: record for record in case.get('runs', [])}


def condition_name(index, condition):
    return 'c%03d_%s_%s_%s_%s_d%d' % (index, condition['model'], condition['group'],
                                      condition['control'], condition['dose'], condition['duration'])


def resolve_remote(run, explicit=None):
    if explicit is not None:
        return long_path(explicit)
    for root in [run/'remote', run]:
        if (root/'screen/campaign.json').is_file() and (root/'pilot/campaign.json').is_file():
            return root
    candidates = sorted((run/'remote').rglob('screen/campaign.json')) if (run/'remote').exists() else []
    candidates = [path.parent.parent for path in candidates if (path.parent.parent/'pilot/campaign.json').is_file()]
    if len(candidates) != 1:
        raise RuntimeError('Need one retrieved root containing screen/campaign.json and pilot/campaign.json; use --remote-root')
    return candidates[0]


def locate(relative, *roots):
    for root in roots:
        candidate = root/relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(relative)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def trajectory_digest(arrays, keys):
    digest = hashlib.sha256()
    for key in sorted(keys):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def compare_array(actual, expected, atol, rtol=0.):
    if actual.shape != expected.shape:
        raise AssertionError('array shape mismatch: %r vs %r' % (actual.shape, expected.shape))
    error = float(np.max(np.abs(actual.astype(float)-expected.astype(float)))) if actual.size else 0.
    if actual.dtype.kind in 'biu' and expected.dtype.kind in 'biu':
        if not np.array_equal(actual, expected):
            raise AssertionError('integer/Boolean arrays differ')
    elif not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError('max_abs=%g exceeds atol=%g, rtol=%g' % (error, atol, rtol))
    return error


def mechanism(arrays, onset):
    goal = arrays['post_progress'] > .95
    near = arrays['post_distance'] <= .2
    contacts = arrays['post_contacts']
    fingers = contacts[:, :-1].sum(axis=1)
    palm = contacts[:, -1].astype(bool)
    strict = (fingers >= 2) & palm
    successful = goal & near & strict
    goal_near = goal & near
    if successful.any():
        label = 'success'
    elif onset is None:
        label = 'no_onset_failure'
    elif not goal.any():
        label = 'post_onset_failure_progress_not_reached'
    elif not goal_near.any():
        label = 'post_onset_failure_goal_without_near'
    else:
        deficits = set('palm_only' if fingers[t] >= 2 and not palm[t]
                       else 'fingers_only' if fingers[t] < 2 and palm[t] else 'palm_and_fingers'
                       for t in np.flatnonzero(goal_near))
        label = 'post_onset_failure_goal_near_without_strict_'+'|'.join(sorted(deficits))
    indices = np.flatnonzero(goal_near)
    strict_after = np.flatnonzero(strict & (np.arange(len(strict)) > indices[-1])) if len(indices) else np.array([], dtype=int)
    return dict(success=bool(successful.any()), success_step=first(successful), mechanism_class=label,
                progress_goal_reached=bool(goal.any()), goal_and_near=bool(goal_near.any()),
                goal_and_strict=bool((goal & strict).any()), strict_contact_anywhere=bool(strict.any()),
                goal_near_steps=int(goal_near.sum()), goal_strict_steps=int((goal_near & strict).sum()),
                goal_missing_palm_steps=int((goal_near & ~palm).sum()),
                goal_insufficient_finger_steps=int((goal_near & (fingers < 2)).sum()),
                first_goal_near_step=int(indices[0]) if len(indices) else None,
                strict_before_first_goal=bool(strict[:indices[0]].any()) if len(indices) else None,
                strict_after_last_goal=bool(len(strict_after)) if len(indices) else None,
                first_later_strict_gap=int(strict_after[0]-indices[-1]) if len(strict_after) else None,
                max_opening_rad=float(arrays['post_opening'].max()), final_opening_rad=float(arrays['post_opening'][-1]))


def audit_arrays(arrays, actor, trigger_threshold, condition=None, frozen_trigger=None):
    required = set(COMMON_TRAJECTORY_KEYS + ['applied'])
    missing = required-set(arrays)
    if missing:
        raise AssertionError('Missing trajectory fields: '+repr(sorted(missing)))
    obs_keys = sorted(key for key in arrays if key.startswith('obs__'))
    if not obs_keys:
        raise AssertionError('Full observations missing')
    length = len(arrays['step'])
    if not length or not np.array_equal(arrays['step'], np.arange(length)):
        raise AssertionError('Empty or noncontiguous full trajectory')
    if not np.array_equal(arrays['post_step'], np.arange(length)+1):
        raise AssertionError('Post-action step misalignment')
    if not np.array_equal(arrays['done'].astype(bool), np.arange(length) == length-1):
        raise AssertionError('Rollout did not terminate exactly once at its last step')
    for key, value in arrays.items():
        if len(value) != length or not np.isfinite(value).all():
            raise AssertionError('Nonfinite or inconsistent length: '+key)
    predicate_errors = 0
    for prefix in ['', 'post_']:
        contacts = arrays[prefix+'contacts']
        if contacts.shape != (length, 5) or not np.isin(contacts, [0, 1]).all():
            raise AssertionError('Unexpected contact matrix')
        strict = (contacts[:, :-1].sum(axis=1) >= 2) & contacts[:, -1].astype(bool)
        success = (arrays[prefix+'progress'] > .95) & (arrays[prefix+'distance'] <= .2) & strict
        predicate_errors += int(np.count_nonzero(strict != arrays[prefix+'strict_contact']))
        predicate_errors += int(np.count_nonzero(success != arrays[prefix+'success']))
    if predicate_errors:
        raise AssertionError('Independent contact/composite endpoint disagrees')
    h = arrays['h_applied'].astype(np.float32)
    if h.shape != (length, 64) or arrays['h_original'].shape != (length, 64) or (h < 0).any() or (arrays['h_original'] < 0).any():
        raise AssertionError('Invalid nonnegative hidden activation')
    # Independent NumPy implementation; no frozen helper/editor is called.
    h2 = np.maximum(h @ actor['mlp_extractor.policy_net.2.weight'].T + actor['mlp_extractor.policy_net.2.bias'], 0)
    mu = h2 @ actor['action_net.weight'].T + actor['action_net.bias']
    mu_error = compare_array(arrays['mu'], mu, ACTOR_ATOL, ACTOR_RTOL)
    action_error = compare_array(arrays['action'], np.clip(mu, -1, 1), ACTOR_ATOL, ACTOR_RTOL)
    recorded_clip_error = compare_array(arrays['action'], np.clip(arrays['mu'], -1, 1), NORM_ATOL)
    h2_error = compare_array(arrays['h2'], h2, ACTOR_ATOL, ACTOR_RTOL) if 'h2' in arrays else None
    if condition is not None and 'h2' not in arrays:
        raise AssertionError('Pilot branch h2 missing')
    actual_norm = np.linalg.norm(arrays['h_applied'].astype(float)-arrays['h_original'].astype(float), axis=1)
    norm_errors = {}
    if 'delta_norm' in arrays:
        norm_errors['delta_norm'] = compare_array(arrays['delta_norm'], actual_norm, NORM_ATOL)
    if np.any(actual_norm > .1*np.linalg.norm(arrays['h_original'].astype(float), axis=1)+NORM_ATOL):
        raise AssertionError('Actual h edit exceeds frozen relative norm cap')
    scheduled = arrays.get('scheduled', np.zeros(length, dtype=bool)).astype(bool)
    applied = arrays['applied'].astype(bool)
    if condition is None:
        if applied.any() or actual_norm.max() != 0:
            raise AssertionError('Natural baseline unexpectedly edited h')
    else:
        expected = np.arange(length) >= frozen_trigger
        if condition['duration'] != -1:
            expected &= np.arange(length) < frozen_trigger+condition['duration']
        compare_array(scheduled, expected, 0)
        for key in ['edit_status_code','edit_budget','edit_actual_norm','edit_norm_error']:
            if key not in arrays:
                raise AssertionError('Editing audit metadata missing: '+key)
        status = arrays['edit_status_code']
        if not np.isin(status, np.arange(7)).all() or (status[~scheduled] != 0).any() or (status[scheduled] == 0).any():
            raise AssertionError('Editing status/schedule mismatch')
        compare_array(applied, scheduled & (status == 1), 0)
        budget = arrays['edit_budget']
        if (budget < 0).any() or (actual_norm > budget+NORM_ATOL).any():
            raise AssertionError('Invalid budget or actual edit exceeds budget')
        norm_errors['edit_actual_norm'] = compare_array(arrays['edit_actual_norm'], actual_norm, NORM_ATOL)
        norm_errors['edit_norm_error'] = compare_array(arrays['edit_norm_error'], np.abs(actual_norm-budget), NORM_ATOL)
        compare_array(actual_norm[status == 1], budget[status == 1], NORM_ATOL)
        if actual_norm[~applied].max(initial=0.) > NORM_ATOL:
            raise AssertionError('Unapplied/unscheduled h changed')
    onset = first(arrays['opening'] >= trigger_threshold)
    outcome = mechanism(arrays, onset)
    nonzero = actual_norm > 1e-9
    if not scheduled.any():
        exposure = 'not_scheduled'
    elif not nonzero.any():
        exposure = 'all_zero_actual_exposure'
    elif np.count_nonzero(nonzero & scheduled) < np.count_nonzero(scheduled):
        exposure = 'partial_actual_exposure'
    else:
        exposure = 'full_actual_exposure'
    return dict(**outcome, steps=length, onset_step=onset,
                scheduled_steps=int(scheduled.sum()), applied_steps=int(applied.sum()),
                actual_nonzero_steps=int(nonzero.sum()), ineffective_scheduled_steps=int((scheduled & ~applied).sum()),
                zero_actual_applied_steps=int((applied & ~nonzero).sum()), exposure_class=exposure,
                cumulative_actual_h_norm=float(actual_norm.sum()), max_actual_h_norm=float(actual_norm.max()),
                max_mu_error=mu_error, max_action_error=action_error, max_recorded_clip_error=recorded_clip_error,
                max_h2_error=h2_error, max_norm_error=max(norm_errors.values(), default=0.), predicate_errors=predicate_errors,
                exact_h_action_sha256=trajectory_digest(arrays, ['h_applied','action']),
                full_trajectory_sha256=trajectory_digest(arrays, COMMON_TRAJECTORY_KEYS+obs_keys))


def audit_prefix(natural, branch, trigger, atol, noop=False):
    errors = {}
    if noop:
        fields = COMMON_TRAJECTORY_KEYS+[key for key in natural if key.startswith('obs__')]
        for key in fields:
            errors[key] = compare_array(branch[key], natural[key], atol)
    else:
        # Current observations/state/h must match through the trigger action input.
        for key in STATE_KEYS+['h_original','observation_state']+[key for key in natural if key.startswith('obs__')]:
            errors[key] = compare_array(branch[key][:trigger+1], natural[key][:trigger+1], atol)
        # Actions and post-action states must match strictly before treatment.
        for key in ['h_applied','mu','action','reward','done']+['post_'+key for key in STATE_KEYS]:
            errors[key] = compare_array(branch[key][:trigger], natural[key][:trigger], atol)
    return dict(prefix_max_abs=max(errors.values(), default=0.), compared_arrays=len(errors),
                prefix_scope='full_noop_trajectory' if noop else 'all_pretrigger_outputs_and_trigger_input')


def check_report(report, case, condition, baseline_report, plan_hash, reference_hash, metadata_hash, prefix_atol):
    if report.get('status') != 'fresh_process_rollout_completed':
        raise AssertionError('Incomplete child report')
    for flag in ['prefix_gate_passed','actor_parameters_unchanged','full_policy_state_unchanged']:
        if report.get(flag) is not True:
            raise AssertionError('Missing/failed report flag: '+flag)
    expected = dict(object=case['object'], seed=case['seed'], model_name=condition['model'],
                    group_name=condition['group'], control=condition['control'], dose=condition['dose'],
                    duration=condition['duration'], group_edit=True, source_plan_sha256=plan_hash,
                    reference_sha256=reference_hash, online_metadata_sha256=metadata_hash)
    for field, value in expected.items():
        if report.get(field) != value:
            raise AssertionError('Report assignment/hash mismatch: '+field)
    for field in ['checkpoint_sha256','full_policy_state_sha256']:
        if not baseline_report.get(field) or report.get(field) != baseline_report[field]:
            raise AssertionError('Policy hash differs from selected baseline: '+field)
    if not report.get('prefix_max_abs') or max(report['prefix_max_abs'].values()) > prefix_atol:
        raise AssertionError('Report prefix gate errors missing or over tolerance')


def csv_output(out, name, rows, columns=None):
    table = pd.DataFrame(rows, columns=columns) if columns is not None else pd.DataFrame(rows)
    table.to_csv(out/name, index=False, encoding='utf-8-sig')
    return table


def summarize(out, rows):
    table = pd.DataFrame(rows)
    # All assigned rows stay in primary tables, regardless of effective exposure.
    table.to_csv(out/'assigned_origin_outcomes.csv', index=False, encoding='utf-8-sig')
    groups = ['baseline_stratum','family','group','control','dose','duration']
    nested_rows = []
    for key, group in table.groupby(['object','environment_seed']+groups, dropna=False):
        valid = group[group.completed.fillna(False)]
        complete = len(valid) == len(group)
        planned_seeds=sorted(int(seed) for seed in group.model_seed.unique())
        seed_scope=('three_seeds_nested_within_origin' if planned_seeds==[0,1,2]
                    else 'seed0_development_mechanism_only' if planned_seeds==[0] else 'other_explicit_seed_subset')
        nested_rows.append(dict(zip(['object','environment_seed']+groups, key), assigned_model_seeds=len(group),
                                planned_model_seed_ids=json.dumps(planned_seeds),seed_scope=seed_scope,
                                completed_model_seeds=len(valid), complete_assigned_outcomes=complete,
                                success_fraction=float(valid.success.astype(float).mean()) if complete else None,
                                flip_fraction=float(valid.flipped.astype(float).mean()) if complete else None,
                                all_zero_conditions=int(group.exposure_class.eq('all_zero_actual_exposure').sum()),
                                partial_exposure_conditions=int(group.exposure_class.eq('partial_actual_exposure').sum())))
    nested = csv_output(out, 'origin_seed_nested_outcomes.csv', nested_rows)
    aggregate_rows = []
    for key, group in table.groupby(groups, dropna=False):
        valid = group[group.completed.fillna(False)]
        aggregate_rows.append(dict(zip(groups,key), assigned_conditions=len(group), completed_conditions=len(valid),
                                   physical_origins=group[['object','environment_seed']].drop_duplicates().shape[0],
                                   objects=group.object.nunique(), success_to_failure_conditions=int(valid.success_to_failure.sum()),
                                   failure_to_success_conditions=int(valid.failure_to_success.sum()),
                                   all_zero_conditions=int(group.exposure_class.eq('all_zero_actual_exposure').sum()),
                                   partial_exposure_conditions=int(group.exposure_class.eq('partial_actual_exposure').sum()),
                                   invalid_condition_count=int((~group.audit_passed.fillna(False)).sum())))
    aggregate = csv_output(out, 'assigned_endpoint_summary.csv', aggregate_rows)
    origin_summary = []
    for key, group in nested.groupby(groups, dropna=False):
        valid = group[group.complete_assigned_outcomes]
        origin_summary.append(dict(zip(groups,key), assigned_origins=len(group), complete_origins=len(valid),
                                   seed_scope='|'.join(sorted(set(group.seed_scope))),
                                   mean_origin_success_fraction=float(valid.success_fraction.mean()) if len(valid)==len(group) else None,
                                   mean_origin_flip_fraction=float(valid.flip_fraction.mean()) if len(valid)==len(group) else None,
                                   missing_origins=len(group)-len(valid)))
    csv_output(out, 'origin_weighted_summary.csv', origin_summary)
    pairs=[]
    for _, target in table.iterrows():
        if target.family != 'assisted' or target.control not in ['decrease','bounded_increase']:
            continue
        peers=table[table.object.eq(target.object) & table.environment_seed.eq(target.environment_seed)
                    & table.model.eq(target.model) & table.group.eq(target.group)
                    & table.dose.eq(target.dose) & table.duration.eq(target.duration)]
        pure=table[table.object.eq(target.object) & table.environment_seed.eq(target.environment_seed)
                   & table.model.eq('pure_seed%d' % target.model_seed) & table.group.eq(target.group)
                   & table.dose.eq(target.dose) & table.duration.eq(target.duration) & table.control.eq(target.control)]
        ridge = 'ridge_decrease' if target.control == 'decrease' else 'ridge_increase'
        for comparator, controls, expected in [('random_mean',peers[peers.control.isin(['random0','random1','random2'])],3),
                                                ('signed_ridge',peers[peers.control.eq(ridge)],1),('pure_same_seed',pure,1)]:
            if len(controls) == 0:
                continue  # This comparator was never assigned for the seed0 mechanism subconditions.
            complete=bool(target.completed and len(controls)==expected and controls.completed.fillna(False).all())
            pairs.append(dict(object=int(target.object),environment_seed=int(target.environment_seed),
                              baseline_stratum=target.baseline_stratum,model_seed=int(target.model_seed),group=target.group,
                              target_control=target.control,dose=target.dose,duration=int(target.duration),comparator=comparator,
                              complete_assigned_pair=complete,assigned_controls=expected,observed_controls=len(controls),
                              target_success=target.success if complete else None,
                              control_mean_success=float(controls.success.astype(float).mean()) if complete else None,
                              target_minus_control_success=float(target.success)-float(controls.success.astype(float).mean()) if complete else None,
                              target_minus_control_flip=float(target.flipped)-float(controls.flipped.astype(float).mean()) if complete else None))
    pairs = csv_output(out, 'seed_paired_comparisons.csv', pairs)
    pair_keys=['baseline_stratum','group','target_control','dose','duration','comparator']
    origin_pairs=[]
    if len(pairs):
        for key, group in pairs.groupby(['object','environment_seed']+pair_keys,dropna=False):
            complete=bool(group.complete_assigned_pair.all())
            origin_pairs.append(dict(zip(['object','environment_seed']+pair_keys,key),assigned_model_seeds=len(group),
                                     complete_model_seed_pairs=int(group.complete_assigned_pair.sum()),complete_origin_pair=complete,
                                     target_minus_control_success=float(group.target_minus_control_success.mean()) if complete else None,
                                     target_minus_control_flip=float(group.target_minus_control_flip.mean()) if complete else None))
    origin_pairs=csv_output(out,'origin_seed_nested_pairs.csv',origin_pairs)
    paired_summary=[]
    if len(origin_pairs):
        for key, group in origin_pairs.groupby(pair_keys,dropna=False):
            valid=group[group.complete_origin_pair]
            complete=len(valid)==len(group)
            paired_summary.append(dict(zip(pair_keys,key),assigned_origins=len(group),complete_origins=len(valid),
                                       objects=group.object.nunique(),
                                       mean_paired_success_difference=float(valid.target_minus_control_success.mean()) if complete else None,
                                       mean_paired_flip_difference=float(valid.target_minus_control_flip.mean()) if complete else None,
                                       smallest_origin_flip_difference=float(valid.target_minus_control_flip.min()) if complete else None,
                                       largest_origin_flip_difference=float(valid.target_minus_control_flip.max()) if complete else None,
                                       weighting='average model seeds within origin, then equal physical origins; no exposure filtering'))
    csv_output(out,'paired_origin_summary.csv',paired_summary)
    return table, aggregate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--remote-root',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    run=long_path(args.run)
    out=long_path(args.out) if args.out else run/'analysis'
    out.mkdir(parents=True,exist_ok=True)
    remote=resolve_remote(run,args.remote_root)
    screen_path=remote/'screen/campaign.json'
    pilot_path=remote/'pilot/campaign.json'
    selection_path=remote/'screen/selected_origins.json'
    plan_path=locate('pilot_plan.json',remote,run)
    screen_plan_path=locate('screen_plan.json',remote,run)
    manifest_path=locate('package/protocol_hashes.json',remote,run)
    bundle=manifest_path.parent/'bundle'
    actor_path=bundle/'actor.npz'
    actor=load_npz(actor_path)
    screen,pilot,selection,plan,screen_plan=map(read_json,[screen_path,pilot_path,selection_path,plan_path,screen_plan_path])
    selected=selection.get('origins',selection.get('selected_origins',[])) if isinstance(selection,dict) else selection
    conditions=plan['conditions']
    errors=[]
    tracked_inputs=[screen_path,pilot_path,selection_path,plan_path,screen_plan_path,manifest_path,actor_path]
    inputs={readable_path(path):sha(path) for path in tracked_inputs}
    def fail(scope,error):
        errors.append(dict(scope=scope,error=str(error)))
    if screen.get('status')!='completed' or pilot.get('status')!='completed':
        fail('campaign_completion','Screen/pilot must both be completed; missing assignments will be retained')
    if len(selected)>8 or len({(item['object'],item['seed']) for item in selected})!=len(selected):
        fail('selection','Selection exceeds eight or duplicates physical origins')
    # Read the explicit assignment list; seed0 mechanism branches do not acquire
    # unassigned seed1/2 replicas merely because the primary comparison has three.
    if len({json.dumps(item,sort_keys=True) for item in conditions})!=len(conditions):
        fail('plan','Duplicated condition assignment')
    if plan.get('version')=='online_bidirectional_development_v1':
        for family,controls in [('assisted',['decrease','bounded_increase','random0','random1','random2','ridge_decrease','ridge_increase']),
                                ('pure',['decrease','bounded_increase'])]:
            for control in controls:
                observed={int(item['model'].split('_seed')[1]) for item in conditions
                          if item['model'].startswith(family+'_seed') and item['group']=='top4'
                          and item['duration']==5 and item['dose']=='full' and item['control']==control}
                if observed!={0,1,2}:
                    fail('main_seed_population','Frozen main Top4 %s/%s must assign all three model seeds' % (family,control))
    for field,path in [('plan_sha256',plan_path),('selection_sha256',selection_path),('protocol_sha256',manifest_path)]:
        if pilot.get(field)!=sha(path):
            fail('pilot_identity',field+' mismatch')
    manifest=read_json(manifest_path)
    for relative,expected in manifest.items():
        path=(manifest_path.parent/relative).resolve()
        try:
            if manifest_path.parent.resolve() not in path.parents or sha(path)!=expected:
                raise AssertionError('Frozen manifest file mismatch: '+relative)
        except Exception as error:
            fail('frozen_package',error)
    screen_cases={(case['object'],case['seed']):case for case in screen['cases']}
    pilot_cases={(case['object'],case['seed']):case for case in pilot['cases']}
    if len(screen_cases)!=len(screen['cases']) or len(pilot_cases)!=len(pilot['cases']):
        fail('campaign_keys','Duplicate physical origin in campaign')
    if set(pilot_cases)!={(item['object'],item['seed']) for item in selected}:
        fail('pilot_population','Pilot origin list differs from frozen selected list')
    initial_seeds=screen_plan['initial_seeds']
    initial_keys={(obj,seed) for obj in screen_plan['objects'] for seed in initial_seeds}
    initial_failures=sum(bool(case.get('natural',{}).get('post_onset_failure')) for key,case in screen_cases.items() if key in initial_keys)
    should_extend=(initial_failures < screen_plan.get('minimum_post_onset_failures',4)
                   and screen_plan.get('allow_single_expansion',True))
    expected_screen_keys=set(initial_keys)
    if should_extend:
        expected_screen_keys.update((obj,seed) for obj in screen_plan['objects'] for seed in screen_plan['expansion_seeds'])
    if set(screen_cases)!=expected_screen_keys or bool(screen.get('extension_used'))!=should_extend:
        fail('screen_population','Screening population/extension differs from the single frozen expansion rule')
    screen_rows=[]
    for (obj,seed),case in screen_cases.items():
        natural=case.get('natural',{})
        if case.get('status')!='completed' or natural.get('post_onset_failure')!=(not natural.get('success') and bool(natural.get('has_trigger'))):
            fail('screen_population','Incomplete or inconsistent screening stratum for %d/%d' % (obj,seed))
        screen_rows.append(dict(object=obj,environment_seed=seed,status=case.get('status'),stage=case.get('stage'),
                                success=natural.get('success'),has_trigger=natural.get('has_trigger'),
                                post_onset_failure=natural.get('post_onset_failure'),mechanism_class=natural.get('mechanism_class')))
    csv_output(out,'screen_population.csv',screen_rows)
    expected_selected=set()
    for obj in screen_plan['objects']:
        available=sorted([case for (object_id,_),case in screen_cases.items() if object_id==obj],key=lambda case:case['seed'])
        for stratum in ['success','post_onset_failure']:
            eligible=[case for case in available if case.get('status')=='completed' and case.get('natural',{}).get('has_trigger')
                      and (case['natural'].get('success') if stratum=='success' else case['natural'].get('post_onset_failure'))]
            if eligible:
                expected_selected.add((obj,eligible[0]['seed'],stratum))
    if expected_selected!={(item['object'],item['seed'],item['baseline_stratum']) for item in selected}:
        fail('selection_rule','Selected origins do not follow first numerical seed per object and natural outcome stratum')
    metadata_hash=sha(bundle/'online_intervention.json')
    plan_hash=sha(plan_path)
    baseline_rows=[]
    assigned=[]
    audit_rows=[]
    duplicates=[]
    mechanism_rows=[]
    audit_steps=0
    for selected_item in selected:
        obj,seed=selected_item['object'],selected_item['seed']
        scope='object%d_seed%d' % (obj,seed)
        case=pilot_cases.get((obj,seed),dict(object=obj,seed=seed,runs=[]))
        s_case=screen_cases.get((obj,seed),{})
        base_path=(remote/'screen'/selected_item.get('natural_relative_path',scope+'/natural/natural.npz')).resolve()
        baseline=None
        baseline_report=None
        baseline_success=None
        trigger=None
        reference_hash=None
        try:
            if sha(base_path)!=selected_item['natural_npz_sha256'] or case.get('reference_sha256')!=sha(base_path):
                raise AssertionError('Frozen selected baseline NPZ hash mismatch')
            natural_record=latest_runs(s_case).get('natural',{})
            if natural_record.get('npz_sha256')!=sha(base_path) or natural_record.get('report_sha256')!=sha(base_path.with_name('report.json')):
                raise AssertionError('Natural baseline differs from screening recorded evidence hashes')
            baseline_report=read_json(base_path.with_name('report.json'))
            if baseline_report.get('status')!='fresh_process_rollout_completed' or any(baseline_report.get(flag) is not True for flag in ['actor_parameters_unchanged','full_policy_state_unchanged']):
                raise AssertionError('Natural policy invariance report missing')
            if baseline_report.get('object')!=obj or baseline_report.get('seed')!=seed or baseline_report.get('source_plan_sha256')!=sha(screen_plan_path):
                raise AssertionError('Natural report identity/plan mismatch')
            if case.get('baseline_report_sha256')!=sha(base_path.with_name('report.json')):
                raise AssertionError('Pilot stored baseline report hash mismatch')
            baseline=load_npz(base_path)
            base_audit=audit_arrays(baseline,actor,plan['trigger_opening_rad'])
            audit_steps+=base_audit['steps']
            baseline_success=base_audit['success']
            trigger=base_audit['onset_step']
            reference_hash=sha(base_path)
            if baseline_success!=baseline_report['summary']['success']:
                raise AssertionError('Natural report success differs from independent predicate')
            if baseline_success!=(selected_item['baseline_stratum']=='success') or trigger!=selected_item['trigger_step']:
                raise AssertionError('Selected baseline outcome/trigger mismatch')
            baseline_rows.append(dict(object=obj,environment_seed=seed,baseline_stratum=selected_item['baseline_stratum'],**base_audit))
            duplicates.append(dict(object=obj,environment_seed=seed,name='natural',kind='baseline',**{key:base_audit[key] for key in ['exact_h_action_sha256','full_trajectory_sha256']}))
            audit_rows.append(dict(object=obj,environment_seed=seed,name='natural',audit_passed=True,**base_audit))
        except Exception as error:
            fail(scope+'/natural',error)
        latest=latest_runs(case)
        noop=dict(model='assisted_seed0',group='top4',control='noop',dose='full',duration=5)
        schedule=[('noop_gate',noop,-1)]+[(condition_name(index,condition),condition,index) for index,condition in enumerate(conditions)]
        for name,condition,index in schedule:
            record=latest.get(name)
            family,model_seed=condition['model'].split('_seed')
            row=dict(object=obj,environment_seed=seed,baseline_stratum=selected_item['baseline_stratum'],baseline_success=baseline_success,
                     condition_index=index,name=name,model=condition['model'],family=family,model_seed=int(model_seed),
                     group=condition['group'],control=condition['control'],dose=condition['dose'],duration=condition['duration'],
                     assigned=True,recorded=record is not None,completed=False,audit_passed=False,
                     exit_code=record.get('exit_code') if record else None,success=None,flipped=None,
                     success_to_failure=None,failure_to_success=None,exposure_class='unknown',audit_error=None)
            try:
                if record is None or record.get('exit_code')!=0:
                    raise AssertionError('Assigned rollout missing or nonzero exit code')
                folder=remote/'pilot'/scope/record['relative_dir']
                archive=folder/(name+'.npz')
                report_path=folder/'report.json'
                if sha(archive)!=record.get('npz_sha256') or sha(report_path)!=record.get('report_sha256'):
                    raise AssertionError('Pilot recorded NPZ/report hash mismatch')
                if record.get('condition')!=condition:
                    raise AssertionError('Campaign stored condition differs from frozen plan')
                if baseline is None or baseline_report is None or baseline_success is None or trigger is None:
                    raise AssertionError('Selected baseline not independently verified')
                report=read_json(report_path)
                check_report(report,case,condition,baseline_report,plan_hash,reference_hash,metadata_hash,plan['restore_repeat_atol'])
                arrays=load_npz(archive)
                independent=audit_arrays(arrays,actor,plan['trigger_opening_rad'],condition,trigger)
                independent.update(audit_prefix(baseline,arrays,trigger,plan['restore_repeat_atol'],noop=(index==-1)))
                if independent['success']!=report['summary']['success'] or independent['steps']!=report['summary']['steps']:
                    raise AssertionError('Pilot report summary disagrees with independent arrays')
                if report.get('trigger_step')!=trigger:
                    raise AssertionError('Child reference trigger differs from natural trigger')
                audit_steps+=independent['steps']
                row.update(independent,completed=True,audit_passed=True,flipped=(independent['success']!=baseline_success),
                           success_to_failure=bool(baseline_success and not independent['success']),
                           failure_to_success=bool(not baseline_success and independent['success']),
                           npz_sha256=sha(archive),report_sha256=sha(report_path))
                audit_rows.append(dict(object=obj,environment_seed=seed,name=name,audit_passed=True,**independent))
                duplicates.append(dict(object=obj,environment_seed=seed,name=name,kind='noop' if index==-1 else 'assigned',
                                       **{key:independent[key] for key in ['exact_h_action_sha256','full_trajectory_sha256']}))
                mechanism_rows.append(dict(object=obj,environment_seed=seed,baseline_stratum=selected_item['baseline_stratum'],
                                           name=name,family=family,control=condition['control'],group=condition['group'],duration=condition['duration'],
                                           mechanism_class=independent['mechanism_class'],success=independent['success']))
            except Exception as error:
                row['audit_error']=str(error)
                fail(scope+'/'+name,error)
            if index>=0:
                assigned.append(row)
            else:
                dump(out/(scope+'_noop_audit.json'),row)
    table,aggregate=summarize(out,assigned)
    csv_output(out,'selected_baselines.csv',baseline_rows)
    csv_output(out,'independent_trajectory_audits.csv',audit_rows)
    duplicates=csv_output(out,'trajectory_hashes.csv',duplicates)
    duplicate_groups=[]
    if len(duplicates):
        for key,group in duplicates.groupby(['object','environment_seed','full_trajectory_sha256']):
            duplicate_groups.append(dict(object=key[0],environment_seed=key[1],full_trajectory_sha256=key[2],
                                         assigned_conditions=int(group.kind.eq('assigned').sum()),conditions_including_baseline_noop=len(group),
                                         names='|'.join(group.name)))
    csv_output(out,'complete_trajectory_duplicate_groups.csv',duplicate_groups)
    mechanism_table=csv_output(out,'mechanism_strata.csv',mechanism_rows)
    if len(mechanism_table):
        mechanism_table.groupby(['baseline_stratum','family','control','group','duration','mechanism_class'],dropna=False).size().rename('condition_count').reset_index().to_csv(out/'mechanism_strata_counts.csv',index=False,encoding='utf-8-sig')
    expected_assignments=len(selected)*len(conditions)
    if len(assigned)!=expected_assignments or not all(row['completed'] for row in assigned):
        fail('assignment_completeness','Not all %d assigned conditions independently verified' % expected_assignments)
    for path in tracked_inputs:
        if sha(path)!=inputs[readable_path(path)]:
            fail('read_only_source','Source changed while being read: '+readable_path(path))
    selected_strata=Counter(item['baseline_stratum'] for item in selected)
    successful=table[table.completed.fillna(False)]
    verification=dict(passed=not errors,status='passed_with_explicit_audit_boundary' if not errors else 'needs_revision',
                      screen_status=screen.get('status'),pilot_status=pilot.get('status'),screened_origins=len(screen_cases),
                      selected_development_origins=len(selected),selected_objects=len({item['object'] for item in selected}),
                      selected_baseline_strata=dict(selected_strata),conditions_per_origin=len(conditions),
                      assigned_condition_rows=len(assigned),verified_assigned_conditions=int(table.completed.sum()),
                      verified_baselines=len(baseline_rows),verified_trajectory_count=len(audit_rows),independently_audited_steps=audit_steps,
                      success_to_failure_conditions=int(successful.success_to_failure.sum()),
                      failure_to_success_conditions=int(successful.failure_to_success.sum()),
                      full_zero_exposure_conditions=int(table.exposure_class.eq('all_zero_actual_exposure').sum()),
                      partial_exposure_conditions=int(table.exposure_class.eq('partial_actual_exposure').sum()),
                      max_mu_error=max((row.get('max_mu_error',0.) for row in audit_rows),default=0.),
                      max_action_error=max((row.get('max_action_error',0.) for row in audit_rows),default=0.),
                      max_h2_error=max((row.get('max_h2_error') or 0. for row in audit_rows),default=0.),
                      max_norm_error=max((row.get('max_norm_error',0.) for row in audit_rows),default=0.),
                      source_hashes=inputs,source_files_unchanged=not any(error['scope']=='read_only_source' for error in errors),
                      actor_tolerances=dict(atol=ACTOR_ATOL,rtol=ACTOR_RTOL),norm_atol=NORM_ATOL,
                      error_count=len(errors),errors=errors,
                      completed_checks=['all selected baseline and pilot/noop NPZ composite endpoints and termination',
                                        'independent NumPy actor tail, recorded h2 where present, raw mu and clipped actions',
                                        'actual h norms, 10 percent cap, assigned schedule and exposure bookkeeping',
                                        'recorded NPZ/report/plan/metadata/reference hashes and reported parameter invariance',
                                        'all untreated prefix state/observations/actions and full noop trajectory',
                                        'frozen package manifest, development selection and all assigned denominators'],
                      not_audited=['full per-step SAE encoding and TopK boundary stability',
                                   'requested edit direction reconstruction and train-support/code bound compliance',
                                   'independent full policy checkpoint tensor rehash (report invariance flags/hashes are reconciled)',
                                   'all nonselected screening NPZ arrays (screening campaign denominators are retained)'],
                      statistics='Development selection, at most eight physical origins; SAE seeds nested within origins; no exposure-based outcome filtering; no confirmatory tests',
                      full_trajectory_hash_fields=COMMON_TRAJECTORY_KEYS+['all obs__ keys; h2 and edit/model diagnostics excluded so natural/noop can match'])
    dump(out/'verification.json',verification)
    lines=['# 在线开发实验结果（草稿）','',
           '2026-11-04，北京时间。本报告由回传证据独立复算生成。', '',
           '**这批是按自然结局筛选的开发起点，不是留出测试。** 筛查%d个起点，选择%d个起点（自然成功%d、开启后自然失败%d），每起点分配%d个处理条件；空操作另作管线门限。独立统计单位为起点/物体，SAE种子嵌套在起点内。' % (len(screen_cases),len(selected),selected_strata.get('success',0),selected_strata.get('post_onset_failure',0),len(conditions)), '',
           '全部%d个分配条件保留；其中%d个完成独立核验，%d个全零实际暴露、%d个部分实际暴露。主结果未按处理后有效性筛选。缺失或核验失败保留为未知，不当作任务失败。' % (len(assigned),int(table.completed.sum()),verification['full_zero_exposure_conditions'],verification['partial_exposure_conditions']), '',
           '在已核验条件中，成功→失败%d个、自然失败→成功%d个。它们是**条件数量**，不同模型、方向和时长共享起点，不能当作相同数量的独立复现。' % (verification['success_to_failure_conditions'],verification['failure_to_success_conditions']), '',
           '|自然基线分层|表示|TopN|处理|时长|分配条件|成功→失败|失败→成功|',
           '|---|---|---|---|---:|---:|---:|---:|']
    primary=aggregate[aggregate['group'].eq('top4') & aggregate.duration.eq(5)]
    for _,row in primary.iterrows():
        lines.append('|%s|%s|%s|%s|%d|%d|%d|%d|' % (row.baseline_stratum,row.family,row['group'],row.control,row.duration,row.assigned_conditions,row.success_to_failure_conditions,row.failure_to_success_conditions))
    lines += ['', '逐起点全部分配条件结局见`assigned_origin_outcomes.csv`。三SAE种子先在同起点内平均，再对起点等权；配对摘要见`paired_origin_summary.csv`及其逐种子、逐起点底表。只有实际分配的随机/Ridge/纯SAE对照参与对应配对；未分配的机制子条件对照不补造。', '',
              'Top1/Top6及持续处理是预先只分配seed0的开发机制条件，汇总中标为`seed0_development_mechanism_only`；不把未分配的seed1/2当缺失，也不与三种子主Top4混为同一复现口径。', '',
              '失败按目标进度、近距离、手掌与手指组接触是否同一步成立分层，见`mechanism_strata_counts.csv`。进度是目标角度接近程度，不是单调开启量；码增加/减少也不等于物理开盖动作增强/抑制。完整轨迹数值重复见`complete_trajectory_duplicate_groups.csv`，相同轨迹不作为独立机制病例。', '',
              '独立核验状态：**%s**。已核对%d条完整轨迹/%d动作步的复合成功、冻结actor尾部与动作裁剪、实际h扰动范数、未处理前缀、空操作完整重放、文件哈希及参数不变报告。%s' % (verification['status'],len(audit_rows),audit_steps,('发现%d个问题，详见`verification.json`；当前不得据此收紧结论。' % len(errors)) if errors else '在上述范围内未发现不一致。'), '',
              '**审计边界：尚未逐步独立重编码SAE，也未重建请求编辑方向及train支持边界。** 因此这里不声称全量码级审计通过、精确删除语义或闭环补偿。完整策略参数不变依赖子进程报告及其哈希一致性，独立数值重算覆盖冻结actor尾部。历史TopK近并列异常不会由本批物理终点核验改标通过。', '',
              '本草稿用于开发结果整理。即便观察到救援，也需保留同起点随机/Ridge/纯SAE比较及机制过程，不能将本批筛选起点包装成泛化确认。']
    (out/'ONLINE_RESULTS_2026-11-04.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({key:verification[key] for key in ['passed','screened_origins','selected_development_origins','assigned_condition_rows','verified_assigned_conditions','success_to_failure_conditions','failure_to_success_conditions','error_count']},ensure_ascii=False))
    return 0 if verification['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
