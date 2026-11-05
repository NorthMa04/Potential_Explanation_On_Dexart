"""Post-hoc dose/direction decomposition of already collected online trajectories.

No simulator, new training, or physical counterfactual is run. New matched-dose
results are frozen-actor fixed-input sensitivity calculations only.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from temp_scripts.audit_online_edits import (
    IndependentModel, actor_forward, independent_edit, independent_ray,
    native_path, display, projected,
)
from temp_scripts.project_paths import PROJECT_ROOT


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def cosine(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(np.ravel(a), np.ravel(b)) / denominator) if denominator > 1e-15 else np.nan


def first_difference(a, b, atol=1e-5):
    length = min(len(a), len(b))
    difference = abs(a[:length].astype(float) - b[:length].astype(float))
    maximum = difference if difference.ndim == 1 else difference.reshape(length, -1).max(axis=1)
    found = np.flatnonzero(maximum > atol)
    return int(found[0]) if len(found) else None


KEYS = ['step', 'h_original', 'h_applied', 'h2', 'mu', 'action', 'scheduled', 'applied',
        'edit_budget', 'edit_actual_norm', 'edit_z_before', 'edit_z_after', 'edit_nontarget_code_l2',
        'edit_cap_active', 'post_opening', 'post_distance', 'post_progress', 'post_contacts',
        'post_strict_contact', 'post_success', 'post_robot_qpos', 'post_object_qpos']


def analyze(run, destination):
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    run, out = native_path(run), native_path(destination)
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def read(path):
        hashes[display(path)] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    def array_file(path, keys=None):
        hashes[display(path)] = sha(path)
        with np.load(path, allow_pickle=False) as arrays:
            return {key: arrays[key].copy() for key in (keys or arrays.files) if key in arrays.files}

    campaign = read(run / 'remote/pilot/campaign.json')
    plan = read(run / 'pilot_plan.json')
    previous_audit = read(run / 'analysis_edit_audit/verification.json')
    if not previous_audit['passed'] or campaign['status'] != 'completed':
        raise RuntimeError('Expected completed, independently audited source campaign')
    bundle = run / 'package/bundle'
    metadata = read(bundle / 'online_intervention.json')
    stats = array_file(bundle / 'online_statistics.npz')
    ridge = array_file(bundle / 'combined_directions.npz')['ridge_oriented']
    actor = {key: torch.from_numpy(value) for key, value in array_file(bundle / 'actor.npz').items()}
    model = IndependentModel(bundle / 'models/assisted_seed0')
    for file in ('model.json', 'weights.npz'):
        path = bundle / 'models/assisted_seed0' / file
        hashes[display(path)] = sha(path)
    hashes[display(native_path(__file__))] = sha(native_path(__file__))
    helper_path = native_path(PROJECT_ROOT / 'temp_scripts/audit_online_edits.py')
    hashes[display(helper_path)] = sha(helper_path)
    lookup, natural, summaries, step_rows, source_records = {}, {}, [], [], []
    source_norm_max_error = 0.
    cache = {}

    def load(record):
        path = record['path']
        if path not in cache:
            cache[path] = array_file(path, KEYS)
        return cache[path]

    for case in campaign['cases']:
        obj, seed, trigger = case['object'], case['seed'], case['trigger_step']
        case_root = run / 'remote/pilot' / ('object%d_seed%d' % (obj, seed))
        # The selection path is relative to screen and includes any retry suffix.
        selection = case['selection']
        baseline_path = run / 'remote/screen' / selection['natural_relative_path']
        natural[obj, seed] = array_file(baseline_path, [key for key in KEYS if not key.startswith('edit_')] + ['opening'])
        latest = {record['name']: record for record in case['runs']}
        for old in latest.values():
            condition = old['condition']
            path = case_root / old['relative_dir'] / (old['name'] + '.npz')
            if old['exit_code'] != 0 or sha(path) != old['npz_sha256']:
                raise RuntimeError('Source archive failed its retained completion hash')
            record = {'object': obj, 'seed': seed, 'trigger': trigger, 'name': old['name'],
                      'condition': condition, 'path': path, 'success': old['archive_audit']['success']}
            lookup[obj, seed, condition['model'], condition['group'], condition['control'], condition['duration']] = record
            if condition['control'] == 'noop':
                continue
            source_records.append(record)
            # All 231 treatment trajectories remain in this denominator.
            arrays = array_file(path, KEYS)
            if condition['model'] == 'assisted_seed0' and condition['control'] in ('decrease', 'bounded_increase'):
                cache[path] = arrays
            positions = np.flatnonzero((arrays['step'] >= trigger) & (arrays['step'] < trigger + 5))
            reconstructed_norm = np.linalg.norm(arrays['h_applied'].astype(float) - arrays['h_original'].astype(float), axis=1)
            source_norm_max_error = max(source_norm_max_error, float(np.max(abs(reconstructed_norm - arrays['edit_actual_norm']))))
            summary = {'object': obj, 'seed': seed, 'name': old['name'], **condition,
                       'baseline_success': case['baseline_audit']['success'], 'success': record['success'],
                       'trigger': trigger, 'steps': len(arrays['step']),
                       'first5_frames': len(positions), 'first5_applied': int(arrays['applied'][positions].sum()),
                       'first5_budget_sum': float(arrays['edit_budget'][positions].sum()),
                       'first5_norm_sum': float(arrays['edit_actual_norm'][positions].sum()),
                       'total_norm_sum': float(arrays['edit_actual_norm'].sum()),
                       'first5_nontarget_code_l2_sum': float(arrays['edit_nontarget_code_l2'][positions].sum()),
                       'total_applied': int(arrays['applied'].sum()), 'total_scheduled': int(arrays['scheduled'].sum())}
            indices = np.array(metadata['models'][condition['model']]['opening_coordinates'][:metadata['groups'][condition['group']]])
            for t in positions:
                h2_before, mu_before = actor_forward(actor, arrays['h_original'][t])
                h2_after, mu_after = actor_forward(actor, arrays['h_applied'][t])
                before, after = arrays['edit_z_before'][t], arrays['edit_z_after'][t]
                if condition['control'] == 'decrease':
                    request = -before[indices]
                elif condition['control'] == 'bounded_increase':
                    request = np.where(before[indices] > 0, np.maximum(0., np.minimum(
                        .5 * stats[condition['model'] + '__active_std'][indices],
                        stats[condition['model'] + '__active_q99'][indices] - before[indices])), 0.)
                else:
                    request = np.zeros(len(indices))
                step_rows.append({'object': obj, 'seed': seed, 'name': old['name'], **condition, 'step': int(t),
                                  'applied': bool(arrays['applied'][t]), 'budget': float(arrays['edit_budget'][t]),
                                  'actual_norm': float(arrays['edit_actual_norm'][t]),
                                  'raw_mu_delta_l2': float(np.linalg.norm(mu_after - mu_before)),
                                  'clipped_action_delta_l2': float(np.linalg.norm(np.clip(mu_after, -1, 1) - np.clip(mu_before, -1, 1))),
                                  'h2_gate_flips': int(np.count_nonzero((h2_before > 0) != (h2_after > 0))),
                                  'h2_delta_l2': float(np.linalg.norm(h2_after - h2_before)),
                                  'target_code_change_l2': float(np.linalg.norm(after[indices] - before[indices])),
                                  'target_request_actual_cosine': cosine(request, after[indices] - before[indices]),
                                  'nontarget_code_l2': float(arrays['edit_nontarget_code_l2'][t]),
                                  'target_before_norm': float(np.linalg.norm(before[indices])),
                                  'target_after_norm': float(np.linalg.norm(after[indices]))})
            summaries.append(summary)
    if len(source_records) != 231:
        raise RuntimeError('Expected all 231 treatment branches')
    write_csv(out / 'all_treatment_exposure.csv', summaries)
    write_csv(out / 'first5_step_mechanisms.csv', step_rows)
    pairs, duration_pairs, fixed_states = [], [], []
    field_specs = [('incoming_h', 'h_original', 1e-5), ('emitted_h', 'h_applied', 1e-5),
                   ('mu', 'mu', 1e-5), ('action', 'action', 1e-5), ('h2', 'h2', 1e-5),
                   ('opening', 'post_opening', 1e-5), ('distance', 'post_distance', 1e-5),
                   ('progress', 'post_progress', 1e-5), ('contacts', 'post_contacts', 0),
                   ('strict_contact', 'post_strict_contact', 0)]

    def get(obj, seed, group, control, duration=5):
        return lookup[obj, seed, 'assisted_seed0', group, control, duration]

    for case in campaign['cases']:
        obj, seed, trigger = case['object'], case['seed'], case['trigger_step']
        baseline = natural[obj, seed]
        for t in range(trigger, min(trigger + 5, len(baseline['h_original']))):
            fixed_states.append({'object': obj, 'seed': seed, 'step': t, 'source': 'natural_first5',
                                 'source_control': 'natural', 'h': baseline['h_original'][t], 'pair': ''})
        for control in ('decrease', 'bounded_increase'):
            for left, right in (('top1', 'top4'), ('top4', 'top6'), ('top1', 'top6')):
                r1, r2 = get(obj, seed, left, control), get(obj, seed, right, control)
                a, b = load(r1), load(r2)
                row = {'object': obj, 'seed': seed, 'control': control, 'left': left, 'right': right,
                       'left_success': r1['success'], 'right_success': r2['success'], 'trigger': trigger}
                for label, key, atol in field_specs:
                    row['first_' + label + '_difference'] = first_difference(a[key], b[key], atol)
                    row['first_' + label + '_raw_difference'] = first_difference(a[key], b[key], 0)
                first = row['first_emitted_h_difference']
                if first is not None:
                    row['incoming_h_max_abs_at_first_output_difference'] = float(np.max(abs(a['h_original'][first] - b['h_original'][first])))
                    row['outgoing_delta_cosine_at_first_output_difference'] = cosine(
                        a['h_applied'][first] - a['h_original'][first], b['h_applied'][first] - b['h_original'][first])
                    if obj == 11477 and (left, right) in (('top1', 'top4'), ('top4', 'top6')):
                        fixed_states.append({'object': obj, 'seed': seed, 'step': first, 'source': 'first_group_output_difference',
                                             'source_control': control, 'h': a['h_original'][first], 'pair': left + '_' + right})
                pairs.append(row)
            pulse, sustained = get(obj, seed, 'top4', control, 5), get(obj, seed, 'top4', control, -1)
            a, b = load(pulse), load(sustained)
            end = trigger + 5
            common_arrays = 0
            all_prefix_exact = True
            with np.load(pulse['path'], allow_pickle=False) as aa, np.load(sustained['path'], allow_pickle=False) as bb:
                for key in sorted(set(aa.files) & set(bb.files)):
                    if len(aa[key]) < end or len(bb[key]) < end:
                        all_prefix_exact = False
                    else:
                        all_prefix_exact &= bool(np.array_equal(aa[key][:end], bb[key][:end]))
                    common_arrays += 1
            row = {'object': obj, 'seed': seed, 'control': control, 'trigger': trigger,
                   'pulse_success': pulse['success'], 'sustained_success': sustained['success'],
                   'through_first5_all_common_arrays_exact': all_prefix_exact, 'prefix_compared_arrays': common_arrays,
                   'pulse_total_norm': float(a['edit_actual_norm'].sum()), 'sustained_total_norm': float(b['edit_actual_norm'].sum()),
                   'pulse_steps': len(a['step']), 'sustained_steps': len(b['step'])}
            for label, key, atol in field_specs:
                row['first_' + label + '_difference'] = first_difference(a[key], b[key], atol)
                row['first_' + label + '_raw_difference'] = first_difference(a[key], b[key], 0)
            row['first_palm_difference'] = first_difference(a['post_contacts'][:, -1], b['post_contacts'][:, -1], 0)
            row['first_finger_group_difference'] = first_difference(a['post_contacts'][:, :-1], b['post_contacts'][:, :-1], 0)
            for label, array in (('pulse', a), ('sustained', b)):
                goal = (array['post_progress'] > .95) & (array['post_distance'] <= .2)
                success = goal & array['post_strict_contact']
                for name, mask in (('goal_near', goal), ('strict_contact', array['post_strict_contact']), ('success', success)):
                    hits = np.flatnonzero(mask)
                    row[label + '_first_' + name] = int(hits[0]) if len(hits) else None
                row[label + '_goal_near_palm_missing_steps'] = int(np.sum(goal & ~array['post_contacts'][:, -1].astype(bool)))
            duration_pairs.append(row)
            if obj == 11406 and seed == 2026110402 and row['first_emitted_h_difference'] is not None:
                t = row['first_emitted_h_difference']
                fixed_states.append({'object': obj, 'seed': seed, 'step': t, 'source': 'first_duration_output_difference',
                                     'source_control': control, 'h': a['h_original'][t], 'pair': 'pulse_sustained'})
    write_csv(out / 'group_first_differences.csv', pairs)
    write_csv(out / 'duration_first_differences.csv', duration_pairs)
    fixed_rows, direction_pairs, component_rows, arrays_out = [], [], [], []
    max_ray_error, max_additivity_error = 0., 0.
    for state_index, state in enumerate(fixed_states):
        h = state['h'].astype(np.float64)
        z, _, _ = model.encode(h)
        original_h2, original_mu = actor_forward(actor, h)
        base = {key: value for key, value in state.items() if key != 'h'}
        for control in ('decrease', 'bounded_increase'):
            contexts = {}
            for n in (1, 4, 6):
                group = 'top%d' % n
                condition = {'model': 'assisted_seed0', 'group': group, 'control': control, 'dose': 'full', 'duration': 5}
                value, detail = independent_edit(h, z, model, stats, ridge, metadata, condition, 'unused-target-key')
                coordinates = detail['coordinates']
                if control == 'decrease':
                    request = -z[coordinates]
                else:
                    request = np.where(z[coordinates] > 0, np.maximum(0., np.minimum(
                        .5 * stats['assisted_seed0__active_std'][coordinates],
                        stats['assisted_seed0__active_q99'][coordinates] - z[coordinates])), 0.)
                raw = model.decoder[:, coordinates] @ request
                contexts[n] = {'native': value, 'detail': detail, 'raw': raw, 'request': request}
            positive = [context['detail']['budget'] for context in contexts.values() if context['detail']['budget'] > 1e-9]
            common = min(positive) if positive else 0.
            full_support = len(positive) == 3
            for n, context in contexts.items():
                detail, raw, native = context['detail'], context['raw'], context['native']
                if detail['budget'] > 1e-9 and common > 1e-9:
                    delta, status = independent_ray(h, raw, common)
                    matched = np.maximum(h + delta, 0).astype(np.float32) if status == 'ok' else h.astype(np.float32)
                    max_ray_error = max(max_ray_error, abs(np.linalg.norm(matched.astype(float) - h) - common))
                else:
                    status, matched = 'inactive_retained', h.astype(np.float32)
                context['common'] = matched
                context['common_status'] = status
                for mode, emission in (('native_budget', native), ('common_budget', matched)):
                    after, _, _ = model.encode(emission)
                    h2, mu = actor_forward(actor, emission)
                    coordinates = detail['coordinates']
                    other = np.ones(64, dtype=bool); other[coordinates] = False
                    actual_delta = emission.astype(float) - h
                    fixed_rows.append({**base, 'state_id': state_index, 'control': control, 'n': n, 'mode': mode,
                                       'native_budget': detail['budget'], 'common_positive_budget': common,
                                       'all_three_groups_supported': full_support, 'native_supported': detail['budget'] > 1e-9,
                                       'actual_norm': float(np.linalg.norm(actual_delta)),
                                       'raw_direction_norm': float(np.linalg.norm(raw)),
                                       'target_coordinates': json.dumps(detail['coordinates'].tolist()),
                                       'active_target_coordinates': json.dumps(detail['coordinates'][z[detail['coordinates']] > 0].tolist()),
                                       'requested_code_change': json.dumps(context['request'].tolist()),
                                       'requested_active_coordinates': int(np.count_nonzero(context['request'])),
                                       'full_request_projected_coordinates': int(np.sum(h + raw < 0)),
                                       'raw_negative_h_proposal_fraction': float(np.mean(h + raw < 0)),
                                       'projected_full_request_norm': float(np.linalg.norm(projected(h, raw))),
                                       'projection_cosine': cosine(raw, projected(h, raw)),
                                       'mu_delta_l2': float(np.linalg.norm(mu - original_mu)),
                                       'action_delta_l2': float(np.linalg.norm(np.clip(mu, -1, 1) - np.clip(original_mu, -1, 1))),
                                       'h2_delta_l2': float(np.linalg.norm(h2 - original_h2)),
                                       'h2_gate_flips': int(np.count_nonzero((original_h2 > 0) != (h2 > 0))),
                                       'other_code_l2': float(np.linalg.norm((after - z)[other])),
                                       'target_change_l2': float(np.linalg.norm(after[coordinates] - z[coordinates])),
                                       'request_actual_code_cosine': cosine(context['request'], after[coordinates] - z[coordinates])})
                    arrays_out.append((state_index, n, control, mode, emission, h2, mu, after))
            # Nested unprojected vector additivity is exact by construction;
            # those components need not survive projection/scaling unchanged.
            raw1, raw4, raw6 = (contexts[n]['raw'] for n in (1, 4, 6))
            atoms = {'A_rank1': raw1, 'B_ranks2to4': raw4 - raw1, 'C_ranks5to6': raw6 - raw4}
            max_additivity_error = max(max_additivity_error, float(np.max(abs(sum(atoms.values()) - raw6))))
            for block, vector in atoms.items():
                component_rows.append({**base, 'state_id': state_index, 'control': control, 'block': block,
                                       'raw_component_norm': float(np.linalg.norm(vector)),
                                       'cosine_with_top4': cosine(vector, raw4), 'cosine_with_top6': cosine(vector, raw6)})
            for left, right in ((1, 4), (4, 6), (1, 6)):
                a, b = contexts[left], contexts[right]
                row = {**base, 'state_id': state_index, 'control': control, 'left_n': left, 'right_n': right,
                       'all_three_groups_supported': full_support,
                       'both_supported': a['detail']['budget'] > 1e-9 and b['detail']['budget'] > 1e-9,
                       'raw_direction_cosine': cosine(a['raw'], b['raw']),
                       'projected_request_cosine': cosine(projected(h, a['raw']), projected(h, b['raw'])),
                       'native_delta_cosine': cosine(a['native'].astype(float) - h, b['native'].astype(float) - h),
                       'common_delta_cosine': cosine(a['common'].astype(float) - h, b['common'].astype(float) - h)}
                for mode in ('native', 'common'):
                    h2a, mua = actor_forward(actor, a[mode]); h2b, mub = actor_forward(actor, b[mode])
                    row[mode + '_h_difference_l2'] = float(np.linalg.norm(a[mode] - b[mode]))
                    row[mode + '_mu_difference_l2'] = float(np.linalg.norm(mua - mub))
                    row[mode + '_h2_gate_disagreement'] = int(np.count_nonzero((h2a > 0) != (h2b > 0)))
                direction_pairs.append(row)
    write_csv(out / 'fixed_input_dose_direction.csv', fixed_rows)
    write_csv(out / 'fixed_input_group_pairs.csv', direction_pairs)
    write_csv(out / 'fixed_input_raw_blocks.csv', component_rows)
    write_csv(out / 'fixed_input_states.csv', [{**{k: v for k, v in s.items() if k != 'h'}, 'state_id': i} for i, s in enumerate(fixed_states)])
    np.savez_compressed(out / 'fixed_input_vectors.npz',
                        source_h=np.asarray([s['h'] for s in fixed_states]),
                        state_id=np.asarray([a[0] for a in arrays_out]), n=np.asarray([a[1] for a in arrays_out]),
                        control=np.asarray([a[2] for a in arrays_out]), mode=np.asarray([a[3] for a in arrays_out]),
                        h_applied=np.asarray([a[4] for a in arrays_out]), h2=np.asarray([a[5] for a in arrays_out]),
                        mu=np.asarray([a[6] for a in arrays_out]), z_after=np.asarray([a[7] for a in arrays_out]))
    timeline = []
    for record in source_records:
        c = record['condition']
        if record['object'] not in (11477, 11406) or record['seed'] not in (2026110400, 2026110402):
            continue
        if c['model'] != 'assisted_seed0' or c['control'] not in ('decrease', 'bounded_increase'):
            continue
        a = load(record)
        for t in range(len(a['step'])):
            timeline.append({'object': record['object'], 'seed': record['seed'], **c, 'step': t,
                             'opening_rad': float(a['post_opening'][t]), 'progress': float(a['post_progress'][t]),
                             'distance_m': float(a['post_distance'][t]), 'palm': bool(a['post_contacts'][t, -1]),
                             'finger_groups': int(a['post_contacts'][t, :-1].sum()),
                             'strict_contact': bool(a['post_strict_contact'][t]), 'success': bool(a['post_success'][t]),
                             'actual_norm': float(a['edit_actual_norm'][t]), 'applied': bool(a['applied'][t])})
    write_csv(out / 'focal_full_timelines.csv', timeline)
    # Separate numerical implementation for every new forward result.
    w = {key: value.numpy() for key, value in actor.items()}
    emitted = np.asarray([a[4] for a in arrays_out], dtype=np.float32)
    numpy_h2 = np.maximum(emitted @ w['mlp_extractor.policy_net.2.weight'].T + w['mlp_extractor.policy_net.2.bias'], 0)
    numpy_mu = numpy_h2 @ w['action_net.weight'].T + w['action_net.bias']
    expected_h2 = np.asarray([a[5] for a in arrays_out])
    expected_mu = np.asarray([a[6] for a in arrays_out])
    numpy_h2_error = float(np.max(abs(numpy_h2 - expected_h2)))
    numpy_mu_error = float(np.max(abs(numpy_mu - expected_mu)))
    native_replays, replay_h_error = 0, 0.
    for state in fixed_states:
        for n in (1, 4, 6):
            for control in ('decrease', 'bounded_increase'):
                record = get(state['object'], state['seed'], 'top%d' % n, control)
                a = load(record); t = state['step']
                if t >= len(a['h_original']) or not np.array_equal(a['h_original'][t], state['h']):
                    continue
                z, _, _ = model.encode(state['h'])
                condition = {'model': 'assisted_seed0', 'group': 'top%d' % n, 'control': control, 'dose': 'full'}
                reconstructed, _ = independent_edit(state['h'], z, model, stats, ridge, metadata, condition, 'unused-target-key')
                native_replays += 1
                replay_h_error = max(replay_h_error, float(np.max(abs(reconstructed - a['h_applied'][t]))))
    plot(out, pd.DataFrame(fixed_rows), pd.DataFrame(timeline), pd.DataFrame(duration_pairs))
    changed = [path for path, expected in hashes.items() if sha(native_path(path)) != expected]
    prefix_passed = all(row['through_first5_all_common_arrays_exact'] for row in duration_pairs)
    verification = {'passed': not changed and len(summaries) == 231 and prefix_passed and max_ray_error < 2e-6 and max_additivity_error < 1e-10
                              and source_norm_max_error < 2e-6 and numpy_h2_error < 2e-5 and numpy_mu_error < 2e-5 and replay_h_error < 2e-5,
                    'all_treatment_branches': len(summaries), 'group_physical_pairs': len(pairs), 'duration_physical_pairs': len(duration_pairs),
                    'all_duration_first5_full_observation_prefixes_exact': prefix_passed,
                    'fixed_inputs': len(fixed_states), 'fixed_input_edits': len(fixed_rows),
                    'max_common_budget_float32_error': max_ray_error, 'max_raw_block_additivity_error': max_additivity_error,
                    'all_source_actual_norm_recompute_max_error': source_norm_max_error,
                    'independent_numpy_all_new_h2_max_error': numpy_h2_error, 'independent_numpy_all_new_mu_max_error': numpy_mu_error,
                    'exact_same_source_h_native_replay_rows': native_replays, 'native_replay_h_max_error': replay_h_error,
                    'source_hashes_unchanged': not changed, 'changed_sources': changed,
                    'new_physical_trajectories': 0, 'new_training': 0,
                    'limits': ['Post-hoc outcome-selected focal cases and divergence-selected fixed states',
                               'Matched-dose results are single-forward frozen-input sensitivity, not physical outcomes',
                               'Common budget is the minimum positive original group budget; inactive groups stay explicit no-ops; all-three support is recorded',
                               'No sustained random/Ridge control; sustained treatment does not prove feedback compensation']}
    dump(out / 'verification.json', verification)
    dump(out / 'source_hashes.json', hashes)
    dump(out / 'protocol.json', {'scope': 'frozen source post-hoc analysis; no new simulation',
                               'first5': 'first five assigned control steps, including unsupported/noop steps',
                               'first_difference': 'both exact nonzero and max absolute >1e-5 recorded; contact Booleans exact',
                               'fixed_states': '7 origins natural first five trigger states plus focal first-output-divergence inputs, labeled separately',
                               'common_budget': 'minimum strictly positive native bidirectional budgets among Top1/4/6 at exactly the same h; groups without support retained as no-op',
                               'h2_gates': 'CPU frozen actor ReLU support at the edited input; readouts, not standalone physical causes'})
    write_report(out, summaries, pairs, duration_pairs, fixed_rows, direction_pairs, verification)
    return verification


def write_report(out, summaries, pairs, durations, fixed_rows, direction_pairs, verification):
    summary = pd.DataFrame(summaries)
    fixed = pd.DataFrame(fixed_rows)
    directions = pd.DataFrame(direction_pairs)
    text = ['# 特征组不单调与持续处理的事后机制分解', '',
            '这批非单调结局同时混合了组方向、组别预算、投影和闭环后续状态差异。Top4/Top6码下降在首次分叉输入上几乎平行；等预算后动作差显著缩小。码增加则仍有方向差异。以下均是开发病例的事后分解，等预算部分只有固定输入前向，没有新物理轨迹。', '',
            '## 实际暴露：更大的扰动没有简单对应更多破坏', '',
            '|起点|操作|Top1前5步norm和/成功|Top4前5步norm和/成功|Top6前5步norm和/成功|',
            '|---|---|---|---|---|']
    for seed in (2026110400, 2026110402):
        for control in ('decrease', 'bounded_increase'):
            values = []
            for group in ('top1', 'top4', 'top6'):
                row = summary[(summary.object == 11477) & (summary.seed == seed) & (summary.model == 'assisted_seed0') &
                              (summary.group == group) & (summary.control == control) & (summary.duration == 5)].iloc[0]
                values.append('%.4f / %s' % (row.first5_norm_sum, '成功' if row.success else '失败'))
            text.append('|11477/%s|%s|%s|' % (str(seed)[-3:], control, '|'.join(values)))
    text += ['', '这排除了“组越大就以更大扰动单调增加破坏”的简单描述，但不能排除预算与方向共同造成不同结局。累计量来自不同闭环输入，不能当等剂量比较。', '',
             '## 在第一次Top4/Top6输出不同处固定相同h', '',
             '这些输入来自Top4分支，且与Top6当时的动作前h逐值相同；Top1在这里是另一个固定输入计算，不是其真实轨迹上的同一状态。组方向和预算均按冻结规则重新计算。', '',
             '|起点/操作/步|Top4与Top6原方向余弦|原预算动作均值差L2|共同预算动作均值差L2|差异缩小|共同预算h2门差|',
             '|---|---:|---:|---:|---:|---:|']
    focal = directions[(directions.object == 11477) & (directions.source == 'first_group_output_difference') &
                       (directions.pair == 'top4_top6') & (directions.source_control == directions.control) &
                       (directions.left_n == 4) & (directions.right_n == 6)]
    for _, row in focal.iterrows():
        shrink = 1 - row.common_mu_difference_l2 / row.native_mu_difference_l2
        text.append('|%s / %s / %d|%.4f|%.5f|%.5f|%.1f%%|%d|' %
                    (str(row.seed)[-3:], row.control, row.step, row.native_delta_cosine, row.native_mu_difference_l2,
                     row.common_mu_difference_l2, shrink * 100, row.common_h2_gate_disagreement))
    text += ['', '三组共同预算落在Top1的较小预算，也降低了Top4/Top6两者的整体剂量，因此上表的差异缩小比例不是可归因于剂量的百分比。码增加的共同预算方向余弦仍约0.79—0.80，动作差和门差仍在。下面另外固定Top4/Top6配对预算，避免以Top1决定预算。所有等预算结果均未运行物理闭环。', '',
             '辅助seed0的Top6为[6,8,35,5,22,42]。400/402码下降首次Top4/6分叉时，新增部分是z42进入活跃支持（z22为0）；402码增加首次分叉时新增部分是z22（z42为0）。新增坐标既改变原方向，也改变可行双向预算；固定组的名义维数不是逐步实际活跃维数。逐坐标支持与请求量保留在fixed_input_dose_direction.csv。', '',
             '![固定输入预算与动作](group_dose_direction.png)', '',
             '## 持续与5步：相同前缀之后的额外施加', '',
             '全部7起点×2操作共14对，触发前和前5处理步的52个共有数组（含完整有噪观测、h、码、动作与状态）逐值完全相同。持续处理并不是另一条初始扰动，区别始于第6处理步以后。', '']
    for row in durations:
        if row['object'] == 11406 and row['seed'] == 2026110402:
            text.append('- 11406/402 `%s`：第%d步触发；首次输出h/动作差在第%s步，输入h首次差在第%s步；首次手指组差在第%s步，手掌与严格接触首次差在第%s/%s步。5步/持续最终为%s/%s，累计实际norm为%.4f/%.4f。' %
                        (row['control'], row['trigger'], row['first_emitted_h_difference'], row['first_incoming_h_difference'],
                         row['first_finger_group_difference'], row['first_palm_difference'], row['first_strict_contact_difference'],
                         '成功' if row['pulse_success'] else '失败', '成功' if row['sustained_success'] else '失败',
                         row['pulse_total_norm'], row['sustained_total_norm']))
    text += ['', '在11406/402有界码增加中，持续分支第65步开始满足严格接触，第69步目标进度、距离和严格接触同时满足而成功；5步分支第66步已到目标进度/近距离，但三次目标窗口均缺手掌，第70步才第一次满足严格接触，故最终失败。持续下降也产生额外动作和接触差，却仍失败。此链条描述处理后时序，不把接触恢复直接当作神经反馈补偿，也未排除一般扰动或慢化开启使接触赶上的解释。持续条件没有持续随机/Ridge对照。', '',
             '![持续处理接触时序](duration_contact_timing.png)', '',
             '## 机器核验与范围', '',
             '保留全部231处理轨迹的实际暴露；42个组对和14个时长对；45个固定输入、540个基础前向变体及8个Top4/6配对预算前向。全部输入哈希复核未变。共同预算float32范数最大误差%.3g；所有新前向的独立NumPy h2/mu最大误差%.3g/%.3g；%d个输入逐值相同的原处理重放最大h差%.3g。' %
             (verification['max_common_budget_float32_error'], verification['independent_numpy_all_new_h2_max_error'],
              verification['independent_numpy_all_new_mu_max_error'], verification['exact_same_source_h_native_replay_rows'], verification['native_replay_h_max_error']), '',
             '状态包括自然触发后5帧，以及按已观察分叉选出的少数输入；后者不能当独立测试。三组共同预算取同一h上所有正支持组的最小原预算，无支持组保留为空操作，并单独记录是否三组都有共同支持。不是在干预后筛出成功或完整生效病例。主来源、计算规则和核验分别见source_hashes.json、protocol.json、verification.json。']
    pair_path = out / 'pair46_budget_vector_decomposition.csv'
    if pair_path.exists():
        table = pd.read_csv(pair_path)
        insert = ['', '## Top4/Top6配对预算及动作向量分解', '',
                  '在这4个首次分叉的共同输入上，取b=min(b4,b6)，不纳入Top1。精确保存向量恒等式：', '',
                  '`mu6_native-mu4_native = (mu6_native-mu6_b) + (mu6_b-mu4_b) + (mu4_b-mu4_native)`', '',
                  '分别称为Top6径向变化项、同剂量方向项、Top4径向变化项。这是固定输入actor向量的代数分解，三个向量的范数不可相加为贡献百分比，也不是物理结局归因。', '',
                  '|起点/操作/步|配对b|原总向量L2|Top6径向项L2|同剂量方向项L2|Top4径向项L2|向量可加残差max|',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for _, r in table.iterrows():
            insert.append('|%s / %s / %d|%.5f|%.5f|%.5f|%.5f|%.5g|%.2g|' %
                          (str(r.seed)[-3:], r.control, r.step, r.pair_budget, r.total_mu_l2,
                           r.radial6_mu_l2, r.same_budget_direction_mu_l2, r.radial4_mu_l2, r.additive_residual_max_abs))
        insert += ['', '![动作向量分解](pair46_vector_decomposition.png)', '',
                   '下降方向的配对预算比较仍显示很小的方向项，而增加方向的方向项较大；原组大小效应同时包含预算与方向差。是否改变成败，需要新等预算闭环及相应对照，不能由这次前向分解决定。']
        marker = text.index('## 持续与5步：相同前缀之后的额外施加')
        text[marker:marker] = insert
    (out / 'GROUP_MECHANISMS_2026-11-05.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def plot(out, fixed, timeline, durations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), constrained_layout=True)
    for col, (seed, control) in enumerate([(2026110400, 'decrease'), (2026110400, 'bounded_increase'),
                                          (2026110402, 'decrease'), (2026110402, 'bounded_increase')]):
        g = fixed[(fixed.object == 11477) & (fixed.seed == seed) & (fixed.source == 'first_group_output_difference') &
                  (fixed.pair == 'top4_top6') & (fixed.source_control == control) & (fixed.control == control)]
        state = int(g.state_id.iloc[0]); g = g[g.state_id == state]
        for mode, color, style, label in [('native_budget', '#386cb0', '-o', 'Original budget'), ('common_budget', '#d97926', '--s', 'Same positive budget')]:
            a = g[g['mode'] == mode].sort_values('n')
            axes[0, col].plot(a.n, a.actual_norm, style, color=color, label=label)
            axes[1, col].plot(a.n, a.mu_delta_l2, style, color=color, label=label)
        axes[0, col].set_title('11477/%s, %s\nshared incoming h at step %d' % (str(seed)[-3:], control.replace('bounded_', ''), int(g.step.iloc[0])))
        for ax in axes[:, col]:
            ax.set_xticks([1, 4, 6]); ax.set_ylim(bottom=0); ax.grid(alpha=.2)
        axes[1, col].set_xlabel('Fixed group size N')
    axes[0, 0].set_ylabel('Actual h perturbation L2')
    axes[1, 0].set_ylabel('Unclipped action mean change L2')
    for ax in axes[0]:
        ax.set_ylim(0, .58)
    for ax in axes[1]:
        ax.set_ylim(0, .30)
    axes[0, 0].legend(frameon=False, loc='lower right')
    fig.suptitle('Fixed-input sensitivity at the first Top4/Top6 output divergence; no new physical outcomes')
    fig.savefig(out / 'group_dose_direction.png', dpi=180)
    plt.close(fig)
    plot_duration(out, timeline)


def augment_existing(run, destination):
    """Add eight pair-budget forwards from saved inputs, without a raw-data rerun."""
    torch.set_num_threads(1)
    run, out = native_path(run), native_path(destination)
    metadata = json.loads((run / 'package/bundle/online_intervention.json').read_text())
    with np.load(run / 'package/bundle/online_statistics.npz') as a:
        stats = {k: a[k].copy() for k in a.files}
    with np.load(run / 'package/bundle/combined_directions.npz') as a:
        ridge = a['ridge_oriented'].copy()
    with np.load(run / 'package/bundle/actor.npz') as a:
        actor = {k: torch.from_numpy(a[k].copy()) for k in a.files}
    model = IndependentModel(run / 'package/bundle/models/assisted_seed0')
    states = pd.read_csv(out / 'fixed_input_states.csv')
    fixed = pd.read_csv(out / 'fixed_input_dose_direction.csv')
    with np.load(out / 'fixed_input_vectors.npz') as a:
        vectors = {k: a[k].copy() for k in a.files}
    focal = states[(states.object == 11477) & (states.source == 'first_group_output_difference') & (states.pair == 'top4_top6')]
    decompositions, raw_vectors = [], []
    pair_norm_error = 0.
    for _, state in focal.iterrows():
        h = vectors['source_h'][int(state.state_id)]
        control = state.source_control
        z, _, _ = model.encode(h)
        contexts = {}
        for n in (4, 6):
            condition = {'model': 'assisted_seed0', 'group': 'top%d' % n, 'control': control, 'dose': 'full'}
            emission, detail = independent_edit(h, z, model, stats, ridge, metadata, condition, 'unused-target-key')
            coordinates = detail['coordinates']
            request = (-z[coordinates] if control == 'decrease' else np.where(z[coordinates] > 0, np.maximum(0., np.minimum(
                .5 * stats['assisted_seed0__active_std'][coordinates], stats['assisted_seed0__active_q99'][coordinates] - z[coordinates])), 0.))
            raw = model.decoder[:, coordinates] @ request
            contexts[n] = {'native': emission, 'budget': detail['budget'], 'raw': raw}
        b = min(contexts[4]['budget'], contexts[6]['budget'])
        if b <= 1e-9:
            raise RuntimeError('Focal pair does not support a common positive budget')
        for n, context in contexts.items():
            delta, status = independent_ray(h, context['raw'], b)
            if status != 'ok':
                raise RuntimeError('Pair budget became infeasible')
            context['pair'] = np.maximum(h + delta, 0).astype(np.float32)
            pair_norm_error = max(pair_norm_error, abs(np.linalg.norm(context['pair'].astype(float) - h) - b))
            context['native_h2'], native_mu = actor_forward(actor, context['native'])
            context['pair_h2'], pair_mu = actor_forward(actor, context['pair'])
            context['native_mu'], context['pair_mu'] = native_mu.astype(float), pair_mu.astype(float)
        a, c = contexts[4], contexts[6]
        total = c['native_mu'] - a['native_mu']
        radial6 = c['native_mu'] - c['pair_mu']
        direction = c['pair_mu'] - a['pair_mu']
        radial4 = a['pair_mu'] - a['native_mu']
        residual = total - radial6 - direction - radial4
        decompositions.append({'object': int(state.object), 'seed': int(state.seed), 'step': int(state.step), 'control': control,
                               'state_id': int(state.state_id), 'pair_budget': b, 'native_b4': a['budget'], 'native_b6': c['budget'],
                               'total_mu_l2': float(np.linalg.norm(total)), 'radial6_mu_l2': float(np.linalg.norm(radial6)),
                               'same_budget_direction_mu_l2': float(np.linalg.norm(direction)), 'radial4_mu_l2': float(np.linalg.norm(radial4)),
                               'additive_residual_max_abs': float(np.max(abs(residual))),
                               'pair_h_delta_cosine': cosine(a['pair'].astype(float) - h, c['pair'].astype(float) - h),
                               'pair_h2_gate_disagreement': int(np.count_nonzero((a['pair_h2'] > 0) != (c['pair_h2'] > 0))),
                               'radial6_direction_cosine': cosine(radial6, direction)})
        raw_vectors.append({'source_h': h, 'h4_native': a['native'], 'h6_native': c['native'],
                            'h4_pair': a['pair'], 'h6_pair': c['pair'],
                            'mu4_native': a['native_mu'], 'mu6_native': c['native_mu'],
                            'mu4_pair': a['pair_mu'], 'mu6_pair': c['pair_mu'],
                            'total': total, 'radial6': radial6, 'direction': direction, 'radial4': radial4, 'residual': residual})
    write_csv(out / 'pair46_budget_vector_decomposition.csv', decompositions)
    np.savez_compressed(out / 'pair46_vector_decomposition.npz', **{key: np.asarray([r[key] for r in raw_vectors]) for key in raw_vectors[0]})
    # Add coordinate/support details using the already saved 45 inputs only.
    details = {}
    for state_id, h in enumerate(vectors['source_h']):
        z, _, _ = model.encode(h)
        for n in (1, 4, 6):
            coordinates = np.asarray(metadata['models']['assisted_seed0']['opening_coordinates'][:n])
            for control in ('decrease', 'bounded_increase'):
                request = -z[coordinates] if control == 'decrease' else np.where(z[coordinates] > 0, np.maximum(0., np.minimum(
                    .5 * stats['assisted_seed0__active_std'][coordinates], stats['assisted_seed0__active_q99'][coordinates] - z[coordinates])), 0.)
                raw = model.decoder[:, coordinates] @ request
                details[state_id, n, control] = {'target_coordinates': json.dumps(coordinates.tolist()),
                                                'active_target_coordinates': json.dumps(coordinates[z[coordinates] > 0].tolist()),
                                                'requested_code_change': json.dumps(request.tolist()),
                                                'requested_active_coordinates': int(np.count_nonzero(request)),
                                                'full_request_projected_coordinates': int(np.sum(h + raw < 0)),
                                                'raw_negative_h_proposal_fraction': float(np.mean(h + raw < 0))}
    for field in next(iter(details.values())):
        fixed[field] = [details[int(r.state_id), int(r.n), r.control][field] for _, r in fixed.iterrows()]
    fixed.to_csv(out / 'fixed_input_dose_direction.csv', index=False, encoding='utf-8-sig')
    w = {key: tensor.numpy() for key, tensor in actor.items()}
    h_all = np.concatenate([vectors['h_applied'], np.asarray([r['h%d_pair' % n] for r in raw_vectors for n in (4, 6)])])
    numpy_h2 = np.maximum(h_all @ w['mlp_extractor.policy_net.2.weight'].T + w['mlp_extractor.policy_net.2.bias'], 0)
    numpy_mu = numpy_h2 @ w['action_net.weight'].T + w['action_net.bias']
    expected_h2 = np.concatenate([vectors['h2'], np.asarray([actor_forward(actor, r['h%d_pair' % n])[0] for r in raw_vectors for n in (4, 6)])])
    expected_mu = np.concatenate([vectors['mu'], np.asarray([r['mu%d_pair' % n] for r in raw_vectors for n in (4, 6)])])
    h2_error, mu_error = float(np.max(abs(numpy_h2 - expected_h2))), float(np.max(abs(numpy_mu - expected_mu)))
    gate_mismatches = int(np.count_nonzero((numpy_h2 > 0) != (expected_h2 > 0)))
    campaign = json.loads((run / 'remote/pilot/campaign.json').read_text())
    replays, replay_max = 0, 0.
    for state, result in zip(focal.to_dict('records'), raw_vectors):
        case = next(c for c in campaign['cases'] if c['object'] == state['object'] and c['seed'] == state['seed'])
        for n in (4, 6):
            record = next(r for r in case['runs'] if r['condition']['model'] == 'assisted_seed0' and r['condition']['group'] == 'top%d' % n
                          and r['condition']['control'] == state['source_control'] and r['condition']['duration'] == 5)
            path = run / 'remote/pilot' / ('object%d_seed%d' % (case['object'], case['seed'])) / record['relative_dir'] / (record['name'] + '.npz')
            with np.load(path) as a:
                t = int(state['step'])
                if not np.array_equal(a['h_original'][t], result['source_h']):
                    raise RuntimeError('Focal pair source h was not exactly shared')
                replay_max = max(replay_max, float(np.max(abs(a['h_applied'][t] - result['h%d_native' % n]))))
                replays += 1
    verification = json.loads((out / 'verification.json').read_text())
    source_hashes = json.loads((out / 'source_hashes.json').read_text())
    implementation = display(native_path(__file__))
    changed = [path for path, expected in source_hashes.items() if path != implementation and sha(native_path(path)) != expected]
    source_hashes[implementation] = sha(native_path(__file__))
    verification.update({'pair_budget_fixed_states': len(decompositions), 'pair_budget_new_forwards': 2 * len(decompositions),
                         'max_pair_budget_float32_error': pair_norm_error,
                         'max_pair_vector_additivity_residual': max(r['additive_residual_max_abs'] for r in decompositions),
                         'independent_numpy_all_new_h2_max_error': h2_error, 'independent_numpy_all_new_mu_max_error': mu_error,
                         'independent_numpy_h2_gate_mismatches': gate_mismatches,
                         'exact_same_source_h_native_replay_rows': replays, 'native_replay_h_max_error': replay_max,
                         'source_hashes_unchanged': not changed, 'changed_sources': changed,
                         'augmentation': '45 saved inputs reused; 4 focal points/8 new forwards, no all-source trajectory recomputation; source archive hashes rechecked'})
    verification['passed'] &= not changed and pair_norm_error < 2e-6 and h2_error < 2e-5 and mu_error < 2e-5 and replay_max < 2e-5 and gate_mismatches == 0
    dump(out / 'verification.json', verification); dump(out / 'source_hashes.json', source_hashes)
    protocol = json.loads((out / 'protocol.json').read_text())
    protocol['pair_budget_decomposition'] = 'b=min(native b4,native b6), same h, complete 22-dimensional mu identity; norms are not additive contribution shares'
    dump(out / 'protocol.json', protocol)
    pairs = pd.read_csv(out / 'group_first_differences.csv').to_dict('records')
    durations = pd.read_csv(out / 'duration_first_differences.csv').to_dict('records')
    plot(out, fixed, pd.read_csv(out / 'focal_full_timelines.csv'), pd.DataFrame(durations))
    plot_decomposition(out, pd.DataFrame(decompositions))
    write_report(out, pd.read_csv(out / 'all_treatment_exposure.csv').to_dict('records'), pairs, durations,
                 fixed.to_dict('records'), pd.read_csv(out / 'fixed_input_group_pairs.csv').to_dict('records'), verification)
    dump(out / 'artifact_hashes.json', {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'artifact_hashes.json'})
    return verification


def plot_decomposition(out, table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, constrained_layout=True)
    colors = ['#222222', '#386cb0', '#d97926', '#6c8e60']
    fields = ['total_mu_l2', 'radial6_mu_l2', 'same_budget_direction_mu_l2', 'radial4_mu_l2']
    for ax, (_, row) in zip(axes, table.iterrows()):
        ax.bar(np.arange(4), [row[f] for f in fields], color=colors)
        ax.set_xticks(np.arange(4), ['Total', 'Radial 6', 'Direction\nat same b', 'Radial 4'], rotation=25)
        ax.set_title('%s: %s\nb=%.4f' % (str(row.seed)[-3:], row.control.replace('bounded_', ''), row.pair_budget))
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylabel('L2 norm of unclipped action-mean vectors')
    fig.suptitle('Pairwise Top4/Top6 budget decomposition: vectors add exactly; their norms do not add')
    fig.savefig(out / 'pair46_vector_decomposition.png', dpi=180)
    plt.close(fig)


def plot_duration(out, timeline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, constrained_layout=True)
    for col, control in enumerate(('decrease', 'bounded_increase')):
        for duration, color, label in [(5, '#386cb0', '5-step pulse'), (-1, '#d97926', 'Continued treatment')]:
            g = timeline[(timeline.object == 11406) & (timeline.seed == 2026110402) &
                         (timeline.group == 'top4') & (timeline.control == control) & (timeline.duration == duration)]
            axes[0, col].plot(g.step, g.progress, color=color, label=label)
            axes[1, col].step(g.step, g.strict_contact.astype(int), where='post', color=color, label=label + ': strict')
            axes[1, col].step(g.step, g.palm.astype(int) * .8, where='post', color=color, linestyle=':', alpha=.8, label=label + ': palm (0.8)')
            axes[2, col].plot(g.step, g.actual_norm, color=color, label=label)
        axes[0, col].axhline(.95, color='#555555', linestyle='--', linewidth=1)
        axes[0, col].set_title('11406/402: ' + control.replace('bounded_', ''))
        for ax in axes[:, col]:
            ax.axvspan(52, 57, color='#999999', alpha=.12); ax.set_xlim(48, 78); ax.grid(alpha=.2)
        axes[2, col].set_xlabel('Absolute control step (post-action state at this step)')
    axes[0, 0].set_ylabel('Progress to target angle')
    axes[1, 0].set_ylabel('Contact indicator')
    axes[2, 0].set_ylabel('Actual h perturbation L2')
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False, fontsize=7, loc='upper left')
    fig.suptitle('Pulse and continued branches share the complete first five treatment steps; later extra exposure differs')
    fig.savefig(out / 'duration_contact_timing.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=PROJECT_ROOT / 'outputs/online_development_20261104')
    parser.add_argument('--out', type=Path, default=PROJECT_ROOT / 'outputs/deeper_analysis_20261105/groups')
    parser.add_argument('--augment-existing', action='store_true', help='Reuse the saved base analysis and add pair-budget vectors only')
    args = parser.parse_args()
    if args.augment_existing:
        result = augment_existing(args.run, args.out)
    else:
        analyze(args.run, args.out)
        result = augment_existing(args.run, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
