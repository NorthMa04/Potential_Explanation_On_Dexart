"""Independent audit and origin-paired descriptive analysis of constrained edits."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from temp_scripts.constrained_intervention import ConstrainedEditor
from temp_scripts.physical_runtime import NumericSAE, actor_tail


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def write_rows(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def stratified_origin_interval(rows, seed=20261102, draws=2000):
    strata = defaultdict(list)
    for row in rows:
        strata[row['object']].append(float(row['effect_degrees']))
    if not rows:
        return {'origins': 0, 'mean': None, 'ci95': None, 'reason': 'no complete paired origins'}
    mean = float(np.mean([r['effect_degrees'] for r in rows]))
    if any(len(values) < 2 for values in strata.values()):
        return {'origins': len(rows), 'mean': mean, 'ci95': None,
                'reason': 'fewer than two complete origins within an observed object stratum; no degenerate bootstrap CI reported'}
    rng = np.random.default_rng(seed)
    total = np.zeros(draws)
    for values in strata.values():
        v = np.asarray(values)
        total += rng.choice(v, size=(draws, len(v)), replace=True).sum(axis=1)
    total /= len(rows)
    return {'origins': len(rows), 'mean': mean, 'ci95': np.quantile(total, [.025, .975]).tolist(),
            'scope': 'whole origin resampling within the sampled objects; dictionaries averaged inside each origin'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--completed-only', action='store_true')
    p.add_argument('--study', type=Path, default=Path('outputs/state_assisted/20261102_122123_738477'))
    args = p.parse_args()
    root = args.run.resolve()
    campaign = json.loads((root / 'remote/campaign/campaign.json').read_text())
    if args.completed_only:
        cases = [c for c in campaign['cases'] if c['status'] in ('completed', 'no_trigger_all_outcomes_retained')
                 and (root / 'remote/campaign' / ('object%d_seed%d' % (c['object'], c['seed']))).exists()]
    else:
        assert campaign['status'] == 'completed', campaign['status']
        cases = campaign['cases']
    numeric = {d.name: NumericSAE(d) for d in (root / 'bundle/models').iterdir()}
    editor = ConstrainedEditor(root / 'bundle', numeric)
    actor = load_npz(root / 'bundle/actor.npz')
    source = load_npz(args.study / 'selected_arrays.npz')
    assert set(source['split']) == {'train', 'validation'}
    assert hashlib.sha256((args.study / 'selected_arrays.npz').read_bytes()).hexdigest() == editor.meta['selected_arrays_sha256']
    train = source['h'][source['split'] == 'train']
    train_min, train_max = editor.directions['train_min'], editor.directions['train_max']
    def support(h):
        d = np.sqrt(np.mean(((train - h) / numeric['assisted_seed0'].w['scale']) ** 2, axis=1))
        return float(d.min()), float(np.mean((h < train_min) | (h > train_max)))
    rows, case_rows = [], []
    max_prefix, max_mu, max_edit, max_norm = 0., 0., 0., 0.
    policies = set()
    for case in cases:
        home = root / 'remote/campaign' / ('object%d_seed%d' % (case['object'], case['seed']))
        baseline = load_npz(home / 'natural/natural.npz')
        eligible = np.flatnonzero(baseline['feature_trigger_eligible'])
        trigger = int(eligible[0]) if len(eligible) else None
        case_rows.append({'object': case['object'], 'environment_seed': case['seed'],
                         'trigger_step': trigger, 'status': case['status'],
                         'initial_any_contact': bool(baseline['contact'][trigger]) if trigger is not None else None,
                         'initial_strict_contact': bool(baseline['strict_contact'][trigger]) if trigger is not None else None,
                         'baseline_success': bool(baseline['post_success'].any()),
                         'baseline_steps': len(baseline['step'])})
        for record in case['runs']:
            assert record['exit_code'] == 0, record
            assert record['actor_parameters_unchanged'] and record['full_policy_state_unchanged']
            policies.add(record['full_policy_state_sha256'])
            name = record['name']
            a = load_npz(home / name / (name + '.npz'))
            assert a['done'][-1] and not a['done'][:-1].any()
            assert all(np.isfinite(v).all() for v in a.values())
            np.testing.assert_array_equal(a['step'], np.arange(len(a['step'])))
            np.testing.assert_array_equal(a['action'], a['mu'].clip(-1, 1))
            for key in ('opening', 'distance', 'robot_qpos', 'object_qpos', 'robot_qvel', 'object_qvel', 'contacts'):
                np.testing.assert_array_equal(a['post_' + key][:-1], a[key][1:])
            mu = actor_tail(actor, a['h_applied'])
            max_mu = max(max_mu, float(np.max(np.abs(mu - a['mu']))))
            np.testing.assert_allclose(mu, a['mu'], rtol=2e-5, atol=2e-5)
            assert np.all(a['h_applied'] >= 0)
            model, op, control = record.get('model_name'), record.get('operation'), record.get('control')
            edited = model is not None
            prefix_length = trigger + 1 if edited else len(a['step'])
            for key in ('opening', 'distance', 'contacts', 'robot_qpos', 'object_qpos', 'robot_qvel', 'object_qvel', 'h_original'):
                err = float(np.max(np.abs(a[key][:prefix_length].astype(float) - baseline[key][:prefix_length].astype(float))))
                assert err <= 1e-5, (name, key, err)
                max_prefix = max(max_prefix, err)
            for key in a:
                if key.startswith('obs__'):
                    err = float(np.max(np.abs(a[key][:prefix_length].astype(float) - baseline[key][:prefix_length].astype(float))))
                    assert err <= 1e-5
                    max_prefix = max(max_prefix, err)
            before_actions = trigger if edited else len(a['step'])
            np.testing.assert_allclose(a['action'][:before_actions], baseline['action'][:before_actions], atol=1e-5, rtol=0)
            row = {'object': case['object'], 'environment_seed': case['seed'], 'name': name,
                   'model': model, 'operation': op, 'control': control,
                   'trigger_step': trigger, 'steps': len(a['step']),
                   'success': bool(a['post_success'].any()), 'baseline_success': bool(baseline['post_success'].any()),
                   'baseline_steps': len(baseline['step']), 'matched': None, 'applied': False,
                   'initial_any_contact': bool(baseline['contact'][trigger]) if trigger is not None else None,
                   'initial_strict_contact': bool(baseline['strict_contact'][trigger]) if trigger is not None else None,
                   'initial_distance_m': float(baseline['distance'][trigger]) if trigger is not None else None,
                   'budget': None, 'actual_norm': None, 'target_code_change': None, 'code_retention': None,
                   'target_before': None, 'target_after': None, 'actual_code_above_train_q99': None,
                   'nontarget_code_l2': None, 'action_l2': None, 'mu_l2': None,
                   'opening_probe_change_degrees': None, 'distance_probe_change_m': None,
                   'contact_probe_change': None,
                   'initial_support_nn': None, 'edited_support_nn': None,
                   'initial_outside_train_fraction': None, 'edited_outside_train_fraction': None}
            if edited:
                assert trigger is not None
                expected_mask = a['step'] == trigger
                np.testing.assert_array_equal(a['scheduled'], expected_mask)
                h = a['h_original'][trigger]
                emitted, detail = editor.edit(h, model, op, control,
                                              '%d|%d|%d' % (case['object'], case['seed'], trigger))
                matched = detail['status'] == 'ok'
                actual = a['h_applied'][trigger]
                expected = emitted if matched else h
                err = float(np.max(np.abs(actual - expected)))
                max_edit = max(max_edit, err)
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
                assert bool(a['applied'][trigger]) == matched
                assert int(a['edit_status_code'][trigger]) == (1 if matched else 2)
                assert not a['applied'][~expected_mask].any()
                np.testing.assert_array_equal(a['h_applied'][~expected_mask], a['h_original'][~expected_mask])
                for field in ('budget', 'target_before', 'target_after', 'actual_code_change', 'code_retention', 'nontarget_code_l2'):
                    if field in detail:
                        np.testing.assert_allclose(a['edit_' + field][trigger], detail[field], atol=2e-5, rtol=2e-5)
                actual_norm = float(np.linalg.norm(actual.astype(float) - h))
                if matched:
                    error = abs(actual_norm - float(a['edit_budget'][trigger]))
                    assert error <= 2e-6
                    max_norm = max(max_norm, error)
                initial_nn, initial_outside = support(h)
                edited_nn, edited_outside = support(actual)
                _, _, state_predictions = numeric[model].infer(np.vstack((h, actual)))
                row.update(matched=matched, applied=bool(a['applied'][trigger]),
                           budget=float(a['edit_budget'][trigger]), actual_norm=actual_norm,
                           target_code_change=float(a['edit_actual_code_change'][trigger]),
                           target_before=float(a['edit_target_before'][trigger]),
                           target_after=float(a['edit_target_after'][trigger]),
                           actual_code_above_train_q99=bool(a['edit_target_after'][trigger] > editor.meta['models'][model]['q99']),
                           code_retention=float(a['edit_code_retention'][trigger]),
                           nontarget_code_l2=float(a['edit_nontarget_code_l2'][trigger]),
                           action_l2=float(np.linalg.norm(a['action'][trigger] - baseline['action'][trigger])),
                           mu_l2=float(np.linalg.norm(a['mu'][trigger] - baseline['mu'][trigger])),
                           opening_probe_change_degrees=float((state_predictions[1, 1] - state_predictions[0, 1]) * 180 / np.pi),
                           distance_probe_change_m=float(state_predictions[1, 0] - state_predictions[0, 0]),
                           contact_probe_change=float(state_predictions[1, 2] - state_predictions[0, 2]),
                           initial_support_nn=initial_nn, edited_support_nn=edited_nn,
                           initial_outside_train_fraction=initial_outside, edited_outside_train_fraction=edited_outside)
            else:
                assert not a['applied'].any()
                np.testing.assert_array_equal(a['h_original'], a['h_applied'])
            for horizon in (1, 3, 5):
                j = trigger + horizon - 1 if trigger is not None else len(a['step'])
                available = j < len(a['step']) and j < len(baseline['step'])
                row['opening_delta_%d_degrees' % horizon] = float((a['post_opening'][j] - baseline['post_opening'][j]) * 180 / np.pi) if available else None
                row['baseline_opening_increment_%d_degrees' % horizon] = float((baseline['post_opening'][j] - baseline['opening'][trigger]) * 180 / np.pi) if available else None
                row['distance_delta_%d_m' % horizon] = float(a['post_distance'][j] - baseline['post_distance'][j]) if available else None
                row['any_contact_at_%d' % horizon] = bool(a['post_contact'][j]) if available else None
                row['strict_contact_at_%d' % horizon] = bool(a['post_strict_contact'][j]) if available else None
            rows.append(row)
    assert len(policies) == 1, 'Full policy state hashes differ'
    # Compare only genuinely matched controls; retain all invalid rows in output.
    paired, seed_paired = [], []
    for op in ('suppress', 'enhance'):
        for case in case_rows:
            differences, seed_values = [], []
            for seed in (0, 1, 2):
                group = [r for r in rows if r['object'] == case['object'] and r['environment_seed'] == case['environment_seed']
                         and r['model'] == 'assisted_seed%d' % seed and r['operation'] == op]
                target = [r for r in group if r['control'] == 'target' and r['matched'] and r['opening_delta_5_degrees'] is not None]
                randoms = [r for r in group if r['control'] in ('random0', 'random1', 'random2') and r['matched'] and r['opening_delta_5_degrees'] is not None]
                if len(target) == 1 and len(randoms) == 3:
                    assert all(abs(r['budget'] - target[0]['budget']) < 2e-6 for r in randoms)
                    value = target[0]['opening_delta_5_degrees'] - float(np.mean([r['opening_delta_5_degrees'] for r in randoms]))
                    differences.append(value); seed_values.append(value)
                    seed_paired.append({'object': case['object'], 'environment_seed': case['environment_seed'],
                                        'operation': op, 'sae_seed': seed, 'effect_degrees': value})
            if len(differences) == 3:
                paired.append({'object': case['object'], 'environment_seed': case['environment_seed'],
                               'operation': op, 'effect_degrees': float(np.mean(differences)),
                               'initial_any_contact': case['initial_any_contact'],
                               'dictionary_seed_sd': float(np.std(seed_values, ddof=1))})
    primary = {op: stratified_origin_interval([r for r in paired if r['operation'] == op]) for op in ('suppress', 'enhance')}
    contact_strata = []
    for op in ('suppress', 'enhance'):
        for contact in (False, True):
            values = [r['effect_degrees'] for r in paired if r['operation'] == op and r['initial_any_contact'] == contact]
            contact_strata.append({'operation': op, 'initial_any_contact': contact, 'origins': len(values),
                                   'mean_effect_degrees': float(np.mean(values)) if values else None,
                                   'scope': 'exploratory pretreatment contact stratification, not a replacement primary comparison'})
    write_rows(root / 'physical_results.csv', rows)
    write_rows(root / 'origin_coverage.csv', case_rows)
    write_rows(root / 'paired_origin_effects.csv', paired)
    write_rows(root / 'paired_dictionary_effects.csv', seed_paired)
    by_seed = []
    for op in ('suppress', 'enhance'):
        for seed in (0, 1, 2):
            values = [r['effect_degrees'] for r in seed_paired if r['operation'] == op and r['sae_seed'] == seed]
            by_seed.append({'operation': op, 'sae_seed': seed, 'origins': len(values),
                            'mean_effect_degrees': float(np.mean(values)) if values else None})
    groups = []
    for op in ('half_suppress', 'suppress', 'enhance'):
        for control in ('target', 'random0', 'random1', 'random2', 'raw', 'pure', 'ridge'):
            group = [r for r in rows if r['operation'] == op and r['control'] == control]
            valid = [r for r in group if r['matched']]
            if not group: continue
            changes = [r['opening_delta_5_degrees'] for r in valid if r['opening_delta_5_degrees'] is not None]
            groups.append({'operation': op, 'control': control, 'assigned': len(group), 'matched': len(valid),
                           'valid_horizon5': len(changes), 'successes_assigned': sum(r['success'] for r in group),
                           'mean_opening_delta5_degrees': float(np.mean(changes)) if changes else None,
                           'mean_action_l2': float(np.mean([r['action_l2'] for r in valid])) if valid else None,
                           'correct_target_code_sign_fraction': float(np.mean([r['code_retention'] > 0 for r in valid])) if valid else None})
    audit = {'passed': True, 'partial_completed_cases_only': args.completed_only,
             'audited_rollouts': len(rows), 'intervention_assignments': sum(r['model'] is not None for r in rows),
             'matched_assignments': sum(r['matched'] is True for r in rows), 'cases': len(cases),
             'triggered_origins': sum(r['trigger_step'] is not None for r in case_rows),
             'baseline_successes': sum(r['baseline_success'] for r in case_rows),
             'max_prefix_abs': max_prefix, 'max_actor_mu_abs': max_mu, 'max_edit_abs': max_edit,
             'max_successful_norm_match_error': max_norm, 'full_policy_hash': next(iter(policies)),
             'all_emitted_h_nonnegative': True, 'primary_contrasts': primary, 'groups': groups,
             'primary_by_dictionary_seed': by_seed,
             'exploratory_contact_strata': contact_strata,
             'coverage': case_rows,
             'interpretation': 'development pilot, conditional on consensus feature support; paired origins are units; no semantic-causality or success-improvement certification'}
    (root / 'independent_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
