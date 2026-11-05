"""Audit every completed branch and summarize origin-paired physical outcomes."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .combined_intervention import CombinedEditor
    from .physical_runtime import NumericSAE, actor_tail
else:
    from combined_intervention import CombinedEditor
    from physical_runtime import NumericSAE, actor_tail


def write_rows(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    root = Path(root).resolve()
    if os.name == 'nt' and not str(root).startswith('\\\\?\\'):
        root = Path('\\\\?\\' + str(root))
    torch.set_num_threads(2)
    home = root / 'remote/campaign'
    campaign = json.loads((home / 'campaign.json').read_text(encoding='utf-8'))
    plan = json.loads((root / 'physical_plan.json').read_text(encoding='utf-8'))
    protocol_hash = hashlib.sha256((root / 'protocol_hashes.json').read_bytes()).hexdigest()
    if campaign['protocol_sha256'] != protocol_hash:
        raise RuntimeError('Retrieved campaign protocol hash differs')
    numeric = {p.name: NumericSAE(p) for p in (root / 'bundle/models').iterdir()}
    editor = CombinedEditor(root / 'bundle', numeric)
    with np.load(root / 'bundle/actor.npz', allow_pickle=False) as data:
        actor = {k: data[k].copy() for k in data.files}
    rows, coverage = [], []
    policies = set()
    maximum_edit_error, maximum_actor_error = 0., 0.
    for case in campaign['cases']:
        folder = home / ('object%d_seed%d' % (case['object'], case['seed']))
        latest = {r['name']: r for r in case['runs']}
        natural_record = latest.get('natural', {})
        baseline = None
        if natural_record.get('exit_code') == 0:
            with np.load(folder / natural_record['relative_dir'] / 'natural.npz', allow_pickle=False) as data:
                baseline = {k: data[k].copy() for k in data.files}
        trigger = case.get('trigger_step')
        coverage.append({'object': case['object'], 'environment_seed': case['seed'], 'status': case['status'],
                         'trigger_step': trigger, 'baseline_success': bool(baseline['post_success'].any()) if baseline is not None else None})
        for name, record in latest.items():
            row = {'object': case['object'], 'environment_seed': case['seed'], 'name': name,
                   'exit_code': record['exit_code'], 'model': record.get('model_name'),
                   'group': record.get('group_name'), 'dose': record.get('dose'),
                   'duration': record.get('duration'), 'control': record.get('control'), 'trigger_step': trigger}
            rows.append(row)
            if record['exit_code']:
                continue
            path = folder / record['relative_dir'] / (name + '.npz')
            report_path = path.with_name('report.json')
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['npz_sha256']:
                raise RuntimeError('Evidence archive hash mismatch')
            if hashlib.sha256(report_path.read_bytes()).hexdigest() != record['report_sha256']:
                raise RuntimeError('Evidence report hash mismatch')
            with np.load(path, allow_pickle=False) as data:
                a = {k: data[k].copy() for k in data.files}
            if not a['done'][-1] or a['done'][:-1].any() or any(not np.isfinite(v).all() for v in a.values()):
                raise RuntimeError('Invalid or partial physical trajectory')
            np.testing.assert_array_equal(a['step'], np.arange(len(a['step'])))
            if not record['actor_parameters_unchanged'] or not record['full_policy_state_unchanged']:
                raise RuntimeError('Frozen policy parameter audit failed')
            policies.add(record['full_policy_state_sha256'])
            mu = actor_tail(actor, a['h_applied'])
            maximum_actor_error = max(maximum_actor_error, float(np.max(np.abs(mu - a['mu']))))
            np.testing.assert_allclose(mu, a['mu'], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(mu.clip(-1, 1), a['action'], atol=2e-5, rtol=2e-5)
            if (a['h_applied'] < 0).any():
                raise RuntimeError('ReLU domain violation')
            if name != 'natural':
                end = trigger + 1 if row['model'] else len(baseline['step'])
                for key in baseline:
                    if key.startswith('obs__') or key in ('opening', 'distance', 'contact', 'strict_contact',
                                                         'robot_qpos', 'object_qpos', 'robot_qvel', 'object_qvel', 'h_original'):
                        np.testing.assert_allclose(a[key][:end], baseline[key][:end], atol=plan['restore_repeat_atol'], rtol=0)
                action_end = trigger if row['model'] else end
                for key in ('mu', 'action'):
                    np.testing.assert_allclose(a[key][:action_end], baseline[key][:action_end], atol=plan['restore_repeat_atol'], rtol=0)
                if not row['model']:
                    if len(a['step']) != len(baseline['step']):
                        raise RuntimeError('Natural repeat length differs')
            scheduled = np.zeros(len(a['step']), dtype=bool)
            if row['model']:
                stop = len(scheduled) if row['duration'] == -1 else trigger + row['duration']
                scheduled[trigger:stop] = True
                np.testing.assert_array_equal(a['scheduled'], scheduled)
                for t in np.flatnonzero(scheduled):
                    expected, detail = editor.edit(a['h_original'][t], row['model'], row['group'], row['dose'], row['control'],
                                                  '%d|%d|%d' % (case['object'], case['seed'], trigger))
                    np.testing.assert_allclose(a['h_applied'][t], expected, atol=2e-5, rtol=2e-5)
                    maximum_edit_error = max(maximum_edit_error, float(np.max(np.abs(a['h_applied'][t] - expected))))
                    if bool(a['applied'][t]) != (detail['status'] == 'ok'):
                        raise RuntimeError('Applied status mismatch')
                    for field in ('budget', 'actual_norm', 'norm_error', 'target_before', 'target_after', 'nontarget_code_l2', 'z_before', 'z_after'):
                        np.testing.assert_allclose(a['edit_' + field][t], detail[field], atol=2e-5, rtol=2e-5)
            np.testing.assert_array_equal(a['h_original'][~scheduled], a['h_applied'][~scheduled])
            row.update(success=bool(a['post_success'].any()), steps=len(a['step']),
                       final_opening_degrees=float(np.rad2deg(a['post_opening'][-1])),
                       scheduled_steps=int(scheduled.sum()), applied_steps=int(a['applied'].sum()),
                       cumulative_delta_norm=float(a['delta_norm'].sum()),
                       complete_exposure=bool(scheduled.any() and a['applied'][scheduled].all()),
                       all_scheduled_steps_failed=bool(scheduled.any() and not a['applied'][scheduled].any()))
            row['success_delta'] = int(row['success']) - int(baseline['post_success'].any())
            for horizon in (1, 5, 20, 50):
                index = trigger + horizon - 1 if trigger is not None else max(len(a['step']), len(baseline['step']))
                available = index < min(len(a['step']), len(baseline['step']))
                row['opening_delta_%d_degrees' % horizon] = float(np.rad2deg(a['post_opening'][index] - baseline['post_opening'][index])) if available else None
                if horizon == 5:
                    row['distance_delta_5_m'] = float(a['post_distance'][index] - baseline['post_distance'][index]) if available else None
                    row['contact_delta_5'] = int(a['post_contact'][index]) - int(baseline['post_contact'][index]) if available else None
        print('Audited object %d seed %d' % (case['object'], case['seed']), flush=True)
    if len(policies) > 1:
        raise RuntimeError('Different policy hashes across branches')
    by_condition = defaultdict(dict)
    for row in rows:
        if row['model']:
            key = tuple(row[k] for k in ('object', 'environment_seed', 'model', 'group', 'duration', 'dose'))
            by_condition[key][row['control']] = row
    paired = []
    for key, controls in by_condition.items():
        target = controls.get('target', {})
        references = [controls.get('random%d' % i, {}) for i in range(3)]
        valid = all(r.get('exit_code') == 0 and r.get('complete_exposure') and r.get('opening_delta_5_degrees') is not None for r in [target] + references)
        paired.append(dict(zip(('object', 'environment_seed', 'model', 'group', 'duration', 'dose'), key),
                           complete_matched_pair=valid,
                           effect_degrees=target['opening_delta_5_degrees'] - float(np.mean([r['opening_delta_5_degrees'] for r in references])) if valid else None))
    origins = defaultdict(list)
    for row in paired:
        key = (row['object'], row['environment_seed'], row['model'].split('_seed')[0], row['group'], row['duration'], row['dose'])
        origins[key].append(row)
    origin_rows = []
    for key, values in origins.items():
        valid = [v for v in values if v['complete_matched_pair']]
        origin_rows.append(dict(zip(('object', 'environment_seed', 'kind', 'group', 'duration', 'dose'), key),
                               complete_model_seeds=len(valid),
                               effect_degrees=float(np.mean([v['effect_degrees'] for v in valid])) if len(valid) == 3 else None))
    destination = root / 'analysis'
    destination.mkdir(exist_ok=True)
    write_rows(destination / 'physical_results.csv', rows)
    write_rows(destination / 'origin_coverage.csv', coverage)
    write_rows(destination / 'paired_dictionary_effects.csv', paired)
    write_rows(destination / 'paired_origin_effects.csv', origin_rows)
    result = {'audit_passed': True, 'campaign_status': campaign['status'], 'planned_maximum_rollouts': plan['maximum_planned_rollouts'],
              'observed_latest_conditions': len(rows), 'failed_conditions': sum(r['exit_code'] != 0 for r in rows),
              'origins': len(coverage), 'no_trigger_origins': sum(c['trigger_step'] is None for c in coverage),
              'max_edit_error': maximum_edit_error, 'max_actor_error': maximum_actor_error,
              'interpretation': 'descriptive development experiment; partial or inactive exposures retain denominators; strict matched contrast requires every scheduled step valid and all 3 dictionaries at origin level'}
    (destination / 'independent_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    analyze(parser.parse_args().run)
