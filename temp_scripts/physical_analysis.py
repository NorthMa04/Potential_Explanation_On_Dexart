"""Independent local audit and descriptive summary of the physical development pilot."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from temp_scripts.physical_runtime import NumericSAE, actor_tail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--completed-only', action='store_true')
    args = p.parse_args()
    root = args.run.resolve()
    campaign = json.loads((root / 'remote/fresh_campaign/campaign.json').read_text())
    if args.completed_only:
        campaign['cases'] = [c for c in campaign['cases'] if c.get('status') == 'completed']
        assert campaign['cases']
    else:
        assert campaign['status'] == 'completed', campaign['status']
    bundle = root / 'bundle_v2'
    models = {d.name: NumericSAE(d) for d in (bundle / 'models').iterdir()}
    with np.load(bundle / 'actor.npz') as data:
        actor = {k: data[k].copy() for k in data.files}
    rows = []
    max_actor, max_replacement, max_prefix = 0., 0., 0.
    def load(path):
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}
    for case in campaign['cases']:
        folder = root / 'remote/fresh_campaign' / ('object%d_seed%d' % (case['object'], case['seed']))
        baseline = load(folder / 'natural/natural.npz')
        hits = np.flatnonzero(baseline['opening'] >= .05)
        trigger = int(hits[0]) if len(hits) else None
        for record in case['runs']:
            name = record['name']
            a = load(folder / name / (name + '.npz'))
            assert record['exit_code'] == 0 and record['actor_parameters_unchanged']
            assert a['done'][-1] and not a['done'][:-1].any()
            n = len(a['action'])
            assert np.array_equal(a['step'], np.arange(n))
            assert all(np.isfinite(x).all() for x in a.values())
            np.testing.assert_array_equal(a['action'], np.clip(a['mu'], -1, 1))
            for key in ('opening', 'distance', 'contacts', 'robot_qpos', 'object_qpos', 'robot_qvel', 'object_qvel'):
                np.testing.assert_array_equal(a['post_' + key][:-1], a[key][1:])
            mu = actor_tail(actor, a['h_applied'])
            actor_error = float(np.max(np.abs(mu - a['mu'])))
            max_actor = max(max_actor, actor_error)
            np.testing.assert_allclose(mu, a['mu'], atol=2e-5, rtol=2e-5)
            intervention = name.startswith(('pure_', 'assisted_'))
            if intervention:
                assert trigger is not None
                kind, seed_string, projection, duration_string = name.split('_')
                duration, sae_seed = int(duration_string), int(seed_string[4:])
                expected_mask = (a['step'] >= trigger) & (a['step'] < trigger + duration)
                np.testing.assert_array_equal(a['applied'], expected_mask)
                _, rec, _ = models[kind + '_' + seed_string].infer(a['h_original'][expected_mask])
                expected = np.maximum(rec, 0) if projection == 'relu' else rec
                error = float(np.max(np.abs(expected.astype(np.float32) - a['h_applied'][expected_mask])))
                max_replacement = max(max_replacement, error)
                np.testing.assert_allclose(a['h_applied'][expected_mask], expected.astype(np.float32), atol=2e-5, rtol=2e-5)
                np.testing.assert_array_equal(a['h_applied'][~expected_mask], a['h_original'][~expected_mask])
                if projection == 'relu':
                    assert np.all(a['h_applied'] >= 0)
                prefix_length = trigger + 1
            else:
                kind, projection, duration, sae_seed = 'original', 'none', 0, -1
                np.testing.assert_array_equal(a['h_applied'], a['h_original'])
                prefix_length = n
            for key in a:
                if key.startswith('obs__') or key in ('h_original', 'opening', 'distance', 'robot_qpos', 'object_qpos', 'contacts'):
                    err = float(np.max(np.abs(a[key][:prefix_length].astype(float) - baseline[key][:prefix_length].astype(float))))
                    max_prefix = max(max_prefix, err)
                    assert err <= 1e-5, (case['object'], name, key, err)
            action_prefix = trigger if intervention else n
            np.testing.assert_allclose(a['action'][:action_prefix], baseline['action'][:action_prefix], atol=1e-5, rtol=0)
            row = {'object': case['object'], 'environment_seed': case['seed'], 'name': name,
                   'kind': kind, 'sae_seed': sae_seed, 'projection': projection, 'duration': duration,
                   'trigger_step': trigger, 'steps': n, 'baseline_steps': len(baseline['step']),
                   'success': bool(np.any(a['post_success'])), 'final_opening_rad': float(a['post_opening'][-1]),
                   'applied_steps': int(a['applied'].sum()), 'max_negative_fraction': float(a['negative_fraction'].max()),
                   'max_delta_norm': float(a['delta_norm'].max()),
                   'action_mu_max_abs_audit': actor_error}
            for horizon in (1, 3, 5):
                index = trigger + horizon - 1 if trigger is not None else n
                available = index < n and index < len(baseline['step'])
                row['opening_delta_at_%d_rad' % horizon] = float(a['post_opening'][index] - baseline['post_opening'][index]) if available else None
                row['contact_at_%d' % horizon] = bool(a['post_contact'][index]) if index < n else None
                row['strict_contact_at_%d' % horizon] = bool(a['post_strict_contact'][index]) if index < n else None
            rows.append(row)
    with (root / 'physical_results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    interventions = [r for r in rows if r['kind'] != 'original']
    groups = []
    for kind in ('pure', 'assisted'):
        for projection in ('raw', 'relu'):
            for duration in (1, 3):
                selected = [r for r in interventions if (r['kind'], r['projection'], r['duration']) == (kind, projection, duration)]
                if not selected:
                    continue
                groups.append({'kind': kind, 'projection': projection, 'duration': duration,
                    'rollouts': len(selected), 'successes': sum(r['success'] for r in selected),
                    'delta_steps_mean': float(np.mean([r['steps'] - r['baseline_steps'] for r in selected])),
                    'opening_delta_at_3_deg_mean': float(np.mean([r['opening_delta_at_3_rad'] for r in selected])) * 180 / np.pi,
                    'opening_delta_at_5_deg_mean': float(np.mean([r['opening_delta_at_5_rad'] for r in selected])) * 180 / np.pi})
    audit = {'passed': True, 'rollouts': len(rows), 'interventions': len(interventions),
             'partial_completed_cases_only': args.completed_only,
             'independent_environment_cases': len(campaign['cases']),
             'max_prefix_abs': max_prefix, 'max_replacement_abs': max_replacement, 'max_actor_tail_abs': max_actor,
             'checks': ['whole trajectories finite and terminal', 'physical T to T+1 chain exact',
                        'raw mean clipping equals executed action exactly', 'frozen actor tail independently reproduced',
                        'applied h matches declared raw/ReLU reconstruction and fixed time mask',
                        'full noisy observations/physics/original h1 match through trigger',
                        'prior executed actions match baseline', 'actor parameters unchanged in each process'],
             'groups': groups, 'limitations': '%d development cases on %d objects; SAE seeds share environmental cases; descriptive only, not an independent success-rate or semantic-causality study' % (len(campaign['cases']), len(set(c['object'] for c in campaign['cases'])))}
    (root / 'independent_physical_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
