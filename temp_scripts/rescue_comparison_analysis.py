"""Audit saved 22D vectors and compare restoration directions on common support."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from temp_scripts.physical_runtime import actor_tail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    root = args.run
    folder = root / 'remote/exploratory/direction_comparison'
    with (folder / 'results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with np.load(folder / 'vectors.npz') as data:
        v = {k: data[k].copy() for k in data.files}
    with np.load(root / 'bundle/actor.npz') as data:
        actor = {k: data[k].copy() for k in data.files}
    assert len(rows) == len(v['h_clean'])
    max_actor, max_norm, max_metric = 0., 0., 0.
    for label in ('clean', 'corrupt', 'edited'):
        prediction = actor_tail(actor, v['h_' + label]).clip(-1, 1)
        np.testing.assert_allclose(prediction, v['action_' + label], atol=2e-5, rtol=2e-5)
        max_actor = max(max_actor, float(np.max(np.abs(prediction - v['action_' + label]))))
    before = np.linalg.norm(v['action_corrupt'].astype(float) - v['action_clean'], axis=1)
    after = np.linalg.norm(v['action_edited'].astype(float) - v['action_clean'], axis=1)
    norm = np.linalg.norm(v['h_edited'].astype(float) - v['h_corrupt'], axis=1)
    assert (v['h_edited'] >= 0).all()
    lookup = {}
    for i, row in enumerate(rows):
        error = max(abs(float(row['baseline_action_error_l2']) - before[i]),
                    abs(float(row['after_action_error_l2']) - after[i]),
                    abs(float(row['absolute_error_reduction']) - (before[i] - after[i])))
        assert error < 2e-6
        max_metric = max(max_metric, error)
        if row['status'] == 'ok':
            e = abs(norm[i] - float(row['budget']))
            assert e < 2e-6
            max_norm = max(max_norm, e)
        key = (row['trajectory'], row['state_index'], row['corruption'], row['sae_seed'], row['coordinates'], row['variant'])
        assert key not in lookup
        lookup[key] = (row, before[i] - after[i])
    # New strong controls must not change the earlier corrected opening treatment.
    with (root / 'remote/exploratory/rescue_results_v2/results.csv').open() as stream:
        earlier = list(csv.DictReader(stream))
    invariant = 0.
    for row in earlier:
        if row['variant'] != 'feature_restore': continue
        key = (row['trajectory'], row['state_index'], row['corruption'], row['sae_seed'], row['coordinates'], 'opening')
        invariant = max(invariant, abs(float(row['absolute_error_reduction']) - lookup[key][1]))
    assert invariant < 2e-5
    comparisons = []
    for corruption in ('handle_jitter_5mm', 'handle_jitter_20mm', 'handle_dropout_half'):
        for size in ('1', '4'):
            for alternative in ('distance', 'contact', 'pure', 'ridge', 'raw', 'random_mean'):
                by_origin = defaultdict(list)
                for key, (row, opening_effect) in lookup.items():
                    tid, state, corr, seed, coordinates, variant = key
                    if corr != corruption or coordinates != size or variant != 'opening' or row['status'] != 'ok': continue
                    alternatives = ['random0', 'random1', 'random2'] if alternative == 'random_mean' else [alternative]
                    other = [lookup.get((tid, state, corr, seed, coordinates, label)) for label in alternatives]
                    if all(value is not None and value[0]['status'] == 'ok' for value in other):
                        assert all(abs(float(value[0]['budget']) - float(row['budget'])) < 2e-6 for value in other)
                        by_origin[(tid, state)].append((opening_effect, float(np.mean([value[1] for value in other]))))
                complete = [values for values in by_origin.values() if len(values) == 3]
                opening = [float(np.mean([x[0] for x in values])) for values in complete]
                other = [float(np.mean([x[1] for x in values])) for values in complete]
                comparisons.append({'corruption': corruption, 'coordinates': int(size), 'alternative': alternative,
                    'complete_3seed_trajectories': len(complete),
                    'opening_mean_error_reduction_l2': float(np.mean(opening)) if opening else None,
                    'alternative_mean_error_reduction_l2': float(np.mean(other)) if other else None,
                    'opening_minus_alternative_error_reduction_l2': float(np.mean(np.asarray(opening)-other)) if opening else None,
                    'scope': 'exploratory train-only paired points; positive favors opening; no physical or independent test outcomes'})
    status_counts = defaultdict(lambda: defaultdict(int))
    for row in rows: status_counts[row['variant']][row['status']] += 1
    report = {'passed': True, 'rows': len(rows), 'all_vectors_22D': True, 'h_nonnegative': True,
              'max_actor_clipped_abs': max_actor, 'max_matched_norm_error': max_norm,
              'max_metric_error': max_metric, 'opening_treatment_vs_v2_max_error': invariant,
              'status_by_method': {k: dict(v) for k, v in status_counts.items()},
              'comparisons': comparisons,
              'interpretation': 'same-budget directions with clean-code oracle; selected group overlap and generic denoising confound semantic specificity; train pilot only'}
    (root / 'rescue_strong_control_audit.json').write_text(json.dumps(report, indent=2))
    with (root / 'rescue_strong_control_comparisons.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0])); writer.writeheader(); writer.writerows(comparisons)
    print(json.dumps({k: v for k, v in report.items() if k != 'comparisons'}, indent=2))
    print(json.dumps([r for r in comparisons if r['corruption']=='handle_jitter_20mm' and r['coordinates']==4], indent=2))


if __name__ == '__main__': main()
