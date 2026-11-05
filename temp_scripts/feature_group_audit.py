"""Exploratory development-only Top1/Top4 edits; no policy training or test data."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from temp_scripts.constrained_intervention import ConstrainedEditor, project_delta, match_projected_norm
from temp_scripts.physical_runtime import NumericSAE, actor_tail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--study', type=Path, default=Path('outputs/state_assisted/20261102_122123_738477'))
    args = p.parse_args()
    with np.load(args.study / 'selected_arrays.npz') as data:
        x = {key: data[key].copy() for key in data.files}
    assert set(x['split']) == {'train', 'validation'}
    numeric = {d.name: NumericSAE(d) for d in (args.run / 'bundle/models').iterdir()}
    editor = ConstrainedEditor(args.run / 'bundle', numeric)
    with np.load(args.run / 'bundle/actor.npz') as data:
        actor = {key: data[key].copy() for key in data.files}
    eligible = [i for i in range(len(x['h'])) if x['y'][i, 1] >= .05 and editor.trigger_eligible(x['h'][i])]
    rows = []
    for seed in (0, 1, 2):
        name = 'assisted_seed%d' % seed
        model = numeric[name]
        folder = args.study / ('slots_s0.3_a1.0_seed%d' % seed)
        with (folder / 'coordinate_ranking.csv').open(encoding='utf-8-sig') as stream:
            rank = [r for r in csv.DictReader(stream) if r['target'] == 'opening_rad' and r['representation'] == 'z']
        top4 = [int(r['coordinate']) for r in rank if int(r['rank']) <= 4]
        for i in eligible:
            h = x['h'][i]
            before_z, _, before_pred = model.infer(h)
            z = before_z[0]
            single, detail = editor.edit(h, name, 'suppress', 'target', 'group-audit')
            request = np.zeros_like(z)
            request[top4] = -z[top4]
            raw = (model.w['decoder_weight'] * model.w['scale'][:, None]) @ request
            delta = project_delta(h, raw)
            cap = editor.meta['relative_norm_cap'] * np.linalg.norm(h)
            if np.linalg.norm(delta) > cap:
                delta *= cap / np.linalg.norm(delta)
            group_matched, status = match_projected_norm(h, raw, detail['actual_norm'])
            variants = [('top1', single), ('top4_native_budget', np.maximum(h + delta, 0))]
            if status == 'ok':
                variants.append(('top4_matched_to_top1', np.maximum(h + group_matched, 0)))
            for condition, new_h in variants:
                new_h = np.asarray(new_h, dtype=np.float32)
                new_z, _, prediction = model.infer(new_h)
                means = actor_tail(actor, np.vstack((h, new_h)))
                rows.append({'split': str(x['split'][i]), 'trajectory': str(x['trajectory_id'][i]),
                    'state_index': int(x['state_index'][i]), 'sae_seed': seed, 'condition': condition,
                    'delta_h_norm': float(np.linalg.norm(new_h.astype(float) - h)),
                    'remaining_top4_code_l2_fraction': float(np.linalg.norm(new_z[0, top4]) / max(np.linalg.norm(z[top4]), 1e-12)),
                    'distance_prediction_change_m': float(prediction[0, 0] - before_pred[0, 0]),
                    'opening_prediction_change_degrees': float((prediction[0, 1] - before_pred[0, 1]) * 180 / np.pi),
                    'contact_prediction_change': float(prediction[0, 2] - before_pred[0, 2]),
                    'action_l2': float(np.linalg.norm(means[1].clip(-1, 1) - means[0].clip(-1, 1)))})
    with (args.run / 'group_development_edits.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries = []
    for split in ('train', 'validation'):
        for condition in ('top1', 'top4_native_budget', 'top4_matched_to_top1'):
            group = [r for r in rows if r['split'] == split and r['condition'] == condition]
            by_trajectory = {}
            for row in group:
                by_trajectory.setdefault(row['trajectory'], []).append(row)
            keys = ('delta_h_norm', 'remaining_top4_code_l2_fraction', 'opening_prediction_change_degrees', 'action_l2')
            summaries.append({'split': split, 'condition': condition, 'rows_times_seeds': len(group),
                'trajectories': len(by_trajectory), **{key: float(np.mean([np.mean([r[key] for r in rs]) for rs in by_trajectory.values()])) for key in keys}})
    report = {'scope': 'exploratory after partial physical pilot; original train/validation only; no physical-rule changes',
              'meaning': 'fixed probe readout and code changes, not physical angle changes or information-theoretic erasure',
              'rows': len(rows), 'summaries': summaries}
    (args.run / 'group_development_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
