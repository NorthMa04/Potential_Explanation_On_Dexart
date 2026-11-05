"""Prepare and audit train-defined legal-domain feature directions; never opens tests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
import numpy as np
from temp_scripts.physical_runtime import NumericSAE, actor_tail
from temp_scripts.constrained_intervention import ConstrainedEditor


def ranking(folder, representation):
    with (folder / 'coordinate_ranking.csv').open(encoding='utf-8-sig') as stream:
        return int(next(r['coordinate'] for r in csv.DictReader(stream)
                        if r['target'] == 'opening_rad' and r['representation'] == representation and r['rank'] == '1'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, default=Path('outputs/state_assisted/20261102_122123_738477'))
    p.add_argument('--numeric-bundle', type=Path, default=Path('outputs/physical_simulation/20261102_pilot_v1/bundle_v2'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    bundle = args.out / 'bundle'
    shutil.copytree(args.numeric_bundle, bundle)
    with np.load(args.study / 'selected_arrays.npz') as data:
        x = {k: data[k].copy() for k in data.files}
    assert set(x['split']) == {'train', 'validation'}
    train = x['split'] == 'train'
    groups, counts = np.unique(x['trajectory_id'][train], return_counts=True)
    count_by_group = dict(zip(groups, counts))
    w = np.asarray([1. / count_by_group[g] for g in x['trajectory_id'][train]])
    w /= w.sum()
    def quantile(v, weights, q):
        order = np.argsort(v, kind='stable')
        return float(v[order[min(np.searchsorted(np.cumsum(weights[order]), q), len(v) - 1)]])
    numeric = {folder.name: NumericSAE(folder) for folder in (bundle / 'models').iterdir()}
    ridge_folder = args.study / 'controls/ridge_rotation_10p0_coordinate'
    with np.load(ridge_folder / 'projection.npz') as data:
        rotation, rscale, rmean = data['rotation'], data['h_scale'], data['h_mean']
    ridge_j = ranking(ridge_folder, 'z')
    rz = ((x['h'][train] - rmean) / rscale) @ rotation
    ridge_corr = np.corrcoef(rz[:, ridge_j], x['y'][train, 1])[0, 1]
    directions, models = {}, {}
    for seed in (0, 1, 2):
        name = 'assisted_seed%d' % seed
        folder = args.study / ('slots_s0.3_a1.0_seed%d' % seed)
        j, raw_j = ranking(folder, 'z'), ranking(folder, 'h1')
        z = numeric[name].infer(x['h'][train])[0][:, j]
        active = z > 0
        wa = w[active] / w[active].sum()
        mean = float(wa @ z[active])
        std = float(np.sqrt(wa @ (z[active] - mean) ** 2))
        model = numeric[name]
        directions[name + '__target'] = model.w['decoder_weight'][:, j] * model.w['scale']
        corr = float(np.corrcoef(x['h'][train, raw_j], x['y'][train, 1])[0, 1])
        raw = np.zeros(64)
        raw[raw_j] = np.sign(corr)
        directions[name + '__raw'] = raw
        pure = numeric['pure_seed%d' % seed]
        pure_j = ranking(args.study / ('dense_l0p0_seed%d' % seed), 'z')
        pure_z = pure.infer(x['h'][train])[0][:, pure_j]
        pure_sign = np.sign(np.corrcoef(pure_z, x['y'][train, 1])[0, 1])
        directions[name + '__pure'] = pure.w['decoder_weight'][:, pure_j] * pure.w['scale'] * pure_sign
        directions[name + '__ridge'] = rotation[:, ridge_j] * rscale * np.sign(ridge_corr)
        models[name] = {'feature': j, 'q80': quantile(z, w, .8), 'q99': quantile(z, w, .99),
                        'active_std': std, 'pure_feature': pure_j, 'raw_feature': raw_j,
                        'ridge_feature': ridge_j, 'target_correlation': float(np.corrcoef(z, x['y'][train, 1])[0, 1])}
    np.savez_compressed(bundle / 'directions.npz', **directions,
                        train_min=x['h'][train].min(axis=0), train_max=x['h'][train].max(axis=0))
    metadata = {'format': 'constrained_opening_v1', 'sae_seeds': [0, 1, 2], 'models': models,
                'relative_norm_cap': .1, 'random_directions': 3,
                'ridge_control': 'shared ridge_rotation_10p0_coordinate, train Top1',
                'constraint': 'ReLU projection; residual retained before projection; controls matched AFTER projection',
                'feature_selection': 'existing train-only opening Top1; not chosen by actor/physical effect',
                'trigger': 'current opening >= .05 and all three target codes in [train q80, q99)',
                'selected_arrays_sha256': hashlib.sha256((args.study / 'selected_arrays.npz').read_bytes()).hexdigest()}
    (bundle / 'intervention.json').write_text(json.dumps(metadata, indent=2))
    editor = ConstrainedEditor(bundle, numeric)
    with np.load(bundle / 'actor.npz') as a:
        actor = {k: a[k] for k in a.files}
    rows = []
    for split in ('train', 'validation'):
        indices = np.flatnonzero(x['split'] == split)
        # State eligibility uses all three models and train-derived thresholds.
        eligible = [i for i in indices if x['y'][i, 1] >= .05 and editor.trigger_eligible(x['h'][i])]
        for i in eligible:
            h = x['h'][i]
            for seed in (0, 1, 2):
                name = 'assisted_seed%d' % seed
                for op in ('half_suppress', 'suppress', 'enhance'):
                    for control in ('target', 'random0', 'random1', 'random2', 'raw', 'pure', 'ridge'):
                        emitted, detail = editor.edit(h, name, op, control, '%s|%d' % (x['trajectory_id'][i], x['state_index'][i]))
                        mu = actor_tail(actor, np.vstack((h, emitted)))
                        detail.update(split=split, trajectory=str(x['trajectory_id'][i]), state_index=int(x['state_index'][i]),
                            model=name, operation=op, control=control,
                            mu_l2=float(np.linalg.norm(mu[1] - mu[0])),
                            action_l2=float(np.linalg.norm(mu[1].clip(-1, 1) - mu[0].clip(-1, 1))))
                        rows.append(detail)
        print(split, 'eligible states', len(eligible), '/', len(indices), flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (args.out / 'development_edits.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    summary = {'data_scope': 'only original train/validation; no test arrays loaded', 'rows': len(rows), 'groups': []}
    for split in ('train', 'validation'):
        for op in ('half_suppress', 'suppress', 'enhance'):
            selected = [r for r in rows if r['split'] == split and r['control'] == 'target' and r['operation'] == op]
            if selected:
                summary['groups'].append({'split': split, 'operation': op, 'states_times_seeds': len(selected),
                    'correct_code_sign_fraction': float(np.mean([r['code_retention'] > 0 for r in selected])),
                    'median_code_retention': float(np.median([r['code_retention'] for r in selected])),
                    'max_negative_fraction': max(r['negative_fraction'] for r in selected),
                    'median_action_l2': float(np.median([r['action_l2'] for r in selected]))})
    summary['control_match_failures'] = sum(r['status'] != 'ok' for r in rows)
    summary['max_match_error'] = max(r.get('norm_error', 0) for r in rows)
    (args.out / 'development_audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
