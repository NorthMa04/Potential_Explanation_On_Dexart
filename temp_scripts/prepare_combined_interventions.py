"""Freeze combined SAE suppression using existing train/validation evidence."""
import argparse
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

sys.dont_write_bytecode = True
import numpy as np
import torch

from temp_scripts.combined_intervention import CombinedEditor, group_definitions
from temp_scripts.physical_runtime import NumericSAE, actor_tail, verify_bundle
from temp_scripts.project_paths import PROJECT_ROOT, OUTPUT_ROOT, output_path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def opening_ranking(folder):
    with (folder / 'coordinate_ranking.csv').open(encoding='utf-8-sig') as stream:
        rows = [r for r in csv.DictReader(stream) if r['target'] == 'opening_rad' and r['representation'] == 'z']
    rows.sort(key=lambda r: int(r['rank']))
    coordinates = [int(r['coordinate']) for r in rows[:6]]
    if len(set(coordinates)) != 6:
        raise ValueError('Need six distinct train-ranked opening coordinates')
    return coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=PROJECT_ROOT / 'outputs/state_assisted/20261102_122123_738477')
    parser.add_argument('--numeric-bundle', type=Path, default=PROJECT_ROOT / 'outputs/physical_simulation/20261102_pilot_v1/bundle_v2')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--group-mode', choices=('top', 'factorial'), default='top')
    parser.add_argument('--origins-per-object', type=int, default=2)
    parser.add_argument('--first-environment-seed', type=int, default=2026110300)
    parser.add_argument('--audit-states-per-split', type=int, default=12)
    parser.add_argument('--random-groups', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.origins_per_object <= 8 or args.audit_states_per_split < 1:
        parser.error('Use 1..8 origins per object and at least one audit state per split')
    torch.set_num_threads(2)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    root = output_path(args.out or OUTPUT_ROOT / 'combined_physical' / stamp)
    root.mkdir(parents=True, exist_ok=False)
    bundle = root / 'bundle'
    shutil.copytree(args.numeric_bundle, bundle)
    verified = verify_bundle(bundle)
    if not verified['passed']:
        raise RuntimeError('Existing numeric model verification failed')
    with np.load(args.study / 'selected_arrays.npz', allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    if set(arrays['split']) != {'train', 'validation'}:
        raise ValueError('Preparation accepts original train and validation only')
    numeric = {p.name: NumericSAE(p) for p in sorted((bundle / 'models').iterdir())}
    metadata = {'version': 'combined_opening_v1', 'groups': group_definitions(args.group_mode),
                'suppression_fraction': 1., 'relative_norm_cap': .1, 'models': {},
                'selection': 'existing train absolute opening correlation rank, frozen before new physical outcomes',
                'common_norm': 'minimum positive capped projected target norm over all six models and planned groups at the SAME current h; inactive candidates remain explicit no-ops',
                'common_norm_scope': 'equal local-state budget among supported candidates; divergent closed-loop branches can have different subsequent budgets',
                'selected_arrays_sha256': digest(args.study / 'selected_arrays.npz')}
    source_hashes = {'selected_arrays.npz': metadata['selected_arrays_sha256']}
    for kind, config in (('pure', 'dense_l0p0'), ('assisted', 'slots_s0.3_a1.0')):
        for seed in (0, 1, 2):
            folder = args.study / ('%s_seed%d' % (config, seed))
            name = '%s_seed%d' % (kind, seed)
            metadata['models'][name] = {'configuration': config, 'seed': seed,
                'opening_coordinates': opening_ranking(folder), 'encoder_top_k': numeric[name].meta['top_k']}
            source_hashes[folder.name + '/coordinate_ranking.csv'] = digest(folder / 'coordinate_ranking.csv')
    ridge_folder = args.study / 'controls/ridge_rotation_10p0_coordinate'
    ridge_indices = opening_ranking(ridge_folder)
    with np.load(ridge_folder / 'projection.npz', allow_pickle=False) as data:
        rotation, scale, mean = data['rotation'], data['h_scale'], data['h_mean']
    train = arrays['split'] == 'train'
    rz = ((arrays['h'][train] - mean) / scale) @ rotation
    signs = np.asarray([np.sign(np.corrcoef(rz[:, j], arrays['y'][train, 1])[0, 1]) for j in ridge_indices])
    if not np.isfinite(signs).all() or (signs == 0).any():
        raise ValueError('Degenerate Ridge train direction')
    np.savez_compressed(bundle / 'combined_directions.npz',
                        ridge_oriented=rotation[:, ridge_indices] * scale[:, None] * signs,
                        train_min=arrays['h'][train].min(axis=0), train_max=arrays['h'][train].max(axis=0))
    source_hashes['ridge_projection.npz'] = digest(ridge_folder / 'projection.npz')
    source_hashes['ridge_ranking.csv'] = digest(ridge_folder / 'coordinate_ranking.csv')
    metadata['ridge_coordinates'] = ridge_indices
    (bundle / 'combined_intervention.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    controls = ['target', 'random0', 'random1', 'random2', 'ridge']
    if args.random_groups:
        controls += ['random_group0', 'random_group1', 'random_group2']
    plan = {'version': 'combined_opening_physical_v1', 'frozen_beijing': datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'stage': 'development physical extension', 'objects': [11395, 11405, 11406, 11477],
            'environment_seeds': list(range(args.first_environment_seed, args.first_environment_seed + args.origins_per_object)),
            'models': list(metadata['models']), 'groups': list(metadata['groups']),
            'durations': [1, 5, -1], 'duration_minus_one': 'from trigger through episode termination',
            'doses': ['nominal', 'common_norm'], 'controls': controls,
            'trigger_opening_rad': .05, 'trigger_rule': 'first pre-action opening >= .05 rad; no SAE-code or eventual-outcome filtering',
            'restore_repeat_atol': 1e-5, 'parallel_cases': 2, 'child_timeout_seconds': 300,
            'render_conditions': ['natural'],
            'repo': '/hy-tmp/embodied_sae/third_party/dexart-release',
            'primary_outcome': 'signed opening difference versus paired natural at trigger + 5 control steps; target minus mean of all 3 matched random directions',
            'secondary_outcomes': ['1/20/50-step signed opening', 'success', 'termination steps', 'contact loss', 'distance', 'cumulative actual h norm', 'per-step achieved target and other codes'],
            'missingness': 'fixed windows missing after either branch terminates; no terminal-value carry forward; no-trigger, inactive edits, capacity failures and failed rollouts retained',
            'statistics': 'average dictionary seeds within each physical origin; origins and object strata are sampling units',
            'interaction_scope': 'factorial branches descriptive; projected/capped input nonadditivity must be checked separately before attributing network coupling'}
    per_origin = 2 + len(plan['models']) * len(plan['groups']) * len(plan['durations']) * len(plan['doses']) * len(controls)
    plan['maximum_planned_rollouts'] = len(plan['objects']) * len(plan['environment_seeds']) * per_origin
    (root / 'physical_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    editor = CombinedEditor(bundle, numeric)
    selected = []
    for split in ('train', 'validation'):
        indices = np.flatnonzero(arrays['split'] == split)
        # Sample one deterministic frame per trajectory before taking evenly spaced trajectories.
        _, first = np.unique(arrays['trajectory_id'][indices], return_index=True)
        candidates = indices[np.sort(first)]
        take = np.linspace(0, len(candidates) - 1, min(len(candidates), args.audit_states_per_split), dtype=int)
        selected.extend(candidates[take].tolist())
    rows, expected, inputs = [], [], []
    with np.load(bundle / 'actor.npz', allow_pickle=False) as data:
        actor = {k: data[k].copy() for k in data.files}
    for number, i in enumerate(selected):
        h = arrays['h'][i]
        for name in plan['models']:
            for group in plan['groups']:
                for dose in plan['doses']:
                    for control in controls:
                        key = '%s|%d' % (arrays['trajectory_id'][i], arrays['state_index'][i])
                        emitted, detail = editor.edit(h, name, group, dose, control, key)
                        if (emitted < 0).any() or not np.isfinite(emitted).all():
                            raise RuntimeError('Invalid emitted activation')
                        if detail['status'] == 'ok' and detail['norm_error'] > 2e-6:
                            raise RuntimeError('Post-emission norm match failed')
                        rows.append(dict(model=name, group=group, dose=dose, control=control, random_key=key,
                            split=str(arrays['split'][i]), trajectory=str(arrays['trajectory_id'][i]),
                            state_index=int(arrays['state_index'][i]), status=detail['status'],
                            budget=detail['budget'], actual_norm=detail['actual_norm'], norm_error=detail['norm_error'],
                            target_before=detail['target_before'], target_after=detail['target_after']))
                        inputs.append(h)
                        expected.append(emitted)
        print('Development state %d/%d audited' % (number + 1, len(selected)), flush=True)
    with (root / 'development_edits.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(bundle / 'combined_reference.npz', h=np.asarray(inputs), emitted=np.asarray(expected),
                        mu=actor_tail(actor, np.asarray(expected)))
    (bundle / 'combined_reference.json').write_text(json.dumps(rows), encoding='utf-8')
    audit = {'passed': True, 'states': len(selected), 'rows': len(rows), 'scope': 'deterministic development sample plus full historical numeric reference',
             'status_counts': {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})},
             'max_successful_norm_error': max((r['norm_error'] for r in rows if r['status'] == 'ok'), default=0.),
             'source_hashes': source_hashes, 'numeric_verification': verified}
    (root / 'development_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    code = root / 'code'
    code.mkdir()
    for filename in ('physical_simulation.py', 'physical_runtime.py', 'constrained_intervention.py',
                     'combined_intervention.py', 'combined_campaign.py', 'combined_analysis.py'):
        shutil.copy2(PROJECT_ROOT / 'temp_scripts' / filename, code / filename)
    shutil.copy2(Path(__file__), code / Path(__file__).name)
    shutil.copy2(PROJECT_ROOT / 'temp_scripts/analyze_combined_physical.py', code / 'analyze_combined_physical.py')
    shutil.copy2(PROJECT_ROOT / '13_run_combined_offline.py', code / '13_run_combined_offline.py')
    files = {p.relative_to(root).as_posix(): digest(p) for folder in (bundle, code) for p in sorted(folder.rglob('*')) if p.is_file()}
    files['physical_plan.json'] = digest(root / 'physical_plan.json')
    (root / 'protocol_hashes.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    with tarfile.open(root / 'deploy.tar.gz', 'w:gz') as archive:
        for name in ('bundle', 'code', 'physical_plan.json', 'protocol_hashes.json'):
            archive.add(root / name, arcname=name)
    latest = OUTPUT_ROOT / 'combined_physical/latest.json'
    latest.parent.mkdir(parents=True, exist_ok=True)
    pending = latest.with_suffix('.pending.json')
    pending.write_text(json.dumps({'run_dir': str(root), 'archive_sha256': digest(root / 'deploy.tar.gz')}, indent=2), encoding='utf-8')
    pending.replace(latest)
    print('READY ' + str(root), flush=True)
    print('Maximum complete rollouts: %d' % plan['maximum_planned_rollouts'], flush=True)


if __name__ == '__main__':
    main()
