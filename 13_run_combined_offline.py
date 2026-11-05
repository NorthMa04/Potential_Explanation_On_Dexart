"""Run full train/validation fixed-input combination tests locally, using the frozen protocol."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch
from temp_scripts.combined_intervention import CombinedEditor
from temp_scripts.physical_runtime import NumericSAE, actor_tail
from temp_scripts.project_paths import PROJECT_ROOT, OUTPUT_ROOT, output_path


METRICS = ('actual_h_norm', 'target_code_remaining_fraction', 'other_code_l2',
           'raw_action_l2', 'clipped_action_l2', 'opening_probe_delta_degrees',
           'distance_probe_delta_m', 'contact_probe_delta')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--study', type=Path, default=PROJECT_ROOT / 'outputs/state_assisted/20261102_122123_738477')
    parser.add_argument('--max-states', type=int, help='Optional deterministic smoke sample, recorded separately')
    args = parser.parse_args()
    if args.max_states is not None and args.max_states < 1:
        parser.error('max-states must be positive')
    if args.run is None:
        args.run = Path(json.loads((OUTPUT_ROOT / 'combined_physical/latest.json').read_text(encoding='utf-8'))['run_dir'])
    root = output_path(args.run)
    metadata = json.loads((root / 'bundle/combined_intervention.json').read_text(encoding='utf-8'))
    source = args.study / 'selected_arrays.npz'
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata['selected_arrays_sha256']:
        raise RuntimeError('Original development source hash differs')
    with np.load(source, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    if set(arrays['split']) != {'train', 'validation'}:
        raise ValueError('Only original train and validation are accepted')
    plan = json.loads((root / 'physical_plan.json').read_text(encoding='utf-8'))
    numeric = {p.name: NumericSAE(p) for p in (root / 'bundle/models').iterdir()}
    editor = CombinedEditor(root / 'bundle', numeric)
    with np.load(root / 'bundle/actor.npz', allow_pickle=False) as data:
        actor = {k: data[k].copy() for k in data.files}
    torch.set_num_threads(2)
    indices = np.arange(len(arrays['h']))
    if args.max_states is not None:
        indices = indices[np.linspace(0, len(indices) - 1, min(len(indices), args.max_states), dtype=int)]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    out = root / 'offline' / stamp
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    manifest = {'status': 'running', 'source_sha256': metadata['selected_arrays_sha256'],
                'protocol_sha256': hashlib.sha256((root / 'protocol_hashes.json').read_bytes()).hexdigest(),
                'states': len(indices), 'sample_only': args.max_states is not None,
                'scope': 'fixed original train/validation inputs; no physical movement inferred from probe readout'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    sums = defaultdict(lambda: {'n': 0, 'valid': 0, 'totals': np.zeros(len(METRICS))})
    statuses = defaultdict(int)
    fieldnames = ['split', 'trajectory', 'state_index', 'model', 'group', 'dose', 'control',
                  'status', 'target_indices', 'requested_budget', 'norm_error'] + list(METRICS)
    rows = 0
    with (out / 'fixed_input_results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for number, i in enumerate(indices):
            h = arrays['h'][i]
            baseline_mu = actor_tail(actor, h[None])[0]
            baseline_action = baseline_mu.clip(-1, 1)
            for name in plan['models']:
                model = numeric[name]
                before_z = model.infer(h)[0]
                before_pred = model.predict(before_z)[0]
                edits, details, labels = [], [], []
                for group in plan['groups']:
                    for dose in plan['doses']:
                        for control in plan['controls']:
                            key = '%s|%d' % (arrays['trajectory_id'][i], arrays['state_index'][i])
                            emitted, detail = editor.edit(h, name, group, dose, control, key)
                            edits.append(emitted)
                            details.append(detail)
                            labels.append((group, dose, control))
                edited = np.asarray(edits)
                means = actor_tail(actor, edited)
                predictions = model.predict(np.asarray([d['z_after'] for d in details]))
                for k, (group, dose, control) in enumerate(labels):
                    detail = details[k]
                    values = np.asarray([
                        detail['actual_norm'], detail['target_after'] / max(detail['target_before'], 1e-12),
                        detail['nontarget_code_l2'], np.linalg.norm(means[k] - baseline_mu),
                        np.linalg.norm(means[k].clip(-1, 1) - baseline_action),
                        np.rad2deg(predictions[k, 1] - before_pred[1]),
                        predictions[k, 0] - before_pred[0], predictions[k, 2] - before_pred[2]])
                    if detail['target_before'] <= 1e-12:
                        values[1] = np.nan
                    if (edited[k] < 0).any() or not np.isfinite(edited[k]).all():
                        raise RuntimeError('Invalid activation')
                    if detail['status'] == 'ok' and detail['norm_error'] > 2e-6:
                        raise RuntimeError('Float32 budget mismatch')
                    row = dict(split=str(arrays['split'][i]), trajectory=str(arrays['trajectory_id'][i]),
                               state_index=int(arrays['state_index'][i]), model=name, group=group, dose=dose,
                               control=control, status=detail['status'], target_indices=json.dumps(detail['target_indices']),
                               requested_budget=detail['budget'], norm_error=detail['norm_error'])
                    row.update({metric: float(value) if np.isfinite(value) else None for metric, value in zip(METRICS, values)})
                    writer.writerow(row)
                    group_key = (row['split'], name, group, dose, control, row['trajectory'])
                    item = sums[group_key]
                    item['n'] += 1
                    # Conditional metrics use valid edits and keep coverage separately.
                    if detail['status'] == 'ok':
                        item['valid'] += 1
                        item['totals'] += values
                    statuses[detail['status']] += 1
                    rows += 1
            if number % 25 == 0 or number + 1 == len(indices):
                stream.flush()
                print('Offline state %d/%d; %d variants' % (number + 1, len(indices), rows), flush=True)
    summaries = defaultdict(list)
    counts = defaultdict(lambda: [0, 0])
    for key, value in sums.items():
        condition = key[:-1]
        counts[condition][0] += value['n']
        counts[condition][1] += value['valid']
        if value['valid']:
            summaries[condition].append(value['totals'] / value['valid'])
    summary_rows = []
    for key, count in counts.items():
        row = dict(zip(('split', 'model', 'group', 'dose', 'control'), key))
        row.update(planned_states=count[0], valid_states=count[1], valid_trajectories=len(summaries[key]))
        average = np.mean(summaries[key], axis=0) if summaries[key] else np.full(len(METRICS), np.nan)
        row.update({metric: float(value) if np.isfinite(value) else None for metric, value in zip(METRICS, average)})
        summary_rows.append(row)
    with (out / 'trajectory_equal_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest.update(status='completed', variants=rows, status_counts=dict(statuses))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (out.parent / 'latest.json').write_text(json.dumps({'run_dir': str(out)}, indent=2), encoding='utf-8')
    print('OFFLINE COMPLETE ' + str(out), flush=True)


if __name__ == '__main__':
    main()
