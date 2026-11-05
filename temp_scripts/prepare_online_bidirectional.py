"""Freeze a portable bidirectional pilot editor; no training or simulation."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from temp_scripts.activation_baseline import sample_weights
from temp_scripts.constrained_intervention import project_delta, match_projected_norm
from temp_scripts.online_bidirectional import OnlineBidirectionalEditor, verify_online_bundle
from temp_scripts.physical_runtime import NumericSAE, actor_tail, verify_bundle
from temp_scripts.project_paths import PROJECT_ROOT
from temp_scripts.state_assisted_intervention import weighted_quantile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def prepare(root):
    torch.set_num_threads(1)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bundle = root / 'bundle'
    if bundle.exists():
        raise RuntimeError('Output bundle already exists; refuse to overwrite')
    source = PROJECT_ROOT / 'outputs/combined_physical/20261103_140606_955714/bundle'
    local = PROJECT_ROOT / 'outputs/local_interpretability/20261104_mainline_v3'
    study = PROJECT_ROOT / 'outputs/state_assisted/20261102_122123_738477'
    shutil.copytree(source, bundle)
    source_hashes = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob('*') if p.is_file()}
    with np.load(study / 'selected_arrays.npz', allow_pickle=False) as arrays:
        saved = {k: arrays[k].copy() for k in arrays.files}
    with np.load(local / 'development_arrays.npz', allow_pickle=False) as arrays:
        development = {k: arrays[k].copy() for k in arrays.files}
    frames = pd.read_csv(local / 'development_frames.csv')
    positions = development['local_positions']
    np.testing.assert_array_equal(development['h'][positions], saved['h'])
    np.testing.assert_array_equal(development['y'][positions], saved['y'])
    np.testing.assert_array_equal(frames.iloc[positions].trajectory.to_numpy(), saved['trajectory_id'])
    np.testing.assert_array_equal(frames.iloc[positions].state_index.to_numpy(), saved['state_index'])
    if len(saved['h']) != 1365 or set(saved['split']) != {'train', 'validation'}:
        raise RuntimeError('Unexpected original development local-window inputs')
    numeric = {d.name: NumericSAE(d) for d in sorted((bundle / 'models').iterdir())}
    ranking = json.loads((local / 'frozen_rankings.json').read_text(encoding='utf-8'))
    old_meta = json.loads((source / 'combined_intervention.json').read_text(encoding='utf-8'))
    train = saved['split'] == 'train'
    train_weights = sample_weights(saved['trajectory_id'][train])
    stats, inference_audits = {}, []
    for name, model in numeric.items():
        z = np.asarray([model.infer(h)[0][0] for h in saved['h']])
        std, upper = np.zeros(64), np.zeros(64)
        for coordinate in range(64):
            active = z[train, coordinate] > 0
            if active.any():
                values, weights = z[train, coordinate][active], train_weights[active]
                mean = np.average(values, weights=weights)
                std[coordinate] = np.sqrt(np.average((values - mean) ** 2, weights=weights))
                upper[coordinate] = weighted_quantile(values, weights, .99)
        stats[name + '__active_std'] = std
        stats[name + '__active_q99'] = upper
        # Verify the v3's *single-state* calling convention, not batch equivalence.
        with np.load(local / ('new_edit_vectors_%s.npz' % name), allow_pickle=False) as previous:
            np.testing.assert_array_equal(z, previous['z_before'])
        batched = model.infer(saved['h'])[0]
        inference_audits.append({'model': name, 'single_state_matches_v3_exactly': True,
                                'batch_vs_single_max_abs': float(np.max(abs(z - batched))),
                                'batch_topk_support_mismatches': int(np.any((z > 0) != (batched > 0), axis=1).sum())})
        print('FROZEN TRAIN BOUNDS', name, flush=True)
    np.savez_compressed(bundle / 'online_statistics.npz', **stats)
    meta = {
        'version': 'bounded_bidirectional_online_v1_20261104',
        'frozen_beijing': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'groups': {'top1': 1, 'top4': 4, 'top6': 6}, 'relative_norm_cap': .1,
        'models': {name: {'opening_coordinates': ranking[name]['opening_rad'][:6]} for name in numeric},
        'dose_mapping': OnlineBidirectionalEditor.DOSES,
        'dose_warning': 'nominal/full=full bidirectional budget; common_norm/half=half. These aliases do NOT use the historical combined common_norm definition.',
        'budget': 'min(projected native full decrease norm, projected bounded increase norm, .1*norm(current h)); per model/group and current state; both signs and all controls use this budget',
        'increase': 'current active coordinates only; min(.5 train-active SD, train-active q99 minus current z), lower bound zero',
        'decrease': 'request current selected codes to zero before projection and norm matching; not exact deletion',
        'bounds': '1029 original train local-window states; equal total trajectory weights; active subsets retain those frame weights; validation excluded',
        'ranking': 'original frozen train absolute opening correlation rankings from v3, checked against old combined bundle',
        'random': 'three deterministic unit Gaussian directions from the supplied fixed origin/model/group key; same directions over pulse steps',
        'ridge': 'positive-opening oriented sum of first N frozen Ridge coordinates, both signs; matched actual h norm/projection, not equivalent SAE semantic operations',
        'semantic_scope': 'coordinate decrease/increase and directed Ridge controls; no claim of monotonic physical opening or pure concepts',
        'reference_code_replacement': 'not implemented; this pilot does not establish complete semantic selectivity',
        'selected_arrays_sha256': digest(study / 'selected_arrays.npz'),
        'development_arrays_sha256': digest(local / 'development_arrays.npz'),
        'frozen_rankings_sha256': digest(local / 'frozen_rankings.json'),
    }
    for name in numeric:
        if meta['models'][name]['opening_coordinates'] != old_meta['models'][name]['opening_coordinates']:
            raise RuntimeError('Original ranking changed')
    dump(bundle / 'online_intervention.json', meta)
    editor = OnlineBidirectionalEditor(bundle, numeric)
    # Original-local strata are represented along with late post-onset and pre-onset
    # states. This is a diagnostic sample, not a fit or an exhaustive experiment.
    phases = np.where(frames.local_window, 'local_window', np.where(frames.post_onset, 'later_post_onset', 'pre_onset'))
    selected = []
    for split in ('train', 'validation'):
        for phase in ('local_window', 'later_post_onset', 'pre_onset'):
            indices = np.flatnonzero((frames.split.to_numpy() == split) & (phases == phase))
            if len(indices):
                _, first = np.unique(frames.trajectory.to_numpy()[indices], return_index=True)
                candidates = indices[np.sort(first)]
                selected.extend(candidates[np.linspace(0, len(candidates)-1, min(3, len(candidates)), dtype=int)].tolist())
    inputs, emissions, codes, reference, diagnostics = [], [], [], [], []
    with np.load(bundle / 'actor.npz', allow_pickle=False) as arrays:
        actor = {k: arrays[k].copy() for k in arrays.files}
    for index in selected:
        h = development['h'][index]
        for name in numeric:
            for group in meta['groups']:
                key = '%s|%d|%s|%s' % (frames.iloc[index].trajectory, frames.iloc[index].state_index, name, group)
                z = numeric[name].infer(h)[0][0]
                base_pred = numeric[name].predict(z[None])[0]
                base_mu = actor_tail(actor, h[None])[0]
                for dose in ('full', 'half'):
                    for control in OnlineBidirectionalEditor.CONTROLS:
                        value, detail = editor.edit(h, name, group, dose, control, key)
                        if not np.isfinite(value).all() or (value < 0).any():
                            raise RuntimeError('Invalid h emission')
                        if detail['status'] == 'ok' and detail['norm_error'] > 2e-6:
                            raise RuntimeError('Float32 norm mismatch')
                        inputs.append(h); emissions.append(value); codes.append(detail['z_after'])
                        reference.append({'model': name, 'group': group, 'dose': dose, 'control': control,
                                          'random_key': key, 'status': detail['status'], 'source_index': int(index)})
                        after_pred = numeric[name].predict(np.asarray(detail['z_after'])[None])[0]
                        mu = actor_tail(actor, value[None])[0]
                        diagnostics.append({'split': frames.iloc[index].split, 'phase': phases[index], 'model': name,
                                            'group': group, 'dose': dose, 'control': control, 'status': detail['status'],
                                            'budget': detail['budget'], 'actual_norm': detail['actual_norm'],
                                            'opening_probe_delta_rad': float(after_pred[1] - base_pred[1]),
                                            'distance_probe_delta_m': float(after_pred[0] - base_pred[0]),
                                            'any_contact_probe_delta': float(after_pred[2] - base_pred[2]),
                                            'raw_mu_l2': float(np.linalg.norm(mu - base_mu)),
                                            'other_code_l2': detail['nontarget_code_l2']})
    dump(bundle / 'online_reference.json', reference)
    np.savez_compressed(bundle / 'online_reference.npz', h=np.asarray(inputs), emitted=np.asarray(emissions),
                        z_after=np.asarray(codes), mu=actor_tail(actor, np.asarray(emissions)))
    # Reproduce existing saved target rows independently of the editor's routine.
    old_matches = []
    for name, model in numeric.items():
        table = pd.read_csv(local / ('new_edits_%s.csv' % name))
        chosen = table[(table.n == 4) & (table.dose == 1.) & table.control.isin(['decrease', 'bounded_increase'])]
        chosen = chosen.iloc[np.linspace(0, len(chosen)-1, 24, dtype=int)]
        with np.load(local / ('new_edit_vectors_%s.npz' % name), allow_pickle=False) as data:
            maximum = 0.
            for row_index, row in chosen.iterrows():
                h = saved['h'][int(row.source_index)]
                actual, detail = editor.edit(h, name, 'top4', 'full', row.control, 'irrelevant-for-target')
                if detail['status'] == 'inactive_or_no_feasible_budget':
                    expected_status = 'no_bidirectional_common_support'
                else:
                    expected_status = detail['status']
                if expected_status != row.status:
                    raise RuntimeError('Saved v3 feasibility does not reproduce')
                expected = data['h_applied'][row_index]
                maximum = max(maximum, float(np.max(abs(actual - expected))))
                np.testing.assert_array_equal(actual, expected)
        old_matches.append({'model': name, 'sampled_target_rows': len(chosen), 'max_abs': maximum, 'exact': True})
    table = pd.DataFrame(diagnostics)
    table.to_csv(root / 'phase_ridge_diagnostic_rows.csv', index=False, encoding='utf-8-sig')
    table.assign(valid=table.status.eq('ok')).groupby(['split', 'phase', 'group', 'dose', 'control']).agg(
        rows=('status', 'size'), valid=('valid', 'sum'), actual_norm=('actual_norm', 'mean'),
        opening_probe_delta_rad=('opening_probe_delta_rad', 'mean'),
        distance_probe_delta_m=('distance_probe_delta_m', 'mean'), any_contact_probe_delta=('any_contact_probe_delta', 'mean'),
        raw_mu_l2=('raw_mu_l2', 'mean')).to_csv(root / 'phase_ridge_diagnostic_summary.csv', encoding='utf-8-sig')
    verification = verify_online_bundle(bundle)
    original_verification = verify_bundle(bundle)
    if not original_verification['passed']:
        raise RuntimeError('Original numeric bundle verification failed')
    if any(digest(source / relative) != value for relative, value in source_hashes.items()):
        raise RuntimeError('Historical bundle changed')
    result = {'passed': True, 'original_local_states': 1365, 'train_local_states': int(train.sum()),
              'new_reference_states': len(selected), 'reference': verification,
              'original_bundle': original_verification, 'inference_calling_convention': inference_audits,
              'saved_v3_target_reproduction': old_matches,
              'phase_diagnostic': '18 or fewer deterministically sampled train/validation states across local, later-post-onset and pre-onset; descriptive frozen-probe readouts, no physical result',
              'historical_bundle_unchanged': True, 'historical_source_hashes': source_hashes,
              'limitations': ['Reference-code replacement not implemented', 'No independent physical generalization established by this preparation',
                              'Batch inference can differ at TopK boundaries; online editor always uses one state at a time']}
    dump(root / 'preparation_verification.json', result)
    code = root / 'code'
    code.mkdir(exist_ok=True)
    for name in ('online_bidirectional.py', 'prepare_online_bidirectional.py', 'physical_runtime.py', 'constrained_intervention.py'):
        shutil.copy2(PROJECT_ROOT / 'temp_scripts' / name, code / name)
    dump(root / 'editor_hashes.json', {str(p.relative_to(root).as_posix()): digest(p) for folder in (bundle, code) for p in folder.rglob('*') if p.is_file()})
    print('READY ' + str(root), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=PROJECT_ROOT / 'outputs/online_development_20261104/package')
    args = parser.parse_args()
    prepare(args.out)
