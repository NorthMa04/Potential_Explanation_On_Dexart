"""Independent mathematical audit of every retained online pilot edit.

Does not call OnlineBidirectionalEditor, NumericSAE, actor_tail, or their editing
helpers. Single-state TopK inference and ray projection are implemented here.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F


H_ATOL = 2e-5
CODE_ATOL = 2e-4
NORM_ATOL = 2e-6
RTOL = 2e-5
STATUS = {'ok': 1, 'inactive_or_no_feasible_budget': 2, 'zero_budget': 3,
          'zero_direction': 4, 'insufficient_feasible_capacity': 5, 'norm_match_failed': 6}


def native_path(path):
    value = str(Path(path).resolve())
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value = '\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value
    return Path(value)


def display(path):
    value = str(path)
    if value.startswith('\\\\?\\UNC\\'):
        return '\\\\' + value[8:]
    return value[4:] if value.startswith('\\\\?\\') else value


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def close(actual, expected, atol):
    return bool(np.allclose(actual, expected, rtol=RTOL, atol=atol))


class IndependentModel:
    def __init__(self, folder):
        self.meta = read_json(folder / 'model.json')
        with np.load(folder / 'weights.npz', allow_pickle=False) as arrays:
            self.w = {k: arrays[k].copy() for k in arrays.files}
        self.encoder_weight = torch.from_numpy(self.w['encoder_weight'])
        self.encoder_bias = torch.from_numpy(self.w['encoder_bias'])
        self.decoder = self.w['decoder_weight'] * self.w['scale'][:, None]

    def encode(self, h):
        # Exactly one row, matching the frozen online calling convention.
        x = ((np.asarray(h, dtype=np.float64) - self.w['mean']) / self.w['scale']).astype(np.float32)
        with torch.no_grad():
            pre = F.linear(torch.from_numpy(x[None]), self.encoder_weight, self.encoder_bias)
            values, indices = torch.topk(pre, int(self.meta['top_k']), dim=-1)
            z = torch.zeros_like(pre)
            z[0, indices[0]] = values[0].clamp(min=0)
        p = pre.numpy()[0].copy()
        order = np.argsort(-p, kind='stable')
        k = int(self.meta['top_k'])
        gap = float(p[order[k - 1]] - p[order[k]])
        # A diagnostic flag only; never changes an audit tolerance.
        threshold = float(8 * np.finfo(np.float32).eps * max(1., np.max(abs(p))))
        boundary = {'gap': gap, 'near_tie': bool(gap <= threshold), 'diagnostic_threshold': threshold,
                    'inside_coordinate': int(order[k - 1]), 'outside_coordinate': int(order[k]),
                    'inside_pre': float(p[order[k - 1]]), 'outside_pre': float(p[order[k]])}
        return z.numpy()[0].copy(), p, boundary


def projected(h, direction):
    return np.clip(h + direction, 0., None) - h


def independent_ray(h, direction, wanted):
    """Solve the monotone squared-norm equation on an unnormalized ray."""
    if wanted <= 1e-10:
        return np.zeros_like(h), 'zero_budget'
    if np.linalg.norm(direction) < 1e-15:
        return np.zeros_like(h), 'zero_direction'
    lo, hi = 0., 1.
    for _ in range(80):
        candidate = projected(h, hi * direction)
        if np.linalg.norm(candidate) >= wanted - 1e-10:
            break
        hi *= 2.
    else:
        return np.zeros_like(h), 'insufficient_feasible_capacity'
    for _ in range(85):
        midpoint = (lo + hi) * .5
        if np.dot(projected(h, midpoint * direction), projected(h, midpoint * direction)) < wanted * wanted:
            lo = midpoint
        else:
            hi = midpoint
    result = projected(h, hi * direction)
    if abs(np.linalg.norm(result) - wanted) > max(1e-9, wanted * 1e-8):
        return np.zeros_like(h), 'norm_match_failed'
    return result, 'ok'


def independent_edit(h, z, model, stats, ridge, metadata, condition, key):
    h = np.asarray(h, dtype=np.float64)
    name, group = condition['model'], condition['group']
    n = int(metadata['groups'][group])
    coordinates = np.asarray(metadata['models'][name]['opening_coordinates'][:n], dtype=int)
    active_std = stats[name + '__active_std'][coordinates]
    upper = stats[name + '__active_q99'][coordinates]
    amount = np.zeros(n, dtype=np.float64)
    for i, coordinate in enumerate(coordinates):
        if z[coordinate] > 0:
            amount[i] = max(0., min(.5 * active_std[i], upper[i] - z[coordinate]))
    down = np.sum(model.decoder[:, coordinates] * (-z[coordinates])[None, :], axis=1)
    up = np.sum(model.decoder[:, coordinates] * amount[None, :], axis=1)
    down_norm, up_norm = float(np.linalg.norm(projected(h, down))), float(np.linalg.norm(projected(h, up)))
    cap = float(metadata['relative_norm_cap'] * np.linalg.norm(h))
    budget = min(down_norm, up_norm, cap) * {'full': 1., 'half': .5}[condition['dose']]
    control = condition['control']
    if control == 'noop':
        delta, status, budget = np.zeros_like(h), 'ok', 0.
    elif budget <= 1e-9:
        delta, status = np.zeros_like(h), 'inactive_or_no_feasible_budget'
    else:
        if control == 'decrease':
            direction = down
        elif control == 'bounded_increase':
            direction = up
        elif control.startswith('random'):
            raw_seed = hashlib.sha256(str(key).encode()).digest()[:8]
            draws = np.random.default_rng(int.from_bytes(raw_seed, 'little')).normal(size=(3, 64))
            directions = draws / np.sqrt(np.sum(draws * draws, axis=1))[:, None]
            direction = directions[int(control[6:])]
        elif control in ('ridge_decrease', 'ridge_increase'):
            direction = np.sum(ridge[:, :n], axis=1) * (-1. if control == 'ridge_decrease' else 1.)
        else:
            raise RuntimeError('Unknown frozen control: ' + control)
        delta, status = independent_ray(h, direction, budget)
    emission = np.clip(h + delta, 0., None).astype(np.float32) if status == 'ok' else h.astype(np.float32)
    return emission, {'status': status, 'budget': budget, 'coordinates': coordinates,
                      'native_down_norm': down_norm, 'native_up_norm': up_norm, 'cap': cap,
                      'cap_active': bool(min(down_norm, up_norm) > cap),
                      'supported_candidates': int(min(down_norm, up_norm, cap) > 1e-9) * 2}


def actor_forward(weights, h):
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(h, dtype=np.float32)[None])
        middle = F.linear(x, weights['mlp_extractor.policy_net.2.weight'], weights['mlp_extractor.policy_net.2.bias']).clamp(min=0)
        mu = F.linear(middle, weights['action_net.weight'], weights['action_net.bias'])
    return middle.numpy()[0], mu.numpy()[0]


def audit(run, destination=None):
    started = time.time()
    run = native_path(run)
    out = native_path(destination) if destination else run / 'analysis_edit_audit'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    package, remote = run / 'package', run / 'remote'
    try:
        hashes = read_json(package / 'protocol_hashes.json')
        frozen_failures = []
        for relative, expected in hashes.items():
            path = (package / relative).resolve()
            if package not in path.parents or not path.is_file() or digest(path) != expected:
                frozen_failures.append(relative)
        if frozen_failures:
            raise RuntimeError('Frozen source hash mismatch: ' + repr(frozen_failures))
        bundle = package / 'bundle'
        metadata = read_json(bundle / 'online_intervention.json')
        metadata_hash = digest(bundle / 'online_intervention.json')
        with np.load(bundle / 'online_statistics.npz', allow_pickle=False) as arrays:
            stats = {k: arrays[k].copy() for k in arrays.files}
        with np.load(bundle / 'combined_directions.npz', allow_pickle=False) as arrays:
            ridge = arrays['ridge_oriented'].copy()
        with np.load(bundle / 'actor.npz', allow_pickle=False) as arrays:
            actor = {k: torch.from_numpy(arrays[k].copy()) for k in arrays.files}
        models = {name: IndependentModel(bundle / 'models' / name) for name in metadata['models']}
        campaign_path = remote / 'pilot/campaign.json'
        campaign = read_json(campaign_path)
        plan_path = remote / 'pilot_plan.json' if (remote / 'pilot_plan.json').is_file() else run / 'pilot_plan.json'
        plan = read_json(plan_path)
        plan_hash = digest(plan_path)
        integrity, errors, rows, actor_rows, diagnostics, assigned = [], [], [], [], [], []
        absent, audited_archives, audited_frames = 0, 0, 0
        for case in campaign['cases']:
            obj, seed = int(case['object']), int(case['seed'])
            case_folder = remote / 'pilot' / ('object%d_seed%d' % (obj, seed))
            names = ['noop_gate'] + ['c%03d_%s_%s_%s_%s_d%d' % (i, c['model'], c['group'], c['control'], c['dose'], c['duration']) for i, c in enumerate(plan['conditions'])]
            latest = {record['name']: record for record in case['runs']}
            for name in names:
                record = latest.get(name)
                assigned.append({'object': obj, 'seed': seed, 'name': name,
                                 'recorded': record is not None, 'exit_code': record.get('exit_code') if record else None})
                if record is None or record.get('exit_code') != 0:
                    absent += 1
            for attempt, record in enumerate(case['runs']):
                folder = case_folder / record['relative_dir']
                archive, report_path = folder / (record['name'] + '.npz'), folder / 'report.json'
                label = {'object': obj, 'seed': seed, 'name': record['name'], 'attempt': attempt}
                if not archive.is_file():
                    if record.get('exit_code') == 0:
                        errors.append({**label, 'error': 'completed archive missing'})
                    continue
                try:
                    report = read_json(report_path)
                    hash_ok = ((not record.get('npz_sha256') or digest(archive) == record['npz_sha256'])
                               and (not record.get('report_sha256') or digest(report_path) == record['report_sha256']))
                    report_ok = (report.get('object') == obj and report.get('seed') == seed
                                 and report.get('source_plan_sha256') == plan_hash
                                 and report.get('online_metadata_sha256') == metadata_hash)
                    integrity.append({**label, 'hash_ok': hash_ok, 'report_identity_ok': report_ok,
                                      'npz_sha256': digest(archive), 'report_sha256': digest(report_path)})
                    condition = record['condition']
                    for field, value in [('model_name', condition['model']), ('group_name', condition['group']),
                                         ('control', condition['control']), ('dose', condition['dose']), ('duration', condition['duration'])]:
                        if report.get(field) != value:
                            raise RuntimeError('Report condition differs from recorded assignment: ' + field)
                    trigger = int(case['trigger_step'])
                    # This is the literal key used in frozen online_physical_simulation.py.
                    key = '%d|%d|%d' % (obj, seed, trigger)
                    model = models[condition['model']]
                    with np.load(archive, allow_pickle=False) as data:
                        length = len(data['step'])
                        if not all(np.isfinite(data[k]).all() for k in data.files):
                            raise RuntimeError('Nonfinite recorded evidence')
                        schedule = (np.arange(length) >= trigger)
                        if condition['duration'] != -1:
                            schedule &= np.arange(length) < trigger + condition['duration']
                        if not np.array_equal(schedule, data['scheduled']):
                            raise RuntimeError('Assigned schedule does not reproduce')
                        if not np.array_equal(data['step'], np.arange(length)) or not np.array_equal(data['done'], np.arange(length) == length - 1):
                            raise RuntimeError('Incomplete/noncontiguous rollout')
                        actor_max = {'h2': 0., 'mu': 0., 'action': 0.}
                        actor_bad = 0
                        per_actor_ok = []
                        for t in range(length):
                            h2, mu = actor_forward(actor, data['h_applied'][t])
                            differences = {'h2': float(np.max(abs(h2 - data['h2'][t]))),
                                           'mu': float(np.max(abs(mu - data['mu'][t]))),
                                           'action': float(np.max(abs(np.clip(mu, -1, 1) - data['action'][t])))}
                            ok = close(h2, data['h2'][t], H_ATOL) and close(mu, data['mu'][t], H_ATOL) and close(np.clip(mu, -1, 1), data['action'][t], H_ATOL)
                            per_actor_ok.append(ok)
                            actor_bad += int(not ok)
                            for field, value in differences.items():
                                actor_max[field] = max(actor_max[field], value)
                        actor_rows.append({**label, 'frames': length, 'failed_frames': actor_bad,
                                           **{'max_' + k + '_abs': v for k, v in actor_max.items()}})
                        audited_frames += length
                        for t in np.flatnonzero(schedule):
                            h = data['h_original'][t].astype(np.float64)
                            actual_h = data['h_applied'][t]
                            before, before_pre, before_edge = model.encode(h)
                            after, after_pre, after_edge = model.encode(actual_h)
                            expected_h, detail = independent_edit(h, before, model, stats, ridge, metadata, condition, key)
                            indices = detail['coordinates']
                            other = np.ones(64, dtype=bool); other[indices] = False
                            actual_norm = float(np.linalg.norm(actual_h.astype(float) - h))
                            before_ok = close(before, data['edit_z_before'][t], CODE_ATOL)
                            after_ok = close(after, data['edit_z_after'][t], CODE_ATOL)
                            h_ok = close(expected_h, actual_h, H_ATOL)
                            budget_ok = abs(detail['budget'] - float(data['edit_budget'][t])) <= NORM_ATOL
                            norm_ok = (abs(actual_norm - float(data['edit_actual_norm'][t])) <= NORM_ATOL
                                       and abs(abs(actual_norm - detail['budget']) - float(data['edit_norm_error'][t])) <= NORM_ATOL
                                       and actual_norm <= detail['budget'] + NORM_ATOL
                                       and (detail['status'] != 'ok' or abs(actual_norm - detail['budget']) <= NORM_ATOL))
                            status_ok = (int(data['edit_status_code'][t]) == STATUS[detail['status']]
                                         and bool(data['applied'][t]) == (detail['status'] == 'ok'))
                            quantities = {'target_before': float(np.linalg.norm(before[indices])),
                                          'target_after': float(np.linalg.norm(after[indices])),
                                          'nontarget_code_l2': float(np.linalg.norm((after - before)[other]))}
                            quantity_ok = all(close(expected, data['edit_' + field][t], CODE_ATOL) for field, expected in quantities.items())
                            cap_ok = (bool(data['edit_cap_active'][t]) == detail['cap_active']
                                      and int(data['edit_common_supported_candidates'][t]) == detail['supported_candidates'])
                            row = {**label, **condition, 'step': int(t), 'logged_status': int(data['edit_status_code'][t]),
                                   'reconstructed_status': detail['status'], 'status_ok': status_ok,
                                   'h_ok': h_ok, 'h_max_abs': float(np.max(abs(expected_h - actual_h))),
                                   'z_before_ok': before_ok, 'z_after_ok': after_ok,
                                   'z_before_max_abs': float(np.max(abs(before - data['edit_z_before'][t]))),
                                   'z_after_max_abs': float(np.max(abs(after - data['edit_z_after'][t]))),
                                   'budget_ok': budget_ok, 'logged_budget': float(data['edit_budget'][t]), 'independent_budget': detail['budget'],
                                   'actual_norm': actual_norm, 'norm_ok': norm_ok,
                                   'target_and_other_quantities_ok': quantity_ok, 'cap_support_ok': cap_ok, 'actor_ok': per_actor_ok[t],
                                   'native_down_norm': detail['native_down_norm'], 'native_up_norm': detail['native_up_norm'],
                                   'before_boundary_gap': before_edge['gap'], 'after_boundary_gap': after_edge['gap'],
                                   'before_near_tie': before_edge['near_tie'], 'after_near_tie': after_edge['near_tie']}
                            row['passed'] = all((h_ok, before_ok, after_ok, budget_ok, norm_ok, status_ok, quantity_ok, cap_ok, per_actor_ok[t]))
                            rows.append(row)
                            if not row['passed'] or before_edge['near_tie'] or after_edge['near_tie']:
                                diagnostic = {**label, **condition, 'step': int(t), 'passed': row['passed'],
                                              'actual_h_matches_independent': h_ok, 'before_boundary': before_edge, 'after_boundary': after_edge,
                                              'pre_before': before_pre.tolist(), 'pre_after_actual_h': after_pre.tolist(),
                                              'independent_z_before': before.tolist(), 'logged_z_before': data['edit_z_before'][t].tolist(),
                                              'independent_z_after': after.tolist(), 'logged_z_after': data['edit_z_after'][t].tolist()}
                                if not before_ok:
                                    alternate_h, alternate = independent_edit(h, data['edit_z_before'][t], model, stats, ridge, metadata, condition, key)
                                    diagnostic['h_reconstructed_from_logged_before_code_max_abs'] = float(np.max(abs(alternate_h - actual_h)))
                                    diagnostic['budget_from_logged_before_code'] = alternate['budget']
                                diagnostics.append(diagnostic)
                        audited_archives += 1
                    print('AUDITED %d %d %s' % (obj, seed, record['name']), flush=True)
                except Exception as error:
                    errors.append({**label, 'error': repr(error), 'archive': display(archive)})
        csv_write(out / 'scheduled_step_audit.csv', rows)
        csv_write(out / 'actor_all_frame_audit.csv', actor_rows)
        csv_write(out / 'evidence_integrity.csv', integrity)
        csv_write(out / 'assigned_condition_coverage.csv', assigned)
        dump(out / 'topk_and_failure_diagnostics.json', diagnostics)
        dump(out / 'archive_errors.json', errors)
        code_failed = sum(not (r['z_before_ok'] and r['z_after_ok']) for r in rows)
        h_failed = sum(not r['h_ok'] for r in rows)
        result = {
            'passed': bool(campaign.get('status') == 'completed' and not absent and not errors
                           and all(r['passed'] for r in rows) and all(r['failed_frames'] == 0 for r in actor_rows)
                           and all(r['hash_ok'] and r['report_identity_ok'] for r in integrity)),
            'campaign_status': campaign.get('status'), 'physical_origins': len(campaign['cases']),
            'assigned_conditions': len(assigned), 'uncompleted_assigned_conditions': absent,
            'audited_archives': audited_archives, 'audited_actor_frames': audited_frames,
            'scheduled_steps': len(rows), 'failed_scheduled_steps': sum(not r['passed'] for r in rows),
            'code_audit_failed_steps': code_failed, 'actual_h_audit_failed_steps': h_failed,
            'code_failure_with_matching_h_steps': sum(not (r['z_before_ok'] and r['z_after_ok']) and r['h_ok'] for r in rows),
            'near_tie_before_steps': sum(r['before_near_tie'] for r in rows),
            'near_tie_after_steps': sum(r['after_near_tie'] for r in rows),
            'actor_failed_frames': sum(r['failed_frames'] for r in actor_rows), 'archive_errors': len(errors),
            'max_abs': {field: max((r[field] for r in rows), default=0.) for field in ('h_max_abs', 'z_before_max_abs', 'z_after_max_abs')},
            'tolerances': {'h_actor_atol': H_ATOL, 'code_atol': CODE_ATOL, 'rtol': RTOL, 'norm_budget_atol': NORM_ATOL,
                           'origin': 'original frozen numeric h/code tolerances retained; scalar actual norm tolerance is 2e-6; near ties never waive failures'},
            'independence': 'explicit one-state F.linear and TopK; independent raw-ray bisection, bound construction, SHA256 RNG and actor forward; no editor/runtime helper calls',
            'historical_topk_anomaly': 'not relabeled by this new audit; code and actual h failures remain separately reported',
            'source_hashes': {'campaign': digest(campaign_path), 'pilot_plan': plan_hash,
                              'package_protocol': digest(package / 'protocol_hashes.json'),
                              'online_metadata': metadata_hash, 'audit_implementation': digest(native_path(__file__))},
            'elapsed_seconds': time.time() - started,
        }
        dump(out / 'verification.json', result)
        return result
    except Exception as error:
        result = {'passed': False, 'fatal_error': repr(error), 'elapsed_seconds': time.time() - started}
        dump(out / 'verification.json', result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
