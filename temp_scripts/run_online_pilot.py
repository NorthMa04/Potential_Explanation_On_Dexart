"""Resumable independent-process bidirectional pilot (Python 3.8 compatible)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def audit_archive(path, trigger_opening_rad, condition=None, trigger_step=None,
                  require_h2=False, expected_obs_shapes=None):
    """Check complete trajectories and independently reconstruct success."""
    with np.load(path, allow_pickle=False) as arrays:
        for key in ('step', 'opening', 'done', 'post_progress', 'post_distance',
                    'post_strict_contact', 'post_contacts', 'post_success',
                    'h_original', 'h_applied', 'mu', 'action'):
            if key not in arrays.files:
                raise RuntimeError('Missing recorded evidence: ' + key)
        length = len(arrays['step'])
        if not length or not bool(arrays['done'][-1]):
            raise RuntimeError('Partial or empty rollout')
        if not np.array_equal(arrays['step'], np.arange(length)):
            raise RuntimeError('Recorded control steps are not contiguous from zero')
        if not np.array_equal(arrays['done'].astype(bool), np.arange(length) == length - 1):
            raise RuntimeError('done must be true only at the last recorded step')
        if 'post_step' in arrays.files and not np.array_equal(arrays['post_step'], np.arange(length) + 1):
            raise RuntimeError('Recorded post-action steps are not contiguous')
        for key in arrays.files:
            if len(arrays[key]) != length or not np.isfinite(arrays[key]).all():
                raise RuntimeError('Invalid evidence array: ' + key)
        obs_keys = sorted(key for key in arrays.files if key.startswith('obs__'))
        required_obs = {'obs__instance_1-seg_gt', 'obs__instance_1-point_cloud',
                        'obs__imagination_robot', 'obs__state', 'obs__oracle_state'}
        if not required_obs.issubset(obs_keys):
            raise RuntimeError('Incomplete full noisy policy observations')
        obs_shapes = {key: list(arrays[key].shape[1:]) for key in obs_keys}
        if expected_obs_shapes is not None and obs_shapes != expected_obs_shapes:
            raise RuntimeError('Observation keys or per-frame shapes differ from the natural baseline')
        for key in ('h_original', 'h_applied'):
            if arrays[key].shape != (length, 64):
                raise RuntimeError('Invalid hidden activation shape: ' + key)
        for key in ('mu', 'action'):
            if arrays[key].shape != (length, 22):
                raise RuntimeError('Invalid actor output shape: ' + key)
        if require_h2 and ('h2' not in arrays.files or arrays['h2'].shape != (length, 64)):
            raise RuntimeError('New physical branches must record finite 64-dimensional h2')
        strict = ((arrays['post_contacts'][:, :-1].sum(axis=1) >= 2)
                  & arrays['post_contacts'][:, -1].astype(bool))
        success = ((arrays['post_progress'] > .95) & (arrays['post_distance'] <= .2) & strict)
        if not np.array_equal(strict, arrays['post_strict_contact']):
            raise RuntimeError('Strict contact reconstruction mismatch')
        if not np.array_equal(success, arrays['post_success']):
            raise RuntimeError('Composite success reconstruction mismatch')
        if (arrays['h_original'] < 0).any() or (arrays['h_applied'] < 0).any():
            raise RuntimeError('Negative h evidence')
        actual = np.linalg.norm(arrays['h_applied'].astype(float) - arrays['h_original'].astype(float), axis=1)
        if 'delta_norm' in arrays.files:
            np.testing.assert_allclose(actual, arrays['delta_norm'], rtol=0, atol=2e-6)
        if np.any(actual > .1 * np.linalg.norm(arrays['h_original'], axis=1) + 2e-6):
            raise RuntimeError('Actual h norm exceeded 10 percent cap')
        hits = np.flatnonzero(arrays['opening'] >= trigger_opening_rad)
        scheduled = arrays['scheduled'].astype(bool) if 'scheduled' in arrays.files else np.zeros(length, dtype=bool)
        applied = arrays['applied'].astype(bool) if 'applied' in arrays.files else np.zeros(length, dtype=bool)
        if np.any(arrays['h_applied'][~scheduled] != arrays['h_original'][~scheduled]):
            raise RuntimeError('Hidden activation changed outside the assigned schedule')
        if condition is not None:
            if trigger_step is None:
                raise RuntimeError('A treatment requires a frozen natural trigger')
            expected_schedule = np.arange(length) >= trigger_step
            if condition['duration'] != -1:
                expected_schedule &= np.arange(length) < trigger_step + condition['duration']
            if not np.array_equal(scheduled, expected_schedule):
                raise RuntimeError('Treatment schedule differs from trigger and assigned duration')
        edit_fields = ('edit_status_code', 'edit_budget', 'edit_actual_norm', 'edit_norm_error')
        if any(key in arrays.files for key in edit_fields):
            if not all(key in arrays.files for key in edit_fields) or 'scheduled' not in arrays.files or 'applied' not in arrays.files:
                raise RuntimeError('Incomplete editing audit metadata')
            status = arrays['edit_status_code']
            if not np.isin(status, np.arange(7)).all():
                raise RuntimeError('Illegal edit status code')
            if np.any(status[~scheduled] != 0) or np.any(status[scheduled] == 0):
                raise RuntimeError('Editing status does not match assigned schedule')
            if not np.array_equal(applied, scheduled & (status == 1)):
                raise RuntimeError('Applied flags do not match successful editing status')
            budget = arrays['edit_budget']
            if (budget < 0).any():
                raise RuntimeError('Negative requested norm budget')
            np.testing.assert_allclose(arrays['edit_actual_norm'], actual, rtol=0, atol=2e-6,
                                       err_msg='Recorded edit norm differs from actual h change')
            np.testing.assert_allclose(arrays['edit_norm_error'], np.abs(actual - budget), rtol=0, atol=2e-6,
                                       err_msg='Recorded norm error differs from actual budget error')
            if np.any(actual > budget + 2e-6):
                raise RuntimeError('Actual hidden change exceeded assigned budget')
            if not np.allclose(actual[status == 1], budget[status == 1], rtol=0, atol=2e-6):
                raise RuntimeError('Successful edits do not achieve the assigned norm budget')
        elif condition is not None:
            raise RuntimeError('A treatment branch did not record editing metadata')
        goal = arrays['post_progress'] > .95
        near = arrays['post_distance'] <= .2
        return {'passed': True, 'steps': length, 'success': bool(success.any()),
                'trigger_step': int(hits[0]) if len(hits) else None,
                'scheduled_steps': int(scheduled.sum()), 'applied_steps': int(applied.sum()),
                'ineffective_scheduled_steps': int((scheduled & ~applied).sum()),
                'applied_zero_norm_steps': int((applied & (actual <= 1e-9)).sum()),
                'cumulative_actual_h_norm': float(actual.sum()), 'max_actual_h_norm': float(actual.max()),
                'full_observation_shapes': obs_shapes, 'h2_recorded': 'h2' in arrays.files,
                'goal_and_near_steps': int((goal & near).sum()),
                'goal_and_near_without_palm_steps': int((goal & near & ~arrays['post_contacts'][:, -1].astype(bool)).sum()),
                'goal_and_near_without_two_finger_groups_steps': int((goal & near & (arrays['post_contacts'][:, :-1].sum(axis=1) < 2)).sum()),
                'max_opening_rad': float(arrays['post_opening'].max()),
                'final_opening_rad': float(arrays['post_opening'][-1]),
                'first_success_step': int(np.flatnonzero(success)[0]) if success.any() else None,
                'success_predicate': 'post_progress>.95 AND post_distance<=.2 AND at least two finger groups plus palm'}


def verify_noop(reference, archive, atol):
    errors, excluded = {}, []
    with np.load(reference, allow_pickle=False) as natural, np.load(archive, allow_pickle=False) as noop:
        if 'h2' not in noop.files or noop['h2'].shape != (len(noop['step']), 64) or not np.isfinite(noop['h2']).all():
            raise RuntimeError('Noop must record finite 64-dimensional h2, even when natural lacks it')
        common = sorted(set(natural.files) & set(noop.files))
        for key in common:
            if key in ('applied', 'scheduled') or key.startswith('edit_'):
                excluded.append(key)
                continue
            if natural[key].shape != noop[key].shape:
                raise RuntimeError('Noop full rollout shape mismatch: ' + key)
            a, b = natural[key], noop[key]
            errors[key] = float(np.max(np.abs(a.astype(float) - b.astype(float))))
            if a.dtype.kind == 'b' or b.dtype.kind == 'b':
                if not np.array_equal(a, b):
                    raise RuntimeError('Noop full rollout Boolean mismatch: ' + key)
            else:
                np.testing.assert_allclose(a, b, rtol=0, atol=atol,
                                           err_msg='Noop full rollout mismatch: ' + key)
        required = {'opening', 'post_opening', 'h_original', 'h_applied', 'mu', 'action', 'reward', 'done', 'post_success'}
        if not required.issubset(errors):
            raise RuntimeError('Noop gate is missing a required common array')
    return {'passed': True, 'atol': atol, 'rtol': 0, 'compared_arrays': errors,
            'excluded_metadata_arrays': excluded,
            'exclusion_reason': 'scheduled/applied/edit_* describe assignment and editor bookkeeping, not physical or policy equality',
            'reference_sha256': digest(reference), 'noop_npz_sha256': digest(archive)}


def resolve_reference(root, selection_file, item):
    if item.get('natural_relative_path'):
        candidate = (selection_file.parent / item['natural_relative_path']).resolve()
        if candidate.is_file():
            return candidate
    for field in ('natural_npz', 'natural_path', 'reference', 'reference_npz', 'baseline_npz'):
        if item.get(field):
            path = Path(item[field])
            if path.is_absolute():
                return path.resolve()
            for base in (root, selection_file.parent):
                candidate = (base / path).resolve()
                if candidate.is_file():
                    return candidate
    obj, seed = int(item['object']), int(item.get('seed', item.get('environment_seed')))
    for candidate in (root / 'screen' / ('object%d_seed%d' % (obj, seed)) / 'natural' / 'natural.npz',
                      root / 'screen' / 'campaign' / ('object%d_seed%d' % (obj, seed)) / 'natural' / 'natural.npz'):
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError('Cannot resolve natural NPZ for %d/%d' % (obj, seed))


def write_analysis(output, summary):
    rows = []
    for case in summary['cases']:
        latest = {}
        for run in case['runs']:
            latest[run['name']] = run
        for run in latest.values():
            audit = run.get('archive_audit', {})
            success = audit.get('success')
            baseline = case.get('baseline_audit', {}).get('success')
            rows.append({'object': case['object'], 'environment_seed': case['seed'], 'case_status': case['status'],
                         'name': run['name'], **run.get('condition', {}), 'exit_code': run['exit_code'],
                         'baseline_success': baseline, 'success': success,
                         'flipped': (success != baseline) if success is not None and baseline is not None else None,
                         'success_to_failure': bool(baseline and not success) if success is not None and baseline is not None else None,
                         'failure_to_success': bool(not baseline and success) if success is not None and baseline is not None else None,
                         'natural_trigger_step': case.get('trigger_step'),
                         **{k: v for k, v in audit.items() if k not in ('passed', 'success', 'success_predicate', 'full_observation_shapes')},
                         'npz_sha256': run.get('npz_sha256'), 'relative_dir': run.get('relative_dir')})
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    pending = output / 'analysis.pending.csv'
    with pending.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(output / 'analysis.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true', help='Accepted for explicit resume; matching existing campaigns also resume automatically')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('workers must be between one and four')
    root = args.root.resolve()
    package = root / 'package'
    plan_path = root / 'pilot_plan.json'
    selection_path = root / 'screen/selected_origins.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    selected_data = json.loads(selection_path.read_text(encoding='utf-8'))
    if isinstance(selected_data, list):
        selected = selected_data
    else:
        selected = next((selected_data[k] for k in ('selected_origins', 'selected', 'origins', 'cases') if isinstance(selected_data.get(k), list)), None)
    if selected is None:
        raise RuntimeError('Selection JSON needs a list of selected origins')
    conditions = plan['conditions']
    for condition in conditions:
        if not {'model', 'group', 'control', 'dose', 'duration'}.issubset(condition):
            raise RuntimeError('Incomplete explicit pilot condition')
        if condition['duration'] not in (1, 5, -1):
            raise RuntimeError('Unexpected duration')
    import fcntl
    lock_file = (root / 'pilot.lock').open('a')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Online pilot is already running')
    hashes_path = package / 'protocol_hashes.json'
    hashes = json.loads(hashes_path.read_text(encoding='utf-8'))
    for relative, expected in hashes.items():
        path = (package / relative).resolve()
        if package not in path.parents or not path.is_file() or digest(path) != expected:
            raise RuntimeError('Frozen package mismatch: ' + relative)
    code = package / 'code'
    sys.path.insert(0, str(code))
    import torch
    torch.set_num_threads(1)
    from online_bidirectional import verify_online_bundle
    verification = verify_online_bundle(package / 'bundle')
    write_json(root / 'pilot_cross_runtime_verification.json', verification)
    output = root / 'pilot'
    output.mkdir(exist_ok=True)
    progress = output / 'campaign.json'
    identity = {'protocol_sha256': digest(hashes_path), 'plan_sha256': digest(plan_path),
                'selection_sha256': digest(selection_path)}
    metadata_sha256 = digest(package / 'bundle/online_intervention.json')
    if progress.exists():
        summary = json.loads(progress.read_text(encoding='utf-8'))
        if any(summary.get(key) != value for key, value in identity.items()):
            raise RuntimeError('Resume protocol, plan, or selected origin list changed')
        summary['status'] = 'running'
    else:
        cases = []
        seen = set()
        for item in selected:
            obj, seed = int(item['object']), int(item.get('seed', item.get('environment_seed')))
            if (obj, seed) in seen:
                raise RuntimeError('Duplicate selected physical origin')
            seen.add((obj, seed))
            reference = resolve_reference(root, selection_path, item)
            if item.get('natural_npz_sha256') and digest(reference) != item['natural_npz_sha256']:
                raise RuntimeError('Selected natural NPZ hash mismatch')
            cases.append({'object': obj, 'seed': seed, 'status': 'pending', 'runs': [],
                          'reference': str(reference), 'reference_sha256': digest(reference), 'selection': item})
        summary = {'status': 'running', 'started_unix': time.time(), 'cases': cases, **identity,
                   'statistics': 'All selected origins retained; no filtering on complete treatment exposure; natural baseline outcome strata',
                   'workers': args.workers}
    mutex = threading.Lock()

    def flush():
        write_json(progress, summary)

    def one_case(item):
        case = output / ('object%d_seed%d' % (item['object'], item['seed']))
        case.mkdir(exist_ok=True)
        reference = Path(item['reference'])

        def run(name, condition):
            if shutil.disk_usage(root).free < 4 * 1024 ** 3:
                raise RuntimeError('Fewer than 4 GiB free; evidence retained')
            previous = [record for record in item['runs'] if record['name'] == name]
            if previous:
                last = previous[-1]
                archive = case / last['relative_dir'] / (name + '.npz')
                report = archive.with_name('report.json')
                if last['exit_code'] == 0:
                    if not archive.is_file() or not report.is_file() or digest(archive) != last['npz_sha256'] or digest(report) != last['report_sha256']:
                        raise RuntimeError('Completed evidence changed: ' + name)
                    return last
            relative = name if not previous else name + '__retry%d' % len(previous)
            target = case / relative
            if target.exists():
                relative += '__%d' % time.time_ns()
                target = case / relative
            command = [sys.executable, '-B', '-u', str(code / 'online_physical_simulation.py'),
                       '--bundle', str(package / 'bundle'), '--plan', str(plan_path), '--repo', plan['repo'],
                       '--out', str(target), '--object', str(item['object']), '--seed', str(item['seed']),
                       '--single-condition', name, '--reference', str(reference), '--group-edit',
                       '--model-name', condition['model'], '--group-name', condition['group'],
                       '--control', condition['control'], '--dose', condition['dose'], '--duration', str(condition['duration'])]
            started = time.time()
            with (case / (relative + '.log')).open('w') as log:
                try:
                    exit_code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                               timeout=plan.get('child_timeout_seconds', 300)).returncode
                except subprocess.TimeoutExpired:
                    exit_code = -999
            record = {'name': name, 'relative_dir': relative, 'condition': condition,
                      'exit_code': exit_code, 'elapsed_seconds': time.time() - started}
            report = target / 'report.json'
            archive = target / (name + '.npz')
            if report.is_file():
                record['report'] = json.loads(report.read_text(encoding='utf-8'))
            if exit_code == 0:
                try:
                    evidence = record.get('report', {})
                    if not archive.is_file() or evidence.get('status') != 'fresh_process_rollout_completed':
                        raise RuntimeError('Missing complete rollout evidence')
                    for flag in ('prefix_gate_passed', 'actor_parameters_unchanged', 'full_policy_state_unchanged'):
                        if evidence.get(flag) is not True:
                            raise RuntimeError('Failed physical evidence flag: ' + flag)
                    if evidence.get('reference_sha256') != item['reference_sha256']:
                        raise RuntimeError('Child reference does not match frozen baseline')
                    for field in ('checkpoint_sha256', 'full_policy_state_sha256'):
                        if evidence.get(field) != item['baseline_policy_hashes'][field]:
                            raise RuntimeError('Child policy differs from natural baseline: ' + field)
                    if evidence.get('source_plan_sha256') != identity['plan_sha256']:
                        raise RuntimeError('Child source plan hash mismatch')
                    if evidence.get('online_metadata_sha256') != metadata_sha256:
                        raise RuntimeError('Child online metadata hash mismatch')
                    expected_fields = {'object': item['object'], 'seed': item['seed'],
                                       'model_name': condition['model'], 'group_name': condition['group'],
                                       'control': condition['control'], 'dose': condition['dose'],
                                       'duration': condition['duration'], 'group_edit': True}
                    for field, expected in expected_fields.items():
                        if evidence.get(field) != expected:
                            raise RuntimeError('Child report assignment mismatch: ' + field)
                    record['archive_audit'] = audit_archive(
                        archive, plan['trigger_opening_rad'], condition=condition,
                        trigger_step=item['trigger_step'], require_h2=True,
                        expected_obs_shapes=item['baseline_audit']['full_observation_shapes'])
                    if record['archive_audit']['success'] != evidence['summary']['success']:
                        raise RuntimeError('Report success disagrees with independent NPZ audit')
                    record.update(npz_sha256=digest(archive), report_sha256=digest(report))
                except Exception as error:
                    record.update(exit_code=-998, audit_error=repr(error))
            with mutex:
                item['runs'].append(record)
                flush()
                print(json.dumps({'object': item['object'], 'seed': item['seed'], 'name': name,
                                  'exit_code': record['exit_code'], 'audit': record.get('archive_audit')}), flush=True)
            return record

        try:
            with mutex:
                item['status'] = 'running'
                flush()
            if digest(reference) != item['reference_sha256']:
                raise RuntimeError('Natural baseline changed after selection')
            baseline_report_path = reference.with_name('report.json')
            baseline_report_hash = digest(baseline_report_path)
            if item.get('baseline_report_sha256') and item['baseline_report_sha256'] != baseline_report_hash:
                raise RuntimeError('Natural baseline report changed after the first pilot attempt')
            baseline_report = json.loads(baseline_report_path.read_text(encoding='utf-8'))
            if baseline_report.get('object') != item['object'] or baseline_report.get('seed') != item['seed']:
                raise RuntimeError('Natural baseline report object/seed mismatch')
            if baseline_report.get('status') != 'fresh_process_rollout_completed':
                raise RuntimeError('Natural baseline report does not describe a complete rollout')
            if not all(baseline_report.get(flag) is True for flag in ('actor_parameters_unchanged', 'full_policy_state_unchanged')):
                raise RuntimeError('Natural baseline policy parameter invariance was not verified')
            policy_hashes = {field: baseline_report.get(field) for field in ('checkpoint_sha256', 'full_policy_state_sha256')}
            if not all(isinstance(value, str) and len(value) == 64 for value in policy_hashes.values()):
                raise RuntimeError('Natural baseline report is missing policy hashes')
            item['baseline_report_sha256'] = baseline_report_hash
            item['baseline_policy_hashes'] = policy_hashes
            item['baseline_audit'] = audit_archive(reference, plan['trigger_opening_rad'])
            if item['baseline_audit']['success'] != baseline_report['summary']['success']:
                raise RuntimeError('Natural baseline report success disagrees with the independent NPZ audit')
            item['trigger_step'] = item['baseline_audit']['trigger_step']
            if item['trigger_step'] is None:
                item['status'] = 'no_trigger_all_outcomes_retained'
                return
            noop = {'model': 'assisted_seed0', 'group': 'top4', 'control': 'noop', 'dose': 'full', 'duration': 5}
            gate = run('noop_gate', noop)
            if gate['exit_code']:
                raise RuntimeError('Noop rollout execution failed')
            gate_result = verify_noop(reference, case / gate['relative_dir'] / 'noop_gate.npz', plan['restore_repeat_atol'])
            write_json(case / 'noop_full_rollout_gate.json', gate_result)
            item['noop_full_rollout_gate_passed'] = True
            for index, condition in enumerate(conditions):
                name = 'c%03d_%s_%s_%s_%s_d%d' % (index, condition['model'], condition['group'], condition['control'], condition['dose'], condition['duration'])
                record = run(name, condition)
                if record['exit_code']:
                    raise RuntimeError('Condition failed; this origin stopped: ' + name)
            item['status'] = 'completed'
            item.pop('exception', None)
        except Exception as error:
            item.update(status='failed', exception=repr(error))
        finally:
            with mutex:
                flush()

    flush()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one_case, summary['cases']))
    summary['status'] = 'completed' if all(item['status'] in ('completed', 'no_trigger_all_outcomes_retained') for item in summary['cases']) else 'completed_with_failures'
    summary['elapsed_seconds'] = time.time() - summary['started_unix']
    summary['completed_conditions'] = sum(sum(record['exit_code'] == 0 for record in item['runs']) for item in summary['cases'])
    flush()
    write_analysis(output, summary)
    if summary['status'] != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
