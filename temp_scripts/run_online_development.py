"""Resumable fresh-process natural/repeat screening for online development.

Python 3.8 compatible. The frozen source package is root/screen_source;
root/screen_plan.json contains the new screening population and child settings.
No intervention pilot is implemented by this entry point.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(rows[0]) if rows else ['object', 'seed', 'status']
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class CampaignLock:
    def __init__(self, path):
        self.stream = path.open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.stream.seek(0)
                self.stream.write(b'0')
                self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):
            self.stream.close()
            raise RuntimeError('This screening campaign is already locked/running')

    def close(self):
        self.stream.close()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def audit_rollout(archive, report, item, plan, expected_checkpoint, plan_hash):
    if report.get('status') != 'fresh_process_rollout_completed':
        raise RuntimeError('Child did not report completed fresh-process rollout')
    if report.get('object') != item['object'] or report.get('seed') != item['seed']:
        raise RuntimeError('Report object/seed mismatch')
    if not report.get('actor_parameters_unchanged') or not report.get('full_policy_state_unchanged'):
        raise RuntimeError('Unchanged policy parameter checks missing/failed')
    if report.get('checkpoint_sha256') != expected_checkpoint or report.get('source_plan_sha256') != plan_hash:
        raise RuntimeError('Child checkpoint/plan hash mismatch')
    if report.get('model_name') is not None or report.get('duration') != 0:
        raise RuntimeError('Screening child unexpectedly used a treatment')
    if report.get('feature_edit') or report.get('group_edit'):
        raise RuntimeError('Screening child unexpectedly enabled an editor')
    if not report.get('full_policy_state_sha256'):
        raise RuntimeError('Full frozen policy hash missing')
    arrays = load_arrays(archive)
    required = ['step', 'post_step', 'opening', 'distance', 'contact', 'strict_contact', 'contacts',
                'success', 'progress', 'post_opening', 'post_distance', 'post_contact',
                'post_strict_contact', 'post_contacts', 'post_success', 'post_progress',
                'h_original', 'h_applied', 'mu', 'action', 'done', 'applied', 'observation_state',
                'robot_qpos', 'robot_qvel', 'object_qpos', 'object_qvel',
                'post_robot_qpos', 'post_robot_qvel', 'post_object_qpos', 'post_object_qvel']
    missing = set(required) - set(arrays)
    if missing or not any(key.startswith('obs__') for key in arrays):
        raise RuntimeError('Incomplete trajectory arrays: ' + repr(sorted(missing)))
    steps = len(arrays['step'])
    if not steps or not bool(arrays['done'][-1]) or np.any(arrays['done'][:-1]):
        raise RuntimeError('Incomplete/incorrectly terminated full trajectory')
    for key, value in arrays.items():
        if len(value) != steps or not np.isfinite(value).all():
            raise RuntimeError('Nonfinite or inconsistent trajectory length: ' + key)
    if not np.array_equal(arrays['step'], np.arange(steps)):
        raise RuntimeError('Trajectory does not start at step zero or has missing steps')
    if not np.array_equal(arrays['post_step'], arrays['step'] + 1):
        raise RuntimeError('Action-before/action-after step alignment failed')
    if arrays['applied'].any() or not np.array_equal(arrays['h_original'], arrays['h_applied']):
        raise RuntimeError('Natural/repeat screening changed h')
    errors = {}
    for prefix in ('', 'post_'):
        contacts = arrays[prefix + 'contacts']
        if contacts.ndim != 2 or contacts.shape[1] != 5 or not np.isin(contacts, [0, 1]).all():
            raise RuntimeError('Unexpected task contact layout')
        strict = (contacts[:, :-1].sum(axis=1) >= 2) & contacts[:, -1].astype(bool)
        predicted = ((arrays[prefix + 'progress'] > .95)
                     & (arrays[prefix + 'distance'] <= .2) & strict)
        errors[prefix + 'strict_mismatches'] = int(np.count_nonzero(strict != arrays[prefix + 'strict_contact']))
        errors[prefix + 'success_mismatches'] = int(np.count_nonzero(predicted != arrays[prefix + 'success']))
    if any(errors.values()):
        raise RuntimeError('Independent task success/contact predicate mismatch: ' + repr(errors))
    success = bool(arrays['post_success'].any())
    child_summary = report.get('summary', {})
    if child_summary.get('steps') != steps or bool(child_summary.get('success')) != success:
        raise RuntimeError('Child summary disagrees with independent full-trajectory endpoint')
    trigger = first(arrays['opening'] >= plan['trigger_opening_rad'])
    goal = arrays['post_progress'] > .95
    near = arrays['post_distance'] <= .2
    strict = arrays['post_strict_contact'].astype(bool)
    goal_near = goal & near
    contacts = arrays['post_contacts']
    fingers = contacts[:, :-1].sum(axis=1)
    palm = contacts[:, -1].astype(bool)
    if success:
        classification = 'success'
    elif trigger is None:
        classification = 'no_onset_failure'
    elif not goal.any():
        classification = 'post_onset_failure_progress_not_reached'
    elif not goal_near.any():
        classification = 'post_onset_failure_goal_without_near'
    else:
        signatures = set()
        for t in np.flatnonzero(goal_near):
            signatures.add('palm_only' if fingers[t] >= 2 and not palm[t]
                           else 'fingers_only' if fingers[t] < 2 and palm[t] else 'palm_and_fingers')
        classification = 'post_onset_failure_goal_near_without_strict_' + '|'.join(sorted(signatures))
    result = dict(status='passed', steps=steps, success=success, trigger_step=trigger,
                  has_trigger=trigger is not None, post_onset_failure=(not success and trigger is not None),
                  mechanism_class=classification, success_step=first(arrays['post_success']),
                  progress_goal_reached=bool(goal.any()), goal_and_near=bool(goal_near.any()),
                  goal_and_strict=bool((goal & strict).any()), strict_contact_anywhere=bool(strict.any()),
                  goal_near_steps=int(goal_near.sum()), first_goal_near_step=first(goal_near),
                  final_opening_rad=float(arrays['post_opening'][-1]),
                  max_opening_rad=float(arrays['post_opening'].max()),
                  predicate_errors=errors, parameters_unchanged=True,
                  full_policy_state_sha256=report['full_policy_state_sha256'])
    return result, arrays


def audit_repeat(natural, repeat, tolerance):
    # Natural alone has model-readout diagnostics. Every other array must match.
    ignored = lambda key: key.startswith('z__') or key.startswith('prediction__')
    expected = {key for key in natural if not ignored(key)}
    actual = {key for key in repeat if not ignored(key)}
    if expected != actual:
        raise RuntimeError('Natural/repeat full array key set differs')
    max_errors = {}
    for key in sorted(expected):
        left, right = natural[key], repeat[key]
        if left.shape != right.shape:
            raise RuntimeError('Natural/repeat full trajectory shape mismatch: ' + key)
        if left.dtype.kind in 'biu' and right.dtype.kind in 'biu':
            error = 0. if np.array_equal(left, right) else 1.
            limit = 0.
        else:
            error = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
            limit = tolerance
        max_errors[key] = error
        if error > limit:
            raise RuntimeError('Natural/repeat full trajectory mismatch: %s max_abs=%r' % (key, error))
    return dict(status='passed', arrays_checked=len(expected), full_trajectory=True,
                absolute_tolerance=tolerance, relative_tolerance=0,
                maximum_abs=max(max_errors.values()) if max_errors else 0., per_array_max_abs=max_errors,
                excluded_natural_only_diagnostics=sorted(key for key in natural if ignored(key)))


def run_screen(root, workers):
    source = root / 'screen_source'
    plan_path = root / 'screen_plan.json'
    plan = read_json(plan_path)
    initial = plan['initial_seeds']
    expansion = plan.get('expansion_seeds', list(range(2026110408, 2026110416)))
    objects = plan['objects']
    if len(initial) != 8 or len(expansion) != 8 or len(set(initial + expansion)) != 16:
        raise RuntimeError('Expected one frozen 8-seed initial and 8-seed extension pool')
    if len(objects) != 4 or len(set(objects)) != 4:
        raise RuntimeError('Expected four distinct development objects')
    if not set(initial + expansion).issubset(set(plan['environment_seeds'])):
        raise RuntimeError('Child plan must authorize all initial and extension seeds up front')
    output = root / 'screen'
    output.mkdir(parents=True, exist_ok=True)
    guard = CampaignLock(output / 'campaign.lock')
    try:
        checkpoint = Path(plan['repo']) / 'assets/rl_checkpoints/laptop/laptop_nopretrain_0.zip'
        expected_checkpoint = digest(checkpoint)
        frozen_paths = [plan_path, Path(__file__).resolve(), source / 'physical_plan.json']
        frozen_paths += sorted((source / 'code').glob('*.py'))
        frozen_paths += sorted(path for path in (source / 'bundle').rglob('*') if path.is_file())
        fingerprint = {str(path.resolve()): digest(path) for path in frozen_paths}
        signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        progress = output / 'campaign.json'
        if progress.exists():
            summary = read_json(progress)
            if summary.get('frozen_inputs_sha256') != signature or summary.get('checkpoint_sha256') != expected_checkpoint:
                raise RuntimeError('Resume frozen source/plan/scheduler/checkpoint differs; use a new root')
        else:
            summary = dict(version='online_natural_screen_v1', status='running', started_unix=time.time(),
                           frozen_inputs_sha256=signature, frozen_inputs=fingerprint,
                           checkpoint_sha256=expected_checkpoint, initial_seed_pool=initial,
                           extension_seed_pool=expansion, extension_used=False, cases=[])
        state_lock = threading.Lock()
        stop = threading.Event()

        def flush():
            summary['updated_unix'] = time.time()
            write_json(progress, summary)

        def mark(item, **fields):
            with state_lock:
                item.update(fields)
                flush()

        def one_case(item):
            if stop.is_set():
                mark(item, status='not_started_after_failure')
                return
            case = output / ('object%d_seed%d' % (item['object'], item['seed']))
            case.mkdir(exist_ok=True)

            def run(name, reference=None):
                if stop.is_set():
                    raise RuntimeError('Another screening task failed; no new child launched')
                if shutil.disk_usage(root).free < 4 * 1024 ** 3:
                    raise RuntimeError('Fewer than 4 GiB free; screening stopped with evidence retained')
                previous = [record for record in item['runs'] if record['name'] == name]
                if previous and previous[-1].get('status') == 'completed':
                    record = previous[-1]
                    folder = case / record['relative_dir']
                    for relative, field in [(name+'.npz', 'npz_sha256'), ('report.json', 'report_sha256'), ('audit.json', 'audit_sha256')]:
                        if digest(folder / relative) != record[field]:
                            raise RuntimeError('Completed screening evidence changed: ' + str(folder / relative))
                    report = read_json(folder / 'report.json')
                    audit, arrays = audit_rollout(folder/(name+'.npz'), report, item, plan, expected_checkpoint, digest(plan_path))
                    return record, audit, arrays
                folder_name = name if not previous else name+'__retry%d' % len(previous)
                folder = case / folder_name
                if folder.exists():
                    folder_name += '__%d' % time.time_ns()
                    folder = case / folder_name
                command = [sys.executable, '-B', '-u', str(source/'code/physical_simulation.py'),
                           '--bundle', str(source/'bundle'), '--plan', str(plan_path), '--repo', plan['repo'],
                           '--out', str(folder), '--object', str(item['object']), '--seed', str(item['seed']),
                           '--single-condition', name]
                if reference is not None:
                    command += ['--reference', str(reference)]
                record = dict(name=name, relative_dir=folder_name, status='running',
                              started_unix=time.time(), command=command, exit_code=None)
                with state_lock:
                    item['runs'].append(record)
                    flush()
                try:
                    with (case/(folder_name+'.log')).open('w', encoding='utf-8') as stream:
                        try:
                            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                                       timeout=plan.get('child_timeout_seconds', 300))
                            code = completed.returncode
                        except subprocess.TimeoutExpired:
                            code = -999
                    record['exit_code'] = code
                    if code != 0:
                        raise RuntimeError('Screen child failed/timeout with exit code %d; see %s' % (code, folder_name+'.log'))
                    report = read_json(folder/'report.json')
                    audit, arrays = audit_rollout(folder/(name+'.npz'), report, item, plan, expected_checkpoint, digest(plan_path))
                    if reference is not None:
                        if report.get('prefix_gate_passed') is not True or report.get('reference_sha256') != digest(reference):
                            raise RuntimeError('Triggered repeat reference gate/hash missing or failed')
                    elif report.get('reference') is not None:
                        raise RuntimeError('Unexpected reference in natural/no-trigger repeat')
                    write_json(folder/'audit.json', audit)
                    with state_lock:
                        record.update(status='completed', completed_unix=time.time(),
                                      npz_sha256=digest(folder/(name+'.npz')), report_sha256=digest(folder/'report.json'),
                                      audit_sha256=digest(folder/'audit.json'), summary=audit)
                        flush()
                    print(json.dumps(dict(object=item['object'], seed=item['seed'], name=name, status='completed', audit=audit)), flush=True)
                    return record, audit, arrays
                except Exception as error:
                    with state_lock:
                        record.update(status='failed', exception=repr(error), completed_unix=time.time())
                        flush()
                    raise

            try:
                mark(item, status='running')
                natural_record, natural_audit, natural = run('natural')
                reference = case/natural_record['relative_dir']/'natural.npz' if natural_audit['has_trigger'] else None
                repeat_record, repeat_audit, repeat = run('fresh_original_control', reference)
                if natural_audit['full_policy_state_sha256'] != repeat_audit['full_policy_state_sha256']:
                    raise RuntimeError('Natural/repeat full frozen policy hashes differ')
                comparison = audit_repeat(natural, repeat, plan['restore_repeat_atol'])
                comparison['triggered_reference_gate'] = reference is not None
                comparison['natural_npz_sha256'] = natural_record['npz_sha256']
                comparison['repeat_npz_sha256'] = repeat_record['npz_sha256']
                write_json(case/'repeat_audit.json', comparison)
                mark(item, status='completed', natural=natural_audit,
                     repeat_audit=comparison, repeat_audit_sha256=digest(case/'repeat_audit.json'),
                     completed_unix=time.time())
            except Exception as error:
                stop.set()
                mark(item, status='failed', exception=repr(error))
                print(json.dumps(dict(object=item['object'], seed=item['seed'], status='failed', exception=repr(error))), flush=True)

        def stage(seeds, name):
            existing = {(item['object'], item['seed']): item for item in summary['cases']}
            items = []
            for obj in objects:
                for seed in seeds:
                    item = existing.get((obj, seed))
                    if item is None:
                        item = dict(object=obj, seed=seed, stage=name, status='pending', runs=[])
                        summary['cases'].append(item)
                    items.append(item)
            summary.update(status='running', active_stage=name)
            flush()
            # Revalidate cached evidence on resume as well as executing missing runs.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(one_case, items))
            if any(item.get('status') != 'completed' for item in items):
                raise RuntimeError('Screening stage has failed/incomplete runs; pilot must not start')

        try:
            stage(initial, 'initial')
            initial_items = [item for item in summary['cases'] if item['seed'] in initial]
            initial_failures = sum(item['natural']['post_onset_failure'] for item in initial_items)
            summary['initial_post_onset_failures'] = initial_failures
            threshold = plan.get('minimum_post_onset_failures', 4)
            extend = initial_failures < threshold and plan.get('allow_single_expansion', True)
            if extend or summary.get('extension_used'):
                summary['extension_used'] = True
                flush()
                stage(expansion, 'extension')
            for path_text, expected in fingerprint.items():
                if digest(path_text) != expected:
                    raise RuntimeError('Frozen source/plan/scheduler changed during screening: ' + path_text)
            if digest(checkpoint) != expected_checkpoint:
                raise RuntimeError('Checkpoint file changed during screening')
            ordered = sorted(summary['cases'], key=lambda item: (objects.index(item['object']), item['seed']))
            results = []
            for item in ordered:
                audit = item['natural']
                results.append(dict(object=item['object'], seed=item['seed'], stage=item['stage'], status=item['status'],
                                    success=audit['success'], has_trigger=audit['has_trigger'], trigger_step=audit['trigger_step'],
                                    post_onset_failure=audit['post_onset_failure'], mechanism_class=audit['mechanism_class'],
                                    steps=audit['steps'], success_step=audit['success_step'],
                                    progress_goal_reached=audit['progress_goal_reached'], goal_and_near=audit['goal_and_near'],
                                    goal_and_strict=audit['goal_and_strict'], strict_contact_anywhere=audit['strict_contact_anywhere'],
                                    natural_npz_sha256=item['repeat_audit']['natural_npz_sha256'],
                                    repeat_npz_sha256=item['repeat_audit']['repeat_npz_sha256']))
            write_csv(output/'screen_results.csv', results)
            selected = []
            missing = []
            for obj in objects:
                candidates = [item for item in ordered if item['object'] == obj]
                for stratum in ['success', 'post_onset_failure']:
                    eligible = [item for item in candidates if item['natural']['has_trigger'] and
                                (item['natural']['success'] if stratum == 'success' else item['natural']['post_onset_failure'])]
                    if eligible:
                        chosen = eligible[0]
                        selected.append(dict(object=obj, seed=chosen['seed'], baseline_stratum=stratum,
                                             trigger_step=chosen['natural']['trigger_step'],
                                             mechanism_class=chosen['natural']['mechanism_class'],
                                             natural_relative_path='object%d_seed%d/%s/natural.npz' %
                                             (obj, chosen['seed'], [record for record in chosen['runs'] if record['name']=='natural' and record['status']=='completed'][-1]['relative_dir']),
                                             natural_npz_sha256=chosen['repeat_audit']['natural_npz_sha256']))
                    else:
                        missing.append(dict(object=obj, baseline_stratum=stratum))
            selection = dict(status='completed', rule='For each object, lowest seed success-with-trigger and lowest seed post-onset natural failure; no backfill',
                             source_frozen_inputs_sha256=signature, screened_origins=len(ordered),
                             initial_origins=len(initial_items), extension_used=summary['extension_used'],
                             post_onset_failures=sum(item['natural']['post_onset_failure'] for item in ordered),
                             no_trigger_origins=sum(not item['natural']['has_trigger'] for item in ordered),
                             all_origins_retained=True, selected_count=len(selected), maximum_selected=8,
                             origins=selected, missing_strata=missing)
            write_json(output/'selected_origins.json', selection)
            summary.update(status='completed', completed_unix=time.time(), selected_count=len(selected),
                           screened_origins=len(ordered), screen_results_sha256=digest(output/'screen_results.csv'),
                           selected_origins_sha256=digest(output/'selected_origins.json'))
            flush()
            print(json.dumps(selection), flush=True)
            return 0
        except Exception as error:
            summary.update(status='failed', exception=repr(error), failed_unix=time.time())
            flush()
            print(json.dumps(dict(status='failed', exception=repr(error))), flush=True)
            return 1
    finally:
        guard.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--phase', choices=['screen', 'pilot'], default='screen')
    parser.add_argument('--resume', action='store_true', help='Accepted for compatibility; safe hash-checked resume is automatic')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.phase == 'pilot':
        parser.error('Pilot is not implemented; screen must complete and a separate frozen pilot runner is required')
    return run_screen(args.root.resolve(), args.workers)


if __name__ == '__main__':
    sys.exit(main())
