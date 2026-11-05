"""Verify the frozen package and run resumable independent-process physical branches."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import threading
import time

import numpy as np
import torch
from combined_intervention import CombinedEditor
from physical_runtime import NumericSAE, actor_tail, verify_bundle


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, indent=2), encoding='utf-8')
    pending.replace(path)


def verify_package(root):
    hashes = json.loads((root / 'protocol_hashes.json').read_text(encoding='utf-8'))
    for relative, expected in hashes.items():
        path = (root / relative).resolve()
        if root not in path.parents or digest(path) != expected:
            raise RuntimeError('Frozen file mismatch: ' + relative)
    bundle = root / 'bundle'
    numeric_check = verify_bundle(bundle)
    if not numeric_check['passed']:
        raise RuntimeError('Numeric model cross-runtime check failed')
    numeric = {p.name: NumericSAE(p) for p in (bundle / 'models').iterdir()}
    editor = CombinedEditor(bundle, numeric)
    rows = json.loads((bundle / 'combined_reference.json').read_text(encoding='utf-8'))
    max_edit = 0.
    with np.load(bundle / 'combined_reference.npz', allow_pickle=False) as data, np.load(bundle / 'actor.npz', allow_pickle=False) as actor_data:
        reference = {k: data[k].copy() for k in data.files}
        actor = {k: actor_data[k] for k in actor_data.files}
        mu = actor_tail(actor, reference['emitted'])
        np.testing.assert_allclose(mu, reference['mu'], atol=2e-5, rtol=2e-5)
        for i, row in enumerate(rows):
            actual, detail = editor.edit(reference['h'][i], row['model'], row['group'], row['dose'], row['control'], row['random_key'])
            if detail['status'] != row['status']:
                raise RuntimeError('Cross-runtime feasibility status mismatch')
            np.testing.assert_allclose(actual, reference['emitted'][i], atol=2e-5, rtol=2e-5)
            max_edit = max(max_edit, float(np.max(np.abs(actual - reference['emitted'][i]))))
    result = {'passed': True, 'reference_rows': len(rows), 'max_edit_abs': max_edit, 'numeric': numeric_check,
              'protocol_hashes_sha256': digest(root / 'protocol_hashes.json')}
    write_json(root / 'cross_runtime_verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(2)
    verification = verify_package(root)
    print(json.dumps(verification), flush=True)
    if args.verify_only:
        return
    import fcntl
    lock_file = (root / 'campaign.lock').open('a')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('This campaign is already running')
    plan = json.loads((root / 'physical_plan.json').read_text(encoding='utf-8'))
    output = root / 'campaign'
    output.mkdir(exist_ok=True)
    progress = output / 'campaign.json'
    if progress.exists():
        if not args.resume:
            raise RuntimeError('Campaign exists; use --resume')
        summary = json.loads(progress.read_text(encoding='utf-8'))
        if summary['protocol_sha256'] != digest(root / 'protocol_hashes.json'):
            raise RuntimeError('Resume protocol differs')
    else:
        summary = {'status': 'running', 'started_unix': time.time(),
                   'protocol_sha256': digest(root / 'protocol_hashes.json'), 'cases': [
                       {'object': obj, 'seed': seed, 'status': 'pending', 'runs': []}
                       for obj in plan['objects'] for seed in plan['environment_seeds']]}
    lock = threading.Lock()

    def flush():
        write_json(progress, summary)

    def one_case(item):
        case = output / ('object%d_seed%d' % (item['object'], item['seed']))
        case.mkdir(exist_ok=True)

        def run(name, reference=None, model=None, group=None, duration=0, dose=None, control=None):
            if shutil.disk_usage(root).free < 4 * 1024 ** 3:
                raise RuntimeError('Fewer than 4 GiB free; campaign stopped with evidence retained')
            previous = [r for r in item['runs'] if r['name'] == name]
            if previous:
                last = previous[-1]
                archive = case / last['relative_dir'] / (name + '.npz')
                report = archive.with_name('report.json')
                if last['exit_code'] == 0 and archive.exists() and report.exists():
                    if digest(archive) != last['npz_sha256'] or digest(report) != last['report_sha256']:
                        raise RuntimeError('Completed evidence changed: ' + name)
                    return last
            folder_name = name if not previous else name + '__retry%d' % len(previous)
            target = case / folder_name
            if target.exists():
                # An interrupted child can leave a directory without a recorded completion.
                folder_name += '__%d' % time.time_ns()
                target = case / folder_name
            command = [sys.executable, '-B', '-u', str(root / 'code/physical_simulation.py'),
                       '--bundle', str(root / 'bundle'), '--plan', str(root / 'physical_plan.json'),
                       '--repo', plan['repo'], '--out', str(target), '--object', str(item['object']),
                       '--seed', str(item['seed']), '--single-condition', name, '--group-edit']
            if reference:
                command += ['--reference', str(reference)]
            if model:
                command += ['--model-name', model, '--group-name', group, '--duration', str(duration),
                            '--dose', dose, '--control', control]
            with (case / (folder_name + '.log')).open('w') as log:
                try:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=plan['child_timeout_seconds']).returncode
                except subprocess.TimeoutExpired:
                    code = -999
            record = {'name': name, 'relative_dir': folder_name, 'exit_code': code}
            report = target / 'report.json'
            archive = target / (name + '.npz')
            if report.exists():
                record.update(json.loads(report.read_text(encoding='utf-8')))
            if code == 0:
                if not archive.exists() or record.get('status') != 'fresh_process_rollout_completed':
                    record['exit_code'] = -998
                else:
                    record.update(npz_sha256=digest(archive), report_sha256=digest(report))
            with lock:
                item['runs'].append(record)
                flush()
                print(json.dumps({'object': item['object'], 'seed': item['seed'], 'name': name,
                                  'exit_code': record['exit_code'], 'summary': record.get('summary')}), flush=True)
            return record

        try:
            with lock:
                item['status'] = 'running'
                flush()
            baseline = run('natural')
            if baseline['exit_code']:
                raise RuntimeError('Natural rollout failed')
            reference = case / baseline['relative_dir'] / 'natural.npz'
            with np.load(reference, allow_pickle=False) as arrays:
                hits = np.flatnonzero(arrays['opening'] >= plan['trigger_opening_rad'])
            item['trigger_step'] = int(hits[0]) if len(hits) else None
            if not len(hits):
                item['status'] = 'no_trigger_all_outcomes_retained'
                return
            repeat = run('fresh_original_control', reference)
            if repeat['exit_code']:
                raise RuntimeError('Full natural-repeat gate failed')
            failed = False
            for model in plan['models']:
                for group in plan['groups']:
                    for duration in plan['durations']:
                        for dose in plan['doses']:
                            for control in plan['controls']:
                                label = '%s_%s_d%d_%s_%s' % (model, group, duration, dose, control)
                                record = run(label, reference, model, group, duration, dose, control)
                                failed |= record['exit_code'] != 0
            item['status'] = 'completed_with_failures' if failed else 'completed'
        except Exception as error:
            item.update(status='failed', exception=repr(error))
        finally:
            with lock:
                flush()

    with ThreadPoolExecutor(max_workers=plan['parallel_cases']) as pool:
        list(pool.map(one_case, summary['cases']))
    summary['status'] = 'completed' if all(c['status'] in ('completed', 'no_trigger_all_outcomes_retained') for c in summary['cases']) else 'completed_with_failures'
    summary['elapsed_seconds'] = time.time() - summary['started_unix']
    flush()
    if summary['status'] != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
