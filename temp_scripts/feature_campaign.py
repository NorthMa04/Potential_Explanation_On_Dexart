"""Bounded feature-intervention pilot; independent-process replay for each condition."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.bundle, args.plan, args.out = args.bundle.resolve(), args.plan.resolve(), args.out.resolve()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'started_unix': time.time(), 'cases': [
        {'object': obj, 'seed': seed, 'status': 'pending', 'runs': []}
        for obj in plan['objects'] for seed in plan['environment_seeds']]}
    lock = threading.Lock()
    def flush():
        pending = args.out / 'campaign.pending.json'
        pending.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        pending.replace(args.out / 'campaign.json')
    def one_case(item):
        obj, seed = item['object'], item['seed']
        case = args.out / ('object%d_seed%d' % (obj, seed))
        case.mkdir()
        with lock:
            item['status'] = 'running'; flush()
        def run(name, reference=None, model=None, operation=None, control=None):
            target = case / name
            cmd = [sys.executable, '-u', str(Path(__file__).with_name('physical_simulation.py')),
                   '--bundle', str(args.bundle), '--plan', str(args.plan), '--out', str(target),
                   '--object', str(obj), '--seed', str(seed), '--single-condition', name, '--feature-edit']
            if reference:
                cmd += ['--reference', str(reference)]
            if model:
                cmd += ['--model-name', model, '--duration', '1', '--operation', operation, '--control', control]
            with (case / (name + '.log')).open('w') as stream:
                try:
                    completed = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                               timeout=plan['child_timeout_seconds'])
                    code = completed.returncode
                except subprocess.TimeoutExpired:
                    code = -999
            record = {'name': name, 'exit_code': code}
            for filename, key in [('report.json', None), ('prefix_failure.json', 'prefix_failure')]:
                if (target / filename).exists():
                    value = json.loads((target / filename).read_text())
                    if key: record[key] = value
                    else: record.update(value)
            with lock:
                item['runs'].append(record); flush()
                print(json.dumps({'object': obj, 'seed': seed, 'name': name, 'code': code,
                                  'summary': record.get('summary')}), flush=True)
            return record
        baseline = run('natural')
        if baseline['exit_code']:
            status = 'baseline_failed'
        else:
            reference = case / 'natural/natural.npz'
            with np.load(reference) as data:
                eligible = np.flatnonzero(data['feature_trigger_eligible'])
            with lock:
                item['eligible_trigger_steps'] = eligible.tolist()
            if not len(eligible):
                status = 'no_trigger_all_outcomes_retained'
            else:
                control = run('fresh_original_control', reference)
                if control['exit_code']:
                    status = 'fresh_repeat_gate_failed_no_interventions'
                else:
                    for sae_seed in plan['sae_seeds']:
                        model = 'assisted_seed%d' % sae_seed
                        for operation in plan['operations']:
                            for condition in plan['controls']:
                                run('%s_%s_%s' % (model, operation, condition), reference,
                                    model, operation, condition)
                        if plan['half_suppression_target_only']:
                            run('%s_half_suppress_target' % model, reference, model, 'half_suppress', 'target')
                    status = 'completed' if all(r['exit_code'] == 0 for r in item['runs']) else 'completed_with_run_failures'
        with lock:
            item['status'] = status; flush()
        return item
    flush()
    with ThreadPoolExecutor(max_workers=plan['parallel_cases']) as pool:
        futures = {pool.submit(one_case, item): item for item in summary['cases']}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                with lock:
                    futures[future]['status'] = 'orchestrator_exception'
                    futures[future]['exception'] = repr(error); flush()
    summary['elapsed_seconds'] = time.time() - summary['started_unix']
    summary['status'] = 'completed' if all(c['status'] in ('completed', 'no_trigger_all_outcomes_retained') for c in summary['cases']) else 'completed_with_failures'
    flush()


if __name__ == '__main__':
    main()
