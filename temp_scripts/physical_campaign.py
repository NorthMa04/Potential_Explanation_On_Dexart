"""Run development pilot in fresh processes, gating every intervention on prefix equality."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    args.bundle, args.plan, args.out = args.bundle.resolve(), args.plan.resolve(), args.out.resolve()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'cases': [], 'started_unix': time.time()}

    def flush():
        (args.out / 'campaign.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    def run(case, name, obj, seed, reference=None, model=None, replacement='raw', duration=0):
        target = case / name
        cmd = [sys.executable, '-u', str(Path(__file__).with_name('physical_simulation.py')),
               '--bundle', str(args.bundle), '--plan', str(args.plan), '--out', str(target),
               '--object', str(obj), '--seed', str(seed), '--single-condition', name]
        if reference:
            cmd += ['--reference', str(reference)]
        if model:
            cmd += ['--model-name', model, '--replacement', replacement, '--duration', str(duration)]
        with (case / (name + '.log')).open('w') as stream:
            completed = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        record = {'name': name, 'exit_code': completed.returncode}
        if (target / 'report.json').exists():
            record.update(json.loads((target / 'report.json').read_text()))
        if (target / 'prefix_failure.json').exists():
            record['prefix_failure'] = json.loads((target / 'prefix_failure.json').read_text())
        print(json.dumps({'object': obj, 'seed': seed, 'name': name, 'exit': completed.returncode,
                          'summary': record.get('summary')}), flush=True)
        return record

    flush()
    for obj in plan['objects']:
        for seed in plan['environment_seeds']:
            case = args.out / ('object%d_seed%d' % (obj, seed))
            case.mkdir()
            item = {'object': obj, 'seed': seed, 'runs': []}
            summary['cases'].append(item)
            baseline = run(case, 'natural', obj, seed)
            item['runs'].append(baseline)
            flush()
            if baseline['exit_code']:
                item['status'] = 'baseline_failed'
                continue
            reference = case / 'natural/natural.npz'
            with np.load(reference) as a:
                has_trigger = bool(np.any(a['opening'] >= plan['trigger_opening_rad']))
            if not has_trigger:
                item['status'] = 'no_trigger_all_outcomes_retained'
                continue
            control = run(case, 'fresh_original_control', obj, seed, reference)
            item['runs'].append(control)
            flush()
            if control['exit_code']:
                item['status'] = 'fresh_repeat_gate_failed_no_replacements'
                continue
            for sae_seed in plan['sae_seeds']:
                for kind in ('pure', 'assisted'):
                    for replacement in ('raw', 'relu'):
                        for duration in plan['durations']:
                            model = '%s_seed%d' % (kind, sae_seed)
                            name = '%s_%s_%d' % (model, replacement, duration)
                            result = run(case, name, obj, seed, reference, model, replacement, duration)
                            item['runs'].append(result)
                            flush()
            item['status'] = 'completed' if all(r['exit_code'] == 0 for r in item['runs']) else 'completed_with_failed_runs'
            flush()
    summary['status'] = 'completed' if all(c.get('status') in ('completed', 'no_trigger_all_outcomes_retained') for c in summary['cases']) else 'completed_with_gate_or_run_failures'
    summary['elapsed_seconds'] = time.time() - summary['started_unix']
    flush()


if __name__ == '__main__':
    main()
