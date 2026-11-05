"""Resume a frozen physical campaign with a recorded runtime worker count."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import hashlib
import importlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parallel-cases', type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / 'physical_plan.json').read_text(encoding='utf-8'))
    case_count = len(plan['objects']) * len(plan['environment_seeds'])
    if not 1 <= args.parallel_cases <= case_count:
        parser.error('parallel-cases must be between 1 and the number of physical origins')
    runtime = root / 'runtime'
    runtime.mkdir(exist_ok=True)
    started = datetime.now(timezone(timedelta(hours=8)))
    record_path = runtime / ('execution_' + started.strftime('%Y%m%d_%H%M%S_%f') + '.json')
    record = {'status': 'starting', 'started_beijing': started.isoformat(),
              'parallel_cases': args.parallel_cases, 'plan_parallel_cases': plan['parallel_cases'],
              'protocol_sha256': hashlib.sha256((root / 'protocol_hashes.json').read_bytes()).hexdigest(),
              'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        pending = record_path.with_suffix('.pending.json')
        pending.write_text(json.dumps(record, indent=2), encoding='utf-8')
        pending.replace(record_path)

    def runtime_executor(max_workers):
        record.update(status='running', executor_plan_workers=max_workers)
        save()
        return ThreadPoolExecutor(max_workers=args.parallel_cases)

    sys.path.insert(0, str(root / 'code'))
    campaign = importlib.import_module('combined_campaign')
    # The frozen entry point supplies its planned count to this executor factory.
    campaign.ThreadPoolExecutor = runtime_executor
    sys.argv = [str(root / 'code/combined_campaign.py'), '--root', str(root), '--resume']
    save()
    print(json.dumps(record), flush=True)
    try:
        campaign.main()
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='stopped', exception=repr(error))
        raise
    finally:
        record['finished_beijing'] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        save()


if __name__ == '__main__':
    main()
