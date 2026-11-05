"""Utility for the local offline audit of downloaded physical trajectories."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from temp_scripts.project_paths import OUTPUT_ROOT, output_path
from temp_scripts.combined_analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.run is None:
        args.run = Path(json.loads((OUTPUT_ROOT / 'combined_physical/latest.json').read_text(encoding='utf-8'))['run_dir'])
    root = output_path(args.run)
    if not (root / 'remote/campaign/campaign.json').is_file():
        raise SystemExit('Physical evidence has not been downloaded yet. Expected ' + str(root / 'remote/campaign/campaign.json'))
    analyze(root)
    print('OFFLINE RESULTS ' + str(root / 'analysis'), flush=True)


if __name__ == '__main__':
    main()
