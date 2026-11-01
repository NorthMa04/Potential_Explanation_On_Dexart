"""Standalone Visual Studio entry: build the dense state/activation information table."""
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path

from temp_scripts.project_paths import bootstrap, default_index
bootstrap()


def main():
    from temp_scripts.data_access import DataRepository, DEV_ROLES
    from temp_scripts.information_table import build_table
    parser = argparse.ArgumentParser(description="生成逐状态信息总表（CSV + SQLite），无需启动可视化窗口")
    parser.add_argument("--index-dir", type=Path, default=default_index())
    parser.add_argument("--development-only", action="store_true", help="仅导出train/validation，默认导出全部固定划分")
    args = parser.parse_args()
    repo = DataRepository(args.index_dir)
    records = repo.filter(DEV_ROLES) if args.development_only else None
    print("Building state/activation table and reconciling all exported rows...", flush=True)
    folder, report = build_table(repo, records=records)
    print("Output:", folder, flush=True)
    print("Verification:", report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
