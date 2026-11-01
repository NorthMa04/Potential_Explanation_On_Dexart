"""Visual Studio startup file. F5/Ctrl+F5 opens the offline trajectory viewer."""
import sys
sys.dont_write_bytecode = True

import argparse
import logging
from pathlib import Path

from temp_scripts.project_paths import bootstrap, default_index, new_run

bootstrap()


def main():
    p = argparse.ArgumentParser(description="DexArt 离线轨迹读取与可视化审阅（实验步骤1、2）")
    p.add_argument("--index-dir", type=Path, default=default_index())
    p.add_argument("--export-demo", action="store_true", help="无窗口导出同物体成功/失败示例图和完整数值")
    p.add_argument("--validate-all", action="store_true", help="逐条检查全部280条轨迹，报告保存在项目outputs下")
    args = p.parse_args()
    run = new_run("export" if args.export_demo else "validation" if args.validate_all else "viewer")
    logging.basicConfig(filename=run / "application.log", level=logging.INFO, encoding="utf-8", format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    try:
        import matplotlib
        if args.export_demo or args.validate_all:
            matplotlib.use("Agg")
        from temp_scripts.data_access import DataRepository
        from temp_scripts.export_results import json_file, record_run, export_trajectory, export_comparison
        from temp_scripts.plots import configure_style
        configure_style()
        repo = DataRepository(args.index_dir)
        record_run(repo, run, "headless" if args.export_demo or args.validate_all else "native_tk")
        print("Data:", repo.root, flush=True)
        print("Outputs:", run, flush=True)
        if args.validate_all:
            report = repo.validate_all()
            json_file(run / "validation" / "data_validation.json", report)
            print(report, flush=True)
            if report["errors"]:
                return 1
        if args.export_demo:
            pair = repo.example_pair()
            if pair is None:
                raise ValueError("开发集没有同物体成功/失败示例。")
            a, b = [repo.load(tid) for tid in pair]
            for traj in [a, b]:
                j = traj.event_anchor("first_clear_opening") or 0
                print("Exported:", export_trajectory(repo, traj, run, j), flush=True)
            print("Comparison:", export_comparison(repo, a, b, run), flush=True)
        if not args.validate_all and not args.export_demo:
            from temp_scripts.trajectory_app import launch
            launch(repo, run)
        return 0
    except Exception:
        logging.exception("Application failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
