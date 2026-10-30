"""Standalone Visual Studio entry for experiment step 3; no training or simulation."""
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path

from project_paths import bootstrap, default_index, PROJECT_ROOT
bootstrap()


def main():
    parser = argparse.ArgumentParser(description="步骤3：恢复冻结actor，逐段前向复现并输出误差图")
    parser.add_argument("--index-dir", type=Path, default=default_index())
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT.parents[2] / "dexart_assets.zip", help="原资源ZIP或laptop_nopretrain_0.zip")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--trajectory-id", help="可选：只校验指定轨迹，默认校验全部280条")
    parser.add_argument("--no-gui", action="store_true", help="只计算和导出，适合命令行检查")
    parser.add_argument("--view-only", nargs="?", const="latest", help="直接查看最近结果或指定结果目录，不重复计算")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    import matplotlib
    matplotlib.use("Agg")
    from plots import configure_style
    configure_style()
    from actor_verification import ACTOR_HOME, verify_actor, latest_result
    log_dir = ACTOR_HOME / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    logging.basicConfig(filename=log_dir / (stamp + ".log"), level=logging.INFO, encoding="utf-8", format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    if args.view_only:
        import json
        folder = latest_result() if args.view_only == "latest" else Path(args.view_only).resolve()
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    else:
        import torch
        from data_access import DataRepository
        torch.set_num_threads(args.threads)
        repo = DataRepository(args.index_dir)
        if args.trajectory_id and args.trajectory_id not in repo.by_id:
            parser.error("Unknown trajectory ID")
        records = [repo.by_id[args.trajectory_id]] if args.trajectory_id else None
        print("Frozen actor forward verification; no training or simulation.", flush=True)
        try:
            folder, summary = verify_actor(repo, args.checkpoint, records=records, device=args.device, batch_size=args.batch_size,
                                           atol=args.atol, rtol=args.rtol, progress=lambda s: print(s, flush=True))
        except Exception:
            logging.exception("Verification failed")
            print("Error details:", log_dir / (stamp + ".log"), flush=True)
            raise
    print("Result:", summary["status"], "Trajectories:", summary["trajectories"], "Steps:", summary["decision_steps"], flush=True)
    print("Output:", folder, flush=True)
    for row in summary["comparisons"]:
        print("%-10s MAE=%.6g max=%.6g failed=%d" % (row["comparison"], row["mae"], row["max_abs_error"], row["failed_values"]), flush=True)
    if not args.no_gui:
        from actor_verification_viewer import launch
        launch(folder)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
