"""Step 5: user-run pure TopK SAE, frozen actor fidelity and state analysis."""
import sys
sys.dont_write_bytecode = True
import argparse
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path

from temp_scripts.project_paths import bootstrap, default_index, PROJECT_ROOT
bootstrap()


def build_parser():
    parser = argparse.ArgumentParser(description="步骤5：纯 TopK SAE（仅开发集），随后打开分析窗口")
    parser.add_argument("--index-dir", type=Path, default=default_index())
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT.parents[2] / "dexart_assets.zip")
    parser.add_argument("--layer", choices=["h1", "h2"], default="h1")
    parser.add_argument("--event", choices=["sustained_opening", "first_clear_opening", "opening_stalled", "opening_reversed"], default="sustained_opening")
    parser.add_argument("--window-radius", type=int, default=5)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50, help="验证MSE连续多少轮未创新低后停止")
    parser.add_argument("--scale-floor", type=float, default=1e-6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--compare-configs", action="store_true", help="依次比较(128,8)/(128,4)/(128,16)/(256,8)；替代width/top-k")
    parser.add_argument("--weighting", choices=["trajectory", "frame"], default="trajectory")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip-probes", action="store_true", help="只训练SAE并评价重建/动作，不拟合状态探针")
    parser.add_argument("--prepare-only", action="store_true", help="检查开发集窗口和原actor，不训练SAE、不更新正式结果指针")
    parser.add_argument("--quick-check", action="store_true", help="最多5轮，仅检查流程，不更新正式结果指针")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--view-only", nargs="?", const="latest", help="复看最近完成结果或指定目录，不重复训练")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    from temp_scripts.sae_model import SAEConfig
    try:
        config = SAEConfig(args.width, args.top_k, args.learning_rate, args.batch_size,
                           args.max_epochs, args.patience, args.scale_floor).validate()
        if args.threads < 1 or args.window_radius < 0:
            raise ValueError("线程数必须为正，窗口半径不能为负")
        if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
            raise ValueError("种子必须非负且不能重复")
    except ValueError as exc:
        parser.error(str(exc))
    import matplotlib
    matplotlib.use("Agg")
    from temp_scripts.plots import configure_style
    configure_style()
    from temp_scripts.sae_training import SAE_HOME, latest_result, run_sae
    if args.view_only:
        folder = latest_result() if args.view_only == "latest" else Path(args.view_only).resolve()
    else:
        import torch
        from temp_scripts.data_access import DataRepository
        torch.set_num_threads(args.threads)
        configurations = [replace(config, width=w, top_k=k) for w, k in
                          [(128, 8), (128, 4), (128, 16), (256, 8)]] if args.compare_configs else [config]
        if args.quick_check:
            configurations = [replace(c, max_epochs=min(c.max_epochs, 5), patience=min(c.patience, 3))
                              for c in configurations]
        logs = SAE_HOME / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        log_file = logs / (stamp + ".log")
        logging.basicConfig(filename=log_file, level=logging.INFO, encoding="utf-8", format="%(asctime)s %(message)s")
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        def progress(message):
            print(message, flush=True)
            logging.info(message)
        progress("Pure SAE: train/validation only; frozen actor; no physical simulation.")
        try:
            folder, summary = run_sae(DataRepository(args.index_dir), args.checkpoint,
                configurations=configurations, seeds=tuple(args.seeds), layer=args.layer,
                event=args.event, radius=args.window_radius, weighting=args.weighting,
                device=args.device, prepare_only=args.prepare_only, skip_probes=args.skip_probes,
                quick=args.quick_check, progress=progress)
        except (Exception, KeyboardInterrupt):
            logging.exception("Step 5 failed or interrupted")
            print("Log:", log_file, flush=True)
            raise
        print("Status:", summary["status"], "Coverage:", summary["counts"], flush=True)
        for result in summary["experiments"]:
            print(result["experiment_id"], "best epoch:", result["best_epoch"],
                  "unused training features:", result["health"]["train"]["never_active_features"], flush=True)
    print("Output:", folder, flush=True)
    if not args.no_gui and not args.prepare_only:
        from temp_scripts.sae_viewer import launch
        launch(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
