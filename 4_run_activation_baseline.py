"""Step 4: fit current-state baseline probes, then open the native result viewer."""
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path

from temp_scripts.project_paths import bootstrap, default_index, OUTPUT_ROOT
bootstrap()


def main():
    parser = argparse.ArgumentParser(description="步骤4：首次持续开启窗口的原始激活 baseline（仅train/validation）")
    parser.add_argument("--index-dir", type=Path, default=default_index())
    parser.add_argument("--event", default="sustained_opening", choices=["sustained_opening", "first_clear_opening", "opening_stalled", "opening_reversed"])
    parser.add_argument("--window-radius", type=int, default=5)
    parser.add_argument("--weighting", choices=["trajectory", "frame"], default="trajectory", help="保留全部窗口帧，仅改变训练损失权重")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="小MLP随机种子；确定性线性模型只拟合一次")
    parser.add_argument("--skip-mlp", action="store_true", help="只运行线性、少量神经元、PCA和时间对照")
    parser.add_argument("--prepare-only", action="store_true", help="只导出窗口覆盖及全部选中帧，不拟合任何模型")
    parser.add_argument("--quick-check", action="store_true", help="缩小调参/迭代预算检查流程，不更新正式结果指针")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--view-only", nargs="?", const="latest", help="打开最近结果或指定结果目录，不重复训练")
    args = parser.parse_args()
    if args.window_radius < 0 or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("窗口半径和种子必须非负；种子不能重复")
    import matplotlib
    matplotlib.use("Agg")
    from temp_scripts.plots import configure_style
    configure_style()
    from temp_scripts.activation_baseline import latest_result, run_baseline
    if args.view_only:
        folder = latest_result() if args.view_only == "latest" else Path(args.view_only).resolve()
    else:
        from temp_scripts.data_access import DataRepository
        log_dir = OUTPUT_ROOT / "activation_baseline" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        logging.basicConfig(filename=log_dir / (stamp + ".log"), level=logging.INFO, encoding="utf-8")
        def progress(message):
            print(message, flush=True)
            logging.info(message)
        try:
            folder, summary = run_baseline(DataRepository(args.index_dir), event=args.event, radius=args.window_radius,
                                          weighting=args.weighting, seeds=tuple(args.seeds), include_mlp=not args.skip_mlp,
                                          quick=args.quick_check, prepare_only=args.prepare_only, progress=progress)
        except Exception:
            logging.exception("Baseline failed")
            print("Error log:", log_dir / (stamp + ".log"), flush=True)
            raise
        print("Status:", summary["status"], flush=True)
        if "counts" in summary:
            print("Coverage:", summary["counts"], flush=True)
            print("Selected models with convergence warnings:", summary["selected_models_with_warnings"], flush=True)
    print("Output:", folder, flush=True)
    if not args.no_gui and not args.prepare_only:
        from temp_scripts.activation_baseline_viewer import launch
        launch(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
