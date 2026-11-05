"""Step 7: resumable state-assisted SAE experiments, with no physical simulation."""
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
from temp_scripts.project_paths import bootstrap
bootstrap()


def main():
    parser = argparse.ArgumentParser(description="状态辅助SAE：分阶段开发集实验")
    parser.add_argument("--study-dir", type=Path)
    parser.add_argument("--stage", choices=["core", "losses", "structure", "fidelity", "refine", "refine_slots", "concentration", "negative"], default="core")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--layer", choices=["h1", "h2"], default=None)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    from temp_scripts.state_assisted_study import prepare_study, run_experiments, experiment_configs
    folder = args.study_dir or prepare_study(layer=args.layer or "h1")
    if args.study_dir and args.layer:
        import json
        existing = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
        if existing["layer"] != args.layer:
            parser.error("--layer differs from the existing study layer")
    print("Study:", folder, flush=True)
    if not args.analyze_only:
        configs = experiment_configs(args.stage)
        if args.only:
            configs = [c for c in configs if c.name in args.only]
            if len(configs) != len(set(args.only)):
                parser.error("Some --only configuration names were not found")
        run_experiments(folder, configs, args.seeds)
    from temp_scripts.state_assisted_analysis import summarize_study
    print(summarize_study(folder)[0], flush=True)
    if args.verify:
        from temp_scripts.state_assisted_analysis import verify_study, match_seed_features
        report = verify_study(folder)
        print("Verification:", report.get("passed"), "evidence:", folder / "analysis" / "independent_verification.json", flush=True)
        matches = match_seed_features(folder)
        print("Cross-seed pairs:", len(matches), "evidence:", folder / "analysis" / "cross_seed_matching.json", flush=True)


if __name__ == "__main__":
    main()
