"""Step 8: evaluate a frozen development plan without fitting or simulation."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path
from temp_scripts.project_paths import bootstrap, output_path
bootstrap()


def main():
    parser = argparse.ArgumentParser(description="冻结方案的独立评价及可选离线动作干预")
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--offline-interventions", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    from temp_scripts.activation_baseline import dump, sha256
    root = output_path(args.study_dir)
    plan = json.loads((root / "frozen_evaluation_plan.json").read_text(encoding="utf-8"))
    for relative, digest in plan.get("input_artifact_hashes", {}).items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError("Frozen study input changed: " + relative)
    for identifier, hashes in plan.get("additional_artifact_hashes", {}).items():
        for filename, digest in hashes.items():
            path = (root / identifier / filename).resolve()
            if not path.is_relative_to(root) or sha256(path) != digest:
                raise ValueError("Frozen auxiliary artifact changed: " + identifier + "/" + filename)
    from temp_scripts.state_assisted_holdout import evaluate_holdout
    folder, summary = evaluate_holdout(root, plan["experiment_ids"])
    dump(root / "evaluation_location.json", {"directory": str(folder),
        "frozen_plan_sha256": sha256(root / "frozen_evaluation_plan.json")})
    if plan.get("rotation_controls"):
        from temp_scripts.state_assisted_controls import evaluate_rotation_holdout
        evaluate_rotation_holdout(root, folder)
        print("Frozen rotation controls completed", flush=True)
    if args.offline_interventions and plan.get("intervention_experiment_ids"):
        from temp_scripts.state_assisted_intervention import run_offline_interventions
        intervention_folder, report = run_offline_interventions(root, folder, plan["intervention_experiment_ids"])
        dump(root / "intervention_location.json", {"directory": str(intervention_folder)})
        print("Offline interventions:", intervention_folder, flush=True)
    print("Holdout:", folder, "Counts:", summary["counts"], flush=True)


if __name__ == "__main__":
    main()
