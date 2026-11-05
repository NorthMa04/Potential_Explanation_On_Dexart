"""Step 6: aggregate completed development-set SAE runs; no fitting."""
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path

from temp_scripts.project_paths import bootstrap
bootstrap()


def main():
    parser = argparse.ArgumentParser(description="步骤6：对比多个SAE运行，按配置汇总训练种子")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    from temp_scripts.sae_comparison import compare_runs
    folder, rows = compare_runs(args.runs)
    import torch
    torch.set_num_threads(1)
    from temp_scripts.sae_controls import pca_fidelity_controls
    pca_fidelity_controls(args.runs[0], folder)
    print("Configurations:", len(rows), "Output:", folder)
    for row in rows:
        print("width", row["width"], "K", row["top_k"], row["preprocessing"], "seeds", row["seeds"],
              "validation action MAE", round(row["validation_action_clipped_mae_mean"], 6))


if __name__ == "__main__":
    main()
