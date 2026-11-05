"""Standalone research figure for the November SAE parameter study."""
import csv
from pathlib import Path

from matplotlib.figure import Figure

from temp_scripts.plots import configure_style
from temp_scripts.project_paths import output_path


def export_study_figure(comparison, destination):
    comparison, destination = Path(comparison), output_path(destination)
    with (comparison / "configurations.csv").open(encoding="utf-8-sig") as stream:
        all_rows = list(csv.DictReader(stream))
    with (comparison / "raw_baselines.csv").open(encoding="utf-8-sig") as stream:
        baselines = {r["target"]: float(r["value"]) for r in csv.DictReader(stream) if r["model"] == "h1_top4"}
    with (comparison / "pca_fidelity.csv").open(encoding="utf-8-sig") as stream:
        pca = {int(r["pca_rank"]): float(r["validation_action_clipped_mae"]) for r in csv.DictReader(stream)
               if r["preprocessing"] == "global"}
    # Explicit successive comparisons; the full 12-configuration table remains available.
    cases = [(128, 8, "coordinate", 500), (64, 16, "coordinate", 500),
             (64, 16, "global", 500), (64, 16, "global", 1500),
             (64, 24, "global", 1500), (64, 32, "global", 1500)]
    selected = []
    for width, k, mode, epochs in cases:
        selected.extend(r for r in all_rows if int(r["width"]) == width and int(r["top_k"]) == k
                        and r["preprocessing"] == mode and int(r["max_epochs"]) == epochs
                        and float(r["learning_rate"]) == .001)
    if not selected:
        raise ValueError("No requested comparison cases")
    configure_style()
    fig = Figure(figsize=(16, 10), layout="constrained")
    axes = fig.subplots(2, 2).ravel()
    labels = [f"{r['width']}/{r['top_k']}\n{'逐维' if r['preprocessing']=='coordinate' else '全局'}·≤{r['max_epochs']}轮" for r in selected]
    panels = [("validation_action_clipped_mae", "裁剪后动作 MAE ↓", None, 1.),
              ("z_top4_distance_m_mae", "4个特征：距离 MAE (cm) ↓", baselines["distance_m"], 100.),
              ("z_top4_opening_rad_mae", "4个特征：开启量 MAE (rad) ↓", baselines["opening_rad"], 1.),
              ("z_top4_any_contact_average_precision", "4个特征：任一组接触 AP ↑", baselines["any_contact"], 1.)]
    for ax, (metric, title, baseline, scale) in zip(axes, panels):
        values = [float(r[metric + "_mean"]) * scale for r in selected]
        errors = [float(r[metric + "_std"]) * scale for r in selected]
        ax.bar(range(len(selected)), values, yerr=errors, capsize=4,
               color=["#8b99aa", "#9b69a8", "#4c95a9", "#397a98", "#16806a", "#217143"])
        ax.set_xticks(range(len(selected)), labels, fontsize=9)
        ax.set(title=title, ylim=(0, 1 if "precision" in metric else None))
        if baseline is not None:
            ax.axhline(baseline * scale, ls="--", color="#b63c51", label="原始4个神经元")
        if metric == "validation_action_clipped_mae":
            ax.axhline(pca[16], ls=":", color="#7758a6", label="全局缩放 PCA16")
            if 32 in pca:
                ax.axhline(pca[32], ls="--", color="#586471", label="全局缩放 PCA32")
        ax.legend(loc="upper right")
        ax.grid(True, axis="y", alpha=.3)
    fig.suptitle("SAE 参数探索 · 仅 validation，轨迹等权\n横轴：字典宽度/K；误差线是3个训练种子的标准差，并非数据置信区间", fontsize=13)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    return destination
