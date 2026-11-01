"""Baseline figures; examples are chosen by ID order, never prediction quality."""
import numpy as np
from matplotlib.figure import Figure

TARGET_LABELS = {"distance_m": "手掌—把手距离 (m)", "opening_rad": "相对初始开启量 (rad)", "any_contact": "任一组接触 / 预测概率"}


def window_figure(fig, rows, y, predictions, target, model, seed, trajectory_id):
    fig.clear()
    ax = fig.add_subplot(111)
    selected = np.array([r["trajectory_id"] == trajectory_id for r in rows])
    column = list(TARGET_LABELS).index(target)
    relative = np.array([int(r["relative_state_index"]) for r in rows])[selected]
    key = f"{target}__{model}__seed{seed}"
    ax.plot(relative, y[selected, column], "o-", label="记录的当前状态", lw=2)
    if key in predictions:
        ax.plot(relative, predictions[key][selected], "s--", label="探针预测", lw=1.5)
    ax.axvline(0, color="purple", alpha=.5, label="事件起点 s_j")
    ax.set(xlabel="相对事件起点的状态步数（仅作图，不输入模型）", ylabel=TARGET_LABELS[target],
           title=f"{trajectory_id}\n{model} / seed {seed}")
    if target == "any_contact":
        ax.set_ylim(-.08, 1.08)
    ax.grid(alpha=.2)
    ax.legend()
    return ax


def summary_figure(fig, summaries):
    fig.clear()
    for i, target in enumerate(TARGET_LABELS, 1):
        ax = fig.add_subplot(3, 1, i)
        metric = "log_loss" if target == "any_contact" else "mae"
        rows = [r for r in summaries if r["target"] == target and r["split"] == "validation" and r["weighting"] == "trajectory"
                and (r["model"] in ("constant", "time_spline", "h1_linear", "h2_linear", "h1_top1", "h1_top2", "h1_top4", "h2_top1", "h2_top2", "h2_top4")
                     or r["model"].endswith("_mlp"))]
        names = list(dict.fromkeys(r["model"] for r in rows))
        values = [[float(r[metric]) for r in rows if r["model"] == n] for n in names]
        ax.bar(range(len(names)), [np.mean(v) for v in values], color="#4b7ca7", alpha=.8)
        for j, v in enumerate(values):
            ax.scatter(np.full(len(v), j), v, s=14, color="#c4643c", zorder=3)
        ax.set_xticks(range(len(names)), names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Log loss" if target == "any_contact" else f"MAE ({'m' if target == 'distance_m' else 'rad'})")
        ax.set_title(TARGET_LABELS[target], fontsize=10)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("验证集（已用于选参）；轨迹等权；越低越好\nMLP柱为种子均值，点为各次训练；不表示置信区间", fontsize=11)


def export_figures(folder, data, predictions, summaries):
    from temp_scripts.plots import configure_style
    configure_style()
    dest = folder / "figures"
    dest.mkdir()
    fig = Figure(figsize=(14, 13), layout="constrained")
    summary_figure(fig, summaries)
    fig.savefig(dest / "validation_summary.png", dpi=140)
    fig.clear()
    examples = sorted(set(data.groups[data.splits == "validation"]))[:3]
    for tid in examples:
        for target in TARGET_LABELS:
            fig = Figure(figsize=(10, 4.5), layout="constrained")
            window_figure(fig, data.rows, data.y, predictions, target, "h1_linear", 0, tid)
            fig.savefig(dest / f"{tid}__{target}.png", dpi=140)
            fig.clear()
