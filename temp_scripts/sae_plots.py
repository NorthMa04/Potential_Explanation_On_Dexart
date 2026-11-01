"""Pure SAE figures shared by PNG export and the native result viewer."""
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def training_figure(fig, experiment, result):
    history = read_csv(experiment / "training_history.csv")
    fidelity = read_csv(experiment / "reconstruction_metrics.csv")
    fig.clear()
    axes = fig.subplots(1, 3)
    for name, label in (("train_mse_standardized", "train"), ("validation_mse_standardized", "validation")):
        axes[0].plot([int(r["epoch"]) for r in history], [max(float(r[name]), 1e-15) for r in history], label=label)
    axes[0].axvline(result["best_epoch"], color="#7758a6", ls="--", label="最佳检查点")
    axes[0].set(xlabel="训练轮数", ylabel="标准化激活 MSE（对数轴）", yscale="log", title="验证MSE选模；含初始化第0轮")
    axes[0].legend()
    for ax, quantity, label in ((axes[1], "action_mean", "未裁剪动作均值"),
                                (axes[2], "action_clipped", "裁剪后控制量")):
        selected = [r for r in fidelity if r["quantity"] == quantity and r["weighting"] == "trajectory"]
        ax.bar([r["split"] for r in selected], [float(r["mae"]) for r in selected], color=["#2563a6", "#16806a"])
        ax.set(ylabel="动作 MAE（轨迹等权）", title=label, ylim=(0, None))
    for ax in axes:
        ax.grid(True, axis="y", alpha=.3)
    fig.suptitle(result["experiment_id"] + " · 开发集；动作误差仅评价，不参与SAE训练")


def usage_figure(fig, experiment, result):
    rows = read_csv(experiment / "feature_usage.csv")
    fig.clear()
    axes = fig.subplots(2, 1, sharex=True)
    for ax, split in zip(axes, ("train", "validation")):
        selected = [r for r in rows if r["split"] == split]
        ax.bar([int(r["feature"]) for r in selected], [float(r["trajectory_weighted_frequency"]) for r in selected], color="#2563a6")
        health = result["health"][split]
        ax.set(ylabel="轨迹等权激活频率", ylim=(0, 1),
               title=f"{split}：未使用 {health['never_active_features']} 个；平均L0={health['mean_l0']:.2f}")
        ax.grid(True, axis="y", alpha=.3)
    axes[-1].set_xlabel("特征编号（不同种子间编号不代表同一含义）")
    fig.suptitle("本批样本中的使用覆盖；未使用不等于永久失活")


def window_figure(fig, run_folder, experiment, trajectory_id, dimension=0):
    rows = read_csv(run_folder / "selected_frames.csv")
    mask = np.array([r["trajectory_id"] == trajectory_id for r in rows])
    if not mask.any() or not 0 <= dimension < 22:
        raise ValueError("Unknown trajectory or action dimension")
    with np.load(run_folder / "selected_arrays.npz", allow_pickle=False) as data:
        original = data["h"][mask]
    with np.load(run_folder / "action_references.npz", allow_pickle=False) as data:
        original_mu, original_action = data["saved_mean"][mask], data["saved_action"][mask]
    with np.load(experiment / "reconstruction.npz", allow_pickle=False) as data:
        reconstructed, z, mu, action = (data[k][mask] for k in
                ("reconstructed_h", "z", "reconstructed_mean", "reconstructed_action"))
    x = np.array([int(r["state_index"]) for r in rows if r["trajectory_id"] == trajectory_id])
    fig.clear()
    axes = fig.subplots(4, 1, sharex=True)
    extent = [x[0] - .5, x[-1] + .5, -.5, original.shape[1] - .5]
    im = axes[0].imshow((reconstructed - original).T, origin="lower", aspect="auto", extent=extent,
                        cmap="coolwarm", vmin=-max(np.abs(reconstructed-original).max(), 1e-9),
                        vmax=max(np.abs(reconstructed-original).max(), 1e-9))
    fig.colorbar(im, ax=axes[0], label="重建 − 原激活")
    axes[0].set_ylabel("原激活坐标")
    im = axes[1].imshow(z.T, origin="lower", aspect="auto", extent=[x[0]-.5, x[-1]+.5, -.5, z.shape[1]-.5], cmap="viridis", vmin=0)
    fig.colorbar(im, ax=axes[1], label="SAE激活")
    axes[1].set_ylabel("全部 SAE 特征")
    for ax, ref, pred, label in ((axes[2], original_mu, mu, "未裁剪均值"),
                                 (axes[3], original_action, action, "实际裁剪控制量")):
        ax.plot(x, ref[:, dimension], label="原策略", marker=".")
        ax.plot(x, pred[:, dimension], label="重建激活接回actor", marker=".", ls="--")
        ax.set_ylabel(label)
        ax.grid(True, alpha=.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("动作前状态索引 t；终止状态无新激活/动作")
    fig.suptitle(f"{trajectory_id} · 动作维度 {dimension} · 连续窗口（离线前向，无仿真）")


def probe_figure(fig, run_folder, experiment, trajectory_id, layer):
    rows = read_csv(run_folder / "selected_frames.csv")
    mask = np.array([r["trajectory_id"] == trajectory_id for r in rows])
    x = [int(r["state_index"]) for r in rows if r["trajectory_id"] == trajectory_id]
    fig.clear()
    axes = fig.subplots(3, 1, sharex=True)
    path = experiment / "probe_predictions.npz"
    if not path.exists():
        axes[0].text(.5, .5, "本次选择了 skip-probes，没有拟合状态探针", ha="center", transform=axes[0].transAxes)
        return
    with np.load(run_folder / "selected_arrays.npz", allow_pickle=False) as data:
        y = data["y"][mask]
    with np.load(path, allow_pickle=False) as predictions:
        for col, (target, label) in enumerate((("distance_m", "距离 (m)"), ("opening_rad", "开启量 (rad)"), ("any_contact", "任一组接触概率"))):
            axes[col].plot(x, y[:, col], color="#111827", lw=2, label="保存的真实状态")
            for name, caption in ((layer + "_linear", "原始完整激活"), ("z_linear", "SAE全部特征"), ("z_top4", "SAE 4个特征")):
                key = f"{target}__{name}__seed0"
                if key in predictions:
                    axes[col].plot(x, predictions[key][mask], marker=".", label=caption)
            axes[col].set_ylabel(label)
            axes[col].grid(True, alpha=.3)
            axes[col].legend(loc="best", ncol=2)
    axes[-1].set_xlabel("动作前状态索引 t；预测的是当前状态")
    fig.suptitle(trajectory_id + " · 新线性状态探针；状态预测不等于动作贡献")


def export_figures(experiment, run_folder, result):
    from temp_scripts.plots import configure_style
    configure_style()
    figures = experiment / "figures"
    figures.mkdir()
    fig = Figure(figsize=(13, 5), layout="constrained")
    training_figure(fig, experiment, result)
    fig.savefig(figures / "training_and_action_fidelity.png", dpi=140)
    fig = Figure(figsize=(13, 6), layout="constrained")
    usage_figure(fig, experiment, result)
    fig.savefig(figures / "feature_usage.png", dpi=140)
    rows = read_csv(run_folder / "selected_frames.csv")
    tid = sorted({r["trajectory_id"] for r in rows if r["split"] == "validation"})[0]
    fig = Figure(figsize=(13, 10), layout="constrained")
    window_figure(fig, run_folder, experiment, tid)
    fig.savefig(figures / "validation_window_example.png", dpi=140)
    if result["probes"] is not None:
        fig = Figure(figsize=(13, 8), layout="constrained")
        layer = json.loads((run_folder / "configuration.json").read_text(encoding="utf-8"))["layer"]
        probe_figure(fig, run_folder, experiment, tid, layer)
        fig.savefig(figures / "validation_probe_example.png", dpi=140)
