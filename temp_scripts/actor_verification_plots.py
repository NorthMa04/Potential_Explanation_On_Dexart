"""Scientific plots for complete numerical checks; sampling affects scatter marks only."""
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from temp_scripts.actor_verification import COMPARISONS, load_replayed
from temp_scripts.plots import COLORS


def reference_for(data, comparison):
    suffix = comparison.split("_", 1)[1]
    return data["reference_" + suffix]


def result_meta(data):
    return json.loads(str(data["metadata_json"]))["trajectory"]


def global_figure(fig, summary):
    fig.clear()
    axes = fig.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
    rows = summary["comparisons"]
    k = np.arange(len(rows))
    names = [r["comparison"] for r in rows]
    fig.suptitle("冻结Actor前向复现：%s | %d条轨迹 / %d步\n统计使用全部数值；A/B/C为不同入口的前向计算，D为原日志裁剪核验" % (summary["status"], summary["trajectories"], summary["decision_steps"]), fontsize=11)
    axes[0].bar(k - .18, [r["mae"] for r in rows], width=.36, label="MAE", color=COLORS["blue"])
    axes[0].bar(k + .18, [r["max_abs_error"] for r in rows], width=.36, label="最大绝对误差", color=COLORS["orange"])
    axes[0].set_yscale("symlog", linthresh=1e-9)
    axes[0].set(ylabel="数值误差（symlog）", xticks=k, xticklabels=names)
    axes[0].legend(loc="upper right", ncol=2)
    for x, r in zip(k, rows):
        axes[0].annotate("%.2g" % r["max_abs_error"], (x + .18, r["max_abs_error"]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    axes[1].bar(k, [r["failed_values"] for r in rows], color=[COLORS["green"] if r["passed"] else COLORS["red"] for r in rows])
    for x, r in zip(k, rows):
        axes[1].text(x, r["failed_values"], str(r["failed_values"]), ha="center", va="bottom", fontsize=9)
    axes[1].set(ylabel="超过固定容差的数值数量", xlabel="比较项（完整定义见报告）", xticks=k, xticklabels=names, ylim=(0, max(1, max(r["failed_values"] for r in rows) * 1.2)))
    for ax in axes:
        ax.grid(axis="y", alpha=.3)
    return axes


def scatter_figure(fig, bank):
    fig.clear()
    axes = fig.subplots(2, 2).ravel()
    fig.suptitle("C：融合特征出发，重新计算 vs 已保存记录\n散点按完整展开数组均匀索引抽样，每项最多5000点；误差指标未抽样", fontsize=11)
    for ax, name in zip(axes, ["C_h1", "C_h2", "C_mu", "C_action"]):
        ref, pred = [np.concatenate(values) for values in bank[name]]
        indices = np.linspace(0, ref.size - 1, min(ref.size, 5000), dtype=int)
        ax.scatter(ref[indices], pred[indices], s=3, alpha=.22, color=COLORS["blue"], rasterized=True)
        low, high = min(float(ref.min()), float(pred.min())), max(float(ref.max()), float(pred.max()))
        pad = max((high - low) * .04, 1e-6)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], "--", lw=1, color=COLORS["red"], label="y=x")
        ax.set(title=name, xlabel="已保存记录", ylabel="重新计算", xlim=(low - pad, high + pad), ylim=(low - pad, high + pad))
        ax.grid(alpha=.3)
        ax.legend(loc="upper left")
    return axes


def trajectory_error_figure(fig, data, comparison, config):
    fig.clear()
    axes = fig.subplots(3, 1, sharex=True)
    pred, ref = data[comparison].astype(float), reference_for(data, comparison).astype(float)
    error = np.abs(pred - ref)
    failed = error > config["atol"] + config["rtol"] * np.abs(ref)
    meta = result_meta(data)
    x = data["time_s"]
    fig.suptitle("%s | %s\n%s；最后一个动作后为终止状态，没有额外激活" % (meta["trajectory_id"], meta["split"], COMPARISONS[comparison][1]), fontsize=10)
    axes[0].plot(x, error.max(axis=1), color=COLORS["red"], label="每步最大绝对误差")
    axes[0].plot(x, error.mean(axis=1), color=COLORS["blue"], label="每步MAE")
    axes[0].set_ylabel("绝对误差")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].step(x, failed.sum(axis=1), where="mid", color=COLORS["orange"])
    axes[1].set(ylabel="超差维数", ylim=(-.1, max(1, int(failed.sum(axis=1).max()) * 1.1)))
    axes[2].plot(x, data["opening_rad"], color=COLORS["green"])
    axes[2].set(ylabel="动作前开启量 (rad)", xlabel="动作前时刻 t (s)")
    for ax in axes:
        ax.grid(alpha=.3)
    return axes


def values_figure(fig, data, comparison, dimension, config):
    fig.clear()
    axes = fig.subplots(2, 1, sharex=True)
    pred, ref = data[comparison][:, dimension].astype(float), reference_for(data, comparison)[:, dimension].astype(float)
    x = data["time_s"]
    tolerance = config["atol"] + config["rtol"] * np.abs(ref)
    fig.suptitle("%s | %s | 维度 %d（从0编号）" % (result_meta(data)["trajectory_id"], comparison, dimension), fontsize=10)
    axes[0].plot(x, ref, color=COLORS["blue"], lw=2, label="已保存记录")
    axes[0].plot(x, pred, color=COLORS["orange"], ls="--", label="重新计算")
    axes[0].set_ylabel("原始数值")
    axes[0].legend(ncol=2)
    axes[1].plot(x, pred - ref, color=COLORS["purple"], label="重算 - 记录")
    axes[1].plot(x, tolerance, color=COLORS["muted"], ls=":", label="该维度允许误差 ±(atol+rtol·|ref|)")
    axes[1].plot(x, -tolerance, color=COLORS["muted"], ls=":")
    axes[1].set(ylabel="有符号误差", xlabel="动作前时刻 t (s)")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.grid(alpha=.3)
    return axes


def error_heatmap_figure(fig, data, comparison):
    fig.clear()
    ax = fig.add_subplot(111)
    delta = data[comparison].astype(float) - reference_for(data, comparison).astype(float)
    maximum = float(np.abs(delta).max())
    scale = max(maximum, 1e-12)
    x = data["time_s"]
    dt = float(x[1] - x[0]) if len(x) > 1 else .04
    im = ax.imshow(delta.T, aspect="auto", origin="lower", interpolation="nearest", cmap="RdBu_r", vmin=-scale, vmax=scale,
                   extent=[x[0] - dt / 2, x[-1] + dt / 2, -.5, delta.shape[1] - .5])
    fig.suptitle("%s | %s\n有符号误差=重算-记录，真实最大绝对误差 %.6g；色标自动对称缩放" % (result_meta(data)["trajectory_id"], comparison, maximum), fontsize=10)
    ax.set(xlabel="动作前时刻 t (s)", ylabel="神经元 / 动作维度编号")
    fig.colorbar(im, ax=ax, label="有符号数值误差")
    return [ax]


def clipping_figure(fig, data, action_index):
    fig.clear()
    axes = fig.subplots(2, 1, sharex=True)
    t = int(action_index)
    k = np.arange(22)
    fig.suptitle("%s | 动作行 %d\n原始均值与实际裁剪动作分开核对；这里未做干预" % (result_meta(data)["trajectory_id"], t), fontsize=10)
    for ax, suffix, label in zip(axes, ["mu", "action"], ["原始均值 μ", "裁剪后实际动作 a"]):
        ax.plot(k, data["reference_" + suffix][t], "o-", color=COLORS["blue"], markersize=4, label="已保存记录")
        ax.plot(k, data["C_" + suffix][t], "x--", color=COLORS["orange"], markersize=5, label="融合特征重新计算")
        ax.axhline(1, color=COLORS["muted"], ls=":")
        ax.axhline(-1, color=COLORS["muted"], ls=":")
        ax.set_ylabel(label)
        ax.grid(alpha=.3)
        ax.legend(ncol=2)
    axes[-1].set(xlabel="动作维度（从0编号；不是相同数值的关节位移）", xticks=k)
    return axes


def export_figures(folder, summary, bank, repo, records):
    output = Path(folder) / "figures"
    output.mkdir(exist_ok=True)
    for name, function, args in [("01_global_errors.png", global_figure, (summary,)), ("02_replayed_vs_recorded.png", scatter_figure, (bank,))]:
        fig = Figure(figsize=(12, 8), layout="constrained")
        function(fig, *args)
        fig.savefig(output / name, dpi=150)
        fig.clear()
    selected = {r["trajectory_id"] for r in records}
    pair = repo.example_pair()
    examples = [tid for tid in (pair or ()) if tid in selected] or [records[0]["trajectory_id"]]
    for tid in examples:
        data = load_replayed(folder, tid)
        error = np.abs(data["C_mu"].astype(float) - data["reference_mu"].astype(float))
        t, k = np.unravel_index(int(np.argmax(error)), error.shape)
        cases = [("03_time_errors", trajectory_error_figure, (data, "C_mu", summary["configuration"])),
                 ("04_values_dimension", values_figure, (data, "C_mu", int(k), summary["configuration"])),
                 ("05_error_heatmap", error_heatmap_figure, (data, "C_mu")),
                 ("06_clipping", clipping_figure, (data, int(t)))]
        for name, function, args in cases:
            fig = Figure(figsize=(12, 8), layout="constrained")
            function(fig, *args)
            fig.savefig(output / (tid + "_" + name + ".png"), dpi=150)
            fig.clear()
