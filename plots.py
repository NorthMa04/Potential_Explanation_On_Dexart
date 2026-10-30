"""Scientific figures shared by the native viewer and reproducible PNG exports."""
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

from data_access import CONTACT_LABELS, EVENT_LABELS

COLORS = {"blue": "#2563a6", "orange": "#b95c08", "green": "#16806a", "red": "#b63c51", "purple": "#7758a6", "muted": "#66758a"}
KEY_EVENTS = {"first_near", "first_any_contact", "first_strict_contact", "first_clear_opening", "first_success", "episode_end"}


def configure_style():
    rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "legend.fontsize": 8,
                     "figure.facecolor": "white", "axes.facecolor": "#fbfcfe", "axes.edgecolor": "#bec8d5",
                     "grid.color": "#d9e0e8", "grid.alpha": .6, "savefig.facecolor": "white"})


def title(traj):
    return "%s  |  %s  |  %s  |  %d 步 / %.2f s" % (traj.tid, traj.meta["split"], "最终成功" if traj.meta["trajectory_success"] == "True" else "最终未成功", traj.n, traj.n * traj.dt)


def setup(fig, nrows, traj, subtitle):
    fig.clear()
    axes = np.atleast_1d(fig.subplots(nrows, 1, sharex=True))
    fig.suptitle(title(traj) + "\n" + subtitle, fontsize=10)
    for ax in axes:
        ax.grid(True, axis="both", linewidth=.6)
    return axes


def mark_events(axes, traj, selected_event=None, all_events=False):
    for event in traj.events:
        j = event["state_index"]
        if j is None or (not all_events and event["event_type"] not in KEY_EVENTS):
            continue
        color = COLORS["red"] if event["event_type"] == "first_success" else COLORS["muted"]
        for ax in axes:
            ax.axvline(j * traj.dt, color=color, ls=":" if event["status"] != "observed" else "--", alpha=.32, lw=.8)
    if selected_event and selected_event["state_index"] is not None:
        j = selected_event["state_index"]
        for ax in axes:
            ax.axvline(j * traj.dt, color=COLORS["purple"], lw=1.6, alpha=.8)
            c = selected_event["confirmation_state_index"]
            if c is not None and c != j:
                ax.axvspan(j * traj.dt, c * traj.dt, color=COLORS["purple"], alpha=.09)


def finish_time(axes, traj, scope=None, selected_state=None):
    start, stop = scope or (0, traj.n)
    for ax in axes:
        ax.set_xlim(start * traj.dt - .25 * traj.dt, stop * traj.dt + .25 * traj.dt)
        if selected_state is not None:
            ax.axvline(selected_state * traj.dt, color="#151b27", lw=1.2, gid="state_cursor")
    axes[-1].set_xlabel("状态 / 动作发生时刻 (s)，黑线为当前选择")


def physical_figure(fig, traj, rules, selected_state=0, event=None, scope=None, all_events=False):
    axes = setup(fig, 5, traj, "状态 s_0…s_T；虚线为物理事件，紫色为选中事件。自动标签为 v1 初稿。")
    x = traj.time
    axes[0].plot(x, traj.state("palm_handle_distance"), color=COLORS["blue"], lw=1.8, label="手掌—把手")
    axes[0].axhline(rules["near_distance_m"], color=COLORS["orange"], ls=":", label="接近阈值")
    axes[0].set_ylabel("距离 (m)")
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].plot(x, traj.opening, color=COLORS["green"], lw=1.8, label="相对固定初始位置")
    axes[1].axhline(rules["clear_opening_rad"], color=COLORS["orange"], ls=":", label="明显开启阈值")
    axes[1].set_ylabel("开启量 (rad)")
    axes[1].legend(loc="upper right", ncol=2)
    dq = traj.arrays["physical__opening_delta_rad"]
    axes[2].bar(x[:-1], dq, width=traj.dt * .8, color=np.where(dq >= 0, COLORS["green"], COLORS["red"]), align="edge")
    axes[2].axhline(rules["moving_delta_rad_per_decision"], color=COLORS["muted"], ls=":")
    axes[2].axhline(0, color=COLORS["muted"], lw=.6)
    axes[2].set_ylabel("执行 a_t 后\nΔ开启量 (rad)")
    contacts = np.column_stack([traj.state("finger_palm_contact"), traj.state("loose_contact"), traj.state("contact")]).T
    axes[3].imshow(contacts, aspect="auto", interpolation="nearest", origin="upper", extent=[-.5 * traj.dt, (traj.n + .5) * traj.dt, 6.5, -.5], cmap=ListedColormap(["#edf1f6", COLORS["blue"]]), vmin=0, vmax=1)
    axes[3].set_yticks(range(7), CONTACT_LABELS, fontsize=8)
    axes[3].set_ylabel("接触组\n深色=接触")
    axes[3].grid(False)
    axes[4].plot(x, traj.state("progress"), color=COLORS["purple"], lw=1.8, label="环境 progress（未裁剪）")
    axes[4].plot(x[1:], traj.arrays["success_after"].astype(int), color=COLORS["red"], ls="--", label="环境成功标记")
    axes[4].set_ylabel("环境进度 /\n成功标记")
    axes[4].legend(loc="upper left", ncol=2)
    mark_events(axes, traj, event, all_events)
    finish_time(axes, traj, scope, selected_state)
    return axes


def heatmap_figure(fig, traj, selected_state=0, event=None, scope=None):
    axes = setup(fig, 4, traj, "动作前的原始向量；颜色为原始数值，不作语义命名。终止状态没有新激活或动作。")
    a = traj.arrays
    hmax = max(float(a["activations"].max()), float(a["activations_l2"].max()), 1e-8)
    rawmax = max(float(np.abs(a["action_mean_raw"]).max()), 1.)
    for ax, key, label, low, high, cmap in zip(axes,
            ["activations", "activations_l2", "action_mean_raw", "actions"],
            ["h1 神经元编号", "h2 神经元编号", "原始动作 μ 维度", "实际动作 a 维度"],
            [0, 0, -rawmax, -1], [hmax, hmax, rawmax, 1], ["viridis", "viridis", "RdBu_r", "RdBu_r"]):
        matrix = a[key]
        # Saved action rows occur at t, not at resulting state t+1.
        im = ax.imshow(matrix.T, origin="lower", aspect="auto", interpolation="nearest",
                       extent=[-.5 * traj.dt, (traj.n - .5) * traj.dt, -.5, matrix.shape[1] - .5],
                       vmin=low, vmax=high, cmap=cmap)
        ax.set_ylabel(label)
        ax.set_yticks([0, matrix.shape[1] // 2, matrix.shape[1] - 1])
        ax.grid(False)
        fig.colorbar(im, ax=ax, pad=.012, fraction=.025, label="数值")
        ax.axvspan((traj.n - .5) * traj.dt, (traj.n + .5) * traj.dt, facecolor="#e2e5eb", hatch="///", edgecolor="#c0c6d0")
    mark_events(axes, traj, event)
    finish_time(axes, traj, scope, selected_state)
    return axes


def snapshot_figure(fig, traj, selected_state):
    fig.clear()
    axes = np.atleast_1d(fig.subplots(3, 1))
    j = int(selected_state)
    rel = traj.relation(j)
    fig.suptitle(title(traj) + "\n选择 s_%d / %.3f s：距离 %.4f m，开启 %.4f rad，严格接触=%s" % (j, rel["time_s"], rel["distance_m"], rel["opening_rad"], rel["strict_contact"]), fontsize=10)
    if rel["policy"] is None:
        fig.clear()
        fig.suptitle(title(traj) + "\n选择终止状态 s_%d：距离 %.4f m，开启 %.4f rad" % (j, rel["distance_m"], rel["opening_rad"]), fontsize=10)
        ax = fig.add_subplot(111)
        ax.text(.5, .5, "这是终止状态 s_T\n\n存在物理状态，未采集 h_T / μ_T / a_T\n最后有效动作是第 %d 行\n\n可切换到物理状态页或空间关系页查看 s_T。" % (traj.n - 1), ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        return [ax]
    p = rel["policy"]
    hmax = max(float(traj.arrays["activations"].max()), float(traj.arrays["activations_l2"].max()), .01)
    for ax, name, color in zip(axes[:2], ["h1", "h2"], [COLORS["blue"], COLORS["green"]]):
        ax.bar(np.arange(64), p[name], color=color, width=.85)
        ax.set(xlabel="神经元编号", ylabel=name + " 原始激活", xlim=(-1, 64), ylim=(0, hmax * 1.12))
        ax.grid(axis="y", lw=.6)
    ax = axes[2]
    k = np.arange(22)
    ax.plot(k, p["mu_raw"], "o--", color=COLORS["orange"], markersize=3, label="原始均值 μ_t")
    ax.plot(k, p["action"], "s-", color=COLORS["blue"], markersize=3, label="实际 a_t=clip(μ_t)")
    ax.axhline(1, color=COLORS["muted"], ls=":")
    ax.axhline(-1, color=COLORS["muted"], ls=":")
    ax.set(xlabel="动作维度（顺序见来源说明）", ylabel="归一化动作数值", xticks=k)
    rawmax = max(float(np.abs(traj.arrays["action_mean_raw"]).max()), 1.) * 1.08
    ax.set_ylim(-rawmax, rawmax)
    ax.grid(axis="y", lw=.6)
    ax.legend(loc="upper right", ncol=2)
    return axes


def spatial_figure(fig, traj, selected_state):
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    j = int(selected_state)
    palm, handle, tips = [traj.state(k) for k in ["palm_position_world", "handle_position_world", "fingertip_position_world"]]
    fig.suptitle(title(traj) + "\n世界坐标的已记录点与轨迹；不是物体实体模型或仿真重放。鼠标拖动可旋转。", fontsize=10)
    ax.plot(*palm.T, color=COLORS["blue"], alpha=.25, label="手掌路径")
    ax.plot(*handle.T, color=COLORS["orange"], alpha=.3, label="把手路径")
    ax.scatter(*palm[j], color=COLORS["blue"], s=65, marker="s", label="当前手掌")
    ax.scatter(*handle[j], color=COLORS["orange"], s=80, marker="*", label="当前把手")
    ax.scatter(*tips[j].T, color=COLORS["green"], s=35, marker="o", label="四指尖")
    for k, pos in enumerate(tips[j]):
        ax.text(*pos, CONTACT_LABELS[k], fontsize=8)
    ax.plot(*np.vstack([palm[j], handle[j]]).T, color=COLORS["red"], ls="--", label="手掌—把手连线")
    cloud = np.vstack([palm, handle, tips.reshape(-1, 3)])
    lo, hi = cloud.min(0), cloud.max(0)
    center, radius = (lo + hi) / 2, max(float((hi - lo).max()) / 2 * 1.12, .03)
    ax.set(xlim=(center[0] - radius, center[0] + radius), ylim=(center[1] - radius, center[1] + radius), zlim=(center[2] - radius, center[2] + radius), xlabel="世界 X (m)", ylabel="世界 Y (m)", zlabel="世界 Z (m)")
    ax.set_box_aspect((1, 1, 1))
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("s_%d，距离 %.4f m" % (j, traj.state("palm_handle_distance")[j]), pad=6)
    return [ax]


def comparison_figure(fig, a, b, anchor="first_any_contact"):
    fig.clear()
    axes = np.atleast_1d(fig.subplots(5, 1, sharex=True))
    fig.suptitle("同物体轨迹对照（示例未做相似状态匹配，不能据此归因）\nA: %s   |   B: %s" % (a.tid, b.tid), fontsize=10)
    missing = []
    for traj, name, color, style in [(a, "A", COLORS["blue"], "-"), (b, "B", COLORS["orange"], "--")]:
        j = 0 if anchor == "start" else traj.event_anchor(anchor)
        if j is None:
            missing.append(name + " 未发生已确认的“" + EVENT_LABELS[anchor] + "”")
            continue
        x = traj.time - j * traj.dt
        outcome = "成功" if traj.meta["trajectory_success"] == "True" else "未成功"
        label = "%s (%s / %s)" % (name, outcome, traj.meta["split"])
        y = [traj.state("palm_handle_distance"), traj.opening, traj.state("contact").astype(int),
             np.linalg.norm(traj.arrays["activations"], axis=1),
             np.mean(np.abs(traj.arrays["action_mean_raw"] - traj.arrays["actions"]) > 1e-6, axis=1)]
        for ax, values in zip(axes, y):
            ax.plot(x[:len(values)], values, color=color, ls=style, lw=1.5, label=label, drawstyle="steps-post" if ax is axes[2] else "default")
    for ax, label in zip(axes, ["距离 (m)", "开启量 (rad)", "严格接触", "h1 L2 范数", "动作裁剪维数占比"]):
        ax.set_ylabel(label)
        ax.grid(lw=.6)
        ax.axvline(0, color=COLORS["muted"], ls=":")
    if axes[0].lines and axes[0].get_legend_handles_labels()[1]:
        axes[0].legend(loc="upper right", ncol=2)
    if missing:
        axes[0].text(.01, .03, "；".join(missing) + "，未伪造对齐点。", transform=axes[0].transAxes, color=COLORS["red"], fontsize=8)
    axes[-1].set_xlabel("相对“%s”的时间 (s)，负值为事件前" % ("轨迹起点" if anchor == "start" else EVENT_LABELS[anchor]))
    return axes
