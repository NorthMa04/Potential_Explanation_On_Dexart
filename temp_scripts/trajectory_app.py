"""Native Tk/Matplotlib viewer for Visual Studio; no server or browser required."""
import logging
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from temp_scripts.data_access import ALL_ROLES, DEV_ROLES, EVENT_LABELS, STATUS_LABELS
from temp_scripts.export_results import export_trajectory, export_comparison, save_review
from temp_scripts.plots import physical_figure, heatmap_figure, snapshot_figure, spatial_figure, comparison_figure, title
from temp_scripts.project_paths import OUTPUT_ROOT

TABS = ["物理状态与事件", "激活与动作热图", "当前步完整向量", "手—物体空间关系", "成功/失败轨迹对照", "来源、字段与规则"]
ANCHORS = {"轨迹起点": "start", "首次任一组接触": "first_any_contact", "首次明显开启": "first_clear_opening",
           "首次严格接触": "first_strict_contact", "严格接触保持期间连续开启": "contact_supported_opening"}


class ProjectToolbar(NavigationToolbar2Tk):
    """Route the toolbar save action through the project-managed exporter too."""
    def __init__(self, canvas, panel, save_callback):
        self.save_callback = save_callback
        super().__init__(canvas, panel, pack_toolbar=False)

    def save_figure(self, *args):
        self.save_callback()


class TrajectoryApp:
    def __init__(self, root, repo, run):
        self.root, self.repo, self.run = root, repo, Path(run)
        self.current, self.selected_event, self.scope = None, None, None
        self.state_index = 0
        self.pair = repo.example_pair()
        self.dirty = set(range(6))
        self.axes, self.figures, self.canvases = {}, {}, {}
        self.pending = None
        self.playing = False
        root.title("DexArt | 轨迹读取与物理事件审阅 | 实验步骤 1、2")
        w, h = min(1640, root.winfo_screenwidth() - 60), min(1020, root.winfo_screenheight() - 80)
        root.geometry("%dx%d+20+20" % (w, h))
        root.minsize(1120, 760)
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Treeview", rowheight=23)
        style.configure("Title.TLabel", font=("Microsoft YaHei", 11, "bold"))
        self._build()
        self._filter(prefer=self.pair[0] if self.pair else None)
        root.bind("<Left>", lambda e: self._keyboard_step(-1))
        root.bind("<Right>", lambda e: self._keyboard_step(1))

    def _build(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        side_container = ttk.Frame(self.root, width=338)
        side_container.grid(row=0, column=0, sticky="nsew")
        side_container.grid_propagate(False)
        side_container.rowconfigure(0, weight=1)
        side_container.columnconfigure(0, weight=1)
        side_canvas = tk.Canvas(side_container, highlightthickness=0, width=320)
        side_canvas.grid(row=0, column=0, sticky="nsew")
        side_scroll = ttk.Scrollbar(side_container, orient="vertical", command=side_canvas.yview)
        side_scroll.grid(row=0, column=1, sticky="ns")
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side = ttk.Frame(side_canvas, padding=10)
        side_window = side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", lambda e: side_canvas.configure(scrollregion=side_canvas.bbox("all")))
        side_canvas.bind("<Configure>", lambda e: side_canvas.itemconfigure(side_window, width=e.width))
        side.columnconfigure(0, weight=1)
        ttk.Label(side, text="轨迹 · 事件 · 每一步", style="Title.TLabel").grid(sticky="w")
        ttk.Label(side, text="默认仅浏览 train / validation\n离线数值记录，没有 RGB 视频", wraplength=290).grid(sticky="w", pady=(4, 8))
        filters = ttk.Frame(side)
        filters.grid(sticky="ew")
        filters.columnconfigure(1, weight=1)
        self.role = tk.StringVar(value="开发集 train + validation")
        self.obj = tk.StringVar(value="全部物体")
        self.outcome = tk.StringVar(value="全部结局")
        specs = [("数据划分", self.role, ["开发集 train + validation", *ALL_ROLES, "全部（含测试集）"]),
                 ("物体", self.obj, ["全部物体", *map(str, sorted({int(r["object_id"]) for r in self.repo.records}))]),
                 ("轨迹结局", self.outcome, ["全部结局", "最终成功", "最终未成功"])]
        for i, (label, var, values) in enumerate(specs):
            ttk.Label(filters, text=label).grid(row=i, column=0, sticky="w", padx=(0, 6), pady=2)
            combo = ttk.Combobox(filters, textvariable=var, values=values, state="readonly", width=24)
            combo.grid(row=i, column=1, sticky="ew", pady=2)
            combo.bind("<<ComboboxSelected>>", lambda e: self._filter())
        self.count = tk.StringVar()
        ttk.Label(side, textvariable=self.count).grid(sticky="w", pady=(5, 2))
        frame = ttk.Frame(side)
        frame.grid(sticky="nsew")
        frame.columnconfigure(0, weight=1)
        self.traj_tree = ttk.Treeview(frame, columns=("object", "ep", "outcome", "n"), show="headings", height=8, selectmode="browse")
        for name, label, width in [("object", "物体", 66), ("ep", "轨迹", 45), ("outcome", "结局", 65), ("n", "步数", 45)]:
            self.traj_tree.heading(name, text=label)
            self.traj_tree.column(name, width=width, anchor="center", stretch=True)
        self.traj_tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.traj_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.traj_tree.configure(yscrollcommand=scroll.set)
        self.traj_tree.bind("<<TreeviewSelect>>", self._on_trajectory)
        row = ttk.Frame(side)
        row.grid(sticky="ew", pady=4)
        ttk.Button(row, text="设为 A", command=lambda: self._set_pair(0)).pack(side="left")
        ttk.Button(row, text="设为 B", command=lambda: self._set_pair(1)).pack(side="left", padx=3)
        ttk.Button(row, text="成功/失败示例", command=self._example_pair).pack(side="left")
        ttk.Separator(side).grid(sticky="ew", pady=5)
        ttk.Label(side, text="物理事件（点击定位到事件状态 s_j）").grid(sticky="w")
        events = ttk.Frame(side)
        events.grid(sticky="nsew", pady=3)
        events.columnconfigure(0, weight=1)
        self.event_tree = ttk.Treeview(events, columns=("event", "j", "status"), show="headings", height=9, selectmode="browse")
        for name, label, width in [("event", "事件", 165), ("j", "s_j", 40), ("status", "标注", 65)]:
            self.event_tree.heading(name, text=label)
            self.event_tree.column(name, width=width, anchor="w", stretch=name == "event")
        self.event_tree.tag_configure("absent", foreground="#7c8590")
        self.event_tree.tag_configure("censored", foreground="#a45300")
        self.event_tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(events, orient="vertical", command=self.event_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.event_tree.configure(yscrollcommand=scroll.set)
        self.event_tree.bind("<<TreeviewSelect>>", self._on_event)
        self.event_tree.bind("<Double-1>", lambda e: self._focus_clip())
        row = ttk.Frame(side)
        row.grid(sticky="ew")
        ttk.Button(row, text="定位触发动作前", command=self._jump_trigger).pack(side="left")
        ttk.Button(row, text="只看事件片段", command=self._focus_clip).pack(side="left", padx=3)
        self.event_info = tk.StringVar(value="选择事件后，显示动作前 / 响应 / 确认行。")
        ttk.Label(side, textvariable=self.event_info, wraplength=295).grid(sticky="w", pady=4)
        self.show_all_events = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="状态图显示全部事件线", variable=self.show_all_events, command=self._redraw_current).grid(sticky="w")
        ttk.Separator(side).grid(sticky="ew", pady=5)
        ttk.Label(side, text="人工审阅（与自动标签分开保存）").grid(sticky="w")
        self.judgement = tk.StringVar(value="未审阅")
        ttk.Combobox(side, textvariable=self.judgement, values=["未审阅", "认可当前标注", "需要调整", "不确定"], state="readonly").grid(sticky="ew", pady=3)
        self.note = tk.StringVar()
        ttk.Entry(side, textvariable=self.note).grid(sticky="ew")
        ttk.Button(side, text="保存所选事件审阅意见", command=self._save_review).grid(sticky="ew", pady=4)
        body = ttk.Frame(self.root, padding=(0, 8, 8, 5))
        body.grid(row=0, column=1, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)
        self.header = tk.StringVar()
        ttk.Label(body, textvariable=self.header, style="Title.TLabel", wraplength=1180).grid(sticky="w", pady=(0, 3))
        tools = ttk.Frame(body)
        tools.grid(sticky="ew", pady=3)
        ttk.Button(tools, text="完整轨迹", command=self._full_scope).pack(side="left")
        ttk.Button(tools, text="导出当前轨迹/片段", command=self._export).pack(side="left", padx=4)
        ttk.Button(tools, text="打开输出目录", command=lambda: os.startfile(self.run)).pack(side="left")
        ttk.Label(tools, text="对照对齐：").pack(side="left", padx=(18, 2))
        self.anchor = tk.StringVar(value="首次任一组接触")
        box = ttk.Combobox(tools, textvariable=self.anchor, values=list(ANCHORS), state="readonly", width=23)
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda e: self._comparison_changed())
        ttk.Button(tools, text="导出对照图", command=self._export_comparison).pack(side="left", padx=4)
        self.notebook = ttk.Notebook(body)
        self.notebook.grid(sticky="nsew")
        for i, label in enumerate(TABS):
            panel = ttk.Frame(self.notebook)
            self.notebook.add(panel, text=label)
            if i == 5:
                self.sources = ScrolledText(panel, wrap="word", font=("Microsoft YaHei", 10), padx=12, pady=12)
                self.sources.pack(fill="both", expand=True)
                continue
            fig = Figure(figsize=(11, 8), layout="constrained")
            canvas = FigureCanvasTkAgg(fig, master=panel)
            toolbar = ProjectToolbar(canvas, panel, self._export_comparison if i == 4 else self._export)
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")
            canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            canvas.mpl_connect("button_press_event", lambda e, idx=i: self._plot_click(idx, e))
            self.figures[i], self.canvases[i] = fig, canvas
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self._render())
        control = ttk.Frame(body)
        control.grid(sticky="ew", pady=(6, 2))
        control.columnconfigure(5, weight=1)
        ttk.Button(control, text="◀ 上一步", command=lambda: self._step(-1)).grid(row=0, column=0)
        self.play_btn = ttk.Button(control, text="播放状态", command=self._toggle_play)
        self.play_btn.grid(row=0, column=1, padx=3)
        ttk.Button(control, text="下一步 ▶", command=lambda: self._step(1)).grid(row=0, column=2)
        ttk.Label(control, text="状态索引 s_j：").grid(row=0, column=3, padx=(12, 2))
        self.spin_var = tk.StringVar(value="0")
        self.spin = ttk.Spinbox(control, from_=0, to=250, width=6, textvariable=self.spin_var, command=self._spin)
        self.spin.grid(row=0, column=4)
        self.spin.bind("<Return>", lambda e: self._spin())
        self.slider_var = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(control, from_=0, to=250, variable=self.slider_var, command=self._slide)
        self.slider.grid(row=0, column=5, sticky="ew", padx=8)
        self.relation_text = tk.StringVar()
        ttk.Label(body, textvariable=self.relation_text, wraplength=1200).grid(sticky="ew", pady=3)
        self.status = tk.StringVar(value="输出目录：" + str(self.run))
        ttk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w").grid(row=1, column=0, columnspan=2, sticky="ew")

    def _roles(self):
        return DEV_ROLES if self.role.get().startswith("开发集") else ALL_ROLES if self.role.get().startswith("全部") else (self.role.get(),)

    def _filter(self, prefer=None):
        obj = None if self.obj.get() == "全部物体" else int(self.obj.get())
        outcome = None if self.outcome.get() == "全部结局" else self.outcome.get() == "最终成功"
        rows = self.repo.filter(self._roles(), obj, outcome)
        self.traj_tree.delete(*self.traj_tree.get_children())
        for row in rows:
            self.traj_tree.insert("", "end", iid=row["trajectory_id"], values=(row["object_id"], row["episode_id"], "成功" if row["trajectory_success"] == "True" else "未成功", row["decisions"]))
        self.count.set("筛选结果：%d 条；共有 %d 条完整轨迹" % (len(rows), len(self.repo.records)))
        if rows:
            tid = prefer if prefer in [r["trajectory_id"] for r in rows] else rows[0]["trajectory_id"]
            self.traj_tree.selection_set(tid)
            self.traj_tree.see(tid)
            self._load(tid)
        else:
            self.current, self.selected_event = None, None
            self.event_tree.delete(*self.event_tree.get_children())
            self.header.set("当前筛选没有轨迹")
            self.relation_text.set("请调整左侧物体、划分或结局筛选。")
            self.sources.configure(state="normal")
            self.sources.delete("1.0", "end")
            self.sources.insert("1.0", "当前筛选没有轨迹。")
            self.sources.configure(state="disabled")
            for i, fig in self.figures.items():
                fig.clear()
                ax = fig.add_subplot(111)
                ax.text(.5, .5, "当前筛选没有轨迹", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                self.canvases[i].draw_idle()
            self.status.set("当前筛选没有轨迹。")

    def _on_trajectory(self, event=None):
        selected = self.traj_tree.selection()
        if selected:
            self._load(selected[0])

    def _load(self, tid):
        if self.current is not None and self.current.tid == tid:
            return
        try:
            self.current = self.repo.load(tid)
            self.playing = False
            self.play_btn.configure(text="播放状态")
            self.scope, self.selected_event, self.state_index = None, None, 0
            self.header.set(title(self.current))
            self.event_tree.delete(*self.event_tree.get_children())
            for e in self.current.events:
                label = "严格接触中连续开启" if e["event_type"] == "contact_supported_opening" else EVENT_LABELS[e["event_type"]]
                self.event_tree.insert("", "end", iid=e["event_id"], values=(label, e["state_index"] if e["state_index"] is not None else "—", STATUS_LABELS[e["status"]]), tags=("absent" if e["status"] == "not_observed" else "censored" if e["status"] == "right_censored" else "observed",))
            self.slider.configure(from_=0, to=self.current.n)
            self.spin.configure(from_=0, to=self.current.n)
            self.event_info.set("事件按状态时间排序；未发生事件没有伪造时刻。")
            self.dirty = set(range(6))
            self._set_state(0)
        except Exception as exc:
            self._error(exc)

    def _on_event(self, unused=None):
        if self.current is None or not self.event_tree.selection():
            return
        eid = self.event_tree.selection()[0]
        self.selected_event = next(e for e in self.current.events if e["event_id"] == eid)
        e = self.selected_event
        self.event_info.set("%s：%s\n事件 s_%s；动作前 h/a 行 %s；响应 h 行 %s；确认 s_%s。" % (
            EVENT_LABELS[e["event_type"]], STATUS_LABELS[e["status"]], self._display(e["state_index"]), self._display(e["trigger_action_index"]), self._display(e["response_action_index"]), self._display(e["confirmation_state_index"])))
        self.dirty.update([0, 1, 5])
        self.scope = None
        self.slider.configure(from_=0, to=self.current.n)
        self._set_state(e["state_index"] if e["state_index"] is not None else self.state_index)

    @staticmethod
    def _display(value):
        return "无" if value is None else str(value)

    def _jump_trigger(self):
        if self.selected_event and self.selected_event["trigger_action_index"] is not None:
            self._set_state(self.selected_event["trigger_action_index"])
        else:
            self.status.set("此事件未发生或属于初始状态，没有可定位的前一动作。")

    def _focus_clip(self):
        if not self.selected_event or not self.selected_event["segment_id"]:
            self.status.set("请选择存在实际片段的事件；未发生事件没有片段。")
            return
        s = self.current.segments[self.selected_event["segment_id"]]
        self.scope = s["action_start_inclusive"], s["action_stop_exclusive"]
        self.slider.configure(from_=self.scope[0], to=self.scope[1])
        self.dirty.update([0, 1])
        self._set_state(self.selected_event["state_index"])
        self.status.set("片段动作 [%d,%d)，状态 [%d,%d]；当前选择 s_%d。" % (*self.scope, *self.scope, self.state_index))

    def _full_scope(self):
        if self.current:
            self.scope = None
            self.slider.configure(from_=0, to=self.current.n)
            self.dirty.update([0, 1])
            self.status.set("完整轨迹；当前 s_%d。输出目录：%s" % (self.state_index, self.run))
            self._render()

    def _slide(self, value):
        if self.current is None:
            return
        if self.pending:
            self.root.after_cancel(self.pending)
        self.pending = self.root.after(70, lambda: self._set_state(round(float(value))))

    def _spin(self):
        try:
            self._set_state(int(self.spin_var.get()))
        except ValueError:
            self.spin_var.set(str(self.state_index))

    def _keyboard_step(self, delta):
        if not isinstance(self.root.focus_get(), (ttk.Entry, ttk.Spinbox, ScrolledText)):
            self._step(delta)

    def _step(self, delta):
        if self.current:
            self._set_state(self.state_index + delta)

    def _set_state(self, j):
        self.pending = None
        if self.current is None:
            return
        lo, hi = self.scope or (0, self.current.n)
        self.state_index = max(lo, min(hi, int(j)))
        self.slider_var.set(self.state_index)
        self.spin_var.set(str(self.state_index))
        if self.scope:
            self.status.set("片段动作 [%d,%d)，状态 [%d,%d]；当前 s_%d。" % (*self.scope, *self.scope, self.state_index))
        else:
            self.status.set("完整轨迹；当前 s_%d。输出目录：%s" % (self.state_index, self.run))
        r = self.current.relation(self.state_index)
        if r["policy"] is None:
            self.relation_text.set("s_%d（%.3f s）是终止状态；距离 %.4f m，开启 %.4f rad。最后有效动作 a_%d；没有 h_%d / a_%d。" % (self.state_index, r["time_s"], r["distance_m"], r["opening_rad"], self.current.n - 1, self.state_index, self.state_index))
        else:
            self.relation_text.set("s_%d → h1_%d / h2_%d → μ_%d → clip(μ)=a_%d → s_%d  |  %.3f s，距离 %.4f m，开启 %.4f rad，严格接触=%s。" % (self.state_index, self.state_index, self.state_index, self.state_index, self.state_index, self.state_index + 1, r["time_s"], r["distance_m"], r["opening_rad"], r["strict_contact"]))
        self.dirty.update([2, 3])
        self._render()

    def _plot_click(self, index, event):
        if index in (0, 1) and self.current and event.inaxes in self.axes.get(index, []) and event.xdata is not None:
            if self.canvases[index].toolbar.mode:
                return
            self._set_state(round(event.xdata / self.current.dt))

    def _redraw_current(self):
        self.dirty.update([0, 1])
        self._render()

    def _render(self):
        if self.current is None:
            return
        i = self.notebook.index(self.notebook.select())
        try:
            if i not in self.dirty:
                if i in (0, 1):
                    for ax in self.axes[i]:
                        for line in ax.lines:
                            if line.get_gid() == "state_cursor":
                                line.set_xdata([self.state_index * self.current.dt] * 2)
                    self.canvases[i].draw_idle()
                return
            if i == 0:
                self.axes[i] = physical_figure(self.figures[i], self.current, self.repo.rules, self.state_index, self.selected_event, self.scope, self.show_all_events.get())
            elif i == 1:
                self.axes[i] = heatmap_figure(self.figures[i], self.current, self.state_index, self.selected_event, self.scope)
            elif i == 2:
                self.axes[i] = snapshot_figure(self.figures[i], self.current, self.state_index)
            elif i == 3:
                previous = self.axes.get(i, [])
                view = (previous[0].elev, previous[0].azim, previous[0].roll) if previous else None
                self.axes[i] = spatial_figure(self.figures[i], self.current, self.state_index)
                if view:
                    self.axes[i][0].view_init(elev=view[0], azim=view[1], roll=view[2])
            elif i == 4:
                if not self.pair:
                    fig = self.figures[i]
                    fig.clear()
                    fig.add_subplot(111).text(.5, .5, "在左侧设置同物体 A、B 两条轨迹。", ha="center")
                else:
                    self.axes[i] = comparison_figure(self.figures[i], *[self.repo.load(tid) for tid in self.pair], ANCHORS[self.anchor.get()])
            else:
                self._sources()
            if i != 5:
                self.canvases[i].draw_idle()
            self.dirty.discard(i)
        except Exception as exc:
            self._error(exc)

    def _sources(self):
        t = self.current
        e = self.selected_event
        text = "数据来源与证据边界\n\n索引目录：%s\n当前数值轨迹：%s\n原始观测/点云所在文件：%s\n原始 SHA256：%s\n事件规则 SHA256：%s\n\n输出目录：%s\n\n当前轨迹 %s\n划分：%s；物体 %s；seed %s；步数 %d；控制间隔 %.9f s。\n\n状态：T+1 行；激活/动作：T 行。终止状态没有响应激活，未发生事件没有伪造关键帧。\n原始动作范围为归一化 [-1,1]；不是关节位移。\n\n维度顺序：\n%s\n\n" % (self.repo.root, t.meta["numeric_archive"], t.meta["source_npz"], t.meta["source_sha256"], self.repo.provenance["rules_sha256"], self.run, t.tid, t.meta["split"], t.meta["object_id"], t.meta["seed"], t.n, t.dt, ", ".join(t.provenance["robot_joint_names"]))
        text += "物理事件规则（固定 v1 初稿）：\n"
        text += "接近距离 ≤%.3f m；相对固定初始关节位置开启 ≥%.3f rad；连续 %d 步判断持续、停滞或回退。\n" % (self.repo.rules["near_distance_m"], self.repo.rules["clear_opening_rad"], self.repo.rules["run_decisions"])
        text += "首次任一组接触与首次严格接触分别标注；严格接触要求至少两个手指组及手掌。\n连续事件确认需要未来状态，只用于离线标注，不是动作前输入。\n人工审阅只保存意见，不修改现有事件、划分或阈值。\n\n"
        text += "学习输入：h-only SAE 只接收一层当前激活；当前状态可作辅助目标。post_*、success_after、结局、事件标签等不能混作编码器输入。\n\n空间图只展示已记录的世界坐标点与路径，不包含物体实体模型或 RGB，也没有执行仿真。\n激活与状态同时出现不证明语义或因果作用；本程序没有训练或干预模型。\n对照示例按同物体和结局选择，未做相似物理状态匹配，不能据此解释成功/失败原因。\n\n"
        if e:
            text += "选中事件完整记录：\n" + "\n".join("%s: %s" % (k, v) for k, v in e.items())
        self.sources.configure(state="normal")
        self.sources.delete("1.0", "end")
        self.sources.insert("1.0", text)
        self.sources.configure(state="disabled")

    def _set_pair(self, index):
        if self.current is None:
            return
        values = list(self.pair) if self.pair else [self.current.tid] * 2
        other = self.repo.by_id[values[1 - index]]
        if other["object_id"] != self.current.meta["object_id"]:
            # Reset both when moving to another object; no silent cross-object pair.
            values = [self.current.tid] * 2
        values[index] = self.current.tid
        self.pair = tuple(values)
        self._comparison_changed()
        self.status.set("A=%s；B=%s（同物体示例，未做相似状态匹配）。" % self.pair)

    def _example_pair(self):
        obj = None if self.obj.get() == "全部物体" else int(self.obj.get())
        pair = self.repo.example_pair(self._roles(), obj)
        if not pair:
            self.status.set("当前物体/划分没有同时包含成功和失败的轨迹，不伪造对照。")
            return
        self.pair = pair
        self.outcome.set("全部结局")
        self._filter(prefer=pair[0])
        self._comparison_changed()
        self.notebook.select(4)

    def _comparison_changed(self):
        self.dirty.add(4)
        self._render()

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.configure(text="暂停" if self.playing else "播放状态")
        if self.playing:
            self._tick()

    def _tick(self):
        if not self.playing or self.current is None:
            return
        _, end = self.scope or (0, self.current.n)
        if self.state_index >= end:
            self.playing = False
            self.play_btn.configure(text="播放状态")
            return
        self._step(1)
        self.root.after(max(40, round(self.current.dt * 1000)), self._tick)

    def _save_review(self):
        if self.current is None:
            return
        try:
            path = save_review(self.run, self.current, self.selected_event, self.judgement.get(), self.note.get())
            self.status.set("已保存人工意见：%s；自动规则未修改。" % path)
        except Exception as exc:
            self._error(exc)

    def _export(self):
        if self.current is None:
            return
        try:
            self.status.set("正在导出四类图表和同步数值，请稍候…")
            self.root.update_idletasks()
            path = export_trajectory(self.repo, self.current, self.run, self.state_index, self.selected_event, self.scope)
            self.status.set("已导出：" + str(path))
        except Exception as exc:
            self._error(exc)

    def _export_comparison(self):
        if not self.pair:
            return
        try:
            self.root.update_idletasks()
            path = export_comparison(self.repo, *[self.repo.load(tid) for tid in self.pair], self.run, ANCHORS[self.anchor.get()])
            self.status.set("已导出对照：" + str(path))
        except Exception as exc:
            self._error(exc)

    def _error(self, exc):
        logging.exception("Viewer action failed")
        self.status.set("出错：" + str(exc))
        messagebox.showerror("DexArt 可视化", str(exc), parent=self.root)


def launch(repo, run):
    root = tk.Tk()
    TrajectoryApp(root, repo, run)
    root.mainloop()
