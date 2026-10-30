"""Native result viewer; launched after verification, or via --view-only."""
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from actor_verification import COMPARISONS, load_replayed
from actor_verification_plots import global_figure, trajectory_error_figure, values_figure, error_heatmap_figure, clipping_figure
from project_paths import output_path


class ManagedToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, parent, save_callback):
        self.save_callback = save_callback
        super().__init__(canvas, parent, pack_toolbar=False)

    def save_figure(self, *args):
        self.save_callback()


class ActorVerificationViewer:
    def __init__(self, root, folder):
        self.root, self.folder = root, Path(folder).resolve()
        self.summary = json.loads((self.folder / "summary.json").read_text(encoding="utf-8"))
        self.sources = json.loads((self.folder / "sources.json").read_text(encoding="utf-8"))
        self.ids = [row["trajectory_id"] for row in self.sources]
        if not self.ids:
            raise ValueError("结果目录没有轨迹")
        preferred = "object_9996__episode_00004"
        first = preferred if preferred in self.ids else next((r["trajectory_id"] for r in self.sources if r["split"] in ("train", "validation")), self.ids[0])
        self.trajectory = tk.StringVar(value=first)
        self.comparison = tk.StringVar(value="C_mu")
        self.dimension, self.action_index = tk.StringVar(value="0"), tk.StringVar(value="0")
        self.figures, self.canvases = {}, {}
        self.dirty = set(range(6))
        root.title("DexArt | 步骤3：冻结Actor前向复现")
        root.geometry("%dx%d" % (min(1500, root.winfo_screenwidth() - 70), min(950, root.winfo_screenheight() - 80)))
        root.minsize(1050, 720)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)
        banner = ttk.Frame(root, padding=8)
        banner.grid(sticky="ew")
        text = "%s  |  %d 条轨迹 / %d 步  |  超差数值 %d  |  原权重不变=%s / 钩子中立=%s" % (
            self.summary["status"].upper(), self.summary["trajectories"], self.summary["decision_steps"], self.summary["failed_values"],
            self.summary["weights_unchanged"], self.summary["hooks_neutral_and_removed"])
        banner_label = ttk.Label(banner, text=text, font=("Microsoft YaHei", 11, "bold"), wraplength=900, foreground="#16704a" if self.summary["status"] == "passed" else "#b43143")
        banner_label.pack(side="left")
        ttk.Button(banner, text="打开结果目录", command=lambda: os.startfile(self.folder)).pack(side="right")
        controls = ttk.Frame(root, padding=(8, 0, 8, 4))
        controls.grid(sticky="ew")
        ttk.Label(controls, text="轨迹：").pack(side="left")
        box = ttk.Combobox(controls, textvariable=self.trajectory, values=self.ids, width=43, state="readonly")
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda e: self._load())
        ttk.Label(controls, text="比较项：").pack(side="left", padx=(14, 0))
        box = ttk.Combobox(controls, textvariable=self.comparison, values=list(COMPARISONS), width=12, state="readonly")
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda e: self._selection_changed())
        ttk.Label(controls, text="维度：").pack(side="left", padx=(14, 0))
        self.dimension_spin = ttk.Spinbox(controls, from_=0, to=21, textvariable=self.dimension, width=5, command=self._controls_changed)
        self.dimension_spin.pack(side="left")
        self.dimension_spin.bind("<Return>", lambda e: self._controls_changed())
        ttk.Label(controls, text="动作行：").pack(side="left", padx=(14, 0))
        self.step_spin = ttk.Spinbox(controls, from_=0, to=1, textvariable=self.action_index, width=6, command=self._controls_changed)
        self.step_spin.pack(side="left")
        self.step_spin.bind("<Return>", lambda e: self._controls_changed())
        ttk.Button(controls, text="保存当前图", command=self.save_figure).pack(side="left", padx=10)
        self.caption = tk.StringVar()
        caption_label = ttk.Label(root, textvariable=self.caption, padding=(8, 2), wraplength=1450)
        caption_label.grid(sticky="w")
        root.bind("<Configure>", lambda event: (caption_label.configure(wraplength=max(500, root.winfo_width() - 25)), banner_label.configure(wraplength=max(500, root.winfo_width() - 160))) if event.widget is root else None)
        self.notebook = ttk.Notebook(root)
        self.notebook.grid(sticky="nsew", padx=6)
        for i, label in enumerate(["全局误差", "重算与记录散点", "逐时刻误差", "指定维度对照", "误差热图", "原始动作与裁剪", "来源与判定说明"]):
            panel = ttk.Frame(self.notebook)
            self.notebook.add(panel, text=label)
            if i == 6:
                self.text = ScrolledText(panel, wrap="word", font=("Microsoft YaHei", 10), padx=12, pady=12)
                self.text.pack(fill="both", expand=True)
                continue
            fig = Figure(figsize=(12, 8), layout="constrained")
            canvas = FigureCanvasTkAgg(fig, master=panel)
            toolbar = ManagedToolbar(canvas, panel, self.save_figure)
            toolbar.pack(side="bottom", fill="x")
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.figures[i], self.canvases[i] = fig, canvas
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self._render())
        self.status = tk.StringVar(value="所有保存均写入本项目 outputs/。终止状态没有策略响应，未填零参加比较。")
        ttk.Label(root, textvariable=self.status, relief="sunken", anchor="w").grid(sticky="ew", pady=(4, 0))
        self._load()

    def _load(self):
        self.data = load_replayed(self.folder, self.trajectory.get())
        self.step_spin.configure(to=len(self.data["time_s"]) - 1)
        self._selection_changed()

    def _selection_changed(self):
        name = self.comparison.get()
        suffix = name.split("_", 1)[1]
        error = np.abs(self.data[name].astype(float) - self.data["reference_" + suffix].astype(float))
        t, k = np.unravel_index(int(np.argmax(error)), error.shape)
        self.dimension_spin.configure(to=error.shape[1] - 1)
        self.dimension.set(str(k))
        self.action_index.set(str(t))
        self.caption.set(COMPARISONS[name][1] + "；默认定位该比较的最大误差处，仅用于检查数值一致性。" +
                         "  当前轨迹：" + json.loads(str(self.data["metadata_json"]))["trajectory"]["split"])
        self.dirty.update([2, 3, 4, 5])
        self._render()

    def _controls_changed(self):
        try:
            k = max(0, min(self.data[self.comparison.get()].shape[1] - 1, int(self.dimension.get())))
            t = max(0, min(len(self.data["time_s"]) - 1, int(self.action_index.get())))
            self.dimension.set(str(k))
            self.action_index.set(str(t))
            self.dirty.update([3, 5])
            self._render()
        except ValueError:
            self.status.set("维度和动作行需填写整数。")

    def _render(self):
        if not hasattr(self, "data"):
            return
        i = self.notebook.index(self.notebook.select())
        config, name = self.summary["configuration"], self.comparison.get()
        if i == 6:
            info = "当前结果目录：%s\n\n" % self.folder
            info += "逐元素判定：abs(replayed-reference) <= atol + rtol*abs(reference)。\n当前atol=%g，rtol=%g。阈值在前向计算前固定，未根据结果调宽。\n\n" % (config["atol"], config["rtol"])
            info += "每一段都沿输入到输出方向计算。A从h2开始，B从h1开始，C从融合特征开始，D单独核对原始裁剪。\n只恢复actor后半部分，不包含PointNet或物理仿真。激活为ReLU之后，动作使用确定性均值。\n\n"
            info += "全部误差指标使用全部选中轨迹和数值。散点图最多抽样5000点，其余图不抽时间步。终止状态无激活/动作，另存terminal_states.csv。最大误差示例用于排查恢复错误，不用于挑选语义特征。\n\n"
            info += "配置及权重来源：\n" + json.dumps(config, ensure_ascii=False, indent=2)
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            self.text.insert("1.0", info)
            self.text.configure(state="disabled")
            return
        if i not in self.dirty:
            return
        fig = self.figures[i]
        if i == 0:
            global_figure(fig, self.summary)
        elif i == 1:
            fig.clear()
            path = self.folder / "figures" / "02_replayed_vs_recorded.png"
            if path.exists():
                ax = fig.add_subplot(111)
                ax.imshow(imread(path))
                ax.axis("off")
            else:
                fig.add_subplot(111).text(.5, .5, "此运行没有导出散点图。", ha="center")
        elif i == 2:
            trajectory_error_figure(fig, self.data, name, config)
        elif i == 3:
            values_figure(fig, self.data, name, int(self.dimension.get()), config)
        elif i == 4:
            error_heatmap_figure(fig, self.data, name)
        elif i == 5:
            clipping_figure(fig, self.data, int(self.action_index.get()))
        self.canvases[i].draw_idle()
        self.dirty.discard(i)

    def save_figure(self):
        try:
            i = self.notebook.index(self.notebook.select())
            if i == 6:
                self.status.set("来源与判定说明已保存在 configuration.json。")
                return
            path = output_path(self.folder / "user_exports" / self.trajectory.get() /
                               ("tab_%d_%s_dim_%s_action_%s.png" % (i, self.comparison.get(), self.dimension.get(), self.action_index.get())))
            path.parent.mkdir(parents=True, exist_ok=True)
            self.figures[i].savefig(path, dpi=160)
            self.status.set("已保存：" + str(path))
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc), parent=self.root)


def launch(folder):
    root = tk.Tk()
    ActorVerificationViewer(root, folder)
    root.mainloop()
