"""Native Tk result viewer. Reopening a run never retrains its SAE or probes."""
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from temp_scripts.project_paths import output_path
from temp_scripts.sae_plots import read_csv, training_figure, usage_figure, window_figure, probe_figure


class SAEViewer:
    def __init__(self, root, folder):
        from temp_scripts.plots import configure_style
        configure_style()
        self.root, self.folder = root, output_path(folder)
        self.summary = json.loads((self.folder / "summary.json").read_text(encoding="utf-8"))
        if self.summary["status"] != "development_run_completed":
            raise ValueError("只有完成的开发集运行可打开结果窗口")
        self.results = {r["experiment_id"]: r for r in self.summary["experiments"]}
        rows = read_csv(self.folder / "selected_frames.csv")
        tids = sorted({r["trajectory_id"] for r in rows if r["split"] == "validation"})
        tids += sorted({r["trajectory_id"] for r in rows if r["split"] == "train"})
        self.experiment = tk.StringVar(value=next(iter(self.results)))
        self.trajectory = tk.StringVar(value=tids[0])
        self.dimension = tk.StringVar(value="0")
        self.status = tk.StringVar()
        root.title("步骤5 · 纯 TopK SAE · 开发集分析")
        root.geometry("1400x950")
        ttk.Label(root, text="仅 train / validation；验证集已用于选参。原actor冻结，动作误差不代表物理仿真或干预结果。", padding=8).pack(fill="x")
        controls = ttk.Frame(root)
        controls.pack(fill="x")
        for caption, variable, choices, width in (("实验", self.experiment, list(self.results), 30),
                ("轨迹（验证集在前）", self.trajectory, tids, 38),
                ("动作维度", self.dimension, [str(i) for i in range(22)], 5)):
            ttk.Label(controls, text=caption).pack(side="left", padx=4)
            box = ttk.Combobox(controls, textvariable=variable, values=choices, state="readonly", width=width)
            box.pack(side="left", pady=5, padx=4)
            box.bind("<<ComboboxSelected>>", self.redraw)
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        self.figures, self.canvases = {}, {}
        for name, caption in (("training", "训练与动作保持"), ("usage", "特征使用"),
                               ("window", "逐步重建与动作"), ("probes", "当前状态预测")):
            page = ttk.Frame(notebook)
            notebook.add(page, text=caption)
            fig = Figure(figsize=(13, 8), layout="constrained")
            canvas = FigureCanvasTkAgg(fig, page)
            class Toolbar(NavigationToolbar2Tk):
                def save_figure(toolbar_self, *args, chart=name):
                    self.save_figure(chart)
            toolbar = Toolbar(canvas, page, pack_toolbar=False)
            toolbar.pack(side="bottom", fill="x")
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.figures[name], self.canvases[name] = fig, canvas
        page = ttk.Frame(notebook)
        notebook.add(page, text="指标明细")
        fields = ("kind", "target_or_quantity", "model", "split", "weighting", "mae", "mse", "AP", "log_loss")
        self.table = ttk.Treeview(page, columns=fields, show="headings")
        for field in fields:
            self.table.heading(field, text=field)
            self.table.column(field, width=140, anchor="center")
        vertical = ttk.Scrollbar(page, orient="vertical", command=self.table.yview)
        horizontal = ttk.Scrollbar(page, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        vertical.pack(side="right", fill="y")
        horizontal.pack(side="bottom", fill="x")
        self.table.pack(fill="both", expand=True)
        ttk.Label(root, textvariable=self.status, padding=4).pack(fill="x")
        self.redraw()

    def redraw(self, event=None):
        result = self.results[self.experiment.get()]
        experiment = self.folder / self.experiment.get()
        training_figure(self.figures["training"], experiment, result)
        usage_figure(self.figures["usage"], experiment, result)
        window_figure(self.figures["window"], self.folder, experiment, self.trajectory.get(), int(self.dimension.get()))
        probe_figure(self.figures["probes"], self.folder, experiment, self.trajectory.get(), self.summary["configuration"]["layer"])
        for canvas in self.canvases.values():
            canvas.draw_idle()
        self.table.delete(*self.table.get_children())
        for row in read_csv(experiment / "reconstruction_metrics.csv"):
            self.table.insert("", "end", values=("重建/动作", row["quantity"], "SAE", row["split"], row["weighting"],
                              row["mae"], row["mse"], "", ""))
        if (experiment / "probe_metrics.csv").exists():
            for row in read_csv(experiment / "probe_metrics.csv"):
                self.table.insert("", "end", values=("状态探针", row["target"], row["model"], row["split"], row["weighting"],
                                  row.get("mae", ""), "", row.get("average_precision", ""), row.get("log_loss", "")))
        self.status.set(str(experiment))

    def save_figure(self, chart):
        path = output_path(self.folder / self.experiment.get() / "figures" /
                           f"review_{chart}_{self.trajectory.get()}_dim{self.dimension.get()}.png")
        self.figures[chart].savefig(path, dpi=150)
        self.status.set("已保存：" + str(path))


def launch(folder):
    root = tk.Tk()
    try:
        SAEViewer(root, Path(folder))
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
