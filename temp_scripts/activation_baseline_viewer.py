"""Read saved baseline results in Tk; viewing never retrains probes."""
import csv
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from temp_scripts.activation_baseline_plots import summary_figure, window_figure, TARGET_LABELS


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def launch(folder):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "development_run_completed":
        raise ValueError("该目录尚未完成探针拟合；不能作为baseline结果查看。")
    rows = read_csv(folder / "selected_frames.csv")
    metrics = read_csv(folder / "metrics.csv")
    with np.load(folder / "selected_arrays.npz", allow_pickle=False) as f:
        y = f["y"]
    with np.load(folder / "predictions.npz", allow_pickle=False) as f:
        predictions = {k: f[k] for k in f.files}
    root = tk.Tk()
    root.title("步骤4 · 原始激活 baseline · train / validation")
    root.geometry("1380x900")
    ttk.Label(root, text="当前为事件窗口的状态可解码性；validation已参与选参。此处未评价SAE、动作贡献或测试集。", padding=8).pack(fill="x")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)
    page = ttk.Frame(notebook)
    notebook.add(page, text="验证集汇总")
    fig = Figure(figsize=(13, 11), layout="constrained")
    summary_figure(fig, metrics)
    canvas = FigureCanvasTkAgg(fig, page)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()
    table = ttk.Frame(notebook)
    notebook.add(table, text="指标明细")
    fields = ("target", "model", "seed", "split", "weighting", "mae", "average_precision", "precision", "recall", "log_loss")
    tree = ttk.Treeview(table, columns=fields, show="headings")
    for field in fields:
        tree.heading(field, text=field)
        tree.column(field, width=120, anchor="center")
    for row in metrics:
        tree.insert("", "end", values=[row.get(f, "") for f in fields])
    scrollbar = ttk.Scrollbar(table, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    tree.pack(fill="both", expand=True)
    detail = ttk.Frame(notebook)
    notebook.add(detail, text="逐轨迹连续窗口")
    controls = ttk.Frame(detail)
    controls.pack(fill="x")
    target = tk.StringVar(value="distance_m")
    keys = sorted(predictions)
    model = tk.StringVar(value="h1_linear__seed0")
    tids = sorted({r["trajectory_id"] for r in rows if r["split"] == "validation"})
    tids += sorted({r["trajectory_id"] for r in rows if r["split"] == "train"})
    trajectory = tk.StringVar(value=tids[0])
    widgets = []
    for title, variable, options, width in (("状态", target, list(TARGET_LABELS), 18),
                                            ("模型", model, sorted({k.split("__", 1)[1] for k in keys}), 28),
                                            ("轨迹（验证集在前）", trajectory, tids, 38)):
        ttk.Label(controls, text=title).pack(side="left", padx=5)
        box = ttk.Combobox(controls, textvariable=variable, values=options, state="readonly", width=width)
        box.pack(side="left", padx=5, pady=6)
        widgets.append(box)
    detail_fig = Figure(figsize=(12, 6), layout="constrained")
    detail_canvas = FigureCanvasTkAgg(detail_fig, detail)
    toolbar = NavigationToolbar2Tk(detail_canvas, detail, pack_toolbar=False)
    toolbar.pack(side="bottom", fill="x")
    detail_canvas.get_tk_widget().pack(fill="both", expand=True)
    def redraw(event=None):
        name, seed = model.get().rsplit("__seed", 1)
        window_figure(detail_fig, rows, y, predictions, target.get(), name, int(seed), trajectory.get())
        detail_canvas.draw_idle()
    for box in widgets:
        box.bind("<<ComboboxSelected>>", redraw)
    redraw()
    ttk.Label(root, text=str(folder), padding=5).pack(fill="x")
    root.mainloop()
