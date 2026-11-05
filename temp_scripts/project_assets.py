"""Read-only source/data inventory. Writes only a fresh outputs/asset_catalog run."""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

from temp_scripts.project_paths import PROJECT_ROOT, OUTPUT_ROOT, output_path

WORKSPACE = PROJECT_ROOT.parents[1]
RESEARCH = WORKSPACE / "research" / "embodied_sae"
SOURCE_SUFFIXES = {".py", ".md", ".txt", ".sh", ".json", ".lock", ".pyproj", ".slnx"}
NS = {"m": "http://schemas.microsoft.com/developer/msbuild/2003"}
LABELS = {
    "_root_files": ("集合根元数据/归档", "保留原名；压缩包不当作新增独立数据"),
    "activation_baseline": ("原始激活状态读出基线", "步骤4；按原train/validation协议使用"),
    "actor_verification": ("冻结actor数值核验", "步骤3；输出复现不是物理效果评价"),
    "constrained_physical": ("受约束Top1闭环开发", "步骤10；独立单位为物理起点"),
    "physical_simulation": ("初始局部替换闭环开发", "步骤9；保留全前缀与空操作审计"),
    "examples": ("早期可视化示例", "用于审阅，不作代表性抽样推断"),
    "information_table": ("逐状态总表与查询", "步骤2；保留原划分与终止缺失值"),
    "runs": ("轨迹审阅与人工导出", "保留原人工意见与来源"),
    "sae_comparison": ("纯SAE配置对比", "步骤6；开发集比较，不是新确认集"),
    "sae_study_20261101": ("纯SAE开发汇总", "36训练的历史汇总与复看入口"),
    "laptop_dataset_001": ("原始自然轨迹", "原划分；禁止重新划分或改标"),
    "laptop_physical_index_v1": ("物理事件索引", "原划分；T动作与T+1状态分开"),
    "laptop_events_v1": ("历史事件索引", "历史规则，使用时核对版本"),
    "laptop_pilot_001": ("早期采集调试", "不混入正式训练或测试"),
    "laptop_resume_bundle": ("恢复与部署资源", "不是独立评价样本"),
    "_cache": ("缓存", "可再生；本轮保留"),
    "_test": ("测试中间文件", "不计研究样本；本轮保留"),
    "document_updates": ("历史文档快照", "不作为最新待办"),
    "server_checks": ("环境核验", "不是研究评价"),
    "combined_physical": ("组合闭环开发实验", "8起点；运行条件不是独立样本"),
    "online_development_20261104": ("双向闭环开发实验", "32自然筛查/7处理起点；不是留出确认"),
    "state_assisted": ("辅助训练及冻结评价", "按各子批冻结协议区分开发与测试"),
    "pure_sae": ("纯SAE开发训练", "开发结果"),
    "local_interpretability": ("本地事后分析", "不增加独立物理样本"),
    "deeper_analysis_20261105": ("本地事后分析", "冻结读出迁移及机制探索"),
    "contact_coordination_20261105": ("接触协调事后分析", "7起点/21轨迹；未新增仿真"),
    "snapshot_matching_20261105": ("快照匹配探索", "9基础匹配/0严格匹配"),
    "action_backtrace_20261105": ("单状态动作反查", "数值分解；不是独立物理验证"),
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def files_under(root):
    """Do not follow junctions/symlinks or silently suppress traversal errors."""
    def onerror(error):
        raise error
    for parent, dirs, names in os.walk(root, followlinks=False, onerror=onerror):
        dirs[:] = sorted(d for d in dirs if d not in {".git", "__pycache__"}
                         and not (Path(parent) / d).is_symlink()
                         and not os.path.isjunction(Path(parent) / d))
        for name in sorted(names):
            path = Path(parent) / name
            if not path.is_symlink():
                yield path


def source_files():
    result = [p for p in PROJECT_ROOT.iterdir() if p.is_file()
              and (p.suffix in SOURCE_SUFFIXES or p.name in {".gitignore", ".gitattributes"})]
    for base in [PROJECT_ROOT / "temp_scripts", PROJECT_ROOT / "tests",
                 RESEARCH / "scripts", RESEARCH / "tests"]:
        result.extend(p for p in files_under(base) if p.suffix in SOURCE_SUFFIXES)
    result.extend(WORKSPACE.glob("*.md"))
    result.extend(p for p in RESEARCH.iterdir() if p.is_file()
                  and (p.suffix in SOURCE_SUFFIXES or p.name == ".gitignore"))
    return sorted(set(result))


def source_role(path):
    if path.name in {"retrieve_combined_run.py", "keep_awake_during_retrieval.py", "switch_combined_parallel.py"}:
        return "运维工具（禁止为检查而导入执行）"
    if path.parent == PROJECT_ROOT and path.suffix == ".py":
        return "编号入口"
    if "tests" in path.parts:
        return "回归测试"
    if path.parent == PROJECT_ROOT / "temp_scripts":
        return "正式公共模块"
    if path.is_relative_to(RESEARCH / "scripts"):
        return "采集/部署历史脚本（保留原名）"
    return "文档/配置"


def audit_sources(paths):
    issues = []
    for p in paths:
        if p.suffix == ".py":
            ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
        if p.parent == PROJECT_ROOT and p.suffix == ".py" and not re.fullmatch(r"\d+_[a-z][a-z0-9_]*\.py", p.name):
            issues.append("Unnumbered root entry: " + p.name)
    project = ET.parse(PROJECT_ROOT / "Potential_Explanation_On_Dexart.pyproj")
    registered = [e.attrib["Include"].replace("\\", "/") for e in project.findall(".//m:Compile", NS)]
    expected = {p.relative_to(PROJECT_ROOT).as_posix() for p in paths
                if p.suffix == ".py" and p.is_relative_to(PROJECT_ROOT)}
    issues += ["Missing VS registration: " + x for x in sorted(expected - set(registered))]
    issues += ["Stale VS registration: " + x for x in sorted(set(registered) - expected)]
    issues += ["Duplicate VS registration: " + x for x, n in Counter(registered).items() if n > 1]
    return issues


def catalog(out):
    paths = source_files()
    issues = audit_sources(paths)
    if issues:
        raise ValueError("\n".join(issues))
    out = output_path(out)
    # Never overwrite a historical catalog, including partial runs.
    out.mkdir(parents=True, exist_ok=False)
    source_rows = []
    with zipfile.ZipFile(out / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            rel = p.relative_to(WORKSPACE).as_posix()
            source_rows.append({"path": rel, "role": source_role(p), "bytes": p.stat().st_size,
                                "sha256": digest(p), "version_scope": "VS Git" if p.is_relative_to(PROJECT_ROOT) else "外部采集/文档快照"})
            archive.write(p, rel)
    write_json(out / "code_inventory.json", source_rows)
    # Verify archived bytes against the live source hashes.
    with zipfile.ZipFile(out / "source_snapshot.zip") as archive:
        assert all(hashlib.sha256(archive.read(r["path"])).hexdigest() == r["sha256"] for r in source_rows)
    registry, reports, manifest_count = [], [], 0
    with (out / "file_inventory.jsonl").open("w", encoding="utf-8") as manifest:
        roots = [("outputs", OUTPUT_ROOT), ("research_runs", RESEARCH / "runs"), ("research_evidence", RESEARCH / "evidence")]
        for scope, base in roots:
            for collection in [base, *sorted(base.iterdir())]:
                if not collection.is_dir() or collection.name == "asset_catalog":
                    continue
                collection_name = "_root_files" if collection == base else collection.name
                role, usage = LABELS.get(collection_name, ("历史派生产物", "按该批协议/报告判断用途；不重新标记结局"))
                row = {"scope": scope, "collection": collection_name, "role": role, "usage": usage,
                       "path": collection.relative_to(WORKSPACE).as_posix(), "file_count": 0, "bytes": 0,
                       "report_count": 0, "verification_count": 0, "metadata_changed_during_scan": 0}
                members = sorted(p for p in base.iterdir() if p.is_file()) if collection == base else files_under(collection)
                for p in members:
                    stat = p.stat()
                    rel = p.relative_to(WORKSPACE).as_posix()
                    record = {"path": rel, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                              "collection": collection_name, "scope": scope}
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    row["file_count"] += 1
                    row["bytes"] += stat.st_size
                    manifest_count += 1
                    if p.suffix.lower() == ".md":
                        row["report_count"] += 1
                        # Exclude embedded code/docs backups from the reader report index.
                        if not any(x in p.parts for x in ("code", "code_snapshot", "review", "document_updates")):
                            reports.append({"collection": collection_name, "path": rel})
                    if "verification" in p.name.lower():
                        row["verification_count"] += 1
                    after = p.stat()
                    row["metadata_changed_during_scan"] += (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns)
                registry.append(row)
    write_json(out / "data_registry.json", registry)
    write_json(out / "report_index.json", reports)
    # Copy original labels exactly; do not infer, refit, or relabel outcomes.
    label_source = RESEARCH / "runs/laptop_physical_index_v1/trajectory_index.csv"
    with label_source.open(encoding="utf-8-sig", newline="") as f:
        trajectories = list(csv.DictReader(f))
    missing_trajectory_paths = [r["trajectory_id"] for r in trajectories
                                if not Path(r["source_npz"]).is_file()
                                or not (label_source.parent / r["numeric_archive"]).is_file()]
    write_json(out / "trajectory_labels.json", trajectories)
    write_json(out / "label_contract.json", {
        "source": label_source.relative_to(WORKSPACE).as_posix(), "sha256": digest(label_source),
        "rule": "原字段原值复制；人工整理备注不回写源标签；test仅登记元数据，不加载轨迹数组或拟合。",
        "split_counts": dict(Counter(r["split"] for r in trajectories)),
        "strict_contact": "至少两个指组加手掌；任一接触不等于严格接触",
        "success": "原判据同一步progress>.95、距离<=.2m、严格接触；本轮不重算或改标",
        "time_alignment": "动作t对应动作前state[t]与动作后state[t+1]；终止状态没有动作/激活",
    })
    packages = sorted({d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}.items())
    (out / "environment_installed.txt").write_text("\n".join(f"{n}=={v}" for n, v in packages) + "\n", encoding="utf-8")
    write_json(out / "environment.json", {"executable": sys.executable, "version": sys.version,
               "platform": sys.platform, "purpose": "当前本地已安装环境记录；不是服务器环境或跨平台安装锁"})
    for name, args in [("git_status.txt", ["status", "--short"]), ("git_log.txt", ["log", "-8", "--format=%h %aI %s"])]:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, check=True)
        (out / name).write_bytes(result.stdout)
    # Final source content verification catches any concurrent code change.
    source_unchanged = all(digest(WORKSPACE / r["path"]) == r["sha256"] for r in source_rows)
    summary = {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
               "source_count": len(paths), "source_snapshot_sha256": digest(out / "source_snapshot.zip"),
               "source_unchanged": source_unchanged, "file_count": manifest_count,
               "data_bytes": sum(r["bytes"] for r in registry), "collection_count": len(registry),
               "trajectory_count": len(trajectories), "unique_trajectory_ids": len({r["trajectory_id"] for r in trajectories}),
               "missing_trajectory_paths": missing_trajectory_paths,
               "metadata_changed_during_scan": sum(r["metadata_changed_during_scan"] for r in registry),
               "data_integrity_scope": "本轮全文件路径/大小/mtime登记；未重新计算全量数据SHA256，既有校验记录保留。",
               "excluded": ["本维护输出asset_catalog", ".git", "__pycache__", "链接目录", "工作区tmp中的SSH/运维材料"],
               "research_advanced": False, "issues": issues}
    write_json(out / "verification.json", summary)
    lines = ["# 代码与数据资产目录", "", "本轮仅整理。原数据、划分、任务结局、冻结包及历史输出均保留原路径。", "",
             "- `code_inventory.json`：代码/文档角色与SHA256；`source_snapshot.zip`：已逐文件核验的源码快照。",
             "- `data_registry.json`：目录用途、文件数与容量；`file_inventory.jsonl`：逐文件路径/大小/mtime。",
             "- `report_index.json`：已有报告入口；`trajectory_labels.json`：原标签逐值复制。",
             "- `label_contract.json`：原划分、接触/成功与时序口径；人工备注应另列，不能覆盖源标签。",
             "- `environment_installed.txt`：本地已安装依赖版本记录，非可移植安装锁。",
             "- `verification.json`：本次整理范围与校验，不代替既有科学结果/全量回传哈希核验。", "",
             "## 数据目录", "", "|目录|用途|文件数|GiB|", "|---|---|---:|---:|"]
    lines.extend(f"|{r['collection']}|{r['role']}|{r['file_count']}|{r['bytes']/2**30:.3f}|" for r in registry)
    (out / "CATALOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not source_unchanged or missing_trajectory_paths or summary["metadata_changed_during_scan"] or summary["unique_trajectory_ids"] != len(trajectories):
        raise RuntimeError("Catalog consistency check failed; inspect verification.json")
    pointer = OUTPUT_ROOT / "asset_catalog" / "latest.json"
    temporary = pointer.with_suffix(".tmp")
    write_json(temporary, {"path": out.relative_to(PROJECT_ROOT).as_posix(), "created_at": summary["created_at"]})
    temporary.replace(pointer)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Fresh directory inside outputs/asset_catalog/")
    args = parser.parse_args()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    out = args.out or OUTPUT_ROOT / "asset_catalog" / stamp
    if not out.resolve().is_relative_to((OUTPUT_ROOT / "asset_catalog").resolve()):
        parser.error("--out must stay inside outputs/asset_catalog/")
    return catalog(out)
