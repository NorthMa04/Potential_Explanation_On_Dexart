"""One row per physical state, with lossless vectors and explicit missing responses."""
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np

from temp_scripts.data_access import EVENT_LABELS
from temp_scripts.project_paths import PROJECT_ROOT, source_path, OUTPUT_ROOT, output_path

TABLE_HOME = OUTPUT_ROOT / "information_table"
VECTOR_GROUPS = [("h1", "activations", 64), ("h2", "activations_l2", 64),
                 ("mu_raw", "action_mean_raw", 22), ("action", "actions", 22)]
VECTOR_COLUMNS = ["%s_%02d" % (prefix, k) for prefix, _, width in VECTOR_GROUPS for k in range(width)]
ALIASES = {"step": "environment_step", "palm_handle_distance": "distance_m", "contact": "strict_contact", "loose_contact": "any_contact"}
OVERVIEW = ["trajectory_id", "object_id", "episode_id", "split", "trajectory_success", "state_index", "time_s",
            "policy_available", "distance_m", "opening_rad", "strict_contact", "any_contact", "progress",
            "confirmed_event_types", "unconfirmed_event_types", "h1_nonzero_count", "h1_l2_norm",
            "h1_top5_ids", "h1_top5_values", "h2_nonzero_count", "h2_l2_norm", "h2_top5_ids", "h2_top5_values",
            "action_clipped_count", "delta_opening_after_action_rad", "previous_action_index", "next_state_index",
            "event_ids", "triggered_event_ids", "segment_action_ids", "segment_state_ids"]


def dump(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    return int(value) if isinstance(value, bool) else value


def state_specs(traj):
    """Flatten recorded state vectors; preserve component coordinates in names."""
    result = []
    for key in traj.arrays:
        if "post_" + key not in traj.arrays:
            continue
        values = traj.state(key)
        shape = values.shape[1:]
        integer = values.dtype.kind in "biu"
        for index in np.ndindex(shape) if shape else [()]:
            name = ALIASES.get(key, key) + "".join("_%02d" % k for k in index)
            result.append((name, key, index, "INTEGER" if integer else "REAL"))
    return result


def activation_summary(vector, prefix):
    nonzero = np.flatnonzero(vector > 0)
    ranked = nonzero[np.argsort(-vector[nonzero], kind="stable")][:5]
    return {prefix + "_nonzero_count": int(len(nonzero)), prefix + "_l2_norm": float(np.linalg.norm(vector)),
            prefix + "_max": float(np.max(vector)),
            prefix + "_top5_ids": json.dumps(ranked.tolist()),
            prefix + "_top5_values": json.dumps([float(vector[k]) for k in ranked])}


def rows_for_trajectory(traj):
    specs = state_specs(traj)
    chains = {key: traj.state(key) for _, key, _, _ in specs}
    at_state, triggered = {}, {}
    state_members, action_members = {}, {}
    for e in traj.events:
        if e["state_index"] is not None:
            at_state.setdefault(e["state_index"], []).append(e)
        if e["trigger_action_index"] is not None:
            triggered.setdefault(e["trigger_action_index"], []).append(e["event_id"])
    for sid, segment in traj.segments.items():
        start, stop = segment["action_start_inclusive"], segment["action_stop_exclusive"]
        for j in range(start, stop + 1):
            state_members.setdefault(j, []).append(sid)
            if j < stop:
                action_members.setdefault(j, []).append(sid)
    for j in range(traj.n + 1):
        available = j < traj.n
        row = {"trajectory_id": traj.tid, "object_id": int(traj.meta["object_id"]), "episode_id": int(traj.meta["episode_id"]),
               "seed": int(traj.meta["seed"]), "split": traj.meta["split"], "policy_seen": int(traj.meta["policy_seen"] == "True"),
               "trajectory_success": int(traj.meta["trajectory_success"] == "True"), "state_index": j, "time_s": j * traj.dt,
               "policy_available": int(available), "is_terminal_state": int(not available),
               "action_index": j if available else None, "previous_action_index": j - 1 if j else None,
               "next_state_index": j + 1 if available else None}
        row.update({name: scalar(chains[key][(j,) + index]) for name, key, index, _ in specs})
        row.update({"opening_rad": float(traj.opening[j]), "q_rad": float(traj.arrays["physical__q_rad"][j]),
                    "opening_from_left_rad": float(traj.arrays["physical__opening_from_left_rad"][j]),
                    "remaining_to_target_rad": float(traj.arrays["physical__remaining_to_target_rad"][j])})
        for prefix, key, width in VECTOR_GROUPS:
            row.update({"%s_%02d" % (prefix, k): float(traj.arrays[key][j, k]) if available else None for k in range(width)})
        for prefix, key in [("h1", "activations"), ("h2", "activations_l2")]:
            summary = activation_summary(traj.arrays[key][j], prefix) if available else {
                prefix + "_" + suffix: None for suffix in ["nonzero_count", "l2_norm", "max", "top5_ids", "top5_values"]}
            row.update(summary)
            row[prefix + "_change_from_previous_l2"] = float(np.linalg.norm(traj.arrays[key][j] - traj.arrays[key][j - 1])) if available and j else None
        row.update({"action_raw_l2_norm": float(np.linalg.norm(traj.arrays["action_mean_raw"][j])) if available else None,
                    "action_l2_norm": float(np.linalg.norm(traj.arrays["actions"][j])) if available else None,
                    "action_clipped_count": int(np.count_nonzero(np.abs(traj.arrays["action_mean_raw"][j] - traj.arrays["actions"][j]) > 1e-6)) if available else None,
                    "next_distance_m": float(chains["palm_handle_distance"][j + 1]) if available else None,
                    "next_opening_rad": float(traj.opening[j + 1]) if available else None,
                    "delta_opening_after_action_rad": float(traj.arrays["physical__opening_delta_rad"][j]) if available else None,
                    "next_strict_contact": int(chains["contact"][j + 1]) if available else None,
                    "success_after_action": int(traj.arrays["success_after"][j]) if available else None,
                    "done_after_action": int(traj.arrays["done"][j]) if available else None})
        events = at_state.get(j, [])
        row.update({"event_ids": "|".join(e["event_id"] for e in events),
                    "event_names_zh": "|".join(EVENT_LABELS[e["event_type"]] for e in events),
                    "confirmed_event_types": "|".join(e["event_type"] for e in events if e["status"] == "observed"),
                    "unconfirmed_event_types": "|".join(e["event_type"] for e in events if e["status"] == "right_censored"),
                    "triggered_event_ids": "|".join(triggered.get(j, [])),
                    "segment_action_ids": "|".join(action_members.get(j, [])), "segment_state_ids": "|".join(state_members.get(j, []))})
        yield row


def field_dictionary(sample, example):
    state_types = {name: dtype for name, _, _, dtype in state_specs(example)}
    text = {"trajectory_id", "split", "event_ids", "event_names_zh", "confirmed_event_types", "unconfirmed_event_types", "triggered_event_ids", "segment_action_ids", "segment_state_ids"}
    text.update(prefix + suffix for prefix in ["h1", "h2"] for suffix in ["_top5_ids", "_top5_values"])
    integers = {"object_id", "episode_id", "seed", "policy_seen", "trajectory_success", "state_index", "policy_available", "is_terminal_state", "action_index", "previous_action_index", "next_state_index", "action_clipped_count", "next_strict_contact", "success_after_action", "done_after_action", "h1_nonzero_count", "h2_nonzero_count"}
    info = []
    for name in sample:
        dtype = "TEXT" if name in text else "INTEGER" if name in integers else state_types.get(name, "REAL")
        role, meaning, unit = "current_state_context", "动作前当前物理状态；多维字段按原数组索引展开", "原始单位"
        if name in VECTOR_COLUMNS:
            role = "current_activation_choose_one_layer" if name.startswith(("h1_", "h2_")) else "current_action_reference"
            meaning = "当前原始 ReLU 激活坐标；两层各64维" if name.startswith(("h1_", "h2_")) else "当前22维原始动作均值或实际裁剪动作"
            unit = "activation" if name.startswith(("h1_", "h2_")) else "normalized action"
        elif name.startswith(("h1_", "h2_", "action_")) and name != "action_index":
            role, meaning, unit = "descriptive_summary", "当前向量摘要；非零指严格大于0，Top5仅供浏览，不表示语义或贡献", "见字段名"
        if name in {"trajectory_id", "object_id", "episode_id", "seed", "split", "policy_seen", "state_index", "policy_available", "is_terminal_state", "action_index", "previous_action_index", "next_state_index"}:
            role, meaning, unit = "identity_and_alignment", "身份、划分、状态/动作索引或响应可用性", "index / flag"
        if name == "trajectory_success":
            role, meaning, unit = "offline_trajectory_outcome", "整条轨迹最终结果，不代替局部行为标签", "flag"
        if name.startswith("next_") and name != "next_state_index" or name in {"delta_opening_after_action_rad", "success_after_action", "done_after_action"}:
            role, meaning, unit = "post_action_future", "执行当前动作后才得到；不能混入动作前编码器", "见字段名"
        if "event" in name or "segment" in name:
            role, meaning, unit = "offline_event_annotation", "物理事件/片段关联；可能用未来确认，不是编码器输入", "IDs / event types"
        if "distance" in name or "position" in name or "minus_palm" in name:
            unit = "m"
        if "quaternion" in name:
            unit = "unitless; w,x,y,z"
        if name.endswith("_rad") or name in {"opening_rad", "q_rad"} or "qpos" in name:
            unit = "rad"
        if "qvel" in name:
            unit = "rad/s"
        if "velocity" in name:
            unit = "rad/s" if "angular" in name else "m/s"
        if name == "time_s":
            unit = "s"
        descriptions = {
            "distance_m": "当前手掌中心到把手位置的欧氏距离", "strict_contact": "至少两个手指接触组及手掌同时接触handle_link",
            "any_contact": "任一手指组或手掌接触handle_link", "phase": "上游距离/接触启发式阶段，不是人工语义真值",
            "progress": "上游环境进度原值，不裁剪，也不按本条轨迹终点归一化",
            "opening_rad": "相对固定物理初始位置left+0.2rad的开启增量，非本轨迹终点归一化",
            "q_rad": "物体任务关节原始位置；同object_qpos_00",
            "opening_from_left_rad": "相对物体任务标注left位置的有向角度增量",
            "remaining_to_target_rad": "朝middle任务目标尚余的有向角度，越过目标可为负",
            "time_fraction": "环境时限内的归一化时间；不能代替物理行为标签",
        }
        if name in descriptions:
            meaning = descriptions[name]
        if name.startswith("finger_palm_contact_"):
            group = ["拇指", "食指", "中指", "无名指", "手掌"][int(name.rsplit("_", 1)[1])]
            meaning, unit = group + "接触handle_link的上游接触组标记", "flag (0/1)"
        if name in {"strict_contact", "any_contact", "policy_available", "is_terminal_state"}:
            unit = "flag (0/1)"
        info.append({"column": name, "sqlite_type": dtype, "role": role, "unit": unit, "description_zh": meaning,
                     "terminal_policy_missing": int(name in VECTOR_COLUMNS or name.startswith(("h1_", "h2_")) or role in {"current_action_reference", "post_action_future"} or name in {"action_index", "next_state_index", "action_raw_l2_norm", "action_l2_norm", "action_clipped_count"})})
    return info


def create_database(conn, dictionary):
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.executescript("""
        CREATE TABLE trajectories(trajectory_id TEXT PRIMARY KEY, object_id INTEGER, split TEXT, trajectory_success INTEGER,
                                  decisions INTEGER, source_npz TEXT, source_sha256 TEXT, metadata_json TEXT);
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value_json TEXT);
    """)
    columns = ",".join('"%s" %s' % (d["column"], d["sqlite_type"]) for d in dictionary)
    conn.execute("CREATE TABLE states(" + columns + ",PRIMARY KEY(trajectory_id,state_index),FOREIGN KEY(trajectory_id) REFERENCES trajectories(trajectory_id))")
    conn.executescript("""
        CREATE TABLE events(event_id TEXT PRIMARY KEY, trajectory_id TEXT REFERENCES trajectories(trajectory_id),
            event_type TEXT, status TEXT, state_index INTEGER, trigger_action_index INTEGER, response_action_index INTEGER,
            confirmation_state_index INTEGER, segment_id TEXT, metadata_json TEXT);
        CREATE TABLE segments(segment_id TEXT PRIMARY KEY, trajectory_id TEXT REFERENCES trajectories(trajectory_id),
            event_id TEXT REFERENCES events(event_id), action_start_inclusive INTEGER, action_stop_exclusive INTEGER,
            state_start_inclusive INTEGER, state_stop_exclusive INTEGER, numeric_clip TEXT, metadata_json TEXT);
        CREATE TABLE event_state_links(event_id TEXT REFERENCES events(event_id), trajectory_id TEXT, state_index INTEGER,
            relation TEXT, PRIMARY KEY(event_id,relation), FOREIGN KEY(trajectory_id,state_index) REFERENCES states(trajectory_id,state_index));
        CREATE TABLE segment_memberships(segment_id TEXT REFERENCES segments(segment_id), trajectory_id TEXT, state_index INTEGER,
            is_action_row INTEGER, PRIMARY KEY(segment_id,state_index), FOREIGN KEY(trajectory_id,state_index) REFERENCES states(trajectory_id,state_index));
        CREATE INDEX states_filter ON states(split,object_id,policy_available);
        CREATE INDEX states_physical ON states(strict_contact,opening_rad,distance_m);
        CREATE INDEX events_filter ON events(event_type,status);
        CREATE VIEW development_states AS SELECT * FROM states WHERE split IN ('train','validation') AND policy_available=1;
    """)
    conn.execute("CREATE VIEW compact_overview AS SELECT " + ",".join(OVERVIEW) + " FROM states")
    conn.execute("CREATE TABLE field_dictionary(column TEXT PRIMARY KEY,sqlite_type TEXT,role TEXT,unit TEXT,description_zh TEXT,terminal_policy_missing INTEGER)")
    conn.executemany("INSERT INTO field_dictionary VALUES(?,?,?,?,?,?)", [tuple(d.values()) for d in dictionary])


def add_trajectory(conn, traj, rows, columns):
    m = traj.meta
    conn.execute("INSERT INTO trajectories VALUES(?,?,?,?,?,?,?,?)", (traj.tid, int(m["object_id"]), m["split"], int(m["trajectory_success"] == "True"), traj.n, m["source_npz"], m["source_sha256"], json.dumps({"trajectory": m, "provenance": traj.provenance}, ensure_ascii=False)))
    conn.executemany("INSERT INTO states VALUES(" + ",".join("?" for _ in columns) + ")", [tuple(r[c] for c in columns) for r in rows])
    for e in traj.events:
        conn.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)", (e["event_id"], traj.tid, e["event_type"], e["status"], e["state_index"], e["trigger_action_index"], e["response_action_index"], e["confirmation_state_index"], e["segment_id"] or None, json.dumps(e, ensure_ascii=False)))
        for key, relation in [("state_index", "event_state"), ("trigger_action_index", "before_trigger_action"), ("response_action_index", "policy_response"), ("confirmation_state_index", "offline_confirmation")]:
            if e[key] is not None:
                conn.execute("INSERT INTO event_state_links VALUES(?,?,?,?)", (e["event_id"], traj.tid, e[key], relation))
    for sid, s in traj.segments.items():
        start, stop = s["action_start_inclusive"], s["action_stop_exclusive"]
        conn.execute("INSERT INTO segments VALUES(?,?,?,?,?,?,?,?,?)", (sid, traj.tid, s["event_id"], start, stop, s["state_start_inclusive"], s["state_stop_exclusive"], s["numeric_clip"], json.dumps(s, ensure_ascii=False)))
        conn.executemany("INSERT INTO segment_memberships VALUES(?,?,?,?)", [(sid, traj.tid, j, int(j < stop)) for j in range(start, stop + 1)])


def verify_database(repo, conn, records):
    conn.row_factory = sqlite3.Row
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or conn.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("SQLite integrity or foreign key failure")
    comparisons = 0
    for meta in records:
        traj = repo.load(meta["trajectory_id"])
        actual = conn.execute("SELECT * FROM states WHERE trajectory_id=? ORDER BY state_index", (traj.tid,)).fetchall()
        if len(actual) != traj.n + 1:
            raise ValueError("Row count mismatch")
        for prefix, key, width in VECTOR_GROUPS:
            cols = ["%s_%02d" % (prefix, k) for k in range(width)]
            np.testing.assert_array_equal([[row[c] for c in cols] for row in actual[:-1]], traj.arrays[key].astype(float))
            if any(actual[-1][c] is not None for c in cols):
                raise ValueError("Fabricated terminal vector")
            comparisons += traj.n * width
        for name, key, index, _ in state_specs(traj):
            expected = traj.state(key)[(slice(None),) + index]
            np.testing.assert_array_equal([row[name] for row in actual], expected)
            comparisons += len(expected)
        for name, expected in [("opening_rad", traj.opening), ("time_s", traj.time)]:
            np.testing.assert_array_equal([row[name] for row in actual], expected)
        if actual[0]["previous_action_index"] is not None or actual[-1]["next_state_index"] is not None:
            raise ValueError("Initial/final alignment error")
        if any(actual[-1][c] is not None for c in ["h1_top5_ids", "h2_nonzero_count", "action_clipped_count", "next_distance_m"]):
            raise ValueError("Terminal unavailable fields must be NULL")
    counts = {name: conn.execute("SELECT COUNT(*) FROM " + name).fetchone()[0] for name in ["trajectories", "states", "events", "segments", "event_state_links", "segment_memberships", "development_states"]}
    counts["policy_rows"] = conn.execute("SELECT COUNT(*) FROM states WHERE policy_available=1").fetchone()[0]
    counts["terminal_rows"] = conn.execute("SELECT COUNT(*) FROM states WHERE policy_available=0").fetchone()[0]
    count_error = conn.execute("SELECT COUNT(*) FROM states s JOIN states n ON s.trajectory_id=n.trajectory_id AND n.state_index=s.state_index+1 WHERE s.policy_available=1 AND (ABS(s.next_opening_rad-n.opening_rad)>1e-12 OR ABS(s.next_distance_m-n.distance_m)>1e-12 OR s.next_strict_contact<>n.strict_contact OR ABS(s.delta_opening_after_action_rad-(n.opening_rad-s.opening_rad))>1e-12)").fetchone()[0]
    if count_error:
        raise ValueError("Post-action fields shifted relative to next state")
    return {"status": "passed", "counts": counts, "source_numeric_values_exactly_reconciled": comparisons,
            "foreign_key_check": "passed", "integrity_check": "ok", "post_action_alignment_errors": count_error}


def verify_csv(path, conn, dictionary):
    types = {d["column"]: d["sqlite_type"] for d in dictionary}
    checked = 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cursor = iter(conn.execute("SELECT * FROM states ORDER BY rowid"))
        for text_row in reader:
            db_row = next(cursor)
            for name, value in text_row.items():
                expected = db_row[name]
                actual = None if value == "" and expected is None else int(value) if types[name] == "INTEGER" else float(value) if types[name] == "REAL" else value
                if actual != expected:
                    raise ValueError("CSV roundtrip mismatch: " + name)
            checked += 1
        if next(cursor, None) is not None:
            raise ValueError("CSV omitted rows")
    return checked


def build_table(repo, records=None, destination=None, publish=True):
    records = repo.filter(roles=("train", "validation", "test_trajectory", "test_object_policy_seen", "test_object_policy_unseen")) if records is None else records
    if not records:
        raise ValueError("No trajectories selected")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = output_path(destination or TABLE_HOME / stamp)
    folder.mkdir(parents=True, exist_ok=False)
    sample_traj = repo.load(records[0]["trajectory_id"])
    sample = next(rows_for_trajectory(sample_traj))
    columns, dictionary = list(sample), field_dictionary(sample, sample_traj)
    database = folder / "state_activation.sqlite"
    conn = sqlite3.connect(database)
    try:
        create_database(conn, dictionary)
        with (folder / "states_full.csv").open("w", encoding="utf-8-sig", newline="") as full, (folder / "states_overview.csv").open("w", encoding="utf-8-sig", newline="") as overview:
            writer = csv.DictWriter(full, fieldnames=columns)
            compact = csv.DictWriter(overview, fieldnames=OVERVIEW, extrasaction="ignore")
            writer.writeheader()
            compact.writeheader()
            for meta in records:
                traj = repo.load(meta["trajectory_id"])
                rows = list(rows_for_trajectory(traj))
                if any(list(row) != columns for row in rows):
                    raise ValueError("Schema drift across trajectories")
                add_trajectory(conn, traj, rows, columns)
                writer.writerows(rows)
                compact.writerows(rows)
        report = verify_database(repo, conn, records)
        report["csv_rows_exactly_reconciled"] = verify_csv(folder / "states_full.csv", conn, dictionary)
        report["columns"] = len(columns)
        metadata = {"source_index": str(repo.root), "rules_sha256": repo.provenance["rules_sha256"], "script_sha256": file_hash(__file__),
                    "grain": "one row per (trajectory_id,state_index), including terminal state",
                    "export_scope": "all_trajectories" if len(records) == len(repo.records) else "selected_trajectories",
                    "default_query_roles": ["train", "validation"], "contact_group_order": ["thumb", "index", "middle", "ring", "palm"],
                    "loader_sha256": file_hash(source_path("data_access.py")),
                    "encoder_contract": "h-only encoders choose one activation layer; do not use identity, outcome, event, or future fields as encoder inputs",
                    "top5_contract": "strictly-positive ReLU values ranked by magnitude for browsing; no semantics, importance or causal contribution inferred",
                    "source_sha256_by_trajectory": {r["trajectory_id"]: r["source_sha256"] for r in records}}
        for key, value in metadata.items():
            conn.execute("INSERT INTO metadata VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))
        conn.commit()
        dump(folder / "verification.json", report)
        dump(folder / "provenance.json", metadata)
        for name in ["information_table.py", "2_build_information_table.py", "2_query_information_table.py", "information_queries.py", "data_access.py", "project_paths.py", "__init__.py"]:
            target = output_path(folder / "source_snapshot" / source_path(name).relative_to(PROJECT_ROOT))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path(name), target)
        with (folder / "field_dictionary.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dictionary[0]))
            w.writeheader()
            w.writerows(dictionary)
        readme = "# 逐状态信息总表\n\n主键为 trajectory_id + state_index。共 %d 条轨迹、%d 行状态，%d 列；其中 %d 行有策略响应，%d 行终止状态的激活与动作留空。\n\n- states_overview.csv：时间、状态、事件、激活摘要，方便快速浏览。\n- states_full.csv：全部状态坐标、两层各64维激活、22维原始和实际动作。\n- state_activation.sqlite：相同总表及 trajectories/events/segments 关联表，可重复查询。\n- field_dictionary.csv：字段单位、用途、未来信息和缺失约定。\n- verification.json / provenance.json：检查与来源记录。\n\nSQLite 的 development_states 默认只含 train/validation 且有响应的状态；states 包含全部固定划分。event_state_links 区分事件状态、触发前、响应、离线确认；segment_memberships 区分片段状态和片段动作行，避免重复连接造成样本膨胀。\n\nTop5 是原始激活强度排序，不是语义或因果贡献排名。接触组顺序及机器人关节名见 trajectories.metadata_json 中的 provenance；所有数组从0编号。状态单位遵循来源，四元数 w,x,y,z。未来字段、整轨迹结局和离线事件均不能当作 h-only 编码器输入。\n" % (report["counts"]["trajectories"], report["counts"]["states"], report["columns"], report["counts"]["policy_rows"], report["counts"]["terminal_rows"])
        (folder / "README.md").write_text(readme, encoding="utf-8")
    finally:
        conn.close()
    if publish:
        pointer = {"directory": str(folder), "database": str(database), "verification": str(folder / "verification.json")}
        dump(TABLE_HOME / "latest.json", pointer)
        relative = folder.relative_to(TABLE_HOME).as_posix()
        (TABLE_HOME / "README.md").write_text("# 逐状态信息总表\n\n当前版本：%s。主键为轨迹ID + 状态索引；每行一个真实物理状态。\n\n- [摘要表（方便浏览）](%s/states_overview.csv)\n- [完整表（全部338列）](%s/states_full.csv)\n- [SQLite数据库（可重复查询）](%s/state_activation.sqlite)\n- [字段字典](%s/field_dictionary.csv)\n- [对齐与缺失检查](%s/verification.json)\n- [来源与使用约定](%s/README.md)\n\n运行项目根目录的 2_build_information_table.py 可独立重建；运行 2_query_information_table.py 可独立查询。默认查询只含train/validation；数据库保留全部固定划分。总表包含终止状态，其缺失激活/动作记为NULL，不能当作零激活。\n" % (folder.name, relative, relative, relative, relative, relative, relative), encoding="utf-8")
    return folder, report


def latest_database():
    pointer = TABLE_HOME / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError("先运行 2_build_information_table.py 生成总表。")
    return Path(json.loads(pointer.read_text(encoding="utf-8"))["database"])
