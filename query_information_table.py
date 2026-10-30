"""Standalone, read-only queries; all query exports remain under project outputs/."""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
from contextlib import closing
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re
import sqlite3

from project_paths import bootstrap, output_path
bootstrap()

from data_access import ALL_ROLES, EVENT_LABELS
from information_table import TABLE_HOME, dump, latest_database


def query(database, *, trajectory_id=None, state_index=None, object_id=None, split="development", neuron=None,
          min_activation=None, contact=None, event_type=None, trigger_before=False, limit=20, include_terminal=False):
    if limit < 1 or limit > 100000:
        raise ValueError("limit must be between 1 and 100000")
    columns = ["s.trajectory_id", "s.split", "s.state_index", "s.time_s", "s.policy_available", "s.distance_m", "s.opening_rad",
               "s.strict_contact", "s.any_contact", "s.confirmed_event_types", "s.h1_nonzero_count", "s.h1_l2_norm", "s.h1_top5_ids",
               "s.h2_nonzero_count", "s.h2_l2_norm", "s.h2_top5_ids", "s.action_clipped_count"]
    predicates, params = [], []
    if split == "development":
        predicates.append("s.split IN ('train','validation')")
    elif split != "all":
        if split not in ALL_ROLES:
            raise ValueError("Unknown split")
        predicates.append("s.split=?")
        params.append(split)
    if not include_terminal and state_index is None and event_type is None:
        predicates.append("s.policy_available=1")
    for col, value in [("trajectory_id", trajectory_id), ("state_index", state_index), ("object_id", object_id), ("strict_contact", contact)]:
        if value is not None:
            predicates.append("s." + col + "=?")
            params.append(value)
    if neuron is not None:
        match = re.fullmatch(r"(h[12])_(\d{1,2})", neuron)
        if not match or not 0 <= int(match[2]) < 64:
            raise ValueError("Neuron must be h1_00…h1_63 or h2_00…h2_63")
        neuron = "%s_%02d" % (match[1], int(match[2]))
        columns.append('s."' + neuron + '"')
        if min_activation is not None:
            predicates.append('s."' + neuron + '" >= ?')
            params.append(float(min_activation))
    elif min_activation is not None:
        raise ValueError("min_activation requires a neuron")
    if event_type is not None:
        if event_type not in EVENT_LABELS:
            raise ValueError("Unknown event type")
        index = "trigger_action_index" if trigger_before else "state_index"
        predicates.append("EXISTS(SELECT 1 FROM events e WHERE e.trajectory_id=s.trajectory_id AND e." + index + "=s.state_index AND e.event_type=? AND e.status='observed')")
        params.append(event_type)
    sql = "SELECT " + ",".join(columns) + " FROM states s" + (" WHERE " + " AND ".join(predicates) if predicates else "") + " ORDER BY s.object_id,s.episode_id,s.state_index LIMIT ?"
    params.append(limit)
    database = Path(database).resolve()
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, params)
        headers = [d[0] for d in cursor.description]
        rows = [dict(row) for row in cursor.fetchall()]
    return rows, headers, {"database": str(database), "sql": sql, "parameters": params,
                           "row_count": len(rows), "read_only": True, "ranking": "trajectory and time order; not ranked by activation",
                           "default_scope": "train/validation only unless explicitly changed"}


def export_query(rows, headers, evidence):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = output_path(TABLE_HOME / "queries" / stamp)
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / "query_result.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    dump(folder / "query_and_source.json", evidence)
    return folder


def main():
    parser = argparse.ArgumentParser(description="查询逐状态信息总表；默认只查看train/validation")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--trajectory-id")
    parser.add_argument("--state-index", type=int)
    parser.add_argument("--object-id", type=int)
    parser.add_argument("--split", default="development", choices=["development", "all", *ALL_ROLES])
    parser.add_argument("--neuron", help="例如 h1_12 或 h2_07；从0编号")
    parser.add_argument("--min-activation", type=float)
    parser.add_argument("--contact", type=int, choices=[0, 1])
    parser.add_argument("--event-type", choices=list(EVENT_LABELS))
    parser.add_argument("--trigger-before", action="store_true", help="查询事件前触发动作的对应行，而非事件状态行")
    parser.add_argument("--include-terminal", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    args = vars(parser.parse_args())
    database = args.pop("database") or latest_database()
    rows, headers, evidence = query(database, **args)
    folder = export_query(rows, headers, evidence)
    selected_neuron = headers[-1] if args["neuron"] is not None else None
    print("trajectory / split / state / time_s / distance_m / opening_rad / contact / h1_norm / h2_norm" + (" / " + selected_neuron if selected_neuron else ""), flush=True)
    def number(x):
        return "NULL" if x is None else "%.5g" % x
    for row in rows:
        values = [row["trajectory_id"], row["split"], str(row["state_index"]), *[number(row[k]) for k in ["time_s", "distance_m", "opening_rad", "strict_contact", "h1_l2_norm", "h2_l2_norm"]]]
        if selected_neuron:
            values.append(number(row[selected_neuron]))
        print(" | ".join(values), flush=True)
    print("Rows:", len(rows), "Output:", folder, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
