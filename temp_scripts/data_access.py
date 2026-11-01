"""Read-only access to preindexed trajectories, with explicit T versus T+1."""
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np

EVENT_LABELS = {
    "first_near": "首次接近", "first_any_contact": "首次任一组接触",
    "first_strict_contact": "首次严格接触", "first_clear_opening": "首次明显开启",
    "first_success": "首次任务成功", "strict_contact_lost": "严格接触丢失",
    "strict_contact_reestablished": "严格接触恢复", "sustained_opening": "连续开启",
    "contact_supported_opening": "严格接触保持期间连续开启",
    "opening_stalled": "开启后停滞", "opening_reversed": "开启后回退", "episode_end": "轨迹终止",
}
STATUS_LABELS = {"observed": "已观测", "not_observed": "未发生", "right_censored": "终止处不足3步"}
DEV_ROLES = ("train", "validation")
ALL_ROLES = (*DEV_ROLES, "test_trajectory", "test_object_policy_seen", "test_object_policy_unseen")
CONTACT_LABELS = ["拇指", "食指", "中指", "无名指", "手掌", "任一组", "严格接触"]


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def optional_int(value):
    return None if value in (None, "") else int(value)


@dataclass
class Trajectory:
    meta: dict
    arrays: dict
    events: list
    segments: dict
    provenance: dict

    @property
    def tid(self):
        return self.meta["trajectory_id"]

    @property
    def n(self):
        return int(self.meta["decisions"])

    @property
    def dt(self):
        return float(self.provenance["control_time_step_seconds"])

    @property
    def time(self):
        return np.arange(self.n + 1) * self.dt

    def state(self, key):
        if "state__" + key in self.arrays:
            return self.arrays["state__" + key]
        return np.concatenate([self.arrays[key][:1], self.arrays["post_" + key]])

    @property
    def opening(self):
        return self.arrays["physical__opening_from_reset_rad"]

    def event_anchor(self, kind):
        return next((e["state_index"] for e in self.events if e["event_type"] == kind and e["status"] == "observed"), None)

    def relation(self, state_index):
        j = int(state_index)
        if not 0 <= j <= self.n:
            raise IndexError(j)
        # Never replace a missing terminal response with a zero activation/action.
        policy = None if j == self.n else {
            "h1": self.arrays["activations"][j], "h2": self.arrays["activations_l2"][j],
            "mu_raw": self.arrays["action_mean_raw"][j], "action": self.arrays["actions"][j],
        }
        return {"state_index": j, "time_s": float(self.time[j]),
                "distance_m": float(self.state("palm_handle_distance")[j]),
                "opening_rad": float(self.opening[j]), "strict_contact": bool(self.state("contact")[j]),
                "policy": policy, "previous_action_index": j - 1 if j else None,
                "post_state_index": j + 1 if policy is not None else None}


def validate_trajectory(traj):
    a, n = traj.arrays, traj.n
    for key, values in a.items():
        if (np.issubdtype(values.dtype, np.number) or values.dtype == np.bool_) and not np.isfinite(values).all():
            raise ValueError("Nonfinite saved values: " + key)
    for name, width in [("activations", 64), ("activations_l2", 64), ("action_mean_raw", 22), ("actions", 22), ("features_fused", 320)]:
        if a[name].shape != (n, width) or not np.isfinite(a[name]).all():
            raise ValueError("Invalid finite shape: " + name)
    if not a["done"][-1] or a["done"][:-1].any():
        raise ValueError("Incomplete trajectory")
    np.testing.assert_array_equal(a["step"], np.arange(n))
    for name in a:
        if name.startswith("post_"):
            np.testing.assert_array_equal(a[name][:-1], a[name[5:]][1:])
    for key in ["object_qpos", "palm_handle_distance", "contact", "finger_palm_contact", "palm_position_world", "handle_position_world"]:
        if len(traj.state(key)) != n + 1:
            raise ValueError("State chain must contain T+1 rows: " + key)
    np.testing.assert_array_equal(traj.state("object_qpos")[:-1], a["object_qpos"])
    np.testing.assert_array_equal(traj.state("object_qpos")[-1], a["post_object_qpos"][-1])
    np.testing.assert_allclose(a["actions"], np.clip(a["action_mean_raw"], -1, 1), atol=1e-6)
    for key in ["activations", "activations_l2"]:
        if (a[key] < 0).any():
            raise ValueError("Saved ReLU output has negative values")
    for e in traj.events:
        j = e["state_index"]
        if e["status"] == "not_observed":
            if j is not None or e["segment_id"]:
                raise ValueError("Absent event must not have a fabricated frame")
            continue
        if j is None or not 0 <= j <= n:
            raise ValueError("Event outside state chain")
        if e["trigger_action_index"] != (j - 1 if j else None):
            raise ValueError("Event/action indexing mismatch")
        if e["response_action_index"] != (j if j < n else None):
            raise ValueError("Fabricated or shifted event response")
        s = traj.segments[e["segment_id"]]
        if not 0 <= s["action_start_inclusive"] < s["action_stop_exclusive"] <= n:
            raise ValueError("Invalid clip bounds")
    if traj.relation(n)["policy"] is not None:
        raise ValueError("Terminal state must not have a policy response")


class DataRepository:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.records = csv_rows(self.root / "trajectory_index.csv")
        self.by_id = {r["trajectory_id"]: r for r in self.records}
        if len(self.by_id) != len(self.records):
            raise ValueError("Duplicate trajectory IDs")
        self.rules = json.loads((self.root / "event_rules.json").read_text(encoding="utf-8"))
        self.provenance = json.loads((self.root / "provenance.json").read_text(encoding="utf-8"))
        self.source_meta = {p["trajectory_id"]: p for p in self.provenance["episodes"]}
        self.events = defaultdict(list)
        self.segments = defaultdict(dict)
        for e in csv_rows(self.root / "events.csv"):
            for key in ["state_index", "confirmation_state_index", "trigger_action_index", "response_action_index", "run_end_state_index_exclusive"]:
                e[key] = optional_int(e[key])
            self.events[e["trajectory_id"]].append(e)
        for s in csv_rows(self.root / "segments.csv"):
            for key in ["event_state_index", "center_action_index", "action_start_inclusive", "action_stop_exclusive", "state_start_inclusive", "state_stop_exclusive"]:
                s[key] = int(s[key])
            self.segments[s["trajectory_id"]][s["segment_id"]] = s
        for events in self.events.values():
            events.sort(key=lambda e: (e["state_index"] is None, e["state_index"] or 0, e["event_id"]))
        self.cache = OrderedDict()

    def filter(self, roles=DEV_ROLES, object_id=None, outcome=None):
        return sorted((r for r in self.records if r["split"] in roles
                       and (object_id is None or int(r["object_id"]) == int(object_id))
                       and (outcome is None or (r["trajectory_success"] == "True") == outcome)),
                      key=lambda r: (int(r["object_id"]), int(r["episode_id"])))

    def load(self, tid):
        if tid in self.cache:
            self.cache.move_to_end(tid)
            return self.cache[tid]
        meta = self.by_id[tid]
        path = (self.root / meta["numeric_archive"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Archive path escaped source root")
        with np.load(path, allow_pickle=False) as f:
            arrays = {key: f[key] for key in f.files}
        traj = Trajectory(meta, arrays, self.events[tid], self.segments[tid], self.source_meta[tid])
        validate_trajectory(traj)
        self.cache[tid] = traj
        if len(self.cache) > 4:
            self.cache.popitem(last=False)
        return traj

    def example_pair(self, roles=DEV_ROLES, object_id=None):
        records = self.filter(roles, object_id)
        for obj in sorted({int(r["object_id"]) for r in records}):
            same = [r for r in records if int(r["object_id"]) == obj]
            yes = [r for r in same if r["trajectory_success"] == "True"]
            no = [r for r in same if r["trajectory_success"] == "False"]
            if yes and no:
                return yes[0]["trajectory_id"], no[0]["trajectory_id"]
        return None

    def validate_all(self):
        errors, n = [], 0
        for row in self.records:
            try:
                n += self.load(row["trajectory_id"]).n
            except Exception as exc:
                errors.append({"trajectory_id": row["trajectory_id"], "error": str(exc)})
        return {"status": "passed" if not errors else "failed", "trajectories": len(self.records),
                "action_rows": n, "state_rows": n + len(self.records), "errors": errors,
                "scope": "Shapes, finite activations/actions, state transitions, clipping, event/clip indices and missing terminal responses; no model fitting or event relabeling."}
