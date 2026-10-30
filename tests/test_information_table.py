"""Regression checks for grain, terminal NULLs and read-only experimental queries."""
import sys
sys.dont_write_bytecode = True
from datetime import datetime
from contextlib import closing
from pathlib import Path
import sqlite3
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import bootstrap, OUTPUT_ROOT, default_index
bootstrap()
import numpy as np
from data_access import DataRepository
from information_table import activation_summary, build_table, rows_for_trajectory, VECTOR_COLUMNS
from query_information_table import query


class InformationTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = DataRepository(default_index())
        cls.tid = "object_9996__episode_00004"
        cls.traj = cls.repo.load(cls.tid)
        cls.rows = list(rows_for_trajectory(cls.traj))
        destination = OUTPUT_ROOT / "_test" / ("information_table_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        cls.folder, cls.report = build_table(cls.repo, records=[cls.repo.by_id[cls.tid]], destination=destination, publish=False)
        cls.database = cls.folder / "state_activation.sqlite"

    def test_unique_state_grain_and_terminal_row(self):
        self.assertEqual(len(self.rows), self.traj.n + 1)
        self.assertEqual(len({(r["trajectory_id"], r["state_index"]) for r in self.rows}), len(self.rows))
        self.assertTrue(all(self.rows[-1][name] is None for name in VECTOR_COLUMNS))
        self.assertEqual(self.rows[-1]["is_terminal_state"], 1)

    def test_raw_vectors_are_lossless_and_current(self):
        t = 31
        for prefix, key, width in [("h1", "activations", 64), ("h2", "activations_l2", 64), ("mu_raw", "action_mean_raw", 22), ("action", "actions", 22)]:
            np.testing.assert_array_equal([self.rows[t]["%s_%02d" % (prefix, k)] for k in range(width)], self.traj.arrays[key][t])
        self.assertEqual(self.rows[t]["next_opening_rad"], self.rows[t + 1]["opening_rad"])

    def test_absent_event_not_attached_to_final_state(self):
        failure = self.repo.load("object_9996__episode_00000")
        self.assertIsNone(failure.event_anchor("first_success"))
        tail = list(rows_for_trajectory(failure))[-1]
        self.assertNotIn("first_success", tail["confirmed_event_types"])

    def test_zero_activation_is_measured_zero_not_missing(self):
        summary = activation_summary(np.zeros(64, dtype=np.float32), "h1")
        self.assertEqual(summary["h1_nonzero_count"], 0)
        self.assertEqual(summary["h1_l2_norm"], 0.)
        self.assertEqual(summary["h1_top5_ids"], "[]")
        self.assertIsNone(self.rows[-1]["h1_l2_norm"])

    def test_top_ids_and_values_correspond_and_only_include_positive_values(self):
        summary = activation_summary(np.array([0., 1., 4., 4., 0.]), "h2")
        self.assertEqual(summary["h2_top5_ids"], "[2, 3, 1]")
        self.assertEqual(summary["h2_top5_values"], "[4.0, 4.0, 1.0]")

    def test_query_distinguishes_event_state_and_trigger_row(self):
        at, _, _ = query(self.database, trajectory_id=self.tid, event_type="first_clear_opening")
        before, _, _ = query(self.database, trajectory_id=self.tid, event_type="first_clear_opening", trigger_before=True)
        self.assertEqual([r["state_index"] for r in at], [31])
        self.assertEqual([r["state_index"] for r in before], [30])

    def test_exact_terminal_query_returns_null_vectors(self):
        result, _, evidence = query(self.database, trajectory_id=self.tid, state_index=self.traj.n, neuron="h1_12")
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]["h1_12"])
        self.assertTrue(evidence["read_only"])

    def test_terminal_event_query_preserves_the_real_terminal_state(self):
        result, _, _ = query(self.database, trajectory_id=self.tid, event_type="first_success", neuron="h2_07")
        self.assertEqual([r["state_index"] for r in result], [self.traj.n])
        self.assertIsNone(result[0]["h2_07"])

    def test_invalid_neuron_cannot_be_used_as_sql(self):
        for name in ["h1_64", "h3_12", "h1_12;DROP TABLE states"]:
            with self.assertRaises(ValueError):
                query(self.database, neuron=name)

    def test_segment_endpoint_is_state_but_not_segment_action(self):
        segment = next(iter(self.traj.segments.values()))
        with closing(sqlite3.connect(self.database)) as conn:
            at_stop = conn.execute("SELECT is_action_row FROM segment_memberships WHERE segment_id=? AND state_index=?", (segment["segment_id"], segment["action_stop_exclusive"])).fetchone()
        self.assertEqual(at_stop, (0,))

    def test_database_and_csv_reconciled(self):
        self.assertEqual(self.report["status"], "passed")
        self.assertEqual(self.report["csv_rows_exactly_reconciled"], self.traj.n + 1)
        self.assertEqual(self.report["counts"]["terminal_rows"], 1)


if __name__ == "__main__":
    unittest.main()
