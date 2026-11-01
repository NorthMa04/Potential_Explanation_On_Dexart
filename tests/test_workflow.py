"""Real-data alignment and export regressions, independent of the native UI."""
import sys
sys.dont_write_bytecode = True
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temp_scripts.project_paths import bootstrap, default_index, OUTPUT_ROOT, output_path
bootstrap()
import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.figure import Figure
from temp_scripts.data_access import DataRepository, DEV_ROLES
from temp_scripts.export_results import export_trajectory, save_review
from temp_scripts.plots import configure_style, comparison_figure, snapshot_figure


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_style()
        cls.repo = DataRepository(default_index())
        cls.a, cls.b = [cls.repo.load(tid) for tid in cls.repo.example_pair()]
        cls.test_root = OUTPUT_ROOT / "_test"
        cls.test_root.mkdir(parents=True, exist_ok=True)

    def test_default_filter_never_includes_test_splits(self):
        rows = self.repo.filter()
        self.assertEqual(len(rows), 128)
        self.assertEqual({r["split"] for r in rows}, set(DEV_ROLES))

    def test_examples_are_same_object_and_different_outcomes(self):
        self.assertEqual(self.a.meta["object_id"], self.b.meta["object_id"])
        self.assertNotEqual(self.a.meta["trajectory_success"], self.b.meta["trajectory_success"])

    def test_terminal_state_has_no_zero_or_invented_policy(self):
        self.assertIsNone(self.a.relation(self.a.n)["policy"])
        self.assertIsNotNone(self.a.relation(self.a.n - 1)["policy"])
        self.assertEqual(len(self.a.state("object_qpos")), self.a.n + 1)

    def test_current_vectors_are_not_shifted_to_post_state(self):
        t = self.a.event_anchor("first_clear_opening")
        r = self.a.relation(t)
        np.testing.assert_array_equal(r["policy"]["h1"], self.a.arrays["activations"][t])
        np.testing.assert_array_equal(r["policy"]["action"], self.a.arrays["actions"][t])
        self.assertEqual(r["post_state_index"], t + 1)
        self.assertEqual(r["previous_action_index"], t - 1)

    def test_event_uses_trigger_before_response_and_terminal_response_is_missing(self):
        for traj in [self.a, self.b]:
            for e in traj.events:
                j = e["state_index"]
                if j is not None:
                    self.assertEqual(e["trigger_action_index"], j - 1 if j else None)
                    self.assertEqual(e["response_action_index"], j if j < traj.n else None)
                else:
                    self.assertFalse(e["segment_id"])

    def test_missing_alignment_is_disclosed_instead_of_synthesized(self):
        b = replace(self.b, events=[])
        fig = Figure(layout="constrained")
        axes = comparison_figure(fig, self.a, b, "first_clear_opening")
        self.assertIn("未发生", axes[0].texts[0].get_text())
        self.assertEqual(len(axes[0].get_legend_handles_labels()[1]), 1)
        fig.clear()

    def test_terminal_snapshot_contains_no_activation_bars(self):
        fig = Figure(layout="constrained")
        axes = snapshot_figure(fig, self.a, self.a.n)
        self.assertTrue(all(not ax.patches for ax in axes))
        self.assertTrue(all("终止状态" in ax.texts[0].get_text() for ax in axes))

    def test_outputs_cannot_escape_project_output_root(self):
        with self.assertRaises(ValueError):
            output_path(OUTPUT_ROOT.parent / "unorganized_export.csv")

    def test_clip_export_keeps_global_indices_and_final_state(self):
        scope = self.a.n - 5, self.a.n
        dest = export_trajectory(self.repo, self.a, self.test_root, self.a.n, scope=scope)
        with np.load(dest / "aligned_numeric.npz", allow_pickle=False) as d:
            np.testing.assert_array_equal(d["source_row_index"], np.arange(*scope))
            np.testing.assert_array_equal(d["state_index"], np.arange(scope[0], scope[1] + 1))
            np.testing.assert_array_equal(d["activations"], self.a.arrays["activations"][slice(*scope)])
            np.testing.assert_array_equal(d["state__object_qpos"][-1], self.a.state("object_qpos")[-1])
            self.assertEqual(len(d["state__object_qpos"]), 6)
            self.assertEqual(len(d["actions"]), 5)

    def test_review_is_separate_from_automatic_labels(self):
        area = self.test_root / ("review_save_test_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        area.mkdir(exist_ok=True)
        event = self.a.events[0]
        before = dict(event)
        with patch("temp_scripts.export_results.OUTPUT_ROOT", area):
            path = save_review(area, self.a, event, "不确定", "automated QA, not a human annotation")
        self.assertEqual(event, before)
        records = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(records[-1]["rules_modified"])
        self.assertTrue(path.is_relative_to(self.test_root))


if __name__ == "__main__":
    unittest.main()
