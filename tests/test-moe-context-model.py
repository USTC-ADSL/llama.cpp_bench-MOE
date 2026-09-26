#!/usr/bin/env python3
"""Semantic checks for context correction and conservative pre-load choice."""
import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location("context_model", pathlib.Path(__file__).with_name("moe-context-model.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ContextContract(unittest.TestCase):
    def setUp(self):
        self.model = {"decision_budget_us": 2, "base": {
            "backends": [[v] + [0] * 25 for v in (10, 20, 5)], "host": [1] + [0] * 7}}
        self.node = {"native_us": 100, "native_base_compute_us": 60, "native_uncertainty_us": 2,
                     "families": {"cpu": {"scales": [2, .5, 3, 2], "outside_us": 230, "uncertainty_us": 23}}}

    def test_uncertainty_can_reject_positive_nominal_gain(self):
        d = m.path_choice(self.model, self.node, 95, 60, 3)
        self.assertLess(d["custom_predicted_us"], d["native_predicted_us"])
        self.assertEqual(d["execution_mode"], "npu")

    def test_clear_gain_and_route_change_can_select_custom(self):
        self.assertEqual(m.path_choice(self.model, self.node, 80, 60, 3)["execution_mode"], "custom")
        self.assertEqual(m.path_choice(self.model, self.node, 95, 80, 3)["execution_mode"], "custom")

    def test_observed_decision_cost_can_exhaust_gain(self):
        self.assertEqual(m.path_choice(self.model, self.node, 90, 60, 3)["execution_mode"], "custom")
        d = m.path_choice(self.model, self.node, 90, 60, 3, observed_decision_us=8)
        self.assertEqual(d["execution_mode"], "npu")
        self.assertEqual(d["decision_budget_us"], 8)

    def test_correct_each_branch_before_critical_path_max(self):
        p = {"owner": [0] * 24 + [1] * 36, "shared": 2}
        d = m.corrected(self.model, self.node, 128, [9] * 32 + [8] * 28, p)
        self.assertEqual(d["backend_us"], [20, 10, 15])
        self.assertEqual(d["total_us"], 22)
        self.assertEqual(d["score_us"], 32)

    def test_native_npu_is_not_a_calibrated_callback_family(self):
        self.assertIsNone(m.family({"owner": [1] * 60, "shared": 1}))
        self.assertEqual(m.family({"owner": [1] * 60, "shared": 2}), "branch")

    def test_fullmodel_residual_is_added_once_despite_24_layers(self):
        p = {"owner": [0] * 24 + [1] * 36, "shared": 2}
        regular = m.corrected(self.model, self.node, 128, [9] * 32 + [8] * 28, p)
        final = m.corrected(self.model, self.node, 1, [1] * 4 + [0] * 56, p)
        self.assertEqual(23 * regular["outside_share_us"] + final["outside_share_us"], 230)

    def test_unsupported_lengths_are_not_silently_extrapolated(self):
        model = {"nodes": [{"tokens": 128}, {"tokens": 512}]}
        for nt in (64, 1024):
            with self.assertRaises(ValueError):
                m.interpolate(model, nt)


if __name__ == "__main__":
    unittest.main()
