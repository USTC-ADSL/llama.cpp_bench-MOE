#!/usr/bin/env python3
"""Contract checks for critical-path prediction and resident feasibility."""
import importlib.util
import pathlib
import unittest

path = pathlib.Path(__file__).with_name("moe-placement-model.py")
spec = importlib.util.spec_from_file_location("placement", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ModelContract(unittest.TestCase):
    def setUp(self):
        self.counts = [8] * 60
        self.plan = {"owner": [0] * 10 + [1] * 50, "shared": 2, "name": "example"}
        self.model = {"backends": [[value] + [0] * 25 for value in (10000, 20000, 5000)],
                      "host": [1000] + [0] * 7}

    def test_parallel_critical_path_not_sum(self):
        prediction = m.predict(self.model, 120, self.counts, self.plan)
        self.assertEqual(prediction["total_us"], 21000)

    def test_no_free_nonresident_fallback(self):
        available = {"experts": [[False, True, False] for _ in range(60)], "shared": [False, True, False]}
        ranking = m.select(self.model, 120, self.counts, available)
        self.assertEqual(len(ranking), 1)
        self.assertEqual(ranking[0]["placement"]["name"], "npu_shared_npu")

    def test_preparation_cost_included(self):
        unavailable = {"experts": [[False] * 3 for _ in range(60)], "shared": [False] * 3}
        with self.assertRaises(ValueError):
            m.select(self.model, 120, self.counts, unavailable)
        ranking = m.select(self.model, 120, self.counts, unavailable, {"npu_shared_npu": 3000})
        self.assertEqual(ranking[0]["score_us"], 24000)
        with self.assertRaises(ValueError):
            m.select(self.model, 120, self.counts, unavailable, {"npu_shared_npu": -1})

    def test_routes_change_tail_expert_identity(self):
        counts = [0] * 60
        counts[2], counts[13], counts[21], counts[42] = 10, 3, 5, 2
        p = next(p for p in m.candidates(counts) if p["name"] == "npu_tail1_cpu_shared_gpu")
        self.assertEqual([i for i, b in enumerate(p["owner"]) if b == 0], [42])
        self.assertEqual(sum(m.group_work(counts, p)[1]), 20)


if __name__ == "__main__":
    unittest.main()
