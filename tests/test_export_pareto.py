import unittest

from tools.export_pareto import aggregate, pareto_members


class ParetoTests(unittest.TestCase):
    def test_constrained_pareto_excludes_infeasible_and_dominated(self):
        rows = [
            {"scene": "A", "method": "m1", "front_1m": 0.1, "t_lpips": 0.2, "temperature_mae_c": 1.0, "gaussian_count": 10, "feasible": True},
            {"scene": "A", "method": "m2", "front_1m": 0.2, "t_lpips": 0.3, "temperature_mae_c": 1.1, "gaussian_count": 11, "feasible": True},
            {"scene": "A", "method": "m3", "front_1m": 0.05, "t_lpips": 0.1, "temperature_mae_c": 0.9, "gaussian_count": 12, "feasible": False},
        ]
        members = pareto_members(rows, "t_lpips")
        self.assertEqual(members, {("A", "m1")})

    def test_macro_and_worst_scene_are_unweighted(self):
        rows = [
            {"scene": "A", "method": "m", "front_1m": 0.1, "t_lpips": 0.2, "temperature_mae_c": 1.0, "gaussian_count": 10, "feasible": True},
            {"scene": "B", "method": "m", "front_1m": 0.3, "t_lpips": 0.4, "temperature_mae_c": 2.0, "gaussian_count": 20, "feasible": False},
        ]
        macro, worst = aggregate(rows)
        self.assertAlmostEqual(macro[0]["front_1m"], 0.2)
        self.assertFalse(macro[0]["feasible"])
        self.assertEqual(worst[0]["worst_scene_temperature_mae_c"], "B")


if __name__ == "__main__":
    unittest.main()
