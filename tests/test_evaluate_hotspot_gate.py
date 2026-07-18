import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.evaluate_hotspot_gate import evaluate


class HotspotGateEvaluationTests(unittest.TestCase):
    def _run(self, gate: np.ndarray, temperature: np.ndarray) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            render_root = root / "render"
            temperature_root = root / "temperature"
            support_root = root / "support"
            for directory in (render_root, temperature_root, support_root):
                directory.mkdir()
            np.save(render_root / "frame.npy", gate.astype(np.float32))
            np.save(temperature_root / "frame.npy", temperature.astype(np.float32))
            np.save(support_root / "frame.npy", np.ones(gate.shape, dtype=bool))
            heldout = root / "heldout.txt"
            heldout.write_text("frame.png\n", encoding="utf-8")
            return evaluate(
                argparse.Namespace(
                    render_root=str(render_root),
                    temperature_root=str(temperature_root),
                    support_root=str(support_root),
                    heldout_list=str(heldout),
                    threshold_c=30.0,
                    max_rank_pixels=1000,
                )
            )

    def test_perfect_gate_is_validated(self):
        temperature = np.arange(16, dtype=np.float32).reshape(4, 4) + 22.0
        gate = (temperature - temperature.min()) / (temperature.max() - temperature.min())
        result = self._run(gate, temperature)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["heldout_only"])
        self.assertEqual(result["semantic_status"], "temperature_gated_fusion_validated")
        self.assertAlmostEqual(result["global"]["rank_correlation_spearman"], 1.0)

    def test_constant_gate_is_downgraded(self):
        temperature = np.arange(16, dtype=np.float32).reshape(4, 4) + 22.0
        result = self._run(np.full((4, 4), 0.25, dtype=np.float32), temperature)
        self.assertEqual(
            result["semantic_status"],
            "downgraded_to_high_thermal_response_visualization",
        )
        self.assertIsNone(result["global"]["precision"])


if __name__ == "__main__":
    unittest.main()
