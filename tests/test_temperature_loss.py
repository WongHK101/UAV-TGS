import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tools.thermal_radiometry.palette_lut import hot_iron_lut
from utils.temperature_loss import (
    TemperatureTargetStore,
    adjacent_lut_tau,
    canonical_lut_tensor,
    soft_lut_inverse,
    temperature_consistency_losses,
)


class TemperatureLossTests(unittest.TestCase):
    def test_soft_inverse_is_monotonic_on_exact_lut(self):
        lut = torch.from_numpy(hot_iron_lut().astype(np.float32) / 255.0)
        image = lut.transpose(0, 1).reshape(3, 1, 256).requires_grad_(True)
        tau = adjacent_lut_tau(lut)
        recovered = soft_lut_inverse(image, tau=tau, chunk_pixels=17).reshape(-1)
        self.assertTrue(bool((recovered[1:] >= recovered[:-1]).all()))
        self.assertLess(float(torch.mean(torch.abs(recovered - torch.linspace(0, 1, 256)))), 0.004)
        recovered.mean().backward()
        self.assertTrue(bool(torch.isfinite(image.grad).all()))

    def test_losses_are_masked_and_differentiable(self):
        lut = canonical_lut_tensor(torch.device("cpu"), torch.float32)
        image = lut[100].view(3, 1, 1).expand(3, 8, 8).clone().requires_grad_(True)
        target = torch.full((1, 8, 8), 100.0 / 255.0)
        mask = torch.ones((1, 8, 8))
        result = temperature_consistency_losses(image, target, mask=mask, chunk_pixels=11)
        total = result["scalar"] + result["gradient"]
        total.backward()
        self.assertTrue(bool(torch.isfinite(image.grad).all()))
        self.assertLess(float(result["scalar"]), 0.01)

    def test_target_store_resizes_float_temperature_and_bool_support(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            temperature_root = root / "temperature"
            support_root = root / "support"
            temperature_root.mkdir()
            support_root.mkdir()
            np.save(temperature_root / "0001.npy", np.linspace(10, 20, 24, dtype=np.float32).reshape(4, 6))
            support = np.ones((4, 6), dtype=np.bool_)
            support[0, 0] = False
            np.save(support_root / "0001.npy", support)
            manifest = root / "range.json"
            manifest.write_text(json.dumps({"Tmin": 10.0, "Tmax": 20.0}), encoding="utf-8")
            store = TemperatureTargetStore(temperature_root, manifest, support_root)
            target, mask = store.get("0001", 2, 3, "cpu")
            self.assertEqual(tuple(target.shape), (1, 2, 3))
            self.assertEqual(tuple(mask.shape), (1, 2, 3))
            self.assertGreaterEqual(float(target.min()), 0.0)
            self.assertLessEqual(float(target.max()), 1.0)
            self.assertEqual(float(mask[0, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
