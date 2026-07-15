from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from tools import stage2_endpoint_audit


class Stage2EndpointAuditTests(unittest.TestCase):
    @staticmethod
    def _write_ply(
        path: Path,
        *,
        geometry_delta: float = 0.0,
        max_sh_degree: int = 1,
        inject_inactive: bool = False,
    ) -> None:
        names = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            *[f"f_rest_{index}" for index in range(45)],
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]
        data = np.zeros(3, dtype=[(name, "f4") for name in names])
        for index, name in enumerate(stage2_endpoint_audit.GEOMETRY_FIELDS):
            data[name] = np.arange(3, dtype=np.float32) + index
        data["x"] += np.float32(geometry_delta)
        active_rest = (max_sh_degree + 1) ** 2 - 1
        for channel in range(3):
            for coefficient in range(active_rest):
                data[f"f_rest_{channel * 15 + coefficient}"] = coefficient + 1
        if inject_inactive:
            data[f"f_rest_{active_rest}"] = 0.25
        path.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))

    def test_strict_sh1_endpoints_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rgb = root / "rgb.ply"
            self._write_ply(rgb, max_sh_degree=3)
            model = root / "model"
            for iteration in (40000, 50000, 60000):
                self._write_ply(
                    model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
                    max_sh_degree=1,
                )
            report = stage2_endpoint_audit.audit_endpoint_group(
                rgb_ply=rgb,
                model_root=model,
                group="F1",
                iterations=(40000, 50000, 60000),
                thermal_max_sh_degree=1,
                strict_geometry=True,
            )
            self.assertEqual(report["status"], "passed")
            self.assertTrue(all(item["geometry"]["passed"] for item in report["endpoints"]))
            self.assertTrue(
                all(
                    item["inactive_sh"]["all_inactive_fields_exact_zero"]
                    for item in report["endpoints"]
                )
            )

    def test_geometry_change_fails_strict_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rgb = root / "rgb.ply"
            endpoint = root / "model" / "point_cloud" / "iteration_40000" / "point_cloud.ply"
            self._write_ply(rgb, max_sh_degree=3)
            self._write_ply(endpoint, geometry_delta=1e-3, max_sh_degree=1)
            report = stage2_endpoint_audit.audit_endpoint_group(
                rgb_ply=rgb,
                model_root=root / "model",
                group="F1",
                iterations=(40000,),
                thermal_max_sh_degree=1,
                strict_geometry=True,
            )
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["endpoints"][0]["geometry"]["fields"]["x"]["unchanged"])

    def test_inactive_sh_nonzero_fails_even_without_strict_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rgb = root / "rgb.ply"
            endpoint = root / "model" / "point_cloud" / "iteration_40000" / "point_cloud.ply"
            self._write_ply(rgb, max_sh_degree=3)
            self._write_ply(endpoint, max_sh_degree=0, inject_inactive=True)
            report = stage2_endpoint_audit.audit_endpoint_group(
                rgb_ply=rgb,
                model_root=root / "model",
                group="F0",
                iterations=(40000,),
                thermal_max_sh_degree=0,
                strict_geometry=False,
            )
            self.assertEqual(report["status"], "failed")
            self.assertIn("f_rest_0", report["endpoints"][0]["inactive_sh"]["nonzero_fields"])


if __name__ == "__main__":
    unittest.main()
