from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from tools import audit_opacity_adaptation


class OpacityAdaptationAuditTests(unittest.TestCase):
    @staticmethod
    def _write_ply(
        path: Path,
        raw_opacity: np.ndarray,
        *,
        changed_field: str | None = None,
    ) -> None:
        raw_opacity = np.asarray(raw_opacity, dtype=np.float32)
        names = [*audit_opacity_adaptation.STRUCTURAL_FIELDS, "opacity"]
        vertices = np.zeros(raw_opacity.size, dtype=[(name, "f4") for name in names])
        base = np.arange(raw_opacity.size, dtype=np.float32)
        for field_index, field in enumerate(audit_opacity_adaptation.STRUCTURAL_FIELDS):
            vertices[field] = base + np.float32(field_index + 1) / np.float32(16.0)
        if changed_field is not None:
            vertices[changed_field][0] += np.float32(0.125)
        vertices["opacity"] = raw_opacity
        path.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def _write_proxy_manifest(
        cls,
        root: Path,
        stems: list[str],
        *,
        value: float,
    ) -> Path:
        proxy_root = root / "opacity_proxy"
        proxy_root.mkdir(parents=True, exist_ok=True)
        entries = []
        for index, stem in enumerate(stems):
            # Two pixels per view exercise pixel-micro and frame-macro aggregation.
            array = np.asarray(
                [[value + index * 1e-4, value + index * 1e-4]],
                dtype=np.float32,
            )
            proxy_path = proxy_root / f"{stem}.npy"
            np.save(proxy_path, array, allow_pickle=False)
            entries.append(
                {
                    "source_image_name": f"{stem}.png",
                    "output": {"opacity_proxy": f"opacity_proxy/{stem}.npy"},
                    "output_sha256": {"opacity_proxy": cls._sha256(proxy_path)},
                }
            )
        manifest = root / "render_mapping.json"
        manifest.write_text(
            json.dumps(
                {
                    "opacity_proxy_saved": True,
                    "opacity_proxy_semantics": (
                        "black_bg_plus_white_override_color_render"
                    ),
                    "entries": entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return manifest

    def test_sigmoid_statistics_thresholds_and_structural_field_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchor_ply = root / "anchor.ply"
            a3_ply = root / "a3.ply"
            anchor_raw = np.zeros(4, dtype=np.float32)
            target_activated = np.asarray([0.50, 0.52, 0.56, 0.70], dtype=np.float64)
            a3_raw = np.log(target_activated / (1.0 - target_activated)).astype(np.float32)
            self._write_ply(anchor_ply, anchor_raw)
            self._write_ply(a3_ply, a3_raw)

            payload, rows = audit_opacity_adaptation.run_audit(
                anchor_ply=anchor_ply,
                a3_ply=a3_ply,
                anchor_ply_sha256=self._sha256(anchor_ply),
                a3_ply_sha256=self._sha256(a3_ply),
            )

            self.assertEqual(payload["status"], "passed")
            self.assertEqual(len(rows), 1)
            ply_audit = payload["ply_audit"]
            self.assertTrue(ply_audit["all_structural_fields_exact"])
            structural = ply_audit["structural_fields"]
            self.assertEqual(set(structural), set(audit_opacity_adaptation.STRUCTURAL_FIELDS))
            field_hashes = []
            for field in audit_opacity_adaptation.STRUCTURAL_FIELDS:
                item = structural[field]
                self.assertTrue(item["exact"], field)
                self.assertEqual(item["dtype_anchor"], item["dtype_a3"])
                self.assertEqual(item["shape"], [4])
                self.assertEqual(item["anchor_sha256"], item["a3_sha256"])
                self.assertEqual(len(item["anchor_sha256"]), 64)
                field_hashes.append(item["anchor_sha256"])
            # Field names and values both participate in each per-field digest.
            self.assertEqual(len(set(field_hashes)), len(field_hashes))

            expected_anchor = np.full(4, 0.5, dtype=np.float64)
            expected_a3 = 1.0 / (1.0 + np.exp(-a3_raw.astype(np.float64)))
            expected_abs_delta = np.abs(expected_a3 - expected_anchor)
            activated = ply_audit["activated_opacity"]
            self.assertEqual(activated["semantics"], "sigmoid(raw PLY opacity logit)")
            self.assertAlmostEqual(activated["anchor"]["mean"], 0.5, places=14)
            for key, expected in (
                ("mean", np.mean(expected_a3)),
                ("median", np.median(expected_a3)),
                ("p95", np.percentile(expected_a3, 95)),
                ("p99", np.percentile(expected_a3, 99)),
                ("max", np.max(expected_a3)),
            ):
                self.assertAlmostEqual(activated["a3"][key], float(expected), places=14)
            absolute = activated["a3_minus_anchor"]["absolute"]
            for key, expected in (
                ("mean", np.mean(expected_abs_delta)),
                ("median", np.median(expected_abs_delta)),
                ("p95", np.percentile(expected_abs_delta, 95)),
                ("p99", np.percentile(expected_abs_delta, 99)),
                ("max", np.max(expected_abs_delta)),
            ):
                self.assertAlmostEqual(absolute[key], float(expected), places=14)
            self.assertEqual(
                activated["a3_minus_anchor"]["fractions_abs_delta_gt"],
                {"0.01": 0.75, "0.05": 0.5, "0.10": 0.25},
            )

    def test_catastrophic_saturation_fraction_is_inclusive_at_0_99(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchor_ply = root / "anchor.ply"
            a3_ply = root / "a3.ply"
            self._write_ply(anchor_ply, np.zeros(100, dtype=np.float32))
            activated = np.concatenate(
                [np.full(99, 1e-5, dtype=np.float64), np.asarray([0.5])]
            )
            raw = np.log(activated / (1.0 - activated)).astype(np.float32)
            self._write_ply(a3_ply, raw)
            payload, _ = audit_opacity_adaptation.run_audit(
                anchor_ply=anchor_ply,
                a3_ply=a3_ply,
                anchor_ply_sha256=self._sha256(anchor_ply),
                a3_ply_sha256=self._sha256(a3_ply),
            )
            saturation = payload["ply_audit"]["activated_opacity"][
                "catastrophic_saturation"
            ]
            self.assertEqual(
                saturation["thresholds"],
                {
                    "low_activated_opacity_lte": 1e-4,
                    "high_activated_opacity_gte": 1.0 - 1e-4,
                    "catastrophic_fraction_gte": 0.99,
                },
            )
            self.assertAlmostEqual(saturation["a3"]["low_fraction"], 0.99)
            self.assertEqual(saturation["a3"]["high_fraction"], 0.0)
            self.assertTrue(saturation["a3"]["detected"])
            self.assertTrue(saturation["detected"])

    def test_catastrophic_saturation_below_0_99_is_not_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchor_ply = root / "anchor.ply"
            a3_ply = root / "a3.ply"
            self._write_ply(anchor_ply, np.zeros(100, dtype=np.float32))
            activated = np.concatenate(
                [np.full(98, 1.0 - 1e-5, dtype=np.float64), np.full(2, 0.5)]
            )
            raw = np.log(activated / (1.0 - activated)).astype(np.float32)
            self._write_ply(a3_ply, raw)
            payload, _ = audit_opacity_adaptation.run_audit(
                anchor_ply=anchor_ply,
                a3_ply=a3_ply,
                anchor_ply_sha256=self._sha256(anchor_ply),
                a3_ply_sha256=self._sha256(a3_ply),
            )
            saturation = payload["ply_audit"]["activated_opacity"][
                "catastrophic_saturation"
            ]
            self.assertAlmostEqual(saturation["a3"]["high_fraction"], 0.98)
            self.assertFalse(saturation["detected"])

    def test_high_endpoint_saturation_is_inclusive_at_0_99(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchor_ply = root / "anchor.ply"
            a3_ply = root / "a3.ply"
            self._write_ply(anchor_ply, np.zeros(100, dtype=np.float32))
            activated = np.concatenate(
                [np.full(99, 1.0 - 1e-5, dtype=np.float64), np.asarray([0.5])]
            )
            raw = np.log(activated / (1.0 - activated)).astype(np.float32)
            self._write_ply(a3_ply, raw)
            payload, _ = audit_opacity_adaptation.run_audit(
                anchor_ply=anchor_ply,
                a3_ply=a3_ply,
                anchor_ply_sha256=self._sha256(anchor_ply),
                a3_ply_sha256=self._sha256(a3_ply),
            )
            saturation = payload["ply_audit"]["activated_opacity"][
                "catastrophic_saturation"
            ]
            self.assertAlmostEqual(saturation["a3"]["high_fraction"], 0.99)
            self.assertTrue(saturation["detected"])

    def test_nonfinite_opacity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchor_ply = root / "anchor.ply"
            a3_ply = root / "a3.ply"
            self._write_ply(anchor_ply, np.zeros(2, dtype=np.float32))
            self._write_ply(a3_ply, np.asarray([0.0, np.nan], dtype=np.float32))
            with self.assertRaisesRegex(
                audit_opacity_adaptation.OpacityAuditError,
                "A3 raw opacity contains non-finite values",
            ):
                audit_opacity_adaptation.run_audit(
                    anchor_ply=anchor_ply,
                    a3_ply=a3_ply,
                    anchor_ply_sha256=self._sha256(anchor_ply),
                    a3_ply_sha256=self._sha256(a3_ply),
                )

    def test_structural_change_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchor_ply = root / "anchor.ply"
            a3_ply = root / "a3.ply"
            raw = np.zeros(3, dtype=np.float32)
            self._write_ply(anchor_ply, raw)
            self._write_ply(a3_ply, raw, changed_field="scale_1")

            with self.assertRaisesRegex(
                audit_opacity_adaptation.OpacityAuditError,
                "structural field is not exact: scale_1",
            ):
                audit_opacity_adaptation.run_audit(
                    anchor_ply=anchor_ply,
                    a3_ply=a3_ply,
                    anchor_ply_sha256=self._sha256(anchor_ply),
                    a3_ply_sha256=self._sha256(a3_ply),
                )

    def test_ply_sha256_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchor_ply = root / "anchor.ply"
            a3_ply = root / "a3.ply"
            raw = np.zeros(2, dtype=np.float32)
            self._write_ply(anchor_ply, raw)
            self._write_ply(a3_ply, raw)

            with self.assertRaisesRegex(
                audit_opacity_adaptation.OpacityAuditError,
                "anchor PLY SHA-256 mismatch",
            ):
                audit_opacity_adaptation.run_audit(
                    anchor_ply=anchor_ply,
                    a3_ply=a3_ply,
                    anchor_ply_sha256="0" * 64,
                    a3_ply_sha256=self._sha256(a3_ply),
                )

    def test_80_view_proxy_manifests_and_cli_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchor_ply = root / "anchor.ply"
            a3_ply = root / "a3.ply"
            raw = np.zeros(2, dtype=np.float32)
            self._write_ply(anchor_ply, raw)
            self._write_ply(a3_ply, raw)
            stems = [f"{index:04d}" for index in range(80)]
            anchor_manifest = self._write_proxy_manifest(
                root / "anchor_render", stems, value=0.20
            )
            a3_manifest = self._write_proxy_manifest(
                root / "a3_render", stems, value=0.22
            )
            report_path = root / "audit.json"
            csv_path = root / "audit.csv"

            result = audit_opacity_adaptation.main(
                [
                    "--anchor-ply",
                    str(anchor_ply),
                    "--a3-ply",
                    str(a3_ply),
                    "--anchor-ply-sha256",
                    self._sha256(anchor_ply),
                    "--a3-ply-sha256",
                    self._sha256(a3_ply),
                    "--anchor-opacity-manifest",
                    str(anchor_manifest),
                    "--a3-opacity-manifest",
                    str(a3_manifest),
                    "--anchor-opacity-manifest-sha256",
                    self._sha256(anchor_manifest),
                    "--a3-opacity-manifest-sha256",
                    self._sha256(a3_manifest),
                    "--expected-proxy-views",
                    "80",
                    "--report",
                    str(report_path),
                    "--csv",
                    str(csv_path),
                ]
            )
            self.assertEqual(result, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            proxy = report["opacity_proxy_audit"]
            self.assertEqual(proxy["view_count"], 80)
            self.assertEqual(proxy["anchor_manifest"]["view_count"], 80)
            self.assertEqual(proxy["a3_manifest"]["view_count"], 80)
            self.assertEqual(proxy["anchor_manifest"]["directory_npy_count"], 80)
            self.assertEqual(proxy["a3_manifest"]["directory_npy_count"], 80)
            micro = proxy["pixel_micro"]
            expected_delta = float(np.float32(0.22) - np.float32(0.20))
            self.assertEqual(micro["count"], 160)
            self.assertAlmostEqual(micro["signed_delta_mean"], expected_delta, places=7)
            self.assertAlmostEqual(micro["abs_delta_mean"], expected_delta, places=7)
            self.assertAlmostEqual(micro["rmse"], expected_delta, places=7)
            self.assertEqual(micro["fraction_abs_delta_gt_0.01"], 1.0)
            self.assertEqual(micro["fraction_abs_delta_gt_0.05"], 0.0)
            self.assertEqual(micro["fraction_abs_delta_gt_0.10"], 0.0)
            self.assertEqual(proxy["frame_macro"]["view_count"], 80)
            self.assertAlmostEqual(
                proxy["frame_macro"]["mean_abs_delta_mean"], expected_delta, places=7
            )

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 82)  # PLY + 80 views + pixel-micro.
            self.assertEqual(csv_rows[-1]["scope"], "opacity_proxy_pixel_micro")
            self.assertEqual(csv_rows[-1]["view_id"], "ALL")

    def test_dynamic_64_and_96_view_proxy_manifests(self) -> None:
        for view_count in (64, 96):
            with self.subTest(view_count=view_count), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                stems = [f"{index:04d}" for index in range(view_count)]
                anchor_manifest = self._write_proxy_manifest(
                    root / "anchor_render", stems, value=0.20
                )
                a3_manifest = self._write_proxy_manifest(
                    root / "a3_render", stems, value=0.22
                )
                payload, rows = audit_opacity_adaptation.compare_opacity_proxy_manifests(
                    anchor_manifest, a3_manifest, expected_proxy_views=view_count
                )
                self.assertEqual(payload["view_count"], view_count)
                self.assertEqual(payload["expected_proxy_views"], view_count)
                self.assertEqual(
                    payload["anchor_manifest"]["directory_npy_count"], view_count
                )
                self.assertEqual(
                    payload["a3_manifest"]["directory_npy_count"], view_count
                )
                self.assertEqual(len(rows), view_count)

    def test_proxy_manifest_directory_extra_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stems = ["0001", "0002"]
            anchor_manifest = self._write_proxy_manifest(
                root / "anchor_render", stems, value=0.20
            )
            a3_manifest = self._write_proxy_manifest(
                root / "a3_render", stems, value=0.22
            )
            np.save(
                anchor_manifest.parent / "opacity_proxy" / "unlisted.npy",
                np.zeros((1, 1), dtype=np.float32),
                allow_pickle=False,
            )
            with self.assertRaisesRegex(
                audit_opacity_adaptation.OpacityAuditError,
                "manifest/directory files differ",
            ):
                audit_opacity_adaptation.compare_opacity_proxy_manifests(
                    anchor_manifest, a3_manifest
                )

    def test_expected_proxy_view_count_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stems = ["0001", "0002"]
            anchor_manifest = self._write_proxy_manifest(
                root / "anchor_render", stems, value=0.20
            )
            a3_manifest = self._write_proxy_manifest(
                root / "a3_render", stems, value=0.22
            )
            with self.assertRaisesRegex(
                audit_opacity_adaptation.OpacityAuditError,
                "view count mismatch: expected=64 actual=2",
            ):
                audit_opacity_adaptation.compare_opacity_proxy_manifests(
                    anchor_manifest, a3_manifest, expected_proxy_views=64
                )

    def test_proxy_manifest_declared_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stems = [f"{index:04d}" for index in range(80)]
            anchor_manifest = self._write_proxy_manifest(
                root / "anchor_render", stems, value=0.20
            )
            a3_manifest = self._write_proxy_manifest(
                root / "a3_render", stems, value=0.22
            )
            manifest_data = json.loads(a3_manifest.read_text(encoding="utf-8"))
            manifest_data["entries"][17]["output_sha256"]["opacity_proxy"] = "f" * 64
            a3_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

            with self.assertRaisesRegex(
                audit_opacity_adaptation.OpacityAuditError,
                "opacity proxy SHA-256 mismatch for 0017.png",
            ):
                audit_opacity_adaptation.compare_opacity_proxy_manifests(
                    anchor_manifest, a3_manifest
                )

    def test_proxy_manifest_view_set_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchor_stems = [f"{index:04d}" for index in range(80)]
            a3_stems = [f"{index:04d}" for index in range(1, 81)]
            anchor_manifest = self._write_proxy_manifest(
                root / "anchor_render", anchor_stems, value=0.20
            )
            a3_manifest = self._write_proxy_manifest(
                root / "a3_render", a3_stems, value=0.22
            )

            with self.assertRaisesRegex(
                audit_opacity_adaptation.OpacityAuditError,
                "opacity proxy view sets differ",
            ):
                audit_opacity_adaptation.compare_opacity_proxy_manifests(
                    anchor_manifest, a3_manifest
                )


if __name__ == "__main__":
    unittest.main()
