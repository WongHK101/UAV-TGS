from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from tools.geometric_repeatability.build_temperature_responsibility_bundle import (
    DIAGNOSTIC_KEYS,
    DIAGNOSTIC_SEMANTICS,
    OUTPUT_CONTRACT_MANIFEST,
    OUTPUT_MODEL_MANIFEST,
    PROTOCOL,
    SEMANTICS,
    SOURCE_RENDER_BINDING_PROTOCOL,
    TEMPERATURE_KEYS,
    UNDISTORTED_SCHEMA,
    build_temperature_responsibility_bundle,
)
from tools.geometric_repeatability.build_thermal_render_binding import (
    PROTOCOL as THERMAL_RENDER_BINDING_PROTOCOL,
    build_thermal_render_binding,
)
from tools.geometric_repeatability.evaluate_depth_definitions import (
    _load_temperature_arrays,
    _temperature_contract,
)
from tools.thermal_radiometry.palette_lut import (
    hot_iron_lut,
    lut_sha256,
    temperature_to_rgb,
    rgb_to_temperature,
)
from tools.thermal_radiometry.bind_formal_scene import binding_basis_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _refresh_receipt_hash(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("receipt_sha256", None)
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(path, payload)


class TemperatureResponsibilityBundleTests(unittest.TestCase):
    def _fixture(self, root: Path, *, view_count: int = 3) -> dict[str, Any]:
        fixture_root = root
        root = fixture_root / "inputs"
        root.mkdir(parents=True)
        source_root = root / "source"
        decoded_root = root / "decoded"
        target_root = root / "temperature_c"
        support_root = root / "valid_support"
        for path in (source_root, decoded_root, target_root, support_root):
            path.mkdir(parents=True)

        if view_count == 3:
            specs = (("0001", "train"), ("0002", "guard"), ("0003", "test"))
        elif view_count == 559:
            specs = tuple(
                (
                    f"{index:04d}",
                    "train" if index <= 470 else "guard" if index <= 495 else "test",
                )
                for index in range(1, 560)
            )
        else:
            raise ValueError(f"Unsupported fixture view_count={view_count}")
        split_counts = {
            split: sum(1 for _, observed in specs if observed == split)
            for split in ("train", "guard", "test")
        }
        formal_records = [
            {
                "scene": "InternalRoad",
                "pair_id": stem,
                "camera_name": f"{stem}.jpg",
                "thermal_camera_name": f"{stem}.png",
                "split": split,
            }
            for stem, split in specs
        ]
        formal_path = _write_json(
            root / "formal.json",
            {
                "scene": "InternalRoad",
                "counts": {"total": view_count, **split_counts},
                "records": formal_records,
            },
        )

        lut = hot_iron_lut()
        exact_and_off_lut = np.array(
            [[lut[0], lut[85]], [[123, 45, 67], lut[255]]], dtype=np.uint8
        )
        source_views: list[dict[str, Any]] = []
        binding_files: list[dict[str, Any]] = []
        undistorted_files: list[dict[str, Any]] = []
        targets: dict[str, np.ndarray] = {}
        source_renders: dict[str, np.ndarray] = {}
        for ordinal, (stem, split) in enumerate(specs):
            decoded = np.full((2, 2), 30.0 + ordinal, dtype=np.float32)
            decoded_path = decoded_root / f"{stem}.npy"
            np.save(decoded_path, decoded, allow_pickle=False)
            target = np.array(
                [
                    [11.125 + ordinal, 22.375 + ordinal],
                    [33.625 + ordinal, 44.875 + ordinal],
                ],
                dtype=np.float32,
            )
            target_path = target_root / f"{stem}.npy"
            np.save(target_path, target, allow_pickle=False)
            support = np.array([[True, True], [ordinal != 1, True]], dtype=np.bool_)
            support_path = support_root / f"{stem}.npy"
            np.save(support_path, support, allow_pickle=False)

            render_u8 = np.roll(exact_and_off_lut, shift=ordinal, axis=0)
            render_chw = np.moveaxis(render_u8.astype(np.float32) / 255.0, -1, 0)
            target_u8, _ = temperature_to_rgb(target, 10.0, 50.0, lut=lut)
            target_chw = np.moveaxis(target_u8.astype(np.float32) / 255.0, -1, 0)
            source_npz = source_root / f"{stem}.npz"
            arrays: dict[str, np.ndarray] = {
                "depth_expected_alpha_normalized": np.full((2, 2), 10.0, np.float32),
                "depth_transmittance_median": np.full((2, 2), 10.0, np.float32),
                "depth_max_contribution": np.full((2, 2), 10.0, np.float32),
                "accumulated_opacity": np.full((2, 2), 0.8, np.float32),
                "top_contributor_index": np.array([[0, 1], [2, 3]], np.int32),
                "top_contributor_weight": np.full((2, 2), 0.5, np.float32),
                "render_thermal_canonical": render_chw,
                "target_thermal_canonical": target_chw,
            }
            np.savez_compressed(source_npz, **arrays)
            source_views.append(
                {
                    "image_name": f"{stem}.png",
                    "width": 2,
                    "height": 2,
                    "fx": 2.0,
                    "fy": 2.0,
                    "cx": 1.0,
                    "cy": 1.0,
                    "camera_to_world": np.eye(4).tolist(),
                    "camera_sha256": hashlib.sha256(stem.encode()).hexdigest(),
                    "split": split,
                    "bound_split": split,
                    "npz_file": source_npz.name,
                    "npz_size_bytes": source_npz.stat().st_size,
                    "npz_sha256": _sha(source_npz),
                }
            )
            binding_files.append(
                {
                    "pair_id": stem,
                    "camera_name": f"{stem}.jpg",
                    "thermal_camera_name": f"{stem}.png",
                    "split": split,
                    "temperature_npy": f"InternalRoad/{stem}.npy",
                    "temperature_sha256": _sha(decoded_path),
                    "verified_sha256": _sha(decoded_path),
                    "raw_thermal": {"adapter_backend": "official-dji-irp"},
                }
            )
            undistorted_files.append(
                {
                    "image_name": f"{stem}.png",
                    "input_temperature": {
                        "relative_path": f"InternalRoad/{stem}.npy",
                        "sha256": _sha(decoded_path),
                        "dtype": "float32",
                        "shape": [2, 2],
                    },
                    "output_temperature": {
                        "relative_path": f"temperature_c/{stem}.npy",
                        "sha256": _sha(target_path),
                        "dtype": "float32",
                        "shape": [2, 2],
                    },
                    "valid_support": {
                        "relative_path": f"valid_support/{stem}.npy",
                        "sha256": _sha(support_path),
                        "dtype": "bool",
                        "shape": [2, 2],
                    },
                }
            )
            targets[stem] = target
            source_renders[stem] = render_u8

        model_point_cloud = source_root / "point_cloud.ply"
        model_point_cloud.write_bytes(b"formal-test-point-cloud")
        model_point_cloud_sha = _sha(model_point_cloud)
        source_manifest_path = _write_json(
            source_root / "split_manifest.json",
            {
                "bundle_type": "gaussian_probe_split_bundle_v1",
                "scene_name": "InternalRoad",
                "appearance_modality": "thermal_canonical",
                "depth_diagnostics": {"enabled": True, **DIAGNOSTIC_SEMANTICS},
                "gaussian_count": 4,
                "model_point_cloud": {
                    "path": str(model_point_cloud.resolve()),
                    "size_bytes": model_point_cloud.stat().st_size,
                    "sha256": model_point_cloud_sha,
                },
                "gaussian_index_anchor": {"sha256": "c" * 64},
                "formal_split_manifest_identity": {"sha256": _sha(formal_path)},
                "views": source_views,
            },
        )
        binding_payload = {
            "schema_name": "uav_tgs_formal_scene_decode_binding",
            "schema_version": 2,
            "scene": "InternalRoad",
            "status": "passed",
            "collection_hash": "2" * 64,
            "collection_split_hash": "4" * 64,
            "formal_rule_hash": "7" * 64,
            "scene_split_hash": "a" * 64,
            "scene_rule_hash": "9" * 64,
            "scene_manifest_sha256": "8" * 64,
            "collection_manifest_sha256": "3" * 64,
            "decode_manifest_sha256": "5" * 64,
            "decode_protocol_sha256": "d" * 64,
            "decode_protocol_hash": "6" * 64,
            "decode_protocol_basis": "legacy_frozen_v1",
            "formal_protocol_schema": "uav-tgs.radiometry-protocol.v1",
            "formal_decode_schema": "uav-tgs.temperature-decode.v1",
            "adapter": "builtin:dji_irp",
            "adapter_backend": "official-dji-irp",
            "adapter_executable_sha256": "b" * 64,
            "dirp_api_version": "1.8",
            "rjpeg_version": "H30T",
            "sfm_image_scope": "shared_sfm_all_images",
            "counts": {"total": view_count, **split_counts},
            "files": binding_files,
        }
        binding_payload["binding_hash"] = binding_basis_sha256(binding_payload)
        binding_path = _write_json(root / "binding.json", binding_payload)
        target_manifest_path = _write_json(
            root / "target_manifest.json",
            {
                "schema": UNDISTORTED_SCHEMA,
                "status": "complete",
                "files": undistorted_files,
            },
        )
        support_manifest_path = _write_json(
            root / "support_manifest.json",
            {
                "schema": UNDISTORTED_SCHEMA,
                "status": "complete",
                "files": undistorted_files,
            },
        )
        range_path = _write_json(
            root / "range.json",
            {
                "schema_name": "uav_tgs_train_only_scene_temperature_range",
                "scene": "InternalRoad",
                "Tmin": 10.0,
                "Tmax": 50.0,
                "source_split_manifest_sha256": _sha(formal_path),
            },
        )
        canonical_path = _write_json(
            root / "canonical_manifest.json",
            {
                "schema": "uav-tgs-canonical-hot-iron-v1",
                "temperature_range": {
                    "tmin_c": 10.0,
                    "tmax_c": 50.0,
                    "source": {"sha256": _sha(range_path)},
                },
                "palette": {
                    "name": "uav-tgs-hot-iron-v1",
                    "sha256_uint8_rgb": lut_sha256(),
                },
                "files": [
                    {"relative_id": stem, "relative_output": f"{stem}.png"}
                    for stem, _ in specs
                ],
            },
        )
        source_render_binding_path = (
            fixture_root / "render_binding_input" / "source_render_binding.json"
        )
        build_thermal_render_binding(
            source_model_manifest_path=source_manifest_path,
            formal_split_manifest_path=formal_path,
            range_manifest_path=range_path,
            canonical_manifest_path=canonical_path,
            output_path=source_render_binding_path,
        )
        return {
            "source_root": source_root,
            "source_manifest": source_manifest_path,
            "source_render_binding": source_render_binding_path,
            "formal": formal_path,
            "binding": binding_path,
            "target_root": target_root,
            "target_manifest": target_manifest_path,
            "support_root": support_root,
            "support_manifest": support_manifest_path,
            "range": range_path,
            "canonical": canonical_path,
            "targets": targets,
            "renders": source_renders,
            "view_count": view_count,
            "split_counts": split_counts,
        }

    def _build(self, fixture: dict[str, Any], output: Path) -> dict[str, Any]:
        return build_temperature_responsibility_bundle(
            source_model_manifest_path=fixture["source_manifest"],
            source_render_binding_manifest_path=fixture["source_render_binding"],
            formal_split_manifest_path=fixture["formal"],
            tsdk_binding_manifest_path=fixture["binding"],
            temperature_root=fixture["target_root"],
            temperature_manifest_path=fixture["target_manifest"],
            valid_support_root=fixture["support_root"],
            valid_support_manifest_path=fixture["support_manifest"],
            range_manifest_path=fixture["range"],
            canonical_manifest_path=fixture["canonical"],
            output_root=output,
            expected_view_count=fixture["view_count"],
            chunk_pixels=2,
        )

    def test_full_all_split_contract_is_evaluator_compatible_and_target_is_direct_float32(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root, view_count=559)
            result = self._build(fixture, root / "output")
            model_path = Path(result["model_manifest"])
            contract_path = Path(result["temperature_responsibility_manifest"])
            model = json.loads(model_path.read_text(encoding="utf-8"))
            contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))

            self.assertEqual(contract_payload["protocol"], PROTOCOL)
            self.assertEqual(contract_payload["semantics"], SEMANTICS)
            self.assertEqual(contract_payload["coverage"]["view_count"], 559)
            self.assertEqual(
                contract_payload["coverage"]["split_counts"],
                {"guard": 25, "test": 64, "train": 470},
            )
            self.assertEqual(
                contract_payload["palette"]["sha256_uint8_rgb"], lut_sha256()
            )
            self.assertEqual(
                len(contract_payload["producer_identity"]["script_sha256"]), 64
            )
            self.assertIn("git_dirty", contract_payload["producer_identity"])
            self.assertFalse(
                contract_payload["target_provenance"][
                    "png_or_palette_inverse_used_for_target"
                ]
            )
            self.assertEqual(len(model["views"]), 559)
            self.assertEqual(
                {row["split"] for row in contract_payload["views"]},
                {"train", "guard", "test"},
            )

            by_name = {row["image_name"]: row for row in model["views"]}
            evaluator_contract = _temperature_contract(
                contract_path,
                scene_name="InternalRoad",
                model_manifest_path=model_path,
                model_manifest=model,
                model_views=by_name,
                temperature_fields_present=True,
                formal_split_sha256=_sha(fixture["formal"]),
                formal_split_counts={"train": 470, "guard": 25, "test": 64},
            )
            self.assertIsNotNone(evaluator_contract)
            first = by_name["0001.png"]
            with np.load(
                model_path.parent / first["npz_file"], allow_pickle=False
            ) as data:
                self.assertEqual(
                    set(data.files),
                    set(DIAGNOSTIC_KEYS) | set(TEMPERATURE_KEYS.values()),
                )
                rendered, target, valid = _load_temperature_arrays(
                    data,
                    evaluator_contract,
                    expected_shape=(2, 2),
                    label="0001.png",
                )
            expected_render, _, _ = rgb_to_temperature(
                fixture["renders"]["0001"], 10.0, 50.0
            )
            np.testing.assert_array_equal(rendered, expected_render)
            # Values are deliberately not palette-bin centers; exact equality proves
            # the target came from the float32 NPY rather than PNG inversion.
            np.testing.assert_array_equal(target, fixture["targets"]["0001"])
            self.assertEqual(target.dtype, np.float32)
            self.assertEqual(valid.dtype, np.bool_)

    def test_off_lut_render_uses_nearest_fixed_palette_and_records_distance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            result = self._build(fixture, root / "output")
            contract = json.loads(
                Path(result["temperature_responsibility_manifest"]).read_text(
                    encoding="utf-8"
                )
            )
            first = next(row for row in contract["views"] if row["pair_id"] == "0001")
            expected_t, expected_distance, _ = rgb_to_temperature(
                fixture["renders"]["0001"], 10.0, 50.0
            )
            self.assertGreater(float(expected_distance[1, 0]), 0.0)
            self.assertAlmostEqual(
                first["off_lut"]["maximum"], float(np.max(expected_distance))
            )
            model = json.loads(
                Path(result["model_manifest"]).read_text(encoding="utf-8")
            )
            view = next(
                row for row in model["views"] if row["image_name"] == "0001.png"
            )
            with np.load(
                Path(result["model_manifest"]).parent / view["npz_file"],
                allow_pickle=False,
            ) as data:
                np.testing.assert_array_equal(
                    data[TEMPERATURE_KEYS["rendered"]], expected_t
                )

    def test_source_and_target_tampering_fail_before_publication(self) -> None:
        cases = ("source", "target")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self._fixture(root)
                if case == "source":
                    with (fixture["source_root"] / "0001.npz").open("ab") as handle:
                        handle.write(b"tamper")
                    expected = "Source view NPZ SHA-256 mismatch"
                else:
                    with (fixture["target_root"] / "0001.npy").open("ab") as handle:
                        handle.write(b"tamper")
                    expected = "temperature target NPY SHA-256 mismatch"
                with self.assertRaisesRegex(RuntimeError, expected):
                    self._build(fixture, root / "output")
                self.assertFalse((root / "output").exists())

    def test_render_palette_and_support_lineage_tampering_fail_closed(self) -> None:
        for case in (
            "render_binding",
            "canonical_palette",
            "support_lineage",
            "source_target_nonbyte",
            "source_target_wrong_byte",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self._fixture(root)
                if case == "render_binding":
                    payload = json.loads(
                        fixture["source_render_binding"].read_text(encoding="utf-8")
                    )
                    payload["range_manifest_sha256"] = "f" * 64
                    _write_json(fixture["source_render_binding"], payload)
                    _refresh_receipt_hash(fixture["source_render_binding"])
                    expected = "render binding mismatch"
                elif case == "canonical_palette":
                    payload = json.loads(
                        fixture["canonical"].read_text(encoding="utf-8")
                    )
                    payload["palette"]["sha256_uint8_rgb"] = "f" * 64
                    _write_json(fixture["canonical"], payload)
                    expected = "fixed Hot-Iron LUT"
                elif case == "support_lineage":
                    payload = json.loads(
                        fixture["support_manifest"].read_text(encoding="utf-8")
                    )
                    payload["files"][0]["input_temperature"]["sha256"] = "f" * 64
                    _write_json(fixture["support_manifest"], payload)
                    expected = "Valid-support/TSDK source lineage mismatch"
                else:
                    npz_path = fixture["source_root"] / "0001.npz"
                    with np.load(npz_path, allow_pickle=False) as data:
                        arrays = {key: np.asarray(data[key]) for key in data.files}
                    arrays["target_thermal_canonical"] = arrays[
                        "target_thermal_canonical"
                    ].copy()
                    arrays["target_thermal_canonical"][0, 0, 0] = (
                        0.123 if case == "source_target_nonbyte" else 1.0
                    )
                    np.savez_compressed(npz_path, **arrays)
                    source = json.loads(
                        fixture["source_manifest"].read_text(encoding="utf-8")
                    )
                    source["views"][0]["npz_size_bytes"] = npz_path.stat().st_size
                    source["views"][0]["npz_sha256"] = _sha(npz_path)
                    _write_json(fixture["source_manifest"], source)
                    binding = json.loads(
                        fixture["source_render_binding"].read_text(encoding="utf-8")
                    )
                    binding["source_model_manifest_sha256"] = _sha(
                        fixture["source_manifest"]
                    )
                    _write_json(fixture["source_render_binding"], binding)
                    _refresh_receipt_hash(fixture["source_render_binding"])
                    expected = (
                        "not byte-exact canonical PNG data"
                        if case == "source_target_nonbyte"
                        else "not the exact fixed-LUT/formal-range"
                    )
                with self.assertRaisesRegex(ValueError, expected):
                    self._build(fixture, root / "output")
                self.assertFalse((root / "output").exists())

    def test_split_shape_and_mask_contracts_fail_closed(self) -> None:
        cases = ("split", "shape", "mask")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self._fixture(root)
                if case == "split":
                    source = json.loads(
                        fixture["source_manifest"].read_text(encoding="utf-8")
                    )
                    source["views"][0]["bound_split"] = "test"
                    source["views"][0]["split"] = "test"
                    _write_json(fixture["source_manifest"], source)
                    binding = json.loads(
                        fixture["source_render_binding"].read_text(encoding="utf-8")
                    )
                    binding["source_model_manifest_sha256"] = _sha(
                        fixture["source_manifest"]
                    )
                    _write_json(fixture["source_render_binding"], binding)
                    _refresh_receipt_hash(fixture["source_render_binding"])
                    expected = "split label mismatch"
                elif case == "shape":
                    path = fixture["target_root"] / "0001.npy"
                    np.save(path, np.zeros((3, 2), np.float32), allow_pickle=False)
                    manifest = json.loads(
                        fixture["target_manifest"].read_text(encoding="utf-8")
                    )
                    manifest["files"][0]["output_temperature"]["sha256"] = _sha(path)
                    manifest["files"][0]["output_temperature"]["shape"] = [3, 2]
                    _write_json(fixture["target_manifest"], manifest)
                    support_manifest = json.loads(
                        fixture["support_manifest"].read_text(encoding="utf-8")
                    )
                    support_manifest["files"][0]["output_temperature"]["sha256"] = _sha(
                        path
                    )
                    support_manifest["files"][0]["output_temperature"]["shape"] = [3, 2]
                    _write_json(fixture["support_manifest"], support_manifest)
                    expected = "Temperature/support shape mismatch"
                else:
                    path = fixture["support_root"] / "0001.npy"
                    np.save(path, np.ones((2, 2), np.uint8), allow_pickle=False)
                    manifest = json.loads(
                        fixture["support_manifest"].read_text(encoding="utf-8")
                    )
                    manifest["files"][0]["valid_support"]["sha256"] = _sha(path)
                    manifest["files"][0]["valid_support"]["dtype"] = "uint8"
                    _write_json(fixture["support_manifest"], manifest)
                    expected = "Valid-support must be boolean"
                with self.assertRaisesRegex((ValueError, TypeError), expected):
                    self._build(fixture, root / "output")
                self.assertFalse((root / "output").exists())

    def test_rgb_bundle_or_rgb_arrays_cannot_be_mislabeled_as_temperature(self) -> None:
        for case in ("modality", "arrays"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self._fixture(root)
                source = json.loads(
                    fixture["source_manifest"].read_text(encoding="utf-8")
                )
                if case == "modality":
                    source["appearance_modality"] = "rgb"
                    expected = "RGB bundles must never be relabeled"
                else:
                    npz_path = fixture["source_root"] / "0001.npz"
                    with np.load(npz_path, allow_pickle=False) as data:
                        arrays = {key: np.asarray(data[key]) for key in data.files}
                    arrays["render_rgb"] = np.zeros((3, 2, 2), np.float32)
                    np.savez_compressed(npz_path, **arrays)
                    source["views"][0]["npz_size_bytes"] = npz_path.stat().st_size
                    source["views"][0]["npz_sha256"] = _sha(npz_path)
                    expected = "could be misreported as RGB"
                _write_json(fixture["source_manifest"], source)
                if case == "arrays":
                    binding = json.loads(
                        fixture["source_render_binding"].read_text(encoding="utf-8")
                    )
                    binding["source_model_manifest_sha256"] = _sha(
                        fixture["source_manifest"]
                    )
                    _write_json(fixture["source_render_binding"], binding)
                    _refresh_receipt_hash(fixture["source_render_binding"])
                with self.assertRaisesRegex(ValueError, expected):
                    self._build(fixture, root / "output")
                self.assertFalse((root / "output").exists())

    def test_binding_self_hash_and_output_tree_isolation_fail_closed(self) -> None:
        for case in ("binding_hash", "render_receipt", "output_isolation"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self._fixture(root)
                output = root / "output"
                if case == "binding_hash":
                    payload = json.loads(
                        fixture["binding"].read_text(encoding="utf-8")
                    )
                    payload["files"][0]["temperature_sha256"] = "f" * 64
                    _write_json(fixture["binding"], payload)
                    expected = "formal binding_hash mismatch"
                elif case == "render_receipt":
                    payload = json.loads(
                        fixture["source_render_binding"].read_text(encoding="utf-8")
                    )
                    payload["range_manifest_sha256"] = "f" * 64
                    _write_json(fixture["source_render_binding"], payload)
                    expected = "receipt_sha256 mismatch"
                else:
                    output = fixture["source_root"] / "derived_output"
                    expected = "tree-isolated"
                with self.assertRaisesRegex(ValueError, expected):
                    self._build(fixture, output)
                self.assertFalse(output.exists())

    def test_thermal_render_binding_builder_pins_all_formal_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            output = root / "receipt" / "thermal_render_binding.json"
            payload = build_thermal_render_binding(
                source_model_manifest_path=fixture["source_manifest"],
                formal_split_manifest_path=fixture["formal"],
                range_manifest_path=fixture["range"],
                canonical_manifest_path=fixture["canonical"],
                output_path=output,
            )
            self.assertEqual(payload["protocol"], THERMAL_RENDER_BINDING_PROTOCOL)
            self.assertEqual(payload["coverage"]["view_count"], 3)
            self.assertEqual(payload["formal_split_manifest_sha256"], _sha(fixture["formal"]))
            self.assertEqual(len(payload["receipt_sha256"]), 64)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
