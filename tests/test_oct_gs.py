import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from oct_gs.field import (
    OCTConfig,
    OCTGaussianField,
    build_oct_optimizer,
    verify_field_only_optimizer,
)
from oct_gs.formal import (
    FORMAL_EXPERIMENT_RECIPE,
    FormalOCTTargetStore,
    build_formal_binding,
    sha256_file,
    sha256_json,
)
from oct_gs.losses import OCTLossWeights, oct_rendering_loss
from oct_gs.protocol import (
    BuildingGradientCalibrator,
    CALIBRATION_SCHEMA,
    OCTStageCostTracker,
    capture_occupancy_snapshot,
    inspect_oct_checkpoint,
    load_frozen_calibration,
    load_oct_checkpoint,
    restore_oct_optimizer_state,
    save_oct_checkpoint,
    validate_training_source_provenance,
    verify_oct_field_finite,
    verify_oct_post_step_finite,
    verify_occupancy_snapshot,
    write_oct_protocol_manifest,
)
from oct_gs.radiance import (
    BandRadianceProxy,
    METHOD_SEMANTICS,
    TARGET_SEMANTICS,
    temperature_to_hot_iron,
)
from oct_gs.rendering import FrozenGaussianView, OCTRendererContext, weak_axis_from_anchor
from tools.thermal_radiometry.palette_lut import (
    hot_iron_lut,
    indices_to_temperature,
    lut_sha256,
    temperature_to_indices,
)
from tools.evaluate_oct_gs_formal_v2 import (
    ALLOWED_POST_TRAINING_PATHS,
    EVALUATION_SCHEMA,
    FROZEN_TRAINING_COMMIT,
    FORMAL_HOTSPOT_BINS,
    FORMAL_HOTSPOT_QUANTILE,
    _checkpoint_compatibility,
    _exact_display_temperature_c,
    _histogram_auprc,
    _load_and_validate_endpoint_receipt,
    _occupancy_invariant_evidence,
    _population_variance_from_moments,
    _training_source_compatibility,
    _update_visible_temperature_moments,
    _validate_formal_hotspot_threshold,
)
from tools.oct_gs_formal import (
    HOTSPOT_SCHEMA,
    Runtime,
    _copy_immutable_reference,
    _formal_source_provenance,
    _load_hotspot_threshold,
    _formal_metadata_cameras,
    _metadata_camera,
    _qvec_to_rotation,
    _remaining_sequence,
    _require_matching_source_provenance,
    _require_isolated_output,
    command_eval,
)
from utils.camera_sequence import build_sequence_manifest, save_sequence_manifest
from utils.graphics_utils import focal2fov


class DummyAnchor:
    def __init__(self, count: int = 4):
        self._xyz = torch.nn.Parameter(
            torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 10.0
        )
        self._scaling = torch.nn.Parameter(torch.zeros((count, 3), dtype=torch.float32))
        self._rotation = torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(count, 1)
        )
        self._opacity = torch.nn.Parameter(torch.zeros((count, 1), dtype=torch.float32))
        self.active_sh_degree = 3
        self.max_sh_degree = 3

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation, dim=1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)


class DummyCamera:
    def __init__(self):
        self.camera_center = torch.tensor([0.0, 0.0, 3.0], dtype=torch.float32)


def fake_colmap_camera(name: str, *, camera_id: int = 7):
    extrinsic = SimpleNamespace(
        name=name,
        camera_id=camera_id,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        tvec=np.asarray([1.0, -2.0, 3.0], dtype=np.float64),
    )
    intrinsic = SimpleNamespace(
        id=camera_id,
        model="PINHOLE",
        width=1280,
        height=1024,
        params=np.asarray([900.0, 880.0, 640.0, 512.0], dtype=np.float64),
    )
    return extrinsic, intrinsic


def weighted_fake_renderer(weights):
    weights = torch.as_tensor(weights, dtype=torch.float32)

    def render(camera, anchor, pipe, background, **kwargs):
        del camera, pipe
        colors = kwargs["override_color"]
        assert isinstance(anchor, FrozenGaussianView)
        weights_on_device = weights.to(colors)
        value = (weights_on_device[:, None] * colors).sum(dim=0)
        value = value + (1.0 - weights_on_device.sum()) * background.to(colors)
        image = value[:, None, None].expand(3, 2, 2)
        package = {
            "render": image,
            "depth": torch.ones((1, 2, 2), device=colors.device),
            "radii": torch.ones(colors.shape[0], device=colors.device),
            "visibility_filter": torch.arange(colors.shape[0], device=colors.device),
            "viewspace_points": torch.zeros_like(anchor.get_xyz),
        }
        return package

    return render


def fake_training_source_provenance() -> dict:
    files = [
        {
            "path": "tools/oct_gs_formal.py",
            "sha256": "1" * 64,
            "bytes": 123,
        }
    ]
    return {
        "schema": "uav-tgs-oct-training-source-v1",
        "git_commit": "2" * 40,
        "git_clean": True,
        "git_status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
        "files": files,
        "files_sha256": sha256_json(files),
    }


def run_toy_adam_steps(
    field: OCTGaussianField,
    optimizer: torch.optim.Optimizer,
    start_step: int,
    end_step: int,
) -> None:
    for step in range(start_step + 1, end_step + 1):
        optimizer.zero_grad(set_to_none=True)
        target_base = field.raw_base_temperature.new_full(
            field.raw_base_temperature.shape,
            float((step % 29) - 14) / 20.0,
        )
        loss = (field.raw_base_temperature - target_base).square().mean()
        if field.raw_residual_amplitude is not None:
            target_residual = field.raw_residual_amplitude.new_full(
                field.raw_residual_amplitude.shape,
                float((step % 17) - 8) / 25.0,
            )
            loss = loss + 0.3 * (
                field.raw_residual_amplitude - target_residual
            ).square().mean()
        loss.backward()
        optimizer.step()
        verify_oct_post_step_finite(field, optimizer, step)


class FormalFixture:
    def __init__(self, root: Path, *, scene="Building", count=2):
        self.root = root
        self.scene = scene
        self.anchor = DummyAnchor(count)
        self.snapshot = capture_occupancy_snapshot(self.anchor)
        self.anchor_artifact = root / "point_cloud.ply"
        self.anchor_artifact.write_bytes(b"fake-ply-anchor")
        self.target_root = root / "temperature"
        self.color_root = root / "canonical"
        self.support_root = root / "support"
        self.evaluation_support_root = root / "evaluation_support"
        for directory in (
            self.target_root,
            self.color_root,
            self.support_root,
            self.evaluation_support_root / "bool",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.pairs = (("train", "train"), ("test", "test"), ("guard", "guard"))
        canonical_rows = []
        support_rows = []
        decode_rows = []
        protocol_rows = []
        self.protocol_hash = "1" * 64
        for index, (pair, split) in enumerate(self.pairs):
            temperature = np.full((4, 5), 10.0 + index, dtype=np.float32)
            target_path = self.target_root / f"{pair}.npy"
            np.save(target_path, temperature, allow_pickle=False)
            color = temperature_to_hot_iron(
                torch.from_numpy(temperature), 0.0, 30.0, exact_display=True
            ).mul(255).round().to(torch.uint8).numpy()
            color_path = self.color_root / f"{pair}.png"
            Image.fromarray(color, mode="RGB").save(color_path)
            support = np.ones((4, 5), dtype=np.bool_)
            support_path = self.support_root / f"{pair}.npy"
            np.save(support_path, support, allow_pickle=False)
            target_sha = sha256_file(target_path)
            canonical_rows.append(
                {
                    "pair_id": pair,
                    "relative_input": target_path.name,
                    "input_sha256": target_sha,
                    "temperature_dtype": "float32",
                    "relative_output": color_path.name,
                    "output_sha256": sha256_file(color_path),
                }
            )
            support_rows.append(
                {
                    "image_name": f"{pair}.png",
                    "input_temperature": {
                        "dtype": "float32",
                        "relative_path": target_path.name,
                        "sha256": target_sha,
                    },
                    "output_temperature": {
                        "dtype": "float32",
                        "relative_path": target_path.name,
                        "sha256": target_sha,
                    },
                    "valid_support": {
                        "dtype": "bool",
                        "relative_path": support_path.name,
                        "sha256": sha256_file(support_path),
                    },
                }
            )
            parameters = {
                "ambient_c": {"value": 25.0, "source": "benchmark_assumption"},
                "distance_m": {"value": 5.0 + index, "source": "lrf"},
                "emissivity": {"value": 0.95, "source": "benchmark_assumption"},
                "humidity_percent": {"value": 70.0, "source": "benchmark_assumption"},
                "reflected_c": {"value": 23.0, "source": "benchmark_assumption"},
            }
            decode_rows.append(
                {
                    "pair_id": pair,
                    "scene": scene,
                    "success": True,
                    "dtype": "float32",
                    "output_sha256": target_sha,
                }
            )
            protocol_rows.append(
                {
                    "pair_id": pair,
                    "scene": scene,
                    "schema_version": "uav-tgs.radiometry-protocol.v1",
                    "protocol_hash": self.protocol_hash,
                    "decode_parameters": parameters,
                }
            )
        self.decode_manifest = root / "decode.jsonl"
        self.decode_protocol = root / "protocol.jsonl"
        self.decode_manifest.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in decode_rows), encoding="utf-8"
        )
        self.decode_protocol.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in protocol_rows), encoding="utf-8"
        )
        self.bound = root / "bound.json"
        split_hash = "2" * 64
        bound_payload = {
            "scene": scene,
            "split_hash": split_hash,
            "counts": {"total": 3, "train": 1, "test": 1, "guard": 1},
            "decode_binding": {
                "adapter_backend": "official-dji-irp",
                "adapter_executable_sha256": "3" * 64,
                "decode_manifest_sha256": sha256_file(self.decode_manifest),
                "decode_protocol_sha256": sha256_file(self.decode_protocol),
                "protocol_hash": self.protocol_hash,
                "verified_temperature_file_hashes": True,
            },
            "records": [
                {"pair_id": pair, "split": split, "thermal_camera_name": f"{pair}.png"}
                for pair, split in self.pairs
            ],
        }
        self.bound.write_text(json.dumps(bound_payload), encoding="utf-8")
        self.range_manifest = root / "range.json"
        range_basis = {
            "scene": scene,
            "split_hash": split_hash,
            "configuration": {
                "source_split": "train",
                "rule": "fixture",
                "guard_role": "not_read",
                "test_role": "qa_only_not_used_for_estimation",
            },
            "Tmin": 0.0,
            "Tmax": 30.0,
        }
        range_payload = {
            "schema_name": "uav_tgs_train_only_scene_temperature_range",
            "schema_version": 1,
            **range_basis,
            "source_split_manifest_sha256": sha256_file(self.bound),
            "range_hash": sha256_json(range_basis),
            "train_estimation": {"frame_count": 1},
            "clipping_stats": {
                "train": {"frame_count": 1},
                "test": {"frame_count": 1},
            },
            "per_frame_quantiles": [
                {"pair_id": "train", "split": "train"},
                {"pair_id": "test", "split": "test"},
            ],
        }
        self.range_manifest.write_text(json.dumps(range_payload), encoding="utf-8")
        self.canonical_manifest = root / "canonical.json"
        canonical_payload = {
            "schema": "uav-tgs-canonical-hot-iron-v1",
            "status": "complete",
            "palette": {"sha256_uint8_rgb": lut_sha256()},
            "image_encoding": {"format": "PNG", "mode": "RGB", "lossless": True, "gamma": 1.0},
            "temperature_range": {
                "tmin_c": 0.0,
                "tmax_c": 30.0,
                "source": {"sha256": sha256_file(self.range_manifest)},
            },
            "files": canonical_rows,
        }
        self.canonical_manifest.write_text(json.dumps(canonical_payload), encoding="utf-8")
        self.support_manifest = root / "support.json"
        self.support_manifest.write_text(
            json.dumps(
                {
                    "schema": "uav-tgs-undistorted-temperature-v1",
                    "status": "complete",
                    "files": support_rows,
                }
            ),
            encoding="utf-8",
        )
        test_support = np.ones((4, 5), dtype=np.bool_)
        test_support_path = self.evaluation_support_root / "bool" / "test.npy"
        np.save(test_support_path, test_support, allow_pickle=False)
        self.evaluation_support_manifest = root / "evaluation_support.json"
        self.evaluation_support_manifest.write_text(
            json.dumps(
                {
                    "schema_name": "uav-tgs-formal-temperature-support",
                    "schema_version": 1,
                    "split": "test",
                    "expected_test_count": 1,
                    "source_manifests": {
                        "split": {"sha256": sha256_file(self.bound)},
                        "valid_support": {"sha256": sha256_file(self.support_manifest)},
                    },
                    "policy": {
                        "expression": "valid_support AND (opacity_proxy > opacity_threshold)",
                        "opacity_threshold": 0.01,
                        "comparison": "strict_greater_than",
                        "opacity_proxy_semantics": "black_bg_plus_white_override_color_render",
                        "threshold_applied_only_by_this_combiner": True,
                    },
                    "records": [
                        {
                            "name": "test",
                            "shape": [4, 5],
                            "outputs": {
                                "bool": {
                                    "dtype": "bool",
                                    "relative_path": "bool/test.npy",
                                    "sha256": sha256_file(test_support_path),
                                }
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.camera_parameters_sha = "4" * 64
        self.sequence = root / "sequence.json"
        metadata = {
            "scene": scene,
            "bound_split_sha256": sha256_file(self.bound),
            "decode_manifest_sha256": sha256_file(self.decode_manifest),
            "decode_protocol_sha256": sha256_file(self.decode_protocol),
            "range_manifest_sha256": sha256_file(self.range_manifest),
            "canonical_manifest_sha256": sha256_file(self.canonical_manifest),
            "support_manifest_sha256": sha256_file(self.support_manifest),
            "evaluation_support_manifest_sha256": sha256_file(
                self.evaluation_support_manifest
            ),
            "anchor_artifact_sha256": sha256_file(self.anchor_artifact),
            "anchor_occupancy_sha256": self.snapshot["overall_sha256"],
            "camera_parameters_sha256": self.camera_parameters_sha,
            "experiment_recipe_sha256": sha256_json(FORMAL_EXPERIMENT_RECIPE),
        }
        save_sequence_manifest(
            self.sequence,
            build_sequence_manifest(["train.png"], steps=30_000, seed=0, metadata=metadata),
        )
        self.proxy = BandRadianceProxy(0.0, 30.0)
        self.configs = {
            variant: OCTConfig(count, 0.0, 30.0, variant=variant)
            for variant in ("oct_scalar", "oct_residual")
        }

    def binding(self):
        return build_formal_binding(
            scene_name=self.scene,
            bound_split_path=self.bound,
            decode_manifest_path=self.decode_manifest,
            decode_protocol_path=self.decode_protocol,
            range_manifest_path=self.range_manifest,
            canonical_manifest_path=self.canonical_manifest,
            temperature_root=self.target_root,
            canonical_root=self.color_root,
            support_manifest_path=self.support_manifest,
            support_root=self.support_root,
            evaluation_support_manifest_path=self.evaluation_support_manifest,
            evaluation_support_root=self.evaluation_support_root,
            camera_sequence_path=self.sequence,
            camera_parameters_sha256=self.camera_parameters_sha,
            anchor_artifact_path=self.anchor_artifact,
            anchor_snapshot=self.snapshot,
            field_configs=self.configs,
            radiance_proxy=self.proxy,
        )

    def calibration(self, path: Path):
        binding = self.binding()
        calibrator = BuildingGradientCalibrator(binding.calibration_receipt())
        for variant in ("oct_scalar", "oct_residual"):
            calibrator.add_gradient_norms(
                "train.png",
                variant,
                {"thermometric": 2.0, "color_l1": 4.0, "color_dssim": 8.0},
            )
        return calibrator.freeze(
            path,
            metadata={
                "experiment_recipe": dict(FORMAL_EXPERIMENT_RECIPE),
                "experiment_recipe_sha256": sha256_json(FORMAL_EXPERIMENT_RECIPE),
            },
        )


class OCTRadianceTests(unittest.TestCase):
    def test_float32_lookup_is_monotonic_roundtrip_differentiable_and_proxy_only(self):
        proxy = BandRadianceProxy(-10.0, 80.0)
        temperature = torch.linspace(-10.0, 80.0, 1001, dtype=torch.float32, requires_grad=True)
        radiance = proxy(temperature)
        self.assertTrue(bool((radiance[1:] > radiance[:-1]).all()))
        self.assertLess(float((proxy.inverse(radiance) - temperature).abs().max()), 1e-3)
        radiance.mean().backward()
        self.assertTrue(bool(torch.isfinite(temperature.grad).all()))
        metadata = proxy.metadata()
        self.assertFalse(metadata["runtime_planck_integration"])
        self.assertFalse(metadata["tsdk_environmental_correction_in_renderer"])
        self.assertEqual(metadata["method_semantics"], METHOD_SEMANTICS)
        self.assertEqual(metadata["target_semantics"], TARGET_SEMANTICS)
        with self.assertRaises(TypeError):
            proxy(torch.ones(2, dtype=torch.float64))

    def test_forward_hot_iron_exact_and_gradient(self):
        temperature = torch.tensor([0.0, 50.0, 100.0], dtype=torch.float32, requires_grad=True)
        smooth = temperature_to_hot_iron(temperature, 0.0, 100.0)
        smooth.sum().backward()
        self.assertTrue(bool(torch.isfinite(temperature.grad).all()))
        exact = temperature_to_hot_iron(torch.tensor([0.0, 100.0]), 0.0, 100.0, exact_display=True)
        lut = torch.from_numpy(hot_iron_lut().astype(np.float32) / 255.0)
        self.assertTrue(torch.equal(exact[0], lut[0]))
        self.assertTrue(torch.equal(exact[1], lut[-1]))


class OCTFieldTests(unittest.TestCase):
    def test_only_two_variants_and_v1_uncertainty_is_disabled(self):
        with self.assertRaises(ValueError):
            OCTConfig(2, 0.0, 10.0, variant="free_rgb_sh").validate()
        with self.assertRaises(ValueError):
            OCTConfig(2, 0.0, 10.0, learn_uncertainty=True).validate()
        field = OCTGaussianField(OCTConfig(3, 0.0, 10.0, variant="oct_scalar"))
        self.assertIsNone(field.raw_residual_amplitude)
        self.assertIsNone(field()["log_uncertainty_c"])

    def test_residual_is_odd_bounded_and_in_range(self):
        config = OCTConfig(2, 0.0, 20.0, variant="oct_residual", residual_bound_fraction=0.1)
        field = OCTGaussianField(config)
        with torch.no_grad():
            field.raw_residual_amplitude.fill_(20.0)
            field.raw_base_temperature[1].fill_(3.0)
        weak = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        positive, negative = field(weak, weak), field(-weak, weak)
        self.assertTrue(torch.equal(positive["view_residual_c"], -negative["view_residual_c"]))
        self.assertLessEqual(float(positive["view_residual_c"].abs().max()), config.residual_bound_c + 1e-6)
        for output in (positive, negative):
            self.assertTrue(bool((output["temperature_c"] >= 0.0).all()))
            self.assertTrue(bool((output["temperature_c"] <= 20.0).all()))

    def test_optimizer_is_field_only_and_rejects_uncertainty_lr(self):
        anchor = DummyAnchor(3)
        field = OCTGaussianField(OCTConfig(3, 0.0, 20.0, variant="oct_residual"))
        optimizer = build_oct_optimizer(field, temperature_lr=1e-2, residual_lr=2e-3)
        verify_field_only_optimizer(field, optimizer, (anchor._xyz, anchor._scaling, anchor._rotation, anchor._opacity))
        with self.assertRaises(ValueError):
            build_oct_optimizer(field, temperature_lr=1e-2, residual_lr=2e-3, uncertainty_lr=1e-3)


class OCTRenderingAndLossTests(unittest.TestCase):
    def test_native_background_composition_is_exact_one_pass_and_field_only(self):
        anchor = DummyAnchor(2)
        field = OCTGaussianField(OCTConfig(2, 0.0, 40.0))
        with torch.no_grad():
            field.raw_base_temperature.copy_(torch.tensor([[-1.0], [1.0]]))
        proxy = BandRadianceProxy(0.0, 40.0)
        weights = torch.tensor([0.25, 0.5], dtype=torch.float32)
        context = OCTRendererContext(
            anchor,
            proxy,
            renderer=weighted_fake_renderer(weights),
        )
        background_c = 7.0
        output = context.render(
            DummyCamera(), field, object(), background_temperature_c=background_c
        )
        self.assertEqual(output["raster_passes"], 1)
        self.assertEqual(output["alpha_backend"], "native_background_composition")
        self.assertIsNone(output["alpha"])
        gaussian_t = output["gaussian_temperature_c"]
        gaussian_u = proxy.normalize(proxy(gaussian_t))[:, 0]
        background_u = proxy.normalize(proxy(torch.tensor(background_c)))
        expected_u = (weights * gaussian_u).sum() + (1.0 - weights.sum()) * background_u
        expected_radiance = proxy.denormalize(expected_u)
        self.assertTrue(
            torch.allclose(
                output["radiance"],
                expected_radiance.expand_as(output["radiance"]),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        output["temperature_c"].mean().backward()
        self.assertGreater(float(field.raw_base_temperature.grad.abs().sum()), 0.0)
        self.assertIsNone(anchor._opacity.grad)

    def test_masked_ssim_ignores_invalid_boundary_and_uint8_target_is_explicit(self):
        proxy = BandRadianceProxy(0.0, 40.0)
        prediction_t = torch.full((1, 16, 16), 20.0, requires_grad=True)
        target_t = prediction_t.detach().clone()
        prediction_rgb = temperature_to_hot_iron(prediction_t[0], 0.0, 40.0).movedim(-1, 0)
        target_rgb = prediction_rgb.detach().mul(255).round().to(torch.uint8)
        target_rgb[:, :, 8:] = 0
        mask = torch.zeros((1, 16, 16), dtype=torch.bool)
        mask[:, :, :8] = True
        result = oct_rendering_loss(
            prediction_t,
            target_t,
            prediction_rgb,
            radiance_proxy=proxy,
            target_hot_iron=target_rgb,
            mask=mask,
            weights=OCTLossWeights(1.0, 1.0, 1.0),
        )
        self.assertGreater(float(result["color_dssim"]), -1e-5)
        self.assertLess(float(result["color_dssim"]), 1e-3)
        result["total"].backward()
        with self.assertRaises(ValueError):
            oct_rendering_loss(
                prediction_t.detach(), target_t, prediction_rgb.detach(), radiance_proxy=proxy,
                prediction_log_uncertainty_c=torch.zeros_like(prediction_t)
            )

    def test_weak_axis_is_cached_from_anchor(self):
        anchor = DummyAnchor(2)
        with torch.no_grad():
            anchor._scaling.copy_(torch.log(torch.tensor([[0.1, 1.0, 2.0], [2.0, 0.1, 1.0]])))
        weak = weak_axis_from_anchor(FrozenGaussianView(anchor))
        self.assertTrue(torch.allclose(weak[0].abs(), torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(weak[1].abs(), torch.tensor([0.0, 1.0, 0.0])))


class OCTFormalProtocolTests(unittest.TestCase):
    def test_formal_metadata_cameras_do_not_decode_source_images(self):
        train_b, intrinsic = fake_colmap_camera("b.png")
        train_a, _ = fake_colmap_camera("a.png")
        test_c, _ = fake_colmap_camera("c.png")
        guard_d, _ = fake_colmap_camera("guard.png")
        bound = {
            "records": [
                {"split": "train", "thermal_camera_name": "b.png"},
                {"split": "guard", "thermal_camera_name": "guard.png"},
                {"split": "test", "thermal_camera_name": "c.png"},
                {"split": "train", "thermal_camera_name": "a.png"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            image_root = source / "images"
            image_root.mkdir()
            for name in ("a.png", "b.png", "c.png"):
                Image.new("RGB", (1280, 1024), (0, 0, 0)).save(image_root / name)
            train, test = _formal_metadata_cameras(
                source,
                bound,
                images="images",
                device="cpu",
                colmap_model=(
                    {1: train_b, 2: train_a, 3: test_c, 4: guard_d},
                    {int(intrinsic.id): intrinsic},
                ),
            )
            Image.new("RGB", (1279, 1024), (0, 0, 0)).save(image_root / "c.png")
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                _formal_metadata_cameras(
                    source,
                    bound,
                    images="images",
                    device="cpu",
                    colmap_model=(
                        {1: train_b, 2: train_a, 3: test_c, 4: guard_d},
                        {int(intrinsic.id): intrinsic},
                    ),
                )
        self.assertEqual([camera.image_name for camera in train], ["a.png", "b.png"])
        self.assertEqual([camera.uid for camera in train], [0, 1])
        self.assertEqual([camera.image_name for camera in test], ["c.png"])
        self.assertEqual(test[0].uid, 0)
        for camera in train + test:
            self.assertTrue(camera.metadata_only)
            self.assertFalse(hasattr(camera, "original_image"))
            self.assertFalse(hasattr(camera, "alpha_mask"))
            self.assertEqual((camera.image_height, camera.image_width), (1024, 1280))
            self.assertEqual(camera.world_view_transform.device.type, "cpu")
            self.assertTrue(torch.isfinite(camera.full_proj_transform).all())
            expected_center = torch.inverse(camera.world_view_transform)[3, :3]
            self.assertTrue(torch.equal(camera.camera_center, expected_center))

        # Current InternalRoad camera 0139 exposes a one-ULP difference if the
        # legacy qvec expression is algebraically simplified (q**2 -> q*q).
        regression_qvec = np.asarray(
            [
                -0.0025306362255026475,
                -0.9996852564913212,
                -0.02495593521926798,
                -0.0004302691102580914,
            ],
            dtype=np.float64,
        )
        self.assertEqual(
            hashlib.sha256(_qvec_to_rotation(regression_qvec).tobytes()).hexdigest(),
            "832b2e0836f21c01acd3acbba8c3e62adc38dfe8ab02eaca553a2ac91e400605",
        )

        simple_extrinsic, simple_intrinsic = fake_colmap_camera("simple.png")
        simple_intrinsic.model = "SIMPLE_PINHOLE"
        simple_intrinsic.params = np.asarray([900.0, 640.0, 512.0])
        simple = _metadata_camera(
            simple_extrinsic, simple_intrinsic, uid=9, device="cpu"
        )
        self.assertEqual(simple.uid, 9)
        self.assertAlmostEqual(simple.FoVx, focal2fov(900.0, 1280))
        self.assertAlmostEqual(simple.FoVy, focal2fov(900.0, 1024))

        invalid_extrinsic, invalid_intrinsic = fake_colmap_camera("bad.png")
        invalid_intrinsic.model = "SIMPLE_RADIAL"
        with self.assertRaisesRegex(ValueError, "PINHOLE/SIMPLE_PINHOLE"):
            _metadata_camera(invalid_extrinsic, invalid_intrinsic, uid=0, device="cpu")

    def test_formal_outputs_are_isolated_and_gt_is_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {
                name: root / name
                for name in (
                    "source",
                    "model",
                    "temperature",
                    "canonical",
                    "support",
                    "evaluation_support",
                )
            }
            for path in inputs.values():
                path.mkdir()
            args = SimpleNamespace(
                source_path=inputs["source"],
                model_path=inputs["model"],
                temperature_root=inputs["temperature"],
                canonical_root=inputs["canonical"],
                support_root=inputs["support"],
                evaluation_support_root=inputs["evaluation_support"],
            )
            isolated = root / "experiments" / "oct"
            self.assertEqual(
                _require_isolated_output(isolated, args, label="test output"),
                isolated.resolve(),
            )
            with self.assertRaises(ValueError):
                _require_isolated_output(
                    inputs["canonical"] / "bad", args, label="test output"
                )
            with self.assertRaises(ValueError):
                _require_isolated_output(root, args, label="test output")

            source = inputs["canonical"] / "reference.png"
            destination = isolated / "gt" / "reference.png"
            source.write_bytes(b"immutable-reference")
            _copy_immutable_reference(source, destination)
            destination.write_bytes(b"changed-evaluator-copy")
            self.assertEqual(source.read_bytes(), b"immutable-reference")
            with self.assertRaises(FileExistsError):
                _copy_immutable_reference(source, destination)

    def test_building_calibrator_observe_runs_real_autograd_components(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = FormalFixture(Path(temp))
            calibrator = BuildingGradientCalibrator(
                fixture.binding().calibration_receipt(),
                thermometric_domain="celsius",
            )
            parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            components = {
                "thermometric": parameter.square().mean(),
                "color_l1": parameter.abs().mean(),
                "color_dssim": (3.0 * parameter).square().mean(),
            }
            norms = calibrator.observe(
                "train.png", "oct_scalar", components, (parameter,)
            )
            self.assertEqual(set(norms), set(components))
            self.assertTrue(all(np.isfinite(value) and value > 0.0 for value in norms.values()))

    def test_binding_pins_tsdk_split_range_lut_support_anchor_and_sequence(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = FormalFixture(Path(temp))
            binding = fixture.binding()
            self.assertEqual(binding.payload["tsdk_target"]["target_semantics"], TARGET_SEMANTICS)
            self.assertFalse(binding.payload["tsdk_target"]["environmental_correction_reapplied_by_oct"])
            self.assertTrue(binding.payload["tsdk_target"]["fixed_decode_parameters_are_metadata_only"])
            self.assertEqual(binding.payload["bound_split"]["counts"], {"total": 3, "train": 1, "test": 1, "guard": 1})
            self.assertEqual(binding.names("train"), ["train.png"])
            self.assertEqual(binding.payload["camera_sequence"]["steps"], 30_000)
            self.assertEqual(binding.payload["camera_sequence"]["seed"], 0)
            temperature, color, support = FormalOCTTargetStore(binding).get(
                "train.png", 4, 5, "cpu"
            )
            self.assertEqual(temperature.dtype, torch.float32)
            self.assertTrue(torch.equal(temperature, torch.full((1, 4, 5), 10.0)))
            self.assertEqual(color.dtype, torch.float32)
            self.assertEqual(support.dtype, torch.bool)
            with self.assertRaises(ValueError):
                FormalOCTTargetStore(binding).get("train.png", 2, 3, "cpu")

    def test_binding_rejects_self_consistent_nonzero_sequence_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = FormalFixture(Path(temp))
            original = json.loads(fixture.sequence.read_text(encoding="utf-8"))
            tampered = build_sequence_manifest(
                ["train.png"],
                steps=30_000,
                seed=1,
                metadata=original["metadata"],
            )
            save_sequence_manifest(fixture.sequence, tampered)
            with self.assertRaisesRegex(ValueError, "seed=0"):
                fixture.binding()

    def test_binding_rejects_target_or_protocol_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = FormalFixture(Path(temp))
            fixture.binding()
            with fixture.target_root.joinpath("train.npy").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(ValueError):
                fixture.binding()

    def test_binding_requires_bound_undistortion_and_formal_test_support(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = FormalFixture(Path(temp))
            binding = fixture.binding()
            self.assertEqual(
                binding.payload["tsdk_target"]["target_geometric_transform"],
                "float32 temperature-domain undistortion with bound valid_support",
            )
            self.assertEqual(
                binding.payload["support"]["evaluation"]["policy"]["opacity_threshold"],
                0.01,
            )
            _, _, support = FormalOCTTargetStore(binding).get(
                "test.png", 4, 5, "cpu", evaluation_support=True
            )
            self.assertTrue(bool(support.all()))
            payload = json.loads(
                fixture.evaluation_support_manifest.read_text(encoding="utf-8")
            )
            payload["policy"]["opacity_threshold"] = 0.02
            fixture.evaluation_support_manifest.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                fixture.binding()

    def test_building_calibration_is_receipt_bound_and_internalroad_requires_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = FormalFixture(root)
            path = root / "calibration.json"
            payload = fixture.calibration(path)
            self.assertEqual(payload["schema"], CALIBRATION_SCHEMA)
            loaded, weights = load_frozen_calibration(
                path,
                building_receipt=fixture.binding().calibration_receipt(),
                expected_calibration_sha256=payload["calibration_sha256"],
                consumer_scene="Building",
            )
            self.assertEqual(loaded["calibration_sha256"], payload["calibration_sha256"])
            self.assertAlmostEqual(weights.color_l1, 0.5)
            self.assertAlmostEqual(weights.color_dssim, 0.25)
            with self.assertRaises(ValueError):
                load_frozen_calibration(path, consumer_scene="InternalRoad")

    def test_checkpoint_is_bound_to_formal_protocol_and_exact_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = FormalFixture(root)
            binding = fixture.binding()
            calibration_path = root / "calibration.json"
            calibration = fixture.calibration(calibration_path)
            field = OCTGaussianField(fixture.configs["oct_scalar"])
            optimizer = build_oct_optimizer(field, temperature_lr=1e-2)
            protocol_path = root / "oct_protocol.json"
            write_oct_protocol_manifest(
                protocol_path,
                field=field,
                anchor_snapshot=fixture.snapshot,
                radiance_proxy=fixture.proxy,
                calibration_manifest=calibration_path,
                formal_binding=binding,
                expected_calibration_sha256=calibration["calibration_sha256"],
                thermometric_domain="celsius",
                optimizer_config={
                    "type": "Adam",
                    "field_only": True,
                    "temperature_lr": 1e-2,
                    "residual_lr": None,
                    "adam_eps": 1e-15,
                    "schedule": "constant",
                    "steps": 30_000,
                },
                source_provenance=fake_training_source_provenance(),
            )
            checkpoint_path = root / "oct.pt"
            run_toy_adam_steps(field, optimizer, 0, 10_000)
            save_oct_checkpoint(
                checkpoint_path,
                field=field,
                anchor=fixture.anchor,
                anchor_snapshot=fixture.snapshot,
                optimizer=optimizer,
                step=10_000,
                protocol_manifest=protocol_path,
                cost_summary={"status": "endpoint"},
            )
            inspected = inspect_oct_checkpoint(checkpoint_path)
            self.assertFalse(inspected["contains_anchor_tensors"])
            self.assertEqual(inspected["sequence_offset"], 10_000)
            with self.assertRaises(FileExistsError):
                save_oct_checkpoint(
                    checkpoint_path,
                    field=field,
                    anchor=fixture.anchor,
                    anchor_snapshot=fixture.snapshot,
                    optimizer=optimizer,
                    step=10_000,
                    protocol_manifest=protocol_path,
                    cost_summary={"status": "duplicate"},
                )
            with self.assertRaises(ValueError):
                save_oct_checkpoint(
                    root / "non_endpoint.pt",
                    field=field,
                    anchor=fixture.anchor,
                    anchor_snapshot=fixture.snapshot,
                    optimizer=optimizer,
                    step=9_999,
                    protocol_manifest=protocol_path,
                    cost_summary={"status": "invalid"},
                )
            restored, metadata = load_oct_checkpoint(
                checkpoint_path,
                anchor=fixture.anchor,
                protocol_manifest=protocol_path,
                formal_binding=binding,
            )
            self.assertTrue(torch.equal(restored.raw_base_temperature, field.raw_base_temperature))
            self.assertEqual(metadata["step"], 10_000)
            self.assertEqual(metadata["sequence_offset"], 10_000)
            try:
                invalid_payload = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
            except TypeError:
                invalid_payload = torch.load(checkpoint_path, map_location="cpu")
            invalid_payload["step"] = 9_999
            invalid_payload["sequence_offset"] = 9_999
            invalid_checkpoint = root / "invalid_load.pt"
            torch.save(invalid_payload, invalid_checkpoint)
            with self.assertRaises(ValueError):
                load_oct_checkpoint(
                    invalid_checkpoint,
                    anchor=fixture.anchor,
                    protocol_manifest=protocol_path,
                    formal_binding=binding,
                )
            other_root = root / "other"
            other_root.mkdir()
            other_binding = FormalFixture(other_root, scene="InternalRoad").binding()
            with self.assertRaises(ValueError):
                load_oct_checkpoint(
                    checkpoint_path,
                    anchor=fixture.anchor,
                    protocol_manifest=protocol_path,
                    formal_binding=other_binding,
                )
            with torch.no_grad():
                fixture.anchor._opacity[0, 0].add_(1e-5)
            with self.assertRaises(RuntimeError):
                load_oct_checkpoint(
                    checkpoint_path,
                    anchor=fixture.anchor,
                    protocol_manifest=protocol_path,
                    formal_binding=binding,
                )

    def test_uninterrupted_and_resumed_adam_are_bit_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = FormalFixture(root)
            binding = fixture.binding()
            calibration_path = root / "calibration.json"
            calibration = fixture.calibration(calibration_path)
            field = OCTGaussianField(fixture.configs["oct_residual"])
            optimizer = build_oct_optimizer(
                field, temperature_lr=1e-2, residual_lr=2.5e-3
            )
            protocol_path = root / "oct_protocol.json"
            write_oct_protocol_manifest(
                protocol_path,
                field=field,
                anchor_snapshot=fixture.snapshot,
                radiance_proxy=fixture.proxy,
                calibration_manifest=calibration_path,
                formal_binding=binding,
                expected_calibration_sha256=calibration["calibration_sha256"],
                thermometric_domain="celsius",
                optimizer_config={
                    "type": "Adam",
                    "field_only": True,
                    "temperature_lr": 1e-2,
                    "residual_lr": 2.5e-3,
                    "adam_eps": 1e-15,
                    "schedule": "constant",
                    "steps": 30_000,
                },
                source_provenance=fake_training_source_provenance(),
            )
            run_toy_adam_steps(field, optimizer, 0, 10_000)
            checkpoint_path = root / "step_10000.pt"
            save_oct_checkpoint(
                checkpoint_path,
                field=field,
                anchor=fixture.anchor,
                anchor_snapshot=fixture.snapshot,
                optimizer=optimizer,
                step=10_000,
                protocol_manifest=protocol_path,
                cost_summary={"status": "endpoint_10000"},
            )
            resumed_field, checkpoint = load_oct_checkpoint(
                checkpoint_path,
                anchor=fixture.anchor,
                protocol_manifest=protocol_path,
                formal_binding=binding,
            )
            resumed_optimizer = build_oct_optimizer(
                resumed_field, temperature_lr=1e-2, residual_lr=2.5e-3
            )
            restored_step = restore_oct_optimizer_state(
                field=resumed_field,
                optimizer=resumed_optimizer,
                checkpoint_metadata=checkpoint,
                anchor=fixture.anchor,
                protocol_manifest=protocol_path,
            )
            self.assertEqual(restored_step, 10_000)
            sequence = [f"camera_{index}" for index in range(30_000)]
            self.assertEqual(
                _remaining_sequence({"sequence": sequence}, restored_step)[0],
                sequence[10_000],
            )
            with self.assertRaises(ValueError):
                _remaining_sequence({"sequence": sequence}, 30_000)

            run_toy_adam_steps(field, optimizer, 10_000, 10_007)
            run_toy_adam_steps(
                resumed_field, resumed_optimizer, restored_step, 10_007
            )
            for (name, uninterrupted), (resumed_name, resumed) in zip(
                field.named_parameters(), resumed_field.named_parameters()
            ):
                self.assertEqual(name, resumed_name)
                self.assertTrue(torch.equal(uninterrupted, resumed), name)
                uninterrupted_state = optimizer.state[uninterrupted]
                resumed_state = resumed_optimizer.state[resumed]
                self.assertEqual(set(uninterrupted_state), set(resumed_state))
                for state_name in uninterrupted_state:
                    left = uninterrupted_state[state_name]
                    right = resumed_state[state_name]
                    if isinstance(left, torch.Tensor):
                        self.assertTrue(
                            torch.equal(left, right), f"{name}:{state_name}"
                        )
                    else:
                        self.assertEqual(left, right)

    def test_post_step_finite_source_and_anchor_iteration_fail_closed(self):
        provenance = fake_training_source_provenance()
        self.assertEqual(
            validate_training_source_provenance(provenance)["git_commit"], "2" * 40
        )
        dirty = dict(provenance)
        dirty["git_clean"] = False
        with self.assertRaisesRegex(ValueError, "clean Git"):
            validate_training_source_provenance(dirty)
        bad_hash = dict(provenance)
        bad_hash["files_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "inventory hash"):
            validate_training_source_provenance(bad_hash)

        runner_path = Path(__file__).resolve().parents[1] / "tools" / "oct_gs_formal.py"
        with patch(
            "tools.oct_gs_formal._git_bytes",
            side_effect=[b"3" * 40 + b"\n", b"", b"tools/oct_gs_formal.py\0"],
        ), patch(
            "tools.oct_gs_formal._formal_source_paths", return_value=[runner_path]
        ):
            actual = _formal_source_provenance()
        self.assertTrue(actual["git_clean"])
        self.assertEqual(actual["git_commit"], "3" * 40)
        self.assertEqual(actual["files"][0]["sha256"], sha256_file(runner_path))
        with patch(
            "tools.oct_gs_formal._git_bytes",
            side_effect=[b"3" * 40 + b"\n", b" M tools/oct_gs_formal.py\n"],
        ):
            with self.assertRaisesRegex(RuntimeError, "dirty Git worktree"):
                _formal_source_provenance()

        current = fake_training_source_provenance()
        self.assertEqual(
            _require_matching_source_provenance(
                current, current, label="unit-test artifact"
            ),
            current,
        )
        drifted = dict(current)
        drifted["git_commit"] = "4" * 40
        with self.assertRaisesRegex(RuntimeError, "differs from the current"):
            _require_matching_source_provenance(
                drifted, current, label="unit-test artifact"
            )
        with self.assertRaisesRegex(RuntimeError, "provenance is invalid"):
            _require_matching_source_provenance(
                {}, current, label="unit-test artifact"
            )

        field = OCTGaussianField(OCTConfig(2, 0.0, 30.0, variant="oct_scalar"))
        optimizer = build_oct_optimizer(field, temperature_lr=1e-2)
        run_toy_adam_steps(field, optimizer, 0, 1)
        with torch.no_grad():
            optimizer.state[field.raw_base_temperature]["exp_avg"][0, 0] = float("inf")
        with self.assertRaises(FloatingPointError):
            verify_oct_post_step_finite(field, optimizer, 1)
        with torch.no_grad():
            optimizer.state[field.raw_base_temperature]["exp_avg"][0, 0] = 0.0
            field.raw_base_temperature[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            verify_oct_field_finite(field, label="unit test")

        for invalid_iteration in (0, -1):
            with self.assertRaisesRegex(ValueError, "anchor_iteration > 0"):
                Runtime(
                    SimpleNamespace(
                        anchor_iteration=invalid_iteration,
                        resolution=-1,
                    )
                )

    def test_occupancy_snapshot_and_cost_tracker(self):
        anchor = DummyAnchor(3)
        snapshot = capture_occupancy_snapshot(anchor)
        verify_occupancy_snapshot(anchor, snapshot)
        evidence = _occupancy_invariant_evidence(snapshot, snapshot)
        self.assertEqual(evidence["status"], "passed")
        self.assertTrue(evidence["exact"])
        self.assertEqual(evidence["topology_count"], 3)
        self.assertEqual(
            evidence["expected_overall_sha256"], snapshot["overall_sha256"]
        )
        self.assertEqual(
            evidence["observed_overall_sha256"], snapshot["overall_sha256"]
        )
        self.assertEqual(set(evidence["field_hashes"]), set(snapshot["ordered_fields"]))
        tracker = OCTStageCostTracker({"variant": "oct_scalar"})
        tracker.start()
        time.sleep(0.001)
        tracker.record_step(raster_passes=1)
        result = tracker.finish()
        self.assertEqual(result["raster_passes_per_view"], 1.0)
        with torch.no_grad():
            anchor._xyz[0, 0].add_(1e-6)
        with self.assertRaises(RuntimeError):
            verify_occupancy_snapshot(anchor, snapshot)

    def test_visible_welford_is_scalar_exact_and_residual_stable(self):
        scalar = torch.tensor([37.123451, 19.876543, 72.34567], dtype=torch.float32)
        mean = torch.zeros_like(scalar)
        m2 = torch.zeros_like(scalar)
        count = torch.zeros(3, dtype=torch.int32)
        visible = torch.tensor([True, True, False])
        for _ in range(257):
            _update_visible_temperature_moments(
                mean, m2, count, scalar, visible
            )
        variance, valid = _population_variance_from_moments(m2, count)
        self.assertTrue(torch.equal(variance[valid], torch.zeros_like(variance[valid])))
        self.assertEqual(count.tolist(), [257, 257, 0])

        mean.zero_()
        m2.zero_()
        count.zero_()
        samples = (
            torch.tensor([10.0, 20.0, 30.0]),
            torch.tensor([12.0, 20.5, 30.0]),
            torch.tensor([14.0, 21.0, 30.0]),
        )
        masks = (
            torch.tensor([True, True, False]),
            torch.tensor([True, True, False]),
            torch.tensor([True, True, True]),
        )
        for sample, mask in zip(samples, masks):
            _update_visible_temperature_moments(mean, m2, count, sample, mask)
        variance, valid = _population_variance_from_moments(m2, count)
        self.assertEqual(valid.tolist(), [True, True, False])
        self.assertAlmostEqual(float(variance[0]), 8.0 / 3.0, places=6)
        self.assertAlmostEqual(float(variance[1]), 1.0 / 6.0, places=6)

    def test_exact_display_temperature_matches_palette_roundtrip(self):
        source = torch.tensor(
            [[-1.0, 0.0, 12.345, 49.999, 50.001, 100.0, 101.0]],
            dtype=torch.float32,
        )
        recovered = _exact_display_temperature_c(source, 0.0, 100.0)
        indices, _ = temperature_to_indices(source.numpy(), 0.0, 100.0)
        expected = indices_to_temperature(indices, 0.0, 100.0)
        self.assertTrue(
            np.allclose(recovered.numpy(), expected, rtol=0.0, atol=1e-6)
        )
        direct = source.clamp(0.0, 100.0)
        self.assertFalse(torch.equal(recovered, direct))

    def test_hotspot_policy_is_pinned_and_histogram_auprc_is_deterministic(self):
        binding = SimpleNamespace(tmin_c=0.0, tmax_c=100.0)
        payload = {
            "source_split": "train",
            "test_statistics_used": False,
            "quantile": FORMAL_HOTSPOT_QUANTILE,
            "histogram_bins": FORMAL_HOTSPOT_BINS,
            "threshold_c": 50.0,
            "range_c": [0.0, 100.0],
            "valid_train_pixels": 1000,
        }
        _validate_formal_hotspot_threshold(payload, binding)
        with self.assertRaises(ValueError):
            _validate_formal_hotspot_threshold(
                {**payload, "quantile": 0.90}, binding
            )
        with self.assertRaises(ValueError):
            _validate_formal_hotspot_threshold(
                {**payload, "histogram_bins": 4096}, binding
            )
        for invalid in (
            {**payload, "threshold_c": float("nan")},
            {**payload, "threshold_c": 101.0},
            {**payload, "range_c": [0.0, 99.0]},
            {**payload, "valid_train_pixels": 0},
        ):
            with self.assertRaises(ValueError):
                _validate_formal_hotspot_threshold(invalid, binding)
        positive = np.asarray([0, 1, 1], dtype=np.int64)
        negative = np.asarray([1, 1, 0], dtype=np.int64)
        self.assertAlmostEqual(_histogram_auprc(positive, negative), 5.0 / 6.0)
        self.assertIsNone(_histogram_auprc(np.zeros(3, dtype=np.int64), negative))

    def test_eval_v2_training_source_compatibility_is_exact_not_a_waiver(self):
        provenance = fake_training_source_provenance()
        provenance["git_commit"] = FROZEN_TRAINING_COMMIT
        protocol = {"source_provenance": provenance}
        changed = ["tools/evaluate_oct_gs_formal_v2.py", "tests/test_oct_gs.py"]
        with patch(
            "tools.evaluate_oct_gs_formal_v2._source_records",
            return_value=provenance["files"],
        ), patch(
            "tools.evaluate_oct_gs_formal_v2._changed_paths_since_frozen_training",
            return_value=changed,
        ):
            result = _training_source_compatibility(protocol)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["generic_commit_mismatch_waiver"])
        self.assertTrue(result["training_source_files_byte_exact"])
        self.assertEqual(result["post_training_changed_paths"], changed)

        self.assertIn(
            "tools/geometric_repeatability/build_temperature_responsibility_bundle.py",
            ALLOWED_POST_TRAINING_PATHS,
        )
        self.assertIn(
            "tests/test_temperature_responsibility_bundle.py",
            ALLOWED_POST_TRAINING_PATHS,
        )
        self.assertIn(
            "tools/geometric_repeatability/evaluate_depth_definitions.py",
            ALLOWED_POST_TRAINING_PATHS,
        )
        self.assertIn(
            "tests/test_depth_definitions.py",
            ALLOWED_POST_TRAINING_PATHS,
        )
        self.assertFalse(
            any(
                path.startswith("oct_gs/") or path == "tools/oct_gs_formal.py"
                for path in ALLOWED_POST_TRAINING_PATHS
            )
        )

        wrong_commit = dict(provenance)
        wrong_commit["git_commit"] = "9" * 40
        with self.assertRaises(RuntimeError):
            _training_source_compatibility({"source_provenance": wrong_commit})
        with patch(
            "tools.evaluate_oct_gs_formal_v2._source_records",
            return_value=[{**provenance["files"][0], "bytes": 999}],
        ):
            with self.assertRaises(RuntimeError):
                _training_source_compatibility(protocol)

    def test_eval_v2_checkpoint_gate_pins_final_exact_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint_path = Path(temp) / "step_30000.pt"
            checkpoint_path.write_bytes(b"formal-checkpoint-fixture")
            snapshot = capture_occupancy_snapshot(DummyAnchor(3))
            field_config = {"variant": "oct_scalar", "num_gaussians": 3}
            formal_sha = "a" * 64
            protocol = {
                "scene_name": "Building",
                "variant": "oct_scalar",
                "manifest_file_sha256": "b" * 64,
                "manifest_sha256": "c" * 64,
                "field": {"config": field_config},
            }
            checkpoint = {
                "step": 30_000,
                "sequence_offset": 30_000,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "field_config": field_config,
                "anchor_snapshot": snapshot,
                "protocol_receipt": {
                    "manifest_file_sha256": protocol["manifest_file_sha256"],
                    "manifest_sha256": protocol["manifest_sha256"],
                    "formal_protocol_sha256": formal_sha,
                },
            }
            result = _checkpoint_compatibility(
                checkpoint=checkpoint,
                protocol=protocol,
                binding=SimpleNamespace(
                    scene_name="Building", formal_protocol_sha256=formal_sha
                ),
                runtime=SimpleNamespace(anchor_snapshot=snapshot),
                variant="oct_scalar",
            )
            self.assertTrue(all(result.values()))
            self.assertEqual(EVALUATION_SCHEMA, "uav-tgs-oct-formal-evaluation-v2")
            checkpoint["step"] = 20_000
            with self.assertRaises(ValueError):
                _checkpoint_compatibility(
                    checkpoint=checkpoint,
                    protocol=protocol,
                    binding=SimpleNamespace(
                        scene_name="Building", formal_protocol_sha256=formal_sha
                    ),
                    runtime=SimpleNamespace(anchor_snapshot=snapshot),
                    variant="oct_scalar",
                )

    def test_eval_v2_endpoint_receipt_is_independent_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            run_root = Path(temp) / "Building" / "oct_scalar"
            endpoint_path = run_root / "endpoints" / "step_30000.json"
            checkpoint_path = run_root / "checkpoints" / "step_30000.pt"
            protocol_path = run_root / "protocol.json"
            endpoint_path.parent.mkdir(parents=True)
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_bytes = b"independent-formal-checkpoint"
            checkpoint_path.write_bytes(checkpoint_bytes)
            protocol_path.write_text("{}\n", encoding="utf-8")

            snapshot = capture_occupancy_snapshot(DummyAnchor(3))
            source = fake_training_source_provenance()
            source["git_commit"] = FROZEN_TRAINING_COMMIT
            formal_sha = "a" * 64
            logical_protocol_sha = "b" * 64
            protocol = {
                "scene_name": "Building",
                "variant": "oct_scalar",
                "manifest_file_sha256": sha256_file(protocol_path),
                "manifest_sha256": logical_protocol_sha,
                "anchor_snapshot": snapshot,
                "source_provenance": source,
            }
            cost = {
                "schema": "uav-tgs-oct-cost-v1",
                "status": "endpoint_30000",
                "cumulative_optimizer_steps": 30_000,
                "segment_start_step": 0,
                "segment_optimizer_steps": 30_000,
                "optimizer_steps": 30_000,
                "rendered_views": 30_000,
                "raster_passes": 30_000,
                "raster_passes_per_view": 1.0,
                "wall_time_s": 123.0,
                "end_to_end_wall_time_s": 125.0,
                "pre_training_setup_wall_time_s": 2.0,
                "ms_per_step": 4.1,
                "peak_memory_reset_succeeded": True,
                "device": {
                    "cuda_available": True,
                    "device_name": "fixture GPU",
                    "peak_torch_allocated_bytes": 100,
                    "peak_torch_reserved_bytes": 200,
                },
                "metadata": {
                    "formal_protocol_sha256": formal_sha,
                    "scene": "Building",
                    "variant": "oct_scalar",
                    "segment_start_step": 0,
                },
            }
            checkpoint = {
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "protocol_receipt": {
                    "manifest_file_sha256": protocol["manifest_file_sha256"],
                    "manifest_sha256": logical_protocol_sha,
                    "formal_protocol_sha256": formal_sha,
                    "anchor_occupancy_sha256": snapshot["overall_sha256"],
                },
                "cost_summary": cost,
            }
            endpoint = {
                "schema": "uav-tgs-oct-endpoint-v2",
                "step": 30_000,
                "sequence_offset": 30_000,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "protocol_manifest_sha256": logical_protocol_sha,
                "formal_protocol_sha256": formal_sha,
                "anchor_occupancy_sha256": snapshot["overall_sha256"],
                "source_files_sha256": source["files_sha256"],
                "recent_loss_mean": 0.25,
                "resumed_from_step": None,
                "cost": cost,
            }
            endpoint["endpoint_sha256"] = sha256_json(endpoint)
            endpoint_path.write_text(json.dumps(endpoint), encoding="utf-8")
            payload, identity, flags = _load_and_validate_endpoint_receipt(
                endpoint_receipt_path=endpoint_path,
                checkpoint_path=checkpoint_path,
                protocol_path=protocol_path,
                checkpoint=checkpoint,
                protocol=protocol,
                binding=SimpleNamespace(formal_protocol_sha256=formal_sha),
                runtime=SimpleNamespace(anchor_snapshot=snapshot),
            )
            self.assertEqual(payload["cost"], cost)
            self.assertEqual(identity["checkpoint_sha256"], sha256_file(checkpoint_path))
            self.assertTrue(all(flags.values()))

            checkpoint_path.write_bytes(checkpoint_bytes + b"tamper")
            with self.assertRaises(RuntimeError):
                _load_and_validate_endpoint_receipt(
                    endpoint_receipt_path=endpoint_path,
                    checkpoint_path=checkpoint_path,
                    protocol_path=protocol_path,
                    checkpoint=checkpoint,
                    protocol=protocol,
                    binding=SimpleNamespace(formal_protocol_sha256=formal_sha),
                    runtime=SimpleNamespace(anchor_snapshot=snapshot),
                )
            checkpoint_path.write_bytes(checkpoint_bytes)

            wrong_sha = dict(endpoint)
            wrong_sha["checkpoint_sha256"] = "0" * 64
            wrong_sha.pop("endpoint_sha256")
            wrong_sha["endpoint_sha256"] = sha256_json(wrong_sha)
            endpoint_path.write_text(json.dumps(wrong_sha), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _load_and_validate_endpoint_receipt(
                    endpoint_receipt_path=endpoint_path,
                    checkpoint_path=checkpoint_path,
                    protocol_path=protocol_path,
                    checkpoint=checkpoint,
                    protocol=protocol,
                    binding=SimpleNamespace(formal_protocol_sha256=formal_sha),
                    runtime=SimpleNamespace(anchor_snapshot=snapshot),
                )

            bad_self_hash = dict(endpoint)
            bad_self_hash["endpoint_sha256"] = "f" * 64
            endpoint_path.write_text(json.dumps(bad_self_hash), encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_and_validate_endpoint_receipt(
                    endpoint_receipt_path=endpoint_path,
                    checkpoint_path=checkpoint_path,
                    protocol_path=protocol_path,
                    checkpoint=checkpoint,
                    protocol=protocol,
                    binding=SimpleNamespace(formal_protocol_sha256=formal_sha),
                    runtime=SimpleNamespace(anchor_snapshot=snapshot),
                )

            bad_cost = dict(cost)
            bad_cost["wall_time_s"] = -1.0
            bad_cost_endpoint = dict(endpoint)
            bad_cost_endpoint["cost"] = bad_cost
            bad_cost_endpoint.pop("endpoint_sha256")
            bad_cost_endpoint["endpoint_sha256"] = sha256_json(bad_cost_endpoint)
            bad_cost_checkpoint = dict(checkpoint)
            bad_cost_checkpoint["cost_summary"] = bad_cost
            endpoint_path.write_text(json.dumps(bad_cost_endpoint), encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_and_validate_endpoint_receipt(
                    endpoint_receipt_path=endpoint_path,
                    checkpoint_path=checkpoint_path,
                    protocol_path=protocol_path,
                    checkpoint=bad_cost_checkpoint,
                    protocol=protocol,
                    binding=SimpleNamespace(formal_protocol_sha256=formal_sha),
                    runtime=SimpleNamespace(anchor_snapshot=snapshot),
                )

    def test_hotspot_receipt_is_train_only_and_eval_rejects_non_r1_before_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binding = FormalFixture(root).binding()
            payload = {
                "schema": HOTSPOT_SCHEMA,
                "scene_name": "Building",
                "source_receipt": binding.hotspot_receipt(),
                "source_split": "train",
                "test_statistics_used": False,
                "quantile": FORMAL_HOTSPOT_QUANTILE,
                "histogram_bins": FORMAL_HOTSPOT_BINS,
                "threshold_c": 25.0,
                "range_c": [binding.tmin_c, binding.tmax_c],
                "valid_train_pixels": 1000,
                "train_view_ids_sha256": sha256_json(binding.names("train")),
            }
            payload["threshold_sha256"] = sha256_json(payload)
            path = root / "hotspot.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_load_hotspot_threshold(path, binding)["threshold_c"], 25.0)
            payload["source_split"] = "test"
            basis = dict(payload)
            basis.pop("threshold_sha256")
            payload["threshold_sha256"] = sha256_json(basis)
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_hotspot_threshold(path, binding)
            payload["source_split"] = "train"
            payload["quantile"] = 0.90
            basis = dict(payload)
            basis.pop("threshold_sha256")
            payload["threshold_sha256"] = sha256_json(basis)
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = _load_hotspot_threshold(path, binding)
            with self.assertRaises(ValueError):
                _validate_formal_hotspot_threshold(loaded, binding)
        with self.assertRaises(ValueError):
            command_eval(SimpleNamespace(resolution=2))

        current_source = fake_training_source_provenance()
        drifted_source = dict(current_source)
        drifted_source["git_commit"] = "5" * 40
        with patch(
            "tools.oct_gs_formal._formal_source_provenance",
            return_value=current_source,
        ), patch(
            "tools.oct_gs_formal.load_oct_protocol_manifest",
            return_value={"source_provenance": drifted_source},
        ):
            with self.assertRaisesRegex(RuntimeError, "evaluation/training protocol"):
                command_eval(
                    SimpleNamespace(
                        resolution=-1,
                        protocol_manifest="unused-protocol.json",
                    )
                )


if __name__ == "__main__":
    unittest.main()
