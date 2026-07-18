import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from tools.geometric_repeatability.build_surface_aware_resplat_anchor import (
    SurfaceAwareResplatError,
    build_resplat_parameters,
    coincident_child_logit,
    moment_matched_axis,
    pca_surface_frame,
    quaternion_to_rotation,
    rotation_to_quaternion,
    run,
    validate_checkpoint_schema,
)


def _planar_neighbors() -> np.ndarray:
    # Deliberately anisotropic so the PCA tangent axes are unique.
    return np.asarray(
        [[float(x), float(y), 2.5] for x in range(-4, 4) for y in (-0.5, 0.5)],
        dtype=np.float64,
    )


def _synthetic_params(count: int = 100, candidate: int = 7):
    xyz = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 10.0
    xyz[candidate] = torch.tensor([0.0, 0.0, 2.5])
    features_dc = torch.arange(count * 3, dtype=torch.float32).reshape(count, 1, 3)
    features_rest = torch.arange(count * 15 * 3, dtype=torch.float32).reshape(count, 15, 3)
    scaling = torch.zeros((count, 3), dtype=torch.float32)
    scaling[candidate] = torch.log(torch.tensor([1.8, 0.8, 0.3], dtype=torch.float32))
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count, dtype=torch.float32)
    opacity = torch.linspace(-2.0, 2.0, count, dtype=torch.float32)[:, None]
    max_radii = torch.arange(count, dtype=torch.float32)
    gradient = torch.arange(count, dtype=torch.float32)[:, None] + 0.25
    denom = torch.arange(count, dtype=torch.float32)[:, None] + 1.0
    return (
        3,
        nn.Parameter(xyz),
        nn.Parameter(features_dc),
        nn.Parameter(features_rest),
        nn.Parameter(scaling),
        nn.Parameter(rotation),
        nn.Parameter(opacity),
        max_radii,
        gradient,
        denom,
        {"state": {"old": "must_not_survive"}, "param_groups": [{"name": "xyz"}]},
        1.0,
    )


class SurfaceAwareResplatMathTests(unittest.TestCase):
    def test_pca_is_float64_deterministic_and_right_handed(self):
        points = _planar_neighbors()
        frame_a, eigen_a, rank_a = pca_surface_frame(points)
        frame_b, eigen_b, rank_b = pca_surface_frame(points[::-1].copy())
        self.assertEqual(frame_a.dtype, np.float64)
        self.assertEqual(rank_a, 2)
        self.assertEqual(rank_b, 2)
        np.testing.assert_allclose(eigen_a, eigen_b, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(frame_a, frame_b, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(frame_a.T @ frame_a, np.eye(3), rtol=0.0, atol=1e-14)
        self.assertGreater(np.linalg.det(frame_a), 0.0)
        np.testing.assert_allclose(frame_a[:, 2], [0.0, 0.0, 1.0], atol=1e-14)

    def test_pca_rejects_rank_one_and_duplicate_centroids(self):
        line = np.asarray([[float(i), 0.0, 0.0] for i in range(16)])
        with self.assertRaisesRegex(SurfaceAwareResplatError, "rank"):
            pca_surface_frame(line)
        duplicate = _planar_neighbors()
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(SurfaceAwareResplatError, "distinct"):
            pca_surface_frame(duplicate)

    def test_quaternion_rotation_round_trip_handles_all_trace_branches(self):
        quaternions = (
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.35, -0.2, 0.4, 0.7],
        )
        for quaternion in quaternions:
            rotation = quaternion_to_rotation(quaternion)
            recovered = rotation_to_quaternion(rotation)
            np.testing.assert_allclose(
                quaternion_to_rotation(recovered), rotation, rtol=0.0, atol=1e-12
            )
            self.assertGreaterEqual(recovered[0], 0.0)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_tangent_moment_matching_is_exact(self):
        for std, count in ((2.75, 1), (2.75, 2), (2.75, 5), (8.0, 9)):
            child_std, spacing = moment_matched_axis(std, count)
            offsets = (
                np.arange(count, dtype=np.float64) - (count - 1.0) / 2.0
            ) * spacing
            recovered = child_std**2 + float(np.mean(offsets**2))
            self.assertAlmostEqual(recovered, std**2, places=12)
            self.assertAlmostEqual(float(np.mean(offsets)), 0.0, places=14)

    def test_coincident_ray_transmittance_is_preserved(self):
        for raw in (-8.0, -1.0, 0.0, 3.0, 12.0):
            parent_alpha = 1.0 / (1.0 + math.exp(-raw))
            for count in (1, 2, 7, 40):
                child_raw = coincident_child_logit(raw, count)
                child_alpha = 1.0 / (1.0 + math.exp(-child_raw))
                recovered = 1.0 - (1.0 - child_alpha) ** count
                self.assertAlmostEqual(recovered, parent_alpha, places=12)


class SurfaceAwareResplatCheckpointTests(unittest.TestCase):
    def test_build_is_deterministic_preserves_schema_and_row_positions(self):
        candidate = 7
        params = _synthetic_params(candidate=candidate)
        centroids = _planar_neighbors()
        output_a, plans_a = build_resplat_parameters(
            params, [candidate], {candidate: 1.0}, centroids
        )
        output_b, plans_b = build_resplat_parameters(
            params, [candidate], {candidate: 1.0}, centroids
        )
        self.assertEqual(validate_checkpoint_schema(output_a), 101)
        self.assertEqual(plans_a[0]["lattice_shape"], (2, 1))
        self.assertEqual(plans_a[0]["child_count"], 2)
        self.assertEqual(plans_a[0]["output_indices"], [candidate, 100])

        for field_index in range(1, 10):
            self.assertTrue(torch.equal(output_a[field_index], output_b[field_index]))
        for key in (
            "child_centers",
            "child_log_scale",
            "child_quaternion",
            "frame",
            "pca_eigenvalues",
        ):
            np.testing.assert_array_equal(plans_a[0][key], plans_b[0][key])

        noncandidate = [index for index in range(100) if index != candidate]
        rows = torch.tensor(noncandidate, dtype=torch.long)
        for field_index in range(1, 10):
            before = params[field_index].detach().index_select(0, rows)
            after = output_a[field_index].detach().index_select(0, rows)
            self.assertTrue(torch.equal(before, after))

        # Both children inherit the exact parent SH coefficients.
        for output_index in plans_a[0]["output_indices"]:
            self.assertTrue(
                torch.equal(output_a[2][output_index], params[2][candidate])
            )
            self.assertTrue(
                torch.equal(output_a[3][output_index], params[3][candidate])
            )
        np.testing.assert_allclose(
            plans_a[0]["child_centers"].mean(axis=0),
            params[1][candidate].detach().numpy(),
            rtol=0.0,
            atol=1e-12,
        )
        normal_offset = (
            plans_a[0]["child_centers"]
            - params[1][candidate].detach().numpy()[None, :]
        ) @ plans_a[0]["frame"][:, 2]
        np.testing.assert_allclose(normal_offset, 0.0, rtol=0.0, atol=1e-12)

        self.assertEqual(output_a[10]["state"], {})
        self.assertEqual(output_a[10]["param_groups"], [])
        self.assertIs(output_a[10]["fresh_optimizer_required"], True)

    def test_output_float32_opacity_preserves_transmittance(self):
        candidate = 7
        params = _synthetic_params(candidate=candidate)
        output, plans = build_resplat_parameters(
            params, [candidate], {candidate: 0.6}, _planar_neighbors()
        )
        child_indices = plans[0]["output_indices"]
        parent_alpha = torch.sigmoid(params[6][candidate, 0]).double()
        child_alpha = torch.sigmoid(output[6][child_indices[0], 0]).double()
        recovered = 1.0 - (1.0 - child_alpha) ** len(child_indices)
        self.assertLess(abs(float(recovered - parent_alpha)), 2e-7)

    def test_schema_and_growth_fail_closed(self):
        params = list(_synthetic_params())
        params[1] = nn.Parameter(params[1].detach().clone())
        params[1].data[0, 0] = float("nan")
        with self.assertRaisesRegex(SurfaceAwareResplatError, "NaN/Inf"):
            validate_checkpoint_schema(params)

        small = _synthetic_params(count=10, candidate=2)
        small[4].data[2] = torch.log(torch.tensor([10.0, 1.0, 0.2]))
        with self.assertRaisesRegex(SurfaceAwareResplatError, "exceeds 5%"):
            build_resplat_parameters(small, [2], {2: 1.0}, _planar_neighbors())

    def test_candidate_contract_is_sorted_unique_and_complete(self):
        params = _synthetic_params()
        with self.assertRaisesRegex(SurfaceAwareResplatError, "unique and sorted"):
            build_resplat_parameters(params, [7, 7], {7: 1.0}, _planar_neighbors())
        with self.assertRaisesRegex(SurfaceAwareResplatError, "index sets differ"):
            build_resplat_parameters(params, [7], {8: 1.0}, _planar_neighbors())

    def test_preflight_writes_no_model_and_build_writes_checkpoint_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "raw_chkpnt30000.pth"
            torch.save((_synthetic_params(), 30000), checkpoint)
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            sparse = root / "points3D.txt"
            sparse.write_text(
                "# synthetic locked sparse points\n"
                + "".join(
                    f"{index + 1} {x} {y} {z} 255 255 255 0.1\n"
                    for index, (x, y, z) in enumerate(_planar_neighbors())
                ),
                encoding="ascii",
            )
            sparse_sha = hashlib.sha256(sparse.read_bytes()).hexdigest()
            raw_ply_sha = "a" * 64
            scsp = {
                "status": "passed",
                "method": "scsp",
                "input": {
                    "checkpoint_sha256": checkpoint_sha,
                    "ply_sha256": raw_ply_sha,
                },
                "invariants": {"no_training": True},
                "sparse_support": {
                    "points3d_sha256": sparse_sha,
                    "voxel_size": 0.25,
                    "raw_point_count": 16,
                    "voxel_centroid_count": 16,
                },
                "modified_rows": [
                    {"gaussian_index": 7, "local_support_distance": 1.0}
                ],
            }
            scsp_path = root / "adaptive_scale_manifest.json"
            scsp_path.write_text(
                json.dumps(scsp, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            scsp_sha = hashlib.sha256(scsp_path.read_bytes()).hexdigest()
            indices_hash = hashlib.sha256(b"[7]").hexdigest()
            candidate = {
                "protocol": "uav-tgs-conditional-resplat-decision-v2",
                "status": "passed",
                "scene_name": "Synthetic",
                "decision_contract": {
                    "constants": {"locked_scsp_manifest_sha256": scsp_sha}
                },
                "selection_policy": {
                    "formal_test_metrics_used": False,
                    "formal_test_npz_open_count_before_receipt": 0,
                },
                "inputs": {
                    "gaussian_count": 100,
                    "gaussian_index_anchor_sha256": raw_ply_sha,
                    "scsp_manifest": {"sha256": scsp_sha},
                    "sparse_support": {
                        "points3d_sha256": sparse_sha,
                        "voxel_size": 0.25,
                    },
                },
                "decision": {
                    "triggered": True,
                    "decision": "execute_one_deterministic_resplat",
                    "eligible_candidate_indices": [7],
                    "candidate_rule": {
                        "candidate_count": 1,
                        "candidate_indices": [7],
                        "candidate_indices_sha256": indices_hash,
                        "candidate_diagnostics": [
                            {
                                "gaussian_index": 7,
                                "local_support_distance_m": 1.0,
                            }
                        ],
                    },
                },
            }
            candidate_path = root / "conditional_resplat_decision_receipt.json"
            candidate_path.write_text(
                json.dumps(candidate, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )

            def arguments(mode: str, output: Path) -> argparse.Namespace:
                return argparse.Namespace(
                    mode=mode,
                    input_checkpoint=str(checkpoint),
                    candidate_receipt=str(candidate_path),
                    scsp_manifest=str(scsp_path),
                    sparse_points=str(sparse),
                    output_dir=str(output),
                    anchor_iteration=30000,
                    expected_checkpoint_sha256="",
                    expected_candidate_receipt_sha256="",
                    expected_scsp_manifest_sha256="",
                    expected_sparse_points_sha256="",
                    code_commit="0" * 40,
                )

            preflight = root / "preflight"
            preflight_manifest = run(arguments("preflight", preflight))
            self.assertEqual(preflight_manifest["counts"]["gaussian_count_after"], 101)
            self.assertEqual(
                sorted(path.name for path in preflight.iterdir()),
                ["surface_aware_resplat_preflight.json"],
            )
            self.assertFalse(any(preflight.rglob("*.ply")))
            self.assertFalse(any(preflight.rglob("*.pth")))

            built = root / "built"
            build_manifest = run(arguments("build", built))
            self.assertEqual(build_manifest["counts"]["gaussian_count_after"], 101)
            self.assertEqual(
                sorted(path.name for path in built.iterdir()),
                ["chkpnt30000.pth", "surface_aware_resplat_manifest.json"],
            )
            self.assertFalse(any(built.rglob("*.ply")))
            payload = torch.load(
                built / "chkpnt30000.pth", map_location="cpu", weights_only=False
            )
            self.assertEqual(payload[1], 30000)
            self.assertEqual(validate_checkpoint_schema(payload[0]), 101)
            self.assertIs(payload[0][10]["fresh_optimizer_required"], True)


if __name__ == "__main__":
    unittest.main()
