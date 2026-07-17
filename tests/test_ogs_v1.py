import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.ogs_v1 import (
    build_ogs_cache,
    compute_observability_from_moments,
    covariance_thickness,
    load_ogs_cache,
    match_scale_controls,
    moments_from_bearings,
    ogs_v1_loss,
    save_ogs_cache,
)


def _cache(
    scales,
    rotations,
    directions,
    observability=None,
    visible_count=None,
    opacity=None,
):
    count = scales.shape[0]
    if observability is None:
        observability = torch.zeros(count, dtype=torch.float64)
    if visible_count is None:
        visible_count = torch.full((count,), 8, dtype=torch.int64)
    if opacity is None:
        opacity = torch.ones(count)
    return build_ogs_cache(
        observability=observability,
        weakest_direction=directions,
        visible_count=visible_count,
        activated_opacity=opacity,
        anchor_scales=scales,
        anchor_rotations=rotations,
        metadata={"anchor_sha256": "anchor", "camera_sha256": "camera"},
    )


class OgsObservabilityTests(unittest.TestCase):
    def test_one_sided_bearings_have_zero_observability_and_aligned_n(self):
        bearings = torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        moments = moments_from_bearings(bearings).unsqueeze(0)
        result = compute_observability_from_moments(
            moments, torch.tensor([len(bearings)]), chunk_size=1
        )
        self.assertAlmostEqual(float(result["observability"][0]), 0.0, places=12)
        self.assertAlmostEqual(
            float(result["weakest_direction"][0].abs().dot(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))),
            1.0,
            places=12,
        )
        torch.testing.assert_close(
            result["eigenvalues"][0],
            torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64),
        )
        self.assertLess(float(result["trace_m_residual"][0]), 1e-12)
        self.assertLess(float(result["trace_h_residual"][0]), 1e-12)

    def test_isotropic_axis_bearings_have_unit_observability(self):
        bearings = torch.eye(3, dtype=torch.float64)
        moments = moments_from_bearings(bearings).unsqueeze(0)
        result = compute_observability_from_moments(
            moments, torch.tensor([3]), chunk_size=1
        )
        self.assertAlmostEqual(float(result["observability"][0]), 1.0, places=10)
        torch.testing.assert_close(
            result["eigenvalues"][0],
            torch.full((3,), 2.0 / 3.0, dtype=torch.float64),
        )

    def test_chunked_eigendecomposition_matches_direct_batch(self):
        generator = torch.Generator().manual_seed(123)
        all_moments = []
        counts = []
        for index in range(17):
            bearings = torch.randn((index + 2, 3), generator=generator, dtype=torch.float64)
            all_moments.append(moments_from_bearings(bearings))
            counts.append(index + 2)
        moments = torch.stack(all_moments)
        counts = torch.tensor(counts, dtype=torch.int64)
        direct = compute_observability_from_moments(moments, counts, chunk_size=17)
        chunked = compute_observability_from_moments(moments, counts, chunk_size=4)
        for key in (
            "observability",
            "eigenvalues",
            "weak_eigengap",
            "min_eigengap",
            "trace_m_residual",
            "trace_h_residual",
        ):
            torch.testing.assert_close(chunked[key], direct[key], rtol=0, atol=1e-13)
        alignment = (chunked["weakest_direction"] * direct["weakest_direction"]).sum(-1).abs()
        torch.testing.assert_close(alignment, torch.ones_like(alignment), atol=1e-12, rtol=0)


class OgsLossTests(unittest.TestCase):
    def test_direction_sign_does_not_change_thickness_or_loss(self):
        scales = torch.tensor([[4.0, 1.0, 2.0]], dtype=torch.float64)
        rotations = torch.tensor([[1.0, 0.2, -0.1, 0.3]], dtype=torch.float64)
        direction = torch.tensor([[0.7, 0.2, -0.4]], dtype=torch.float64)
        positive = covariance_thickness(scales, rotations, direction)
        negative = covariance_thickness(scales, rotations, -direction)
        torch.testing.assert_close(positive["parallel"], negative["parallel"])
        cache_positive = _cache(scales, rotations, direction)
        cache_negative = _cache(scales, rotations, -direction)
        value_positive = ogs_v1_loss(scales, rotations, cache_positive)["loss"]
        value_negative = ogs_v1_loss(scales, rotations, cache_negative)["loss"]
        torch.testing.assert_close(value_positive, value_negative)

    def test_loss_is_zero_at_and_below_fixed_threshold(self):
        anchor_scales = torch.ones((2, 3), dtype=torch.float64)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64)
        directions = torch.tensor([[1.0, 0.0, 0.0]] * 2, dtype=torch.float64)
        cache = _cache(anchor_scales, rotations, directions)
        current_scales = torch.tensor(
            [[3.0, 1.0, 1.0], [2.5, 1.0, 1.0]], dtype=torch.float64
        )
        result = ogs_v1_loss(current_scales, rotations, cache)
        self.assertAlmostEqual(float(result["loss"]), 0.0, places=12)
        self.assertEqual(result["active_count"], 0)

    def test_gradient_scope_is_only_scale_and_rotation(self):
        anchor_scales = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
        anchor_rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        directions = torch.tensor([[0.6, 0.4, 0.7]], dtype=torch.float64)
        cache = _cache(anchor_scales, anchor_rotations, directions)
        scales = torch.tensor([[5.0, 1.0, 0.8]], dtype=torch.float64, requires_grad=True)
        rotations = torch.tensor(
            [[1.0, 0.1, -0.2, 0.3]], dtype=torch.float64, requires_grad=True
        )
        unrelated = [
            torch.randn((1, 3), dtype=torch.float64, requires_grad=True),  # xyz
            torch.randn((1,), dtype=torch.float64, requires_grad=True),  # opacity
            torch.randn((1, 3), dtype=torch.float64, requires_grad=True),  # f_dc
            torch.randn((1, 4, 3), dtype=torch.float64, requires_grad=True),  # f_rest
            torch.randn((1,), dtype=torch.float64, requires_grad=True),  # exposure
        ]
        loss = ogs_v1_loss(scales, rotations, cache)["loss"]
        loss.backward()
        self.assertIsNotNone(scales.grad)
        self.assertIsNotNone(rotations.grad)
        self.assertGreater(float(scales.grad.norm()), 0.0)
        self.assertGreater(float(rotations.grad.norm()), 0.0)
        for tensor in unrelated:
            self.assertIsNone(tensor.grad)

    def test_fixed_eligible_denominator_not_active_denominator(self):
        anchor_scales = torch.ones((2, 3), dtype=torch.float64)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64)
        directions = torch.tensor([[1.0, 0.0, 0.0]] * 2, dtype=torch.float64)
        cache = _cache(anchor_scales, rotations, directions)
        scales = torch.tensor([[6.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        result = ogs_v1_loss(scales, rotations, cache)
        expected_penalty = torch.log(torch.tensor(2.0, dtype=torch.float64)).square()
        self.assertEqual(result["eligible_count"], 2)
        self.assertEqual(result["active_count"], 1)
        torch.testing.assert_close(result["loss"], expected_penalty / 2.0, atol=1e-8, rtol=1e-8)

    def test_zero_eligible_fails_closed(self):
        scales = torch.ones((1, 3), dtype=torch.float64)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        directions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        cache = _cache(
            scales,
            rotations,
            directions,
            visible_count=torch.tensor([7]),
        )
        with self.assertRaisesRegex(RuntimeError, "zero fixed eligible"):
            ogs_v1_loss(scales, rotations, cache)


class OgsCacheTests(unittest.TestCase):
    def test_float64_audit_is_compacted_to_float32_with_controlled_error(self):
        generator = torch.Generator().manual_seed(9)
        bearings = torch.randn((20, 3), generator=generator, dtype=torch.float64)
        audit = compute_observability_from_moments(
            moments_from_bearings(bearings).unsqueeze(0),
            torch.tensor([20]),
        )
        scales = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        cache = build_ogs_cache(
            audit["observability"],
            audit["weakest_direction"],
            torch.tensor([20]),
            torch.tensor([0.5]),
            scales,
            rotations,
            metadata={"anchor_sha256": "abc"},
        )
        self.assertEqual(cache["observability"].dtype, torch.float32)
        self.assertEqual(cache["weakest_direction"].dtype, torch.float32)
        self.assertEqual(cache["perpendicular_thickness"].dtype, torch.float32)
        torch.testing.assert_close(
            cache["observability"].double(),
            audit["observability"],
            atol=1e-7,
            rtol=1e-7,
        )
        alignment = (
            cache["weakest_direction"].double() * audit["weakest_direction"]
        ).sum(-1).abs()
        torch.testing.assert_close(alignment, torch.ones_like(alignment), atol=1e-6, rtol=0)

    def test_saved_cache_roundtrip_and_hash_mismatch_fail_closed(self):
        scales = torch.ones((2, 3), dtype=torch.float64)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64)
        directions = torch.tensor([[1.0, 0.0, 0.0]] * 2, dtype=torch.float64)
        cache = _cache(scales, rotations, directions)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            save_ogs_cache(path, cache)
            loaded = load_ogs_cache(
                path,
                expected_gaussian_count=2,
                expected_metadata={"anchor_sha256": "anchor"},
            )
            self.assertEqual(loaded["cache_sha256"], cache["cache_sha256"])
            with self.assertRaisesRegex(ValueError, "Gaussian count mismatch"):
                load_ogs_cache(path, expected_gaussian_count=3)
        tampered = copy.deepcopy(cache)
        tampered["observability"][0] = 0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.pt"
            torch.save(tampered, path)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_ogs_cache(path)


class OgsControlMatchingTests(unittest.TestCase):
    def test_matching_is_deterministic_tie_broken_and_records_reuse(self):
        # Targets 0 and 1 are identical. Eligible controls are deliberately
        # symmetric, making the Gaussian index the deterministic tie-break.
        scales = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        )
        eligible = np.ones(9, dtype=bool)
        first = match_scale_controls(scales, eligible, [1, 0], controls_per_target=5)
        second = match_scale_controls(scales, eligible, [0, 1], controls_per_target=5)
        self.assertEqual(first, second)
        self.assertEqual(first["records"][0]["target_index"], 0)
        self.assertEqual(first["records"][0]["control_indices"], [2, 3, 4, 5, 6])
        self.assertEqual(first["records"][1]["control_indices"][:2], [7, 8])
        self.assertEqual(first["records"][1]["control_indices"][2:], [2, 3, 4])
        self.assertEqual(first["reuse_count"], 3)
        self.assertTrue(first["reuse_required"])


if __name__ == "__main__":
    unittest.main()
