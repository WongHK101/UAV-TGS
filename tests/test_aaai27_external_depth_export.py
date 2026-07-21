from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.aaai27_external.depth_export_common import (
    camera_z_from_ray_distance,
    validate_render_binding,
    write_view_npz,
)
from tools.aaai27_external.benchmark_thermonerf_render import _thermal_u8
from tools.aaai27_external.export_thermonerf_expected_depth import (
    _rescale_cameras_exact,
)


def test_thermonerf_ray_distance_to_metric_camera_z() -> None:
    distance = np.array([[4.0, 12.0]], dtype=np.float32)
    directions_norm = np.array([[1.0, 2.0]], dtype=np.float32)
    result = camera_z_from_ray_distance(distance, directions_norm, 0.5)
    np.testing.assert_allclose(result, np.array([[8.0, 12.0]], dtype=np.float32))


def test_thermonerf_camera_z_rejects_invalid_scale() -> None:
    with pytest.raises(ValueError, match="dataparser_scale"):
        camera_z_from_ray_distance(np.ones((1, 1)), np.ones((1, 1)), 0.0)


def test_render_binding_requires_authoritative_order(tmp_path: Path) -> None:
    records = [
        {"pair_id": "0001", "split": "test"},
        {"pair_id": "0009", "split": "test"},
    ]
    path = tmp_path / "binding.json"
    path.write_text(
        json.dumps(
            {
                "protocol": "aaai27_hold8_v2",
                "rows": [
                    {"raw_index": 0, "pair_id": "0001", "formal_gt_name": "0001.png"},
                    {"raw_index": 1, "pair_id": "0009", "formal_gt_name": "0009.png"},
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = validate_render_binding(path, records)
    assert [row["pair_id"] for row in rows] == ["0001", "0009"]

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"].reverse()
    for index, row in enumerate(payload["rows"]):
        row["raw_index"] = index
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authoritative"):
        validate_render_binding(path, records)


def test_view_npz_preserves_missing_support(tmp_path: Path) -> None:
    expected = np.array([[2.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    weight = np.array([[0.5, 0.0], [1.0e-10, 0.7]], dtype=np.float32)
    path, identity = write_view_npz(
        output_root=tmp_path,
        index=0,
        expected_depth=expected,
        weight_sum=weight,
    )
    assert identity["npz_size_bytes"] == path.stat().st_size
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["has_finite_positive_depth_sample"],
            np.array([[True, False], [False, True]]),
        )


def test_thermonerf_exact_reference_raster_rescales_axes_independently() -> None:
    class Cameras:
        width = torch.tensor([[1257]], dtype=torch.int64)
        height = torch.tensor([[1006]], dtype=torch.int64)
        fx = torch.tensor([[2000.0]])
        fy = torch.tensor([[2100.0]])
        cx = torch.tensor([[628.5]])
        cy = torch.tensor([[503.0]])

    cameras = Cameras()
    _rescale_cameras_exact(cameras, 314, 252)
    assert int(cameras.width.item()) == 314
    assert int(cameras.height.item()) == 252
    assert cameras.fx.item() == pytest.approx(2000.0 * 314.0 / 1257.0)
    assert cameras.fy.item() == pytest.approx(2100.0 * 252.0 / 1006.0)


def test_thermonerf_thermal_u8_is_finite_and_clipped() -> None:
    value = torch.tensor([[[-0.1], [0.5], [1.2]]], dtype=torch.float32)
    np.testing.assert_array_equal(
        _thermal_u8(value), np.array([[0, 128, 255]], dtype=np.uint8)
    )
    with pytest.raises(ValueError, match="finite"):
        _thermal_u8(torch.tensor([[[float("nan")]]]))
