from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.geometric_repeatability.evaluate_zero_update_resplat_diagnostic import (
    ALPHA_THRESHOLD,
    SELECTION_GROUPS,
    _canonical_sha256,
    compute_pair_metrics,
    create_selection_receipt,
    load_and_verify_selection_receipt,
    render_model_view,
    restore_checkpoint_for_render,
    save_comparison_panel,
    select_fixed_formal_records,
)


def _records(*, include_test: bool = True) -> list[dict]:
    records: list[dict] = []
    for split, orientation in SELECTION_GROUPS:
        for index in range(12):
            records.append(
                {
                    "scene": "InternalRoad",
                    "split": split,
                    "stratum": f"{orientation}:p-090",
                    "strip_id": f"{orientation}-strip",
                    "block_index": index // 4,
                    "block_offset": index % 4,
                    "position_in_strip": index,
                    "pair_id": f"{split[0]}{orientation[0]}_{index:03d}",
                    "camera_name": f"{split[0]}{orientation[0]}_{index:03d}.JPG",
                }
            )
    if include_test:
        # Deliberately lacks orientation: the selector must skip test before
        # inspecting/classifying any of its fields.
        records.append({"split": "test", "pair_id": "forbidden_test"})
    return list(reversed(records))


def _probe(selected: list[dict]) -> dict:
    views = []
    for index, row in enumerate(selected):
        name = f"{row['image_stem']}.JPG"
        c2w = np.eye(4, dtype=np.float64)
        c2w[0, 3] = float(index)
        views.append(
            {
                "view_id": f"{index:05d}",
                "image_name": name,
                "width": 8,
                "height": 6,
                "fx": 7.0,
                "fy": 7.0,
                "cx": 4.0,
                "cy": 3.0,
                "camera_to_world": c2w.tolist(),
                "native_camera_to_world": c2w.tolist(),
                "bound_split": row["split"],
                "split": row["split"],
            }
        )
    return {"camera_manifest_type": "fixture", "views": views}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_fixed_selection_is_four_groups_at_ordered_offset_7_and_ignores_test() -> None:
    with_test = select_fixed_formal_records({"records": _records(include_test=True)})
    without_test = select_fixed_formal_records({"records": _records(include_test=False)})

    assert [(row["split"], row["orientation"]) for row in with_test] == list(SELECTION_GROUPS)
    assert [row["image_stem"] for row in with_test] == [
        "tn_007",
        "to_007",
        "gn_007",
        "go_007",
    ]
    assert _canonical_sha256(with_test) == _canonical_sha256(without_test)
    assert all(row["ordered_group_offset_zero_based"] == 7 for row in with_test)
    assert all(row["split"] != "test" for row in with_test)


def test_fixed_selection_fails_closed_when_group_is_too_short() -> None:
    records = _records(include_test=False)
    records = [
        row
        for row in records
        if not (row["split"] == "guard" and row["stratum"].startswith("oblique") and row["position_in_strip"] > 5)
    ]
    with pytest.raises(ValueError, match="guard/oblique"):
        select_fixed_formal_records({"records": records})


def test_selection_receipt_is_precomputed_and_tamper_evident(tmp_path: Path) -> None:
    split_path = tmp_path / "bound_split.json"
    probe_path = tmp_path / "probe.json"
    receipt_path = tmp_path / "selection_receipt.json"
    selected = select_fixed_formal_records({"records": _records()})
    _write_json(split_path, {"records": _records()})
    _write_json(probe_path, _probe(selected))

    create_selection_receipt(
        scene_name="InternalRoad",
        formal_split=split_path,
        probe_camera_manifest=probe_path,
        native_cameras_json=None,
        receipt_path=receipt_path,
    )
    verified = load_and_verify_selection_receipt(
        receipt_path=receipt_path,
        scene_name="InternalRoad",
        formal_split=split_path,
        probe_camera_manifest=probe_path,
        native_cameras_json=None,
    )
    assert len(verified["frozen_views"]) == 4
    assert verified["selection_protocol"]["test_views_selected"] is False

    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["frozen_views"][0]["image_stem"] = "changed"
    _write_json(receipt_path, tampered)
    with pytest.raises(ValueError, match="content hash"):
        load_and_verify_selection_receipt(
            receipt_path=receipt_path,
            scene_name="InternalRoad",
            formal_split=split_path,
            probe_camera_manifest=probe_path,
            native_cameras_json=None,
        )


def test_pair_metrics_cover_alpha_and_keep_identical_psnr_finite() -> None:
    raw_rgb = np.zeros((2, 3, 3), dtype=np.float32)
    resplat_rgb = raw_rgb.copy()
    raw_alpha = np.array([[0.0, 0.02, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32)
    resplat_alpha = np.array([[0.0, 0.02, 0.0], [0.03, 0.0, 0.0]], dtype=np.float32)
    metrics = compute_pair_metrics(
        raw_rgb,
        resplat_rgb,
        raw_alpha,
        resplat_alpha,
        ssim_value=1.0,
        lpips_value=0.0,
    )
    assert metrics["alpha_coverage_iou"] == pytest.approx(1.0 / 3.0)
    assert metrics["alpha_mae"] == pytest.approx(float(np.mean(np.abs(raw_alpha - resplat_alpha))))
    assert metrics["alpha_p95_abs"] == pytest.approx(float(np.percentile(np.abs(raw_alpha - resplat_alpha), 95)))
    assert metrics["rgb_psnr_db"] == pytest.approx(120.0)
    assert all(np.isfinite(value) for value in metrics.values())
    assert ALPHA_THRESHOLD == 0.01


def test_pair_metrics_reject_nonfinite() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    alpha = np.zeros((2, 2), dtype=np.float32)
    rgb[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        compute_pair_metrics(rgb, rgb, alpha, alpha, ssim_value=1.0, lpips_value=0.0)


def test_fake_renderer_uses_white_override_black_background_and_panel(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class FakeGaussians:
        def __init__(self) -> None:
            self.get_xyz = torch.zeros((3, 3), dtype=torch.float32)

    calls: list[dict] = []

    def fake_render(camera, gaussians, pipeline, background, **kwargs):
        calls.append(
            {
                "background": background.detach().cpu().numpy().copy(),
                "override": kwargs["override_color"],
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        fill = 0.4 if kwargs["override_color"] is not None else 0.2
        return {"render": torch.full((3, 4, 5), fill, dtype=torch.float32)}

    rgb, alpha, _ = render_model_view(
        object(), FakeGaussians(), object(), render_fn=fake_render, white_background=False
    )
    assert rgb.shape == (4, 5, 3)
    assert alpha.shape == (4, 5)
    assert len(calls) == 2
    assert np.array_equal(calls[1]["background"], np.zeros(3, dtype=np.float32))
    assert tuple(calls[1]["override"].shape) == (3, 3)
    assert all(call["grad_enabled"] is False for call in calls)

    panel_path = tmp_path / "panel.png"
    save_comparison_panel(
        panel_path,
        raw_rgb=rgb,
        resplat_rgb=rgb,
        raw_alpha=alpha,
        resplat_alpha=alpha,
        title="fixture",
    )
    assert panel_path.is_file()
    with Image.open(panel_path) as panel:
        assert panel.size == (20, 46)


def test_checkpoint_restore_builds_no_optimizer_and_disables_grad(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class FakeModel:
        def __init__(self, sh_degree: int) -> None:
            self.max_sh_degree = sh_degree
            self.optimizer = "must_be_removed"

        @property
        def get_xyz(self):
            return self._xyz

    count = 2
    tensors = (
        torch.zeros((count, 3), requires_grad=True),
        torch.zeros((count, 1, 3), requires_grad=True),
        torch.zeros((count, 15, 3), requires_grad=True),
        torch.zeros((count, 3), requires_grad=True),
        torch.zeros((count, 4), requires_grad=True),
        torch.zeros((count, 1), requires_grad=True),
        torch.zeros((count,)),
        torch.zeros((count, 1)),
        torch.zeros((count, 1)),
    )
    checkpoint = tmp_path / "raw.pth"
    torch.save(((3, *tensors, {"state": {}, "param_groups": []}, 1.25), 30000), checkpoint)

    model, iteration = restore_checkpoint_for_render(
        checkpoint,
        sh_degree=3,
        device="cpu",
        model_factory=FakeModel,
    )
    assert iteration == 30000
    assert model.optimizer is None
    assert model.active_sh_degree == 3
    assert model.spatial_lr_scale == 1.25
    assert model.get_xyz.shape == (count, 3)
    assert all(
        not getattr(model, name).requires_grad
        for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")
    )
