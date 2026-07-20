from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "aaai27_external" / "normalize_sequence_renders.py"
SPEC = importlib.util.spec_from_file_location("normalize_sequence_renders", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


@pytest.mark.skipif(os.name == "nt", reason="Windows test host has no symlink privilege")
def test_sequence_binding_and_gt_identity(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw_gt = tmp_path / "raw_gt"
    formal = tmp_path / "formal"
    for directory in (raw, raw_gt, formal):
        directory.mkdir()
    names = ["0001.JPG", "0009.JPG"]
    (tmp_path / "test.txt").write_text("\n".join(names) + "\n")
    for index, name in enumerate(names):
        color = (index * 20, 10, 30)
        Image.new("RGB", (4, 3), color).save(raw / f"{index:05d}.png")
        Image.new("RGB", (4, 3), color).save(raw_gt / f"{index:05d}.png")
        Image.new("RGB", (4, 3), color).save(formal / name, format="PNG")
    value = mod.bind(
        raw_render_root=raw,
        raw_gt_root=raw_gt,
        formal_gt_root=formal,
        test_list=tmp_path / "test.txt",
        output_model_root=tmp_path / "bound",
        method="fixture",
        modality="thermal",
    )
    assert value["test_count"] == 2
    assert value["raw_gt_verified_pixel_exact"] is True
    assert (tmp_path / "bound" / "test" / "ours_formal" / "renders" / "0001.png").is_symlink()


def test_official_pil_default_resize_verification(tmp_path: Path) -> None:
    formal = tmp_path / "formal.png"
    raw = tmp_path / "raw.png"
    image = Image.new("RGB", (7, 5), (20, 40, 60))
    image.putpixel((3, 2), (200, 100, 10))
    image.save(formal)
    image.resize((9, 6)).save(raw)
    drift, formal_resolution, raw_resolution = mod._verify_raw_gt(
        raw, formal, "pil-default-resize-to-raw"
    )
    assert drift == 0
    assert formal_resolution == (7, 5)
    assert raw_resolution == (9, 6)


def test_resize_verification_rejects_unrelated_gt(tmp_path: Path) -> None:
    formal = tmp_path / "formal.png"
    raw = tmp_path / "raw.png"
    Image.new("RGB", (7, 5), (20, 40, 60)).save(formal)
    Image.new("RGB", (9, 6), (21, 40, 60)).save(raw)
    with pytest.raises(ValueError, match="raw/formal GT mismatch"):
        mod._verify_raw_gt(raw, formal, "pil-default-resize-to-raw")


def test_hotiron_gray_inverse_returns_exact_lut_colors(tmp_path: Path) -> None:
    source = tmp_path / "gray.png"
    output = tmp_path / "normalized.png"
    table = mod._hotiron_grayscale_table()
    Image.fromarray(table[[0, 64, 128, 255]][None, :], mode="L").save(source)
    mod._normalize_render(
        source, output, (4, 1), "hotiron-grayscale-inverse-to-formal"
    )
    result = mod._rgb(output)[0]
    lut = mod.hot_iron_lut()
    assert all(any((pixel == color).all() for color in lut) for pixel in result)


def test_exact_render_policy_rejects_shape_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "render.png"
    Image.new("RGB", (4, 3), (0, 0, 0)).save(source)
    with pytest.raises(ValueError, match="render/formal resolution mismatch"):
        mod._normalize_render(source, tmp_path / "out.png", (5, 3), "exact")


def test_right_half_policy_extracts_prediction_panel(tmp_path: Path) -> None:
    source = tmp_path / "combined.png"
    destination = tmp_path / "prediction.png"
    ground_truth = np.full((2, 3, 3), 25, dtype=np.uint8)
    prediction = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[90, 80, 70], [60, 50, 40], [30, 20, 10]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(np.concatenate([ground_truth, prediction], axis=1)).save(source)

    mod._normalize_render(source, destination, (3, 2), "right-half-to-formal")

    assert np.array_equal(mod._rgb(destination), prediction)


def test_right_half_policy_rejects_nonpaired_width(tmp_path: Path) -> None:
    source = tmp_path / "bad.png"
    Image.fromarray(np.zeros((2, 5, 3), dtype=np.uint8)).save(source)

    with pytest.raises(ValueError, match="side-by-side"):
        mod._normalize_render(
            source,
            tmp_path / "prediction.png",
            (3, 2),
            "right-half-to-formal",
        )


def test_parser_accepts_method_specific_render_glob() -> None:
    args = mod.build_parser().parse_args(
        [
            "--raw-render-root",
            "renders",
            "--raw-render-glob",
            "thermal_*.jpg",
            "--formal-gt-root",
            "gt",
            "--test-list",
            "test.txt",
            "--output-model-root",
            "normalized",
            "--method",
            "ThermoNeRF",
            "--modality",
            "thermal",
        ]
    )
    assert args.raw_render_glob == "thermal_*.jpg"
