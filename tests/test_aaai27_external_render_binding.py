from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
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
