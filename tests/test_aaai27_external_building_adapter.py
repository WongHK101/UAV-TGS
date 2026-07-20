from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "aaai27_external" / "building_adapter.py"
SPEC = importlib.util.spec_from_file_location("building_adapter", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_hold8_natural_order() -> None:
    names = [f"{index:04d}.JPG" for index in range(1, 33)]
    test = names[::8]
    train = [name for index, name in enumerate(names) if index % 8]
    assert mod.validate_hold8(train, test) == names


def test_hold8_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        mod.validate_hold8(["0002.JPG", "0001.JPG"], ["0001.JPG"])


def test_colmap_pose_is_finite_and_flips_axes() -> None:
    matrix = mod._colmap_to_nerfstudio_c2w([1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    assert matrix == [
        [1.0, -0.0, -0.0, -1.0],
        [0.0, -1.0, -0.0, -2.0],
        [0.0, -0.0, -1.0, -3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_compact_colmap_images_preserves_headers_and_drops_points(tmp_path: Path) -> None:
    source = tmp_path / "images.txt"
    output = tmp_path / "compact.txt"
    source.write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 0001.JPG\n"
        "10 20 3 30 40 -1\n"
        "2 1 0 0 0 1 0 0 1 0002.JPG\n"
        "50 60 4\n",
        encoding="utf-8",
    )
    assert mod._write_compact_colmap_images(source, output) == 2
    assert output.read_text(encoding="utf-8") == (
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 0001.JPG\n\n"
        "2 1 0 0 0 1 0 0 1 0002.JPG\n\n"
    )
    second = tmp_path / "compact_second_pass.txt"
    assert mod._write_compact_colmap_images(output, second) == 2
    assert second.read_bytes() == output.read_bytes()
    parsed = mod._parse_colmap_images_text(output)
    assert list(parsed) == ["0001.JPG", "0002.JPG"]
    assert parsed["0002.JPG"]["tvec"] == [1.0, 0.0, 0.0]


@pytest.mark.skipif(os.name == "nt", reason="Windows test host has no symlink privilege")
def test_thermal_alias_preserves_png_bytes(tmp_path: Path) -> None:
    source = tmp_path / "canonical" / "0001.png"
    source.parent.mkdir()
    source.write_bytes(b"\x89PNG\r\n\x1a\ncanonical")
    alias = tmp_path / "view" / "thermal" / "0001.JPG"
    mod._relative_symlink(source, alias)
    assert alias.is_symlink()
    assert alias.read_bytes() == source.read_bytes()
