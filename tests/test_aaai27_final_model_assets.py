from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest

from tools.aaai27_final.collect_deployable_model_assets import _row, merge


def test_same_asset_is_counted_once(tmp_path: Path) -> None:
    asset = tmp_path / "endpoint.ply"
    asset.write_bytes(b"12345")
    row = _row(
        "Building",
        "raw_f3",
        "test",
        (("rgb", asset), ("thermal_same_file", asset)),
    )
    assert row["asset_count"] == 1
    assert row["model_size_bytes"] == 5
    assert row["assets"][0]["role"] == "rgb+thermal_same_file"


def test_missing_deployable_asset_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _row("Building", "raw_f3", "test", (("endpoint", tmp_path / "missing.ply"),))


def test_merge_rejects_duplicate_scene_method(tmp_path: Path) -> None:
    inputs = []
    for index in range(2):
        path = tmp_path / f"host{index}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "uav-tgs-aaai27-deployable-model-assets-v1",
                    "rows": [
                        {
                            "scene": "Building",
                            "method": "raw_f3",
                            "host": str(index),
                            "assets": [],
                            "asset_count": 0,
                            "model_size_bytes": 0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        inputs.append(path)
    with pytest.raises(ValueError, match="duplicate scene/method"):
        merge(Namespace(input=inputs, output=tmp_path / "merged.json"))
