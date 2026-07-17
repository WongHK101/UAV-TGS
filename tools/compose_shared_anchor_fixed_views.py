#!/usr/bin/env python3
"""Compose preregistered fixed-view RGB comparisons for shared-anchor clamp."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw

try:
    from tools.analyze_formal_test_blocks import _load_blocks, _read_test_list
except ModuleNotFoundError:
    from analyze_formal_test_blocks import _load_blocks, _read_test_list


class FixedViewError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FixedViewError(f"render is missing: {path}")
    try:
        return Image.open(path).convert("RGB")
    except OSError as exc:
        raise FixedViewError(f"cannot read render: {path}") from exc


def _label_panel(image: Image.Image, label: str) -> Image.Image:
    header = 28
    output = Image.new("RGB", (image.width, image.height + header), "white")
    output.paste(image, (0, header))
    ImageDraw.Draw(output).text((8, 7), label, fill="black")
    return output


def compose(args: argparse.Namespace) -> dict[str, Any]:
    test_list = Path(args.test_list).resolve()
    bound_split = Path(args.bound_split).resolve()
    anchor_dir = Path(args.anchor_render_dir).resolve()
    shared_dir = Path(args.shared_render_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FixedViewError(f"refusing to overwrite output: {output_dir}")
    for path in (test_list, bound_split):
        if not path.is_file():
            raise FixedViewError(f"input is missing: {path}")
    for path in (anchor_dir, shared_dir):
        if not path.is_dir():
            raise FixedViewError(f"render directory is missing: {path}")

    test_names = _read_test_list(test_list)
    blocks, selected_blocks_hash, _ = _load_blocks(bound_split, test_names)
    partial = output_dir.with_name(f".{output_dir.name}.partial-{uuid.uuid4().hex}")
    partial.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    try:
        for ordinal, block in enumerate(blocks):
            names = list(block["views"])
            if len(names) != 16:
                raise FixedViewError("fixed-view protocol requires complete 16-view blocks")
            # This choice is fixed before looking at any metric or render:
            # select the lower of the two central positions in each 16-view block.
            name = names[7]
            anchor_path = anchor_dir / name
            shared_path = shared_dir / name
            anchor = _open_rgb(anchor_path)
            shared = _open_rgb(shared_path)
            if anchor.size != shared.size:
                raise FixedViewError(f"render size mismatch for {name}")
            difference = ImageChops.difference(anchor, shared)
            difference = Image.fromarray(
                np.clip(np.asarray(difference, dtype=np.uint16) * 4, 0, 255).astype(
                    np.uint8
                ),
                mode="RGB",
            )
            panels = [
                _label_panel(anchor, "Raw RGB anchor"),
                _label_panel(shared, "Shared-clamped S anchor"),
                _label_panel(difference, "4x absolute RGB difference"),
            ]
            composite = Image.new(
                "RGB",
                (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
                "white",
            )
            offset = 0
            for panel in panels:
                composite.paste(panel, (offset, 0))
                offset += panel.width
            output_name = f"block_{ordinal:02d}_offset07_{Path(name).stem}.png"
            output_path = partial / output_name
            composite.save(output_path, format="PNG")
            records.append(
                {
                    "block_ordinal": ordinal,
                    "strip_id": block["strip_id"],
                    "block_index": block["block_index"],
                    "stratum": block["stratum"],
                    "fixed_block_offset": 7,
                    "image_name": name,
                    "anchor_render": {
                        "path": str(anchor_path),
                        "sha256": _sha256(anchor_path),
                    },
                    "shared_render": {
                        "path": str(shared_path),
                        "sha256": _sha256(shared_path),
                    },
                    "composite_file": output_name,
                    "composite_sha256": _sha256(output_path),
                }
            )

        manifest = {
            "schema": "uav-tgs-shared-anchor-fixed-view-comparison-v1",
            "status": "complete",
            "scene": str(args.scene),
            "selection": {
                "rule": (
                    "block_offset=7 (lower central view) from every fixed "
                    "16-view test block; selected before inspecting metrics/renders"
                ),
                "selected_test_blocks_hash": selected_blocks_hash,
                "block_count": len(blocks),
                "views_per_block": 16,
            },
            "inputs": {
                "test_list": {"path": str(test_list), "sha256": _sha256(test_list)},
                "bound_split": {
                    "path": str(bound_split),
                    "sha256": _sha256(bound_split),
                },
                "anchor_render_dir": str(anchor_dir),
                "shared_render_dir": str(shared_dir),
            },
            "records": records,
        }
        manifest_path = partial / "fixed_view_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (partial / "FIXED_VIEW_STATUS").write_text("complete\n", encoding="ascii")
        shutil.move(str(partial), str(output_dir))
        return manifest
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--bound-split", required=True)
    parser.add_argument("--anchor-render-dir", required=True)
    parser.add_argument("--shared-render-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        payload = compose(args)
    except (FixedViewError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
