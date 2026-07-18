#!/usr/bin/env python3
"""Compose preregistered fixed-view panels for multiple scale anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from PIL import Image, ImageDraw

try:
    from tools.analyze_formal_test_blocks import _load_blocks, _read_test_list
except ModuleNotFoundError:
    from analyze_formal_test_blocks import _load_blocks, _read_test_list


class MultiPanelError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_groups(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    seen = set()
    for value in values:
        label, separator, root = value.partition("=")
        if not separator or not label or not root or label in seen:
            raise MultiPanelError("--render must use unique LABEL=DIR entries")
        path = Path(root).resolve()
        if not path.is_dir():
            raise MultiPanelError(f"render directory is missing: {path}")
        result.append((label, path))
        seen.add(label)
    if len(result) < 2:
        raise MultiPanelError("at least two render groups are required")
    return result


def labeled(image: Image.Image, label: str) -> Image.Image:
    header = 28
    output = Image.new("RGB", (image.width, image.height + header), "white")
    output.paste(image, (0, header))
    ImageDraw.Draw(output).text((8, 7), label, fill="black")
    return output


def compose(args: argparse.Namespace) -> dict:
    test_list = Path(args.test_list).resolve()
    bound_split = Path(args.bound_split).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise MultiPanelError(f"refusing to overwrite output: {output_dir}")
    groups = parse_groups(args.render)
    names = _read_test_list(test_list)
    blocks, selected_hash, _ = _load_blocks(bound_split, names)
    partial = output_dir.with_name(f".{output_dir.name}.partial-{uuid.uuid4().hex}")
    partial.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for ordinal, block in enumerate(blocks):
            views = list(block["views"])
            if len(views) != 16:
                raise MultiPanelError("fixed-view protocol requires 16-view blocks")
            name = views[7]
            panels = []
            inputs = {}
            reference_size = None
            for label, root in groups:
                path = root / name
                if not path.is_file():
                    raise MultiPanelError(f"render is missing: {path}")
                image = Image.open(path).convert("RGB")
                reference_size = image.size if reference_size is None else reference_size
                if image.size != reference_size:
                    raise MultiPanelError(f"render size mismatch: {path}")
                panels.append(labeled(image, label))
                inputs[label] = {"path": str(path), "sha256": sha256(path)}
            columns = 2
            rows = (len(panels) + columns - 1) // columns
            width = columns * panels[0].width
            height = rows * panels[0].height
            composite = Image.new("RGB", (width, height), "white")
            for index, panel in enumerate(panels):
                composite.paste(
                    panel,
                    ((index % columns) * panel.width, (index // columns) * panel.height),
                )
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
                    "inputs": inputs,
                    "composite_file": output_name,
                    "composite_sha256": sha256(output_path),
                }
            )
        manifest = {
            "schema": "uav-tgs-multi-anchor-fixed-view-comparison-v1",
            "status": "complete",
            "scene": str(args.scene),
            "groups": [label for label, _ in groups],
            "selection": {
                "rule": "block_offset=7 from every fixed 16-view test block, fixed before metrics",
                "selected_test_blocks_hash": selected_hash,
                "block_count": len(blocks),
            },
            "test_list": {"path": str(test_list), "sha256": sha256(test_list)},
            "bound_split": {"path": str(bound_split), "sha256": sha256(bound_split)},
            "records": records,
        }
        (partial / "fixed_view_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
    parser.add_argument("--render", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        payload = compose(args)
    except (MultiPanelError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
