"""Build the pre-generated camera sequence shared by R35 and O35.

The input is an ordered UTF-8 JSON list of image names, a JSON object with an
``ordered_camera_names`` list, or a plain text file containing one image name
per line.  This utility deliberately does not instantiate ``Scene`` and never
touches the global training RNG.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.camera_sequence import build_sequence_manifest, save_sequence_manifest


def _load_names(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        names = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        if isinstance(payload, dict):
            payload = payload.get("ordered_camera_names")
        if not isinstance(payload, list):
            raise ValueError(
                "Camera-name JSON must be a list or contain ordered_camera_names"
            )
        names = payload
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("Every ordered camera name must be a non-empty string")
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-names", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", type=int, default=5000, choices=[5000])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--anchor-sha256", required=True)
    args = parser.parse_args()

    names = _load_names(args.camera_names)
    manifest = build_sequence_manifest(
        names,
        steps=args.steps,
        seed=args.seed,
        metadata={
            "rule": "private-rng-random-pop-without-replacement-cycles",
            "scene": args.scene,
            "split_sha256": args.split_sha256,
            "anchor_sha256": args.anchor_sha256,
            "camera_names_source": str(args.camera_names.resolve()),
        },
    )
    save_sequence_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "steps": manifest["steps"],
                "camera_count": len(manifest["ordered_camera_names"]),
                "ordered_camera_sha256": manifest["ordered_camera_sha256"],
                "sequence_sha256": manifest["sequence_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
