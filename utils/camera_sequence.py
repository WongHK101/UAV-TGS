"""Deterministic, pre-generated camera sequences for paired continuations.

The training loop only *loads* these manifests.  Sequence generation uses a
private ``random.Random`` instance, so preparing or loading a sequence never
consumes the process-global Python/PyTorch training RNG.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "uav-tgs-fixed-camera-sequence-v1"


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def ordered_camera_names(cameras: Iterable[object]) -> list[str]:
    names = []
    for camera in cameras:
        name = getattr(camera, "image_name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Every training camera must have a non-empty image_name")
        names.append(name)
    if not names:
        raise ValueError("Cannot build a camera sequence for an empty camera set")
    if len(names) != len(set(names)):
        raise ValueError("Training camera image_name values must be unique")
    return names


def ordered_camera_hash(camera_names: Sequence[str]) -> str:
    return sha256_json(list(camera_names))


def camera_parameters_payload(cameras: Sequence[object]) -> list[dict]:
    """Return the canonical ordered camera payload used by the OGS audit.

    The explicit float64 conversion is part of the hash contract.  Keep this
    representation shared between the no-grad audit and continuation trainer.
    """

    rows = []
    for camera in cameras:
        rows.append(
            {
                "image_name": str(camera.image_name),
                "uid": int(camera.uid),
                "colmap_id": int(camera.colmap_id),
                "R": np.asarray(camera.R, dtype=np.float64).tolist(),
                "T": np.asarray(camera.T, dtype=np.float64).tolist(),
                "FoVx": float(camera.FoVx),
                "FoVy": float(camera.FoVy),
                "image_width": int(camera.image_width),
                "image_height": int(camera.image_height),
            }
        )
    return rows


def camera_parameters_hash(cameras: Sequence[object]) -> str:
    """Hash ordered camera identities, extrinsics, intrinsics, and dimensions."""

    return sha256_json(camera_parameters_payload(cameras))


def sequence_hash(sequence: Sequence[str]) -> str:
    return sha256_json(list(sequence))


def build_sequence_manifest(
    camera_names: Sequence[str],
    *,
    steps: int = 5000,
    seed: int = 0,
    metadata: Mapping | None = None,
) -> dict:
    """Build a balanced deterministic sequence without touching global RNG."""

    camera_names = list(camera_names)
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if not camera_names:
        raise ValueError("camera_names must not be empty")
    if len(camera_names) != len(set(camera_names)):
        raise ValueError("camera_names must be unique")

    generator = random.Random(int(seed))
    sequence: list[str] = []
    while len(sequence) < int(steps):
        # Mirror train.py's historical random-pop-without-replacement rule,
        # but use a private RNG so sequence preparation cannot perturb a run.
        cycle = list(camera_names)
        while cycle and len(sequence) < int(steps):
            sequence.append(cycle.pop(generator.randint(0, len(cycle) - 1)))

    payload = {
        "schema": SCHEMA,
        "steps": int(steps),
        "seed": int(seed),
        "ordered_camera_names": camera_names,
        "ordered_camera_sha256": ordered_camera_hash(camera_names),
        "sequence": sequence,
        "sequence_sha256": sequence_hash(sequence),
        "metadata": dict(metadata or {}),
    }
    payload["manifest_sha256"] = sha256_json(payload)
    return payload


def validate_sequence_manifest(
    payload: Mapping,
    *,
    camera_names: Sequence[str],
    expected_steps: int = 5000,
) -> dict:
    payload = dict(payload)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported camera sequence schema: {payload.get('schema')!r}")
    if int(payload.get("steps", -1)) != int(expected_steps):
        raise ValueError(
            f"Camera sequence must contain exactly {expected_steps} steps; "
            f"manifest declares {payload.get('steps')!r}"
        )

    expected_names = list(camera_names)
    stored_names = payload.get("ordered_camera_names")
    if stored_names != expected_names:
        raise ValueError("Ordered training-camera list does not match sequence manifest")
    actual_camera_hash = ordered_camera_hash(expected_names)
    if payload.get("ordered_camera_sha256") != actual_camera_hash:
        raise ValueError("Ordered camera hash mismatch")

    sequence = payload.get("sequence")
    if not isinstance(sequence, list) or len(sequence) != int(expected_steps):
        raise ValueError(
            f"Camera sequence must contain exactly {expected_steps} entries"
        )
    camera_set = set(expected_names)
    unknown = sorted(set(sequence) - camera_set)
    if unknown:
        raise ValueError(f"Camera sequence references unknown cameras: {unknown[:5]}")
    actual_sequence_hash = sequence_hash(sequence)
    if payload.get("sequence_sha256") != actual_sequence_hash:
        raise ValueError("Camera sequence SHA-256 mismatch")

    try:
        stored_seed = int(payload["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Camera sequence manifest must declare an integer seed") from exc
    expected_sequence = build_sequence_manifest(
        expected_names,
        steps=int(expected_steps),
        seed=stored_seed,
    )["sequence"]
    if sequence != expected_sequence:
        raise ValueError(
            "Camera sequence does not match the deterministic sequence generated "
            "from ordered_camera_names, steps, and seed"
        )

    expected_manifest_hash = payload.pop("manifest_sha256", None)
    actual_manifest_hash = sha256_json(payload)
    payload["manifest_sha256"] = expected_manifest_hash
    if expected_manifest_hash != actual_manifest_hash:
        raise ValueError("Camera sequence manifest SHA-256 mismatch")
    return payload


def save_sequence_manifest(path: str | os.PathLike, payload: Mapping) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    return path


def load_sequence_manifest(
    path: str | os.PathLike,
    *,
    camera_names: Sequence[str],
    expected_steps: int = 5000,
) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_sequence_manifest(
        payload,
        camera_names=camera_names,
        expected_steps=expected_steps,
    )


def camera_lookup(cameras: Sequence[object]) -> dict[str, object]:
    names = ordered_camera_names(cameras)
    return dict(zip(names, cameras))
