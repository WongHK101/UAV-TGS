"""Build a verified fixed-topology Gaussian index-space receipt.

The receipt is only needed when a continuation changed Gaussian centres, so
ordered XYZ equality can no longer bind a rendered PLY to its pre-continuation
index anchor.  It proves that both endpoint PLYs match their checkpoint tensor
order and that the bound continuation protocol disabled every topology-changing
operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from plyfile import PlyData


PROTOCOL = "uav-tgs-gaussian-index-binding-v1"
EVIDENCE_SCHEMA = "uav-tgs-rgb-continuation-protocol-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "size_bytes": int(path.stat().st_size), "sha256": _sha256(path)}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _ordered_xyz_identity(array: np.ndarray) -> dict[str, Any]:
    xyz = np.asarray(array, dtype="<f4")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError(f"Expected finite Nx3 XYZ, got {xyz.shape}")
    digest = hashlib.sha256()
    digest.update(np.asarray(xyz.shape, dtype="<i8").tobytes(order="C"))
    digest.update(np.ascontiguousarray(xyz).tobytes(order="C"))
    return {
        "gaussian_count": int(xyz.shape[0]),
        "dtype": "float32_le",
        "sequence_sha256": digest.hexdigest(),
    }


def _ply_xyz(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")
    vertex = ply["vertex"]
    for field in ("x", "y", "z"):
        if field not in vertex.data.dtype.names:
            raise ValueError(f"PLY vertex element is missing {field!r}: {path}")
    xyz = np.column_stack([vertex[field] for field in ("x", "y", "z")]).astype(np.float32, copy=False)
    return xyz, _ordered_xyz_identity(xyz)


def _checkpoint_xyz(path: Path) -> tuple[np.ndarray, int, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError(f"Unexpected checkpoint container: {path}")
    captured, iteration = payload
    if not isinstance(captured, (tuple, list)) or len(captured) < 2 or not torch.is_tensor(captured[1]):
        raise ValueError(f"Checkpoint does not contain Gaussian capture XYZ: {path}")
    xyz = captured[1].detach().cpu().numpy().astype(np.float32, copy=False)
    return xyz, int(iteration), _ordered_xyz_identity(xyz)


def _producer_identity() -> dict[str, Any]:
    script = Path(__file__).resolve()
    repo = script.parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = ""
    return {"script": _identity(script), "git_commit": commit}


def _require_fixed_topology_protocol(payload: Mapping[str, Any]) -> tuple[int, int, str]:
    if str(payload.get("schema", "")) != EVIDENCE_SCHEMA:
        raise ValueError(f"Continuation evidence must use schema {EVIDENCE_SCHEMA!r}")
    if payload.get("topology_fixed") is not True:
        raise ValueError("Continuation evidence does not declare topology_fixed=true")
    for key in ("densification", "pruning", "opacity_reset"):
        if payload.get(key) is not False:
            raise ValueError(f"Continuation evidence must declare {key}=false")
    if str(payload.get("artifact_save_semantics", "")) != "aligned":
        raise ValueError("Continuation evidence must declare aligned endpoint save semantics")
    start_iteration = int(payload.get("anchor_iteration", -1))
    final_iteration = int(payload.get("final_iteration", -1))
    start_checkpoint_sha = str(payload.get("start_checkpoint_sha256", "")).strip().lower()
    if start_iteration <= 0 or final_iteration <= start_iteration or len(start_checkpoint_sha) != 64:
        raise ValueError("Continuation evidence has invalid endpoint iteration/checkpoint identity")
    return start_iteration, final_iteration, start_checkpoint_sha


def build_receipt(
    *,
    scene_name: str,
    anchor_ply_path: Path,
    rendered_ply_path: Path,
    start_checkpoint_path: Path,
    final_checkpoint_path: Path,
    evidence_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    paths = [
        anchor_ply_path.resolve(),
        rendered_ply_path.resolve(),
        start_checkpoint_path.resolve(),
        final_checkpoint_path.resolve(),
        evidence_manifest_path.resolve(),
    ]
    anchor_ply_path, rendered_ply_path, start_checkpoint_path, final_checkpoint_path, evidence_manifest_path = paths
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace Gaussian index binding receipt: {output_path}")
    evidence = _load_json(evidence_manifest_path)
    start_iteration, final_iteration, declared_start_sha = _require_fixed_topology_protocol(evidence)
    identities = {
        "anchor_ply": _identity(anchor_ply_path),
        "rendered_ply": _identity(rendered_ply_path),
        "start_checkpoint": _identity(start_checkpoint_path),
        "final_checkpoint": _identity(final_checkpoint_path),
        "evidence_manifest": _identity(evidence_manifest_path),
    }
    if identities["start_checkpoint"]["sha256"] != declared_start_sha:
        raise ValueError("Continuation evidence/start checkpoint SHA-256 mismatch")

    anchor_ply_xyz, anchor_ply_sequence = _ply_xyz(anchor_ply_path)
    rendered_ply_xyz, rendered_ply_sequence = _ply_xyz(rendered_ply_path)
    start_xyz, observed_start_iteration, start_sequence = _checkpoint_xyz(start_checkpoint_path)
    final_xyz, observed_final_iteration, final_sequence = _checkpoint_xyz(final_checkpoint_path)
    if observed_start_iteration != start_iteration or observed_final_iteration != final_iteration:
        raise ValueError("Continuation evidence/checkpoint iteration mismatch")
    if not np.array_equal(anchor_ply_xyz, start_xyz):
        raise ValueError("Anchor PLY ordered XYZ differs from the start checkpoint")
    if not np.array_equal(rendered_ply_xyz, final_xyz):
        raise ValueError("Rendered PLY ordered XYZ differs from the final checkpoint")
    counts = {
        anchor_ply_sequence["gaussian_count"],
        rendered_ply_sequence["gaussian_count"],
        start_sequence["gaussian_count"],
        final_sequence["gaussian_count"],
    }
    if len(counts) != 1:
        raise ValueError(f"Fixed-topology endpoint Gaussian counts differ: {sorted(counts)}")

    receipt = {
        "protocol": PROTOCOL,
        "status": "verified",
        "scene_name": str(scene_name),
        "rendered_model_point_cloud_sha256": identities["rendered_ply"]["sha256"],
        "gaussian_index_anchor_sha256": identities["anchor_ply"]["sha256"],
        "gaussian_count": counts.pop(),
        "topology_fixed": True,
        "index_order_preserved": True,
        "evidence_schema": EVIDENCE_SCHEMA,
        "evidence_manifest_identity": identities["evidence_manifest"],
        "anchor_checkpoint_identity": identities["start_checkpoint"],
        "final_checkpoint_identity": identities["final_checkpoint"],
        "anchor_ply_identity": identities["anchor_ply"],
        "rendered_ply_identity": identities["rendered_ply"],
        "anchor_iteration": start_iteration,
        "final_iteration": final_iteration,
        "ordered_xyz_proof": {
            "anchor_ply": anchor_ply_sequence,
            "anchor_checkpoint": start_sequence,
            "rendered_ply": rendered_ply_sequence,
            "final_checkpoint": final_sequence,
            "anchor_ply_equals_start_checkpoint": True,
            "rendered_ply_equals_final_checkpoint": True,
            "continuation_has_no_topology_mutation": True,
        },
        "producer_identity": _producer_identity(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--anchor_ply", required=True, type=Path)
    parser.add_argument("--rendered_ply", required=True, type=Path)
    parser.add_argument("--start_checkpoint", required=True, type=Path)
    parser.add_argument("--final_checkpoint", required=True, type=Path)
    parser.add_argument("--evidence_manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    receipt = build_receipt(
        scene_name=args.scene_name,
        anchor_ply_path=args.anchor_ply,
        rendered_ply_path=args.rendered_ply,
        start_checkpoint_path=args.start_checkpoint,
        final_checkpoint_path=args.final_checkpoint,
        evidence_manifest_path=args.evidence_manifest,
        output_path=args.output,
    )
    print(json.dumps({"output": str(args.output.resolve()), "gaussian_count": receipt["gaussian_count"]}))


if __name__ == "__main__":
    main()
