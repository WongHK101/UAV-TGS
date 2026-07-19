"""Bind an immutable Gaussian probe manifest to the formal geometry contract.

The probe exporter predates the formal eight-threshold evaluator in a number of
locked runs.  This tool deliberately does not rewrite the exporter output or
touch any rendered arrays.  It validates the source bundle semantics, copies
the source JSON object, and adds only a fixed formal metric contract plus a
receipt that identifies the immutable source manifest.

The derived manifest must live beside the source manifest so that relative
``npz_file`` entries retain exactly the same meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.geometric_repeatability.evaluate_depth_definitions import (
    DEFAULT_THRESHOLDS_M,
    DIAGNOSTIC_DEPTH_SEMANTICS,
    FORMAL_DEPTH_MIN_M,
    FORMAL_DEPTH_SEMANTICS,
    FORMAL_GEOMETRY_PROTOCOL,
    FORMAL_GEOMETRY_PROTOCOL_SHA256,
    FORMAL_OPACITY_THRESHOLD,
    FORMAL_SUPPORT_POLICY_SHA256,
)


BINDING_PROTOCOL = "uav-tgs-formal-geometry-manifest-binding-v1"
SOURCE_DEPTH_SEMANTICS = "inverse_camera_z_from_renderer"
SOURCE_OPACITY_SEMANTICS = "exact_sum_of_accepted_alpha_times_transmittance"
GAUSSIAN_SUPPORT_MODE = "gaussian_accumulated_opacity"
ALLOWED_INDEX_PROOFS = frozenset(
    {
        "identical_ply_sha256",
        "exact_ordered_xyz_sequence",
        "fixed_topology_invariant_audit_receipt",
    }
)
EVALUATOR_PATH = (
    REPO_ROOT
    / "tools"
    / "geometric_repeatability"
    / "evaluate_depth_definitions.py"
)


def _is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _git_output(*args: str, strip: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip() if strip else result.stdout


def _binding_producer_identity() -> Dict[str, Any]:
    binder = _file_identity(Path(__file__))
    evaluator = _file_identity(EVALUATOR_PATH)
    try:
        commit = _git_output("rev-parse", "HEAD").lower()
        status = _git_output(
            "status", "--porcelain=v1", "--untracked-files=all", strip=False
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot record binder repository provenance") from exc
    if not (len(commit) == 40 and all(character in "0123456789abcdef" for character in commit)):
        raise RuntimeError(f"Binder repository returned an invalid commit: {commit!r}")
    status_bytes = status.encode("utf-8")
    return {
        "binder_script_identity": binder,
        "evaluator_script_identity": evaluator,
        "repo_root": str(REPO_ROOT),
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "git_status_sha256": _sha256_bytes(status_bytes),
    }


def _load_source_snapshot(path: Path) -> tuple[Dict[str, Any], Dict[str, Any], bytes]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    raw = resolved.read_bytes()
    identity = {
        "path": str(resolved),
        "size_bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Source manifest is not valid UTF-8 JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Source manifest must contain a JSON object")
    return payload, identity, raw


def _require_identity(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a file identity object")
    if not _is_sha256(value.get("sha256", "")):
        raise ValueError(f"{label}.sha256 must be a lowercase-compatible SHA-256")
    try:
        size = int(value.get("size_bytes", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.size_bytes must be a non-negative integer") from exc
    if size < 0:
        raise ValueError(f"{label}.size_bytes must be a non-negative integer")
    return value


def _validate_source_producer_identity(source: Mapping[str, Any]) -> Dict[str, Any]:
    producer = source.get("producer_identity")
    if not isinstance(producer, Mapping):
        raise ValueError("Source manifest is missing producer_identity")
    script_path = str(producer.get("script_path", "")).strip()
    if not script_path:
        raise ValueError("source producer_identity.script_path is missing")
    if not _is_sha256(producer.get("script_sha256", "")):
        raise ValueError("source producer_identity.script_sha256 is invalid")
    commit = str(producer.get("git_commit", "")).strip().lower()
    if not (len(commit) == 40 and all(character in "0123456789abcdef" for character in commit)):
        raise ValueError("source producer_identity.git_commit is invalid")
    if not isinstance(producer.get("git_dirty"), bool):
        raise ValueError("source producer_identity.git_dirty must be boolean")
    if not _is_sha256(producer.get("git_status_sha256", "")):
        raise ValueError("source producer_identity.git_status_sha256 is invalid")
    return dict(producer)


def _require_positive_count(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be a positive integer")
    if count <= 0 or isinstance(value, float) and float(count) != value:
        raise ValueError(f"{label} must be a positive integer")
    return count


def formal_geometry_metric_contract() -> Dict[str, Any]:
    """Return the exact contract accepted by the locked formal evaluator."""

    return {
        "protocol": FORMAL_GEOMETRY_PROTOCOL,
        "protocol_sha256": FORMAL_GEOMETRY_PROTOCOL_SHA256,
        "support_policy_sha256": FORMAL_SUPPORT_POLICY_SHA256,
        "thresholds_m": list(DEFAULT_THRESHOLDS_M),
        "depth_definitions": dict(FORMAL_DEPTH_SEMANTICS),
        "opacity_threshold": FORMAL_OPACITY_THRESHOLD,
        "depth_min_m": FORMAL_DEPTH_MIN_M,
        "support_mode": GAUSSIAN_SUPPORT_MODE,
    }


def _validate_existing_contract(source: Mapping[str, Any]) -> None:
    observed = source.get("formal_geometry_metric_contract")
    if observed is None:
        return
    expected = formal_geometry_metric_contract()
    if observed != expected:
        raise ValueError(
            "Source manifest already contains a different formal_geometry_metric_contract"
        )


def _validate_diagnostic_semantics(source: Mapping[str, Any]) -> None:
    if source.get("depth_semantics") != SOURCE_DEPTH_SEMANTICS:
        raise ValueError(
            "Source depth_semantics mismatch: expected "
            f"{SOURCE_DEPTH_SEMANTICS!r}"
        )
    if source.get("opacity_semantics") != SOURCE_OPACITY_SEMANTICS:
        raise ValueError(
            "Source opacity_semantics mismatch: expected "
            f"{SOURCE_OPACITY_SEMANTICS!r}"
        )
    diagnostics = source.get("depth_diagnostics")
    if not isinstance(diagnostics, Mapping) or diagnostics.get("enabled") is not True:
        raise ValueError("Source manifest must declare enabled depth_diagnostics")
    for key, expected in DIAGNOSTIC_DEPTH_SEMANTICS.items():
        if diagnostics.get(key) != expected:
            raise ValueError(
                f"Source depth_diagnostics semantics mismatch for {key}: "
                f"observed={diagnostics.get(key)!r} expected={expected!r}"
            )


def _validate_gaussian_support(source: Mapping[str, Any]) -> None:
    count = _require_positive_count(source.get("gaussian_count"), label="gaussian_count")
    anchor = _require_identity(source.get("gaussian_index_anchor"), label="gaussian_index_anchor")
    model_ply = _require_identity(source.get("model_point_cloud"), label="model_point_cloud")

    binding = source.get("gaussian_index_binding")
    if not isinstance(binding, Mapping) or binding.get("status") != "verified":
        raise ValueError("Source manifest must include a verified gaussian_index_binding")
    if _require_positive_count(
        binding.get("gaussian_count"), label="gaussian_index_binding.gaussian_count"
    ) != count:
        raise ValueError("Gaussian index binding count mismatch")

    bound_anchor = _require_identity(
        binding.get("gaussian_index_anchor"),
        label="gaussian_index_binding.gaussian_index_anchor",
    )
    rendered_ply = _require_identity(
        binding.get("rendered_model_point_cloud"),
        label="gaussian_index_binding.rendered_model_point_cloud",
    )
    if str(bound_anchor["sha256"]).lower() != str(anchor["sha256"]).lower():
        raise ValueError("Gaussian index binding anchor SHA-256 mismatch")
    if str(rendered_ply["sha256"]).lower() != str(model_ply["sha256"]).lower():
        raise ValueError("Gaussian index binding rendered PLY SHA-256 mismatch")

    proof = str(binding.get("proof", ""))
    if proof not in ALLOWED_INDEX_PROOFS:
        raise ValueError(f"Unsupported Gaussian index-space proof: {proof!r}")
    if proof == "exact_ordered_xyz_sequence":
        rendered_xyz = binding.get("rendered_ordered_xyz")
        anchor_xyz = binding.get("anchor_ordered_xyz")
        if not isinstance(rendered_xyz, Mapping) or not isinstance(anchor_xyz, Mapping):
            raise ValueError("Ordered-XYZ Gaussian index proof is missing sequence identities")
        rendered_sha = str(rendered_xyz.get("sequence_sha256", "")).lower()
        anchor_sha = str(anchor_xyz.get("sequence_sha256", "")).lower()
        if not _is_sha256(rendered_sha) or rendered_sha != anchor_sha:
            raise ValueError("Ordered-XYZ Gaussian index proof is inconsistent")
    elif proof == "fixed_topology_invariant_audit_receipt":
        _require_identity(
            binding.get("binding_receipt_identity"),
            label="gaussian_index_binding.binding_receipt_identity",
        )


def _validate_formal_split_binding(
    source: Mapping[str, Any], formal_split_identity: Mapping[str, Any]
) -> None:
    declared = _require_identity(
        source.get("formal_split_manifest_identity"),
        label="formal_split_manifest_identity",
    )
    if str(declared["sha256"]).lower() != str(formal_split_identity["sha256"]).lower():
        raise ValueError("Source/formal split manifest SHA-256 mismatch")
    if int(declared["size_bytes"]) != int(formal_split_identity["size_bytes"]):
        raise ValueError("Source/formal split manifest size mismatch")


def _validate_probe_binding(
    source: Mapping[str, Any], probe_identity: Mapping[str, Any]
) -> None:
    declared = _require_identity(
        source.get("probe_camera_manifest_identity"),
        label="probe_camera_manifest_identity",
    )
    if str(declared["sha256"]).lower() != str(probe_identity["sha256"]).lower():
        raise ValueError("Source/probe camera manifest SHA-256 mismatch")
    if int(declared["size_bytes"]) != int(probe_identity["size_bytes"]):
        raise ValueError("Source/probe camera manifest size mismatch")


def _validate_views(source: Mapping[str, Any]) -> None:
    views = source.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("Source manifest must contain a non-empty views list")
    names: set[str] = set()
    for index, view in enumerate(views):
        if not isinstance(view, Mapping):
            raise ValueError(f"views[{index}] must be an object")
        name = str(view.get("image_name", "")).strip().lower()
        if not name or name in names:
            raise ValueError(f"views[{index}] has a missing or duplicate image_name")
        names.add(name)
        npz_file = str(view.get("npz_file", "")).strip()
        if not npz_file:
            raise ValueError(f"views[{index}] is missing npz_file")
        if not _is_sha256(view.get("npz_sha256", "")):
            raise ValueError(f"views[{index}].npz_sha256 is invalid")
        try:
            npz_size = int(view.get("npz_size_bytes", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"views[{index}].npz_size_bytes is invalid") from exc
        if npz_size <= 0:
            raise ValueError(f"views[{index}].npz_size_bytes must be positive")


def _validate_source(
    source: Mapping[str, Any],
    formal_split_identity: Mapping[str, Any],
    probe_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    _validate_existing_contract(source)
    _validate_diagnostic_semantics(source)
    _validate_gaussian_support(source)
    _validate_formal_split_binding(source, formal_split_identity)
    _validate_probe_binding(source, probe_identity)
    _validate_views(source)
    return _validate_source_producer_identity(source)


def _serialized_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> str:
    serialized = _serialized_json(payload)
    if path.exists():
        if not path.is_file() or path.read_bytes() != serialized:
            raise FileExistsError(
                f"Output manifest already exists with different bytes: {path}"
            )
        return "idempotent_existing"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != serialized:
                raise FileExistsError(
                    f"Output manifest was concurrently created with different bytes: {path}"
                )
            return "idempotent_concurrent"
        return "created"
    except BaseException:
        raise
    finally:
        temporary.unlink(missing_ok=True)


def bind_formal_geometry_manifest(
    *,
    source_manifest_path: Path,
    probe_camera_manifest_path: Path,
    formal_split_manifest_path: Path,
    output_manifest_path: Path,
    expected_source_sha256: str | None = None,
    _precommit_hook: Callable[[], None] | None = None,
) -> Dict[str, Any]:
    """Create a derived formal manifest while leaving ``source`` untouched.

    ``_precommit_hook`` exists solely to make the source-mutation guard
    deterministic in the directed test suite.
    """

    source_path = source_manifest_path.resolve()
    probe_path = probe_camera_manifest_path.resolve()
    split_path = formal_split_manifest_path.resolve()
    output_path = output_manifest_path.resolve()
    if output_path == source_path:
        raise ValueError("Output manifest must not overwrite the source manifest")
    if output_path.parent != source_path.parent:
        raise ValueError(
            "Derived manifest must be written beside the source manifest so relative NPZ paths remain unchanged"
        )
    source, source_identity, source_bytes = _load_source_snapshot(source_path)
    if expected_source_sha256 is not None:
        expected_sha = str(expected_source_sha256).strip().lower()
        if not _is_sha256(expected_sha):
            raise ValueError("expected_source_sha256 must be a valid SHA-256")
        if source_identity["sha256"] != expected_sha:
            raise ValueError(
                "Source manifest SHA-256 does not match the operator-pinned expected hash"
            )

    formal_split_identity = _file_identity(split_path)
    probe_identity = _file_identity(probe_path)
    source_producer_identity = _validate_source(
        source, formal_split_identity, probe_identity
    )
    binding_producer_identity = _binding_producer_identity()

    derived = dict(source)
    derived["formal_geometry_metric_contract"] = formal_geometry_metric_contract()
    derived["formal_geometry_metric_contract_binding"] = {
        "protocol": BINDING_PROTOCOL,
        "derivation": "read_only_source_copy_plus_fixed_formal_contract",
        "source_manifest_identity": dict(source_identity),
        "source_manifest_payload_sha256": _sha256_bytes(source_bytes),
        "source_producer_identity": source_producer_identity,
        "probe_camera_manifest_identity": dict(probe_identity),
        "formal_split_manifest_identity": dict(formal_split_identity),
        "binding_producer_identity": binding_producer_identity,
        "relative_view_paths_preserved": True,
        "source_manifest_modified": False,
    }

    if _precommit_hook is not None:
        _precommit_hook()
    current_identity = _file_identity(source_path)
    if (
        current_identity["sha256"] != source_identity["sha256"]
        or current_identity["size_bytes"] != source_identity["size_bytes"]
    ):
        raise RuntimeError("Source manifest changed while formal contract binding was in progress")

    _atomic_create_json(output_path, derived)
    final_source_identity = _file_identity(source_path)
    if final_source_identity != source_identity:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Source manifest changed during formal contract commit")
    return derived


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind a read-only Gaussian probe manifest to the locked formal geometry metric contract"
    )
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--probe-camera-manifest", required=True, type=Path)
    parser.add_argument("--formal-split-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument(
        "--expected-source-sha256",
        default=None,
        help="Optional operator-pinned source manifest SHA-256",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    payload = bind_formal_geometry_manifest(
        source_manifest_path=args.source_manifest,
        probe_camera_manifest_path=args.probe_camera_manifest,
        formal_split_manifest_path=args.formal_split_manifest,
        output_manifest_path=args.output_manifest,
        expected_source_sha256=args.expected_source_sha256,
    )
    binding = payload["formal_geometry_metric_contract_binding"]
    print(
        json.dumps(
            {
                "status": "passed",
                "output_manifest": str(args.output_manifest.resolve()),
                "source_manifest_sha256": binding["source_manifest_identity"]["sha256"],
                "formal_split_manifest_sha256": binding["formal_split_manifest_identity"]["sha256"],
                "formal_geometry_protocol_sha256": payload["formal_geometry_metric_contract"][
                    "protocol_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
