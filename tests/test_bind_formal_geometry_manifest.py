from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.geometric_repeatability.bind_formal_geometry_manifest import (
    BINDING_PROTOCOL,
    SOURCE_DEPTH_SEMANTICS,
    SOURCE_OPACITY_SEMANTICS,
    bind_formal_geometry_manifest,
    formal_geometry_metric_contract,
)
from tools.geometric_repeatability.evaluate_depth_definitions import (
    DIAGNOSTIC_DEPTH_SEMANTICS,
    _validate_formal_geometry_contract,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return path


class BindFormalGeometryManifestTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        formal = _write_json(
            root / "formal_split.json",
            {
                "scene": "InternalRoad",
                "records": [
                    {"image_name": "0001.jpg", "split": "train"},
                    {"image_name": "0002.jpg", "split": "guard"},
                    {"image_name": "0003.jpg", "split": "test"},
                ],
            },
        )
        formal_identity = {
            "path": str(formal.resolve()),
            "size_bytes": formal.stat().st_size,
            "sha256": _sha256(formal),
        }
        probe = _write_json(
            root / "probe_camera_manifest.json",
            {"camera_manifest_type": "formal_all_split_probe_camera_manifest_v1"},
        )
        probe_identity = {
            "path": str(probe.resolve()),
            "size_bytes": probe.stat().st_size,
            "sha256": _sha256(probe),
        }
        anchor_sha = "a" * 64
        ply_sha = "b" * 64
        source = _write_json(
            root / "bundle" / "split_manifest.json",
            {
                "bundle_type": "gaussian_probe_split_bundle_v1",
                "producer_identity": {
                    "script_path": "/immutable/export_gaussian_probe_bundle.py",
                    "script_sha256": "d" * 64,
                    "repo_root": "/immutable/repo",
                    "git_commit": "e" * 40,
                    "git_dirty": False,
                    "git_status_sha256": hashlib.sha256(b"").hexdigest(),
                    "git_error": "",
                },
                "scene_name": "InternalRoad",
                "gaussian_count": 12,
                "gaussian_index_anchor": {
                    "path": "/immutable/anchor.ply",
                    "size_bytes": 100,
                    "sha256": anchor_sha,
                },
                "model_point_cloud": {
                    "path": "/immutable/model.ply",
                    "size_bytes": 101,
                    "sha256": ply_sha,
                },
                "gaussian_index_binding": {
                    "status": "verified",
                    "proof": "identical_ply_sha256",
                    "gaussian_count": 12,
                    "gaussian_index_anchor": {
                        "path": "/immutable/anchor.ply",
                        "size_bytes": 100,
                        "sha256": anchor_sha,
                    },
                    "rendered_model_point_cloud": {
                        "path": "/immutable/model.ply",
                        "size_bytes": 101,
                        "sha256": ply_sha,
                    },
                },
                "appearance_modality": "rgb",
                "depth_semantics": SOURCE_DEPTH_SEMANTICS,
                "opacity_semantics": SOURCE_OPACITY_SEMANTICS,
                "depth_diagnostics": {
                    "enabled": True,
                    **DIAGNOSTIC_DEPTH_SEMANTICS,
                },
                "formal_split_manifest_identity": formal_identity,
                "probe_camera_manifest_identity": probe_identity,
                "views": [
                    {
                        "image_name": "0001.jpg",
                        "npz_file": "views/0001.npz",
                        "npz_size_bytes": 123,
                        "npz_sha256": "c" * 64,
                    }
                ],
            },
        )
        return {
            "formal": formal,
            "probe": probe,
            "source": source,
            "output": source.parent / "formal_geometry_split_manifest.json",
        }

    def test_valid_source_is_copied_and_bound_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            source_before = fixture["source"].read_bytes()
            source_sha = _sha256(fixture["source"])

            result = bind_formal_geometry_manifest(
                source_manifest_path=fixture["source"],
                probe_camera_manifest_path=fixture["probe"],
                formal_split_manifest_path=fixture["formal"],
                output_manifest_path=fixture["output"],
                expected_source_sha256=source_sha,
            )

            self.assertEqual(fixture["source"].read_bytes(), source_before)
            self.assertTrue(fixture["output"].is_file())
            self.assertEqual(
                result["formal_geometry_metric_contract"],
                formal_geometry_metric_contract(),
            )
            _validate_formal_geometry_contract(result, metric_only=False)
            binding = result["formal_geometry_metric_contract_binding"]
            self.assertEqual(binding["protocol"], BINDING_PROTOCOL)
            self.assertEqual(binding["source_manifest_identity"]["sha256"], source_sha)
            self.assertEqual(
                binding["source_manifest_payload_sha256"],
                source_sha,
            )
            self.assertEqual(
                binding["probe_camera_manifest_identity"]["sha256"],
                _sha256(fixture["probe"]),
            )
            producer = binding["binding_producer_identity"]
            self.assertEqual(len(producer["git_commit"]), 40)
            self.assertIsInstance(producer["git_dirty"], bool)
            self.assertEqual(
                producer["binder_script_identity"]["sha256"],
                _sha256(
                    Path(__file__).resolve().parents[1]
                    / "tools"
                    / "geometric_repeatability"
                    / "bind_formal_geometry_manifest.py"
                ),
            )
            source_payload = json.loads(source_before.decode("utf-8"))
            for key, value in source_payload.items():
                self.assertEqual(result[key], value)

    def test_missing_required_fields_fail_closed(self) -> None:
        mutations = (
            ("depth_diagnostics", lambda payload: payload.pop("depth_diagnostics")),
            ("gaussian_count", lambda payload: payload.pop("gaussian_count")),
            (
                "gaussian_index_binding",
                lambda payload: payload.pop("gaussian_index_binding"),
            ),
            ("producer_identity", lambda payload: payload.pop("producer_identity")),
            ("views", lambda payload: payload.pop("views")),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                fixture = self._fixture(Path(tmp))
                payload = json.loads(fixture["source"].read_text(encoding="utf-8"))
                mutate(payload)
                _write_json(fixture["source"], payload)
                with self.assertRaises(ValueError):
                    bind_formal_geometry_manifest(
                        source_manifest_path=fixture["source"],
                        probe_camera_manifest_path=fixture["probe"],
                        formal_split_manifest_path=fixture["formal"],
                        output_manifest_path=fixture["output"],
                    )
                self.assertFalse(fixture["output"].exists())

    def test_wrong_depth_or_opacity_semantics_fail_closed(self) -> None:
        mutations = (
            lambda payload: payload.__setitem__("depth_semantics", "metric_depth"),
            lambda payload: payload.__setitem__("opacity_semantics", "approximate"),
            lambda payload: payload["depth_diagnostics"].__setitem__(
                "accumulated_opacity", "sum(alpha)"
            ),
        )
        for mutate in mutations:
            with self.subTest(mutation=repr(mutate)), tempfile.TemporaryDirectory() as tmp:
                fixture = self._fixture(Path(tmp))
                payload = json.loads(fixture["source"].read_text(encoding="utf-8"))
                mutate(payload)
                _write_json(fixture["source"], payload)
                with self.assertRaisesRegex(ValueError, "semantics"):
                    bind_formal_geometry_manifest(
                        source_manifest_path=fixture["source"],
                        probe_camera_manifest_path=fixture["probe"],
                        formal_split_manifest_path=fixture["formal"],
                        output_manifest_path=fixture["output"],
                    )

    def test_source_producer_hash_and_commit_formats_fail_closed(self) -> None:
        mutations = (
            lambda payload: payload["producer_identity"].__setitem__(
                "script_sha256", "not-a-sha"
            ),
            lambda payload: payload["producer_identity"].__setitem__(
                "git_commit", "short"
            ),
            lambda payload: payload["producer_identity"].__setitem__(
                "git_dirty", "false"
            ),
        )
        for mutate in mutations:
            with self.subTest(mutation=repr(mutate)), tempfile.TemporaryDirectory() as tmp:
                fixture = self._fixture(Path(tmp))
                payload = json.loads(fixture["source"].read_text(encoding="utf-8"))
                mutate(payload)
                _write_json(fixture["source"], payload)
                with self.assertRaisesRegex(ValueError, "producer_identity"):
                    bind_formal_geometry_manifest(
                        source_manifest_path=fixture["source"],
                        probe_camera_manifest_path=fixture["probe"],
                        formal_split_manifest_path=fixture["formal"],
                        output_manifest_path=fixture["output"],
                    )

    def test_pinned_source_and_formal_split_hashes_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "operator-pinned"):
                bind_formal_geometry_manifest(
                    source_manifest_path=fixture["source"],
                    probe_camera_manifest_path=fixture["probe"],
                    formal_split_manifest_path=fixture["formal"],
                    output_manifest_path=fixture["output"],
                    expected_source_sha256="0" * 64,
                )

        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            payload = json.loads(fixture["source"].read_text(encoding="utf-8"))
            payload["probe_camera_manifest_identity"]["sha256"] = "0" * 64
            _write_json(fixture["source"], payload)
            with self.assertRaisesRegex(ValueError, "probe camera manifest SHA-256"):
                bind_formal_geometry_manifest(
                    source_manifest_path=fixture["source"],
                    probe_camera_manifest_path=fixture["probe"],
                    formal_split_manifest_path=fixture["formal"],
                    output_manifest_path=fixture["output"],
                )

        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            payload = json.loads(fixture["source"].read_text(encoding="utf-8"))
            payload["formal_split_manifest_identity"]["sha256"] = "0" * 64
            _write_json(fixture["source"], payload)
            with self.assertRaisesRegex(ValueError, "formal split manifest SHA-256"):
                bind_formal_geometry_manifest(
                    source_manifest_path=fixture["source"],
                    probe_camera_manifest_path=fixture["probe"],
                    formal_split_manifest_path=fixture["formal"],
                    output_manifest_path=fixture["output"],
                )

    def test_source_mutation_during_binding_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))

            def mutate_source() -> None:
                with fixture["source"].open("ab") as handle:
                    handle.write(b" ")

            with self.assertRaisesRegex(RuntimeError, "changed while"):
                bind_formal_geometry_manifest(
                    source_manifest_path=fixture["source"],
                    probe_camera_manifest_path=fixture["probe"],
                    formal_split_manifest_path=fixture["formal"],
                    output_manifest_path=fixture["output"],
                    _precommit_hook=mutate_source,
                )
            self.assertFalse(fixture["output"].exists())

    def test_existing_output_is_idempotent_only_when_bytes_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            kwargs = {
                "source_manifest_path": fixture["source"],
                "probe_camera_manifest_path": fixture["probe"],
                "formal_split_manifest_path": fixture["formal"],
                "output_manifest_path": fixture["output"],
            }
            first = bind_formal_geometry_manifest(**kwargs)
            first_bytes = fixture["output"].read_bytes()
            second = bind_formal_geometry_manifest(**kwargs)
            self.assertEqual(first, second)
            self.assertEqual(fixture["output"].read_bytes(), first_bytes)

            fixture["output"].write_text("{}\n", encoding="utf-8")
            conflicting_bytes = fixture["output"].read_bytes()
            with self.assertRaisesRegex(FileExistsError, "different bytes"):
                bind_formal_geometry_manifest(**kwargs)
            self.assertEqual(fixture["output"].read_bytes(), conflicting_bytes)

    def test_output_must_be_derived_and_colocated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                bind_formal_geometry_manifest(
                    source_manifest_path=fixture["source"],
                    probe_camera_manifest_path=fixture["probe"],
                    formal_split_manifest_path=fixture["formal"],
                    output_manifest_path=fixture["source"],
                )
            with self.assertRaisesRegex(ValueError, "beside"):
                bind_formal_geometry_manifest(
                    source_manifest_path=fixture["source"],
                    probe_camera_manifest_path=fixture["probe"],
                    formal_split_manifest_path=fixture["formal"],
                    output_manifest_path=Path(tmp) / "elsewhere" / "manifest.json",
                )


if __name__ == "__main__":
    unittest.main()
