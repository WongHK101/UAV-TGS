from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.geometric_repeatability import materialize_train_only_colmap as materializer
from utils.read_write_model import Camera, Image, Point3D, read_model, write_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(
    image_id: int,
    camera_id: int,
    name: str,
    point_ids: list[int],
) -> Image:
    count = len(point_ids)
    return Image(
        id=image_id,
        qvec=np.asarray([1.0, image_id / 100.0, 0.0, 0.0], dtype=np.float64),
        tvec=np.asarray([float(image_id), 2.0, 3.0], dtype=np.float64),
        camera_id=camera_id,
        name=name,
        xys=np.asarray([[float(index), float(index + 1)] for index in range(count)], dtype=np.float64),
        point3D_ids=np.asarray(point_ids, dtype=np.int64),
    )


class TrainOnlyColmapMaterializerTests(unittest.TestCase):
    def _fixture(self, root: Path, ext: str = ".bin") -> dict[str, Path]:
        model_root = root / "all_view" / "sparse" / "0"
        image_root = root / "all_view" / "images"
        model_root.mkdir(parents=True)
        image_root.mkdir(parents=True)

        cameras = {
            1: Camera(1, "PINHOLE", 16, 12, np.asarray([10.0, 11.0, 8.0, 6.0])),
            2: Camera(2, "PINHOLE", 16, 12, np.asarray([12.0, 13.0, 8.0, 6.0])),
            3: Camera(3, "PINHOLE", 16, 12, np.asarray([14.0, 15.0, 8.0, 6.0])),
        }
        images = {
            1: _image(1, 1, "0001.JPG", [100, 102]),
            2: _image(2, 2, "0002.JPG", [102, -1]),
            3: _image(3, 1, "0003.JPG", [100, 101]),
            4: _image(4, 2, "0004.JPG", [101]),
        }
        points = {
            100: Point3D(
                100,
                np.asarray([1.0, 0.0, 3.0]),
                np.asarray([100, 101, 102], dtype=np.uint8),
                0.1,
                np.asarray([1, 3], dtype=np.int32),
                np.asarray([0, 0], dtype=np.int32),
            ),
            101: Point3D(
                101,
                np.asarray([2.0, 0.0, 3.0]),
                np.asarray([110, 111, 112], dtype=np.uint8),
                0.2,
                np.asarray([3, 4], dtype=np.int32),
                np.asarray([1, 0], dtype=np.int32),
            ),
            102: Point3D(
                102,
                np.asarray([3.0, 0.0, 3.0]),
                np.asarray([120, 121, 122], dtype=np.uint8),
                0.3,
                np.asarray([1, 2], dtype=np.int32),
                np.asarray([1, 0], dtype=np.int32),
            ),
        }
        write_model(cameras, images, points, str(model_root), ext=ext)
        for index in range(1, 6):
            (image_root / f"{index:04d}.JPG").write_bytes(f"rgb-frame-{index}".encode("ascii"))

        train = root / "train.txt"
        test = root / "test.txt"
        guard = root / "guard.txt"
        train.write_text("0001.JPG\n0002.JPG\n0005.JPG\n", encoding="utf-8")
        test.write_text("0003.JPG\n", encoding="utf-8")
        guard.write_text("0004.JPG\n", encoding="utf-8")
        binding = root / "binding_manifest.json"
        binding.write_text(
            json.dumps(
                {
                    "schema_name": "uav_tgs_formal_scene_decode_binding",
                    "schema_version": 1,
                    "status": "passed",
                    "binding_hash": "fixture-binding-hash",
                    "scene": "Fixture",
                    "sfm_image_scope": "shared_sfm_all_images",
                    "counts": {"total": 5, "train": 3, "test": 1, "guard": 1},
                    "collection_hash": "collection-hash",
                    "collection_split_hash": "collection-split-hash",
                    "formal_rule_hash": "rule-hash",
                    "scene_split_hash": "scene-split-hash",
                    "outputs": {
                        "train_list.txt": {
                            "path": str(train),
                            "sha256": _sha256(train),
                        },
                        "test_list.txt": {
                            "path": str(test),
                            "sha256": _sha256(test),
                        },
                        "guard_list.txt": {
                            "path": str(guard),
                            "sha256": _sha256(guard),
                        },
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "model": model_root,
            "images": image_root,
            "binding": binding,
            "train": train,
            "test": test,
            "guard": guard,
            "output": root / "train_only",
        }

    def _run(self, paths: dict[str, Path], **overrides):
        kwargs = {
            "source_model_root": paths["model"],
            "binding_manifest_path": paths["binding"],
            "sfm_image_scope": "shared_sfm_all_images",
            "train_list_path": paths["train"],
            "test_list_path": paths["test"],
            "guard_list_path": paths["guard"],
            "image_root": paths["images"],
            "output_workspace": paths["output"],
            "image_mode": "copy",
        }
        kwargs.update(overrides)
        return materializer.materialize_train_only_workspace(**kwargs)

    def test_filters_images_tracks_reverse_references_and_empty_points(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            source_cameras, source_images, _ = read_model(str(paths["model"]), ext=".bin")
            manifest = self._run(paths)

            output_model = paths["output"] / "sparse" / "0"
            cameras, images, points = read_model(str(output_model), ext=".bin")
            self.assertEqual({image.name for image in images.values()}, {"0001.JPG", "0002.JPG"})
            self.assertEqual(set(cameras), {1, 2})
            self.assertEqual(set(points), {100, 102})
            np.testing.assert_array_equal(points[100].image_ids, np.asarray([1]))
            np.testing.assert_array_equal(points[100].point2D_idxs, np.asarray([0]))
            np.testing.assert_array_equal(points[102].image_ids, np.asarray([1, 2]))
            for image_id in (1, 2):
                np.testing.assert_array_equal(images[image_id].qvec, source_images[image_id].qvec)
                np.testing.assert_array_equal(images[image_id].tvec, source_images[image_id].tvec)
                np.testing.assert_array_equal(cameras[images[image_id].camera_id].params, source_cameras[images[image_id].camera_id].params)

            output_files = sorted(path.name for path in (paths["output"] / "images").iterdir())
            self.assertEqual(output_files, ["0001.JPG", "0002.JPG", "0005.JPG"])
            self.assertFalse((paths["output"] / "images" / "0003.JPG").exists())
            self.assertFalse((paths["output"] / "images" / "0004.JPG").exists())

            loaded_manifest = json.loads(
                (paths["output"] / "train_only_colmap_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded_manifest, manifest)
            self.assertEqual(manifest["partition"]["train_list"]["count"], 3)
            self.assertEqual(manifest["partition"]["unregistered_train_count"], 1)
            self.assertEqual(manifest["partition"]["registered"], {"train": 2, "test": 1, "guard": 1})
            self.assertEqual(manifest["filtering"]["removed_points_with_empty_train_track"], 1)
            self.assertEqual(manifest["filtering"]["removed_nontrain_track_entries"], 3)
            self.assertEqual(
                manifest["source"]["sfm_image_scope"], "shared_sfm_all_images"
            )
            self.assertIn("not an independently reconstructed", manifest["output"]["semantics"])
            self.assertEqual(
                manifest["formal_provenance"]["file_sha256"],
                _sha256(paths["binding"]),
            )
            self.assertEqual(
                manifest["formal_provenance"]["binding_hash"],
                "fixture-binding-hash",
            )
            self.assertEqual(
                len(manifest["formal_provenance"]["partition_semantic_sha256"]),
                64,
            )
            preservation = manifest["filtering"]["reload_exact_camera_pose_preservation"]
            self.assertEqual(preservation["status"], "passed")
            self.assertEqual(
                preservation["source_semantic_sha256"],
                preservation["output_semantic_sha256"],
            )
            self.assertTrue(manifest["invariants"]["retained_camera_parameters_exact_after_reload"])
            self.assertTrue(manifest["invariants"]["retained_image_poses_exact_after_reload"])
            self.assertTrue(manifest["partition"]["validation"]["test_images_rejected_from_output"])
            self.assertTrue(manifest["partition"]["validation"]["guard_images_rejected_from_output"])
            self.assertEqual(
                manifest["source"]["selected_train_images_bundle_sha256"],
                manifest["output"]["images_bundle_sha256"],
            )

    def test_preserves_text_model_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary), ext=".txt")
            manifest = self._run(paths)
            sparse = paths["output"] / "sparse" / "0"
            self.assertTrue((sparse / "cameras.txt").is_file())
            self.assertTrue((sparse / "images.txt").is_file())
            self.assertTrue((sparse / "points3D.txt").is_file())
            self.assertFalse((sparse / "images.bin").exists())
            self.assertEqual(manifest["source"]["model_format"], "txt")

    def test_overlap_fails_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            paths["guard"].write_text("0002.JPG\n0004.JPG\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                self._run(paths)
            self.assertFalse(paths["output"].exists())

    def test_unclassified_registered_image_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            paths["train"].write_text("0001.JPG\n0002.JPG\n", encoding="utf-8")
            paths["guard"].write_text("0005.JPG\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the supplied"):
                self._run(paths)
            self.assertFalse(paths["output"].exists())

    def test_corrupt_reverse_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            cameras, images, points = read_model(str(paths["model"]), ext=".bin")
            images[1] = images[1]._replace(point3D_ids=np.asarray([100, -1], dtype=np.int64))
            write_model(cameras, images, points, str(paths["model"]), ext=".bin")
            with self.assertRaisesRegex(ValueError, "reverse reference"):
                self._run(paths)
            self.assertFalse(paths["output"].exists())

    def test_binding_scope_and_list_hash_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "shared_sfm_all_images"):
                self._run(paths, sfm_image_scope="train_only")
            self.assertFalse(paths["output"].exists())

            binding = json.loads(paths["binding"].read_text(encoding="utf-8"))
            binding["outputs"]["train_list.txt"]["sha256"] = "0" * 64
            paths["binding"].write_text(
                json.dumps(binding, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "list SHA mismatch"):
                self._run(paths)
            self.assertFalse(paths["output"].exists())

    def test_reload_comparison_rejects_pose_identity_or_camera_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            cameras, images, _ = read_model(str(paths["model"]), ext=".bin")
            retained_images = {image_id: images[image_id] for image_id in (1, 2)}
            retained_cameras = {camera_id: cameras[camera_id] for camera_id in (1, 2)}

            image_mutations = {
                "qvec": {
                    "qvec": np.asarray(retained_images[1].qvec)
                    + np.asarray([0.0, 0.1, 0.0, 0.0])
                },
                "tvec": {
                    "tvec": np.asarray(retained_images[1].tvec)
                    + np.asarray([0.0, 0.0, 0.1])
                },
                "camera_id": {"camera_id": 2},
                "name": {"name": "renamed.JPG"},
            }
            for field, replacement in image_mutations.items():
                with self.subTest(field=field):
                    changed_images = dict(retained_images)
                    changed_images[1] = changed_images[1]._replace(**replacement)
                    with self.assertRaisesRegex(RuntimeError, "camera/name/qvec/tvec"):
                        materializer._validate_reloaded_camera_pose_preservation(
                            retained_cameras,
                            retained_images,
                            retained_cameras,
                            changed_images,
                        )

            changed_cameras = dict(retained_cameras)
            changed_cameras[1] = changed_cameras[1]._replace(
                params=np.asarray(changed_cameras[1].params) + 0.5
            )
            with self.assertRaisesRegex(RuntimeError, "camera 1"):
                materializer._validate_reloaded_camera_pose_preservation(
                    retained_cameras,
                    retained_images,
                    changed_cameras,
                    retained_images,
                )

    def test_formal_provenance_cli_arguments_are_required(self) -> None:
        actions = {
            action.dest: action for action in materializer.build_parser()._actions
        }
        self.assertTrue(actions["binding_manifest"].required)
        self.assertTrue(actions["sfm_image_scope"].required)
        self.assertEqual(
            tuple(actions["sfm_image_scope"].choices),
            ("shared_sfm_all_images",),
        )

    def test_existing_output_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            paths["output"].mkdir()
            sentinel = paths["output"] / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to replace"):
                self._run(paths)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_output_inside_source_image_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            unsafe_output = paths["images"] / "derived"
            with self.assertRaisesRegex(ValueError, "inside the RGB image root"):
                self._run(paths, output_workspace=unsafe_output)
            self.assertFalse(unsafe_output.exists())


if __name__ == "__main__":
    unittest.main()
