from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_render_module():
    scene = types.ModuleType("scene")
    scene.Scene = object

    gaussian_renderer = types.ModuleType("gaussian_renderer")
    gaussian_renderer.render = lambda *args, **kwargs: None
    gaussian_renderer.GaussianModel = object

    torchvision = types.ModuleType("torchvision")
    torchvision.utils = types.SimpleNamespace(save_image=lambda *args, **kwargs: None)

    general_utils = types.ModuleType("utils.general_utils")
    general_utils.safe_state = lambda quiet: None

    arguments = types.ModuleType("arguments")
    arguments.ModelParams = object
    arguments.PipelineParams = object
    arguments.get_combined_args = lambda parser: parser.parse_args([])

    rasterization = types.ModuleType("diff_gaussian_rasterization")
    rasterization.SparseGaussianAdam = object

    stubs = {
        "scene": scene,
        "gaussian_renderer": gaussian_renderer,
        "torchvision": torchvision,
        "utils.general_utils": general_utils,
        "arguments": arguments,
        "diff_gaussian_rasterization": rasterization,
    }
    module_name = "render_formal_output_test_module"
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "render.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class _View:
    def __init__(self, image_name: str):
        self.image_name = image_name
        self.original_image = torch.zeros((3, 2, 3), dtype=torch.float32)


class RenderFormalOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_render_module()

    @staticmethod
    def _save_image(_tensor, path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture-png")

    def _run_named_render(self, model_path: Path) -> dict:
        self.module.torchvision.utils.save_image = self._save_image
        self.module.render = lambda *args, **kwargs: {
            "render": torch.zeros((3, 2, 3), dtype=torch.float32),
            "alpha": torch.tensor(
                [[[0.0, 0.25, 0.5], [0.75, 1.0, 0.125]]],
                dtype=torch.float32,
            ),
        }
        self.module.render_set(
            str(model_path),
            "test",
            40000,
            [_View("Building_0007.JPG"), _View("Building_0099.png")],
            object(),
            object(),
            object(),
            False,
            False,
            save_by_image_name=True,
            save_renderer_support=True,
        )
        manifest_path = (
            model_path / "test" / "ours_40000" / "render_mapping_manifest.json"
        )
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def test_named_outputs_support_npy_and_deterministic_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first_root = Path(tmp) / "first"
            second_root = Path(tmp) / "second"
            first = self._run_named_render(first_root)
            second = self._run_named_render(second_root)

            self.assertEqual(first, second)
            output_root = first_root / "test" / "ours_40000"
            self.assertTrue((output_root / "renders" / "Building_0007.png").is_file())
            self.assertTrue((output_root / "gt" / "Building_0099.png").is_file())
            support = np.load(
                output_root / "renderer_support" / "Building_0007.npy",
                allow_pickle=False,
            )
            self.assertEqual(support.dtype, np.float32)
            self.assertEqual(support.shape, (2, 3))
            self.assertAlmostEqual(float(support[1, 1]), 1.0)

            self.assertEqual(first["split"], "test")
            self.assertEqual(first["iteration"], 40000)
            self.assertEqual(first["name_mode"], "image_name_stem")
            self.assertFalse(first["support_threshold_applied"])
            self.assertIsNone(first["support_threshold"])
            self.assertEqual(
                first["entries"][0]["source_image_name"],
                "Building_0007.JPG",
            )
            self.assertEqual(
                first["entries"][0]["output"]["render"],
                "renders/Building_0007.png",
            )
            self.assertEqual(
                first["entries"][0]["renderer_support_source_key"],
                "alpha",
            )
            self.assertEqual(
                first["entries"][0]["output_sha256"]["renderer_support"],
                self.module._sha256(
                    output_root / "renderer_support" / "Building_0007.npy"
                ),
            )

    def test_duplicate_stems_fail_before_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            with self.assertRaisesRegex(ValueError, "Duplicate image_name stem"):
                self.module.render_set(
                    str(root),
                    "test",
                    1,
                    [_View("flight/A.JPG"), _View("other/a.png")],
                    object(),
                    object(),
                    object(),
                    False,
                    False,
                    save_by_image_name=True,
                )
            self.assertFalse(root.exists())

    def test_support_request_fails_closed_when_renderer_has_no_native_map(self) -> None:
        saved = []
        self.module.torchvision.utils.save_image = lambda tensor, path: saved.append(path)
        self.module.render = lambda *args, **kwargs: {
            "render": torch.zeros((3, 2, 3), dtype=torch.float32),
            "depth": torch.ones((1, 2, 3), dtype=torch.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "Refusing to synthesize"):
                self.module.render_set(
                    tmp,
                    "test",
                    1,
                    [_View("Building_0001.JPG")],
                    object(),
                    object(),
                    object(),
                    False,
                    False,
                    save_by_image_name=True,
                    save_renderer_support=True,
                )
        self.assertEqual(saved, [])

    def test_opacity_proxy_is_separate_and_uses_established_white_override(self) -> None:
        calls = []

        def fake_render(*args, **kwargs):
            calls.append((args, kwargs))
            if kwargs.get("override_color") is not None:
                return {
                    "render": torch.tensor(
                        [
                            [[0.0, 0.25, 0.5], [0.75, 1.0, 0.125]],
                            [[0.0, 0.25, 0.5], [0.75, 1.0, 0.125]],
                            [[0.0, 0.25, 0.5], [0.75, 1.0, 0.125]],
                        ],
                        dtype=torch.float32,
                    )
                }
            return {"render": torch.zeros((3, 2, 3), dtype=torch.float32)}

        class _Gaussians:
            get_xyz = torch.zeros((4, 3), dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.module.render = fake_render
            self.module.torchvision.utils.save_image = self._save_image
            self.module.render_set(
                str(root),
                "test",
                3,
                [_View("Building_0003.JPG")],
                _Gaussians(),
                object(),
                torch.zeros(3, dtype=torch.float32),
                False,
                False,
                save_by_image_name=True,
                save_opacity_proxy=True,
            )

            self.assertEqual(len(calls), 2)
            self.assertIsNone(calls[0][1].get("override_color"))
            self.assertTrue(torch.all(calls[1][0][3] == 0))
            self.assertTrue(torch.all(calls[1][1]["override_color"] == 1))
            self.assertFalse(calls[1][1]["separate_sh"])
            self.assertFalse(calls[1][1]["use_trained_exp"])

            output_root = root / "test" / "ours_3"
            proxy = np.load(
                output_root / "opacity_proxy" / "Building_0003.npy",
                allow_pickle=False,
            )
            self.assertEqual(proxy.dtype, np.float32)
            self.assertAlmostEqual(float(proxy[1, 1]), 1.0)
            manifest = json.loads(
                (output_root / "render_mapping_manifest.json").read_text(encoding="utf-8")
            )
            semantics = "black_bg_plus_white_override_color_render"
            self.assertTrue(manifest["opacity_proxy_saved"])
            self.assertEqual(manifest["opacity_proxy_semantics"], semantics)
            self.assertEqual(
                manifest["entries"][0]["opacity_proxy_semantics"],
                semantics,
            )
            self.assertEqual(
                manifest["entries"][0]["output_sha256"]["opacity_proxy"],
                self.module._sha256(
                    output_root / "opacity_proxy" / "Building_0003.npy"
                ),
            )
            self.assertFalse(manifest["support_threshold_applied"])
            self.assertIsNone(manifest["support_threshold"])

    def test_default_output_contract_remains_sequential_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.module.torchvision.utils.save_image = self._save_image
            self.module.render = lambda *args, **kwargs: {
                "render": torch.zeros((3, 2, 3), dtype=torch.float32),
            }
            self.module.render_set(
                str(root),
                "test",
                2,
                [_View("any-name.JPG")],
                object(),
                object(),
                object(),
                False,
                False,
            )
            output_root = root / "test" / "ours_2"
            self.assertTrue((output_root / "renders" / "00000.png").is_file())
            self.assertTrue((output_root / "gt" / "00000.png").is_file())
            self.assertFalse((output_root / "render_mapping_manifest.json").exists())
            self.assertFalse((output_root / "renderer_support").exists())


if __name__ == "__main__":
    unittest.main()
