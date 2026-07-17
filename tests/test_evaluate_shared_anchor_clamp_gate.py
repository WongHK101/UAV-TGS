import argparse
import json
from pathlib import Path
import tempfile
import unittest

from tools.evaluate_shared_anchor_clamp_gate import evaluate


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _results(psnr, ssim, lpips):
    return {"ours_30000": {"PSNR": psnr, "SSIM": ssim, "LPIPS": lpips}}


def _depth(scene, front, mean, median, missing, reference="a" * 64):
    return {
        "scene_name": scene,
        "reference_manifest_sha256": reference,
        "secondary_metrics": {
            "MissingRate": missing,
            "AbsDepthError_Mean": mean,
            "AbsDepthError_Median": median,
        },
        "threshold_metrics": [
            {
                "threshold_m": threshold,
                "FrontIntrusionRate": front if threshold == 1.0 else front / threshold,
                "TooDeepRate": 0.1,
                "DepthAgreementRate": 0.8,
            }
            for threshold in (1.0, 2.0, 5.0)
        ],
    }


class SharedAnchorClampGateTests(unittest.TestCase):
    def _case(self, root: Path, scene: str, shared_front=0.32):
        expected = 20 if scene == "InternalRoad" else 35
        files = {
            "anchor_results": _write(root / "anchor_results.json", _results(25.0, 0.8, 0.2)),
            "shared_results": _write(root / "shared_results.json", _results(24.95, 0.799, 0.203)),
            "anchor_depth": _write(root / "anchor_depth.json", _depth(scene, 0.42, 1.7, 1.1, 0.001)),
            "shared_depth": _write(root / "shared_depth.json", _depth(scene, shared_front, 1.2, 0.8, 0.0015)),
            "legacy_depth": _write(root / "legacy_depth.json", _depth(scene, 0.30, 1.0, 0.7, 0.001)),
            "shared_manifest": _write(
                root / "manifest.json",
                {
                    "status": "passed",
                    "scene": scene,
                    "anchor_iteration": 30000,
                    "operation": {
                        "training_updates": 0,
                        "max_activated_scale": 10.0,
                    },
                    "counts": {"actual_clamped_gaussians": expected},
                    "invariants": {"only_scaling_changed": True},
                },
            ),
            "block_analysis": _write(
                root / "blocks.json",
                {
                    "status": "complete",
                    "scene": scene,
                    "groups": ["Anchor", "S"],
                },
            ),
            "qualitative_assessment": _write(
                root / "qualitative.json",
                {
                    "status": "complete",
                    "scene": scene,
                    "fixed_views_reviewed": True,
                    "mechanism_render_reviewed": True,
                    "obvious_thin_structure_collapse": False,
                    "visible_structural_failure": False,
                },
            ),
        }
        return argparse.Namespace(
            scene=scene,
            anchor_iteration=30000,
            **{key: str(value) for key, value in files.items()},
        )

    def test_internalroad_passes_on_front_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            payload = evaluate(self._case(Path(temp), "InternalRoad"))
            self.assertTrue(payload["first_stage_passed"])
            self.assertGreaterEqual(
                payload["depth"]["recovery"]["front_at_1m"]["value"], 0.70
            )

    def test_building_uses_safety_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            payload = evaluate(
                self._case(Path(temp), "Building", shared_front=0.425)
            )
            self.assertTrue(payload["first_stage_passed"])
            self.assertLessEqual(
                payload["depth"]["delta"]["front_at_1m_increase_pp"], 1.0
            )

    def test_nonpositive_recovery_denominator_is_not_applicable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._case(root, "InternalRoad")
            _write(Path(args.legacy_depth), _depth("InternalRoad", 0.45, 1.8, 1.2, 0.001))
            payload = evaluate(args)
            self.assertEqual(
                payload["depth"]["recovery"]["front_at_1m"]["status"],
                "not_applicable",
            )
            self.assertEqual(
                payload["depth"]["recovery"]["mean_error"]["status"],
                "not_applicable",
            )
            self.assertFalse(payload["first_stage_passed"])


if __name__ == "__main__":
    unittest.main()
