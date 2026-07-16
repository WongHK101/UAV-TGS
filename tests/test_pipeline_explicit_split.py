from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import run_uavfgs_pipeline as pipeline


class PipelineExplicitSplitTests(unittest.TestCase):
    def test_requires_both_lists_and_rejects_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.txt"
            test = root / "test.txt"
            train.write_text("0001.JPG\n0002.JPG\n", encoding="utf-8")
            test.write_text("0002.JPG\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "supplied together"):
                pipeline._resolve_explicit_camera_lists(str(train), "")
            with self.assertRaisesRegex(ValueError, "overlap"):
                pipeline._resolve_explicit_camera_lists(str(train), str(test))

    def test_normalises_and_forwards_lists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.txt"
            test = root / "test.txt"
            train.write_text("0001.JPG\n", encoding="utf-8")
            test.write_text("0002.JPG\n", encoding="utf-8")
            resolved_train, resolved_test = pipeline._resolve_explicit_camera_lists(
                str(train), str(test)
            )
            command = ["python", "train.py"]
            pipeline._append_explicit_camera_lists(command, resolved_train, resolved_test)
            self.assertEqual(
                command[-8:],
                [
                    "--train_list", str(train.resolve()),
                    "--test_list", str(test.resolve()),
                    "--train_list_sha256", "",
                    "--test_list_sha256", "",
                ],
            )


if __name__ == "__main__":
    unittest.main()
