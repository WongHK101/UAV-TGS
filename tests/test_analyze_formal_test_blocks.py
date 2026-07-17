from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools import analyze_formal_test_blocks


class FormalTestBlockAnalysisTests(unittest.TestCase):
    BLOCKS = (
        ("tg-0000", 0, range(1, 17), "nadir:p-090"),
        ("tg-0000", 5, range(81, 97), "nadir:p-090"),
        ("tg-0001", 5, range(270, 286), "oblique:p-045"),
        ("tg-0002", 2, range(327, 343), "oblique:p-060"),
        ("tg-0004", 4, range(570, 586), "oblique:p-075"),
    )
    FOUR_BLOCKS = BLOCKS[:4]
    SIX_BLOCKS = (
        *BLOCKS,
        ("tg-0005", 1, range(600, 616), "oblique:p-030"),
    )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _hex(index: int, salt: int = 0) -> str:
        return f"{index + salt:064x}"[-64:]

    def _build_fixture(
        self,
        root: Path,
        *,
        blocks: tuple | None = None,
        groups: tuple[str, ...] = analyze_formal_test_blocks.GROUPS,
    ) -> dict:
        blocks = self.BLOCKS if blocks is None else blocks
        records = []
        test_names = []
        block_by_name = {}
        global_index = 0
        for block_order, (strip_id, block_index, identifiers, stratum) in enumerate(blocks):
            for block_offset, identifier in enumerate(identifiers):
                stem = f"{identifier:04d}"
                name = f"{stem}.png"
                record = {
                    "split": "test",
                    "thermal_camera_name": name,
                    "strip_id": strip_id,
                    "block_index": block_index,
                    "block_offset": block_offset,
                    "position_in_strip": block_index * 16 + block_offset,
                    "pair_id": stem,
                    "stratum": stratum,
                    "hash": self._hex(global_index, 1000),
                }
                records.append(record)
                test_names.append(name)
                block_by_name[name] = block_order
                global_index += 1

        test_list_path = root / "thermal_test_list.txt"
        test_list_path.write_text("\n".join(test_names) + "\n", encoding="utf-8")
        bound_split_path = root / "bound_split.json"
        self._write_json(
            bound_split_path,
            {
                "selected_test_blocks_hash": "a" * 64,
                "records": records,
            },
        )
        bound_sha = analyze_formal_test_blocks.sha256_file(bound_split_path)

        # A3-vs-F3 has three combined non-degraded blocks and two declined
        # blocks.  This makes the >=3/5 rule and per-metric counts observable.
        a3_image_deltas = (
            {"PSNR": 0.06, "SSIM": 0.004, "LPIPS": -0.004},
            {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0},
            {"PSNR": -0.04, "SSIM": -0.002, "LPIPS": 0.002},
            {"PSNR": -0.06, "SSIM": 0.0, "LPIPS": 0.0},
            {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0},
        )
        image_base = {"PSNR": 20.0, "SSIM": 0.8, "LPIPS": 0.2}
        per_view_paths = {}
        for group in groups:
            method = {}
            for metric, base in image_base.items():
                values = {}
                for name in test_names:
                    block_order = block_by_name[name]
                    delta = (
                        a3_image_deltas[block_order % len(a3_image_deltas)][metric]
                        if group == "A3"
                        else 0.0
                    )
                    values[name] = base + delta
                method[metric] = values
            path = root / f"per_view_{group}.json"
            self._write_json(path, {"ours_60000": method})
            per_view_paths[group] = path

        a3_temperature_deltas = (
            {
                "mae_c": -0.11,
                "rmse_c": -0.11,
                "signed_bias_c": -0.11,
                "p95_abs_error_c": -0.11,
                "mean_rgb_distance": -0.1,
                "p95_rgb_distance": -0.1,
            },
            {},
            {
                "mae_c": 0.05,
                "rmse_c": 0.05,
                "signed_bias_c": 0.05,
                "p95_abs_error_c": 0.05,
                "mean_rgb_distance": 0.01,
                "p95_rgb_distance": 0.01,
            },
            {},
            {"mae_c": 0.11},
        )
        temperature_base = {
            "mae_c": 1.0,
            "rmse_c": 1.5,
            "signed_bias_c": 0.5,
            "p95_abs_error_c": 2.0,
            "mean_rgb_distance": 3.0,
            "p95_rgb_distance": 4.0,
        }
        temperature_paths = {}
        for group in groups:
            files = []
            for global_index, record in enumerate(records):
                name = record["thermal_camera_name"]
                stem = Path(name).stem
                block_order = block_by_name[name]
                delta = (
                    a3_temperature_deltas[block_order % len(a3_temperature_deltas)]
                    if group == "A3"
                    else {}
                )
                temperature = {
                    key: value + delta.get(key, 0.0)
                    for key, value in temperature_base.items()
                }
                files.append(
                    {
                        "relative_id": stem,
                        "status": "complete",
                        "missing_pixels": 0,
                        "ground_truth_sha256": self._hex(global_index, 2000),
                        "mask_sha256": self._hex(global_index, 3000),
                        "supported_pixels": 16,
                        "split_assignment": {
                            "split": "test",
                            "strip_id": record["strip_id"],
                            "block_index": record["block_index"],
                            "position_in_strip": record["position_in_strip"],
                            "pair_id": record["pair_id"],
                            "stratum": record["stratum"],
                            "hash": record["hash"],
                        },
                        "supported_pixel_temperature_error": {
                            "mae_c": temperature["mae_c"],
                            "rmse_c": temperature["rmse_c"],
                            "signed_bias_c": temperature["signed_bias_c"],
                            "p95_abs_error_c": temperature["p95_abs_error_c"],
                        },
                        "supported_pixel_off_lut_distance": {
                            "mean_rgb_distance": temperature["mean_rgb_distance"],
                            "p95_rgb_distance": temperature["p95_rgb_distance"],
                        },
                    }
                )
            path = root / f"temperature_{group}.json"
            self._write_json(
                path,
                {
                    "status": "complete",
                    "completed_with_missing": False,
                    "split": {"subset": "test", "sha256": bound_sha},
                    "summary": {"evaluated_file_count": len(test_names)},
                    "files": files,
                },
            )
            temperature_paths[group] = path

        return {
            "test_list_path": test_list_path,
            "bound_split_path": bound_split_path,
            "per_view_paths": per_view_paths,
            "temperature_paths": temperature_paths,
            "test_names": test_names,
            "groups": groups,
            "blocks": blocks,
        }

    @staticmethod
    def _declarations(fixture: dict) -> list[str]:
        declarations = [
            "test_list="
            + analyze_formal_test_blocks.sha256_file(fixture["test_list_path"]),
            "bound_split="
            + analyze_formal_test_blocks.sha256_file(fixture["bound_split_path"]),
        ]
        for group, path in fixture["per_view_paths"].items():
            declarations.append(
                f"per_view:{group}="
                + analyze_formal_test_blocks.sha256_file(path)
            )
        for group, path in fixture["temperature_paths"].items():
            declarations.append(
                f"temperature:{group}="
                + analyze_formal_test_blocks.sha256_file(path)
            )
        return declarations

    def _run(self, fixture: dict) -> dict:
        return analyze_formal_test_blocks.run_analysis(
            test_list_path=fixture["test_list_path"],
            bound_split_path=fixture["bound_split_path"],
            per_view_paths=fixture["per_view_paths"],
            temperature_paths=fixture["temperature_paths"],
            expected_sha256=self._declarations(fixture),
        )

    def test_exact_five_blocks_paired_counts_and_combined_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            payload = self._run(self._build_fixture(Path(temp)))

            self.assertEqual(payload["iteration"], 60000)
            self.assertIn("40k, 50k, and 60k", payload["claim_boundary"])
            self.assertEqual(payload["protocol"]["groups"], ["L", "C3", "F3", "A3"])
            self.assertEqual(payload["protocol"]["test_views"], 80)
            self.assertEqual(payload["protocol"]["block_count"], 5)
            self.assertEqual(len(payload["blocks"]), 5)
            actual_keys = []
            for record in payload["blocks"]:
                block = record["block"]
                actual_keys.append((block["strip_id"], block["block_index"]))
                self.assertEqual(block["size"], 16)
                self.assertEqual(len(block["views"]), 16)
                self.assertEqual(
                    set(record["paired_comparisons"]),
                    {"A3_minus_C3", "A3_minus_F3", "A3_minus_L"},
                )
            self.assertEqual(actual_keys, [(item[0], item[1]) for item in self.BLOCKS])

            judgment = payload["a3_vs_f3_combined_judgment"]
            self.assertEqual(judgment["non_degraded_blocks"], 3)
            self.assertEqual(judgment["required_blocks"], 3)
            self.assertTrue(judgment["passed"])
            psnr_counts = payload["classification_counts"]["A3_minus_F3"]["PSNR"]
            self.assertEqual(psnr_counts, {"improved": 1, "tied": 3, "declined": 1})
            mae_counts = payload["classification_counts"]["A3_minus_F3"][
                "temperature_mae_c"
            ]
            self.assertEqual(mae_counts, {"improved": 1, "tied": 3, "declined": 1})
            first = payload["blocks"][0]
            self.assertAlmostEqual(first["group_means"]["A3"]["PSNR"], 20.06)
            self.assertAlmostEqual(
                first["group_means"]["A3"]["temperature_abs_bias_c"], 0.39
            )
            bias_delta = first["paired_comparisons"]["A3_minus_F3"][
                "diagnostic_deltas"
            ]["temperature_signed_bias_c"]
            self.assertAlmostEqual(
                bias_delta["raw_a3_minus_baseline"],
                first["group_means"]["A3"]["temperature_signed_bias_c"]
                - first["group_means"]["F3"]["temperature_signed_bias_c"],
            )
            self.assertEqual(bias_delta["classification"], "diagnostic_only")

    def test_cli_writes_json_and_long_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self._build_fixture(root)
            output = root / "analysis.json"
            csv_path = root / "analysis.csv"
            argv = [
                "--test-list",
                str(fixture["test_list_path"]),
                "--bound-split",
                str(fixture["bound_split_path"]),
            ]
            for group in fixture["groups"]:
                argv.extend(["--per-view", f"{group}={fixture['per_view_paths'][group]}"])
                argv.extend(
                    ["--temperature", f"{group}={fixture['temperature_paths'][group]}"]
                )
            for declaration in self._declarations(fixture):
                argv.extend(["--expected-sha256", declaration])
            argv.extend(["--output", str(output), "--csv", str(csv_path)])

            self.assertEqual(analyze_formal_test_blocks.main(argv), 0)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["schema"], analyze_formal_test_blocks.SCHEMA)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 350)
            self.assertEqual(
                sum(row["row_type"] == "paired_delta" for row in rows), 150
            )
            self.assertEqual(
                sum(
                    row["metric"] == "temperature_signed_bias_c"
                    and row["row_type"] == "paired_delta"
                    for row in rows
                ),
                15,
            )

    def test_sha_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = self._build_fixture(Path(temp))
            declarations = self._declarations(fixture)
            declarations = [
                "per_view:L=" + "0" * 64 if item.startswith("per_view:L=") else item
                for item in declarations
            ]
            with self.assertRaisesRegex(
                analyze_formal_test_blocks.BlockAnalysisError,
                "SHA-256 mismatch for per_view:L",
            ):
                analyze_formal_test_blocks.run_analysis(
                    test_list_path=fixture["test_list_path"],
                    bound_split_path=fixture["bound_split_path"],
                    per_view_paths=fixture["per_view_paths"],
                    temperature_paths=fixture["temperature_paths"],
                    expected_sha256=declarations,
                )

    def test_incomplete_16_view_bound_block_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = self._build_fixture(Path(temp))
            payload = json.loads(fixture["bound_split_path"].read_text(encoding="utf-8"))
            payload["records"][-1]["block_index"] = self.BLOCKS[-2][1]
            payload["records"][-1]["strip_id"] = self.BLOCKS[-2][0]
            self._write_json(fixture["bound_split_path"], payload)
            with self.assertRaisesRegex(
                analyze_formal_test_blocks.BlockAnalysisError,
                "expected exactly 5 complete test blocks|has 15 views|has 17 views",
            ):
                self._run(fixture)

    def test_per_view_set_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = self._build_fixture(Path(temp))
            path = fixture["per_view_paths"]["A3"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["ours_60000"]["LPIPS"][fixture["test_names"][0]]
            self._write_json(path, payload)
            with self.assertRaisesRegex(
                analyze_formal_test_blocks.BlockAnalysisError,
                "per_view set mismatch for A3/LPIPS",
            ):
                self._run(fixture)

    def test_duplicate_test_frame_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = self._build_fixture(Path(temp))
            names = list(fixture["test_names"])
            names[1] = names[0]
            fixture["test_list_path"].write_text(
                "\n".join(names) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                analyze_formal_test_blocks.BlockAnalysisError,
                "unique entries",
            ):
                self._run(fixture)

    def test_l_a3_internalroad_64_views_four_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self._build_fixture(
                root, blocks=self.FOUR_BLOCKS, groups=("L", "A3")
            )
            payload = self._run(fixture)

            self.assertEqual(payload["protocol"]["groups"], ["L", "A3"])
            self.assertEqual(payload["protocol"]["test_views"], 64)
            self.assertEqual(payload["protocol"]["block_count"], 4)
            self.assertEqual(len(payload["blocks"]), 4)
            for record in payload["blocks"]:
                self.assertEqual(set(record["group_means"]), {"L", "A3"})
                self.assertEqual(
                    set(record["paired_comparisons"]), {"A3_minus_L"}
                )
            self.assertEqual(set(payload["classification_counts"]), {"A3_minus_L"})
            self.assertNotIn("a3_vs_l_combined_judgment", payload)
            self.assertEqual(
                sum(
                    record["paired_comparisons"]["A3_minus_L"][
                        "combined_non_degraded"
                    ]
                    for record in payload["blocks"]
                ),
                3,
            )

            output = root / "l_a3_analysis.json"
            csv_path = root / "l_a3_analysis.csv"
            argv = [
                "--test-list",
                str(fixture["test_list_path"]),
                "--bound-split",
                str(fixture["bound_split_path"]),
            ]
            for group in fixture["groups"]:
                argv.extend(
                    ["--per-view", f"{group}={fixture['per_view_paths'][group]}"]
                )
                argv.extend(
                    [
                        "--temperature",
                        f"{group}={fixture['temperature_paths'][group]}",
                    ]
                )
            for declaration in self._declarations(fixture):
                argv.extend(["--expected-sha256", declaration])
            argv.extend(["--output", str(output), "--csv", str(csv_path)])
            self.assertEqual(analyze_formal_test_blocks.main(argv), 0)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["protocol"]["groups"], ["L", "A3"])
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 120)

    def test_l_a3_urban20k_96_views_six_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = self._build_fixture(
                Path(temp), blocks=self.SIX_BLOCKS, groups=("L", "A3")
            )
            payload = self._run(fixture)

            self.assertEqual(payload["protocol"]["groups"], ["L", "A3"])
            self.assertEqual(payload["protocol"]["test_views"], 96)
            self.assertEqual(payload["protocol"]["block_count"], 6)
            self.assertEqual(len(payload["blocks"]), 6)
            self.assertNotIn("a3_vs_l_combined_judgment", payload)
            self.assertEqual(
                sum(
                    record["paired_comparisons"]["A3_minus_L"][
                        "combined_non_degraded"
                    ]
                    for record in payload["blocks"]
                ),
                4,
            )

    def test_per_view_and_temperature_group_sets_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = self._build_fixture(
                Path(temp), blocks=self.FOUR_BLOCKS, groups=("L", "A3")
            )
            del fixture["temperature_paths"]["A3"]
            with self.assertRaisesRegex(
                analyze_formal_test_blocks.BlockAnalysisError,
                "group sets must match exactly",
            ):
                self._run(fixture)

    def test_temperature_assignment_and_support_fairness_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = self._build_fixture(Path(temp))
            path = fixture["temperature_paths"]["A3"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["files"][0]["split_assignment"]["block_index"] = 999
            self._write_json(path, payload)
            with self.assertRaisesRegex(
                analyze_formal_test_blocks.BlockAnalysisError,
                "split assignment differs from bound split for A3/0001",
            ):
                self._run(fixture)

        with tempfile.TemporaryDirectory() as temp:
            fixture = self._build_fixture(Path(temp))
            path = fixture["temperature_paths"]["A3"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["files"][0]["mask_sha256"] = "f" * 64
            self._write_json(path, payload)
            with self.assertRaisesRegex(
                analyze_formal_test_blocks.BlockAnalysisError,
                "ground truth/support/split metadata differs for A3/0001",
            ):
                self._run(fixture)

    def test_tolerance_boundaries_are_ties(self) -> None:
        self.assertEqual(
            analyze_formal_test_blocks._classify("PSNR", 20.05, 20.0)[
                "classification"
            ],
            "tied",
        )
        self.assertEqual(
            analyze_formal_test_blocks._classify("LPIPS", 0.203, 0.2)[
                "classification"
            ],
            "tied",
        )
        self.assertEqual(
            analyze_formal_test_blocks._classify("temperature_mae_c", 1.101, 1.0)[
                "classification"
            ],
            "declined",
        )


if __name__ == "__main__":
    unittest.main()
