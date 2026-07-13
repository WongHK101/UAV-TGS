from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import build_depth_reference as backend
import evaluate_depth_reference as evaluator
import package_depth_reference_mask_nomask_v2 as mask_packager
import package_depth_reference_results as result_packager


def _fake_stage_log(stage: dict, extra: str = "") -> str:
    if stage.get("cuda_evidence_device") is None:
        return "Interface complete\n" + extra
    text = f"CUDA device {int(stage['cuda_evidence_device'])} initialized: Fake GPU\n"
    if stage.get("stage") == "refine_mesh":
        text += backend.OPENMVS_REFINE_CUDA_FAIL_CLOSED_MARKER + "\n"
    return text + extra


def _write_images_binary(path: Path, names: list[str]) -> None:
    payload = bytearray(struct.pack("<Q", len(names)))
    for image_id, name in enumerate(names, 1):
        payload.extend(
            struct.pack(
                "<idddddddi",
                image_id,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1,
            )
        )
        payload.extend(name.encode("utf-8") + b"\x00")
        payload.extend(struct.pack("<Q", 0))
    path.write_bytes(payload)


def _fixture(root: Path) -> tuple[Path, Path, list[Path]]:
    source = root / "source"
    (source / "images").mkdir(parents=True)
    (source / "images" / "0001.jpg").write_bytes(b"image")
    sparse = source / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.bin").write_bytes(b"model")
    (sparse / "points3D.bin").write_bytes(b"model")
    _write_images_binary(sparse / "images.bin", ["0001.jpg"])
    thermal = root / "thermal"
    thermal.mkdir()
    train_list = root / "train.txt"
    probe_list = root / "probe.txt"
    train_list.write_text("0001.jpg\n", encoding="utf-8")
    probe_list.write_text("0002.jpg\n", encoding="utf-8")
    manifest = root / "strict.json"
    manifest.write_text(
        json.dumps(
            {
                "scene_name": "ToyScene",
                "artifacts": {
                    "train_union_source_root": str(source),
                    "strict_thermal_root": str(thermal),
                },
                "lists": {
                    "train_union": str(train_list),
                    "probe_test": str(probe_list),
                },
            }
        ),
        encoding="utf-8",
    )
    tools = []
    for name in ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh"):
        path = root / name
        payload = b"fake"
        if name == "RefineMesh":
            payload += b"\n" + backend.OPENMVS_REFINE_CUDA_FAIL_CLOSED_MARKER.encode("utf-8")
        path.write_bytes(payload)
        path.chmod(0o755)
        tools.append(path)
    return manifest, root / "dry_out", tools


def _test_cli_dry_run() -> None:
    with tempfile.TemporaryDirectory(prefix="openmvs_reference_dry_") as tmp:
        manifest, out_dir, tools = _fixture(Path(tmp))
        script = Path(__file__).with_name("build_depth_reference.py")
        cmd = [
            sys.executable,
            str(script),
            "--strict_protocol_manifest",
            str(manifest),
            "--out_dir",
            str(out_dir),
            "--openmvs_interface_colmap_cmd",
            str(tools[0]),
            "--openmvs_densify_cmd",
            str(tools[1]),
            "--openmvs_reconstruct_mesh_cmd",
            str(tools[2]),
            "--openmvs_refine_mesh_cmd",
            str(tools[3]),
            "--dry_run",
        ]
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise AssertionError(f"Dry-run CLI failed:\n{completed.stdout}\n{completed.stderr}")
        if "OPENMVS_REFERENCE_DRY_RUN_OK" not in completed.stdout:
            raise AssertionError(f"Dry-run success marker missing:\n{completed.stdout}")
        for stage in ("interface_colmap", "densify_point_cloud", "reconstruct_mesh", "refine_mesh"):
            if f'"stage": "{stage}"' not in completed.stdout:
                raise AssertionError(f"Dry-run command plan is missing stage {stage}")
        forbidden = ("patch_match_stereo", "stereo_fusion", "poisson_mesher", "delaunay_mesher")
        if any(token in completed.stdout for token in forbidden):
            raise AssertionError(f"Dry-run unexpectedly contains a COLMAP-MVS command:\n{completed.stdout}")
        if out_dir.exists():
            raise AssertionError(f"Dry-run must not create its output directory: {out_dir}")

        tools[3].write_bytes(b"stock-openmvs-refiner-without-fail-closed-marker")
        unpatched = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if unpatched.returncode == 0 or "lacks the required CUDA fail-closed marker" not in (
            unpatched.stdout + unpatched.stderr
        ):
            raise AssertionError("Unpatched RefineMesh binary passed the fail-closed preflight")

        skip_cmd = list(cmd)
        refine_option_idx = skip_cmd.index("--openmvs_refine_mesh_cmd")
        skip_cmd[refine_option_idx + 1] = str(Path(tmp) / "missing_RefineMesh")
        skip_cmd.insert(-1, "--skip_openmvs_refine_mesh")
        skipped = subprocess.run(skip_cmd, check=False, capture_output=True, text=True)
        if skipped.returncode != 0:
            raise AssertionError(
                "Dry-run with explicitly skipped RefineMesh must not resolve/hash its executable:\n"
                f"{skipped.stdout}\n{skipped.stderr}"
            )
        payload_text = skipped.stdout.split("OPENMVS_REFERENCE_DRY_RUN_OK", 1)[0].strip()
        payload = json.loads(payload_text)
        if "refine_mesh" in payload["openmvs_executable_sha256"]:
            raise AssertionError("Skipped RefineMesh executable was unexpectedly hashed")
        refine_stage = next(stage for stage in payload["commands"] if stage["stage"] == "refine_mesh")
        if bool(refine_stage["enabled"]):
            raise AssertionError("RefineMesh command is enabled despite --skip_openmvs_refine_mesh")


def _test_refine_failure_has_no_fallback() -> None:
    with tempfile.TemporaryDirectory(prefix="openmvs_no_fallback_") as tmp:
        root = Path(tmp)
        args = backend._build_argparser().parse_args(
            ["--strict_protocol_manifest", str(root / "unused.json"), "--out_dir", str(root / "out")]
        )
        paths = backend._openmvs_paths(root / "out")
        paths["workspace"].mkdir(parents=True)
        executables = {
            "interface_colmap": "InterfaceCOLMAP",
            "densify": "DensifyPointCloud",
            "reconstruct_mesh": "ReconstructMesh",
            "refine_mesh": "RefineMesh",
        }
        plan = backend._build_openmvs_command_plan(
            args,
            paths=paths,
            executables=executables,
            colmap_binary_model=True,
        )
        original_runner = backend._run_openmvs_stage

        def fake_runner(stage, *, cwd, log_path):
            del cwd
            if stage["stage"] == "refine_mesh":
                raise RuntimeError("intentional refine failure")
            for output in stage["required_outputs"]:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"artifact")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(_fake_stage_log(stage), encoding="utf-8")

        backend._run_openmvs_stage = fake_runner
        try:
            try:
                backend._run_openmvs_pipeline(
                    args,
                    paths=paths,
                    command_plan=plan,
                    out_dir=root / "out",
                    plan_sha256="a" * 64,
                )
            except RuntimeError as exc:
                if "intentional refine failure" not in str(exc):
                    raise
            else:
                raise AssertionError("RefineMesh failure was silently accepted")
        finally:
            backend._run_openmvs_stage = original_runner


def _test_openmvs_v24_contract_and_cuda_guard() -> None:
    with tempfile.TemporaryDirectory(prefix="openmvs_v24_contract_") as tmp:
        root = Path(tmp)
        args = backend._build_argparser().parse_args(
            ["--strict_protocol_manifest", str(root / "unused.json"), "--out_dir", str(root / "out")]
        )
        backend._validate_args(args)
        paths = backend._openmvs_paths(root / "out")
        executables = {
            "interface_colmap": "InterfaceCOLMAP",
            "densify": "DensifyPointCloud",
            "reconstruct_mesh": "ReconstructMesh",
            "refine_mesh": "RefineMesh",
        }
        plan = backend._build_openmvs_command_plan(
            args,
            paths=paths,
            executables=executables,
            colmap_binary_model=True,
        )
        by_stage = {stage["stage"]: stage for stage in plan}
        for stage in plan:
            command = stage["command"]
            archive_idx = command.index("--archive-type")
            if command[archive_idx + 1] != "-1":
                raise AssertionError(f"OpenMVS v2.4 stage is not interface-archive mode: {stage}")

        reconstruct = by_stage["reconstruct_mesh"]
        reconstruct_cmd = reconstruct["command"]
        if reconstruct["required_outputs"] != [str(paths["mesh_ply"])]:
            raise AssertionError(f"ReconstructMesh output contract is wrong: {reconstruct}")
        if reconstruct_cmd[reconstruct_cmd.index("--output-file") + 1] != str(paths["mesh_ply"]):
            raise AssertionError(f"ReconstructMesh must output mesh PLY directly: {reconstruct_cmd}")

        refine = by_stage["refine_mesh"]
        refine_cmd = refine["command"]
        if refine["required_outputs"] != [str(paths["refined_ply"])]:
            raise AssertionError(f"RefineMesh output contract is wrong: {refine}")
        expected_refine_args = {
            "--input-file": str(paths["dense_mvs"]),
            "--mesh-file": str(paths["mesh_ply"]),
            "--output-file": str(paths["refined_ply"]),
        }
        for option, expected in expected_refine_args.items():
            if refine_cmd[refine_cmd.index(option) + 1] != expected:
                raise AssertionError(f"RefineMesh {option} contract mismatch: {refine_cmd}")

        for stage_name in ("densify_point_cloud", "reconstruct_mesh", "refine_mesh"):
            if by_stage[stage_name]["cuda_evidence_device"] != 0:
                raise AssertionError(f"CUDA evidence is not required for {stage_name}")
        if not by_stage["refine_mesh"].get("cuda_fallback_fail_closed"):
            raise AssertionError("RefineMesh is not declared fail-closed for CUDA fallback")
        try:
            backend._validate_openmvs_cuda_log(
                by_stage["refine_mesh"],
                "CUDA device 0 initialized: Fake GPU\n",
                log_path=root / "missing-refine-marker.log",
            )
        except RuntimeError as exc:
            if "fail-closed CUDA completion" not in str(exc):
                raise
        else:
            raise AssertionError("RefineMesh CUDA initialization was accepted without fail-closed completion")

        invalid_args = backend._build_argparser().parse_args(
            [
                "--strict_protocol_manifest",
                str(root / "unused.json"),
                "--out_dir",
                str(root / "out"),
                "--openmvs_cuda_device",
                "-1",
            ]
        )
        try:
            backend._validate_args(invalid_args)
        except ValueError as exc:
            if "non-negative CUDA device" not in str(exc):
                raise
        else:
            raise AssertionError("Automatic/fallback-capable CUDA device -1 was accepted")

        output = root / "cuda_stage_output.ply"
        log_path = root / "cuda_stage.log"
        success_stage = {
            "stage": "densify_point_cloud",
            "command": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path({str(output)!r}).write_bytes(b'ok'); "
                    "print('CUDA device 0 initialized: Fake GPU')"
                ),
            ],
            "required_outputs": [str(output)],
            "cuda_evidence_device": 0,
        }
        backend._run_openmvs_stage(success_stage, cwd=root, log_path=log_path)

        for bad_log in (
            "no CUDA initialization line",
            "CUDA device 0 initialized: Fake GPU\nfalling back to CPU",
            "CUDA error: device [999] is not a valid GPU device",
            "CUDA device 0 initialized: Fake GPU\nCUDA unavailable",
            "CUDA device 0 initialized: Fake GPU\nCPU-only build",
        ):
            try:
                backend._validate_openmvs_cuda_log(success_stage, bad_log, log_path=log_path)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"Invalid CUDA log was accepted: {bad_log!r}")

        partial_output = root / "nonzero_partial.ply"
        nonzero_stage = {
            "stage": "interface_colmap",
            "command": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import sys; "
                    f"Path({str(partial_output)!r}).write_bytes(b'partial'); sys.exit(7)"
                ),
            ],
            "required_outputs": [str(partial_output)],
            "cuda_evidence_device": None,
        }
        try:
            backend._run_openmvs_stage(nonzero_stage, cwd=root, log_path=root / "nonzero.log")
        except RuntimeError as exc:
            if "exit code 7" not in str(exc):
                raise
        else:
            raise AssertionError("Non-zero OpenMVS stage exit was silently accepted")
        if partial_output.exists():
            raise AssertionError("Partial output survived a non-zero OpenMVS stage exit")


def _test_cached_stages_require_cuda_evidence() -> None:
    with tempfile.TemporaryDirectory(prefix="openmvs_cached_cuda_") as tmp:
        root = Path(tmp)
        out_dir = root / "out"
        args = backend._build_argparser().parse_args(
            ["--strict_protocol_manifest", str(root / "unused.json"), "--out_dir", str(out_dir)]
        )
        paths = backend._openmvs_paths(out_dir)
        paths["workspace"].mkdir(parents=True)
        plan = backend._build_openmvs_command_plan(
            args,
            paths=paths,
            executables={
                "interface_colmap": "InterfaceCOLMAP",
                "densify": "DensifyPointCloud",
                "reconstruct_mesh": "ReconstructMesh",
                "refine_mesh": "RefineMesh",
            },
            colmap_binary_model=True,
        )
        plan_sha256 = "b" * 64
        for stage in plan:
            for output in stage["required_outputs"]:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"cached")
            log_path = out_dir / "logs" / f"openmvs_{stage['stage']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(_fake_stage_log(stage), encoding="utf-8")
            backend._write_stage_receipt(
                stage,
                out_dir=out_dir,
                plan_sha256=plan_sha256,
                log_path=log_path,
            )

        original_runner = backend._run_openmvs_stage

        def unexpected_runner(*_args, **_kwargs):
            raise AssertionError("Complete cached stages should not rerun")

        backend._run_openmvs_stage = unexpected_runner
        try:
            _, mesh_path, mesh_backend = backend._run_openmvs_pipeline(
                args,
                paths=paths,
                command_plan=plan,
                out_dir=out_dir,
                plan_sha256=plan_sha256,
            )
            if mesh_path != paths["refined_ply"] or mesh_backend != "openmvs_refine_mesh":
                raise AssertionError("Valid cached OpenMVS pipeline returned the wrong mesh")
            evidence = backend._collect_openmvs_cuda_evidence(plan, out_dir=out_dir)
            expected_stages = {"densify_point_cloud", "reconstruct_mesh", "refine_mesh"}
            if set(evidence["stages"]) != expected_stages:
                raise AssertionError(f"CUDA evidence manifest has wrong stages: {evidence}")
            for row in evidence["stages"].values():
                if row["expected_cuda_device"] != 0 or len(row["log_sha256"]) != 64:
                    raise AssertionError(f"CUDA evidence manifest row is incomplete: {row}")

        finally:
            backend._run_openmvs_stage = original_runner

        repaired: list[str] = []
        require_downstream_preinvalidation = [False]

        def repair_runner(stage, *, cwd, log_path):
            del cwd
            repaired.append(str(stage["stage"]))
            if require_downstream_preinvalidation[0] and stage["stage"] == "densify_point_cloud":
                if any(paths["workspace"].rglob("*.dmap")):
                    raise AssertionError("Stale OpenMVS depth-map cache survived until densification began")
                for downstream_stage in plan[2:]:
                    if backend._stage_receipt_path(out_dir, downstream_stage).exists():
                        raise AssertionError(
                            "Downstream receipt survived until after an upstream replacement began"
                        )
                    if any(Path(value).exists() for value in downstream_stage["required_outputs"]):
                        raise AssertionError(
                            "Downstream output survived until after an upstream replacement began"
                        )
            for output in stage["required_outputs"]:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(f"repaired-{stage['stage']}".encode("utf-8"))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(_fake_stage_log(stage), encoding="utf-8")

        backend._run_openmvs_stage = repair_runner
        try:
            bad_log = out_dir / "logs" / "openmvs_refine_mesh.log"
            bad_log.write_text(
                "CUDA device 0 initialized: Fake GPU\nfalling back to CPU\n",
                encoding="utf-8",
            )
            backend._run_openmvs_pipeline(
                args,
                paths=paths,
                command_plan=plan,
                out_dir=out_dir,
                plan_sha256=plan_sha256,
            )
            if repaired != ["refine_mesh"]:
                raise AssertionError(f"Invalid cached CUDA log did not rebuild exactly its stage: {repaired}")

            repaired.clear()
            bad_log.unlink()
            backend._run_openmvs_pipeline(
                args,
                paths=paths,
                command_plan=plan,
                out_dir=out_dir,
                plan_sha256=plan_sha256,
            )
            if repaired != ["refine_mesh"]:
                raise AssertionError(f"Missing cached CUDA log did not rebuild exactly its stage: {repaired}")

            repaired.clear()
            paths["dense_ply"].write_bytes(b"tampered")
            (paths["workspace"] / "depth0001.dmap").write_bytes(b"stale-partial-cache")
            require_downstream_preinvalidation[0] = True
            backend._run_openmvs_pipeline(
                args,
                paths=paths,
                command_plan=plan,
                out_dir=out_dir,
                plan_sha256=plan_sha256,
            )
            if repaired != ["densify_point_cloud", "reconstruct_mesh", "refine_mesh"]:
                raise AssertionError(f"Tampered upstream output did not invalidate downstream stages: {repaired}")
            require_downstream_preinvalidation[0] = False
        finally:
            backend._run_openmvs_stage = original_runner


def _test_cuda_fallback_terminates_immediately() -> None:
    with tempfile.TemporaryDirectory(prefix="openmvs_immediate_cuda_guard_") as tmp:
        root = Path(tmp)
        output = root / "partial.ply"
        dmap_cache = root / "depth0001.dmap"
        log_path = root / "stage.log"
        stage = {
            "stage": "densify_point_cloud",
            "command": [
                sys.executable,
                "-u",
                "-c",
                (
                    "from pathlib import Path; import time; "
                    f"Path({str(output)!r}).write_bytes(b'partial'); "
                    f"Path({str(dmap_cache)!r}).write_bytes(b'partial-cache'); "
                    "print('CUDA error: device [999] is not a valid GPU device', flush=True); "
                    "time.sleep(20)"
                ),
            ],
            "required_outputs": [str(output)],
            "cuda_evidence_device": 0,
            "cache_cleanup_root": str(root),
            "cache_cleanup_globs": ["*.dmap"],
        }
        started = time.monotonic()
        try:
            backend._run_openmvs_stage(stage, cwd=root, log_path=log_path)
        except RuntimeError as exc:
            if "terminated immediately" not in str(exc):
                raise
        else:
            raise AssertionError("OpenMVS CUDA fallback was silently accepted")
        elapsed = time.monotonic() - started
        if elapsed >= 12.0:
            raise AssertionError(f"OpenMVS CUDA fallback was not stopped promptly: {elapsed:.3f}s")
        if output.exists():
            raise AssertionError("Partial OpenMVS output survived immediate CUDA-fallback termination")
        if dmap_cache.exists():
            raise AssertionError("Partial OpenMVS depth-map cache survived CUDA-fallback termination")


def _test_probe_camera_in_sparse_model_is_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="openmvs_sparse_leakage_") as tmp:
        manifest, _, _ = _fixture(Path(tmp))
        strict = json.loads(manifest.read_text(encoding="utf-8"))
        source = Path(strict["artifacts"]["train_union_source_root"])
        train_list = Path(strict["lists"]["train_union"])
        probe_list = Path(strict["lists"]["probe_test"])
        _write_images_binary(source / "sparse" / "0" / "images.bin", ["0002.jpg"])
        try:
            backend._validate_training_only_partition(source, train_list, probe_list)
        except RuntimeError as exc:
            if "outside train_union" not in str(exc):
                raise
        else:
            raise AssertionError("Probe camera in sparse model was silently accepted")


def _test_formal_launcher_revalidates_reference_on_resume() -> None:
    launcher = Path(__file__).with_name("run_depth_reference_formal_5scene_8method.ps1")
    source = launcher.read_text(encoding="utf-8")
    if 'if (-not (Test-Path -LiteralPath $referenceManifest))' in source:
        raise AssertionError(
            "Formal launcher still trusts reference_depth_manifest existence without "
            "re-entering the OpenMVS builder's source/plan/CUDA validation"
        )
    invocation = 'Invoke-PythonChecked -ArgsList $referenceArgs'
    if source.count(invocation) != 1:
        raise AssertionError("Formal launcher must invoke the OpenMVS reference builder exactly once per scene")
    for token in (
        "Test-MetricsManifestMatches",
        "reference_manifest_sha256",
        "model_manifest_sha256",
        "adapter_manifest_sha256",
        "-ExpectedModelPath $modelPath",
        "-ExpectedTrainList $trainUnionList",
        "-ExpectedTestList $probeList",
        'if ([string]$manifest.camera_frame_mode -ne $ExpectedCameraFrameMode)',
        "producer_identity",
        "ExpectedExporterSha256",
        "ExpectedEvaluatorSha256",
        "requires a clean Git worktree",
    ):
        if token not in source:
            raise AssertionError(f"Formal launcher does not bind resumed metrics to current inputs: {token}")


def _test_result_packager_rejects_non_openmvs_reference() -> None:
    with tempfile.TemporaryDirectory(prefix="openmvs_packager_guard_") as tmp:
        scene_root = Path(tmp) / "ToyScene"
        reference_root = scene_root / "reference_openmvs_v1"
        reference_root.mkdir(parents=True)
        build_path = reference_root / "reference_build_manifest.json"
        reference_path = reference_root / "reference_depth_manifest.json"
        mesh_path = reference_root / "mesh.ply"
        mesh_path.write_bytes(b"mesh")
        dense_path = reference_root / "dense.ply"
        dense_path.write_bytes(b"dense")
        plan_path = reference_root / "openmvs_command_plan.json"
        plan_path.write_text("{}", encoding="utf-8")
        receipt_rows = {}
        for stage_name in ("interface_colmap", "densify_point_cloud", "reconstruct_mesh", "refine_mesh"):
            receipt_path = reference_root / f"{stage_name}.success.json"
            receipt_path.write_text(json.dumps({"stage": stage_name, "status": "complete"}), encoding="utf-8")
            receipt_rows[stage_name] = {
                "path": str(receipt_path),
                "sha256": result_packager._sha256_file(receipt_path),
                "size_bytes": receipt_path.stat().st_size,
            }
        view_path = reference_root / "views" / "probe.npz"
        view_path.parent.mkdir()
        view_path.write_bytes(b"view")
        evidence_stages = {}
        for stage_name in ("densify_point_cloud", "reconstruct_mesh", "refine_mesh"):
            log_path = reference_root / f"{stage_name}.log"
            log_text = "CUDA device 0 initialized: Fake GPU\n"
            if stage_name == "refine_mesh":
                log_text += backend.OPENMVS_REFINE_CUDA_FAIL_CLOSED_MARKER + "\n"
            log_path.write_text(log_text, encoding="utf-8")
            evidence_stages[stage_name] = {
                "expected_cuda_device": 0,
                "log_path": str(log_path),
                "log_sha256": result_packager._sha256_file(log_path),
                "log_size_bytes": log_path.stat().st_size,
                "cuda_fallback_fail_closed": stage_name != "reconstruct_mesh",
            }
        valid_build = {
            "scene_name": "ToyScene",
            "reference_construction_protocol": "openmvs-reference-mesh-v1",
            "reference_dense_backend": "openmvs_densify_point_cloud",
            "reference_mesh_backend": "openmvs_refine_mesh",
            "reference_mesh_path": str(mesh_path),
            "reference_mesh_sha256": result_packager._sha256_file(mesh_path),
            "reference_mesh_size_bytes": mesh_path.stat().st_size,
            "reference_dense_ply": str(dense_path),
            "reference_dense_ply_sha256": result_packager._sha256_file(dense_path),
            "reference_dense_ply_size_bytes": dense_path.stat().st_size,
            "openmvs_command_plan": str(plan_path),
            "openmvs_command_plan_sha256": result_packager._sha256_file(plan_path),
            "openmvs_stage_receipts": receipt_rows,
            "reference_construction_overrides": {
                "reference_geometry_backend": "openmvs",
                "colmap_mvs_fallback_allowed": False,
                "openmvs_archive_type": -1,
                "openmvs_interface_normalize": False,
                "openmvs_cuda_device": 0,
                "openmvs_cuda_log_evidence_required": True,
                "openmvs_refine_mesh": True,
                "openmvs_refine_cuda_fail_closed_required": True,
                "openmvs_refine_cuda_fail_closed_marker": (
                    backend.OPENMVS_REFINE_CUDA_FAIL_CLOSED_MARKER
                ),
            },
            "openmvs_cuda_evidence": {
                "status": "verified",
                "stages": evidence_stages,
            },
        }
        build_path.write_text(json.dumps(valid_build), encoding="utf-8")
        reference_path.write_text(
            json.dumps(
                {
                    "reference_construction_protocol": "openmvs-reference-mesh-v1",
                    "reference_mesh_path": str(mesh_path),
                    "reference_mesh_sha256": result_packager._sha256_file(mesh_path),
                    "views": [
                        {
                            "npz_file": "views/probe.npz",
                            "npz_sha256": result_packager._sha256_file(view_path),
                            "npz_size_bytes": view_path.stat().st_size,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        resolved = result_packager._validate_openmvs_reference("ToyScene", scene_root)
        if resolved != reference_path.resolve():
            raise AssertionError("Result packager resolved the wrong OpenMVS reference manifest")

        invalid_build = dict(valid_build)
        invalid_build["reference_construction_protocol"] = "legacy-colmap-mvs"
        build_path.write_text(json.dumps(invalid_build), encoding="utf-8")
        try:
            result_packager._validate_openmvs_reference("ToyScene", scene_root)
        except RuntimeError as exc:
            if "Non-OpenMVS" not in str(exc):
                raise
        else:
            raise AssertionError("Result packager accepted a legacy COLMAP-MVS reference")


def _test_mask_nomask_packager_recomputes_current_openmvs_metrics() -> None:
    source = Path(__file__).with_name("package_depth_reference_mask_nomask_v2.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "reference_openmvs_v1",
        "recompute_metrics_from_bundles(refs)",
        "assert_metrics_match_current_inputs(refs)",
        "reference_manifest_sha256",
        "openmvs_archive_type",
        "OPENMVS_REFINE_CUDA_FAIL_CLOSED_MARKER",
    ):
        if token not in source:
            raise AssertionError(f"Mask/no-mask packager is missing current OpenMVS metric binding: {token}")
    if "def copy_metrics_from_v1" in source:
        raise AssertionError("Mask/no-mask packager still has an old-metric copy path")


def _test_nomask_view_identity_is_rewritten_and_enforced() -> None:
    with tempfile.TemporaryDirectory(prefix="openmvs_nomask_identity_") as tmp:
        root = Path(tmp)
        masked_root = root / "masked"
        view_path = masked_root / "views" / "0001.npz"
        view_path.parent.mkdir(parents=True)
        import numpy as np

        np.savez_compressed(
            view_path,
            depth=np.asarray([[1.0]], dtype=np.float64),
            support_count=np.asarray([[1]], dtype=np.int32),
            valid_mask=np.asarray([[1]], dtype=np.uint8),
            inside_roi=np.asarray([[1]], dtype=np.uint8),
        )
        masked_manifest = masked_root / "reference_depth_manifest.json"
        masked_manifest.write_text(
            json.dumps(
                {
                    "views": [
                        {
                            "image_name": "0001.jpg",
                            "npz_file": "views/0001.npz",
                            "npz_size_bytes": int(view_path.stat().st_size),
                            "npz_sha256": backend._sha256_file(view_path),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        old_out = mask_packager.OUT
        try:
            mask_packager.OUT = root / "package"
            nomask_manifest, _, _ = mask_packager.create_nomask_reference("ToyScene", masked_manifest)
        finally:
            mask_packager.OUT = old_out
        nomask = json.loads(nomask_manifest.read_text(encoding="utf-8"))
        nomask_view = nomask["views"][0]
        verified = evaluator._verified_view_npz(nomask_manifest, nomask_view, label="No-mask")
        if int(nomask_view["npz_size_bytes"]) != int(verified.stat().st_size):
            raise AssertionError("No-mask view size was not refreshed")
        verified.write_bytes(verified.read_bytes() + b"tampered")
        try:
            evaluator._verified_view_npz(nomask_manifest, nomask_view, label="No-mask")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Evaluator accepted a tampered no-mask view NPZ")


def _test_result_packager_rejects_stale_metric_input_hashes() -> None:
    with tempfile.TemporaryDirectory(prefix="openmvs_metric_binding_") as tmp:
        root = Path(tmp)
        scene_root = root / "ToyScene"
        method_root = scene_root / "ToyMethod"
        reference = scene_root / "reference_openmvs_v1" / "reference_depth_manifest.json"
        model = method_root / "bundle" / "split_manifest.json"
        adapter = method_root / "depth_adapter_manifest.json"
        metrics = method_root / "evaluation" / "metrics_summary.json"
        for path in (reference, model, adapter):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        def producer_identity(script_name: str) -> dict:
            script_path = Path(__file__).with_name(script_name).resolve()
            return {
                "script_path": str(script_path),
                "script_sha256": result_packager._sha256_file(script_path),
                "git_commit": result_packager._current_git_commit(),
                "git_dirty": False,
                "git_error": "",
            }

        model.write_text(
            json.dumps(
                {
                    "producer_identity": producer_identity("export_gaussian_probe_bundle.py"),
                }
            ),
            encoding="utf-8",
        )
        metrics.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol_name": "reference-depth-based-geometric-evaluation-v1",
            "producer_identity": producer_identity("evaluate_depth_reference.py"),
            "scene_name": "ToyScene",
            "method_name": "ToyMethod",
            "reference_manifest": str(reference),
            "reference_manifest_sha256": result_packager._sha256_file(reference),
            "model_manifest": str(model),
            "model_manifest_sha256": result_packager._sha256_file(model),
            "adapter_manifest": str(adapter),
            "adapter_manifest_sha256": result_packager._sha256_file(adapter),
            "evaluation_options": {"enable_agreement_metrics": True},
            "counts": {
                "reference_valid_pixels": 1,
                "model_valid_on_reference_pixels": 1,
                "missing_pixels": 0,
            },
            "secondary_metrics": {name: 0.0 for name in result_packager.SECONDARY_METRICS},
            "threshold_metrics": [
                {
                    "threshold_m": 0.1,
                    **{name: 0.0 for name in result_packager.THRESHOLD_METRICS},
                }
            ],
        }
        metrics.write_text(json.dumps(payload), encoding="utf-8")
        old_scenes = result_packager.SCENE_ORDER
        old_methods = result_packager.METHOD_ORDER
        try:
            result_packager.SCENE_ORDER = ["ToyScene"]
            result_packager.METHOD_ORDER = ["ToyMethod"]
            result_packager._collect_metrics(
                {"ToyScene": scene_root},
                {"ToyScene": reference},
            )
            payload["reference_manifest_sha256"] = "0" * 64
            metrics.write_text(json.dumps(payload), encoding="utf-8")
            try:
                result_packager._collect_metrics(
                    {"ToyScene": scene_root},
                    {"ToyScene": reference},
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError("Result packager accepted stale metric input hashes")
        finally:
            result_packager.SCENE_ORDER = old_scenes
            result_packager.METHOD_ORDER = old_methods


def main() -> None:
    _test_cli_dry_run()
    _test_refine_failure_has_no_fallback()
    _test_openmvs_v24_contract_and_cuda_guard()
    _test_cached_stages_require_cuda_evidence()
    _test_cuda_fallback_terminates_immediately()
    _test_probe_camera_in_sparse_model_is_rejected()
    _test_formal_launcher_revalidates_reference_on_resume()
    _test_result_packager_rejects_non_openmvs_reference()
    _test_mask_nomask_packager_recomputes_current_openmvs_metrics()
    _test_nomask_view_identity_is_rewritten_and_enforced()
    _test_result_packager_rejects_stale_metric_input_hashes()
    print("OPENMVS_REFERENCE_BACKEND_SANITY_OK")


if __name__ == "__main__":
    main()
