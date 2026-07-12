param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$PythonExe = "python",
    [Parameter(Mandatory = $true)][string]$InventoryCsv,
    [Parameter(Mandatory = $true)][string]$GeoRoot,
    [Parameter(Mandatory = $true)][string]$OutRoot,
    [string]$ColmapCmd = "colmap",
    [string[]]$Scenes = @("Building", "PVpanel", "Orchard", "Road", "TransmissionTower"),
    [string]$ThresholdsM = "0.10,0.25,0.50,1.00,2.00,5.00,10.00,20.00,30.00",
    [int]$ResolutionArg = 4,
    [double]$RankThresholdM = 1.00,
    [switch]$EnableAgreementMetrics,
    [switch]$EnableReferenceMeshHoleFix,
    [switch]$UseProbeManifestNativeAlign
)

$ErrorActionPreference = "Stop"

function Assert-Exists {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Label not found: $PathValue"
    }
}

function Write-JsonUtf8 {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$PathValue
    )
    $parent = Split-Path -Parent $PathValue
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    $Object | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $PathValue -Encoding UTF8
}

function Ensure-PointCloudJunction {
    param(
        [Parameter(Mandatory = $true)][string]$ModelRoot
    )
    $junctionPath = Join-Path $ModelRoot "point_cloud"
    if (Test-Path -LiteralPath $junctionPath) {
        return
    }
    $thermalDir = Join-Path $ModelRoot "point_cloud_thermal"
    Assert-Exists -PathValue $thermalDir -Label "Thermal point cloud dir"
    cmd /c mklink /J "$junctionPath" "$thermalDir" | Out-Null
    if (-not (Test-Path -LiteralPath $junctionPath)) {
        throw "Failed to create point_cloud junction for $ModelRoot"
    }
}

function Get-MethodIteration {
    param(
        [Parameter(Mandatory = $true)][string]$MethodName
    )
    if ($MethodName -like "Ours_*") {
        return 60000
    }
    return 30000
}

function Get-MethodOutputName {
    param(
        [Parameter(Mandatory = $true)][string]$MethodName
    )
    return "${MethodName}_full"
}

function Get-StrictProtocolManifest {
    param(
        [Parameter(Mandatory = $true)][string]$SceneName
    )
    return (Join-Path $GeoRoot "$SceneName\M01_Strict_v1\strict_dataset\strict_protocol_manifest.json")
}

function Normalize-PathForCompare {
    param(
        [string]$PathValue
    )
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return ""
    }
    try {
        return ([System.IO.Path]::GetFullPath($PathValue)).TrimEnd("\").ToLowerInvariant()
    }
    catch {
        return ([string]$PathValue).TrimEnd("\").ToLowerInvariant()
    }
}

function Get-NativeCamerasJson {
    param(
        [Parameter(Mandatory = $true)][string]$ModelPath
    )
    $candidate = Join-Path $ModelPath "cameras.json"
    if (Test-Path -LiteralPath $candidate) {
        return $candidate
    }
    $matches = @(Get-ChildItem -LiteralPath $ModelPath -Recurse -Filter "cameras.json" -File -ErrorAction SilentlyContinue | Sort-Object FullName)
    if ($matches.Count -eq 1) {
        return $matches[0].FullName
    }
    if ($matches.Count -gt 1) {
        throw "Multiple cameras.json files found under $ModelPath; add a root-level cameras.json before native-align export."
    }
    throw "Native cameras.json not found for $ModelPath"
}

function Update-BatchStatus {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [string]$SceneName = "",
        [string]$MethodName = "",
        [string]$Note = ""
    )
    $script:BatchStatus.stage = $Stage
    $script:BatchStatus.scene = $SceneName
    $script:BatchStatus.method = $MethodName
    $script:BatchStatus.note = $Note
    $script:BatchStatus.updated_at = (Get-Date).ToString("s")
    Write-JsonUtf8 -Object $script:BatchStatus -PathValue $script:BatchStatusPath
}

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)][string[]]$ArgsList,
        [Parameter(Mandatory = $true)][string]$FailureMessage,
        [string]$WorkingDirectory = ""
    )
    if ($WorkingDirectory) {
        Push-Location $WorkingDirectory
    }
    try {
        & $PythonExe @ArgsList
        if ($LASTEXITCODE -ne 0) {
            throw $FailureMessage
        }
    }
    finally {
        if ($WorkingDirectory) {
            Pop-Location
        }
    }
}

function Invoke-SceneSummary {
    param(
        [Parameter(Mandatory = $true)][string]$SceneOutDir,
        [Parameter(Mandatory = $true)][object[]]$MethodRows
    )
    $summaryDir = Join-Path $SceneOutDir "summary"
    New-Item -ItemType Directory -Force -Path $summaryDir | Out-Null
    $metricsJsons = @()
    foreach ($row in $MethodRows) {
        $methodOutName = Get-MethodOutputName -MethodName ([string]$row.method)
        $metricsPath = Join-Path $SceneOutDir "$methodOutName\evaluation\metrics_summary.json"
        if (Test-Path -LiteralPath $metricsPath) {
            $metricsJsons += $metricsPath
        }
    }
    if ($metricsJsons.Count -eq 0) {
        throw "No metrics_summary.json files found under $SceneOutDir"
    }
    $summaryArgs = @(
        (Join-Path $RepoRoot "tools\geometric_repeatability\summarize_depth_reference_methods.py"),
        "--metrics_json"
    ) + $metricsJsons + @(
        "--out_dir", $summaryDir,
        "--rank_threshold_m", ([string]$RankThresholdM)
    )
    Invoke-PythonChecked -ArgsList $summaryArgs -FailureMessage "Scene summary failed for $SceneOutDir"
}

function Test-BundleManifestMatches {
    param(
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][int]$ExpectedResolutionArg,
        [string]$ExpectedCameraFrameMode = "scene_test",
        [string]$ExpectedProbeCameraManifest = "",
        [string]$ExpectedNativeCamerasJson = ""
    )
    if (-not (Test-Path -LiteralPath $ManifestPath)) {
        return $false
    }
    try {
        $manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
        $actualResolution = $manifest.render_resolution.resolution_arg
        if ($null -eq $actualResolution) {
            return $false
        }
        if ([int]$actualResolution -ne [int]$ExpectedResolutionArg) {
            return $false
        }
        if ($ExpectedCameraFrameMode -eq "probe_manifest_native_align") {
            if ([string]$manifest.camera_frame_mode -ne $ExpectedCameraFrameMode) {
                return $false
            }
            if ($null -eq $manifest.strict_to_native_alignment) {
                return $false
            }
            if ([string]::IsNullOrWhiteSpace([string]$manifest.probe_camera_manifest)) {
                return $false
            }
            if ([string]::IsNullOrWhiteSpace([string]$manifest.native_cameras_json)) {
                return $false
            }
            if ((Normalize-PathForCompare $manifest.probe_camera_manifest) -ne (Normalize-PathForCompare $ExpectedProbeCameraManifest)) {
                return $false
            }
            if ((Normalize-PathForCompare $manifest.native_cameras_json) -ne (Normalize-PathForCompare $ExpectedNativeCamerasJson)) {
                return $false
            }
            $views = @($manifest.views)
            if ($views.Count -eq 0) {
                return $false
            }
            if ($null -eq $views[0].native_camera_to_world) {
                return $false
            }
        }
        return $true
    }
    catch {
        return $false
    }
}

Assert-Exists -PathValue $RepoRoot -Label "Repo root"
Assert-Exists -PathValue $PythonExe -Label "FGS python"
Assert-Exists -PathValue $InventoryCsv -Label "Inventory CSV"
Assert-Exists -PathValue $ColmapCmd -Label "COLMAP command"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$TranscriptPath = Join-Path $OutRoot "run_transcript.txt"
$script:BatchStatusPath = Join-Path $OutRoot "status.json"
Start-Transcript -Path $TranscriptPath -Append -Force | Out-Null
Write-JsonUtf8 -Object ([ordered]@{
    enable_agreement_metrics = [bool]$EnableAgreementMetrics
    enable_reference_mesh_hole_fix = [bool]$EnableReferenceMeshHoleFix
    use_probe_manifest_native_align = [bool]$UseProbeManifestNativeAlign
    reference_mvs_max_image_size = 2000
    reference_patch_match_auto_source_count = 30
    reference_poisson_depth = 12
    reference_poisson_trim = 6
    reference_poisson_point_weight = 6
}) -PathValue (Join-Path $OutRoot "depth_reference_runtime_flags.json")

$script:BatchStatus = [ordered]@{
    protocol = "reference-depth-based-geometric-evaluation-v1"
    stage = "starting"
    scene = ""
    method = ""
    note = ""
    thresholds_m = $ThresholdsM
    resolution_arg = $ResolutionArg
    rank_threshold_m = $RankThresholdM
    enable_agreement_metrics = [bool]$EnableAgreementMetrics
    enable_reference_mesh_hole_fix = [bool]$EnableReferenceMeshHoleFix
    use_probe_manifest_native_align = [bool]$UseProbeManifestNativeAlign
    reference_mvs_max_image_size = 2000
    reference_patch_match_auto_source_count = 30
    reference_poisson_depth = 12
    reference_poisson_trim = 6
    reference_poisson_point_weight = 6
    repo_root = $RepoRoot
    inventory_csv = $InventoryCsv
    out_root = $OutRoot
    started_at = (Get-Date).ToString("s")
    updated_at = (Get-Date).ToString("s")
    transcript = $TranscriptPath
}
Write-JsonUtf8 -Object $script:BatchStatus -PathValue $script:BatchStatusPath

try {
    Copy-Item -LiteralPath $InventoryCsv -Destination (Join-Path $OutRoot "model_inventory_snapshot.csv") -Force
    $inventory = Import-Csv -LiteralPath $InventoryCsv
    $selectedScenes = @($Scenes)
    if ($selectedScenes.Count -eq 0) {
        throw "At least one scene must be provided"
    }

    $globalManifest = [ordered]@{
        protocol = "reference-depth-based-geometric-evaluation-v1"
        thresholds_m = $ThresholdsM
        resolution_arg = $ResolutionArg
        rank_threshold_m = $RankThresholdM
        use_probe_manifest_native_align = [bool]$UseProbeManifestNativeAlign
        reference_parameters = [ordered]@{
            patch_match_max_image_size = 2000
            patch_match_auto_source_count = 30
            patch_match_geom_consistency = 1
            patch_match_filter = 1
            patch_match_window_radius = 7
            patch_match_num_iterations = 7
            stereo_fusion_min_num_pixels = 3
            stereo_fusion_max_reproj_error = 2.0
            stereo_fusion_max_depth_error = 0.02
            stereo_fusion_max_normal_error = 12.0
            mesh_backend_preference = "poisson"
            poisson_depth = 12
            poisson_trim = 6
            poisson_point_weight = 6
        }
        scenes = @()
    }

    foreach ($sceneName in $selectedScenes) {
        Update-BatchStatus -Stage "prepare_scene" -SceneName $sceneName -Note "Resolving strict manifest and model list"
        $strictManifestPath = Get-StrictProtocolManifest -SceneName $sceneName
        Assert-Exists -PathValue $strictManifestPath -Label "Strict protocol manifest for $sceneName"
        $strict = Get-Content -LiteralPath $strictManifestPath -Raw | ConvertFrom-Json
        $strictThermalRoot = [string]$strict.artifacts.strict_thermal_root
        $trainUnionList = [string]$strict.lists.train_union
        $probeList = [string]$strict.lists.probe_test

        Assert-Exists -PathValue $strictThermalRoot -Label "Strict thermal root for $sceneName"
        Assert-Exists -PathValue $trainUnionList -Label "Train union list for $sceneName"
        Assert-Exists -PathValue $probeList -Label "Probe list for $sceneName"

        $sceneRows = @($inventory | Where-Object { $_.scene -eq $sceneName -and $_.ready_for_eval -eq "True" })
        if ($sceneRows.Count -eq 0) {
            throw "No ready_for_eval models found for scene $sceneName"
        }

        $sceneOutDir = Join-Path $OutRoot $sceneName
        $referenceOutDir = Join-Path $sceneOutDir "reference"
        New-Item -ItemType Directory -Force -Path $sceneOutDir | Out-Null

        $referenceManifest = Join-Path $referenceOutDir "reference_depth_manifest.json"
        $probeCameraManifest = Join-Path $referenceOutDir "probe_camera_manifest.json"
        if (-not (Test-Path -LiteralPath $referenceManifest)) {
            Update-BatchStatus -Stage "build_reference" -SceneName $sceneName -Note "$sceneName training-only reference depth artifacts"
            $referenceArgs = @(
                (Join-Path $RepoRoot "tools\geometric_repeatability\build_depth_reference.py"),
                "--strict_protocol_manifest", $strictManifestPath,
                "--out_dir", $referenceOutDir,
                "--colmap_cmd", $ColmapCmd,
                "--resolution_arg", ([string]$ResolutionArg),
                "--thresholds_m", $ThresholdsM,
                "--support_min_count", "1",
                "--support_radius_px", "1",
                "--support_depth_tolerance_m", "0.10",
                "--patch_match_max_image_size", "2000",
                "--patch_match_auto_source_count", "30",
                "--patch_match_window_radius", "7",
                "--patch_match_num_iterations", "7",
                "--patch_match_geom_consistency", "1",
                "--patch_match_filter", "1",
                "--stereo_fusion_min_num_pixels", "3",
                "--stereo_fusion_max_reproj_error", "2.0",
                "--stereo_fusion_max_depth_error", "0.02",
                "--stereo_fusion_max_normal_error", "12.0",
                "--mesh_backend_preference", "poisson",
                "--poisson_depth", "12",
                "--poisson_trim", "6",
                "--poisson_point_weight", "6"
            )
            if ($EnableReferenceMeshHoleFix) {
                $referenceArgs += @(
                    "--mesh_backend_preference", "poisson",
                    "--poisson_depth", "12",
                    "--poisson_trim", "6",
                    "--poisson_point_weight", "6",
                    "--stereo_fusion_min_num_pixels", "3",
                    "--stereo_fusion_max_depth_error", "0.02",
                    "--stereo_fusion_max_reproj_error", "2.0",
                    "--stereo_fusion_max_normal_error", "12.0",
                    "--patch_match_window_radius", "7",
                    "--patch_match_num_iterations", "7",
                    "--patch_match_geom_consistency", "1",
                    "--patch_match_filter", "1",
                    "--patch_match_auto_source_count", "30"
                )
            }
            Invoke-PythonChecked -ArgsList $referenceArgs -FailureMessage "Reference build failed for $sceneName"
        }
        Assert-Exists -PathValue $referenceManifest -Label "Reference depth manifest for $sceneName"
        if ($UseProbeManifestNativeAlign) {
            Assert-Exists -PathValue $probeCameraManifest -Label "Probe camera manifest for $sceneName"
        }

        $sceneManifest = [ordered]@{
            scene = $sceneName
            strict_protocol_manifest = $strictManifestPath
            reference_manifest = $referenceManifest
            strict_thermal_root = $strictThermalRoot
            train_union_list = $trainUnionList
            probe_list = $probeList
            methods = @()
        }

        foreach ($row in $sceneRows) {
            $methodName = [string]$row.method
            $modelPath = [string]$row.model_path
            $methodOutName = Get-MethodOutputName -MethodName $methodName
            $methodOutDir = Join-Path $sceneOutDir $methodOutName
            $bundleOutDir = Join-Path $methodOutDir "bundle"
            $evalOutDir = Join-Path $methodOutDir "evaluation"
            $adapterPath = Join-Path $methodOutDir "depth_adapter_manifest.json"
            $iteration = Get-MethodIteration -MethodName $methodName
            $expectedCameraFrameMode = "scene_test"
            $expectedProbeCameraManifest = ""
            $expectedNativeCamerasJson = ""

            Assert-Exists -PathValue $modelPath -Label "$methodName model path"
            if ((-not (Test-Path -LiteralPath (Join-Path $modelPath "point_cloud"))) -and (Test-Path -LiteralPath (Join-Path $modelPath "point_cloud_thermal"))) {
                Update-BatchStatus -Stage "ensure_point_cloud_alias" -SceneName $sceneName -MethodName $methodName -Note "Creating point_cloud junction for thermal-only layout"
                Ensure-PointCloudJunction -ModelRoot $modelPath
            }

            New-Item -ItemType Directory -Force -Path $methodOutDir | Out-Null
            New-Item -ItemType Directory -Force -Path $evalOutDir | Out-Null

            if (-not (Test-Path -LiteralPath $adapterPath)) {
                $adapterPayload = [ordered]@{
                    protocol_name = "reference-depth-based-geometric-evaluation-v1"
                    method_name = $methodOutName
                    model_path = $modelPath
                    source_path = $strictThermalRoot
                    iteration = $iteration
                    depth_semantics = "inverse_camera_z_from_renderer"
                    validity_rule = [ordered]@{
                        mode = "opacity_threshold"
                        opacity_threshold = 0.5
                        depth_min = 1e-6
                    }
                    notes = "Frozen v1 adapter for formal 5-scene 8-method batch"
                }
                Write-JsonUtf8 -Object $adapterPayload -PathValue $adapterPath
            }

            if ($UseProbeManifestNativeAlign) {
                $expectedCameraFrameMode = "probe_manifest_native_align"
                $expectedProbeCameraManifest = $probeCameraManifest
                $expectedNativeCamerasJson = Get-NativeCamerasJson -ModelPath $modelPath
                Assert-Exists -PathValue $expectedNativeCamerasJson -Label "$methodName native cameras.json"
            }

            $bundleManifest = Join-Path $bundleOutDir "split_manifest.json"
            $bundleWasExported = $false
            if (-not (Test-BundleManifestMatches -ManifestPath $bundleManifest -ExpectedResolutionArg $ResolutionArg -ExpectedCameraFrameMode $expectedCameraFrameMode -ExpectedProbeCameraManifest $expectedProbeCameraManifest -ExpectedNativeCamerasJson $expectedNativeCamerasJson)) {
                if (Test-Path -LiteralPath $bundleOutDir) {
                    Remove-Item -LiteralPath $bundleOutDir -Recurse -Force
                }
                Update-BatchStatus -Stage "export_bundle" -SceneName $sceneName -MethodName $methodName -Note "Exporting held-out probe depth bundle"
                $exportArgs = @(
                    (Join-Path $RepoRoot "tools\geometric_repeatability\export_gaussian_probe_bundle.py"),
                    "--model_path", $modelPath,
                    "--source_path", $strictThermalRoot,
                    "--images", "images",
                    "--resolution", ([string]$ResolutionArg),
                    "--train_list", $trainUnionList,
                    "--test_list", $probeList,
                    "--eval",
                    "--iteration", ([string]$iteration),
                    "--split_label", "heldout_probe",
                    "--scene_name_override", $sceneName,
                    "--out_dir", $bundleOutDir,
                    "--quiet"
                )
                if ($UseProbeManifestNativeAlign) {
                    $exportArgs += @(
                        "--camera_frame_mode", "probe_manifest_native_align",
                        "--probe_camera_manifest", $probeCameraManifest,
                        "--native_cameras_json", $expectedNativeCamerasJson
                    )
                }
                Invoke-PythonChecked -ArgsList $exportArgs -FailureMessage "Bundle export failed for $sceneName / $methodName"
                $bundleWasExported = $true
            }

            $metricsJson = Join-Path $evalOutDir "metrics_summary.json"
            if ($bundleWasExported -and (Test-Path -LiteralPath $evalOutDir)) {
                Remove-Item -LiteralPath $evalOutDir -Recurse -Force
                New-Item -ItemType Directory -Force -Path $evalOutDir | Out-Null
            }
            if (-not (Test-Path -LiteralPath $metricsJson)) {
                Update-BatchStatus -Stage "evaluate_depth_reference" -SceneName $sceneName -MethodName $methodName -Note "Evaluating model depth against reference"
                $evalArgs = @(
                    (Join-Path $RepoRoot "tools\geometric_repeatability\evaluate_depth_reference.py"),
                    "--reference_manifest", $referenceManifest,
                    "--model_manifest", $bundleManifest,
                    "--adapter_manifest", $adapterPath,
                    "--out_dir", $evalOutDir
                )
                if ($EnableAgreementMetrics) {
                    $evalArgs += "--enable_agreement_metrics"
                }
                Invoke-PythonChecked -ArgsList $evalArgs -FailureMessage "Depth-reference evaluation failed for $sceneName / $methodName"
            }

            Assert-Exists -PathValue $metricsJson -Label "Metrics summary for $sceneName / $methodName"
            $sceneManifest.methods += [ordered]@{
                method_name = $methodOutName
                model_path = $modelPath
                iteration = $iteration
                camera_frame_mode = $expectedCameraFrameMode
                probe_camera_manifest = $expectedProbeCameraManifest
                native_cameras_json = $expectedNativeCamerasJson
                bundle_manifest = $bundleManifest
                adapter_manifest = $adapterPath
                metrics_summary = $metricsJson
            }
        }

        Update-BatchStatus -Stage "scene_summary" -SceneName $sceneName -Note "Summarizing all methods for scene"
        Invoke-SceneSummary -SceneOutDir $sceneOutDir -MethodRows $sceneRows

        $sceneManifestPath = Join-Path $sceneOutDir "scene_run_manifest.json"
        Write-JsonUtf8 -Object $sceneManifest -PathValue $sceneManifestPath
        $globalManifest.scenes += [ordered]@{
            scene = $sceneName
            scene_run_manifest = $sceneManifestPath
            summary_dir = (Join-Path $sceneOutDir "summary")
        }
    }

    Write-JsonUtf8 -Object $globalManifest -PathValue (Join-Path $OutRoot "batch_run_manifest.json")
    Update-BatchStatus -Stage "completed" -Note "All requested scenes completed"
}
catch {
    Update-BatchStatus -Stage "failed" -Note $_.Exception.Message
    throw
}
finally {
    Stop-Transcript | Out-Null
}
