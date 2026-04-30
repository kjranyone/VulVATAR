param(
    [int]$Select = 0,
    [switch]$RegisterMfCamera
)

$ErrorActionPreference = "Stop"

function Install-Font {
    # Load JP / KR / SC subsets so the egui font fallback chain covers
    # Hangul (only in KR) and SC-specific glyph forms in addition to
    # Japanese kana/kanji. Without KR, Korean text renders as tofu;
    # without SC, simplified-only characters fall back to JP shapes.
    Write-Host "Setting up CJK fonts (Noto Sans JP/KR/SC)..." -ForegroundColor Cyan
    $fontDir = "assets"

    if (!(Test-Path $fontDir)) {
        New-Item -ItemType Directory -Force -Path $fontDir | Out-Null
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem

    $fonts = @(
        @{ Zip = "16_NotoSansJP.zip"; Otf = "NotoSansJP-Regular.otf" },
        @{ Zip = "17_NotoSansKR.zip"; Otf = "NotoSansKR-Regular.otf" },
        @{ Zip = "18_NotoSansSC.zip"; Otf = "NotoSansSC-Regular.otf" }
    )

    foreach ($f in $fonts) {
        $fontPath = "$fontDir\$($f.Otf)"
        if (Test-Path $fontPath) {
            Write-Host "  $($f.Otf) already present, skipping" -ForegroundColor Green
            continue
        }

        $zipUrl = "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/$($f.Zip)"
        $zipPath = "$fontDir\$($f.Zip)"

        Write-Host "  Downloading $($f.Zip)..."
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath

        $zip = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path $zipPath))
        try {
            $entry = $zip.Entries | Where-Object { $_.Name -eq $f.Otf }
            if ($null -eq $entry) {
                throw "$($f.Otf) not found in $($f.Zip)"
            }
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile(
                $entry,
                (Resolve-Path $fontDir).Path + "\$($f.Otf)",
                $true
            )
        } finally {
            $zip.Dispose()
        }

        Remove-Item $zipPath -Force
        Write-Host "  Installed: $fontPath" -ForegroundColor Green
    }
}

function Install-Models {
    # Pulls RTMW3D-x for body / hands, MediaPipe FaceMeshV2 +
    # BlendshapeV2 for face expression, and YOLOX-m for human-art
    # person detection.
    #
    #   * RTMW3D-x   — Soykaf/RTMW3D-x              (370 MB, 133 3D landmarks)
    #   * FaceMesh   — PINTO 410 FaceMeshV2         (4.8 MB, 478 face landmarks)
    #   * Blendshape — PINTO 390 BlendshapeV2       (1.8 MB, 52 ARKit weights)
    #   * YOLOX-m    — mmpose rtmposev1 onnx_sdk    (94 MB,  human-art bbox)
    #
    # YOLOX-m is optional — without it, RTMW3D runs on the whole frame
    # (Kinemotion-style). With it, we crop to the largest detected
    # person bbox before RTMW3D so small / distant subjects get model-
    # input-resolution treatment.
    #
    # RTMW3D's body keypoints 0..=4 (nose / eyes / ears) are used to
    # crop the face for FaceMesh, so we don't need a separate face
    # detector. The 52 ARKit blendshape coefficients drive the avatar's
    # expression channel directly (with a few VRM 1.0 preset
    # aggregations layered on for stock rigs).
    Write-Host "Setting up VulVATAR ONNX models..." -ForegroundColor Cyan
    if (!(Test-Path "models")) {
        New-Item -ItemType Directory -Force -Path "models" | Out-Null
    }

    Install-DirectFiles -Name "RTMW3D-x whole-body 3D pose" -Files @(
        @{ Url = "https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx";
           OutName = "rtmw3d.onnx" }
    )

    Install-ZipArchive -Name "YOLOX-m human-art person detector" `
        -ArchiveUrl "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip" `
        -KeepGlobs @("end2end.onnx") `
        -RenameMap @{ "end2end.onnx" = "yolox.onnx" }

    # Face mesh + blendshape from PINTO_model_zoo. PINTO ships
    # MediaPipe FaceMeshV2 (478 landmarks) and BlendshapeV2 (52 ARKit
    # weights) as separate ONNX files inside per-project tar.gz
    # archives. The OpenCV HF org does not publish these particular
    # models, so we fall back to PINTO's Wasabi S3 mirror.
    Install-PintoArchive -Name "MediaPipe FaceMeshV2 (478 landmarks)" `
        -ArchiveUrl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/410_FaceMeshV2/resources.tar.gz" `
        -KeepGlobs @("face_landmarks_detector_1x3x256x256.onnx")
    Install-PintoArchive -Name "MediaPipe BlendshapeV2 (52 ARKit blendshapes)" `
        -ArchiveUrl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/390_BlendShapeV2/resources.tar.gz" `
        -KeepGlobs @("face_blendshapes.onnx")

    # Rename to the canonical filename the loader expects.
    if ((Test-Path "models\face_landmarks_detector_1x3x256x256.onnx") -and
        (-not (Test-Path "models\face_landmark.onnx"))) {
        Move-Item "models\face_landmarks_detector_1x3x256x256.onnx" "models\face_landmark.onnx"
        Write-Host "    renamed to face_landmark.onnx" -ForegroundColor Green
    }

    Write-Host "VulVATAR ONNX models installed successfully." -ForegroundColor Green
}

function Install-ExperimentalPoseModels {
    # Optional models for the provider-based CIGPose + metric-depth
    # pipeline. CIGPose is downloaded from the cigpose-onnx release
    # bundle and normalized to models\cigpose.onnx. Metric-depth models
    # vary a lot in size, license, and ONNX packaging, so the URL is
    # supplied by environment variables instead of hardcoded here.
    Write-Host "Setting up experimental pose-provider ONNX models..." -ForegroundColor Cyan
    if (!(Test-Path "models")) {
        New-Item -ItemType Directory -Force -Path "models" | Out-Null
    }

    Install-ZipArchive -Name "CIGPose-x UBody whole-body 2D pose" `
        -ArchiveUrl "https://github.com/namas191297/cigpose-onnx/releases/latest/download/cigpose_models.zip" `
        -KeepGlobs @("cigpose-x_coco-ubody_384x288.onnx") `
        -RenameMap @{ "cigpose-x_coco-ubody_384x288.onnx" = "cigpose.onnx" }

    Install-MetricDepthModelFromEnv

    Write-Host ""
    Write-Host "Experimental provider examples:" -ForegroundColor Cyan
    Write-Host '  $env:VULVATAR_POSE_PROVIDER="cigpose-metric-depth"' -ForegroundColor DarkGray
    Write-Host '  $env:VULVATAR_METRIC_DEPTH_KIND="unidepth-v2"  # or moge / depth-pro / custom' -ForegroundColor DarkGray
    Write-Host '  $env:VULVATAR_METRIC_DEPTH_MODEL="models\unidepth_v2.onnx"' -ForegroundColor DarkGray
}

function Install-MetricDepthModelFromEnv {
    $url = $env:VULVATAR_METRIC_DEPTH_URL
    if ([string]::IsNullOrWhiteSpace($url)) {
        Write-Host "  Metric depth model: no VULVATAR_METRIC_DEPTH_URL set; skipping" -ForegroundColor Yellow
        Write-Host "    Set VULVATAR_METRIC_DEPTH_URL to a UniDepthV2 / MoGe / Depth Pro ONNX URL to download it." -ForegroundColor DarkGray
        Write-Host "    Optional: VULVATAR_METRIC_DEPTH_OUT, VULVATAR_METRIC_DEPTH_DATA_URL, VULVATAR_METRIC_DEPTH_DATA_OUT." -ForegroundColor DarkGray
        return
    }

    $kind = $env:VULVATAR_METRIC_DEPTH_KIND
    if ([string]::IsNullOrWhiteSpace($kind)) {
        $kind = "custom"
    }

    $outName = $env:VULVATAR_METRIC_DEPTH_OUT
    if ([string]::IsNullOrWhiteSpace($outName)) {
        switch -Regex ($kind.ToLowerInvariant()) {
            '^(unidepth|unidepth-v2|unidepthv2)$' { $outName = "unidepth_v2.onnx"; break }
            '^moge$' { $outName = "moge.onnx"; break }
            '^(depth-pro|depthpro)$' { $outName = "depth_pro.onnx"; break }
            default { $outName = "metric_depth.onnx"; break }
        }
    }

    $files = @(
        @{ Url = $url; OutName = $outName }
    )

    $dataUrl = $env:VULVATAR_METRIC_DEPTH_DATA_URL
    if (-not [string]::IsNullOrWhiteSpace($dataUrl)) {
        $dataOut = $env:VULVATAR_METRIC_DEPTH_DATA_OUT
        if ([string]::IsNullOrWhiteSpace($dataOut)) {
            try {
                $dataOut = [System.IO.Path]::GetFileName(([System.Uri]$dataUrl).AbsolutePath)
            } catch {
                $dataOut = "$outName.data"
            }
        }
        $files += @{ Url = $dataUrl; OutName = $dataOut }
    }

    Install-DirectFiles -Name "Metric depth model ($kind)" -Files $files
}

function Select-PoseProvider {
    $current = $env:VULVATAR_POSE_PROVIDER
    if ([string]::IsNullOrWhiteSpace($current)) {
        $current = "rtmw3d"
    }

    Write-Host ""
    Write-Host "Select pose provider for this run:" -ForegroundColor Cyan
    Write-Host "   1. RTMW3D (default/current stable)" -ForegroundColor Green
    Write-Host "   2. CIGPose + metric depth (experimental)" -ForegroundColor Yellow
    Write-Host "   0. keep current: $current" -ForegroundColor DarkGray
    $choice = Read-Host "Provider"

    switch ($choice) {
        "0" {
            Write-Host "Using existing provider: $current" -ForegroundColor DarkGray
        }
        "2" {
            $env:VULVATAR_POSE_PROVIDER = "cigpose-metric-depth"
            Select-MetricDepthKind
            Test-ExperimentalPoseModelFiles
        }
        default {
            $env:VULVATAR_POSE_PROVIDER = "rtmw3d"
            Write-Host "Using pose provider: rtmw3d" -ForegroundColor Green
        }
    }
}

function Select-MetricDepthKind {
    $current = $env:VULVATAR_METRIC_DEPTH_KIND
    if ([string]::IsNullOrWhiteSpace($current)) {
        $current = "unidepth-v2"
    }

    Write-Host ""
    Write-Host "Select metric depth backend:" -ForegroundColor Cyan
    Write-Host "   1. UniDepthV2" -ForegroundColor Green
    Write-Host "   2. MoGe" -ForegroundColor Green
    Write-Host "   3. Depth Pro" -ForegroundColor Green
    Write-Host "   0. keep current: $current" -ForegroundColor DarkGray
    $choice = Read-Host "Metric depth"

    switch ($choice) {
        "0" { $env:VULVATAR_METRIC_DEPTH_KIND = $current }
        "2" { $env:VULVATAR_METRIC_DEPTH_KIND = "moge" }
        "3" { $env:VULVATAR_METRIC_DEPTH_KIND = "depth-pro" }
        default { $env:VULVATAR_METRIC_DEPTH_KIND = "unidepth-v2" }
    }

    Write-Host "Using pose provider: cigpose-metric-depth ($env:VULVATAR_METRIC_DEPTH_KIND)" -ForegroundColor Yellow
}

function Get-DefaultMetricDepthModelPath {
    $kind = $env:VULVATAR_METRIC_DEPTH_KIND
    if ([string]::IsNullOrWhiteSpace($kind)) {
        $kind = "unidepth-v2"
    }

    switch -Regex ($kind.ToLowerInvariant()) {
        '^(unidepth|unidepth-v2|unidepthv2)$' { return "models\unidepth_v2.onnx" }
        '^moge$' { return "models\moge.onnx" }
        '^(depth-pro|depthpro)$' { return "models\depth_pro.onnx" }
        default { return "models\metric_depth.onnx" }
    }
}

function Test-ExperimentalPoseModelFiles {
    $metricModel = $env:VULVATAR_METRIC_DEPTH_MODEL
    if ([string]::IsNullOrWhiteSpace($metricModel)) {
        $metricModel = Get-DefaultMetricDepthModelPath
    }

    $missing = @()
    if (-not (Test-Path "models\cigpose.onnx")) {
        $missing += "models\cigpose.onnx"
    }
    if (-not (Test-Path $metricModel)) {
        $missing += $metricModel
    }

    if ($missing.Count -gt 0) {
        Write-Host "  WARNING: experimental provider model files are missing:" -ForegroundColor Yellow
        $missing | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
        Write-Host "  Run menu entry 'setup experimental pose provider models' or set VULVATAR_*_MODEL paths." -ForegroundColor DarkGray
    }
}

# Download a `.zip` archive, extract to a temp dir, copy ONNX files
# matching `KeepGlobs` into `models\`, optionally renaming via
# `RenameMap`, and remove the temp dir. Used for OpenMMLab mmdeploy
# bundles that ship as zip rather than tar.gz.
function Install-ZipArchive {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$ArchiveUrl,
        [Parameter(Mandatory)] [string[]]$KeepGlobs,
        [hashtable]$RenameMap = @{}
    )

    # Resolve the post-rename target file name for each glob and skip
    # the download if every target is already present.
    $allPresent = $true
    foreach ($glob in $KeepGlobs) {
        $finalName = if ($RenameMap.ContainsKey($glob)) { $RenameMap[$glob] } else { $glob }
        if (-not (Test-Path "models\$finalName")) {
            $allPresent = $false
            break
        }
    }
    if ($allPresent) {
        Write-Host "  ${Name}: already installed, skipping" -ForegroundColor Green
        return
    }

    Write-Host "  Fetching ${Name}..." -ForegroundColor Cyan
    $tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("vulvatar_models_" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
    try {
        $archivePath = Join-Path $tempDir "archive.zip"
        Write-Host "    downloading $ArchiveUrl" -ForegroundColor DarkGray
        & curl.exe --fail --silent --show-error --location $ArchiveUrl -o $archivePath
        if ($LASTEXITCODE -ne 0) {
            throw "curl download failed for $Name (exit $LASTEXITCODE): $ArchiveUrl"
        }

        Write-Host "    extracting..." -ForegroundColor DarkGray
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($archivePath, $tempDir)

        $copied = 0
        foreach ($glob in $KeepGlobs) {
            $matched = Get-ChildItem -Path $tempDir -Recurse -Filter $glob -File
            foreach ($file in $matched) {
                $finalName = if ($RenameMap.ContainsKey($glob)) { $RenameMap[$glob] } else { $file.Name }
                $dest = Join-Path "models" $finalName
                Copy-Item -Path $file.FullName -Destination $dest -Force
                Write-Host "    kept $finalName" -ForegroundColor Green
                $copied++
            }
        }
        if ($copied -eq 0) {
            throw "$Name archive contained no files matching: $($KeepGlobs -join ', ')"
        }
    } finally {
        Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Download a PINTO_model_zoo `resources*.tar.gz` archive, extract to a
# temp dir, copy ONNX files matching `KeepGlobs` into `models\`, and
# remove the temp dir. PINTO archives often bundle 10+ variants per
# model; keeping only the ones we need keeps `models\` lean.
function Install-PintoArchive {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$ArchiveUrl,
        [Parameter(Mandatory)] [string[]]$KeepGlobs
    )

    # If every kept glob already resolves to at least one file in
    # models/, the bundle is already installed — skip the redownload.
    $allPresent = $true
    foreach ($glob in $KeepGlobs) {
        $matched = Get-ChildItem -Path "models\$glob" -ErrorAction SilentlyContinue
        if (-not $matched) {
            $allPresent = $false
            break
        }
    }
    if ($allPresent) {
        Write-Host "  ${Name}: already installed, skipping" -ForegroundColor Green
        return
    }

    Write-Host "  Fetching ${Name}..." -ForegroundColor Cyan
    $tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("vulvatar_models_" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
    try {
        $archivePath = Join-Path $tempDir "resources.tar.gz"
        Write-Host "    downloading $ArchiveUrl" -ForegroundColor DarkGray
        # curl.exe is more reliable than Invoke-WebRequest for archives
        # in the 100MB+ range — IWR's progress UI slows the transfer to a
        # crawl and (on flaky links) can return without writing the full
        # body, leaving tar to error out with "Truncated tar archive".
        # `--fail` exits non-zero on HTTP errors; `-L` follows redirects.
        & curl.exe --fail --silent --show-error --location $ArchiveUrl -o $archivePath
        if ($LASTEXITCODE -ne 0) {
            throw "curl download failed for $Name (exit $LASTEXITCODE): $ArchiveUrl"
        }

        Write-Host "    extracting..." -ForegroundColor DarkGray
        # tar.exe has shipped with Windows 10 1803+ and Windows 11.
        # PowerShell's Expand-Archive does not handle .tar.gz natively.
        $tarOutput = & tar.exe -xzf $archivePath -C $tempDir 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "tar -xzf failed for $Name (exit $LASTEXITCODE): $tarOutput"
        }

        $copied = 0
        foreach ($glob in $KeepGlobs) {
            $matched = Get-ChildItem -Path $tempDir -Recurse -Filter $glob -File
            foreach ($file in $matched) {
                $dest = Join-Path "models" $file.Name
                Copy-Item -Path $file.FullName -Destination $dest -Force
                Write-Host "    kept $($file.Name)" -ForegroundColor Green
                $copied++
            }
        }
        if ($copied -eq 0) {
            throw "$Name archive contained no files matching: $($KeepGlobs -join ', ')"
        }
    } finally {
        Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Download individual .onnx files directly. Used for sources that
# publish pre-built ONNX as release assets (no archive unpacking).
function Install-DirectFiles {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [array]$Files
    )

    Write-Host "  Fetching ${Name}..." -ForegroundColor Cyan
    foreach ($f in $Files) {
        $dest = Join-Path "models" $f.OutName
        if (Test-Path $dest) {
            Write-Host "    $($f.OutName): already installed, skipping" -ForegroundColor Green
            continue
        }
        Write-Host "    downloading $($f.Url)" -ForegroundColor DarkGray
        & curl.exe --fail --silent --show-error --location $f.Url -o $dest
        if ($LASTEXITCODE -ne 0) {
            throw "curl download failed for $($f.OutName) (exit $LASTEXITCODE): $($f.Url)"
        }
        Write-Host "    kept $($f.OutName)" -ForegroundColor Green
    }
}

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Build-MfCameraDll {
    Write-Host "Building MediaFoundation virtual camera DLL..." -ForegroundColor Cyan
    $oldRustFlags = $env:RUSTFLAGS
    try {
        if ([string]::IsNullOrWhiteSpace($oldRustFlags)) {
            $env:RUSTFLAGS = "-C target-feature=+crt-static"
        } elseif ($oldRustFlags -notmatch "crt-static") {
            $env:RUSTFLAGS = "$oldRustFlags -C target-feature=+crt-static"
        }
        cargo build -p vulvatar-mf-camera
        if ($LASTEXITCODE -ne 0) {
            throw "cargo build -p vulvatar-mf-camera failed (exit $LASTEXITCODE)"
        }
    } finally {
        if ($null -eq $oldRustFlags) {
            Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
        } else {
            $env:RUSTFLAGS = $oldRustFlags
        }
    }
}

# Split from Install-MfCameraSystem so the HKLM/icacls/FrameServer work
# can run in a separate elevated PowerShell without re-running cargo
# build as Admin (which would leave target/ files owned by Admin and
# break subsequent non-elevated cargo invocations).
function Register-MfCameraSystem {
    if (!(Test-Admin)) {
        throw "Register-MfCameraSystem must be invoked from an elevated PowerShell."
    }

    $clsid = "{B5F1C320-2B8F-4A9C-9BDC-43B0E8E6B2E1}"
    $friendly = "VulVATAR Virtual Camera"
    $source = Resolve-Path "target\debug\vulvatar_mf_camera.dll"
    $runtime = Join-Path $env:ProgramFiles "VulVATAR\VirtualCamera"
    New-Item -ItemType Directory -Force -Path $runtime | Out-Null

    $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $targetDir = Resolve-Path $runtime
    $target = Join-Path $targetDir "vulvatar_mf_camera_system_$timestamp.dll"
    Copy-Item $source $target -Force

    # Frame Server runs the camera-source proxy under NT AUTHORITY\LocalService.
    # Default Program Files ACL only grants BUILTIN\Users (which does NOT
    # include LocalService) read+execute on inherited files, so without an
    # explicit grant Frame Server's svchost cannot LoadLibrary the DLL
    # and clients see ReadSample return MF_SOURCE_READERF_ENDOFSTREAM
    # immediately. Grant LocalService RX on every install — the alternative
    # is installing outside Program Files which loses the system-managed
    # uninstall semantics.
    Write-Host "Granting NT AUTHORITY\LocalService:(RX) on installed DLL..." -ForegroundColor Cyan
    Write-Host "  target: $target" -ForegroundColor DarkGray
    $aclOutput = icacls.exe $target /grant 'NT AUTHORITY\LocalService:(RX)' 2>&1 | Out-String
    Write-Host $aclOutput.TrimEnd() -ForegroundColor DarkGray
    if ($LASTEXITCODE -ne 0) {
        throw "icacls grant LocalService failed (exit $LASTEXITCODE). Frame Server will not be able to load the DLL."
    }
    # Verify the explicit grant landed. icacls input syntax accepts
    # `LocalService` (no space) but Windows displays the resolved name as
    # `LOCAL SERVICE` (with space, S-1-5-19). Match either form.
    $verify = (icacls.exe $target | Out-String)
    if ($verify -notmatch 'LOCAL\s*SERVICE') {
        throw "icacls reported success but the LOCAL SERVICE grant is not visible on $target.`nicacls output:`n$verify"
    }
    Write-Host "LOCAL SERVICE grant confirmed." -ForegroundColor Green

    # Frame-buffer directory. vulvatar.exe (Medium-integrity, no
    # SeCreateGlobalPrivilege under UAC filtering) cannot create
    # `Global\` named sections, so the producer / consumer instead
    # share frames through a file-backed mapping at
    #   $env:ProgramData\VulVATAR\camera_frame_buffer.bin
    # Inherited LocalService:(RX) on the directory means whatever file
    # vulvatar.exe creates inside automatically inherits read access
    # for the MF camera DLL hosted in FrameServer's svchost.
    $bufferDir = Join-Path $env:ProgramData "VulVATAR"
    Write-Host "Setting up frame-buffer dir: $bufferDir" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $bufferDir | Out-Null
    $aclDirOutput = icacls.exe $bufferDir /grant 'NT AUTHORITY\LocalService:(OI)(CI)(RX)' 2>&1 | Out-String
    Write-Host $aclDirOutput.TrimEnd() -ForegroundColor DarkGray
    if ($LASTEXITCODE -ne 0) {
        throw "icacls grant LocalService on $bufferDir failed (exit $LASTEXITCODE). MF camera DLL will not be able to read the frame buffer."
    }

    $key = "HKLM:\Software\Classes\CLSID\$clsid"
    $inproc = Join-Path $key "InprocServer32"
    $legacyUserKey = "HKCU:\Software\Classes\CLSID\$clsid"
    if (Test-Path $legacyUserKey) {
        Remove-Item $legacyUserKey -Recurse -Force
        Write-Host "Removed legacy per-user COM registration at $legacyUserKey" -ForegroundColor DarkGray
    }
    New-Item -Force -Path $key | Out-Null
    New-Item -Force -Path $inproc | Out-Null
    Set-Item -Path $key -Value $friendly
    Set-Item -Path $inproc -Value $target
    New-ItemProperty -Path $inproc -Name ThreadingModel -Value Both -PropertyType String -Force | Out-Null

    Write-Host "Registered $friendly in HKLM:" -ForegroundColor Green
    Write-Host "  $target"

    # Restart FrameServer so it drops any cached IClassFactory pointing
    # at the previously-installed DLL. Without this step, svchost keeps
    # the old DLL `LoadLibrary`-held and serves it to new clients even
    # though HKLM now points at the fresh timestamped copy — clients
    # like Google Meet would keep seeing the old behaviour (e.g. the
    # missing-IMF2DBuffer all-green NV12 bug) until the next reboot.
    # Stopping the service kicks out any active camera consumer, so
    # close Meet/Chrome/Camera app first if you want a graceful test.
    Write-Host "Restarting Windows Camera Frame Server (FrameServer)..." -ForegroundColor Cyan
    try {
        $svc = Get-Service -Name 'FrameServer' -ErrorAction Stop
        if ($svc.Status -eq 'Running') {
            Stop-Service -Name 'FrameServer' -Force -ErrorAction Stop
            Write-Host "  stopped" -ForegroundColor DarkGray
        }
        Start-Service -Name 'FrameServer' -ErrorAction Stop
        Write-Host "  started — svchost will load the new DLL on the next ActivateObject." -ForegroundColor Green
    } catch {
        Write-Host ("  WARNING: could not restart FrameServer: " + $_.Exception.Message) -ForegroundColor Yellow
        Write-Host "  Reboot or run 'sc stop FrameServer; sc start FrameServer' as admin to pick up the new DLL." -ForegroundColor Yellow
    }

    Write-Host "You can now run VulVATAR without elevation until you rebuild/reinstall this DLL." -ForegroundColor DarkGray
}

function Install-MfCameraSystem {
    # Always build in the caller's (non-elevated) shell so target/ stays
    # owned by the developer. Only the registration step needs Admin.
    Build-MfCameraDll

    if (Test-Admin) {
        Register-MfCameraSystem
        return
    }

    Write-Host ""
    Write-Host "Registration step requires Administrator privileges." -ForegroundColor Yellow
    Write-Host "A UAC prompt will appear; accept it to continue." -ForegroundColor Yellow

    $scriptPath = $PSCommandPath
    if ([string]::IsNullOrWhiteSpace($scriptPath)) {
        throw "Cannot determine script path for elevation — run dev.ps1 from its file path, not dot-sourced."
    }
    $workDir = (Get-Location).Path
    $escapedScript = $scriptPath.Replace("'", "''")
    $escapedWork = $workDir.Replace("'", "''")
    # -NoExit keeps the elevated window open so the user can read icacls
    # / FrameServer output and any errors. They close it manually.
    $inner = "Set-Location -LiteralPath '$escapedWork'; & '$escapedScript' -RegisterMfCamera"
    try {
        Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-NoExit",
            "-Command", $inner
        ) -ErrorAction Stop | Out-Null
    } catch {
        throw "Failed to launch elevated registration: $($_.Exception.Message). If you cancelled the UAC prompt, re-run this menu entry and accept it."
    }

    Write-Host ""
    Write-Host "An elevated PowerShell window is now running the registration step." -ForegroundColor Green
    Write-Host "Check that window for icacls / HKLM / FrameServer output; close it when done." -ForegroundColor DarkGray
}

function Uninstall-MfCamera {
    Write-Host "Removing VulVATAR virtual camera COM registration..." -ForegroundColor Cyan
    $clsid = "{B5F1C320-2B8F-4A9C-9BDC-43B0E8E6B2E1}"
    $keys = @("HKCU:\Software\Classes\CLSID\$clsid")
    if (Test-Admin) {
        $keys += "HKLM:\Software\Classes\CLSID\$clsid"
    } else {
        Write-Host "Not elevated; HKLM removal will be skipped." -ForegroundColor Yellow
    }

    foreach ($key in $keys) {
        if (Test-Path $key) {
            Remove-Item $key -Recurse -Force
            Write-Host "Removed $key" -ForegroundColor Green
        } else {
            Write-Host "No registration found at $key (nothing to do)" -ForegroundColor DarkGray
        }
    }
    # The registration the main app owns is MFVirtualCameraLifetime_Session,
    # so any IMFVirtualCamera disappears with its process — nothing else to
    # clean up.
}

# Locate signtool.exe. Prefer PATH; fall back to the latest x64 build
# under the Windows 10 SDK install. Returns the absolute path or throws.
function Find-SignTool {
    $cmd = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $sdkRoot = "${env:ProgramFiles(x86)}\Windows Kits\10\bin"
    if (Test-Path $sdkRoot) {
        $candidates = Get-ChildItem -Path $sdkRoot -Recurse -Filter signtool.exe -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like '*\x64\signtool.exe' } |
            Sort-Object FullName -Descending
        if ($candidates) { return $candidates[0].FullName }
    }
    throw "signtool.exe not found. Install the Windows 10/11 SDK (or add signtool.exe to PATH)."
}

# Locate iscc.exe (Inno Setup compiler). Returns the absolute path or throws.
function Find-IsccTool {
    $cmd = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\iscc.exe",
        "${env:ProgramFiles}\Inno Setup 6\iscc.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    throw "iscc.exe not found. Install Inno Setup 6 from https://jrsoftware.org/isinfo.php (or add iscc.exe to PATH)."
}

# Find the (single) currently-valid code-signing certificate. Sectigo
# USB tokens (SafeNet eToken / Authentic Key) surface their cert in
# CurrentUser\My through the eToken CSP/CNG when plugged in, so the
# cert appearing in the store is exactly equivalent to "token is
# inserted and unlocked at the OS layer". Filtering by EKU
# 1.3.6.1.5.5.7.3.3 (Code Signing) keeps SSL/email certs on the same
# token from being picked accidentally.
function Find-CodeSigningCertificate {
    $now = Get-Date
    $codeSigningOid = '1.3.6.1.5.5.7.3.3'
    foreach ($store in @('Cert:\CurrentUser\My', 'Cert:\LocalMachine\My')) {
        $certs = Get-ChildItem -Path $store -ErrorAction SilentlyContinue | Where-Object {
            $_.HasPrivateKey -and
            $_.NotAfter -gt $now -and
            $_.NotBefore -lt $now -and
            (($_.EnhancedKeyUsageList | Where-Object { $_.ObjectId -eq $codeSigningOid }) -ne $null)
        }
        if ($certs) { return @($certs)[0] }
    }
    return $null
}

# Block until a USB code-signing token is detected, or fail after
# $TimeoutSeconds. Polling the cert store at 2s intervals is cheap
# (Get-ChildItem on a small store) and avoids any reliance on
# WMI / Win32_USB device-arrival events.
function Wait-ForCodeSigningCertificate {
    param([int]$TimeoutSeconds = 120)

    $cert = Find-CodeSigningCertificate
    if ($cert) {
        Write-Host "Code-signing cert: $($cert.Subject)" -ForegroundColor Green
        Write-Host "  thumbprint: $($cert.Thumbprint)" -ForegroundColor DarkGray
        Write-Host "  not after:  $($cert.NotAfter)" -ForegroundColor DarkGray
        return $cert
    }

    Write-Host "Insert the code-signing USB token. Polling for ${TimeoutSeconds}s..." -ForegroundColor Yellow
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        $cert = Find-CodeSigningCertificate
        if ($cert) {
            Write-Host "Detected: $($cert.Subject)" -ForegroundColor Green
            Write-Host "  thumbprint: $($cert.Thumbprint)" -ForegroundColor DarkGray
            return $cert
        }
    }
    throw "No code-signing certificate found in CurrentUser\My or LocalMachine\My within ${TimeoutSeconds}s. Is the USB token plugged in?"
}

# Sign one or more files with the supplied cert. signtool accepts
# multiple files in a single invocation, which lets the SafeNet PIN
# dialog appear once for the whole batch (the developer types the
# PIN once and signtool reuses the unlocked private key handle for
# every file in the list).
function Invoke-SignTool {
    param(
        [Parameter(Mandatory)] [string]$SignTool,
        [Parameter(Mandatory)] $Cert,
        [Parameter(Mandatory)] [string[]]$Files
    )

    $timestampUrl = if ($env:VULVATAR_SIGN_TIMESTAMP_URL) {
        $env:VULVATAR_SIGN_TIMESTAMP_URL
    } else {
        'http://timestamp.sectigo.com'
    }

    $signArgs = @(
        'sign',
        '/sha1', $Cert.Thumbprint,
        '/tr',   $timestampUrl,
        '/td',   'sha256',
        '/fd',   'sha256',
        '/v'
    ) + $Files

    Write-Host "  signtool sign /sha1 $($Cert.Thumbprint) /tr $timestampUrl /td sha256 /fd sha256 ..." -ForegroundColor DarkGray
    foreach ($f in $Files) { Write-Host "    $f" -ForegroundColor DarkGray }
    & $SignTool @signArgs
    if ($LASTEXITCODE -ne 0) {
        throw "signtool failed (exit $LASTEXITCODE). If the SafeNet PIN dialog timed out or was cancelled, re-run."
    }
}

# Verify the asset/model files installer\vulvatar.iss copies into the
# install image are present. cargo handles the binaries; the asset
# pipeline (fonts + ONNX) is a manual `setup` step the developer has
# to have run once. Failing fast here beats iscc reporting "file not
# found" 30s into the compile.
function Test-DistributionPrereqs {
    $required = @(
        "assets\NotoSansJP-Regular.otf",
        "assets\NotoSansKR-Regular.otf",
        "assets\NotoSansSC-Regular.otf",
        "models\rtmw3d.onnx",
        "models\yolox.onnx",
        "models\face_landmark.onnx",
        "models\face_blendshapes.onnx"
    )
    $missing = $required | Where-Object { -not (Test-Path $_) }
    if ($missing) {
        Write-Host "Missing files required for installer build:" -ForegroundColor Red
        $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        throw "Run the 'setup' menu entry first to download fonts + ONNX models."
    }
}

# Run cargo with $env:RUSTFLAGS extended to include +crt-static, then
# restore RUSTFLAGS. Used for the camera DLL so it has no VC runtime
# dependency when FrameServer's svchost loads it.
function Invoke-CargoStaticCrt {
    param([Parameter(Mandatory)] [string[]]$CargoArgs)

    $oldRustFlags = $env:RUSTFLAGS
    try {
        if ([string]::IsNullOrWhiteSpace($oldRustFlags)) {
            $env:RUSTFLAGS = "-C target-feature=+crt-static"
        } elseif ($oldRustFlags -notmatch "crt-static") {
            $env:RUSTFLAGS = "$oldRustFlags -C target-feature=+crt-static"
        }
        & cargo @CargoArgs
        if ($LASTEXITCODE -ne 0) {
            throw "cargo $($CargoArgs -join ' ') failed (exit $LASTEXITCODE)"
        }
    } finally {
        if ($null -eq $oldRustFlags) {
            Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
        } else {
            $env:RUSTFLAGS = $oldRustFlags
        }
    }
}

# Build the redistributable installer. With `-Sign`, Authenticode-sign
# the release binaries and the produced setup .exe via signtool +
# whichever code-signing cert is currently in the user's cert store
# (typically a Sectigo USB token). Two PIN prompts total: one for the
# binary batch, one for the installer.
function Build-Distribution {
    param([switch]$Sign)

    Test-DistributionPrereqs
    $iscc = Find-IsccTool

    if ($Sign) {
        $signtool = Find-SignTool
        $cert     = Wait-ForCodeSigningCertificate
    }

    # 1. Main app — default CRT linkage. ORT's prebuilt onnxruntime.lib
    #    expects /MD; forcing /MT here would hit CRT symbol conflicts at
    #    link time.
    Write-Host "[1/5] cargo build --release..." -ForegroundColor Cyan
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build --release failed (exit $LASTEXITCODE)" }

    # 2. Camera DLL — static CRT. svchost loads this in Session 0 with
    #    its own DLL search rules; static CRT removes any VC runtime
    #    dependency from the dependency walk.
    Write-Host "[2/5] Rebuilding vulvatar-mf-camera with +crt-static..." -ForegroundColor Cyan
    Invoke-CargoStaticCrt -CargoArgs @('build', '--release', '-p', 'vulvatar-mf-camera')

    # 3. Sign release binaries (single signtool call → one PIN prompt).
    if ($Sign) {
        Write-Host "[3/5] Signing release binaries (PIN prompt 1/2)..." -ForegroundColor Cyan
        Invoke-SignTool -SignTool $signtool -Cert $cert -Files @(
            "target\release\vulvatar.exe",
            "target\release\vulvatar_mf_camera.dll"
        )
    } else {
        Write-Host "[3/5] Skipping binary signing (unsigned build)." -ForegroundColor DarkGray
    }

    # 4. Build the installer. iscc resolves OutputDir relative to the
    #    .iss location, so output lands in installer\output\.
    Write-Host "[4/5] Compiling Inno Setup installer..." -ForegroundColor Cyan
    & $iscc "installer\vulvatar.iss"
    if ($LASTEXITCODE -ne 0) { throw "iscc failed (exit $LASTEXITCODE)" }

    $setupExe = Get-ChildItem "installer\output\VulVATAR-Setup-*.exe" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $setupExe) { throw "Setup .exe not found in installer\output\." }

    # 5. Sign the installer itself.
    if ($Sign) {
        Write-Host "[5/5] Signing installer (PIN prompt 2/2)..." -ForegroundColor Cyan
        Invoke-SignTool -SignTool $signtool -Cert $cert -Files @($setupExe.FullName)
        Write-Host "Signed installer: $($setupExe.FullName)" -ForegroundColor Green
    } else {
        Write-Host "[5/5] Unsigned installer: $($setupExe.FullName)" -ForegroundColor Yellow
    }
}

$commands = @(
    @{ Label = "setup (download pose models + CJK font)"; Cmd = "Install-Models; Install-Font" },
    @{ Label = "build (debug)";    Cmd = "cargo build" },
    @{ Label = "build (release)";  Cmd = "cargo build --release" },
    @{ Label = "run (debug)";      Cmd = 'Select-PoseProvider; $env:RUST_LOG="vulvatar=info"; cargo run' },
    @{ Label = "run (debug+lipsync)"; Cmd = 'Select-PoseProvider; $env:RUST_LOG="vulvatar=info"; cargo run --features lipsync' },
    @{ Label = "run (release)";    Cmd = "Select-PoseProvider; cargo run --release" },
    @{ Label = "setup experimental pose provider models"; Cmd = "Install-ExperimentalPoseModels" },
    @{ Label = "install mf virtual camera (HKLM)"; Cmd = "Install-MfCameraSystem" },
    @{ Label = "uninstall mf virtual camera"; Cmd = "Uninstall-MfCamera" },
    @{ Label = "package installer (unsigned)";        Cmd = "Build-Distribution" },
    @{ Label = "package installer (signed, USB token)"; Cmd = "Build-Distribution -Sign" },
    @{ Label = "test";             Cmd = "cargo test" },
    @{ Label = "clippy";           Cmd = "cargo clippy" },
    @{ Label = "fmt";              Cmd = "cargo fmt" },
    @{ Label = "fmt check";        Cmd = "cargo fmt -- --check" },
    @{ Label = "check";            Cmd = "cargo check" },
    @{ Label = "clean";            Cmd = "cargo clean" },
    @{ Label = "doc";              Cmd = "cargo doc --open" },
    @{ Label = "update";           Cmd = "cargo update" },
    @{ Label = "tree";             Cmd = "cargo tree" }
)

function Show-Menu {
    Write-Host ""
    Write-Host "=== VulVATAR dev menu ===" -ForegroundColor Cyan
    for ($i = 0; $i -lt $commands.Count; $i++) {
        Write-Host ("  {0,2}. {1}" -f ($i + 1), $commands[$i].Label)
    }
    Write-Host "   0. exit"
    Write-Host ""
}

function Invoke-DevCommand {
    param([int]$Index)

    if ($Index -lt 1 -or $Index -gt $commands.Count) {
        throw "Invalid selection: $Index"
    }

    $cmd = $commands[$Index - 1].Cmd
    Write-Host "> $cmd" -ForegroundColor Yellow
    Write-Host ""
    Invoke-Expression $cmd
}

if ($RegisterMfCamera) {
    # Entry point used by the elevated PowerShell spawned from
    # Install-MfCameraSystem when the outer menu was not running as Admin.
    Register-MfCameraSystem
    return
}

if ($Select -ne 0) {
    Invoke-DevCommand -Index $Select
    return
}

# When dot-sourced (e.g. `. .\dev.ps1; Install-Font`) skip the
# interactive menu loop — sourcing must only register the helper
# functions. Detected via $MyInvocation.InvocationName == '.'.
if ($MyInvocation.InvocationName -eq '.') {
    return
}

while ($true) {
    Show-Menu
    $input = Read-Host "Select"

    if ($input -eq "0" -or $input -eq "q") {
        break
    }

    $idx = 0
    if ([int]::TryParse($input, [ref]$idx) -and $idx -ge 1 -and $idx -le $commands.Count) {
        Invoke-DevCommand -Index $idx
        Write-Host ""
        Write-Host "Done. Press any key to continue..." -ForegroundColor DarkGray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    } else {
        Write-Host "Invalid selection." -ForegroundColor Red
    }
}
