param(
    [int]$Select = 0,
    [switch]$RegisterMfCamera
)

$ErrorActionPreference = "Stop"

#region Asset & model downloads
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

    # Material Symbols Rounded variable font — drives the GUI's icon
    # glyphs (mode-nav, top-bar actions, status indicators). Single
    # ~3.7 MB file from Google's official material-design-icons repo;
    # static-weight builds aren't published, so the variable font is
    # the only embeddable distribution path.
    $symbolsPath = "$fontDir\MaterialSymbolsRounded.ttf"
    if (Test-Path $symbolsPath) {
        Write-Host "  MaterialSymbolsRounded.ttf already present, skipping" -ForegroundColor Green
    } else {
        $symbolsUrl = "https://github.com/google/material-design-icons/raw/master/variablefont/MaterialSymbolsRounded%5BFILL%2CGRAD%2Copsz%2Cwght%5D.ttf"
        Write-Host "  Downloading MaterialSymbolsRounded.ttf..."
        & curl.exe --fail --silent --show-error --location $symbolsUrl -o $symbolsPath
        if ($LASTEXITCODE -ne 0) {
            throw "curl download failed for MaterialSymbolsRounded.ttf (exit $LASTEXITCODE)"
        }
        Write-Host "  Installed: $symbolsPath" -ForegroundColor Green
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

#endregion

#region MediaFoundation virtual camera
function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Build-MfCameraDll {
    # Static CRT (see Invoke-CargoStaticCrt) so svchost can load the DLL in
    # Session 0 with no VC runtime dependency in its walk.
    Write-Host "Building MediaFoundation virtual camera DLL..." -ForegroundColor Cyan
    Invoke-CargoStaticCrt -CargoArgs @('build', '-p', 'vulvatar-mf-camera')
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

#endregion

#region Code signing
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

#endregion

#region Build & packaging
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
        "assets\MaterialSymbolsRounded.ttf",
        "models\rtmw3d.onnx",
        "models\yolox.onnx",
        "models\face_landmark.onnx",
        "models\face_blendshapes.onnx",
        "THIRD_PARTY_LICENSES.md",
        "docs\USER_GUIDE_JA.md"
    )
    $missing = $required | Where-Object { -not (Test-Path $_) }
    if ($missing) {
        Write-Host "Missing files required for installer build:" -ForegroundColor Red
        $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        throw "Run the 'setup' menu entry first to download fonts + ONNX models, and ensure docs/licenses are present."
    }
}

# Run a script block with the given $env vars temporarily set, restoring
# each to its prior value — or removing it if it was previously unset — on
# exit, so a toolchain-specific build env never leaks into the caller's
# shell. Shared by the crt-static and realsense cargo wrappers below.
function Invoke-WithEnv {
    param(
        [Parameter(Mandatory)] [hashtable]$Vars,
        [Parameter(Mandatory)] [scriptblock]$Script
    )
    $saved = @{}
    foreach ($k in $Vars.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k) }
    try {
        foreach ($k in $Vars.Keys) { Set-Item "Env:$k" -Value $Vars[$k] }
        & $Script
    } finally {
        foreach ($k in $saved.Keys) {
            if ($null -eq $saved[$k]) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
            else { Set-Item "Env:$k" -Value $saved[$k] }
        }
    }
}

# Run cargo with +crt-static appended to RUSTFLAGS. Used for the camera
# DLL so it carries no VC runtime dependency when FrameServer's svchost
# loads it in Session 0.
function Invoke-CargoStaticCrt {
    param([Parameter(Mandatory)] [string[]]$CargoArgs)

    $flags = if ([string]::IsNullOrWhiteSpace($env:RUSTFLAGS)) {
        "-C target-feature=+crt-static"
    } elseif ($env:RUSTFLAGS -notmatch "crt-static") {
        "$env:RUSTFLAGS -C target-feature=+crt-static"
    } else {
        $env:RUSTFLAGS
    }
    Invoke-WithEnv -Vars @{ RUSTFLAGS = $flags } -Script {
        & cargo @CargoArgs
        if ($LASTEXITCODE -ne 0) {
            throw "cargo $($CargoArgs -join ' ') failed (exit $LASTEXITCODE)"
        }
    }
}

# Run cargo with the `realsense` feature appended, wiring up the three
# env vars realsense-sys needs on Windows (see docs/realsense-build.md):
#   * PKG_CONFIG_PATH → our hand-written build-support\pkgconfig\realsense2.pc
#   * LIBCLANG_PATH   → libclang for buildtime-bindgen
#   * PATH            → WinGet pkg-config + the SDK's bin\x64 (realsense2.dll)
# Paths default to the documented install locations and are overridable
# with $env:VULVATAR_REALSENSE_SDK (SDK root) and $env:LIBCLANG_PATH.
function Invoke-CargoRealsense {
    param([Parameter(Mandatory)] [string[]]$CargoArgs)

    $pkgConfig   = Join-Path (Get-Location).Path "build-support\pkgconfig"
    $sdkRoot     = if ($env:VULVATAR_REALSENSE_SDK) { $env:VULVATAR_REALSENSE_SDK }
                   else { Join-Path $env:USERPROFILE "Documents\RealSense SDK 2.0" }
    $sdkBin      = Join-Path $sdkRoot "bin\x64"
    $llvmBin     = if ($env:LIBCLANG_PATH) { $env:LIBCLANG_PATH }
                   else { Join-Path $env:ProgramFiles "LLVM\bin" }
    $wingetLinks = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"

    if (-not (Test-Path (Join-Path $pkgConfig "realsense2.pc"))) {
        throw "build-support\pkgconfig\realsense2.pc missing — see docs/realsense-build.md."
    }
    if (-not (Test-Path $llvmBin)) {
        throw "libclang not found at '$llvmBin'. Install LLVM (winget install LLVM.LLVM) or set `$env:LIBCLANG_PATH. See docs/realsense-build.md."
    }
    if (-not (Test-Path $sdkBin)) {
        throw "RealSense SDK not found at '$sdkBin'. Install it or set `$env:VULVATAR_REALSENSE_SDK to its root. See docs/realsense-build.md."
    }
    if (-not (Get-Command pkg-config -ErrorAction SilentlyContinue) -and
        -not (Test-Path (Join-Path $wingetLinks "pkg-config.exe"))) {
        throw "pkg-config not found (winget install bloodrock.pkg-config-lite). See docs/realsense-build.md."
    }

    Invoke-WithEnv -Vars @{
        PKG_CONFIG_PATH = $pkgConfig
        LIBCLANG_PATH   = $llvmBin
        PATH            = "$wingetLinks;$sdkBin;$($env:PATH)"
    } -Script {
        & cargo @CargoArgs --features realsense
        if ($LASTEXITCODE -ne 0) {
            throw "cargo $($CargoArgs -join ' ') --features realsense failed (exit $LASTEXITCODE)"
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

    # 1. Main app — default CRT linkage with RealSense toolchain.
    #    ORT's prebuilt onnxruntime.lib expects /MD; forcing /MT here
    #    would hit CRT symbol conflicts at link time.
    Write-Host "[1/5] Building release with RealSense toolchain..." -ForegroundColor Cyan
    Invoke-CargoRealsense -CargoArgs @('build', '--release')

    # Copy realsense2.dll from the RealSense SDK to target\release\ so the
    # installer can bundle it alongside vulvatar.exe.
    $sdkRoot = if ($env:VULVATAR_REALSENSE_SDK) { $env:VULVATAR_REALSENSE_SDK }
               else { Join-Path $env:USERPROFILE "Documents\RealSense SDK 2.0" }
    $rsDll = Join-Path $sdkRoot "bin\x64\realsense2.dll"
    if (Test-Path $rsDll) {
        Copy-Item $rsDll "target\release\realsense2.dll" -Force
        Write-Host "  Staged realsense2.dll into target\release\" -ForegroundColor Green
    } else {
        throw "realsense2.dll not found at $rsDll"
    }

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

#endregion

#region Depth capture
# Launch the RealSense D435 depth-capture / calibration utility
# (scripts/depth_capture.py) via uv: live RGB|depth preview, labelled
# `.db3` recording into diagnostics/depth/ for the depth-camera pose
# rebuild. The Python version + deps (pyrealsense2 / opencv / numpy)
# are declared inline in the script (PEP 723); uv resolves them into
# its own cache and runs in an ephemeral env, so nothing touches the
# system Python. First launch downloads Qt essentials + deps (~90 MB),
# cached thereafter.
function Start-DepthCapture {
    $script = "scripts\depth_capture.py"
    if (-not (Test-Path $script)) {
        throw "$script not found — run this from the repo root."
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv not found on PATH. Install it (https://docs.astral.sh/uv/) — it provisions the capture deps from the script's inline PEP 723 metadata."
    }

    Write-Host "Launching D435 capture via uv: 1-6 pick label, space rec, d dump, q/esc quit." -ForegroundColor Cyan
    & uv run --script $script
    if ($LASTEXITCODE -ne 0) {
        # depth_capture.py sys.exit()s with its own message (e.g. no
        # camera); surface the code but don't throw, so the dev menu
        # stays alive for another selection.
        Write-Host "depth_capture.py exited with code $LASTEXITCODE (see message above)." -ForegroundColor Yellow
    }
}

#endregion

#region Dev menu
$commands = @(
    @{ Group = "Setup";          Label = "setup (download pose models + CJK fonts)"; Cmd = "Install-Models; Install-Font" },

    # D435-exclusive build: `realsense` ships in default features and its
    # build.rs needs the pkg-config + LIBCLANG env, so every build/run goes
    # through Invoke-CargoRealsense (sets the env, appends --features realsense).
    # There is no webcam path — no camera means the tracker idles.
    @{ Group = "Build & run (RealSense D435 depth)"; Label = "build (debug)";   Cmd = "Invoke-CargoRealsense -CargoArgs @('build')" },
    @{ Group = "Build & run (RealSense D435 depth)"; Label = "build (release)"; Cmd = "Invoke-CargoRealsense -CargoArgs @('build','--release')" },
    @{ Group = "Build & run (RealSense D435 depth)"; Label = "run (debug)";     Cmd = 'Install-Models; $env:RUST_LOG="vulvatar=info"; Invoke-CargoRealsense -CargoArgs @(''run'')' },
    @{ Group = "Build & run (RealSense D435 depth)"; Label = "run (release)";   Cmd = 'Install-Models; $env:RUST_LOG="vulvatar=info"; Invoke-CargoRealsense -CargoArgs @(''run'',''--release'')' },

    @{ Group = "Camera & depth"; Label = "depth capture / calib data (RealSense D435)"; Cmd = "Start-DepthCapture" },
    @{ Group = "Camera & depth"; Label = "install mf virtual camera (HKLM)"; Cmd = "Install-MfCameraSystem" },
    @{ Group = "Camera & depth"; Label = "uninstall mf virtual camera"; Cmd = "Uninstall-MfCamera" },

    @{ Group = "Packaging"; Label = "package installer (unsigned)";        Cmd = "Build-Distribution" },
    @{ Group = "Packaging"; Label = "package installer (signed, USB token)"; Cmd = "Build-Distribution -Sign" },

    @{ Group = "Cargo utilities"; Label = "test";      Cmd = "cargo test" },
    @{ Group = "Cargo utilities"; Label = "clippy";    Cmd = "cargo clippy" },
    @{ Group = "Cargo utilities"; Label = "fmt";       Cmd = "cargo fmt" },
    @{ Group = "Cargo utilities"; Label = "fmt check"; Cmd = "cargo fmt -- --check" },
    @{ Group = "Cargo utilities"; Label = "check";     Cmd = "cargo check" },
    @{ Group = "Cargo utilities"; Label = "clean";     Cmd = "cargo clean" },
    @{ Group = "Cargo utilities"; Label = "doc";       Cmd = "cargo doc --open" },
    @{ Group = "Cargo utilities"; Label = "update";    Cmd = "cargo update" },
    @{ Group = "Cargo utilities"; Label = "tree";      Cmd = "cargo tree" }
)

function Show-Menu {
    Write-Host ""
    Write-Host "=== VulVATAR dev menu ===" -ForegroundColor Cyan
    $lastGroup = $null
    for ($i = 0; $i -lt $commands.Count; $i++) {
        if ($commands[$i].Group -ne $lastGroup) {
            $lastGroup = $commands[$i].Group
            Write-Host ""
            Write-Host "  $lastGroup" -ForegroundColor DarkCyan
        }
        Write-Host ("   {0,2}. {1}" -f ($i + 1), $commands[$i].Label)
    }
    Write-Host ""
    Write-Host "    0. exit"
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
    # NB: not $input — that is a PowerShell automatic variable (the pipeline
    # enumerator); shadowing it here is a well-known footgun.
    $selection = Read-Host "Select"

    if ($selection -eq "0" -or $selection -eq "q") {
        break
    }

    $idx = 0
    if ([int]::TryParse($selection, [ref]$idx) -and $idx -ge 1 -and $idx -le $commands.Count) {
        Invoke-DevCommand -Index $idx
        Write-Host ""
        Write-Host "Done. Press any key to continue..." -ForegroundColor DarkGray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    } else {
        Write-Host "Invalid selection." -ForegroundColor Red
    }
}

#endregion
