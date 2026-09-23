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
    # Downloads the RTMPose-Face-WFLW LiteRT model for the face sidecar
    # and provisions its Python env. Everything else the runtime loads is
    # produced locally: YOLO26-pose is exported from the repo's .pt via
    # the 'export yolo26-pose ONNX' menu entry, and the RTMPose hand
    # exports + face blendshape MLP are offline-distilled (cannot be
    # downloaded — missing files are noted below).
    #
    # The RTMW3D / YOLOX / MediaPipe model downloads were removed with the
    # backends that consumed them (RTMW3D + YOLOX: YOLO26 migration;
    # MediaPipe: hand 2026-09-18, face 2026-09-22). Offline tooling that
    # still wants them (e.g. the face-distillation teacher, PINTO 410/390)
    # fetches them manually.
    Write-Host "Setting up VulVATAR ONNX models..." -ForegroundColor Cyan
    if (!(Test-Path "models")) {
        New-Item -ItemType Directory -Force -Path "models" | Out-Null
    }

    # Face mesh + blendshape from PINTO_model_zoo were removed with the
    # MediaPipe face backend (2026-09-22); the RTMPose-face sidecar below
    # replaced them.

    # RTMPose-Face-WFLW LiteRT — the default face backend's landmark model
    # (ready-made Apache-2 export from Google's litert-community HF org).
    # It runs in a Python sidecar (scripts/face98_service.py via
    # ai-edge-litert) because the Rust runtime is onnxruntime and cannot
    # load tflite. The distilled blendshape MLP
    # (models/rtmpose-face-blendshape_98.onnx) is a locally trained
    # artifact — without it the sidecar still runs, geometric-only.
    Install-DirectFiles -Name "RTMPose-Face-WFLW LiteRT (face sidecar)" -Files @(
        @{ Url = "https://huggingface.co/litert-community/RTMPose-Face-WFLW-LiteRT/resolve/main/rtm_face_fp16.tflite";
           OutName = "rtm_face_fp16.tflite" }
    )

    # Isolated Python env the sidecar runs in (ai-edge-litert + numpy).
    # The run menu entries export its interpreter as
    # VULVATAR_FACE_SIDECAR_PYTHON; without it the app would spawn bare
    # `python` off PATH, which has no ai-edge-litert.
    Install-FaceSidecarEnv

    if (-not (Test-Path "models\yolo26-pose.onnx") -and
        -not (Test-Path "models\yolo26n-pose_480.onnx")) {
        Write-Host "  WARNING: YOLO26-pose ONNX not exported yet - tracking will not start." -ForegroundColor Yellow
        Write-Host "    Run the 'export yolo26-pose ONNX' menu entry (Setup group) once." -ForegroundColor Yellow
    }

    # Trained artifacts that setup cannot download: the RTMPose hand
    # exports and the face blendshape MLP are distilled offline (see
    # AGENTS.md "Tracking"). Without them the hand chain stays disabled
    # and the face sidecar runs geometric-only expressions.
    foreach ($trained in @(
            "models\rtmpose-m-hand_256.onnx",
            "models\rtmpose-hand-presence_64.onnx",
            "models\rtmpose-hand-palm_256.onnx",
            "models\rtmpose-face-blendshape_98.onnx")) {
        if (-not (Test-Path $trained)) {
            Write-Host "  NOTE: $trained missing (offline-distilled; cannot be downloaded)." -ForegroundColor DarkYellow
        }
    }

    Write-Host "VulVATAR ONNX models installed successfully." -ForegroundColor Green
}

#region Face sidecar & YOLO26 export
# Interpreter path of the face-sidecar venv. Install-FaceSidecarEnv
# provisions it; the run menu entries export it as
# VULVATAR_FACE_SIDECAR_PYTHON so the app spawns the RTMPose-face
# sidecar with ai-edge-litert importable instead of whatever `python`
# is on PATH.
function Get-FaceSidecarPython {
    return Join-Path (Get-Location).Path "tools\face98-venv\Scripts\python.exe"
}

# Create (once) the isolated venv the face landmark sidecar runs in and
# install its two deps. Idempotent: an existing venv with importable
# deps is left untouched, so the per-launch Install-Models call stays
# cheap.
function Install-FaceSidecarEnv {
    $venvPython = Get-FaceSidecarPython
    if (Test-Path $venvPython) {
        & $venvPython -c "import ai_edge_litert, numpy" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  face sidecar venv: already provisioned, skipping" -ForegroundColor Green
            return
        }
    }

    Write-Host "  Provisioning face sidecar venv (tools\face98-venv)..." -ForegroundColor Cyan
    if (-not (Test-Path "tools")) { New-Item -ItemType Directory -Force -Path "tools" | Out-Null }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv "tools\face98-venv"
    } else {
        & py -3 -m venv "tools\face98-venv"
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
        throw "could not create tools\face98-venv (needs python or py -3 on PATH)"
    }
    & $venvPython -m pip install --quiet ai-edge-litert numpy
    if ($LASTEXITCODE -ne 0) {
        throw "pip install ai-edge-litert numpy failed - the face sidecar cannot run without it"
    }
    Write-Host "  face sidecar venv ready: $venvPython" -ForegroundColor Green
}

# One-time ONNX export of the YOLO26-pose body detector. models/ is
# gitignored and the repo carries only the .pt sources, so a fresh
# checkout has no exported detector and tracking fails to start
# (AGENTS.md "Tracking"). ultralytics pulls torch, so the first run of
# this entry is heavyweight; the venv under $TEMP is reused after.
function Export-Yolo26PoseOnnx {
    $venv = Join-Path $env:TEMP "yolo_export_venv"
    $venvPython = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Host "Creating ultralytics venv ($venv) - downloads torch, one-time..." -ForegroundColor Cyan
        if (Get-Command python -ErrorAction SilentlyContinue) {
            & python -m venv $venv
        } else {
            & py -3 -m venv $venv
        }
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
            throw "could not create $venv (needs python or py -3 on PATH)"
        }
        & $venvPython -m pip install --quiet ultralytics onnx onnxslim
        if ($LASTEXITCODE -ne 0) { throw "pip install ultralytics failed" }
    }
    if (-not (Test-Path "models")) { New-Item -ItemType Directory -Force -Path "models" | Out-Null }
    foreach ($weight in @("yolo26n-pose.pt", "yolo26s-pose.pt")) {
        $out = "models\$($weight.Replace('.pt', ''))_480.onnx"
        if (Test-Path $out) {
            Write-Host "  $out already exported, skipping" -ForegroundColor Green
            continue
        }
        if (-not (Test-Path $weight)) {
            Write-Host "  $weight not found in repo root, skipping" -ForegroundColor Yellow
            continue
        }
        Write-Host "  exporting $weight @ imgsz=480 opset=17..." -ForegroundColor Cyan
        & $venvPython -c "from ultralytics import YOLO; YOLO('$weight').export(format='onnx', imgsz=480, opset=17, simplify=True)"
        if ($LASTEXITCODE -ne 0) { throw "ONNX export failed for $weight" }
        Move-Item $weight.Replace('.pt', '.onnx') $out -Force
        Write-Host "  kept $out" -ForegroundColor Green
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
        # Default (RTMPose-face sidecar) face chain: landmark tflite +
        # canonical-mesh anchors + the sidecar script itself. The MediaPipe
        # face/hand ONNX bundles are no longer shipped (runtime removed
        # 2026-09-22); the trained hand exports + blendshape MLP stay
        # offline artifacts and are not installer-prerequisites.
        "models\rtm_face_fp16.tflite",
        "models\mp_canonical478.npy",
        "models\mp_wflw98_idx.json",
        "scripts\face98_service.py",
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
# Generate build-support\pkgconfig\realsense2.pc from the .example template
# (docs/realsense-build.md §"The hand-written realsense2.pc"). The prefix uses
# the 8.3 short path because the default Documents install contains a space,
# which breaks pkg-config / linker arg splitting; Version is read from the
# SDK's own rs.h so bindgen regenerates against the right headers.
function New-Realsense2Pc {
    param(
        [Parameter(Mandatory)] [string]$PkgConfigDir,
        [Parameter(Mandatory)] [string]$SdkRoot
    )

    $example = Join-Path $PkgConfigDir "realsense2.pc.example"
    if (-not (Test-Path $example)) {
        throw "template not found: $example"
    }

    $fso = New-Object -ComObject Scripting.FileSystemObject
    $prefix = $fso.GetFolder($SdkRoot).ShortPath -replace '\\', '/'   # 8.3 path, no spaces
    if ($prefix -eq ($SdkRoot -replace '\\', '/')) {
        # 8.3 generation disabled on this volume — fall back to the real path
        # and let the user rename if the linker chokes on the space.
        Write-Warning "no 8.3 short path for '$SdkRoot'; using the full path in realsense2.pc"
    }

    $rsH = Join-Path $SdkRoot "include\librealsense2\rs.h"
    if (-not (Test-Path $rsH)) { $rsH = Join-Path $SdkRoot "include\librealsense2\h\rs.h" }
    $version = "2.58.2"
    if (Test-Path $rsH) {
        $macros = @{}
        switch -Regex (Get-Content $rsH) {
            '#define\s+RS2_API_(MAJOR|MINOR|PATCH)_VERSION\s+(\d+)' { $macros[$Matches[1]] = $Matches[2] }
        }
        if ($macros.Count -eq 3) { $version = "$($macros['MAJOR']).$($macros['MINOR']).$($macros['PATCH'])" }
    }

    (Get-Content $example) `
        -replace 'prefix=.*', "prefix=$prefix" `
        -replace 'Version:.*', "Version: $version" |
        Set-Content (Join-Path $PkgConfigDir "realsense2.pc")
}

function Invoke-CargoRealsense {
    param([Parameter(Mandatory)] [string[]]$CargoArgs)

    $pkgConfig   = Join-Path (Get-Location).Path "build-support\pkgconfig"
    $sdkRoot     = if ($env:VULVATAR_REALSENSE_SDK) { $env:VULVATAR_REALSENSE_SDK }
                   else { Join-Path $env:USERPROFILE "Documents\RealSense SDK 2.0" }
    $sdkBin      = Join-Path $sdkRoot "bin\x64"
    $llvmBin     = if ($env:LIBCLANG_PATH) { $env:LIBCLANG_PATH }
                   else { Join-Path $env:ProgramFiles "LLVM\bin" }
    $wingetLinks = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"

    if (-not (Test-Path $llvmBin)) {
        throw "libclang not found at '$llvmBin'. Install LLVM (winget install LLVM.LLVM) or set `$env:LIBCLANG_PATH. See docs/realsense-build.md."
    }
    if (-not (Test-Path $sdkBin)) {
        throw "RealSense SDK not found at '$sdkBin'. Install it or set `$env:VULVATAR_REALSENSE_SDK to its root. See docs/realsense-build.md."
    }
    if (-not (Test-Path (Join-Path $pkgConfig "realsense2.pc"))) {
        Write-Host "realsense2.pc missing — generating from realsense2.pc.example (SDK: $sdkRoot)"
        New-Realsense2Pc -PkgConfigDir $pkgConfig -SdkRoot $sdkRoot
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
    @{ Group = "Setup";          Label = "export yolo26-pose ONNX (models/, one-time)"; Cmd = "Export-Yolo26PoseOnnx" },

    # D435-exclusive build: `realsense` ships in default features and its
    # build.rs needs the pkg-config + LIBCLANG env, so every build/run goes
    # through Invoke-CargoRealsense (sets the env, appends --features realsense).
    # There is no webcam path — no camera means the tracker idles.
    @{ Group = "Build & run (RealSense D435 depth)"; Label = "build (debug)";   Cmd = "Invoke-CargoRealsense -CargoArgs @('build')" },
    @{ Group = "Build & run (RealSense D435 depth)"; Label = "build (release)"; Cmd = "Invoke-CargoRealsense -CargoArgs @('build','--release')" },
    @{ Group = "Build & run (RealSense D435 depth)"; Label = "run (debug)";     Cmd = 'Install-Models; $env:VULVATAR_FACE_SIDECAR_PYTHON = (Get-FaceSidecarPython); $env:RUST_LOG="vulvatar=info"; Invoke-CargoRealsense -CargoArgs @(''run'')' },
    @{ Group = "Build & run (RealSense D435 depth)"; Label = "run (release)";   Cmd = 'Install-Models; $env:VULVATAR_FACE_SIDECAR_PYTHON = (Get-FaceSidecarPython); $env:RUST_LOG="vulvatar=info"; Invoke-CargoRealsense -CargoArgs @(''run'',''--release'')' },

    @{ Group = "Camera & depth"; Label = "diagnose realsense (enumerate + stream test)"; Cmd = "Invoke-CargoRealsense -CargoArgs @('run','--bin','diagnose_realsense')" },
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
