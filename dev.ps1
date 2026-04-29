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
    # Pulls the MediaPipe Holistic ONNX bundle. Sourced from the OpenCV
    # Hugging Face org because they publish small (~5 MB each), single-
    # file ONNX exports of MediaPipe's pose / palm / hand-landmark
    # models — far leaner than PINTO_model_zoo's `resources*.tar.gz`
    # bundles which inflate to 100s of MB by including every precision
    # × backend variant.
    #
    #   * Body  — opencv/pose_estimation_mediapipe       (5.5 MB, 33 3D landmarks)
    #   * Palm  — opencv/palm_detection_mediapipe        (3.9 MB, hand bbox)
    #   * Hand  — opencv/handpose_estimation_mediapipe   (3.9 MB, 21 3D landmarks)
    #
    # Replaces the old CIGPose (~1 GB) + YOLOX (~10 MB) bundle. Total
    # download is ~13 MB.
    #
    # Face landmarker / blendshapes are deferred to P3 — there is no
    # first-party-quality face-mesh ONNX export on Hugging Face yet, so
    # we will fetch from PINTO 282 (Wasabi mirror) when that phase
    # lands. Until then the Head bone runs from face-pose Euler angles
    # that the body landmarker derives from the head/shoulder joints.
    Write-Host "Setting up MediaPipe Holistic ONNX models..." -ForegroundColor Cyan
    if (!(Test-Path "models")) {
        New-Item -ItemType Directory -Force -Path "models" | Out-Null
    }

    Install-DirectFiles -Name "MediaPipe pose+hand bundle" -Files @(
        @{ Url = "https://huggingface.co/opencv/pose_estimation_mediapipe/resolve/main/pose_estimation_mediapipe_2023mar.onnx";
           OutName = "pose_landmark.onnx" }
        @{ Url = "https://huggingface.co/opencv/palm_detection_mediapipe/resolve/main/palm_detection_mediapipe_2023feb.onnx";
           OutName = "palm_detection.onnx" }
        @{ Url = "https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx";
           OutName = "hand_landmark.onnx" }
    )

    Write-Host "MediaPipe ONNX models installed successfully." -ForegroundColor Green
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

$commands = @(
    @{ Label = "setup (download pose models + CJK font)"; Cmd = "Install-Models; Install-Font" },
    @{ Label = "build (debug)";    Cmd = "cargo build" },
    @{ Label = "build (release)";  Cmd = "cargo build --release" },
    @{ Label = "run (debug)";      Cmd = '$env:RUST_LOG="vulvatar=info"; cargo run' },
    @{ Label = "run (debug+lipsync)"; Cmd = '$env:RUST_LOG="vulvatar=info"; cargo run --features lipsync' },
    @{ Label = "run (release)";    Cmd = "cargo run --release" },
    @{ Label = "install mf virtual camera (HKLM)"; Cmd = "Install-MfCameraSystem" },
    @{ Label = "uninstall mf virtual camera"; Cmd = "Uninstall-MfCamera" },
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
