$ErrorActionPreference = "Stop"

function Install-Models {
    Write-Host "Setting up ONNX models..." -ForegroundColor Cyan
    $modelName = "cigpose-m_coco-wholebody_256x192.onnx"
    $modelPath = "models\$modelName"
    
    if (!(Test-Path "models")) {
        New-Item -ItemType Directory -Force -Path "models" | Out-Null
    }

    if (Test-Path $modelPath) {
        Write-Host "Model $modelName already exists. Skipping download." -ForegroundColor Green
        return
    }

    $zipPath = "cigpose_models.zip"
    Write-Host "Downloading cigpose_models.zip (this may take a while as it is >1GB)..."
    Invoke-WebRequest -Uri "https://github.com/namas191297/cigpose-onnx/releases/latest/download/cigpose_models.zip" -OutFile $zipPath

    Write-Host "Extracting..."
    # Warning: The zip contains many models. Expand-Archive might take time.
    Expand-Archive -Path $zipPath -DestinationPath "models" -Force

    Write-Host "Cleaning up zip archive..."
    Remove-Item $zipPath -Force

    Write-Host "ONNX Models installed successfully." -ForegroundColor Green
}

$commands = @(
    @{ Label = "setup (download models)"; Cmd = "Install-Models" },
    @{ Label = "build (debug)";    Cmd = "cargo build" },
    @{ Label = "build (release)";  Cmd = "cargo build --release" },
    @{ Label = "run (debug)";      Cmd = '$env:RUST_LOG="vulvatar=info"; cargo run' },
    @{ Label = "run (release)";    Cmd = "cargo run --release" },
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

while ($true) {
    Show-Menu
    $input = Read-Host "Select"

    if ($input -eq "0" -or $input -eq "q") {
        break
    }

    $idx = 0
    if ([int]::TryParse($input, [ref]$idx) -and $idx -ge 1 -and $idx -le $commands.Count) {
        $cmd = $commands[$idx - 1].Cmd
        Write-Host "> $cmd" -ForegroundColor Yellow
        Write-Host ""
        Invoke-Expression $cmd
        Write-Host ""
        Write-Host "Done. Press any key to continue..." -ForegroundColor DarkGray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    } else {
        Write-Host "Invalid selection." -ForegroundColor Red
    }
}
