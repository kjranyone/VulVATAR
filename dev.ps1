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

function Install-VisemeModel {
    Write-Host "Training viseme lip-sync model..." -ForegroundColor Cyan
    $modelPath = "models\viseme_nn.onnx"

    if (Test-Path $modelPath) {
        Write-Host "Viseme model already exists at $modelPath." -ForegroundColor Green
        $ans = Read-Host "Retrain? (y/N)"
        if ($ans -ne "y") { return }
    }

    # Check Python + torch availability.
    try {
        python -c "import torch; import numpy" 2>$null
        if ($LASTEXITCODE -ne 0) { throw "missing" }
    } catch {
        Write-Host "Error: Python with torch and numpy is required." -ForegroundColor Red
        Write-Host "  pip install torch numpy" -ForegroundColor Yellow
        return
    }

    $env:PYTHONIOENCODING = "utf-8"
    python scripts/train_viseme_model.py --output $modelPath
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Viseme model ready at $modelPath." -ForegroundColor Green
    } else {
        Write-Host "Training failed." -ForegroundColor Red
    }
}

$commands = @(
    @{ Label = "setup (download pose models)"; Cmd = "Install-Models" },
    @{ Label = "setup (train viseme model)";   Cmd = "Install-VisemeModel" },
    @{ Label = "build (debug)";    Cmd = "cargo build" },
    @{ Label = "build (release)";  Cmd = "cargo build --release" },
    @{ Label = "run (debug)";      Cmd = '$env:RUST_LOG="vulvatar=info"; cargo run' },
    @{ Label = "run (debug+lipsync)"; Cmd = '$env:RUST_LOG="vulvatar=info"; cargo run --features lipsync' },
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
