# Create or update the native-Windows GPU environment in .venv-win.
# Run from the repo root in PowerShell:  .\scripts\setup_windows.ps1
# Uses the community triton-windows build (upstream Triton ships Linux wheels only).
# Package versions come from requirements-tested.txt. Safe to re-run: it also replaces
# a CPU-only PyTorch. Delete .venv-win to start over.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv-win"
$python = Join-Path $venv "Scripts\python.exe"
$constraints = Join-Path $root "requirements-tested.txt"
# PyTorch's CUDA wheels live on its own index; the version is pinned in the constraints.
$torchIndex = "https://download.pytorch.org/whl/cu130"

function Assert-Success($message) {
    if ($LASTEXITCODE -ne 0) { throw $message }
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi was not found: install the NVIDIA Windows driver first."
}
if (-not (Test-Path $python)) {
    # Checked first: with ErrorActionPreference=Stop, calling a missing python throws
    # before any friendlier message could be shown.
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "python was not found: install Python 3.10 or newer from python.org."
    }
    $version = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
    Assert-Success "python did not run: check the python.org installation."
    if ([version]$version -lt [version]"3.10") {
        throw "Python $version found; 3.10 or newer is required."
    }
    Write-Host "Creating $venv with Python $version"
    python -m venv $venv
    Assert-Success "python -m venv failed."
}

& $python -m pip install --upgrade pip
Assert-Success "Upgrading pip failed."
& $python -m pip install -c $constraints torch --index-url $torchIndex
Assert-Success "PyTorch install failed: no CUDA wheel for this Python version?"
# pip treats any installed torch 2.11.0 as satisfying the pin, including a CPU-only
# build from another index, so check the build and replace it if needed.
$torchCuda = & $python -c "import torch; print(torch.version.cuda or '')"
Assert-Success "PyTorch was installed but cannot be imported."
if (-not $torchCuda) {
    Write-Host "Replacing the CPU-only PyTorch with the CUDA build"
    & $python -m pip install --force-reinstall --no-deps -c $constraints torch --index-url $torchIndex
    Assert-Success "Reinstalling the CUDA build of PyTorch failed."
}
Push-Location $root
try {
    # gpu adds triton-windows on Windows; dev adds Ruff and NumPy for interpreter tests.
    & $python -m pip install -c $constraints -e ".[gpu,dev]"
    Assert-Success "Installing this repo and triton-windows failed."
} finally {
    Pop-Location
}

& $python -m kernel_portfolio.environment
Assert-Success "The environment report failed."
& $python -c "import torch, triton; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU; update the NVIDIA driver (CUDA 13.0 wheels need a recent one).'"
Assert-Success "Setup finished, but the GPU check failed (see the message above)."
Write-Host ""
Write-Host "Done. In each new PowerShell terminal, from the repo root, run:"
Write-Host "    .\.venv-win\Scripts\Activate.ps1"
