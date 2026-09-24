# Create or update the native-Windows GPU environment in .venv-win.
# Run from the repo root in PowerShell:  .\scripts\setup_windows.ps1
# Uses the community triton-windows build (upstream Triton ships Linux wheels only).
# Package versions come from requirements-tested.txt. Safe to re-run; delete .venv-win
# to start over.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv-win"
$python = Join-Path $venv "Scripts\python.exe"
$constraints = Join-Path $root "requirements-tested.txt"
# PyTorch's CUDA wheels live on its own index; the version is pinned in the constraints.
$torchIndex = "https://download.pytorch.org/whl/cu130"

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi was not found: install the NVIDIA Windows driver first."
}
if (-not (Test-Path $python)) {
    $version = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $version) {
        throw "python was not found: install Python 3.10 or newer from python.org."
    }
    if ([version]$version -lt [version]"3.10") {
        throw "Python $version found; 3.10 or newer is required."
    }
    Write-Host "Creating $venv with Python $version"
    python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "python -m venv failed." }
}

& $python -m pip install --upgrade pip
& $python -m pip install -c $constraints torch --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed: no CUDA wheel for this Python version?" }
Push-Location $root
try {
    # gpu adds triton-windows on Windows; dev adds Ruff and NumPy for interpreter tests.
    & $python -m pip install -c $constraints -e ".[gpu,dev]"
    if ($LASTEXITCODE -ne 0) { throw "Installing this repo and triton-windows failed." }
} finally {
    Pop-Location
}

& $python -m kernel_portfolio.environment
Write-Host ""
Write-Host "Done. In each new PowerShell terminal, from the repo root, run:"
Write-Host "    .\.venv-win\Scripts\Activate.ps1"
