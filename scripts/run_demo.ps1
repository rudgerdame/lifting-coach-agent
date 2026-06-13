# One-command demo: synthetic data -> train -> eval report + SHAP plot
# Usage: .\scripts\run_demo.ps1
#        .\scripts\run_demo.ps1 -GravityOS   # local personal data (not committed)

param(
    [switch]$GravityOS,
    [string]$GravityOSDir = $env:GRAVITYOS_DATA_DIR
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Pip = Join-Path $Root ".venv\Scripts\pip.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "Installing dependencies..."
& $Pip install -r requirements.txt -q

if (-not $GravityOS) {
    Write-Host "Generating synthetic data..."
    & $Python -m ingestion.synthetic --out data/synthetic

    Write-Host "Training on synthetic data..."
    & $Python -m models.train --data-dir data/synthetic
} else {
    if (-not $GravityOSDir) {
        throw "Set GRAVITYOS_DATA_DIR or pass -GravityOSDir"
    }
    Write-Host "Training on Gravity OS data at $GravityOSDir ..."
    & $Python -m models.train --gravityos --gravityos-dir $GravityOSDir
}

Write-Host ""
Write-Host "Done. Review:"
Write-Host "  eval/model_report.md"
Write-Host "  eval/shap_summary.png"
Write-Host "  models/artifacts/lgb_readiness.pkl"
