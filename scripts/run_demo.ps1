# One-command demo: synthetic data -> train -> FAISS index -> agent eval
# Usage: .\scripts\run_demo.ps1

param()

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

Write-Host "Generating synthetic data..."
& $Python -m ingestion.synthetic --out data/synthetic

Write-Host "Training on synthetic data..."
& $Python -m models.train --data-dir data/synthetic

Write-Host "Building FAISS corpus index..."
& $Python -m index.build

Write-Host "Running agent eval harness..."
& $Python -m eval.run_all --data-dir data/synthetic

Write-Host ""
Write-Host "Done. Review:"
Write-Host "  eval/model_report.md"
Write-Host "  eval/classification_report.png  (primary: confusion + reliability)"
Write-Host "  eval/shap_summary_clf.png       (classifier SHAP)"
Write-Host "  eval/shap_summary.png           (regression SHAP)"
Write-Host "  eval/results.md"
Write-Host "  models/artifacts/lgb_readiness_clf.pkl  (primary classifier)"
Write-Host "  models/artifacts/lgb_readiness.pkl      (regression)"
