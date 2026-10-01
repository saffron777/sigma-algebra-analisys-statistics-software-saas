# Prepara Sigma desde cero: entorno virtual, dependencias, CSS/JS del frontend y datos de ejemplo.
#   .\scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".venv")) {
    Write-Host "Creando el entorno virtual (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
}
$python = ".\.venv\Scripts\python.exe"

Write-Host "Instalando dependencias de Python..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements-dev.txt

Write-Host "Instalando y compilando el frontend (Tailwind, Plotly, Alpine)..." -ForegroundColor Cyan
npm install
npm run build

Write-Host "Cargando los datos de ejemplo..." -ForegroundColor Cyan
& $python scripts\make_sample_data.py
& $python scripts\load_sample_data.py

Write-Host ""
Write-Host "Listo. Inicie Sigma con:  .\scripts\run.ps1" -ForegroundColor Green
