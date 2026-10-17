# Prepara Sigma Algebra desde cero: entorno virtual, dependencias, CSS/JS del frontend y archivos de ejemplo.
#   .\scripts\setup.ps1
# La primera cuenta (administrador) se crea al abrir Sigma Algebra por primera vez en el navegador (/setup),
# o sin navegador con:  .\.venv\Scripts\python.exe scripts\create_user.py
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".venv")) {
    Write-Host "Creando el entorno virtual (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
}
$python = ".\.venv\Scripts\python.exe"

Write-Host "Instalando dependencias de Python..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip
if (Test-Path "requirements-dev.lock") {
    & $python -m pip install -r requirements-dev.lock
} elseif (Test-Path "requirements-dev.txt") {
    & $python -m pip install -r requirements-dev.txt
} else {
    & $python -m pip install -r requirements.lock
}

Write-Host "Instalando y compilando el frontend (Tailwind, Plotly, Alpine, KaTeX)..." -ForegroundColor Cyan
npm ci
npm run build

Write-Host "Generando los archivos de ejemplo (sample_data)..." -ForegroundColor Cyan
& $python scripts\make_sample_data.py

Write-Host ""
Write-Host "Listo. Inicie Sigma Algebra con:  .\scripts\run.ps1" -ForegroundColor Green
Write-Host "Al abrirlo por primera vez cree la cuenta de administrador; luego, desde Datos, use 'Cargar datos de ejemplo'." -ForegroundColor Green
