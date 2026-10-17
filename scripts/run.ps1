# Inicia Sigma Algebra en http://127.0.0.1:5000  (usa SIGMA_DEBUG=1 para recarga automática en desarrollo)
#   .\scripts\run.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\python.exe run.py
