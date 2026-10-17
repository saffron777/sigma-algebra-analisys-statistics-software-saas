#!/usr/bin/env bash
# Prepara Sigma Algebra desde cero en Linux/macOS: entorno virtual, dependencias, frontend y datos de ejemplo.
#   bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "Creando el entorno virtual (.venv)..."
  python3 -m venv .venv
fi
PYTHON=".venv/bin/python"

echo "Instalando dependencias de Python..."
"$PYTHON" -m pip install --upgrade pip
if [ -f requirements-dev.lock ]; then
  "$PYTHON" -m pip install -r requirements-dev.lock
elif [ -f requirements-dev.txt ]; then
  "$PYTHON" -m pip install -r requirements-dev.txt
else
  "$PYTHON" -m pip install -r requirements.lock
fi

echo "Instalando y compilando el frontend (Tailwind, Plotly, Alpine, KaTeX...)..."
npm ci
npm run build

echo "Generando los archivos de ejemplo..."
"$PYTHON" scripts/make_sample_data.py

echo
echo "Listo. Inicie Sigma Algebra con: ./scripts/run.sh"
echo "La primera vez abra /setup para crear la cuenta de administrador."
