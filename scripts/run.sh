#!/usr/bin/env bash
# Inicia Sigma Algebra en http://127.0.0.1:5000  (use SIGMA_DEBUG=1 para recarga automática).
#   bash scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python run.py
