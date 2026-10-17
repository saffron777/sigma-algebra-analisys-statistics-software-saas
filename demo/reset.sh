#!/usr/bin/env bash
# Restaura la demo desde la instantánea semilla. Ejecútalo desde cron cada hora:
#   0 * * * * /ruta/a/sigma/demo/reset.sh
set -euo pipefail
cd "$(dirname "$0")"

docker compose stop sigma >/dev/null 2>&1 || true
docker compose --profile tools run --rm reset
docker compose start sigma >/dev/null 2>&1 || true
echo "Demo restaurada: $(date -Is)"
