#!/usr/bin/env bash
# Crea la instantánea semilla (usuario demo + datos de ejemplo) y la copia al volumen de la demo.
#   DEMO_DOMAIN=demo.tudominio.org bash demo/seed.sh
set -euo pipefail
cd "$(dirname "$0")"

docker compose pull sigma >/dev/null 2>&1 || docker compose build sigma
docker compose --profile tools run --rm seed
docker compose --profile tools run --rm reset
echo "Semilla lista. Levanta la demo con: docker compose up -d"
