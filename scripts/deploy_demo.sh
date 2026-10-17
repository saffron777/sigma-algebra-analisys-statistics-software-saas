#!/usr/bin/env bash
# Despliega la demo en un VPS con Docker. Debe ejecutarse desde el repositorio.
#   VPS_HOST=203.0.113.10 SSH_USER=root SSH_KEY=~/.ssh/id_ed25519 \
#     bash scripts/deploy_demo.sh
# Si el usuario no es root, necesita sudo sin contraseña o pertenecer al grupo docker.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${VPS_HOST:?Define VPS_HOST con la IP o el host del VPS}"
SSH_USER="${SSH_USER:-root}"
DEMO_DOMAIN="${DEMO_DOMAIN:-demo.polluxnetsystem.com}"
TAG="${TAG:-v1.0.0}"
REMOTE_DIR="${REMOTE_DIR:-/opt/sigma-demo}"
SSH_KEY="${SSH_KEY:-}"

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)
if [[ -n "$SSH_KEY" ]]; then
  SSH+=(-i "$SSH_KEY")
fi
TARGET="${SSH_USER}@${VPS_HOST}"

echo "==> Comprobando Docker en ${TARGET}"
"${SSH[@]}" "$TARGET" 'docker --version && docker compose version' || {
  echo "El VPS no tiene Docker/Compose. Instálalos antes de continuar." >&2
  exit 1
}

echo "==> Clonando ${TAG} en ${REMOTE_DIR}"
"${SSH[@]}" "$TARGET" "set -e
  if [ -d '${REMOTE_DIR}/.git' ]; then
    git -C '${REMOTE_DIR}' fetch --tags --depth 1 origin '${TAG}'
    git -C '${REMOTE_DIR}' checkout -f '${TAG}'
  else
    mkdir -p '$(dirname "${REMOTE_DIR}")'
    git clone --depth 1 --branch '${TAG}' https://github.com/saffron777/sigma-algebra-analisys-statistics-software-saas.git '${REMOTE_DIR}'
  fi"

echo "==> Sembrando la demo y levantando el stack"
"${SSH[@]}" "$TARGET" "set -e
  cd '${REMOTE_DIR}/demo'
  export DEMO_DOMAIN='${DEMO_DOMAIN}'
  bash seed.sh
  docker compose up -d"

echo "==> Comprobando HTTPS"
"${SSH[@]}" "$TARGET" "curl -fsS --retry 20 --retry-delay 5 --retry-all-errors https://${DEMO_DOMAIN}/health"

echo "Demo lista en https://${DEMO_DOMAIN}/"
echo "Recuerda abrir 80/443 en el firewall y crear el registro DNS A ${DEMO_DOMAIN} -> ${VPS_HOST}."
