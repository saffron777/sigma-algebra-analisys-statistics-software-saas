# Publicar la imagen Docker

La imagen se publica en **GitHub Container Registry**:

```
ghcr.io/saffron777/sigma-algebra-analisys-statistics-software-saas
```

También puedes usar el workflow de release (`.github/workflows/release.yml`) empujando el tag
`v1.0.0`; es la vía recomendada.

## Requisitos

- Docker Desktop/Engine en marcha.
- Una cuenta de GitHub con permiso `write:packages` sobre el paquete.
- Idealmente, el repositorio ya publicado y el tag `v1.0.0`.

## Opción A: workflow de release (recomendada)

Empuja el tag y deja que `.github/workflows/release.yml` compile y publique:

```bash
git tag -s v1.0.0 -m "Sigma Algebra 1.0.0"
git push origin v1.0.0
```

El workflow publica `:v1.0.0` y `:latest`. Después, en GitHub → Packages, pon la visibilidad del
paquete en **Public**.

## Opción B: manual

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u saffron777 --password-stdin
bash scripts/publish_docker.sh 1.0.0
```

Comprueba:

```bash
docker pull ghcr.io/saffron777/sigma-algebra-analisys-statistics-software-saas:1.0.0
docker run --rm -d -p 127.0.0.1:5000:5000 \
  -v sigma_data:/app/instance \
  ghcr.io/saffron777/sigma-algebra-analisys-statistics-software-saas:1.0.0
curl -fsS http://127.0.0.1:5000/health
curl -fsS http://127.0.0.1:5000/licencia | head
```

> No se pudo construir ni publicar la imagen desde esta sesión porque el daemon de Docker no está
> en marcha. El Dockerfile y el workflow están preparados; ejecuta los comandos en tu máquina o deja
> que CI lo haga al empujar el tag.
