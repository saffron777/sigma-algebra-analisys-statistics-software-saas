# Publicar una versión de Sigma Algebra

Checklist para el mantenedor. Sustituye `1.0.0` por la versión correspondiente.

## Antes de etiquetar

1. Confirma que tienes derecho a relicenciar todo el código y los assets (revisa contratos
   con clientes si los hubo) y que las aportaciones con IA cumplen sus términos.
2. Actualiza la versión en `sigma/__init__.py`, `pyproject.toml`, `package.json` y
   `package-lock.json`; añade la entrada en `CHANGELOG.md`.
3. Revisa `THIRD-PARTY-NOTICES.md` y ejecuta:

   ```bash
   python scripts/collect_licenses.py collect
   python scripts/collect_licenses.py check
   python scripts/check_versions.py
   python -m pytest
   python -m ruff check .
   ```

4. Comprueba que `git status` no incluye `.env`, `instance/`, `.gitconfig.backup`, bases de
   datos ni archivos temporales. Si toca reescribir el historial, haz copia con
   `git bundle create ... --all` y verifica después que `.gitconfig.backup` no aparece en
   ningún commit.
5. Construye y prueba Docker:

   ```bash
   docker compose build
   docker compose up -d
   curl -fsS http://127.0.0.1:5000/health
   curl -fsS http://127.0.0.1:5000/licencia | head
   docker compose down
   ```

## Empaquetar

```bash
npm ci
npm run build
python scripts/build_release.py
```

Se generan en `dist/` el código fuente, el paquete listo para ejecutar (con assets) y
`SHA256SUMS`. Revisa el contenido del tar/zip: no debe haber secretos ni datos de usuario.
Para verificar las sumas:

```bash
cd dist && sha256sum -c SHA256SUMS && cd ..
gpg --detach-sign --armor dist/sigma-algebra-analisys-statistics-software-saas-1.0.0.tar.gz
```

## Publicar

1. `git commit -s` con los cambios; `git tag -s v1.0.0 -m "Sigma Algebra 1.0.0"`.
2. `git push origin main --tags`.
3. En GitHub, crea la release `v1.0.0` y adjunta los archivos de `dist/` y `SHA256SUMS`.
4. Opcional: publica la imagen en GHCR:

   ```bash
   bash scripts/publish_docker.sh 1.0.0
   ```

5. Opcional: archiva la release en Zenodo para obtener un DOI y añade el badge al README.
6. Difunde siguiendo [docs/launch/checklist.md](launch/checklist.md): demo estable, Show HN,
   LinkedIn, Reddit y las fichas de directorios. Incluye el quick start de Docker.

## Después

- Revisa incidencias y reportes privados de seguridad.
- Mantén Dependabot/Renovate y repite la auditoría de licencias en cada release.
- No borres los artefactos de releases antiguas si la AGPL exige ofrecer el código fuente
  correspondiente; conserva al menos el código fuente de las versiones publicadas.
