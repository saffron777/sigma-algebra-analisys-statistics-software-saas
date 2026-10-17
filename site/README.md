# Sitio estático

Landing page de Sigma Algebra, sin build ni CDN. Se publica con GitHub Pages.

## Antes de publicar

1. Sustituye `https://demo.tudominio.org` por la URL real de la demo en `index.html`.
2. Comprueba que `assets/` tiene el logotipo, el GIF y las tres capturas.
3. Revisa los enlaces a GitHub, licencia, terceros y marcas.
4. Activa GitHub Pages con **Source: GitHub Actions** en la configuración del repositorio.

## Probar en local

```bash
python -m http.server 8000 --directory site
```

Abre <http://127.0.0.1:8000>.

## Publicar

El workflow `.github/workflows/pages.yml` sube la carpeta `site/` al hacer push a `main`. También se
puede ejecutar manualmente desde **Actions → Pages → Run workflow**.

Para un dominio propio, configura el CNAME en GitHub Pages y actualiza los enlaces absolutos de la
landing si cambia la URL del repositorio.
