# Licencias de terceros

`LICENSE` (en la raíz) es la licencia del proyecto: **AGPL-3.0-or-later**.

Este directorio contiene los textos de licencia de componentes de terceros distribuidos
con Sigma Algebra:

- `frontend/`: bibliotecas JavaScript que se copian a
  `sigma/presentation/web/static/vendor/`.
- `icons/`: licencias de los conjuntos de iconos Lucide (ISC) y Feather (MIT).
- `fonts/`: licencia de las fuentes DejaVu usadas en los PDFs y en el logotipo.
- `python/`: textos de las dependencias de Python, generados en el artefacto de release por
  `scripts/collect_licenses.py` (no es necesario versionarlos todos en el repositorio).

`THIRD-PARTY-NOTICES.md` explica qué componente usa cada licencia. Si añades una dependencia
nueva, ejecuta:

```bash
python scripts/collect_licenses.py collect
python scripts/collect_licenses.py check
```

y revisa que su licencia sea compatible con AGPL-3.0-or-later.
