# Avisos de terceros

Sigma Algebra se distribuye bajo **AGPL-3.0-or-later** (ver `LICENSE`). Este archivo enumera los
componentes de terceros que se incluyen en el repositorio, en el paquete de distribución o
en los assets que genera la aplicación.

Las licencias completas de los componentes de frontend, fuentes y datos de ejemplo están en
`LICENSES/`. El script `scripts/collect_licenses.py` regenera esta tabla y copia los textos
desde las dependencias instaladas.

## Frontend incluido en la aplicación

| Componente | Versión | Licencia | Uso |
|---|---:|---|---|
| Alpine.js | 3.17.4 | MIT | Reactividad de la interfaz. |
| DOMPurify | 3.4.16 | MPL-2.0 OR Apache-2.0 | Saneado de HTML. Se distribuye bajo Apache-2.0. |
| JSpreadsheet CE | 5.0.4 | MIT | Grilla de datos. |
| jSuites | 5.13.5 | MIT | Componentes de la grilla. |
| KaTeX | 0.19.0 | MIT | Fórmulas matemáticas y sus fuentes. |
| marked | 12.0.2 | MIT | Renderizado de Markdown. |
| Plotly.js (dist-min) | 4.1.1 | MIT | Gráficos. |
| Tailwind CSS (solo build) | 4.x | MIT | Generación de `app.css`. No se distribuye su código fuente en el runtime. |

Los archivos minificados de estos componentes se copian a
`sigma/presentation/web/static/vendor/` con `scripts/copy_vendor.mjs`. Los textos de licencia
se conservan en `LICENSES/frontend/` y deben acompañar a cualquier artefacto que incluya
esos archivos.

## Dependencias de Python

Se instalan por separado con `pip` (no se copian al repositorio). Sigma Algebra no las relicencia;
cada una conserva su licencia. Las principales:

| Componente | Licencia |
|---|---|
| Flask, Jinja2, Werkzeug, click, itsdangerous | BSD-3-Clause |
| pydantic | MIT |
| waitress | ZPL-2.1 |
| numpy, pandas, scipy, statsmodels, scikit-learn | BSD-3-Clause (numpy incluye 0BSD, MIT, Zlib y CC0-1.0) |
| semopy | MIT |
| plotly | MIT |
| openpyxl | MIT |
| xlrd | BSD |
| SQLAlchemy | MIT |
| pyodbc | MIT |
| requests | Apache-2.0 |
| certifi | MPL-2.0 |
| jupyter_client, ipykernel, nbformat, jupyter_core | BSD-3-Clause |
| matplotlib | PSF-2.0 (basada en BSD) |
| reportlab | BSD |
| python-docx | MIT |
| pillow | MIT-CMU |
| anthropic | MIT |
| PyMySQL | MIT |
| psycopg / psycopg-binary | LGPL-3.0-only |
| oracledb | UPL-1.0 OR Apache-2.0 |
| pymongo | Apache-2.0 |
| arch, linearmodels | NCSA |

`psycopg` se usa como biblioteca enlazada dinámicamente; se aplica su licencia LGPL-3.0.
`certifi` se usa bajo MPL-2.0. `oracledb` se usa bajo Apache-2.0. Estas licencias son
compatibles con la AGPL-3.0-or-later; los textos completos se incluyen en el artefacto de
release generado por `scripts/build_release.py`.

**No se incluye PyMuPDF.** Es AGPL-3.0/comercial y no forma parte de `requirements.txt`; si
alguna vez se añade, debe revisarse y declararse aquí.

## Iconos de la interfaz

Los iconos SVG de `sigma/presentation/web/templates/partials/macros.html` usan formas
geométricas simples, algunas procedentes de los conjuntos **Lucide** (ISC) y **Feather**
(MIT), junto con formas propias del proyecto. Los textos completos están en
`LICENSES/icons/`.

## Fuentes

- **KaTeX**: las fuentes de `static/vendor/fonts/` forman parte de KaTeX y se distribuyen
  bajo su licencia MIT.
- **DejaVu**: los PDFs generados por ReportLab y el logotipo usan DejaVu (incluida con
  matplotlib). La licencia de DejaVu (Bitstream Vera + Dominant Good) permite usar, copiar,
  modificar y distribuir las fuentes, incluida la inclusión de subconjuntos en documentos.
  El texto está en `LICENSES/fonts/DejaVu.txt`.

## Datos de ejemplo

Los archivos de `sample_data/` son sintéticos, generados por
`scripts/make_sample_data.py`, y no contienen datos de personas reales. Se distribuyen bajo
**CC0-1.0** para que puedan reutilizarse libremente.

## Marcas

La licencia de software no concede derechos sobre marcas o logotipos. Consulta
`TRADEMARKS.md`.

## Contacto

Para correcciones en estos avisos, abre una incidencia en el repositorio o escribe a
`frank.alexis@gmail.com`.
