# Show HN

> Sustituye `https://demo.tudominio.org` por la URL real antes de publicar. Publica un martes,
> miércoles o jueves por la mañana (hora de la costa este de EE. UU.) y quédate a responder.

## Título

`Show HN: Sigma Algebra – Self-hosted statistics and reporting app (AGPL, Python)`

## Cuerpo

I built Sigma Algebra because I wanted a statistics web app I could host myself, without per-seat
licenses, that students and colleagues could use from a browser.

It imports CSV, Excel, text, SQL Server, MySQL/MariaDB, PostgreSQL, Oracle, MongoDB and JSON APIs.
It covers descriptive statistics, hypothesis tests, regression/ANOVA, surveys and sampling,
SPC/DOE, charts and PDF/Word/Excel reports. Each account sees only its own datasets, results and
notebooks. Optional AI interpretation is per user and per click; there is no telemetry.

It is AGPL-3.0-or-later and self-hosted ("SAAS" is part of the product name, not a hosted service).
Docker one-liner and demo:

- Demo: https://demo.tudominio.org
- Source: https://github.com/saffron777/sigma-algebra-analisys-statistics-software-saas
- Quick start: `docker compose up --build -d` in the repo

The UI and most docs are in Spanish; the README and landing page are in English. I would love
feedback on onboarding, the English translation and which analyses are missing for teaching or
small-team use. I am the author.

## Primer comentario (técnico)

Stack: Flask + Jinja + Alpine.js + JSpreadsheet + Tailwind, SQLAlchemy/pyodbc/psycopg/pymongo,
pandas/NumPy/SciPy/statsmodels/scikit-learn, Plotly, Jupyter kernels for the optional Python/R
console. Clean architecture (`presentation → application → domain`), 1,600+ tests, Docker image,
and non-root runtime. The demo is read-mostly: imports, console, AI and admin are disabled and the
volume resets periodically.

Happy to answer questions about the architecture, the AGPL choice or the statistics catalogue.
