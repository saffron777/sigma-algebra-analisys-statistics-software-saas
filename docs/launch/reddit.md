# Reddit

> Lee las reglas de cada subreddit antes de publicar. No publiques el mismo texto en todos el mismo
> día y responde a los comentarios. Sustituye la URL de demo.

## r/selfhosted

**Título:** Sigma Algebra – self-hosted statistics and reporting web app (AGPL, Docker)

**Cuerpo:**

Hi r/selfhosted,

I built Sigma Algebra, a self-hosted statistics web app for small teams and teaching. It runs in
Docker, stores data locally, and supports multiple accounts with per-user isolation.

What it does:
- Import CSV, Excel, text, SQL Server, MySQL/MariaDB, PostgreSQL, Oracle, MongoDB and JSON APIs.
- Descriptive stats, hypothesis tests, regression/ANOVA, surveys/sampling, SPC/DOE, charts.
- Export PDF/Word/Excel reports; optional Jupyter console (Python/R) per account.
- No telemetry; optional AI interpretation is disabled unless configured and allowed per user.

Stack: Flask, SQLAlchemy, pandas/NumPy/SciPy/statsmodels, Plotly, Alpine.js, Tailwind. AGPL-3.0-or-later.

Demo: https://demo.tudominio.org (sample data, resets periodically; no real data)
Source: https://github.com/saffron777/sigma-algebra-analisys-statistics-software-saas

I am the author. Feedback on deployment and missing features is welcome.

## r/Python

**Título:** Showcase: a statistics web app built with Flask, Alpine.js and a lot of pandas

**Cuerpo:**

I have been building Sigma Algebra, a self-hosted statistics app in Python. The interesting part
for this subreddit is the architecture: clean layers (`presentation → application → domain`), thin
Flask blueprints, use cases, repositories for SQL/Mongo, and a large pytest suite (1,600+ tests).

It covers descriptive statistics, tests, regression/ANOVA, surveys, SPC/DOE, charts, reports and an
optional Jupyter console. The frontend is Jinja + Alpine.js + JSpreadsheet; charts are Plotly.

Source (AGPL): https://github.com/saffron777/sigma-algebra-analisys-statistics-software-saas
Demo: https://demo.tudominio.org

Happy to discuss the architecture, testing strategy or the stats catalogue. I am the author.

## r/statistics (solo si las reglas lo permiten)

No lo publiques como promoción directa. Usa el hilo semanal de herramientas o pregunta antes a los
moderadores. Enfoca el mensaje en un caso de uso concreto (docencia, encuestas, control de calidad)
y ofrece la demo sin insistir. Aporta primero respuestas útiles en el subreddit.
