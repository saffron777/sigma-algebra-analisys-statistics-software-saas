# syntax=docker/dockerfile:1
# Imagen de Sigma Algebra Analisys Statistics Software (SAAS): compila los assets del navegador y ejecuta la app con Waitress como usuario sin privilegios.
FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY scripts/copy_vendor.mjs scripts/copy_vendor.mjs
COPY sigma/presentation/web/templates sigma/presentation/web/templates
COPY sigma/presentation/web/static/js sigma/presentation/web/static/js
COPY sigma/presentation/web/static/src sigma/presentation/web/static/src
COPY sigma/presentation/web/static/img sigma/presentation/web/static/img
RUN mkdir -p sigma/presentation/web/static/css && npm run build

FROM python:3.13-slim-bookworm AS runtime
LABEL org.opencontainers.image.title="Sigma Algebra Analisys Statistics Software (SAAS)" \
      org.opencontainers.image.description="Self-hosted statistics and reporting web app" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/saffron777/sigma-algebra-analisys-statistics-software-saas" \
      org.opencontainers.image.documentation="https://github.com/saffron777/sigma-algebra-analisys-statistics-software-saas#readme"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SIGMA_HOST=0.0.0.0 \
    SIGMA_PORT=5000 \
    SIGMA_INSTANCE_DIR=/app/instance
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential unixodbc-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY sigma sigma
COPY sample_data sample_data
COPY scripts/load_sample_data.py scripts/make_sample_data.py scripts/seed_demo.py scripts/
COPY run.py wsgi.py ./
COPY LICENSE NOTICE THIRD-PARTY-NOTICES.md TRADEMARKS.md PRIVACY.md SECURITY.md ./
COPY LICENSES LICENSES

COPY --from=frontend /build/sigma/presentation/web/static/vendor sigma/presentation/web/static/vendor
COPY --from=frontend /build/sigma/presentation/web/static/css/app.css sigma/presentation/web/static/css/app.css

RUN useradd --create-home --uid 10001 sigma \
    && mkdir -p /app/instance \
    && chown -R sigma:sigma /app
USER sigma

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)"

CMD ["waitress-serve", "--listen=0.0.0.0:5000", "wsgi:app"]
