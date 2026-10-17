# Demo pública

La demo es una instancia **autoalojada y efímera** de Sigma Algebra. Sirve para que la comunidad
pruebe la aplicación sin datos reales y sin tener que instalar nada.

## Qué incluye

- Un usuario regular `demo` con los datasets de ejemplo; **sin consola, sin IA y sin administración**.
- Entrada directa por `GET /demo` (sin contraseña pública).
- Banner de demo en todas las páginas.
- Bloqueo de importaciones (archivos, bases de datos y APIs), consola, IA, administración y
  `DELETE`/`PATCH`.
- Reinicio periódico desde una instantánea semilla.

## Puesta en marcha

El stack completo está en [demo/](../demo/README.md). Resumen:

```bash
cd demo
export DEMO_DOMAIN=demo.tudominio.org
bash seed.sh
docker compose up -d
```

Caddy obtiene el certificado TLS y hace de proxy inverso; la aplicación queda en
`127.0.0.1:5001` y no se expone directamente.

## Variables de entorno relevantes

| Variable | Valor en la demo |
|---|---|
| `SIGMA_DEMO_MODE` | `1` |
| `SIGMA_DEMO_USER` | `demo` |
| `SIGMA_DEMO_URL` | `https://demo.tudominio.org` |
| `SIGMA_ALLOWED_HOSTS` | `demo.tudominio.org` |
| `SIGMA_SECURE_COOKIES` | `1` |
| `SIGMA_EXTERNAL_DB_HOSTS` | `disabled.invalid` |
| `SIGMA_ANALYSIS_MAX_PARALLEL` | `1` |
| `SIGMA_ANALYSIS_TIMEOUT_SECONDS` | `60` |

## Reinicio

```cron
0 * * * * /ruta/a/sigma/demo/reset.sh >> /var/log/sigma-demo-reset.log 2>&1
```

La demo no debe usarse con datos personales o confidenciales. Si el dominio cambia, actualiza
`DEMO_DOMAIN` en `demo/Caddyfile`, `demo/docker-compose.yml` y este documento.
