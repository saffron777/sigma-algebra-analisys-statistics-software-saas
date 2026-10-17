# Despliegue de Sigma Algebra

## Docker (recomendado)

```bash
docker compose up --build -d
```

La aplicación escucha en `127.0.0.1:5000` y guarda sus datos en el volumen
`sigma_instance` (`/app/instance`). Para actualizar:

```bash
docker compose pull
docker compose up -d
```

Si construyes localmente, usa `docker compose build` antes de `up`.

### Variables de entorno

Copia `.env.example` a `.env` y ajusta lo que necesites. Docker Compose lee `.env` para
interpolar las variables del servicio. Las más importantes:

| Variable | Uso |
|---|---|
| `SIGMA_SOURCE_URL` | URL pública del código fuente (se muestra en la interfaz). |
| `SIGMA_SECRET_KEY` | Secreto de sesión. Si se deja vacío, Sigma Algebra genera `instance/secret.key`. |
| `SIGMA_DATABASE_URL` | Base de datos externa; vacío usa SQLite en `instance/sigma.db`. |
| `SIGMA_HOST` / `SIGMA_PORT` | Escucha; en Docker se fija `0.0.0.0:5000`. |
| `SIGMA_SECURE_COOKIES` | `1` cuando se sirve por HTTPS. |
| `SIGMA_ALLOWED_HOSTS` | Nombres de host permitidos si se accede por red. |
| `SIGMA_EXTERNAL_DB_HOSTS` | Limita los servidores de bases externas que se pueden importar. |
| `SIGMA_AI_REQUESTS_PER_HOUR` | Límite de interpretaciones con IA por persona y hora. |

No publiques `.env` ni el volumen `instance/`: contienen secretos y datos de usuarios.

## HTTPS con proxy inverso

No expongas Waitress directamente a Internet. Ejemplo con Caddy:

```caddyfile
sigma.example.org {
    reverse_proxy 127.0.0.1:5000
    encode gzip
}
```

Con Nginx, reenvía a `http://127.0.0.1:5000` y configura TLS (Certbot). En ambos casos:

- fija `SIGMA_SECURE_COOKIES=1`;
- añade el dominio a `SIGMA_ALLOWED_HOSTS` si corresponde;
- revisa el límite de subida (`SIGMA_MAX_UPLOAD_MB`) y el tamaño máximo del cuerpo en el proxy;
- limita quién puede tener permiso de consola (ejecuta código con los permisos del servidor).

## Copias de seguridad

El estado vive en `instance/` (base SQLite, `secret.key`, `ai.json`, notebooks, importaciones
pendientes) o en la base externa configurada. Para una instalación SQLite, detén el servicio
o usa la copia en caliente de SQLite y guarda todo `instance/` con permisos restringidos.
Prueba periódicamente la restauración.

## Actualizaciones

1. Haz copia de seguridad.
2. `docker compose pull && docker compose up -d` (o reconstruye).
3. Comprueba `/health` y la página `/ayuda#acerca`.
4. Las migraciones de esquema se ejecutan al arrancar; revisa los mensajes del contenedor.

Si usas una base externa, el administrador puede cambiar de motor desde **Base de datos**;
haz copia antes de migrar y reinicia los procesos de Sigma Algebra que estén en ejecución.
