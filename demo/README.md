# Demo pública de Sigma Algebra

Esta carpeta levanta una instancia **efímera y aislada** para que la comunidad pruebe la aplicación
sin subir datos reales.

## Requisitos

- Un VPS Linux con Docker y Docker Compose.
- Un dominio/subdominio (por ejemplo `demo.polluxnetsystem.com`) con un registro DNS `A` apuntando al VPS.
- Puertos **80** y **443** abiertos; no expongas el 5001.

## Puesta en marcha

1. Copia esta carpeta al servidor y entra en ella:
   ```bash
   cd demo
   ```
2. Exporta el dominio y crea la semilla:
   ```bash
   export DEMO_DOMAIN=demo.polluxnetsystem.com
   bash seed.sh
   ```
   `seed.sh` crea el usuario regular `demo` (sin consola ni IA) con los datos de ejemplo y copia la
   instantánea al volumen de la demo.
3. Levanta la aplicación y Caddy:
   ```bash
   docker compose up -d
   ```
   La primera vez Caddy obtiene el certificado TLS. Comprueba:
   ```bash
   curl -fsS https://$DEMO_DOMAIN/health
   curl -fsSI https://$DEMO_DOMAIN/demo
   ```
4. Abre `https://demo.polluxnetsystem.com` y pulsa «Probar la demo pública». La ruta `/demo` inicia sesión
   como el usuario demo; no hay contraseña pública.

## Reinicio periódico

La demo se reinicia para borrar datos de prueba. Añade una tarea cron (cada hora o cada día):

```cron
0 * * * * /ruta/a/sigma/demo/reset.sh >> /var/log/sigma-demo-reset.log 2>&1
```

`reset.sh` detiene la app, restaura la semilla y la vuelve a arrancar. Las sesiones abiertas se
invalidan (se conserva la `secret.key` de la semilla).

## Seguridad y límites

- `SIGMA_DEMO_MODE=1` muestra el banner de demo y bloquea importaciones de archivos/bases/API,
  consola, IA, administración y operaciones destructivas (`DELETE`/`PATCH`).
- Los análisis, gráficos, resultados, exportaciones y la edición manual de datos siguen disponibles.
- No hay claves de IA ni acceso a bases externas (`SIGMA_EXTERNAL_DB_HOSTS=disabled.invalid`).
- La app corre como usuario sin privilegios; Caddy pone TLS y limita el cuerpo a 10 MB.
- No uses esta configuración para datos reales: el volumen se restaura desde la semilla.

## Actualizar la imagen

```bash
docker compose pull
docker compose up -d
```

Para una demo local antes de publicar la imagen, construye desde el repositorio:

```bash
docker compose build
bash seed.sh
docker compose up -d
```
