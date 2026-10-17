# Desplegar la demo en polluxnetsystem.com

La demo pública se sirve en **https://demo.polluxnetsystem.com** con el stack
[demo/docker-compose.yml](../demo/docker-compose.yml) y Caddy.

## Requisitos

- Un **VPS Linux** con Docker y Docker Compose (no el hosting compartido del dominio).
- Puertos **80** y **443** libres.
- Acceso SSH (preferible clave privada) y permiso para ejecutar Docker.
- El token de Cloudflare con permiso **Zone → DNS → Edit** para `polluxnetsystem.com`.

## 1. DNS

Con el token guardado en `.cf-token` (ignorado por Git) o en `CLOUDFLARE_API_TOKEN`:

```bash
python scripts/setup_demo_dns.py --ip IP_DEL_VPS
```

Crea `demo.polluxnetsystem.com` como **A** y **DNS only** (proxy desactivado), para que Caddy
pueda obtener el certificado con HTTP-01. Si prefieres el proxy naranja de Cloudflare, usa
`--proxy` y configura SSL/TLS **Full (strict)** con un certificado de origen.

Comprueba:

```bash
nslookup demo.polluxnetsystem.com
```

## 2. Despliegue

```bash
VPS_HOST=IP_DEL_VPS SSH_USER=root SSH_KEY=~/.ssh/id_ed25519 \
  bash scripts/deploy_demo.sh
```

El script:

1. Comprueba Docker/Compose en el VPS.
2. Clona el tag `v1.0.0` en `/opt/sigma-demo`.
3. Ejecuta `demo/seed.sh` (usuario demo + datos de ejemplo).
4. Levanta Caddy y la aplicación.
5. Verifica `https://demo.polluxnetsystem.com/health`.

Si usas otro usuario, debe poder ejecutar Docker (grupo `docker` o `sudo` configurado).

## 3. Firewall y seguridad

- Abre 80/443 en el firewall del VPS; no expongas el 5001.
- La demo bloquea importaciones, consola, IA, administración y `DELETE`/`PATCH`.
- No uses la demo con datos reales.
- Programa el reinicio:

```cron
0 * * * * /opt/sigma-demo/demo/reset.sh >> /var/log/sigma-demo-reset.log 2>&1
```

## 4. Verificación

```bash
curl -fsS https://demo.polluxnetsystem.com/health
curl -fsSI https://demo.polluxnetsystem.com/demo
```

Abre la URL, pulsa **Probar la demo** y comprueba que aparece el banner, que las importaciones
devuelven 403 y que el reinicio funciona.

## 5. Después de desplegar

- Cambia los botones `#deploy` de `site/index.html` por `https://demo.polluxnetsystem.com`.
- Actualiza los enlaces de demo en `README.md`, `README.es.md` y `docs/launch/`.
- Solo entonces envía las fichas de directorios y publica Show HN/LinkedIn.
