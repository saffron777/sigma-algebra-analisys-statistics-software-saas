# Contribuir a Sigma

Gracias por querer mejorar Sigma. Este documento resume cómo preparar el entorno, qué
estilo seguimos y cómo aceptamos contribuciones.

## Licencia de las contribuciones

Sigma se distribuye bajo **AGPL-3.0-or-later**. Al enviar una contribución aceptas que se
publique bajo esa misma licencia y que tienes derecho a hacerlo. No pedimos un CLA: usamos
el **Developer Certificate of Origin (DCO)**. Firma tus commits con:

```bash
git commit -s -m "fix: ..."
```

La firma `Signed-off-by:` certifica que puedes contribuir con ese código. Consulta
<https://developercertificate.org/>.

## Entorno de desarrollo

Requisitos: Python 3.11+ y Node.js 20+. En Windows:

```powershell
.\scripts\setup.ps1
.\scripts\run.ps1
```

En Linux/macOS:

```bash
bash scripts/setup.sh
bash scripts/run.sh
```

También puedes usar Docker (`docker compose up -d`). La primera vez abre `/setup` para crear
la cuenta de administrador.

## Antes de enviar un cambio

```bash
python -m pytest
python -m ruff check .
node --check sigma/presentation/web/static/js/modules/<modulo>.js   # si tocas JS
```

- Añade o actualiza pruebas para los cambios de comportamiento.
- Mantén la arquitectura: `presentation → application → domain`, con `infrastructure`
  implementando puertos.
- Sigue el estilo existente: Python con type hints y mensajes de error en español; JS
  vanilla/Alpine sin frameworks nuevos; plantillas Jinja con Tailwind.
- No incluyas datos personales, credenciales, bases de datos de `instance/` ni archivos
  generados (`.venv`, `node_modules`, `static/vendor`, `static/css/app.css`).
- Si añades una dependencia, revisa su licencia (debe ser compatible con AGPLv3) y actualiza
  `THIRD-PARTY-NOTICES.md` con `python scripts/collect_licenses.py collect`.

## Pull requests

1. Abre una incidencia o explica el problema en la PR.
2. Un cambio por PR, con un título claro y conciso.
3. Describe cómo probarlo y adjunta capturas si cambia la interfaz.
4. Los tests y el lint deben pasar; si algo no se puede probar en tu entorno, indícalo.
5. No hagas force-push a `main` ni publiques releases; eso lo hace el mantenedor.

## Código de conducta

Al participar aceptas el [Código de Conducta](CODE_OF_CONDUCT.md). Para reportar un
problema de conducta, escribe a `frank.alexis@gmail.com`.

## Seguridad

No reportes vulnerabilidades en una incidencia pública: sigue [SECURITY.md](SECURITY.md).
