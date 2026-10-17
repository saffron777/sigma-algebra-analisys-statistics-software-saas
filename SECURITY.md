# Seguridad

## Versiones soportadas

La rama `main` y la última versión publicada reciben correcciones de seguridad. Las
versiones anteriores pueden no recibirlas; actualiza siempre a la última release.

| Versión | Soporte |
|---|---|
| 1.0.x | Sí |
| < 1.0 | No |

## Cómo reportar una vulnerabilidad

**No abras una incidencia pública con detalles de una vulnerabilidad.**

Usa el canal privado de GitHub Security Advisories del repositorio o escribe a
`frank.alexis@gmail.com` con:

- versión de Sigma Algebra y sistema operativo;
- pasos para reproducir el problema;
- impacto estimado;
- cualquier prueba de concepto (sin datos reales de terceros).

Acuse de recibo en un plazo razonable y coordinación de la divulgación. No se ofrece un SLA
formal.

## Modelo de seguridad

- Sigma Algebra escucha en `127.0.0.1` por defecto. Para exponerlo en red usa HTTPS mediante un proxy
  inverso y revisa quién tiene acceso.
- La **consola Python/R ejecuta código arbitrario** con los permisos del proceso del
  servidor. Está desactivada por defecto y solo se concede por usuario desde la
  administración. No la habilites a cuentas no confiables.
- La página **Base de datos** y la gestión de usuarios son solo para administradores.
- Las contraseñas se guardan con un hash scrypt; las sesiones se invalidan al cambiar la
  contraseña o desactivar la cuenta.
- Las peticiones que modifican datos exigen token CSRF; se valida `Host`/`Origin` y se aplica
  una política de seguridad de contenido restrictiva.
- Los datos importados y el HTML de notebooks se sanean antes de mostrarse.
- Las claves de IA nunca se muestran completas ni llegan al navegador. No publiques
  `instance/`, `.env` ni `ai.json`.
- Las importaciones desde bases externas se abren en modo lectura y se validan; aun así,
  configura `SIGMA_EXTERNAL_DB_HOSTS` si quieres limitar los servidores accesibles.

## Buenas prácticas de despliegue

- Usa HTTPS y `SIGMA_SECURE_COOKIES=1` cuando el servicio no esté solo en localhost.
- Define un `SIGMA_SECRET_KEY` fuerte o deja que Sigma Algebra genere y guarde uno.
- Restringe el acceso a `instance/` y haz copias de seguridad cifradas.
- Mantén las dependencias actualizadas (Dependabot/Renovate) y revisa los avisos de
  seguridad de Python y npm.
- No ejecutes Sigma Algebra como administrador/root; usa un usuario sin privilegios.
