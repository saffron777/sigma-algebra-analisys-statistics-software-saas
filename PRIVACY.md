# Privacidad y tratamiento de datos

Sigma está diseñado para ejecutarse en un servidor propio. Esta página describe qué datos
se guardan y cuándo pueden salir del servidor.

## Qué se guarda y dónde

- **Datos, resultados, notebooks y preferencias** se guardan en la base de datos configurada
  (SQLite en `instance/` por defecto) y en `instance/notebooks/`.
- **Claves de sesión**: `instance/secret.key` se genera al azar en la primera ejecución y
  nunca se publica.
- **Claves de proveedores de IA** (opcionales): se guardan en `instance/ai.json` con
  permisos restringidos, o se leen de variables de entorno del servidor. No se envían al
  navegador ni se muestran completas.
- **Importaciones pendientes** (archivos y vistas previas) se guardan temporalmente en
  `instance/` y se descartan según `SIGMA_DRAFT_RETENTION_DAYS`.

El administrador del servidor es responsable de las copias de seguridad y del acceso al
sistema de archivos.

## Qué sale del servidor

- **No hay telemetría, analítica ni comprobación de actualizaciones.** Sigma no llama a
  servicios externos por su cuenta.
- Las **importaciones desde bases de datos o APIs** solo se ejecutan cuando el usuario las
  pide.
- La **interpretación con IA es opcional**. Solo se activa cuando el usuario pulsa el botón
  y, en ese momento, se envía al proveedor configurado el **resumen del resultado**
  (parámetros, cifras, tablas recortadas y avisos), nunca el dataset completo ni las filas.
  La primera vez se pide consentimiento. El proveedor puede estar en otro país y se rige por
  sus propias condiciones y política de privacidad. Los modelos locales (Ollama, LM Studio)
  no envían nada a internet.
- El **registro del servidor** puede contener rutas, tiempos y errores. No incluye
  contraseñas ni claves (el código las enmascara en los mensajes de error).

## Cookies y sesión

Sigma usa una cookie de sesión para mantener la cuenta autenticada y una preferencia local
del navegador para el tema claro/oscuro. No usa cookies publicitarias ni de seguimiento.

## Derechos y contacto

Para ejercer derechos sobre los datos alojados en una instalación de Sigma, contacta con el
administrador de esa instalación. Para cuestiones sobre el proyecto, abre una incidencia en
el repositorio público.
