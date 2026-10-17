# Changelog

Todos los cambios relevantes de Sigma se documentan en este archivo. El formato sigue
[Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el proyecto usa
[versionado semántico](https://semver.org/lang/es/).

## [1.0.0] - 2026-10-17

Primera versión pública de Sigma como software libre.

### Añadido

- Licencia GNU AGPL-3.0-or-later, avisos de copyright y textos de terceros.
- Avisos legales visibles en la interfaz, con enlaces a la licencia, al código fuente y a
  los avisos de terceros.
- Empaquetado Docker y scripts de instalación para Windows, Linux y macOS.
- Integración continua, auditoría de licencias y comprobación de versiones.
- Guía de publicación, política de seguridad, privacidad, contribución y código de conducta.
- Definición de variables categóricas con código y etiqueta (leyenda) en el editor de
  datasets manuales, con mini-grilla, validación y persistencia del libro de códigos.

### Cambiado

- Metadatos del proyecto y versión 1.0.0.
- Instalación reproducible con `npm ci` y bloqueo de dependencias de Python.
- Documentación de instalación, despliegue, privacidad y marcas.
- La frase sobre privacidad aclara que la interpretación con IA es opcional y puede enviar
  el resumen del resultado al proveedor configurado.

### Corregido

- Pruebas de páginas desactualizadas y controles de exportación/limpieza en Resultados.
