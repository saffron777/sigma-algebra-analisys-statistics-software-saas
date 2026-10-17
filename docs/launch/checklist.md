# Checklist de lanzamiento

## Antes de publicar

- [ ] Nombre oficial decidido y aplicado: **Sigma Algebra Analisys Statistics Software (SAAS)**;
      forma corta **Sigma Algebra**.
- [ ] Historial de Git limpio: `.gitconfig.backup` no aparece en ningún commit; sin rutas de clientes.
- [ ] Repositorio público creado y README renderizado sin enlaces roto.
- [ ] Release `v1.0.0` publicada con los artefactos y `SHA256SUMS`.
- [ ] Imagen `ghcr.io/saffron777/sigma-algebra-analisys-statistics-software-saas` pública.
- [ ] Demo HTTPS estable, con datos de ejemplo, banner y reinicio; `/demo` funciona.
- [ ] Importaciones, consola, IA y administración bloqueadas en la demo.
- [ ] 3 capturas + GIF en el README y en la landing.
- [ ] `ruff check .`, tests, `collect_licenses.py check` y CI en verde.
- [ ] Sin secretos, `.env`, `instance/` ni datos personales en el repositorio/artefactos.
- [ ] Enlaces de demo/source/licencia comprobados desde una ventana incógnito.
- [ ] Textos de Show HN/Reddit/LinkedIn revisados y con la URL real de demo.

## Secuencia recomendada

1. **Lanzamiento suave (semana -1):** envía el enlace a 15–20 profesores/analistas, uno a uno.
   Corrige la fricción de instalación y los errores de onboarding.
2. **Directorios:** envía las fichas a [awesome-selfhosted](../listings/awesome-selfhosted.md),
   [selfh.st](../listings/selfh.st.md) y
   [Free Software Directory](../listings/free-software-directory.md).
3. **Show HN:** martes–jueves, 8–10 h (hora de la costa este de EE. UU.). Responde durante 48 h.
4. **LinkedIn y Mastodon/Bluesky:** el mismo día o el siguiente, en español, con GIF.
5. **Reddit:** r/selfhosted primero; r/Python al día siguiente; r/statistics solo si las reglas lo
   permiten. No publiques el mismo texto en todos a la vez.
6. **Después:** blog «qué aprendí lanzando Sigma Algebra», propuesta a JOSS/Zenodo y búsqueda de 2–3 casos
   de uso reales.

## Qué responder

- **«¿Es un SPSS gratis?»** No exactamente: cubre pruebas, modelos, encuestas, calidad e informes,
  pero no la sintaxis de SPSS ni todos sus procedimientos. Es una alternativa autoalojada para
  docencia y equipos pequeños; sé claro con las limitaciones.
- **«¿Por qué AGPL?»** Para que las mejoras de una versión servida por red vuelvan a la comunidad.
  El uso interno y la autoinstalación son libres; la obligación de ofrecer el código surge al
  ofrecer una versión modificada como servicio de red. No des asesoría legal.
- **«¿Es seguro?»** Cuentas por usuario, CSRF, cookies seguras con HTTPS, sin telemetría; la demo
  bloquea importaciones/consola/IA y reinicia. En producción, HTTPS y copias de seguridad.
- **«¿Habrá inglés?»** El README y la landing ya están en inglés; la traducción completa de la UI es
  el siguiente paso. Pide ayuda si alguien quiere contribuir.

## Métricas (no solo estrellas)

- Clones/descargas de releases y `docker pull`.
- Visitas a la demo y tiempo de uso.
- Issues/discusiones de usuarios reales y solicitudes de taller.
- Contribuidores y `good first issue` resueltos.
- Citas/DOI y uso en cursos.

Usa enlaces UTM en los posts y una analítica respetuosa en la landing (Plausible/Umami), nunca
telemetría dentro de la app.

## Reglas de comunidad

- Di siempre que eres el autor.
- No pidas votos ni uses cuentas falsas.
- Lee las reglas de cada subreddit/foro antes de publicar.
- No uses logos de SPSS/Minitab ni afirmes afiliación.
- Responde a las críticas con datos; no discutas por la licencia.
