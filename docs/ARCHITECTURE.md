# Arquitectura de Sigma

Sigma sigue **Clean Architecture** adaptada a Flask. La regla es una sola: **las dependencias apuntan hacia
dentro**.

```
presentation/web ──► application ──► domain
        ▲                ▲
        └──── infrastructure ┘        (infrastructure implementa los puertos de application)
```

| Capa | Contiene | No debe importar |
|---|---|---|
| `domain` | Entidades, errores, resultados, **toda la estadística** (`statistics/`, `distributions/`), el modelo de documentos de los informes, el prompt de la IA, el informe de un notebook | Flask, SQLAlchemy, sqlite3, requests, nada de E/S |
| `application` | Casos de uso (una clase por acción), comandos (`dto.py`) y **puertos** (`ports.py`, `typing.Protocol`) | Flask, adaptadores concretos |
| `infrastructure` | Adaptadores: SQLite, importadores, kernel Jupyter, almacén de `.ipynb`, exportadores (PDF/Word/Excel), cliente de Anthropic, seguridad (hash, limitador) | Flask |
| `presentation/web` | Blueprints, esquemas pydantic, reportes/gráficos, plantillas, JS | Lógica estadística (solo la presenta) |
| `bootstrap.py` | **Composition root**: único lugar donde se eligen los adaptadores y se arman los casos de uso | — |

> El dominio sí importa numpy, pandas, scipy y statsmodels: son librerías numéricas puras, no infraestructura.
> Lo que el dominio nunca hace es leer/escribir disco o red.

## Cuentas y aislamiento por usuario

`bootstrap.build_container` crea lo **compartido** (base de datos, importadores, pool de kernels, exportadores, ajustes
de la IA, casos de uso de cuentas). `Container.for_user(user)` arma un `UserServices` con **todos los casos de uso ya
acotados a esa cuenta**: los repositorios reciben el `user_id` y lo incluyen en *cada* consulta, de modo que el id de un
recurso ajeno se comporta exactamente como uno inexistente (404, nunca «prohibido»). Los blueprints solo ven
`g.services` (`_helpers.container()`).

- `presentation/web/auth.py`: guardia global (`before_request`): sin sesión, las páginas redirigen a `/login?next=…`
  y la API responde 401; `/login`, `/setup`, `/health` y `/static` son públicos. La sesión es una cookie firmada con
  `uid`, `epoch` y `csrf`; `users.auth_epoch` sube al cambiar la contraseña, el rol o el estado, lo que invalida las
  demás sesiones. Iniciar sesión renueva la sesión y el token CSRF.
- Contraseñas con *scrypt* (werkzeug), mensajes de error genéricos, igualación de tiempos con un hash ficticio y un
  limitador en memoria (5 fallos → bloqueo; también por IP).
- Roles `admin`/`user`; permisos por cuenta `can_use_console` y `can_use_ai`; siempre queda un administrador activo.
- La **consola** ejecuta código con los permisos del sistema operativo: no hay aislamiento real entre usuarios con
  consola, por eso es un permiso explícito. Los kernels se identifican por `(usuario, notebook)`, `SIGMA_USER_ID` acota
  `sigma.load/save` a la cuenta y `ANTHROPIC_API_KEY` se retira del entorno del kernel.

## Flujo de una petición (regresión)

1. `POST /api/regression/run` → `blueprints/regression_api.py` valida el JSON con `schemas.RegressionRequest`
   (pydantic) y lo convierte en un `RegressionCommand`.
2. `application/use_cases/analysis.RunRegression` obtiene el dataset por el puerto `DatasetRepository` (el del usuario
   actual) y llama a `domain/statistics/regression.fit_regression`.
3. El dominio devuelve un `RegressionResult` (dataclass con números y, aparte, los arrays de los gráficos marcados
   con `transient()`, que no se serializan).
4. `reports/regression.build_report` convierte el resultado en **bloques de reporte** (tablas, tiles, alertas,
   figuras) y `charts/regression_plots` dibuja las figuras Plotly.
5. El blueprint responde `{"meta", "result", "report", "figures"}`; `static/js/report.js` pinta los bloques.

### Contrato de las respuestas de análisis

```jsonc
{
  "meta":    { "module", "analysis_type", "title", "dataset_id", "params" },
  "result":  { ... },   // resultado estructurado completo (todos los decimales)
  "report":  { "title", "subtitle", "blocks": [ {type: heading|tiles|table|text|alert|figure}, ... ] },
  "figures": [ { "data": [...], "layout": {...} } ]   // Plotly; los bloques "figure" apuntan por índice
}
```

Errores: `{"error": {"code", "message", "details"}}` con 401/403/404/409/422/429/503 (un único manejador traduce
`SigmaError` y los errores de validación de pydantic al español).

Como el reporte ya está en formato de presentación, **un resultado guardado se reabre idéntico** y **los
exportadores consumen los mismos bloques** que la pantalla.

## Persistencia

SQLite (`instance/sigma.db`, modo WAL, claves foráneas activas). El esquema se versiona con migraciones SQL
(`infrastructure/persistence/sqlite/migrations.py`, `PRAGMA user_version`; cada una se aplica de forma atómica y no
se edita una publicada):

| Versión | Contenido |
|---|---|
| v1 | `datasets` y `saved_results` (versión de un solo usuario) |
| v2 | `users`, `user_id` en datasets/resultados (nombre único **por usuario**), `sync_state` y *triggers* que suben la revisión de cada ámbito |
| v3 | `workspace`: formulario y último resultado de cada módulo |
| v4 | `import_drafts`: importaciones pendientes (archivo en un BLOB + tabla `stg_<id>` con la vista previa) |
| v5 | `report_templates` (colores, textos, página, logotipo) |
| v6 | `app_settings` (modelo e interruptor de la IA), `ai_usage`, `saved_results.interpretation` |

- Los datos de cada dataset viven en su tabla `ds_<uuid>` (nombre generado desde un UUID, nunca desde texto del usuario);
  `frame_codec.py` conserva los dtypes de pandas. Las tablas `stg_<uuid>` de los borradores usan el mismo códec.
- Al adoptar los datos de la versión anterior (primer administrador) las filas sin dueño y los `.ipynb` pasan a esa
  cuenta; borrar una cuenta elimina sus filas (cascada), sus tablas de datos, sus borradores y su carpeta de notebooks.
- Notebooks: archivos `.ipynb` estándar por usuario (`instance/notebooks/<id>/`), escritura atómica y **revisión**
  (mtime+tamaño) para detectar guardados concurrentes (`409`).

## Espacio de trabajo y sincronización

- `workspace(user, module)` guarda `state` (el formulario, pequeño, con *debounce*) y `payload` (el último resultado,
  grande, tras cada ejecución) con una **revisión que solo crece**; «Limpiar» vacía el módulo sin retroceder la revisión.
  La interpretación de la IA forma parte del `payload`.
- `GET /api/sync` devuelve las revisiones de datasets, resultados, borradores, plantillas, notebooks y de cada módulo
  (una consulta a `sync_state` + `workspace` + `stat` de la carpeta de notebooks). Las revisiones de `sync_state` las suben
  *triggers* de SQLite, así que también cuentan las escrituras de la consola (`sigma.save`).
- Front-end: `sync.js` consulta cada ~3 s mientras la pestaña está visible; `workspace.js` (mixin) restaura al
  abrir, guarda al editar, y ante un cambio ajeno aplica la versión nueva o, con ediciones sin guardar, muestra el
  aviso *Cargar la versión nueva / Mantener la mía* **sin sobrescribir en silencio**.

## Importaciones pendientes

`PreviewImport` ya no usa memoria: `SqliteImportDraftStore` (puerto `ImportDraftStore`) guarda el archivo subido y,
tras leerlo, la tabla parseada; hay un borrador por usuario y tipo de fuente. Cambiar las opciones relee el archivo
guardado (`draft_id`); de SQL Server y de las APIs solo se conserva el *resultado* (`safe_options` descarta
contraseñas, cabeceras, cuerpos, parámetros y la *query string* de las URL). Se limpian al guardar, al quitar, con
«Limpiar», al borrar la cuenta y al arrancar (más de 14 días).

## Informes: exportar e imprimir

`domain/documents.py` define `ReportDocument → ReportSection → bloques`: los de la pantalla más `list`, `code`, `output`,
`pagebreak` y `markdown`. El navegador envía los bloques ya renderizados y **cada gráfico como PNG** dibujado con el
tema claro (`Plotly.toImage`); `clean_blocks` sanea todo lo que llega (tipos conocidos, tamaños acotados) y
`PillowImageProcessor` valida los PNG. Tres adaptadores implementan `DocumentExporter` y comparten `common.py`:

- `pdf_exporter` (reportlab, fuentes DejaVu de matplotlib): portada, índice, marcadores, cabecera/pie con la plantilla
  y «Página n de N» (lienzo de dos pasadas); todo texto pasa por *escape* antes de llegar a un `Paragraph`.
- `docx_exporter` (python-docx): estilos de encabezado, tablas con cabecera repetida, campo de índice y de página.
- `xlsx_exporter` (openpyxl): «Índice», «Informe», una hoja por tabla (números como números, texto nunca como fórmula),
  «Gráficos» y hojas de datos completos.

`DocumentAssembler` aplica la plantilla del usuario (o la predeterminada, o la marca de Sigma) y las opciones del
diálogo. Casos de uso: `ExportReport` (resultado/informe combinado), `ExportDatasetReport` (ficha; Excel con todas las
filas) y `ExportNotebookReport` (`domain/notebook_report.py`: texto, código y salidas —texto, tablas HTML, PNG—).
La impresión usa CSS `@media print`, gráficos congelados como imágenes blancas y, para el notebook, un diálogo con
`@page` y márgenes de página (`notebook-print.js`).

## Interpretación con IA (opcional)

`domain/interpretation.py` convierte el reporte en texto compacto y construye el prompt (criterios por módulo, datos
entre `<resultado>…</resultado>`, regla de ignorar instrucciones dentro de ellos). `InterpretResult` comprueba permiso,
interruptor, clave y límite horario, y registra el consumo; el puerto `AiGateway` lo implementa `AnthropicGateway`
(SDK oficial, `claude-opus-5-5`, esfuerzo medio, sin parámetros de muestreo, reserva a `claude-opus-4-8`, errores
traducidos sin exponer la clave). Solo viaja el resumen del resultado: la ficha de un dataset la construye el servidor y
omite filas y valores de columnas tipo identificador. La clave (`ANTHROPIC_API_KEY` o `instance/ai_key.txt`) nunca sale
del servidor.

## Consola de código

`JupyterKernelPool` (puerto `CodeExecutor`) lanza **un kernel ipykernel por notebook y usuario** con `jupyter_client`:

- Las celdas de un notebook se ejecutan de una en una (cerrojo por sesión); las salidas se normalizan al formato
  nbformat y se limitan a 2 MB. Límites por usuario y totales, con expulsión de los inactivos más antiguos.
- Tiempo máximo: se interrumpe el kernel; si no responde en 8 s se descarta y se reinicia en la siguiente celda.
- `infrastructure/kernel/helpers.py` es el objeto `sigma` del kernel; reutiliza `SqliteDatasetRepository(user_id)`.

## Seguridad

`presentation/web/security.py`: lista de `Host` permitidos, comprobación de `Origin`, token CSRF (`X-CSRF-Token`) en
toda petición mutante, cabeceras, CSP restrictiva y `Cache-Control: no-store` salvo en `/static`. El HTML procedente de
notebooks, datos o de la IA se sanea con DOMPurify o se inserta con `textContent`. Las importaciones SQL pasan por
`importers/sql_guard.py`. Los logotipos se validan, se reescalan y se guardan como PNG (nunca SVG).

## Identidad visual

`scripts/make_logo.py` genera todo el logotipo (SVG, PNG, ICO) desde una sola geometría; la variante compacta se usa a
tamaño de favicon.

## Cómo extender

- **Nueva prueba de hipótesis**: una entrada en `SPECS` de `domain/statistics/hypothesis.py`; el formulario se genera
  solo desde el catálogo (`/api/hypothesis/catalog`).
- **Nueva distribución**: una entrada en `domain/distributions/registry.py`.
- **Nueva fuente de datos**: una clase con `source_type` y `read(request)` en `infrastructure/importers/`, registrarla
  en `registry.py` y añadir su formulario en `templates/pages/datos.html` (y sus secretos en `SECRET_OPTIONS`).
- **Nuevo módulo de análisis**: función de dominio + resultado → comando en `dto.py` → caso de uso → esquema pydantic →
  reporte/gráficos → blueprint → página (reutilizando `analysisPage(modulo, campos)` y `result_area`, que ya traen
  guardar, exportar, imprimir e interpretar) → registrar en `bootstrap.py`, en `domain/saved_results.MODULES`, en
  `domain/workspace.WORKSPACE_MODULES` y en `domain/interpretation.MODULES`.
- **Nuevo formato de exportación**: una clase con `format`, `content_type`, `extension` y `export(document)` en
  `infrastructure/exporters/`, añadida al `ExporterRegistry` de `bootstrap.py`.
- **Cambiar la base de datos o el proveedor de IA**: implementar los puertos (`DatasetRepository`, `ResultRepository`,
  `AiGateway`…) y cambiarlos en `bootstrap.py`.

## Pruebas

- `tests/unit/domain`: cada resultado estadístico se contrasta con `scipy`/`statsmodels` o con valores de referencia;
  documentos, Markdown, notebook, ficha e interpretación.
- `tests/unit/infrastructure`: SQLite (migraciones, aislamiento, *triggers*, borradores), importadores (con un servidor
  HTTP local real), exportadores (PDF con `pypdf`, DOCX y XLSX reabiertos), adaptador de Anthropic con un cliente
  simulado y los activos del logotipo.
- `tests/integration`: la API completa con el cliente de pruebas de Flask (autenticación, matriz de aislamiento entre
  cuentas, CSRF/Host, espacio de trabajo y sincronización, borradores, exportación, plantillas, IA), las páginas y la
  consola con **kernels reales**. `pytest -m live` activa la prueba opcional contra el servicio real de IA.
