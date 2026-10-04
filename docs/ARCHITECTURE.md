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

Todo lo que Sigma guarda pasa por **puertos** (`application/ports.py`: `UserRepository`, `DatasetRepository`,
`ResultRepository`, `WorkspaceRepository`, `SyncRepository`, `ImportDraftStore`, `TemplateRepository`,
`SettingsRepository`, `AiUsageRepository`, `NotebookStore`…) y el conjunto de adaptadores de **una** base de datos es un
`Backend` (`persistence/backend.py: create_backend(settings)`). El administrador elige el motor; `bootstrap.py` solo
pide al backend sus repositorios (`backend.datasets(user_id)`…), así que casos de uso, API y front-end no saben qué motor hay.

| Backend | Motores | Implementación |
|---|---|---|
| `SqlBackend` (`persistence/sql/`) | SQLite · SQL Server · MySQL/MariaDB · PostgreSQL · Oracle | SQLAlchemy Core: el esquema es código (`schema.py`) y cada repositorio es el mismo para todos los motores |
| `MongoBackend` (`persistence/mongo/`) | MongoDB | `pymongo`: un documento por fila del esquema SQL (mismos campos y valores) |

**Qué base se usa** (`persistence/database_settings.py`): `SIGMA_DATABASE_URL` > `instance/database.json` (lo escribe el
asistente o la página *Base de datos*; contiene la contraseña, nunca se devuelve por la API) > SQLite en
`instance/sigma.db`. La configuración no puede vivir en la propia base, de ahí el archivo. Lo que llega del navegador
(`DatabaseSettings.from_request`) se valida de forma estricta: SQLite siempre significa el archivo del servidor, solo
pasan las opciones de una lista blanca por motor y una contraseña en blanco solo se conserva si no cambian motor,
servidor, puerto y usuario (un servidor nuevo nunca recibe la contraseña del anterior).

**Reglas de portabilidad del esquema SQL** (por qué funciona igual en cinco motores):

- Sin *triggers*: los repositorios suben la revisión de `sync_state` dentro de la misma transacción que escribe (`helpers.bump_sync`;
  también cuenta `sigma.save` desde un kernel, que usa los mismos repositorios).
- Sin `COLLATE NOCASE`: la unicidad sin distinguir mayúsculas usa una columna `*_key` en minúsculas (`username_key`,
  `name_key`; con `utf8mb4_bin` en MySQL para no confundir acentos).
- *Upserts* atómicos sin depender de un dialecto: `ON CONFLICT` (SQLite/PostgreSQL), `ON DUPLICATE KEY` (MySQL) o
  `UPDATE` + `INSERT` con reintento (SQL Server/Oracle) — `helpers.upsert`.
- Sin cascadas múltiples: `ON DELETE CASCADE` solo desde `users`; el `SET NULL` de `saved_results.dataset_id` lo hace el repositorio.
- Fechas ISO-8601 en texto, booleanos 0/1, texto largo `LongText` (`NVARCHAR(MAX)`/`LONGTEXT`/`CLOB`), binarios en trozos de 1 MiB
  (el controlador de MySQL envía los bytes en hexadecimal: así ninguna sentencia se acerca a `max_allowed_packet`).
- Oracle guarda `''` como `NULL`: las columnas de texto que pueden estar vacías son anulables y se leen como `""`.
- Los ids enteros usan `IDENTITY` (y `AUTOINCREMENT` en SQLite: un id nunca se reutiliza, así la sesión de una cuenta borrada no
  puede acabar apuntando a otra).

**Datasets**: cada uno tiene su tabla con nombre generado (`ds_<24 hex>`, cabe en Oracle) guardado en `datasets.table_name`, columnas
físicas `c0…cN` más `_row` (orden y paginación) y el mapeo nombre→columna en `datasets.col_map`; así ningún nombre de columna
del usuario toca las reglas de identificadores del motor. `frame_codec.py` conserva los dtypes de pandas. Los datasets creados
por versiones anteriores (`col_map` nulo, columnas con su nombre original y orden por `rowid`) se siguen leyendo sin reescribirlos.
Límite de columnas por motor (`table_io.MAX_COLUMNS`), con un error claro. En MongoDB las filas van en documentos de unos miles
de filas (`frame_chunks`).

**Versionado del esquema**: la tabla `sigma_schema` guarda la versión (`SCHEMA_VERSION = 7`, la base portable). Una base vacía se
crea con `create_all`; un archivo SQLite de versiones anteriores (`PRAGMA user_version` 1–6, scripts históricos en
`sql/legacy_sqlite.py`) se actualiza en el sitio hasta la 7; una base con tablas ajenas se rechaza sin tocarla. Cambios futuros:
una función por versión en `engine._STEPS`.

**Notebooks**: con SQLite siguen siendo archivos `.ipynb` estándar (`instance/notebooks/<id>/`, revisión = mtime+tamaño); con los
demás motores viven en la tabla `notebooks` (`SqlNotebookStore`) o en una colección (`MongoNotebookStore`) con una revisión entera y
guardado por *compare-and-swap* (`UPDATE … WHERE revision = :base`), de modo que el `409` por guardados concurrentes funciona entre
procesos y servidores. Las reglas de nbformat son comunes (`notebook_format.py`).

**Cambiar de motor en caliente** (`presentation/web/switching.py`): se construye un `Container` nuevo sobre la base elegida, se
guarda `database.json` y se intercambia `app.extensions["sigma"]` (la guarda de autenticación lo lee en cada petición); el
anterior se cierra poco después. Si la nueva base no abre, la aplicación sigue como estaba.

**Migración** (`application/use_cases/migration.py: MigrateData`): cada backend expone un `DataTransfer` (`counts`, `read`,
`write`, `finish`, `clear`) con registros planos en el orden de dependencias de `MIGRATION_ENTITIES`. Se exige un destino vacío, se
copian los registros con sus ids (por eso las sesiones siguen valiendo), se ajustan los generadores de ids del servidor, se
verifican los recuentos y, si algo falla, el destino vuelve a quedar vacío. El origen solo se lee. La usan
`scripts/migrate_database.py` y la página *Base de datos* (`presentation/web/migration.py`: hilo en segundo plano con progreso,
puerta de mantenimiento que responde 503 al resto de usuarios y cierre de kernels). Las importaciones pendientes se copian sin su vista
previa (se reconstruye desde el archivo).

**Al adoptar los datos de la versión de un solo usuario** (primer administrador) las filas sin dueño y los `.ipynb` pasan a esa cuenta;
borrar una cuenta elimina sus filas (cascada), sus tablas de datos, sus borradores y sus notebooks.

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

## Lectura de bases de datos externas

La pestaña *Base de datos* de la pantalla Datos no es una fuente más sino un selector: cada motor es su propio `source_type`
(`sqlserver`, `mysql`, `postgresql`, `oracle`, `mongodb`), con su borrador pendiente y su bloque de opciones, y
`GET /api/datasets/import/databases` dice cuáles tienen controlador. `/import/inspect` ya existía para las hojas de Excel: aquí es el
**explorador** (`inspect(request)` devuelve tablas, esquemas y columnas, o colecciones y campos).

Seguridad (los importadores son los únicos sitios por donde el usuario hace llegar texto a un servidor ajeno):

- **Conexión**: los campos del formulario pasan por `DatabaseSettings.from_request` (caracteres estrictos, lista blanca de opciones) y la URL la
  construye SQLAlchemy; nada se concatena. Lista blanca opcional de servidores (`SIGMA_EXTERNAL_DB_HOSTS`, `common.check_host_allowed`).
- **Solo lectura, dos capas**. (1) `sql_scrub.py` lee el SQL como cada motor (comentarios anidados y `$tag$…$tag$` en PostgreSQL, `\` y `/*! */`
  en MySQL, `q'[…]'` en Oracle, corchetes en SQL Server) y deja solo lo que no es texto; `sql_guard.py` exige una única sentencia
  `SELECT/WITH` y rechaza palabras y funciones peligrosas por dialecto; una regex no basta (`SELECT $$'$$; DROP …`). (2) La sesión del servidor es de
  solo lectura (`SET SESSION TRANSACTION READ ONLY` en MySQL, `default_transaction_read_only=on` en PostgreSQL, `SET TRANSACTION READ ONLY` en
  Oracle) y los ajustes de los que depende el *scanner* se fuerzan (`standard_conforming_strings=on`, sin `NO_BACKSLASH_ESCAPES`/`ANSI_QUOTES`).
  MongoDB solo usa `find` y rechaza `$where`/`$function`/`$accumulator`/`$expr`. El tiempo de consulta está acotado (120 s) y el de conexión (10 s).
- **Secretos**: `SECRET_OPTIONS` quita la contraseña de lo que se guarda con el borrador y el espacio de trabajo no la conserva; los mensajes de
  error de los controladores (`database_errors.py`, compartido con la prueba de conexión) la ocultan.
- **Nombres**: se resuelven contra el catálogo ignorando mayúsculas (Oracle guarda en mayúsculas lo que no va entre comillas) y se citan con las
  reglas del motor; el tope de filas se pide al servidor (`LIMIT`, `ROWNUM`, `TOP`) y se vuelve a imponer al leer (`sql_fetch.py`).

## Catálogo de análisis (marco común)

Los módulos «de catálogo» (`/transformar`, `/graficos`, `/potencia`, `/remuestreo`, `/modelos`, `/anova-avanzado`, `/series`,
`/econometria`, `/multivariado`, `/aprendizaje`) no tienen blueprint, esquema ni JS propios: una sola página (`catalogo.html` +
`catalogo.js`) y una sola API (`/api/analysis/<módulo>/catalog|run|code`) los sirven a todos.

- **Dominio** `domain/analysis/`: `modules.py` es el **único listado de módulos** (clave de página = clave de espacio de trabajo =
  clave de resultado guardado = clave de interpretación); de él se derivan `saved_results.MODULES`, `workspace.WORKSPACE_MODULES`,
  `interpretation.MODULES/_CRITERIA`, `pages.PAGES`, el menú y la portada. `spec.py` define `AnalysisSpec` (roles de columna,
  parámetros, salidas que son datasets, función `compute`, `code` y `example`) y `prepare_inputs`, que valida lo que envía el usuario
  con mensajes en español. `compute(frame, roles, params)` es puro y devuelve un `AnalysisOutput` de **datos planos** (`output.py`:
  bloques `heading/tiles/table/text/alert/chart`, datasets ofrecidos y cifras para IA/guardado), sin Plotly; los gráficos se
  describen con los constructores de `charts.py`. Un área = un módulo Python (`power.py`, `series.py`…) que llama a
  `registry.register(...)` y se añade a `AREA_MODULES`.
- **Aplicación**: `RunAnalysis` valida, ejecuta mediante el puerto `AnalysisRunner`, comprueba las salidas y guarda los datasets
  pedidos (`save_as`, o `replace` para sustituir el original); `ListAnalyses` y `GetAnalysisCode` completan el catálogo.
- **Infraestructura** `compute/process_runner.py`: los análisis `heavy=True` corren en un **proceso aparte** (`spawn`) con tiempo
  máximo (`SIGMA_ANALYSIS_TIMEOUT_SECONDS`, 120 s) y un tope de ejecuciones simultáneas (`SIGMA_ANALYSIS_MAX_PARALLEL`, 2); los
  errores de dominio cruzan la frontera con su tipo y las advertencias numéricas relevantes llegan como aviso. Los ligeros corren en el hilo
  de la petición.
- **Presentación**: `charts/render.py` convierte las descripciones de gráficos a Plotly con el tema de Sigma; `reports/analysis.py`
  convierte la salida al contrato de bloques de informe (`ReportBuilder`: valores p, formato numérico, celdas con tono).
- Las pruebas añaden análisis internos (ids con `_`, ocultos del catálogo) con `SIGMA_EXTRA_ANALYSIS_MODULES`, variable que también
  heredan los procesos de cálculo.

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
- **Nueva fuente de datos**: una clase con `source_type` y `read(request)` (y, si ofrece un explorador, `inspect(request)`) en
  `infrastructure/importers/`, registrarla en `registry.py`, añadir su formulario en `templates/pages/datos.html`, sus opciones en
  `datos.js` (`defaultOptions`, `SECRET_FIELDS`) y sus secretos en `SECRET_OPTIONS`. Una base de datos nueva: un `ExternalSqlImporter`
  más (dialecto en `sql_scrub.py`/`sql_guard.py`, sesión de solo lectura en `external_engine`) o un importador propio como el de MongoDB.
- **Nuevo análisis de catálogo**: un `AnalysisSpec` (con `example=`, que la prueba de todo el catálogo ejecuta, y `code=` si puede
  mostrarse como Python) en el módulo de su área de `domain/analysis/`; el formulario, la ejecución, los datasets de salida, el guardado y
  la interpretación salen del marco. Una **nueva página de catálogo**: una entrada en `domain/analysis/modules.py` (+ su icono en `macros.html`).
- **Nuevo módulo de análisis a medida**: función de dominio + resultado → comando en `dto.py` → caso de uso → esquema pydantic →
  reporte/gráficos → blueprint → página (reutilizando `analysisPage(modulo, campos)` y `result_area`, que ya traen
  guardar, exportar, imprimir e interpretar) → registrar en `bootstrap.py`, en `domain/saved_results.MODULES`, en
  `domain/workspace.WORKSPACE_MODULES` y en `domain/interpretation.MODULES`.
- **Nuevo formato de exportación**: una clase con `format`, `content_type`, `extension` y `export(document)` en
  `infrastructure/exporters/`, añadida al `ExporterRegistry` de `bootstrap.py`.
- **Nuevo motor de base de datos**: un `Backend` (todos los repositorios + `transfer`) y una rama en `persistence/backend.py`;
  añadirlo a `ENGINES` (`database_settings.py`) y pasar la batería de `tests/conformance/` (se activa con su `SIGMA_TEST_*_URL`).
  Para un motor SQL basta, normalmente, con el dialecto de SQLAlchemy y un controlador.
- **Cambiar el proveedor de IA**: implementar `AiGateway` y cambiarlo en `bootstrap.py`.

## Pruebas

- `tests/unit/domain`: cada resultado estadístico se contrasta con `scipy`/`statsmodels` o con valores de referencia;
  documentos, Markdown, notebook, ficha e interpretación.
- `tests/conformance`: **la misma batería sobre cada motor** (cuentas, datasets con dtypes y paginación, resultados, espacio de
  trabajo y carreras de escritura, borradores con archivos grandes, plantillas, notebooks con conflictos, esquemas, prueba de conexión y
  migración en ambos sentidos). Siempre corre en SQLite y en MongoDB en memoria (`mongomock`); con `SIGMA_TEST_MSSQL_URL`,
  `SIGMA_TEST_MYSQL_URL`, `SIGMA_TEST_POSTGRES_URL`, `SIGMA_TEST_ORACLE_URL` y `SIGMA_TEST_MONGO_URL` también contra servidores reales
  (bases vacías y desechables: cada prueba empieza borrando sus tablas). `docker-compose.test.yml` levanta PostgreSQL, MariaDB,
  MySQL, Oracle y MongoDB para ello. Con `SIGMA_TEST_APP_ENGINE=postgresql` (…) la suite de `integration/` completa corre sobre ese motor.
- `tests/unit/infrastructure`: SQLite (actualización de archivos antiguos, aislamiento, borradores), `DatabaseSettings`, importadores (con un servidor
  HTTP local real), exportadores (PDF con `pypdf`, DOCX y XLSX reabiertos), adaptador de Anthropic con un cliente
  simulado y los activos del logotipo.
- `tests/integration`: la API completa con el cliente de pruebas de Flask (autenticación, matriz de aislamiento entre
  cuentas, CSRF/Host, espacio de trabajo y sincronización, borradores, exportación, plantillas, IA), las páginas y la
  consola con **kernels reales**. `pytest -m live` activa la prueba opcional contra el servicio real de IA.
