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
| `domain` | Entidades, errores, resultados (`results.py`) y **toda la estadística** (`statistics/`, `distributions/`) | Flask, SQLAlchemy, sqlite3, requests, nada de E/S |
| `application` | Casos de uso (una clase por acción), comandos (`dto.py`) y **puertos** (`ports.py`, `typing.Protocol`) | Flask, adaptadores concretos |
| `infrastructure` | Adaptadores: SQLite, importadores, kernel Jupyter, almacén de `.ipynb` | Flask |
| `presentation/web` | Blueprints, esquemas pydantic, reportes/gráficos, plantillas, JS | Lógica estadística (solo la presenta) |
| `bootstrap.py` | **Composition root**: único lugar donde se eligen los adaptadores y se arman los casos de uso | — |

> El dominio sí importa numpy, pandas, scipy y statsmodels: son librerías numéricas puras, no infraestructura.
> Lo que el dominio nunca hace es leer/escribir disco o red.

## Flujo de una petición (regresión)

1. `POST /api/regression/run` → `blueprints/regression_api.py` valida el JSON con `schemas.RegressionRequest`
   (pydantic) y lo convierte en un `RegressionCommand`.
2. `application/use_cases/analysis.RunRegression` obtiene el dataset por el puerto `DatasetRepository` y llama a
   `domain/statistics/regression.fit_regression`.
3. El dominio devuelve un `RegressionResult` (dataclass con números y, aparte, los arrays de los gráficos marcados
   con `transient()`, que no se serializan).
4. `reports/regression.build_report` convierte el resultado en **bloques de reporte** (tablas, tiles, alertas,
   figuras) y `charts/regression_plots` dibuja las figuras Plotly. El gráfico de la distribución de referencia
   (F, t) lo dibuja siempre `charts/distribution_plot.distribution_figure`.
5. El blueprint responde `{"meta", "result", "report", "figures"}`; `static/js/report.js` pinta los bloques.

### Contrato de las respuestas de análisis

```jsonc
{
  "meta":    { "module", "analysis_type", "title", "dataset_id", "params" },  // params = lo que se envió
  "result":  { ... },   // resultado estructurado completo (todos los decimales), apto para exportar
  "report":  { "title", "subtitle", "blocks": [ {type: heading|tiles|table|text|alert|figure}, ... ] },
  "figures": [ { "data": [...], "layout": {...} } ]   // Plotly; los bloques "figure" apuntan por índice
}
```

Errores: `{"error": {"code", "message", "details"}}` con 404/422 (un único manejador traduce
`SigmaError` y los errores de validación de pydantic al español).

Como el reporte ya está en formato de presentación, **un resultado guardado se reabre idéntico** con el mismo
código de front-end (`report.js`).

## Persistencia

- **SQLite** (`instance/sigma.db`, modo WAL, claves foráneas activas, esquema versionado con `PRAGMA user_version`;
  `infrastructure/persistence/sqlite/database.py`).
- `datasets` guarda metadatos; los datos de cada dataset viven en su tabla `ds_<uuid>` (nombre generado desde un
  UUID, nunca desde texto del usuario). `frame_codec.py` conserva los dtypes de pandas (fechas, categorías, enteros
  nulables…) en el viaje de ida y vuelta.
- `saved_results` guarda `params`, `result`, `report` y `figures` en JSON; `dataset_id` es una FK
  `ON DELETE SET NULL` y `dataset_name` conserva el nombre aunque el dataset se borre.
- Notebooks: archivos `.ipynb` estándar (`FileNotebookStore`), escritura atómica.

## Consola de código

`JupyterKernelPool` (puerto `CodeExecutor`) lanza **un kernel ipykernel por notebook** con `jupyter_client`:

- Las celdas de un notebook se ejecutan de una en una (cerrojo por sesión); las salidas se normalizan al formato
  nbformat (stream, display_data, execute_result, error) y se limitan a 2 MB.
- Tiempo máximo: se interrumpe el kernel; si no responde en 8 s se descarta y se reinicia en la siguiente celda.
- Un hilo cierra los kernels inactivos y `atexit` cierra todos al salir.
- `infrastructure/kernel/helpers.py` es el objeto `sigma` del kernel; reutiliza `SqliteDatasetRepository` (SQLite WAL
  permite lectura/escritura concurrente con la aplicación).

## Seguridad

Ver `presentation/web/security.py`: lista de `Host` permitidos, comprobación de `Origin`, token CSRF
(`X-CSRF-Token`) en toda petición mutante, cabeceras y CSP. El HTML procedente de notebooks o datos se sanea con
DOMPurify o se inserta con `textContent`. Las importaciones SQL pasan por `importers/sql_guard.py`.

## Cómo extender

- **Nueva prueba de hipótesis**: añadir una entrada a `SPECS` en `domain/statistics/hypothesis.py` (roles, parámetros,
  `prepare`, `compute`). El formulario del navegador se genera solo desde el catálogo (`/api/hypothesis/catalog`).
- **Nueva distribución**: una entrada en `domain/distributions/registry.py`.
- **Nueva fuente de datos**: una clase con `source_type` y `read(request)` en `infrastructure/importers/` y
  registrarla en `registry.py`; añadir su formulario en `templates/pages/datos.html`.
- **Nuevo módulo de análisis**: función de dominio + resultado en `results.py` → comando en `dto.py` → caso de uso →
  esquema pydantic → reporte/gráficos → blueprint → página (reutilizando `analysisPage()` y la macro `result_area`) →
  registrar en `bootstrap.py`. El botón *Guardar resultado* y la página *Resultados guardados* funcionan sin más
  cambios (solo hay que añadir el módulo a `domain/saved_results.MODULES`).
- **Cambiar la base de datos**: implementar `DatasetRepository`/`ResultRepository` y cambiarlo en `bootstrap.py`.

## Pruebas

- `tests/unit/domain`: cada resultado estadístico se contrasta con `scipy`/`statsmodels` o con valores de referencia
  (n = 385 para p = 0,5; e = 5 %; 95 %, etc.).
- `tests/unit/infrastructure`: SQLite (dtypes, restricciones, rendimiento), importadores (incluido un servidor HTTP
  local real para la API JSON) y la protección de solo lectura de SQL.
- `tests/integration`: la API completa con el cliente de pruebas de Flask (incluidas CSRF/Host), las páginas y la consola
  con **kernels reales**.
