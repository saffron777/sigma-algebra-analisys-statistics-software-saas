# Σ Sigma

Plataforma web de análisis estadístico, hecha con **Flask** y **Tailwind CSS**, con arquitectura limpia.
Se ejecuta en su propio equipo: los datos, los resultados y el código que ejecuta nunca salen de él.

| Módulo | Qué hace |
|---|---|
| **Datos** | Importa desde **CSV, Excel, texto, SQL Server y APIs JSON**; vista previa, tipos de columna editables; todo se guarda en SQLite. |
| **Estadística descriptiva** | Posición, dispersión, forma, IC de la media, frecuencias, normalidad (Shapiro, D'Agostino), ajuste de distribuciones (KS/AIC), correlaciones con valor p. |
| **Distribuciones** | 15 distribuciones: densidad/masa, acumulada, cuantiles, probabilidad entre dos valores y **valor p con el área sombreada**. |
| **Pruebas de hipótesis** | 16 pruebas: z, t (1 muestra, 2 muestras Welch/agrupada, pareada), proporciones, exacta binomial, χ² (varianza, independencia, bondad de ajuste), F, Fisher, correlación, Mann-Whitney, Wilcoxon, Shapiro. Desde columnas o desde estadísticos resumidos. |
| **Regresión lineal** | Simple y múltiple (con variables categóricas), IC, ANOVA del modelo, VIF, supuestos (Durbin-Watson, Jarque-Bera, Breusch-Pagan), observaciones influyentes y predicción con IC/IP. |
| **ANOVA** | Un factor (clásico y Welch) y dos factores con interacción (SC tipo I/II/III), supuestos, Tukey HSD, η²/ω², Kruskal-Wallis. |
| **Análisis de mercados** | Encuestas y sondeos con muestreo probabilístico: tamaño de muestra, selección de muestras (simple, sistemático, estratificado, conglomerados), estimación con márgenes de error (Wald/Wilson/Agresti-Coull/Clopper-Pearson, pesos y efecto de diseño), tabulación cruzada con χ² y escalas Likert (alfa de Cronbach). |
| **Consola Python** | Editor de celdas con un **kernel Jupyter real** por notebook; abre, ejecuta y exporta archivos `.ipynb`. Acceso directo a sus datasets con `sigma.load()` / `sigma.save()`. |
| **Resultados guardados** | Botón **Guardar resultado** en cada módulo: se conservan parámetros, tablas y gráficos en SQLite; se pueden buscar, reabrir, editar, exportar a JSON y reabrir en su módulo. |

Cada resultado que depende de una distribución de referencia (t, F, χ², normal…) muestra la curva con la región de
rechazo y el valor p sombreados.

## Requisitos

- Windows 10/11 (también funciona en Linux/macOS), **Python 3.11+** y **Node.js 20+** (solo para compilar el CSS).
- Para importar desde **SQL Server**: *ODBC Driver 17 o 18 for SQL Server* instalado.

## Instalación

```powershell
.\scripts\setup.ps1
```

El script crea `.venv`, instala las dependencias, compila Tailwind, copia las librerías del navegador
(Plotly, Alpine, marked, DOMPurify) a `static/vendor` —así funciona sin internet— y carga los datos de ejemplo.

Manual, paso a paso:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
npm install
npm run build
.\.venv\Scripts\python.exe scripts\make_sample_data.py
.\.venv\Scripts\python.exe scripts\load_sample_data.py
```

## Ejecución

```powershell
.\scripts\run.ps1            # o:  .\.venv\Scripts\python.exe run.py
```

Abra <http://127.0.0.1:5000>. Para un servidor más robusto:

```powershell
.\.venv\Scripts\waitress-serve.exe --listen=127.0.0.1:5000 wsgi:app
```

Variables de entorno (opcionales, también en un archivo `.env`, ver `.env.example`):

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `SIGMA_HOST` / `SIGMA_PORT` | `127.0.0.1` / `5000` | Dirección de escucha. |
| `SIGMA_DB_PATH` | `instance/sigma.db` | Base SQLite (datasets + resultados). |
| `SIGMA_NOTEBOOKS_DIR` | `instance/notebooks` | Carpeta de los `.ipynb`. |
| `SIGMA_MAX_UPLOAD_MB` | `100` | Tamaño máximo de los archivos subidos. |
| `SIGMA_CELL_TIMEOUT_SECONDS` | `60` | Tiempo máximo por celda de la consola. |
| `SIGMA_KERNEL_IDLE_MINUTES` | `30` | Los kernels inactivos se cierran pasado este tiempo. |
| `SIGMA_DEBUG` | — | `1` activa la recarga automática (solo desarrollo). |

## Datos de ejemplo

`sample_data/` incluye `ventas.csv`, `marco_muestral.csv`, `encuesta.csv` y `analisis_demo.ipynb`
(`python scripts/make_sample_data.py` los regenera; `python scripts/load_sample_data.py` carga los CSV en Sigma).
Para probar la consola, importe `analisis_demo.ipynb` desde **Consola Python → Importar .ipynb**.

## Fuentes de datos

- **CSV / texto**: delimitador, codificación y separador decimal se detectan solos (editables). El texto admite
  columnas de ancho fijo.
- **Excel** (`.xlsx`, `.xls`): elija la hoja y la fila de encabezados.
- **SQL Server**: servidor, base, autenticación Windows o SQL; consulta o tabla completa. Es **solo lectura**: solo se
  aceptan `SELECT`/`WITH` (se rechazan `INSERT`, `UPDATE`, `DELETE`, `DROP`, `EXEC`, `SELECT … INTO`, varias sentencias…),
  se conecta con `ApplicationIntent=ReadOnly` y nada se confirma en el servidor. La contraseña **no se guarda**. Si el
  servidor usa un certificado autofirmado marque *Confiar en el certificado del servidor*. Para mayor seguridad use una
  cuenta con permisos solo de lectura.
- **API JSON**: URL, método GET/POST, cabeceras, parámetros, cuerpo y ruta a los registros (`data.items`); si la omite,
  Sigma detecta la lista de objetos. Cabeceras y tokens no se guardan. Límite de 50 MB por respuesta.

## Persistencia y respaldo

Todo se guarda en `instance/sigma.db` (SQLite en modo WAL): los **datasets importados** (cada uno en su tabla, con
sus tipos) y los **resultados guardados**. Los notebooks se guardan como archivos `.ipynb` en `instance/notebooks/`.
**Respaldo:** copie la carpeta `instance/` (con la aplicación detenida, o copie `sigma.db`, `sigma.db-wal` y
`sigma.db-shm` juntos).

## Consola Python y notebooks

Cada notebook tiene su propio proceso de Python (kernel de Jupyter): el estado se mantiene entre celdas y un bucle
infinito o un fallo no afectan a Sigma. **Shift+Enter** ejecuta y avanza, **Ctrl+Enter** ejecuta y se queda, **Ctrl+S**
guarda (también hay guardado automático). Si una celda excede el tiempo máximo se interrumpe; si el kernel no responde se
reinicia. Ya están disponibles `np`, `pd`, `stats` (scipy), `plt` y `sigma`:

```python
sigma.datasets()  # lista los datasets de Sigma
df = sigma.load("ventas")  # un dataset como DataFrame
sigma.save(df, "ventas_limpias")  # guarda un DataFrame como dataset de Sigma
```

## Seguridad

Sigma está pensado para **uso local de un solo usuario** y la consola ejecuta código arbitrario por diseño, así que:

- Escucha solo en `127.0.0.1`. **No lo exponga a la red ni a internet sin añadir autenticación.**
- Rechaza peticiones con cabecera `Host` u `Origin` ajenas (DNS-rebinding / otras páginas web) y exige un token CSRF
  en toda petición que modifica datos.
- Política de seguridad de contenido (CSP) restrictiva; no se carga ningún recurso externo.
- Los resultados importados, el HTML de los notebooks y los datos se muestran saneados (DOMPurify / `textContent`).
- Las sentencias SQLite usan parámetros y los nombres de tabla se generan desde identificadores internos.

## Estructura

Arquitectura limpia: `presentation → application → domain` e `infrastructure → application/domain`. Detalle y
decisiones en [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
sigma/
  domain/           estadística pura (numpy/scipy/statsmodels), entidades y errores; sin Flask ni E/S
  application/      casos de uso y puertos (interfaces)
  infrastructure/   SQLite, importadores (CSV/Excel/texto/SQL Server/API), kernel Jupyter, almacén de notebooks
  presentation/web/ blueprints Flask, esquemas pydantic, reportes y gráficos, plantillas Jinja, JS y Tailwind
tests/              unit/ (dominio, infraestructura) e integration/ (API, páginas, consola con kernel real)
```

## Desarrollo

```powershell
.\.venv\Scripts\python.exe -m pytest              # 237 pruebas (la consola arranca kernels reales: ~1 min)
.\.venv\Scripts\python.exe -m ruff check .        # lint
.\.venv\Scripts\python.exe -m ruff format .       # formato
npm run watch:css                                 # recompila Tailwind al editar plantillas/JS
$env:SIGMA_DEBUG = "1"; .\.venv\Scripts\python.exe run.py   # recarga automática
```

## Limitaciones conocidas

- Un solo usuario, sin autenticación. Sin ANOVA de medidas repetidas ni post-estratificación (raking).
- La salida de la consola se entrega al terminar la celda (sin transmisión en vivo).
- En Windows, una llamada bloqueante como `time.sleep(120)` no se puede interrumpir: tras el tiempo máximo el kernel se
  reinicia (se pierden las variables en memoria).
- La importación desde SQL Server no puede probarse automáticamente sin un servidor: las pruebas cubren la construcción
  de la conexión, la protección de solo lectura y toda la ruta de lectura contra SQLite.
