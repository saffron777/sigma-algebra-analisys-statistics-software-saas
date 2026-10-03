<p align="center"><img src="docs/logo.png" alt="Sigma" height="90"></p>

# Sigma

Plataforma web de análisis estadístico, hecha con **Flask** y **Tailwind CSS**, con arquitectura limpia.
Funciona con **cuentas de usuario**: cada persona ve únicamente sus propios datasets, resultados y notebooks.
Todo se ejecuta y se guarda en su propio servidor; nada sale de él.

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
- Para importar desde **SQL Server**, o para usarlo como base de Sigma: *ODBC Driver 17 o 18 for SQL Server* instalado.
- Para guardar en MySQL/MariaDB, PostgreSQL, Oracle o MongoDB basta el controlador de Python de `requirements.txt` (ya se instala); necesita un servidor accesible.

## Instalación

```powershell
.\scripts\setup.ps1
```

El script crea `.venv`, instala las dependencias, compila Tailwind, copia las librerías del navegador
(Plotly, Alpine, marked, DOMPurify) a `static/vendor` —así funciona sin internet— y genera los archivos de ejemplo.

Manual, paso a paso:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
npm install
npm run build
.\.venv\Scripts\python.exe scripts\make_sample_data.py
```

## Ejecución

```powershell
.\scripts\run.ps1            # o:  .\.venv\Scripts\python.exe run.py
```

Abra <http://127.0.0.1:5000>. La primera vez Sigma pide **crear la cuenta de administrador** (`/setup`); si ya había
datos de la versión de un solo usuario, esa cuenta los hereda. Para un servidor más robusto:

```powershell
.\.venv\Scripts\waitress-serve.exe --listen=127.0.0.1:5000 wsgi:app
```

Variables de entorno (opcionales, también en un archivo `.env`, ver `.env.example`):

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `SIGMA_HOST` / `SIGMA_PORT` | `127.0.0.1` / `5000` | Dirección de escucha. |
| `SIGMA_DATABASE_URL` | — | Base de datos de Sigma (SQLite, SQL Server, MySQL/MariaDB, PostgreSQL, Oracle o MongoDB); ver *Base de datos*. |
| `SIGMA_DB_PATH` | `instance/sigma.db` | Archivo SQLite que se usa cuando no se ha elegido otra base. |
| `SIGMA_NOTEBOOKS_DIR` | `instance/notebooks` | Carpeta de los `.ipynb`. |
| `SIGMA_MAX_UPLOAD_MB` | `100` | Tamaño máximo de los archivos subidos. |
| `SIGMA_CELL_TIMEOUT_SECONDS` | `60` | Tiempo máximo por celda de la consola. |
| `SIGMA_KERNEL_IDLE_MINUTES` | `30` | Los kernels inactivos se cierran pasado este tiempo. |
| `SIGMA_SESSION_HOURS` | `12` | Duración de una sesión abierta. |
| `SIGMA_SECURE_COOKIES` | — | `1` si Sigma se sirve por HTTPS (la cookie de sesión solo viaja cifrada). |
| `SIGMA_ALLOWED_HOSTS` | — | Nombres extra con los que se accede (uso en red). |
| `SIGMA_MAX_KERNELS_PER_USER` / `SIGMA_MAX_KERNELS_TOTAL` | `3` / `8` | Límite de kernels simultáneos. |
| `SIGMA_DRAFT_RETENTION_DAYS` | `14` | Días tras los que se descarta una importación pendiente sin tocar. |
| `SIGMA_AI_REQUESTS_PER_HOUR` | `20` | Interpretaciones con IA por persona y hora. |
| `ANTHROPIC_API_KEY` | — | Clave de API de la IA (alternativa a guardarla desde *Usuarios*). |
| `SIGMA_DEBUG` | — | `1` activa la recarga automática (solo desarrollo). |

## Cuentas de usuario

- Roles: **administrador** (gestiona cuentas desde *Usuarios*) y **usuario**. Todo lo demás es **privado por cuenta**:
  un recurso ajeno responde siempre «no encontrado».
- Contraseñas: mínimo 10 caracteres, sin el nombre de usuario ni contraseñas triviales; se guardan con *scrypt*.
  Tras 5 intentos fallidos el acceso se bloquea unos minutos. Cambiar o restablecer la contraseña, o desactivar la
  cuenta, cierra sus demás sesiones. Siempre queda al menos un administrador activo.
- La **Consola Python** requiere un permiso explícito del administrador (desactivado por defecto): un kernel
  ejecuta código con los permisos del sistema operativo, así que **no hay aislamiento real entre usuarios con consola**.
  `sigma.load()` / `sigma.save()` solo alcanzan los datos de la cuenta que ejecuta el kernel.
- Sin navegador: `.\.venv\Scripts\python.exe scripts\create_user.py` crea cuentas o restablece contraseñas.
- Para usarlo en red: `SIGMA_HOST=0.0.0.0`, `SIGMA_ALLOWED_HOSTS` y **HTTPS mediante un proxy inverso** (por HTTP las
  contraseñas viajan en claro).

## Espacio de trabajo y varias ventanas

- Cada módulo **conserva lo que hizo** (formulario, dataset elegido y resultado con sus gráficos) al cambiar de módulo,
  recargar la página o volver otro día, hasta que pulse **Limpiar** (o **Limpiar todos los módulos**, en Inicio).
  Se guarda en el servidor (SQLite), por cuenta; las contraseñas, cabeceras y parámetros de las APIs nunca se guardan.
- Si abre Sigma en **varias ventanas o navegadores con la misma cuenta**, se mantienen en sintonía: un análisis, un
  dataset importado, un resultado guardado o un notebook modificado en una ventana aparece en las demás en unos
  segundos (sondeo de `/api/sync` mientras la pestaña está visible; también cuenta lo que guarde la consola con
  `sigma.save`). Si otra ventana cambia un módulo mientras usted tiene ediciones sin guardar, Sigma pregunta:
  *Cargar la versión nueva* o *Mantener la mía*. En la consola, guardar un notebook que otra ventana modificó
  responde con un conflicto (`409`): *Cargar la versión nueva* o *Sobrescribir con la mía*.

## Interpretación con IA (opcional)

En cada módulo, en la ficha de **Datos**, en **Resultados guardados** y en cada celda de código de la **Consola**, el
botón **Interpretar con IA** pide a Claude una explicación en español (*Resumen · Hallazgos · Supuestos y advertencias ·
Conclusión · Siguientes pasos*) con criterios propios de cada análisis; un campo opcional («¿Qué quieres saber?») orienta
el enfoque (p. ej. «explícalo para gerencia»). **Solo se ejecuta cuando usted lo pulsa.**

- **Qué se envía**: únicamente el *resumen del resultado* (parámetros, cifras, tablas recortadas y avisos; hasta ~24 000
  caracteres), dentro de etiquetas de datos y con instrucciones de ignorar cualquier orden que aparezca en él. Nunca el
  dataset (ni sus filas ni las categorías de columnas tipo identificador), ni los gráficos. La primera vez se pide
  consentimiento. La respuesta es una ayuda: lleva el aviso «Generado por IA: verifique las cifras».
- **Dónde queda**: la interpretación forma parte del resultado: se conserva en el espacio de trabajo, se guarda con
  *Guardar resultado* (y se puede añadir o quitar después en *Resultados guardados*), va en los metadatos de la celda
  del notebook y aparece en los PDF, Word y Excel exportados.
- **Configuración (administrador)**: la clave de API es única del servidor, en la variable de entorno
  `ANTHROPIC_API_KEY` o guardada desde **Usuarios → Interpretación con IA** (`instance/ai_key.txt`; nunca se vuelve a
  mostrar ni llega al navegador ni a los kernels). Allí también se elige el modelo (por defecto `claude-opus-5-5`, con
  esfuerzo medio y, si el modelo declina, reserva en `claude-opus-4-8`), se activa o desactiva la función, se prueba la
  conexión y se ve el consumo de los últimos 30 días por cuenta. El permiso **IA** se concede por usuario (activo para
  administradores, desactivado por defecto para el resto).
- **Límites y costo**: `SIGMA_AI_REQUESTS_PER_HOUR` (20) solicitudes por persona y hora; cada una consume tokens de su
  cuenta de Anthropic. Sin clave, sin conexión o con la IA desactivada el resto de Sigma funciona igual.

## Informes: exportar e imprimir

- **Exportar** (PDF, Word o Excel) desde todos los módulos con el botón del resultado: título, autor (se recuerda),
  plantilla, portada, índice, tamaño y orientación de página. Los gráficos se dibujan en el navegador sobre fondo blanco
  y viajan al servidor como PNG; el servidor compone el archivo (reportlab, python-docx, openpyxl). El PDF lleva
  marcadores, cabecera y pie y «Página n de N»; Word, un campo de índice (se actualiza al abrirlo); Excel, una hoja por
  tabla (números como números), «Informe» y «Gráficos».
- **Informe combinado**: en *Resultados guardados* marque varios resultados y pulse *Exportar informe*: un único
  documento con portada, índice y una sección por resultado (con sus notas, dataset y fecha).
- **Datos**: *Ficha* exporta el resumen del dataset (columnas, estadísticos, frecuencias, primeras filas); en Excel
  incluye además **todas las filas**. **Consola**: *Exportar informe* convierte el notebook (texto, código, salidas,
  tablas y gráficos) y permite elegir qué incluir.
- **Plantillas** (menú *Plantillas de informe*, privadas por cuenta): organización, logotipo PNG/JPG (≤ 2 MB, se
  reescala y se guarda como PNG; no se admite SVG), colores, texto de pie y de portada, tamaño y orientación, portada e
  índice. Una puede ser la predeterminada.
- **Imprimir**: cada resultado, la vista de datos y *Resultados guardados* tienen su botón *Imprimir* (gráficos en blanco
  y sin barra lateral ni controles). El notebook tiene un diálogo propio: tamaño y orientación de página, márgenes, qué
  incluir, salto de página antes de los encabezados `#` / `##` o de las celdas marcadas con el botón ⤒, no partir las
  celdas cortas, cabecera con el título y pie con «Página X de Y» (los números de página requieren Chrome/Edge 131+; el
  PDF exportado los incluye en cualquier navegador).

## Importaciones pendientes

El archivo que sube en **Datos** (y, una vez generada, su vista previa) se **conserva en el servidor**: puede cambiar de
módulo, recargar, cerrar el navegador o reiniciar Sigma y la tarjeta del archivo y la vista previa siguen ahí; al
cambiar el delimitador, la hoja o la codificación se vuelve a leer el archivo guardado sin subirlo otra vez. Hay una
importación pendiente por tipo de fuente y cuenta; desaparece al guardarla como dataset, con **Quitar**, con
**Limpiar** o pasados `SIGMA_DRAFT_RETENTION_DAYS` días (14) sin tocarla. De SQL Server y de las APIs solo se conserva el
resultado de la consulta: las contraseñas, cabeceras, cuerpos y parámetros nunca se escriben en la base.

## Datos de ejemplo

`sample_data/` incluye `ventas.csv`, `marco_muestral.csv`, `encuesta.csv` y `analisis_demo.ipynb`
(`python scripts/make_sample_data.py` los regenera). En **Datos → Cargar datos de ejemplo** se cargan los CSV en su
cuenta (o `python scripts/load_sample_data.py --user <nombre>`).
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

## Base de datos: elegir dónde se guarda todo

Sigma guarda en **una sola base de datos** las cuentas, los datasets importados (con sus tipos), los resultados
guardados, el espacio de trabajo, las importaciones pendientes, las plantillas de informe, los ajustes de la IA y, con los
motores de servidor, también los notebooks. El administrador elige el motor:

| Motor | Controlador (`requirements.txt`) | Notas |
|---|---|---|
| **SQLite** (por defecto) | incluido en Python | Un archivo en `instance/sigma.db` (modo WAL). Los notebooks siguen siendo `.ipynb` en `instance/notebooks/<id>/`. |
| **Microsoft SQL Server** | `pyodbc` + «ODBC Driver 17/18 for SQL Server» | Autenticación de SQL Server o de Windows. Esquema propio opcional. Hasta 280 columnas por dataset. |
| **MySQL / MariaDB** | `pymysql` | Se crea con `utf8mb4`. Hasta 900 columnas por dataset. |
| **PostgreSQL** | `psycopg[binary]` | Esquema propio opcional y modo SSL configurable. |
| **Oracle Database** | `oracledb` (modo *thin*, sin cliente Oracle) | Servicio indicado como «nombre del servicio». Oracle trata `''` como `NULL`: una celda de texto vacía vuelve como valor ausente. |
| **MongoDB** | `pymongo` | Colecciones equivalentes a las tablas; los datasets se guardan en documentos de unos miles de filas. |

**Cómo se elige**

1. **Primera vez:** el asistente de `/setup` tiene un paso «Base de datos» (SQLite ya está seleccionada; «Usar otra base
   de datos…» abre el formulario, con *Probar conexión*).
2. **Después:** *Base de datos* en el menú del administrador. *Probar conexión* comprueba el controlador, el acceso,
   el contenido (vacía / datos de Sigma / tablas ajenas) y que se puedan crear tablas, con mensajes claros (usuario o
   contraseña, base inexistente, servidor inalcanzable, certificado TLS…). Si la base elegida está **vacía** y la actual
   tiene datos, la casilla **«Copiar los datos actuales»** los lleva a la nueva base (ids incluidos: **las sesiones abiertas
   siguen válidas**); si ya contiene datos de Sigma se usan tal cual. Mientras dura la copia Sigma queda en
   mantenimiento; la base de origen no se modifica y sirve de copia de seguridad.
3. **Despliegues:** `SIGMA_DATABASE_URL` (tiene prioridad; entonces la interfaz solo muestra la configuración):

```
sqlite:///instance/sigma.db
mssql+pyodbc://usuario:clave@servidor/sigma?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes
mssql+pyodbc://@servidor\SQLEXPRESS/sigma?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes
mysql+pymysql://usuario:clave@servidor/sigma
postgresql+psycopg://usuario:clave@servidor/sigma
oracle+oracledb://usuario:clave@servidor:1521/?service_name=FREEPDB1
mongodb://usuario:clave@servidor:27017/sigma
```

La elección hecha desde la interfaz se guarda en `instance/database.json` (incluye la contraseña de la base: trátelo como
un secreto, igual que `secret.key`). La interfaz nunca la devuelve al navegador. Sigma crea sus propias tablas; la base
(o el esquema) debe existir y estar vacía, y el usuario necesita permiso para crear tablas.

**Migración por línea de comandos** (con Sigma detenido):

```powershell
.\.venv\Scripts\python.exe scripts\migrate_database.py --to "postgresql+psycopg://usuario:clave@servidor/sigma" --use
```

**Qué se ha probado.** La batería de conformidad (cuentas, datasets, resultados, espacio de trabajo con escrituras concurrentes,
borradores con archivos de varios MB, plantillas, notebooks, esquemas, prueba de conexión y migración en ambos sentidos) y toda la
API de Sigma se ejecutaron contra servidores reales: SQLite 3.50, SQL Server 2025 Express (ODBC Driver 18), MySQL 8, MariaDB 11.8,
PostgreSQL 16, Oracle AI Database 26ai Free (23.26) y MongoDB 7. Otras versiones deberían funcionar (se usan dialectos estándar de SQLAlchemy 2),
pero no se han probado.

**Actualización de una instalación anterior:** el archivo `instance/sigma.db` de una versión previa se actualiza solo al
arrancar; los datasets y los notebooks existentes siguen funcionando sin copiarlos.

**Respaldo:** con SQLite, copie la carpeta `instance/` (con la aplicación detenida, o `sigma.db`, `sigma.db-wal` y
`sigma.db-shm` juntos); con un motor de servidor, use las herramientas de respaldo de ese motor (más la carpeta
`instance/` para `secret.key`, `ai_key.txt` y `database.json`).

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

La consola ejecuta código arbitrario por diseño (solo para cuentas autorizadas), así que:

- Escucha solo en `127.0.0.1` por defecto. Para exponerlo en red use HTTPS (proxy inverso) y revise quién tiene
  permiso de consola.
- Toda ruta (salvo `/login`, `/setup`, `/health` y los estáticos) exige sesión; el inicio de sesión renueva la sesión y
  el token CSRF, y las respuestas no se guardan en caché.
- Rechaza peticiones con cabecera `Host` u `Origin` ajenas (DNS-rebinding / otras páginas web) y exige un token CSRF
  en toda petición que modifica datos.
- Política de seguridad de contenido (CSP) restrictiva; no se carga ningún recurso externo.
- Los resultados importados, el HTML de los notebooks y los datos se muestran saneados (DOMPurify / `textContent`).
- Las sentencias SQL usan parámetros y los nombres de tabla se generan desde identificadores internos.
- **Base de datos:** la página *Base de datos* es solo para administradores y el paso del asistente `/setup` solo existe
  mientras no haya ninguna cuenta (complete el asistente nada más instalar). Lo que se escribe en el formulario se valida
  de forma estricta (host, base, opciones permitidas), SQLite siempre es el archivo del servidor y la contraseña guardada no
  se envía a un servidor distinto del que se configuró. El kernel de la consola recibe la URL de la base: quien tiene
  permiso de consola puede leer su contraseña (como ya podía leer `instance/` y `.env`).

## Identidad visual

El logotipo (anillo Ω = espacio muestral, disco A con una Σ = el suceso y la σ-álgebra, y la estrella Pollux como un
resultado ω del complemento) se genera con `python scripts/make_logo.py` a partir de una única definición geométrica:
SVG (marca completa, variante compacta para favicons y marca con texto), PNG (512, 256, *apple-touch* 180) y `favicon.ico`
(16/32/48). Encabeza la barra lateral y las pantallas de acceso, y es el aspecto por defecto de los informes sin plantilla.

## Estructura

Arquitectura limpia: `presentation → application → domain` e `infrastructure → application/domain`. Detalle y
decisiones en [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
sigma/
  domain/           estadística pura (numpy/scipy/statsmodels), entidades y errores; sin Flask ni E/S
  application/      casos de uso y puertos (interfaces)
  infrastructure/   persistencia (SQL: SQLite/SQL Server/MySQL/PostgreSQL/Oracle · MongoDB), importadores, kernel Jupyter
  presentation/web/ blueprints Flask, esquemas pydantic, reportes y gráficos, plantillas Jinja, JS y Tailwind
tests/              unit/ (dominio, infraestructura), conformance/ (la misma batería en cada motor de base de datos)
                    e integration/ (API, páginas, consola con kernel real)
```

## Desarrollo

```powershell
.\.venv\Scripts\python.exe -m pytest              # 660 pruebas con SQLite y MongoDB en memoria (la consola arranca kernels reales: ~4 min)
.\.venv\Scripts\python.exe -m ruff check .        # lint
.\.venv\Scripts\python.exe -m ruff format .       # formato
npm run watch:css                                 # recompila Tailwind al editar plantillas/JS
$env:SIGMA_DEBUG = "1"; .\.venv\Scripts\python.exe run.py   # recarga automática
```

**Probar otros motores de base de datos.** La batería `tests/conformance/` corre siempre en SQLite y en MongoDB en memoria;
para ejecutarla contra servidores reales levante los contenedores de prueba y defina las URL (ver
[docker-compose.test.yml](docker-compose.test.yml)):

```powershell
docker compose -f docker-compose.test.yml up -d
$env:SIGMA_TEST_POSTGRES_URL = "postgresql+psycopg://postgres:sigmatest@127.0.0.1:55432/sigma_test"
$env:SIGMA_TEST_MYSQL_URL    = "mysql+pymysql://root:sigmatest@127.0.0.1:53306/sigma_test"
$env:SIGMA_TEST_ORACLE_URL   = "oracle+oracledb://sigma:sigmatest@127.0.0.1:51521/?service_name=FREEPDB1"
$env:SIGMA_TEST_MONGO_URL    = "mongodb://127.0.0.1:28017/sigma_test"
.\.venv\Scripts\python.exe -m pytest tests\conformance
$env:SIGMA_TEST_APP_ENGINE = "postgresql"; .\.venv\Scripts\python.exe -m pytest tests\integration   # toda la API sobre ese motor
```

Las bases de esas URL deben estar vacías y ser desechables: cada prueba empieza borrando sus tablas.

## Limitaciones conocidas

- Sin recuperación de contraseña por correo (la restablece el administrador), sin 2FA/SSO y sin compartir recursos
  entre cuentas. Sin ANOVA de medidas repetidas ni post-estratificación (raking).
- La salida de la consola se entrega al terminar la celda (sin transmisión en vivo).
- En Windows, una llamada bloqueante como `time.sleep(120)` no se puede interrumpir: tras el tiempo máximo el kernel se
  reinicia (se pierden las variables en memoria).
- La importación desde SQL Server no puede probarse automáticamente sin un servidor: las pruebas cubren la construcción
  de la conexión, la protección de solo lectura y toda la ruta de lectura contra SQLite.
- Base de datos: Oracle no distingue `''` de `NULL` (una celda de texto vacía vuelve como ausente); cada motor limita las
  columnas de un dataset (SQL Server 280, MySQL/Oracle/MariaDB 900, PostgreSQL 1000, SQLite 1900) y lo dice con un error claro; MongoDB
  no tiene transacciones entre documentos, así que borrar una cuenta o un dataset es una secuencia de operaciones (con limpieza si falla).
  Cambiar de base desde la página afecta a este proceso: si ejecuta varios procesos de Sigma, reinicie los demás. La migración copia
  las importaciones pendientes sin su vista previa (se vuelve a generar desde el archivo).
