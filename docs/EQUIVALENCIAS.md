# Equivalencias: de Minitab, SPSS y EViews a Sigma

Esta guía responde «¿dónde está en Sigma lo que yo hacía en…?». Cada fila nombra la ruta del programa de origen y la página y el análisis de
Sigma. Los análisis de las páginas de catálogo (`/transformar`, `/graficos`, `/pruebas`, `/potencia`, `/remuestreo`, `/modelos`,
`/anova-avanzado`, `/series`, `/econometria`, `/multivariado`, `/aprendizaje`) se eligen en la lista de la izquierda de cada página; todos
tienen el botón **Ver código** (Python equivalente, el mismo `sigma.load(...)` de la consola) y pueden guardar columnas nuevas como dataset.

Convenciones: **M** = Minitab (menú *Stat*, *Calc*, *Graph*, *Data*), **S** = SPSS (menú *Analizar*, *Datos*, *Transformar*, *Gráficos*),
**E** = EViews. «Página › Análisis» es la ruta dentro de Sigma. Lo que aún no existe está al final, en [Lo que falta](#lo-que-falta).

## Datos: preparar y transformar (`/transformar`)

| Tarea | Minitab | SPSS | EViews | Sigma |
|---|---|---|---|---|
| Calcular una variable con una fórmula | Calc › Calculator | Transformar › Calcular variable | `genr` | Transformar › Calcular variable (fórmula) — evaluador propio y seguro, con `where`, `lag`, `rolling_*`, `zscore`… |
| Recodificar | Data › Recode / Code | Transformar › Recodificar | `@recode` | Transformar › Recodificar valores · Agrupar en intervalos |
| Variables ficticias | Calc › Make Indicator Variables | Transformar › Crear variables ficticias | `@expand` | Transformar › Variables ficticias (dummies) |
| Tipificar / normalizar | Calc › Standardize | Descriptivos › Guardar valores tipificados | — | Transformar › Tipificar y normalizar |
| Log, raíz, Box-Cox | Stat › Control Charts › Box-Cox | Transformar › Calcular | `genr` | Transformar › Transformaciones matemáticas (log, √, Box-Cox, Yeo-Johnson) |
| Rezagos, diferencias, tasas | Stat › Time Series › Lag / Differences | Transformar › Crear serie temporal | `d()`, `dlog()`, `x(-1)` | Transformar › Rezagos, diferencias, tasas y acumulados |
| Fechas y texto | Data › Extract from Date/Time | Asistente de fecha y hora | `@year`, `@month` | Transformar › Extraer partes de una fecha · Limpiar texto |
| Filtrar / ordenar | Data › Subset Worksheet · Sort | Datos › Seleccionar casos · Ordenar casos | `smpl` | Transformar › Filtrar filas · Ordenar filas |
| Duplicados | Data › Unique | Datos › Identificar casos duplicados | — | Transformar › Detectar y quitar duplicados |
| Muestra aleatoria y partición | Calc › Random Data › Sample | Datos › Seleccionar casos (aleatoria) · Partición | — | Transformar › Muestra aleatoria · Partición entrenamiento / prueba |
| Agregar por grupos | Stat › Tables · Data › Merge | Datos › Agregar | `@sum` por grupo | Transformar › Agregar por grupos (resumen) |
| Apilar / desapilar | Data › Stack · Unstack | Datos › Reestructurar | Reshape | Transformar › Apilar columnas · Pivotar |
| Unir datasets | Data › Merge Worksheets | Datos › Fusionar archivos | `copy` | Transformar › Unir con otro dataset (merge) · Apilar otro dataset |
| Faltantes | Data › Replace Missing | Analizar › Reemplazar valores perdidos | Interpolar | Transformar › Valores faltantes · Imputación multivariante (KNN / MICE) |
| Atípicos | Stat › Basic Statistics › Outlier Test (Grubbs) | Explorar · Identificar casos inusuales | — | Transformar › Valores atípicos (IQR, z, MAD, Grubbs) · Aprendizaje › Detección de anomalías |

## Gráficos (`/graficos`)

| Gráfico | Minitab | SPSS | EViews | Sigma |
|---|---|---|---|---|
| Histograma (con curva) | Graph › Histogram | Gráficos › Histograma | `hist` | Gráficos › Histograma |
| Caja, violín, valores individuales | Graph › Boxplot · Individual Value Plot | Gráficos › Diagrama de caja | `boxplot` | Gráficos › Diagrama de caja, violín y valores individuales |
| Probabilidad | Graph › Probability Plot | Analizar › Gráficos Q-Q | `qqplot` | Gráficos › Gráfico de probabilidad (Q-Q) |
| Dispersión y burbujas | Graph › Scatterplot · Bubble Plot | Gráficos › Dispersión | `scat` | Gráficos › Dispersión y burbujas (con ajuste) |
| Matriz de dispersión | Graph › Matrix Plot | Dispersión matricial | `scat(m)` | Gráficos › Matriz de dispersión |
| Mapa de calor / contornos / 3D | Graph › Contour · 3D Surface | Gráficos › 3-D | — | Gráficos › Mapa de calor · Contornos y superficie · Dispersión 3D |
| Líneas / series | Graph › Time Series Plot | Gráficos › Secuencia | `line` | Gráficos › Gráfico de líneas · Series › Gráfico de la serie |
| Barras, Pareto, circular | Graph › Bar Chart · Stat › Quality Tools › Pareto · Pie Chart | Gráficos › Barras · Circular | `bar` | Gráficos › Gráfico de barras · Diagrama de Pareto · Gráfico circular |
| Intervalos de confianza | Graph › Interval Plot | Gráficos › Barras de error | — | Gráficos › Gráfico de intervalos de confianza |

## Estadística básica

| Análisis | Minitab | SPSS | EViews | Sigma |
|---|---|---|---|---|
| Estadísticos descriptivos | Stat › Basic Statistics › Display Descriptive Statistics | Analizar › Estadísticos descriptivos › Frecuencias / Descriptivos / Explorar | View › Descriptive Statistics | Estadística descriptiva |
| Distribuciones de probabilidad | Calc › Probability Distributions | Transformar › Calcular (funciones) | `@dnorm`… | Distribuciones |
| Normalidad | Stat › Basic Statistics › Normality Test | Explorar › Gráficos con pruebas de normalidad | Histogram and Stats (Jarque-Bera) | Pruebas de hipótesis › Shapiro-Wilk · Pruebas adicionales › Anderson-Darling · Lilliefors · Kolmogorov-Smirnov · Jarque-Bera |
| t de 1 muestra, 2 muestras, pareada; z | Stat › Basic Statistics › 1-Sample t · 2-Sample t · Paired t · 1-Sample Z | Comparar medias › Prueba T | Test of Hypothesis | Pruebas de hipótesis › t de una media · t de dos medias · t pareada · z de una media |
| Proporciones | 1 Proportion · 2 Proportions | Pruebas no paramétricas › Binomial | — | Pruebas de hipótesis › z de una y dos proporciones · Binomial exacta |
| Varianzas | 1 Variance · 2 Variances · Test for Equal Variances | Levene (en ANOVA y t) | Test Equality › Variance | Pruebas de hipótesis › χ² de una varianza · F de dos varianzas · Pruebas adicionales › Igualdad de varianzas (Levene, Brown-Forsythe, Bartlett, Fligner) |
| Tasas de Poisson | Stat › Basic Statistics › 1-Sample Poisson Rate · 2-Sample | — | — | Pruebas adicionales › Tasas de Poisson |
| Equivalencia | Stat › Equivalence Tests | — | — | Pruebas adicionales › Equivalencia de medias (TOST) · Potencia › Equivalencia (TOST) |
| Correlación | Stat › Basic Statistics › Correlation | Analizar › Correlaciones › Bivariadas | Cross-Correlogram | Pruebas de hipótesis › Prueba de correlación · Gráficos › Mapa de calor |
| Correlación parcial | Partial Correlation | Correlaciones › Parciales | — | Pruebas adicionales › Correlación parcial |
| Tablas: χ², Fisher, McNemar, CMH | Stat › Tables › Chi-Square · Cross Tabulation · Cochran-Mantel-Haenszel | Estadísticos descriptivos › Tablas de contingencia | — | Pruebas de hipótesis › χ² de independencia · χ² de bondad · Fisher · Pruebas adicionales › McNemar · Q de Cochran · Cochran-Mantel-Haenszel · Odds ratio y riesgo relativo |
| Acuerdo entre evaluadores | Stat › Quality Tools › Attribute Agreement Analysis (kappa) | Tablas de contingencia › Kappa · Escala › Fiabilidad (CCI) | — | Pruebas adicionales › Kappa de Cohen y de Fleiss · Correlación intraclase (CCI) |
| No paramétricas | Stat › Nonparametrics › Sign · Wilcoxon · Mann-Whitney · Kruskal-Wallis · Mood's Median · Friedman · Runs | Pruebas no paramétricas | — | Pruebas de hipótesis › Mann-Whitney · Wilcoxon · ANOVA (Kruskal-Wallis) · Pruebas adicionales › Signos · Mediana de Mood · Friedman · Rachas |
| Intervalo de confianza de una variable | Display Descriptive Statistics (IC) | Explorar | — | Pruebas adicionales › Intervalos de confianza de una variable |
| T² de Hotelling | Stat › Multivariate › (Hotelling) | GLM multivariante | — | Pruebas adicionales › T² de Hotelling |
| Potencia y tamaño de muestra | Stat › Power and Sample Size | Analizar › Análisis de potencia | — | Potencia y tamaño de muestra (t, proporciones, ANOVA, correlación, TOST) |
| Bootstrap y permutación | Stat › Resampling | Bootstrap · Pruebas exactas | Bootstrap | Remuestreo › Bootstrap · Prueba de permutación |
| Simulación | Calc › Random Data | Transformar › Generadores de números aleatorios | `rnd` | Remuestreo › Simulación Monte Carlo · Generar datos aleatorios |
| Encuestas: tamaño de muestra, selección y estimación | Stat › Power and Sample Size (proporciones) · Calc › Random Data › Sample From Columns | Datos › Seleccionar casos (aleatoria) · Analizar › Muestras complejas (parcial) | — | Análisis de mercados (muestreo simple, sistemático, estratificado y por conglomerados; estimación con márgenes de error y efecto de diseño; tabulación cruzada; escalas Likert) |

## Modelos lineales, ANOVA y regresión

| Análisis | Minitab | SPSS | EViews | Sigma |
|---|---|---|---|---|
| Regresión lineal múltiple | Stat › Regression › Regression › Fit Regression Model | Analizar › Regresión › Lineal | Estimate Equation (LS) | Regresión lineal (con diagnósticos) |
| Polinómica e interacciones | Fit Regression Model (términos) · Fitted Line Plot | Estimación curvilínea | Estimate (polinomios) | Modelos › Regresión polinómica e interacciones |
| Paso a paso / mejores subconjuntos | Stepwise · Best Subsets | Lineal › Método: pasos sucesivos | Stepwise LS | Modelos › Regresión paso a paso · Mejores subconjuntos |
| Ridge, lasso, red elástica | Regression › Elastic Net (Predictive Analytics) | Regresión › Lazo/Red elástica | Estimate › ENET | Modelos › Ridge, lasso y red elástica · Aprendizaje › Modelo lineal regularizado |
| Logística binaria / probit | Regression › Binary Logistic | Regresión › Logística binaria · Probit | Estimate › BINARY | Modelos › Regresión logística / probit |
| Logística ordinal / nominal | Ordinal Logistic · Nominal Logistic | Ordinal · Logística multinomial | Estimate › ORDERED | Modelos › Regresión ordinal · Regresión logística multinomial |
| Conteos | Poisson Regression | Modelos lineales generalizados | Estimate › COUNT | Modelos › Regresión de conteos · GLM |
| No lineal | Nonlinear Regression | Regresión › No lineal | Estimate (ecuación libre) | Modelos › Regresión no lineal |
| Robusta y errores robustos | Robust Regression (Huber) | Regresión › Estimación robusta | Heteroskedasticity-consistent · Newey-West | Modelos › Regresión robusta y errores robustos |
| Mínimos cuadrados ponderados | Regression › Weighted | Regresión › Estimación de ponderaciones | Estimate › Weighted LS | Modelos › Mínimos cuadrados ponderados |
| Cuantílica | Quantile Regression | Regresión cuantílica | Estimate › QREG | Modelos › Regresión cuantílica |
| Variables instrumentales | — | Regresión › Mínimos cuadrados en dos fases | Estimate › TSLS | Modelos › Variables instrumentales (2MCO) |
| Efectos mixtos | Stat › ANOVA › Mixed Effects Model | Analizar › Modelos mixtos | Estimate › Mixed | Modelos › Modelo lineal mixto |
| Diagnósticos: White, RESET, Breusch-Godfrey, Chow | Residual plots | Lineal › Estadísticos / Gráficos | View › Residual Diagnostics · Stability Diagnostics | Modelos › Pruebas de especificación de un modelo lineal |
| ANOVA de 1 y 2 factores, Tukey | Stat › ANOVA › One-Way · Two-Way | Analizar › Comparar medias › ANOVA de un factor | — | ANOVA |
| Varios factores, ANCOVA | General Linear Model · Balanced ANOVA | MLG › Univariante | — | ANOVA avanzado › ANOVA de varios factores y ANCOVA |
| Medidas repetidas | General Linear Model (random) | MLG › Medidas repetidas | — | ANOVA avanzado › ANOVA de medidas repetidas (Mauchly, Greenhouse-Geisser) |
| MANOVA | General MANOVA | MLG › Multivariante | — | ANOVA avanzado › MANOVA |
| Anidado | Fully Nested ANOVA | Componentes de varianza | — | ANOVA avanzado › ANOVA anidado |
| Comparaciones múltiples | Tukey · Dunnett · Games-Howell · Hsu MCB | Post hoc | — | ANOVA avanzado › Comparaciones múltiples (post hoc) |

## Series de tiempo y econometría (`/series`, `/econometria`)

| Análisis | Minitab | SPSS | EViews | Sigma |
|---|---|---|---|---|
| Gráfico y estadísticos móviles | Time Series › Time Series Plot | Secuencia | Graph | Series › Gráfico de la serie y estadísticos móviles |
| ACF, PACF, Ljung-Box | Autocorrelation · Partial Autocorrelation | Autocorrelaciones | Correlogram | Series › Autocorrelación (ACF, PACF y Ljung-Box) |
| Correlación cruzada | Cross Correlation | Correlaciones cruzadas | Cross-Correlogram | Series › Correlación cruzada |
| Periodograma | — | Análisis espectral | Spectrum | Series › Periodograma y periodicidades |
| Descomposición | Decomposition | Descomponer estacionalmente | Seasonal Adjustment (STL) | Series › Descomposición (clásica y STL) |
| Tendencia | Trend Analysis | Estimación curvilínea | Trend regressions | Series › Análisis de tendencia y pronóstico · Tendencia de Mann-Kendall |
| Filtros | Moving Average | — | HP · Band-Pass (BK, CF) | Series › Filtros: Hodrick-Prescott, Baxter-King, Christiano-Fitzgerald |
| Suavizamiento exponencial | Single/Double Exponential Smoothing · Winters | Modelador de series (suavizamiento) | Exponential Smoothing | Series › Suavizamiento exponencial y ETS |
| ARIMA / SARIMA / SARIMAX | ARIMA | Modelador experto · ARIMA | Estimate › ARMA / ARIMAX | Series › ARIMA / SARIMA / SARIMAX (selección automática, diagnóstico, pronóstico) |
| Evaluación de pronósticos | — | — | Forecast Evaluation | Series › Comparar métodos de pronóstico (origen móvil) |
| Raíces unitarias | — | — | Unit Root Test (ADF, PP, DF-GLS, KPSS, Zivot-Andrews) | Series › Raíces unitarias y estacionariedad |
| VAR, impulso-respuesta | — | — | VAR | Econometría › VAR (rezagos óptimos, estabilidad, impulso-respuesta, descomposición de la varianza, pronóstico) |
| Causalidad de Granger | — | — | Granger Causality | Econometría › Causalidad de Granger entre dos series · VAR |
| Cointegración, VEC | — | — | Johansen · Engle-Granger · VEC | Econometría › Cointegración · VECM |
| GARCH y variantes | — | — | ARCH/GARCH (E, T, GJR, APARCH) | Econometría › GARCH, EGARCH, GJR y volatilidad (valor en riesgo, Kupiec) |
| ARDL y prueba de límites | — | — | ARDL (bounds test) | Econometría › ARDL y prueba de límites |
| Panel | — | — | Panel (fixed/random, Hausman) | Econometría › Datos de panel: efectos fijos y aleatorios |
| Cambio estructural | — | — | Chow · Quandt-Andrews · Bai-Perron | Modelos › Pruebas de especificación (Chow) · Econometría › Quandt-Andrews · Bai-Perron |
| Cambio de régimen | — | — | Markov switching | Econometría › Cambio de régimen de Markov |
| Espacio de estados | — | — | Sspace · Kalman | Econometría › Componentes no observados (Kalman) |

## Multivariado y ciencia de datos (`/multivariado`, `/aprendizaje`)

| Análisis | Minitab | SPSS | EViews | Sigma |
|---|---|---|---|---|
| Componentes principales | Stat › Multivariate › Principal Components | Reducción de dimensiones › Factor (componentes) | Group › Principal Components | Multivariado › Componentes principales (PCA) |
| Análisis factorial | Factor Analysis | Reducción de dimensiones › Factor | Group › Factor | Multivariado › Análisis factorial |
| Fiabilidad (alfa de Cronbach) | Item Analysis | Escala › Análisis de fiabilidad | — | Multivariado › Fiabilidad de una escala |
| Correlación canónica | — | MANOVA (sintaxis) | — | Multivariado › Correlación canónica |
| Correspondencias | Simple Correspondence Analysis | Reducción de dimensiones › Análisis de correspondencias | — | Multivariado › Análisis de correspondencias |
| Escalamiento multidimensional | — | Escala › Escalamiento multidimensional | — | Multivariado › Escalamiento multidimensional |
| Mahalanobis, atípicos multivariados | Distance | Regresión › Guardar distancia de Mahalanobis | — | Multivariado › Distancia de Mahalanobis |
| Clúster jerárquico | Cluster Observations · Cluster Variables | Clasificar › Conglomerado jerárquico | — | Multivariado › Clúster jerárquico y dendrograma |
| K-medias | Cluster K-Means | Clasificar › Conglomerado de K-medias | — | Multivariado › K-medias |
| Mezclas / TwoStep | — | Clasificar › TwoStep | — | Multivariado › Mezcla de gaussianas |
| Discriminante | Discriminant Analysis | Clasificar › Discriminante | — | Multivariado › Análisis discriminante |
| Árboles CART | Predictive Analytics › CART® | Clasificar › Árbol | — | Aprendizaje › Árbol de decisión (CART) |
| Bosques aleatorios | Random Forests® | Bosque aleatorio (Modeler) | — | Aprendizaje › Bosque aleatorio |
| TreeNet / boosting | TreeNet® | Boosting (Modeler) | — | Aprendizaje › Gradient boosting |
| Vecinos, SVM, Naive Bayes, red neuronal | — | Vecino más próximo · Redes neuronales · SVM/Bayes (Modeler) | — | Aprendizaje › k vecinos · SVM · Naive Bayes · Red neuronal |
| Comparar modelos y elegir | Discover Best Model | Auto Classifier (Modeler) | — | Aprendizaje › Comparar modelos · Ajuste de hiperparámetros · Curva de aprendizaje |
| Importancia de variables | Discover Key Predictors | Importancia de predictores | — | Aprendizaje › importancia por permutación (en cada modelo) · Selección de variables |
| Curva ROC | Stat › ROC Curve | Analizar › Curva COR | — | Modelos › Regresión logística · Aprendizaje (cada clasificador) · Evaluar predicciones existentes (con IC de DeLong) |
| Dependencia parcial | Partial Dependence Plot | — | — | Aprendizaje › Dependencia parcial y curvas ICE |

## Lo que falta

Sigma no cubre (todavía) estas áreas de los programas de origen; están en la hoja de ruta:

- **Calidad industrial (Minitab):** cartas de control (I-MR, X̄-R/S, P, NP, C, U, CUSUM, EWMA), capacidad del proceso, Gage R&R y acuerdo de atributos, muestreo de aceptación.
- **Diseño de experimentos (Minitab):** factoriales completos y fraccionados, Plackett-Burman, superficie de respuesta, mezclas, Taguchi.
- **Supervivencia y confiabilidad (Minitab, SPSS):** Kaplan-Meier, log-rank, Cox, Weibull y distribuciones con censura, tablas de vida.
- **Muestras complejas y análisis de valores perdidos (SPSS):** diseños de encuesta complejos, prueba de Little, EM.
- **Otros modelos:** mínimos cuadrados parciales (PLS), MARS, regresión ortogonal, GAM, modelos loglineales, conjoint.
- **Econometría avanzada (EViews):** GMM, VAR bayesiano, MIDAS, proyecciones locales, DSGE, ajuste estacional X-13.
- **Asistente guiado** «qué prueba debo usar».
