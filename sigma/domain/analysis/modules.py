"""The pages served by the generic catalogue, and the one place that names them.

A module's ``key`` is at once its page key, its workspace key, the key of its saved results and of its AI
interpretation, so the registries that used to be kept in step by hand are derived from this listing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleDef:
    key: str
    label: str
    path: str
    description: str
    criteria: str  # what an AI interpretation of a result of this module should look at


CATALOG_MODULES: tuple[ModuleDef, ...] = (
    ModuleDef(
        "transformar",
        "Transformar datos",
        "/transformar",
        "Calcule, recodifique, filtre, una y limpie columnas; trate valores faltantes y atípicos.",
        "Explica qué se hizo a los datos (filas y columnas antes y después), si la transformación es razonable para "
        "el análisis posterior y qué precauciones tomar (valores imputados, filas eliminadas, pérdida de información).",
    ),
    ModuleDef(
        "graficos",
        "Gráficos",
        "/graficos",
        "Histogramas, cajas, dispersión, matrices, mapas de calor, gráficos de probabilidad y más, por grupos.",
        "Describe lo que muestra el gráfico: forma de la distribución, relación entre variables, diferencias entre "
        "grupos, valores atípicos y patrones; no inventes valores que no se vean en el resumen.",
    ),
    ModuleDef(
        "pruebas",
        "Pruebas adicionales",
        "/pruebas",
        "Normalidad, igualdad de varianzas, pruebas no paramétricas, tasas de Poisson, equivalencia, acuerdo entre "
        "evaluadores, odds ratio, Mantel-Haenszel, T² de Hotelling y correlación parcial.",
        "Explica la hipótesis nula de la prueba, el estadístico y el valor p, la decisión al nivel de significancia "
        "elegido, el tamaño del efecto o el intervalo de confianza cuando exista, y los supuestos que conviene "
        "verificar (independencia, normalidad, tamaño de muestra, datos pareados).",
    ),
    ModuleDef(
        "potencia",
        "Potencia y tamaño de muestra",
        "/potencia",
        "Calcule el tamaño de muestra, la potencia o el efecto detectable de pruebas t, proporciones, ANOVA y más.",
        "Explica qué tamaño de muestra o potencia resultó, con qué supuestos (efecto, α, varianza), cuánto cambia al "
        "variar el efecto y qué riesgo se corre de no detectar un efecto real.",
    ),
    ModuleDef(
        "remuestreo",
        "Remuestreo",
        "/remuestreo",
        "Bootstrap, pruebas de permutación y simulación Monte Carlo sin supuestos de distribución.",
        "Interpreta el intervalo o el valor p obtenido por remuestreo, la estabilidad de la distribución simulada, el "
        "número de réplicas y la semilla, y compáralo con el método paramétrico cuando se muestre.",
    ),
    ModuleDef(
        "modelos",
        "Modelos de regresión",
        "/modelos",
        "Logística, Poisson, GLM, robusta, cuantílica, paso a paso, regularizada, no lineal, instrumentales y mixtos.",
        "Evalúa el ajuste y la significancia de cada término, el sentido y la magnitud de los efectos (odds ratio, "
        "razón de tasas, coeficientes), la capacidad predictiva (AUC, clasificación, error), los supuestos del modelo "
        "y sus problemas (separación, multicolinealidad, sobredispersión, heterocedasticidad).",
    ),
    ModuleDef(
        "anova_avanzado",
        "ANOVA avanzado",
        "/anova-avanzado",
        "ANCOVA, medidas repetidas, MANOVA, varios factores, efectos mixtos y comparaciones múltiples.",
        "Indica qué efectos son significativos y su tamaño, si se cumplen los supuestos (esfericidad, normalidad, "
        "homogeneidad), qué pares de grupos difieren según la corrección de comparaciones múltiples usada y la "
        "interpretación de las interacciones.",
    ),
    ModuleDef(
        "series",
        "Series de tiempo",
        "/series",
        "Autocorrelación, raíces unitarias, descomposición, suavizamiento, ARIMA y pronósticos con intervalos.",
        "Comenta la tendencia y la estacionalidad, la estacionariedad (pruebas de raíz unitaria), la autocorrelación "
        "de los residuos, la calidad del ajuste y de los pronósticos (MAPE, RMSE) y cuánta confianza tienen los "
        "intervalos de pronóstico.",
    ),
    ModuleDef(
        "econometria",
        "Econometría",
        "/econometria",
        "VAR, cointegración y VECM, GARCH, ARDL, datos de panel, variables instrumentales y quiebres estructurales.",
        "Interpreta las relaciones dinámicas o de panel (causalidad de Granger, impulso-respuesta, cointegración, "
        "volatilidad, efectos fijos o aleatorios), las pruebas de especificación y diagnóstico y las limitaciones "
        "del modelo (estacionariedad, endogeneidad, poca muestra).",
    ),
    ModuleDef(
        "multivariado",
        "Multivariado",
        "/multivariado",
        "Componentes principales, análisis factorial, clúster, discriminante, correspondencias y fiabilidad.",
        "Explica cuántos componentes o grupos se justifican (varianza explicada, silueta), qué variables los "
        "definen (cargas), la calidad de la clasificación y de la fiabilidad y las precauciones (escala, outliers, "
        "tamaño de muestra).",
    ),
    ModuleDef(
        "supervivencia",
        "Supervivencia y confiabilidad",
        "/supervivencia",
        "Kaplan-Meier con comparación de grupos, regresión de Cox y modelos paramétricos con censura.",
        "Interpreta la curva de supervivencia y su mediana, las diferencias entre grupos (log-rank), las razones de "
        "riesgo del modelo de Cox y los parámetros del modelo paramétrico; señala la censura y los supuestos "
        "(riesgos proporcionales).",
    ),
    ModuleDef(
        "calidad",
        "Control de calidad (SPC)",
        "/calidad",
        "Cartas de control de Shewhart (I-MR, X̄-R, X̄-S), CUSUM/EWMA y capacidad del proceso (Cp, Cpk, Pp, Ppk).",
        "Interpreta si el proceso está bajo control estadístico (puntos fuera de los límites o patrones), la "
        "capacidad frente a las especificaciones (Cp, Cpk) y la fracción esperada fuera de especificación.",
    ),
    ModuleDef(
        "encuestas",
        "Regresión de encuestas",
        "/encuestas",
        "Regresión lineal y logística con pesos muestrales (SURVEYREG y SURVEYLOGISTIC).",
        "Interpreta los coeficientes (y odds ratios) del modelo ponderado, su significancia y su ajuste, señalando "
        "el papel de los pesos muestrales y las limitaciones del diseño.",
    ),
    ModuleDef(
        "doe",
        "Diseño de experimentos",
        "/doe",
        "Diseños factoriales 2^k y superficie de respuesta (compuesto central), con análisis de la respuesta.",
        "Interpreta qué factores e interacciones son significativos, la magnitud de sus efectos y la calidad del "
        "ajuste del modelo (R²), señalando la estructura del diseño (codificación ±1, corridas centrales y axiales).",
    ),
    ModuleDef(
        "optimizacion",
        "Optimización (SAS/OR)",
        "/optimizacion",
        "Programación lineal y entera: asignación óptima de recursos con una restricción de presupuesto.",
        "Interpreta el valor objetivo alcanzado, qué elementos se seleccionan y en qué cantidad, el recurso "
        "consumido y la holgura, y contrasta la solución entera con la continua cuando aplica.",
    ),
    ModuleDef(
        "sem",
        "Ecuaciones estructurales",
        "/sem",
        "Análisis de rutas (path analysis) sobre variables observadas con coeficientes de ruta estandarizados.",
        "Interpreta las relaciones directas entre variables (coeficientes de ruta β), su significancia y el R² de "
        "cada ecuación, señalando que es un modelo recursivo sobre variables observadas (sin variables latentes).",
    ),
    ModuleDef(
        "aprendizaje",
        "Aprendizaje automático",
        "/aprendizaje",
        "Árboles, bosques, boosting, SVM y redes con validación cruzada, ROC, importancia de variables y comparación.",
        "Evalúa el desempeño con validación cruzada (media y dispersión de las métricas), el sobreajuste, la "
        "importancia de las variables, el desbalance de clases y si el modelo es mejor que una línea base.",
    ),
)

CATALOG_KEYS = tuple(module.key for module in CATALOG_MODULES)
BY_KEY = {module.key: module for module in CATALOG_MODULES}
