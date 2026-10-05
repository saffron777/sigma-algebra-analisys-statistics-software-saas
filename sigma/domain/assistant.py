"""Asistente guiado: qué análisis usar según la pregunta y los tipos de variables."""

from __future__ import annotations

from dataclasses import dataclass

GOALS = (
    ("comparar", "Comparar grupos o condiciones"),
    ("relacionar", "Relacionar dos o más variables"),
    ("predecir", "Predecir una variable"),
    ("describir", "Describir una variable"),
    ("serie", "Analizar una serie temporal"),
    ("reducir", "Reducir dimensiones o agrupar casos"),
)

KINDS = (
    ("numerica", "Numérica"),
    ("categorica", "Categórica (grupos)"),
    ("tiempo", "Fecha / tiempo"),
)

PAGE_PATHS = {
    "descriptiva": "/descriptiva",
    "hipotesis": "/hipotesis",
    "regresion": "/regresion",
    "anova": "/anova",
    "series": "/series",
    "multivariado": "/multivariado",
    "modelos": "/modelos",
    "pruebas": "/pruebas",
    "mercados": "/mercados",
    "aprendizaje": "/aprendizaje",
}


@dataclass(frozen=True)
class Recommendation:
    goal: str
    module: str
    label: str
    href: str
    reason: str


def recommend(goal: str, primary: str, secondary: str | None = None) -> Recommendation:
    """The analysis that fits a question and the kinds of the variables involved."""
    if goal == "comparar":
        if primary == "categorica":
            return Recommendation(
                goal, "hipotesis", "χ² de independencia", PAGE_PATHS["hipotesis"],
                "Compara proporciones entre grupos con una tabla de contingencia.",
            )
        return Recommendation(
            goal, "anova", "ANOVA de un factor", PAGE_PATHS["anova"],
            "Compara la media de una variable numérica entre dos o más grupos.",
        )
    if goal == "relacionar":
        if primary == "numerica" and secondary == "numerica":
            return Recommendation(
                goal, "regresion", "Regresión lineal", PAGE_PATHS["regresion"],
                "Mide la relación entre dos variables numéricas.",
            )
        if primary == "categorica" and secondary == "categorica":
            return Recommendation(
                goal, "hipotesis", "χ² de independencia", PAGE_PATHS["hipotesis"],
                "Examina la asociación entre dos variables categóricas.",
            )
        return Recommendation(
            goal, "anova", "ANOVA", PAGE_PATHS["anova"],
            "Compara una variable numérica entre categorías (o viceversa).",
        )
    if goal == "predecir":
        if primary == "categorica":
            return Recommendation(
                goal, "modelos", "Regresión logística", PAGE_PATHS["modelos"],
                "Predice una variable categórica (0/1 o de dos niveles).",
            )
        return Recommendation(
            goal, "regresion", "Regresión lineal", PAGE_PATHS["regresion"],
            "Predice una variable numérica a partir de otras.",
        )
    if goal == "describir":
        if primary == "categorica":
            return Recommendation(
                goal, "descriptiva", "Estadística descriptiva", PAGE_PATHS["descriptiva"],
                "Resume frecuencias y proporciones.",
            )
        return Recommendation(
            goal, "descriptiva", "Estadística descriptiva", PAGE_PATHS["descriptiva"],
            "Resume centro, dispersión y forma de la distribución.",
        )
    if goal == "serie":
        return Recommendation(
            goal, "series", "Series de tiempo", PAGE_PATHS["series"],
            "Analiza tendencia, estacionalidad y autocorrelación de una serie.",
        )
    if goal == "reducir":
        return Recommendation(
            goal, "multivariado", "Componentes principales / clúster", PAGE_PATHS["multivariado"],
            "Reduce variables o agrupa observaciones parecidas.",
        )
    return Recommendation(goal, "descriptiva", "Estadística descriptiva", PAGE_PATHS["descriptiva"], "Un buen punto de partida.")
