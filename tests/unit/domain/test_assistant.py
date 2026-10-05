"""Asistente: recomienda el análisis correcto según la pregunta y los tipos de variable."""

from sigma.domain.assistant import recommend


def test_comparar_grupos_recomienda_anova_o_chi_cuadrado():
    assert recommend("comparar", "numerica").module == "anova"
    assert recommend("comparar", "categorica").module == "hipotesis"


def test_predecir_recomienda_regresion_o_logistica():
    assert recommend("predecir", "numerica").module == "regresion"
    assert recommend("predecir", "categorica").module == "modelos"


def test_relacionar_distingue_por_los_tipos():
    assert recommend("relacionar", "numerica", "numerica").module == "regresion"
    assert recommend("relacionar", "categorica", "categorica").module == "hipotesis"


def test_serie_y_reducir_recomiendan_sus_modulos():
    assert recommend("serie", "numerica").module == "series"
    assert recommend("reducir", "numerica").module == "multivariado"


def test_toda_recomendacion_tiene_enlace_y_razon():
    for goal in ("comparar", "relacionar", "predecir", "describir", "serie", "reducir"):
        for primary in ("numerica", "categorica", "tiempo"):
            rec = recommend(goal, primary)
            assert rec.href.startswith("/") and rec.reason and rec.label
