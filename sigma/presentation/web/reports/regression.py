"""Report for linear regression."""

from __future__ import annotations

from sigma.domain.results import RegressionResult
from sigma.presentation.web.charts import regression_plots as plots
from sigma.presentation.web.charts.distribution_plot import distribution_figure
from sigma.presentation.web.reports import blocks as b
from sigma.presentation.web.reports.fmt import integer, interval, num, pval

MAX_T_MARKERS = 6


def build_report(result: RegressionResult, dataset_name: str) -> b.ReportBuilder:
    kind = "simple" if result.is_simple else "múltiple"
    builder = b.ReportBuilder(
        f"Regresión lineal {kind}: {result.response} ~ {' + '.join(result.predictors)}",
        f"{dataset_name} · confianza {num(100 * result.confidence, 3)} %",
    )
    significant = result.f_p_value is not None and result.f_p_value < 0.05
    builder.add(
        b.tiles(
            [
                ("R²", num(result.r_squared, 4), f"{num(100 * result.r_squared, 3)} % de la variabilidad explicada"),
                ("R² ajustado", num(result.adj_r_squared, 4)),
                ("Error estándar residual", num(result.residual_std_error, 4), f"{num(result.df_resid)} gl"),
                ("Estadístico F", num(result.f_statistic, 4), f"gl = {num(result.df_model)}, {num(result.df_resid)}"),
                (
                    "Valor p (F)",
                    pval(result.f_p_value),
                    "modelo significativo" if significant else "modelo no significativo (α = 0.05)",
                ),
                ("Observaciones", integer(result.n_obs)),
            ]
        ),
        b.text(f"Ecuación estimada: {result.equation}"),
    )
    for note in result.notes:
        builder.add(b.alert("info", note))

    rows = []
    for c in result.coefficients:
        sig = c.p_value < 0.05
        rows.append(
            [
                c.term,
                num(c.estimate, 5),
                num(c.std_error, 5),
                num(c.t_value, 4),
                b.cell(pval(c.p_value), "strong" if sig else None),
                interval(c.ci_low, c.ci_high, 5),
                num(c.vif, 3) if c.vif is not None else "—",
            ]
        )
    builder.add(b.heading("Coeficientes"))
    builder.add(
        b.table(
            None,
            ["Término", "Coeficiente", "Error est.", "t", "Valor p", f"IC {num(100 * result.confidence, 3)} %", "VIF"],
            rows,
            "H₀ de cada coeficiente: β = 0. VIF > 5–10 indica multicolinealidad.",
        )
    )
    builder.add(b.heading("ANOVA del modelo"))
    builder.add(
        b.table(
            None,
            ["Fuente", "Suma de cuadrados", "gl", "Cuadrado medio", "F", "Valor p"],
            [
                [
                    r.source,
                    num(r.sum_sq, 5),
                    num(r.df),
                    num(r.mean_sq, 5),
                    num(r.f_value, 4),
                    b.cell(pval(r.p_value), "strong" if r.p_value is not None and r.p_value < 0.05 else None)
                    if r.p_value is not None
                    else "—",
                ]
                for r in result.anova
            ],
        )
    )
    builder.add(
        b.table(
            "Criterios de ajuste",
            ["AIC", "BIC", "Log-verosimilitud"],
            [[num(result.aic, 6), num(result.bic, 6), num(result.log_likelihood, 6)]],
        )
    )

    builder.add(b.heading("Diagnóstico de supuestos"))
    builder.add(
        b.table(
            None,
            ["Prueba", "Estadístico", "Valor p", "Interpretación"],
            [
                [
                    d.name,
                    num(d.statistic, 4),
                    pval(d.p_value) if d.p_value is not None else "—",
                    b.cell(d.interpretation, "good" if d.passed else "bad"),
                ]
                for d in result.diagnostics
            ],
        )
    )
    builder.add(b.heading("Distribuciones de referencia"))
    if result.f_distribution and result.f_statistic is not None:
        builder.add_figure(
            distribution_figure(
                result.f_distribution,
                title=f"Prueba F global — p = {pval(result.f_p_value)}",
                x_title="F",
                statistic=result.f_statistic,
                statistic_label="F",
                tail="right",
                alpha=0.05,
                show_p_value=True,
            ),
            "H₀: todos los coeficientes (excepto el intercepto) son cero.",
        )
    if result.t_distribution:
        shown = [c for c in result.coefficients if c.term != "Intercepto"][:MAX_T_MARKERS] or result.coefficients[
            :MAX_T_MARKERS
        ]
        builder.add_figure(
            distribution_figure(
                result.t_distribution,
                title=f"Pruebas t de los coeficientes (gl = {num(result.df_resid)})",
                x_title="t",
                alpha=0.05,
                tail="two-sided",
                extra_markers=[(c.t_value, c.term) for c in shown],
            ),
            "Cada línea es el estadístico t de un coeficiente; las zonas rojas son la región de rechazo al 5 %.",
        )

    builder.add(b.heading("Gráficos"))
    if result.is_simple:
        builder.add_figure(plots.fit_figure(result))
    else:
        builder.add_figure(plots.observed_vs_fitted_figure(result))
    builder.add_figure(plots.residuals_vs_fitted_figure(result))
    builder.add_figure(plots.residual_qq_figure(result))
    builder.add_figure(plots.scale_location_figure(result))
    builder.add_figure(plots.cooks_figure(result))

    if result.influential:
        builder.add(
            b.table(
                "Observaciones más influyentes",
                ["Fila", "Observado", "Ajustado", "Residuo", "Leverage", "Cook"],
                [
                    [
                        integer(p.row),
                        num(p.observed),
                        num(p.fitted),
                        num(p.residual),
                        num(p.leverage, 3),
                        b.cell(num(p.cooks_distance, 3), "bad" if p.cooks_distance > 4 / result.n_obs else None),
                    ]
                    for p in result.influential[:5]
                ],
                "Cook > 4/n sugiere una observación influyente.",
            )
        )
    if result.predictions:
        names = result.predictors
        builder.add(b.heading("Predicciones"))
        builder.add(
            b.table(
                None,
                [*names, "Predicción", "IC de la media", "IC de predicción"],
                [
                    [
                        *[str(p.inputs.get(n, "")) for n in names],
                        num(p.prediction, 5),
                        interval(p.mean_ci_low, p.mean_ci_high, 5),
                        interval(p.pred_ci_low, p.pred_ci_high, 5),
                    ]
                    for p in result.predictions
                ],
            )
        )
    return builder
