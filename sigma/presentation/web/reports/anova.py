"""Report for ANOVA."""

from __future__ import annotations

from sigma.domain.distributions import DistributionRef
from sigma.domain.results import AnovaResult
from sigma.presentation.web.charts import anova_plots as plots
from sigma.presentation.web.charts.distribution_plot import distribution_figure
from sigma.presentation.web.reports import blocks as b
from sigma.presentation.web.reports.fmt import integer, interval, num, pval

_ROMAN = {1: "I", 2: "II", 3: "III"}


def _significant(p: float | None, alpha: float) -> bool:
    return p is not None and p < alpha


def build_report(result: AnovaResult, dataset_name: str) -> b.ReportBuilder:
    one_way = result.design == "one_way"
    design = "un factor" if one_way else "dos factores"
    subtitle = f"{dataset_name} · α = {num(result.alpha)}"
    if result.ss_type:
        subtitle += f" · suma de cuadrados tipo {_ROMAN[result.ss_type]}"
    builder = b.ReportBuilder(f"ANOVA de {design}: {result.response} ~ {' × '.join(result.factors)}", subtitle)
    for test in result.f_tests:
        sig = _significant(test.p_value, result.alpha)
        if one_way:
            verdict = (
                "hay diferencias significativas entre las medias."
                if sig
                else "no hay evidencia de diferencias entre las medias."
            )
        else:
            verdict = "efecto significativo." if sig else "efecto no significativo."
        builder.add(
            b.alert(
                "success" if sig else "info",
                f"{test.label}: F({num(test.df_num)}, {num(test.df_den)}) = {num(test.statistic, 4)}, "
                f"p = {pval(test.p_value)} → {verdict}",
            )
        )
    if result.n_dropped:
        builder.add(b.alert("info", f"Se excluyeron {result.n_dropped} fila(s) con valores faltantes."))

    builder.add(b.heading("Tabla ANOVA"))
    builder.add(
        b.table(
            None,
            ["Fuente", "Suma de cuadrados", "gl", "Cuadrado medio", "F", "Valor p", "η²", "η² parcial", "ω²"],
            [
                [
                    r.source,
                    num(r.sum_sq, 5),
                    num(r.df),
                    num(r.mean_sq, 5),
                    num(r.f_value, 4),
                    b.cell(pval(r.p_value), "strong" if _significant(r.p_value, result.alpha) else None)
                    if r.p_value is not None
                    else "—",
                    num(r.eta_sq, 3),
                    num(r.partial_eta_sq, 3),
                    num(r.omega_sq, 3),
                ]
                for r in result.table
            ],
            "η² y ω² miden el tamaño del efecto (≈0.01 pequeño, 0.06 mediano, 0.14 grande).",
        )
    )
    if result.welch:
        w = result.welch
        builder.add(
            b.table(
                "ANOVA de Welch (varianzas desiguales)",
                ["F", "gl numerador", "gl denominador", "Valor p"],
                [
                    [
                        num(w["statistic"], 4),
                        num(w["df_num"]),
                        num(w["df_den"], 4),
                        b.cell(pval(w["p_value"]), "strong" if w["p_value"] < result.alpha else None),
                    ]
                ],
            )
        )
    if result.kruskal:
        k = result.kruskal
        builder.add(
            b.table(
                "Kruskal-Wallis (alternativa no paramétrica)",
                ["H", "gl", "Valor p"],
                [
                    [
                        num(k["statistic"], 4),
                        num(k["df"]),
                        b.cell(pval(k["p_value"]), "strong" if k["p_value"] < result.alpha else None),
                    ]
                ],
            )
        )

    builder.add(b.heading("Estadísticos por grupo"))
    builder.add(
        b.table(
            None,
            ["Grupo", "n", "Media", "Desv. est.", "Error est.", f"IC {num(100 * (1 - result.alpha), 3)} %"],
            [
                [g.group, integer(g.n), num(g.mean, 5), num(g.std, 4), num(g.sem, 4), interval(g.ci_low, g.ci_high, 5)]
                for g in result.groups
            ],
        )
    )

    builder.add(b.heading("Verificación de supuestos"))
    builder.add(
        b.table(
            None,
            ["Supuesto", "Prueba", "Estadístico", "Valor p", "Conclusión"],
            [
                [a.name, a.test, num(a.statistic, 4), pval(a.p_value), b.cell(a.note, "good" if a.passed else "bad")]
                for a in result.assumptions
            ],
        )
    )

    if result.post_hoc:
        builder.add(b.heading(f"Comparaciones múltiples — {result.post_hoc_method}"))
        builder.add(
            b.table(
                None,
                [
                    "Grupo 1",
                    "Grupo 2",
                    "Diferencia (2 − 1)",
                    "Valor p ajustado",
                    f"IC {num(100 * (1 - result.alpha), 3)} %",
                    "¿Significativa?",
                ],
                [
                    [
                        p.group1,
                        p.group2,
                        num(p.mean_diff, 5),
                        b.cell(pval(p.p_adj), "strong" if p.reject else None),
                        interval(p.ci_low, p.ci_high, 5),
                        b.cell("Sí" if p.reject else "No", "good" if p.reject else "muted"),
                    ]
                    for p in result.post_hoc
                ],
            )
        )

    builder.add(b.heading("Distribución F"))
    for test in result.f_tests:
        builder.add_figure(
            distribution_figure(
                DistributionRef("f", {"dfn": test.df_num, "dfd": test.df_den}),
                title=f"{test.label}: F({num(test.df_num)}, {num(test.df_den)}) — p = {pval(test.p_value)}",
                x_title="F",
                statistic=test.statistic,
                statistic_label="F",
                tail="right",
                alpha=result.alpha,
                show_p_value=True,
            ),
            "Zona roja: región de rechazo. Zona azul: valor p.",
        )

    builder.add(b.heading("Gráficos"))
    builder.add_figure(plots.groups_figure(result))
    builder.add_figure(plots.means_figure(result))
    if result.design == "two_way":
        builder.add_figure(plots.interaction_figure(result))
    builder.add_figure(plots.residual_qq_figure(result))
    return builder
