"""Report for hypothesis tests."""

from __future__ import annotations

from sigma.domain.results import HypothesisTestResult
from sigma.presentation.web.charts.distribution_plot import distribution_figure
from sigma.presentation.web.reports import blocks as b
from sigma.presentation.web.reports.distributions import TAIL_LABELS
from sigma.presentation.web.reports.fmt import interval, num, pval

DISCRETE_REFERENCES = {"binomial", "poisson", "geometric", "hypergeometric", "negative_binomial"}


def _df_text(df) -> str:
    if df is None:
        return "—"
    if isinstance(df, list):
        return ", ".join(num(d) for d in df)
    return num(df)


def build_report(result: HypothesisTestResult) -> b.ReportBuilder:
    builder = b.ReportBuilder(
        result.test_name,
        f"Nivel de significancia α = {num(result.alpha)} · {TAIL_LABELS.get(result.tail, result.tail)}",
    )
    builder.add(
        b.alert("success" if result.reject_null else "info", result.conclusion),
        b.table("Hipótesis", ["Nula", "Alternativa"], [[result.null_hypothesis, result.alternative_hypothesis]]),
        b.tiles(
            [
                (f"Estadístico ({result.statistic_name})", num(result.statistic, 5)),
                ("Grados de libertad", _df_text(result.df)),
                (
                    "Valor p",
                    pval(result.p_value),
                    "p < α → rechazar H₀" if result.reject_null else "p ≥ α → no rechazar H₀",
                ),
                ("Decisión", "Rechazar H₀" if result.reject_null else "No rechazar H₀"),
            ]
        ),
    )

    rows: list[list] = []
    if result.estimate is not None:
        rows.append([result.estimate_label or "Estimación", num(result.estimate, 5)])
    if result.confidence_interval:
        low, high = result.confidence_interval
        rows.append(
            [f"Intervalo de confianza al {num(100 * (result.confidence or 0.95), 3)} %", interval(low, high, 5)]
        )
    if result.effect_size is not None:
        rows.append([result.effect_size_label or "Tamaño del efecto", num(result.effect_size, 4)])
    if result.critical_values:
        rows.append(["Valor(es) crítico(s)", ", ".join(num(c, 5) for c in result.critical_values)])
    if rows:
        builder.add(b.table("Resultados", ["Medida", "Valor"], rows))
    if result.sample_info:
        builder.add(
            b.table(
                "Datos de la muestra",
                ["Concepto", "Valor"],
                [[str(k), num(v, 5) if isinstance(v, (int, float)) else str(v)] for k, v in result.sample_info.items()],
            )
        )
    for table in result.tables:
        builder.add(
            b.table(
                table.title,
                table.columns,
                [[num(c, 5) if isinstance(c, float) else c for c in row] for row in table.rows],
                table.note or None,
            )
        )
    for note in result.notes:
        builder.add(b.alert("warning" if note.startswith("Aviso") else "info", note))

    if result.distribution is not None:
        discrete = result.distribution.key in DISCRETE_REFERENCES
        marker = result.plot_data.get("t_statistic", result.statistic)
        label = "t" if "t_statistic" in result.plot_data else result.statistic_name
        builder.add_figure(
            distribution_figure(
                result.distribution,
                title=f"Distribución de referencia bajo H₀ — p = {pval(result.p_value)}",
                x_title=label,
                statistic=marker,
                statistic_label=label,
                tail=result.tail,
                alpha=None if discrete else result.alpha,
                show_p_value=True,
            ),
            "Zona roja: región de rechazo. Zona azul: valor p. Línea punteada: estadístico observado.",
        )
    return builder
