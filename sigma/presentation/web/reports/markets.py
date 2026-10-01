"""Reports for the survey / market-research module."""

from __future__ import annotations

from sigma.domain.distributions import DistributionRef
from sigma.domain.results import (
    CrosstabResult,
    LikertResult,
    SampleDrawResult,
    SampleSizeResult,
    SurveyEstimateResult,
)
from sigma.presentation.web.charts import survey_plots as plots
from sigma.presentation.web.charts.distribution_plot import distribution_figure
from sigma.presentation.web.reports import blocks as b
from sigma.presentation.web.reports.fmt import integer, num, pct, pval, ratio_pct

SAMPLE_PREVIEW_ROWS = 20


def sample_size_report(result: SampleSizeResult) -> b.ReportBuilder:
    proportion = result.kind == "proportion"
    what = "una proporción" if proportion else "una media"
    margin = ratio_pct(result.margin_of_error) if proportion else num(result.margin_of_error)
    builder = b.ReportBuilder(
        f"Tamaño de muestra para estimar {what}",
        f"Confianza {num(100 * result.confidence, 3)} % · margen de error ±{margin}",
    )
    tiles = [
        ("Tamaño de muestra requerido", integer(result.n_final), "unidades a encuestar"),
        ("n sin corrección", num(result.n_infinite, 5), "población infinita"),
        ("Valor crítico z", num(result.z, 5)),
    ]
    if result.population_size:
        tiles.insert(2, ("n con población finita", num(result.n_adjusted, 5), f"N = {integer(result.population_size)}"))
    if result.design_effect != 1:
        tiles.append(("Efecto de diseño", num(result.design_effect, 3)))
    builder.add(b.tiles(tiles))
    inputs = [["Nivel de confianza", ratio_pct(result.confidence)], ["Margen de error", margin]]
    if proportion:
        inputs.append(["Proporción esperada (p)", num(result.inputs.get("proportion"))])
    else:
        inputs.append(["Desviación estándar esperada (σ)", num(result.inputs.get("std_dev"))])
    inputs += [
        ["Población (N)", integer(result.population_size) if result.population_size else "infinita"],
        ["Efecto de diseño", num(result.design_effect, 3)],
    ]
    builder.add(b.table("Supuestos del cálculo", ["Parámetro", "Valor"], inputs))
    if proportion and (result.inputs.get("proportion") or 0.5) != 0.5:
        builder.add(b.alert("info", "Con p = 0.5 se obtiene el tamaño máximo (caso más conservador)."))
    if result.allocation:
        builder.add(b.heading("Asignación por estratos"))
        builder.add(
            b.table(
                None,
                ["Estrato", "Población (N_h)", "Peso", "Muestra (n_h)"],
                [
                    [a.stratum, integer(a.population), pct(100 * a.share), integer(a.sample_size)]
                    for a in result.allocation
                ],
            )
        )
        builder.add_figure(plots.allocation_figure(result.allocation, result.allocation_method))
    builder.add_figure(
        plots.sample_size_curve_figure(result), "Cuanto menor el margen de error deseado, mayor la muestra necesaria."
    )
    return builder


def draw_report(result: SampleDrawResult, saved_name: str | None) -> b.ReportBuilder:
    builder = b.ReportBuilder(
        f"Muestra aleatoria — {_DESIGN[result.design]}",
        f"N = {integer(result.population_size)} · n = {integer(result.sample_size)} · semilla {result.seed}",
    )
    builder.add(
        b.tiles(
            [
                ("Población (N)", integer(result.population_size)),
                ("Muestra (n)", integer(result.sample_size)),
                ("Semilla", str(result.seed), "reproduce la misma muestra"),
            ]
        )
    )
    if saved_name:
        builder.add(
            b.alert(
                "success",
                f"La muestra se guardó como dataset «{saved_name}» con las columnas «prob_inclusion» y «peso_diseno».",
            )
        )
    for note in result.notes:
        builder.add(b.alert("info", note))
    for table in result.summary:
        builder.add(
            b.table(
                table.title,
                table.columns,
                [[num(c, 5) if isinstance(c, float) else c for c in row] for row in table.rows],
            )
        )
    sample = result.sample
    if sample is not None and len(sample):
        head = sample.head(SAMPLE_PREVIEW_ROWS)
        builder.add(
            b.table(
                f"Primeras {len(head)} unidades seleccionadas",
                [str(c) for c in head.columns],
                [[_plain(v) for v in row] for row in head.itertuples(index=False)],
            )
        )
    return builder


_DESIGN = {
    "srs": "aleatorio simple",
    "systematic": "sistemático",
    "stratified": "estratificado",
    "cluster": "por conglomerados",
}


def _plain(value) -> str:
    if isinstance(value, float):
        return num(value, 5)
    return "—" if value is None or value != value else str(value)  # noqa: PLR0124 - NaN check


def estimate_report(result: SurveyEstimateResult) -> b.ReportBuilder:
    proportion = result.kind == "proportion"
    builder = b.ReportBuilder(
        f"Estimación de {'proporciones' if proportion else 'la media'} — {result.variable}",
        f"{result.design} · confianza {num(100 * result.confidence, 3)} %",
    )
    tiles = [("Observaciones válidas", integer(result.n))]
    if result.population_size:
        tiles.append(("Población (N)", integer(result.population_size)))
    if result.design_effect is not None:
        tiles += [
            ("Efecto de diseño", num(result.design_effect, 3)),
            ("Tamaño efectivo", num(result.effective_n, 4), "n / efecto de diseño"),
        ]
    builder.add(b.tiles(tiles))
    for note in result.notes:
        builder.add(b.alert("info", note))
    if proportion:
        builder.add(
            b.table(
                None,
                [
                    "Categoría",
                    "Casos",
                    "Estimación",
                    "Error est.",
                    "Margen de error",
                    "IC inferior",
                    "IC superior",
                    "Método",
                ],
                [
                    [
                        r.label,
                        integer(r.count),
                        ratio_pct(r.estimate, 2),
                        ratio_pct(r.std_error, 2),
                        "±" + ratio_pct(r.margin_of_error, 2),
                        ratio_pct(r.ci_low, 2),
                        ratio_pct(r.ci_high, 2),
                        r.method,
                    ]
                    for r in result.rows
                ],
            )
        )
    else:
        builder.add(
            b.table(
                None,
                ["Variable", "Media", "Error est.", "Margen de error", "IC inferior", "IC superior", "Método"],
                [
                    [
                        r.label,
                        num(r.estimate, 5),
                        num(r.std_error, 4),
                        "±" + num(r.margin_of_error, 4),
                        num(r.ci_low, 5),
                        num(r.ci_high, 5),
                        r.method,
                    ]
                    for r in result.rows
                ],
            )
        )
    builder.add_figure(plots.estimates_figure(result))
    lead = result.rows[0]
    scale = 100 if proportion else 1
    builder.add_figure(
        distribution_figure(
            DistributionRef("normal", {"mu": lead.estimate * scale, "sigma": max(lead.std_error * scale, 1e-12)}),
            title=f"Distribución del estimador de «{lead.label}»",
            x_title="Porcentaje (%)" if proportion else result.variable,
            interval=(lead.ci_low * scale, lead.ci_high * scale),
        ),
        "Distribución normal del estimador; la zona verde es el intervalo de confianza.",
    )
    return builder


def crosstab_report(result: CrosstabResult) -> b.ReportBuilder:
    builder = b.ReportBuilder(
        f"Tabulación cruzada — {result.row_variable} × {result.col_variable}", f"n = {integer(result.n)}"
    )
    significant = result.p_value < 0.05
    builder.add(
        b.alert(
            "success" if significant else "info",
            f"χ²({result.df}) = {num(result.chi2, 4)}, p = {pval(result.p_value)} → "
            + (
                "existe asociación estadísticamente significativa entre las variables."
                if significant
                else "no hay evidencia de asociación entre las variables."
            ),
        ),
        b.tiles(
            [
                ("χ²", num(result.chi2, 4)),
                ("gl", integer(result.df)),
                ("Valor p", pval(result.p_value)),
                ("V de Cramér", num(result.cramers_v, 3), "intensidad de la asociación (0–1)"),
            ]
        ),
    )
    for note in result.notes:
        builder.add(b.alert("warning", note))
    columns = [result.row_variable + " \\ " + result.col_variable, *result.col_labels]

    def matrix(title: str, data, fmt) -> dict:
        return b.table(title, columns, [[result.row_labels[i], *[fmt(v) for v in row]] for i, row in enumerate(data)])

    builder.add(matrix("Frecuencias observadas", result.observed, integer))
    builder.add(matrix("Porcentaje por fila", result.row_percent, pct))
    builder.add(matrix("Porcentaje por columna", result.col_percent, pct))
    builder.add(matrix("Frecuencias esperadas bajo independencia", result.expected, lambda v: num(v, 4)))
    builder.add(
        b.table(
            "Residuos estandarizados",
            columns,
            [
                [result.row_labels[i], *[b.cell(num(v, 3), "bad" if abs(v) > 1.96 else None) for v in row]]
                for i, row in enumerate(result.standardized_residuals)
            ],
            "|residuo| > 1.96 señala celdas que contribuyen significativamente a la asociación.",
        )
    )
    builder.add_figure(plots.crosstab_figure(result))
    if result.distribution:
        builder.add_figure(
            distribution_figure(
                result.distribution,
                title=f"Distribución χ²({result.df}) bajo independencia — p = {pval(result.p_value)}",
                x_title="χ²",
                statistic=result.chi2,
                statistic_label="χ²",
                tail="right",
                alpha=0.05,
                show_p_value=True,
            ),
            "Zona roja: región de rechazo al 5 %. Zona azul: valor p.",
        )
    return builder


def likert_report(result: LikertResult) -> b.ReportBuilder:
    builder = b.ReportBuilder("Escala Likert", f"{result.n_items} ítem(s) · niveles: {', '.join(result.levels)}")
    tiles = []
    if result.cronbach_alpha is not None:
        quality = (
            "excelente"
            if result.cronbach_alpha >= 0.9
            else "buena"
            if result.cronbach_alpha >= 0.8
            else "aceptable"
            if result.cronbach_alpha >= 0.7
            else "cuestionable"
            if result.cronbach_alpha >= 0.6
            else "pobre"
        )
        tiles.append(
            (
                "Alfa de Cronbach",
                num(result.cronbach_alpha, 3),
                f"consistencia interna {quality} (n = {result.n_complete})",
            )
        )
    if tiles:
        builder.add(b.tiles(tiles))
    rows = []
    for i, item in enumerate(result.items):
        rows.append([item, *[pct(p) for p in result.percents[i]], num(result.means[i], 3), pct(result.top2box[i])])
    builder.add(
        b.table(
            "Distribución de respuestas",
            ["Ítem", *result.levels, "Media", "Top-2-box"],
            rows,
            "Top-2-box: porcentaje que eligió las dos opciones más altas de la escala.",
        )
    )
    builder.add_figure(plots.likert_figure(result))
    return builder
