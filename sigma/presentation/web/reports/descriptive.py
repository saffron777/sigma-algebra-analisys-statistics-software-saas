"""Report for descriptive statistics."""

from __future__ import annotations

from sigma.domain.distributions import DistributionRef
from sigma.domain.results import DescriptiveResult
from sigma.presentation.web.charts import descriptive_plots as plots
from sigma.presentation.web.charts.distribution_plot import distribution_figure
from sigma.presentation.web.reports import blocks as b
from sigma.presentation.web.reports.fmt import integer, num, pct, pval

PERCENTILES = ("1", "5", "10", "25", "50", "75", "90", "95", "99")


def build_report(result: DescriptiveResult, dataset_name: str) -> b.ReportBuilder:
    builder = b.ReportBuilder(
        f"Estadística descriptiva — {dataset_name}",
        f"Confianza del {num(100 * result.confidence, 3)} % · agrupado por «{result.group_by}»"
        if result.group_by
        else f"Confianza del {num(100 * result.confidence, 3)} %",
    )
    if result.numeric:
        _numeric_tables(builder, result)
    if result.normality:
        _normality(builder, result)
    if result.fits:
        _fits(builder, result)
    _numeric_charts(builder, result)
    for table in result.categorical:
        builder.add(b.heading(f"Variable categórica: {table.name}"))
        note = f"{integer(table.n)} observaciones válidas" + (
            f", {table.n_missing} faltantes" if table.n_missing else ""
        )
        builder.add(
            b.table(
                None,
                ["Categoría", "Frecuencia", "Porcentaje", "Frec. acumulada", "% acumulado"],
                [
                    [r.value, integer(r.count), pct(r.percent), integer(r.cumulative_count), pct(r.cumulative_percent)]
                    for r in table.rows
                ],
                note,
            )
        )
        builder.add_figure(plots.frequency_bar_figure(table))
    if result.correlation:
        _correlation(builder, result)
    return builder


def _label(summary) -> str:
    return summary.name if summary.group is None else f"{summary.name} [{summary.group}]"


def _numeric_tables(builder: b.ReportBuilder, result: DescriptiveResult) -> None:
    rows = result.numeric
    builder.add(b.heading("Variables numéricas"))
    builder.add(
        b.table(
            "Posición y dispersión",
            [
                "Variable",
                "n",
                "Faltantes",
                "Media",
                "Mediana",
                "Moda",
                "Desv. est.",
                "Varianza",
                "CV",
                "Error est.",
                "IC inferior",
                "IC superior",
            ],
            [
                [
                    _label(s),
                    integer(s.n),
                    integer(s.n_missing),
                    num(s.mean),
                    num(s.median),
                    num(s.mode),
                    num(s.std),
                    num(s.variance),
                    num(s.cv),
                    num(s.sem),
                    num(s.ci_low),
                    num(s.ci_high),
                ]
                for s in rows
            ],
            f"IC de la media al {num(100 * result.confidence, 3)} % con distribución t de Student.",
        )
    )
    builder.add(
        b.table(
            "Rango y forma",
            ["Variable", "Mínimo", "Q1", "Q3", "Máximo", "Rango", "RIC", "Asimetría", "Curtosis (exceso)"],
            [
                [
                    _label(s),
                    num(s.minimum),
                    num(s.q1),
                    num(s.q3),
                    num(s.maximum),
                    num(s.range),
                    num(s.iqr),
                    num(s.skewness),
                    num(s.kurtosis),
                ]
                for s in rows
            ],
        )
    )
    builder.add(
        b.table(
            "Percentiles",
            ["Variable", *[f"P{p}" for p in PERCENTILES]],
            [[_label(s), *[num(s.percentiles[p]) for p in PERCENTILES]] for s in rows],
        )
    )
    first = next((s for s in rows if s.group is None and s.n > 1), None)
    if first:
        df = first.n - 1
        builder.add_figure(
            distribution_figure(
                DistributionRef("t", {"df": float(df)}),
                title=f"Distribución t usada en el IC de «{first.name}» (gl = {df})",
                x_title="t",
                alpha=1 - result.confidence,
                tail="two-sided",
            ),
            f"Las zonas rojas dejan {num(100 * (1 - result.confidence), 3)} % de probabilidad en las colas; "
            "el IC de la media usa el valor crítico t de esos límites.",
        )


def _normality(builder: b.ReportBuilder, result: DescriptiveResult) -> None:
    rows = []
    for t in result.normality:
        label = t.name if t.group is None else f"{t.name} [{t.group}]"
        normal = t.p_value >= 0.05
        rows.append(
            [
                label,
                t.test,
                num(t.statistic),
                integer(t.n),
                b.cell(pval(t.p_value), None if normal else "strong"),
                b.cell("compatible con normalidad" if normal else "se rechaza normalidad", "good" if normal else "bad"),
            ]
        )
    builder.add(b.heading("Pruebas de normalidad"))
    builder.add(
        b.table(
            None,
            ["Variable", "Prueba", "Estadístico", "n", "Valor p", "Conclusión (α = 0.05)"],
            rows,
            "H₀: los datos provienen de una distribución normal.",
        )
    )
    k2 = next((t for t in result.normality if t.distribution is not None and t.group is None), None)
    if k2:
        builder.add_figure(
            distribution_figure(
                k2.distribution,
                title=f"Distribución χ²(2) del estadístico K² de «{k2.name}»",
                x_title="K²",
                statistic=k2.statistic,
                statistic_label="K²",
                tail="right",
                alpha=0.05,
                show_p_value=True,
            ),
            "El estadístico K² de D'Agostino-Pearson sigue aproximadamente una χ² con 2 grados de libertad bajo H₀.",
        )


def _fits(builder: b.ReportBuilder, result: DescriptiveResult) -> None:
    builder.add(b.heading("Ajuste de distribuciones"))
    builder.add(
        b.table(
            None,
            ["Variable", "Distribución", "Parámetros", "D de KS", "Valor p (KS)", "AIC"],
            [
                [
                    f.column,
                    f.label,
                    ", ".join(f"{k} = {num(v)}" for k, v in f.params.items()),
                    num(f.ks_statistic),
                    b.cell(pval(f.ks_p_value), "good" if f.ks_p_value >= 0.05 else None),
                    num(f.aic),
                ]
                for f in result.fits
            ],
            "Ordenadas por AIC (menor = mejor). El valor p de KS es aproximado porque los parámetros "
            "se estiman de los mismos datos.",
        )
    )


def _numeric_charts(builder: b.ReportBuilder, result: DescriptiveResult) -> None:
    for name, groups in result.plot_data.get("numeric", {}).items():
        if not groups:
            continue
        fits = [f for f in result.fits if f.column == name]
        builder.add(b.heading(f"Gráficos de {name}"))
        builder.add_figure(plots.histogram_figure(name, groups, fits))
        if fits and len(groups) > 1:  # the fitted histogram covers all data; also show the groups side by side
            builder.add_figure(plots.histogram_figure(name, groups, []))
        builder.add_figure(plots.box_figure(name, groups))
        if "Todos" in groups and len(groups["Todos"]) >= 5:
            builder.add_figure(plots.qq_figure(name, groups["Todos"]))


def _correlation(builder: b.ReportBuilder, result: DescriptiveResult) -> None:
    corr = result.correlation
    names = corr.columns
    builder.add(b.heading(f"Correlaciones ({corr.method})"))
    builder.add(
        b.table(
            "Coeficientes", ["", *names], [[names[i], *[num(v, 3) for v in row]] for i, row in enumerate(corr.matrix)]
        )
    )
    builder.add(
        b.table(
            "Valores p",
            ["", *names],
            [
                [
                    names[i],
                    *[
                        ("" if i == j else b.cell(pval(p), "strong" if p is not None and p < 0.05 else None))
                        for j, p in enumerate(row)
                    ],
                ]
                for i, row in enumerate(corr.p_values)
            ],
            "Resaltado: p < 0.05. H₀: la correlación poblacional es cero.",
        )
    )
    builder.add_figure(plots.correlation_figure(corr))
