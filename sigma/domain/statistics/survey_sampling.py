"""Probability sampling: sample-size calculation and selection of random samples."""

from __future__ import annotations

import math
import secrets

import numpy as np
import pandas as pd
from scipy import stats

from sigma.domain.errors import InvalidInputError
from sigma.domain.results import SampleDrawResult, SampleSizeResult, StratumAllocation, TableData
from sigma.domain.statistics.common import check_confidence, label_of, numeric_series, require_columns

DESIGNS = ("srs", "systematic", "stratified", "cluster")
ALLOCATIONS = ("proportional", "equal", "neyman")
WEIGHT_COLUMN = "peso_diseno"
INCLUSION_COLUMN = "prob_inclusion"
MAX_STRATA = 200
CURVE_POINTS = 25


def z_value(confidence: float) -> float:
    return float(stats.norm.isf((1 - check_confidence(confidence)) / 2))


def _check_population(population: int | None) -> int | None:
    if population is None:
        return None
    if population < 2 or population != int(population):
        raise InvalidInputError("El tamaño de la población N debe ser un entero mayor o igual a 2.")
    return int(population)


def _check_design_effect(deff: float) -> float:
    if not deff >= 1:
        raise InvalidInputError("El efecto de diseño debe ser mayor o igual a 1.")
    return float(deff)


def finite_population_correction(n0: float, population: int | None) -> float:
    """Sample size after correcting an infinite-population size ``n0`` for population ``N``."""
    return n0 if population is None else n0 / (1 + (n0 - 1) / population)


def _finish(
    kind: str,
    confidence: float,
    margin: float,
    n0: float,
    population: int | None,
    deff: float,
    inputs: dict[str, float | None],
) -> SampleSizeResult:
    adjusted = finite_population_correction(n0, population)
    final = math.ceil(adjusted * deff - 1e-9)
    if population is not None:
        final = min(final, population)
    return SampleSizeResult(
        kind=kind,
        confidence=confidence,
        margin_of_error=margin,
        z=z_value(confidence),
        n_infinite=n0,
        n_adjusted=adjusted,
        n_final=int(final),
        design_effect=deff,
        population_size=population,
        inputs=inputs,
    )


def sample_size_proportion(
    confidence: float,
    margin_of_error: float,
    proportion: float = 0.5,
    population: int | None = None,
    design_effect: float = 1.0,
) -> SampleSizeResult:
    if not 0 < margin_of_error < 1:
        raise InvalidInputError("El margen de error debe estar entre 0 y 1 (p. ej. 0.05 para ±5 %).")
    if not 0 < proportion < 1:
        raise InvalidInputError("La proporción esperada debe estar entre 0 y 1 (use 0.5 si la desconoce).")
    population, deff = _check_population(population), _check_design_effect(design_effect)
    n0 = z_value(confidence) ** 2 * proportion * (1 - proportion) / margin_of_error**2
    result = _finish("proportion", confidence, margin_of_error, n0, population, deff, {"proportion": proportion})
    result.curve = _proportion_curve(confidence, proportion, population, deff)
    return result


def sample_size_mean(
    confidence: float,
    margin_of_error: float,
    std_dev: float,
    population: int | None = None,
    design_effect: float = 1.0,
) -> SampleSizeResult:
    if margin_of_error <= 0:
        raise InvalidInputError("El error máximo admitido debe ser mayor que 0.")
    if std_dev <= 0:
        raise InvalidInputError("La desviación estándar esperada debe ser mayor que 0.")
    population, deff = _check_population(population), _check_design_effect(design_effect)
    n0 = (z_value(confidence) * std_dev / margin_of_error) ** 2
    result = _finish("mean", confidence, margin_of_error, n0, population, deff, {"std_dev": std_dev})
    result.curve = _mean_curve(confidence, std_dev, margin_of_error, population, deff)
    return result


def _proportion_curve(confidence: float, p: float, population: int | None, deff: float) -> list[list[float]]:
    z = z_value(confidence)
    margins = np.linspace(0.01, 0.10, CURVE_POINTS)
    return [
        [float(e), float(math.ceil(finite_population_correction(z**2 * p * (1 - p) / e**2, population) * deff))]
        for e in margins
    ]


def _mean_curve(confidence: float, sd: float, margin: float, population: int | None, deff: float) -> list[list[float]]:
    z = z_value(confidence)
    margins = np.linspace(margin * 0.25, margin * 2.0, CURVE_POINTS)
    return [
        [float(e), float(math.ceil(finite_population_correction((z * sd / e) ** 2, population) * deff))]
        for e in margins
    ]


# --------------------------------------------------------------------------- allocation
def _largest_remainder(shares: np.ndarray, total: int, caps: np.ndarray) -> np.ndarray:
    """Integer allocation summing to ``total``, each stratum at least 1 (if possible) and at most its cap."""
    allocation = np.minimum(np.floor(shares).astype(int), caps)
    if total >= len(shares):
        allocation = np.maximum(allocation, 1)
    for _ in range(10 * len(shares) + total):  # bounded fix-up loop
        gap = total - int(allocation.sum())
        if gap == 0:
            break
        remainders = shares - allocation
        if gap > 0:
            candidates = np.where(allocation < caps)[0]
            if len(candidates) == 0:
                break
            allocation[candidates[np.argmax(remainders[candidates])]] += 1
        else:
            candidates = np.where(allocation > (1 if total >= len(shares) else 0))[0]
            if len(candidates) == 0:
                break
            allocation[candidates[np.argmin(remainders[candidates])]] -= 1
    return allocation


def allocate_strata(
    frame: pd.DataFrame,
    strata_column: str,
    total: int,
    method: str = "proportional",
    sd_column: str | None = None,
) -> list[StratumAllocation]:
    if method not in ALLOCATIONS:
        raise InvalidInputError(f"Asignación inválida «{method}». Use: {', '.join(ALLOCATIONS)}.")
    require_columns(frame, [strata_column])
    labels = frame[strata_column]
    if labels.isna().any():
        raise InvalidInputError(f"La columna de estratos «{strata_column}» tiene valores vacíos.")
    keys = labels.map(label_of)
    sizes = keys.value_counts().sort_index()
    if len(sizes) > MAX_STRATA:
        raise InvalidInputError(f"Hay más de {MAX_STRATA} estratos; revise la columna «{strata_column}».")
    if total > int(sizes.sum()):
        raise InvalidInputError("El tamaño de muestra no puede superar al de la población.")
    if total < len(sizes):
        raise InvalidInputError(f"La muestra ({total}) es menor que el número de estratos ({len(sizes)}).")
    population = sizes.to_numpy(dtype=float)
    if method == "proportional":
        weights = population
    elif method == "equal":
        weights = np.ones(len(sizes))
    else:
        if not sd_column:
            raise InvalidInputError("La asignación de Neyman requiere una variable numérica para estimar S_h.")
        values = numeric_series(frame, sd_column)
        spreads = values.groupby(keys).std(ddof=1).reindex(sizes.index).fillna(0.0).to_numpy()
        weights = population * spreads
        if weights.sum() == 0:
            raise InvalidInputError("La desviación estándar es cero en todos los estratos.")
    shares = total * weights / weights.sum()
    counts = _largest_remainder(shares, total, sizes.to_numpy(dtype=int))
    return [
        StratumAllocation(str(label), int(N), float(N / sizes.sum()), int(n))
        for label, N, n in zip(sizes.index, sizes.to_numpy(), counts, strict=True)
    ]


# --------------------------------------------------------------------------- drawing samples
def _attach_design(frame: pd.DataFrame, rows: np.ndarray, probability: np.ndarray | float) -> pd.DataFrame:
    sample = frame.iloc[rows].copy()
    sample[INCLUSION_COLUMN] = probability
    sample[WEIGHT_COLUMN] = 1.0 / sample[INCLUSION_COLUMN]
    return sample


def draw_sample(
    frame: pd.DataFrame,
    design: str,
    sample_size: int | None = None,
    seed: int | None = None,
    strata_column: str | None = None,
    allocation: str = "proportional",
    sd_column: str | None = None,
    cluster_column: str | None = None,
    n_clusters: int | None = None,
) -> SampleDrawResult:
    if design not in DESIGNS:
        raise InvalidInputError(f"Diseño muestral inválido «{design}». Use: {', '.join(DESIGNS)}.")
    for reserved in (WEIGHT_COLUMN, INCLUSION_COLUMN):
        if reserved in frame.columns:
            raise InvalidInputError(f"El marco muestral ya contiene la columna reservada «{reserved}».")
    population = len(frame)
    if population < 2:
        raise InvalidInputError("El marco muestral necesita al menos 2 unidades.")
    seed = int(seed) if seed is not None else secrets.randbelow(2**31)
    rng = np.random.default_rng(seed)
    notes: list[str] = []
    summary: list[TableData] = []
    interval: float | None = None

    if design == "cluster":
        sample, n = _cluster_sample(frame, cluster_column, n_clusters, rng, summary, notes)
    else:
        if sample_size is None or not 1 <= sample_size <= population:
            raise InvalidInputError(f"El tamaño de muestra debe estar entre 1 y {population}.")
        n = int(sample_size)
        if design == "srs":
            rows = np.sort(rng.choice(population, size=n, replace=False))
            sample = _attach_design(frame, rows, n / population)
        elif design == "systematic":
            interval = population / n
            start = rng.uniform(0, interval)
            rows = np.minimum(np.floor(start + interval * np.arange(n)).astype(int), population - 1)
            sample = _attach_design(frame, rows, n / population)
            notes.append(
                f"Intervalo k = N/n = {interval:.4g}; arranque aleatorio = {start:.4g}. "
                "Se respeta el orden del marco muestral."
            )
        else:
            sample = _stratified_sample(frame, strata_column, n, allocation, sd_column, rng, summary)

    summary.insert(
        0,
        TableData(
            "Resumen del diseño",
            ["Concepto", "Valor"],
            [
                ["Diseño", _DESIGN_LABELS[design]],
                ["Población (N)", population],
                ["Muestra (n)", len(sample)],
                ["Semilla", seed],
                *([["Intervalo (k)", interval]] if interval else []),
            ],
        ),
    )
    return SampleDrawResult(
        design=design,
        population_size=population,
        sample_size=len(sample),
        seed=seed,
        interval=interval,
        notes=notes,
        summary=summary,
        sample=sample.reset_index(drop=True),
    )


_DESIGN_LABELS = {
    "srs": "Aleatorio simple sin reemplazo",
    "systematic": "Sistemático",
    "stratified": "Estratificado",
    "cluster": "Por conglomerados (una etapa)",
}


def _stratified_sample(
    frame: pd.DataFrame,
    strata_column: str | None,
    n: int,
    allocation: str,
    sd_column: str | None,
    rng: np.random.Generator,
    summary: list[TableData],
) -> pd.DataFrame:
    if not strata_column:
        raise InvalidInputError("Seleccione la columna de estratos.")
    plan = allocate_strata(frame, strata_column, n, allocation, sd_column)
    keys = frame[strata_column].map(label_of)
    pieces: list[pd.DataFrame] = []
    for stratum in plan:
        members = np.flatnonzero((keys == stratum.stratum).to_numpy())
        chosen = np.sort(rng.choice(members, size=stratum.sample_size, replace=False))
        pieces.append(_attach_design(frame, chosen, stratum.sample_size / stratum.population))
    summary.append(
        TableData(
            "Asignación por estrato",
            ["Estrato", "N_h", "Peso poblacional", "n_h", "Fracción de muestreo", "Peso de diseño"],
            [
                [
                    s.stratum,
                    s.population,
                    s.share,
                    s.sample_size,
                    s.sample_size / s.population,
                    s.population / s.sample_size,
                ]
                for s in plan
            ],
        )
    )
    return pd.concat(pieces).sort_index()


def _cluster_sample(
    frame: pd.DataFrame,
    cluster_column: str | None,
    n_clusters: int | None,
    rng: np.random.Generator,
    summary: list[TableData],
    notes: list[str],
) -> tuple[pd.DataFrame, int]:
    if not cluster_column:
        raise InvalidInputError("Seleccione la columna de conglomerados.")
    require_columns(frame, [cluster_column])
    if frame[cluster_column].isna().any():
        raise InvalidInputError(f"La columna de conglomerados «{cluster_column}» tiene valores vacíos.")
    keys = frame[cluster_column].map(label_of)
    sizes = keys.value_counts().sort_index()
    total_clusters = len(sizes)
    if n_clusters is None or not 1 <= n_clusters <= total_clusters:
        raise InvalidInputError(f"El número de conglomerados a seleccionar debe estar entre 1 y {total_clusters}.")
    chosen = sorted(rng.choice(sizes.index.to_numpy(), size=int(n_clusters), replace=False))
    rows = np.flatnonzero(keys.isin(chosen).to_numpy())
    sample = _attach_design(frame, rows, n_clusters / total_clusters)
    summary.append(
        TableData(
            "Conglomerados seleccionados", ["Conglomerado", "Unidades"], [[str(c), int(sizes[c])] for c in chosen]
        )
    )
    notes.append(
        f"Se seleccionaron {n_clusters} de {total_clusters} conglomerados y se encuestan todas sus unidades; "
        "el tamaño final de muestra depende del tamaño de los conglomerados elegidos."
    )
    return sample, len(sample)
