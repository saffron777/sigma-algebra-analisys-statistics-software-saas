"""Multivariate analysis (Minitab *Multivariate*, SPSS *Dimension Reduction*/*Scale*): components, factors, reliability, canonical correlation."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import MinCovDet
from sklearn.manifold import MDS
from statsmodels.multivariate.cancorr import CanCorr
from statsmodels.multivariate.factor import Factor

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import bar_chart, heatmap_chart, line_chart, scatter_chart
from sigma.domain.analysis.output import Output, P, bad, good, strong
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.analysis.support import load_line
from sigma.domain.errors import ComputationError, InvalidInputError

MODULE = "multivariado"
GROUP_REDUCE = "Reducción de dimensión"
GROUP_SCALES = "Escalas y asociación"
GROUP_MAPS = "Mapas y atípicos"
VARS = RoleSpec("vars", "Variables numéricas (2 o más)", "numeric", multiple=True, min_count=2)
ALPHA = FieldSpec("alpha", "Nivel de significancia α", "number", 0.05, minimum=0.001, maximum=0.5)
LABEL = RoleSpec("label", "Etiqueta de cada fila (opcional)", "any", optional=True)
COLOR = RoleSpec("group", "Colorear por (opcional)", "categorical", optional=True)
MAX_POINTS = 4000


def complete(frame: pd.DataFrame, names: list[str], minimum: int = 8) -> pd.DataFrame:
    """The chosen numeric columns with every incomplete row left out."""
    data = frame[names].apply(pd.to_numeric, errors="coerce")
    clean = data.dropna()
    if len(clean) < minimum:
        raise InvalidInputError(f"Se necesitan al menos {minimum} filas completas en esas columnas (hay {len(clean)}).")
    if (clean.std(ddof=1) == 0).any():
        raise InvalidInputError(
            f"«{clean.columns[(clean.std(ddof=1) == 0).to_numpy()][0]}» es constante: no aporta información."
        )
    return clean


def dropped_alert(out: Output, frame: pd.DataFrame, clean: pd.DataFrame) -> None:
    if len(clean) < len(frame):
        out.alert(
            "info", f"Se omitieron {len(frame) - len(clean)} fila(s) con datos faltantes en las columnas elegidas."
        )


def correlation_matrix(data: np.ndarray) -> np.ndarray:
    return np.corrcoef(data, rowvar=False)


def kmo(r: np.ndarray) -> tuple[float, np.ndarray]:
    """Kaiser-Meyer-Olkin measure of sampling adequacy: (overall, per variable)."""
    inverse = np.linalg.pinv(r)
    scale = np.sqrt(np.outer(np.diag(inverse), np.diag(inverse)))
    partial = -inverse / scale
    np.fill_diagonal(partial, 0.0)
    off = r - np.eye(len(r))
    r2, q2 = (off**2).sum(axis=0), (partial**2).sum(axis=0)
    return float(r2.sum() / (r2.sum() + q2.sum())), r2 / (r2 + q2)


def bartlett_sphericity(r: np.ndarray, n: int) -> tuple[float, int, float]:
    """Bartlett's test that the correlation matrix is the identity: (chi-square, df, p-value)."""
    p = r.shape[0]
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.linalg.slogdet(r)[1]
    df = p * (p - 1) // 2
    return float(chi2), df, float(stats.chi2.sf(chi2, df))


def _kmo_label(value: float) -> str:
    for limit, text in ((0.5, "inaceptable"), (0.6, "mediocre"), (0.7, "regular"), (0.8, "buena"), (0.9, "meritoria")):
        if value < limit:
            return text
    return "maravillosa"


def _fix_signs(vectors: np.ndarray) -> np.ndarray:
    """Make the largest loading of each component positive so that results are reproducible."""
    signs = np.sign(vectors[np.abs(vectors).argmax(axis=0), np.arange(vectors.shape[1])])
    return vectors * np.where(signs == 0, 1, signs)


def _result_frame(frame: pd.DataFrame, index: pd.Index, **columns: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    for name, values in columns.items():
        series = pd.Series(np.nan, index=frame.index, dtype=float)
        series.loc[index] = values
        result[name] = series
    return result


# ============================================================================ principal components
def parallel_analysis(n: int, p: int, reps: int = 200, seed: int = 20240601) -> np.ndarray:
    """Horn's parallel analysis: the 95th percentile of the eigenvalues of correlation matrices of random normal data."""
    rng = np.random.default_rng(seed)
    eigen = np.empty((reps, p))
    for i in range(reps):
        eigen[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False)))[::-1]
    return np.quantile(eigen, 0.95, axis=0)


def _pca(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names = roles["vars"]
    data = complete(frame, names)
    n, k = data.shape
    if n <= k:
        raise InvalidInputError(f"Se necesitan más filas ({n}) que variables ({k}).")
    x = data.to_numpy(float)
    centred = x - x.mean(axis=0)
    standardize = bool(p["standardize"])
    z = centred / x.std(axis=0, ddof=1) if standardize else centred
    matrix = np.cov(z, rowvar=False)
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0), _fix_signs(vectors[:, order])
    total = values.sum()
    share = values / total
    rule = p["rule"]
    horn = parallel_analysis(n, k) if standardize else None
    if rule == "fixed":
        count = int(min(max(p["components"], 1), k))
    elif rule == "variance":
        count = int(np.searchsorted(np.cumsum(share), p["variance"] / 100) + 1)
    elif rule == "parallel":
        if horn is None:
            raise InvalidInputError("El análisis paralelo necesita la matriz de correlaciones: marque «Estandarizar».")
        count = int(max(np.sum(values > horn), 1))
    else:
        threshold = 1.0 if standardize else float(values.mean())
        count = int(max(np.sum(values > threshold), 1))
    count = min(count, k)
    loadings = vectors * np.sqrt(values)
    rotation = p["rotation"]
    if rotation == "varimax" and count >= 2:
        from statsmodels.multivariate.factor_rotation import rotate_factors

        rotated, transform = rotate_factors(loadings[:, :count], "varimax")
        loadings[:, :count] = rotated
        rotation_text = "varimax"
    else:
        transform, rotation_text = np.eye(count), "sin rotar"
    scores = z @ vectors[:, :count]
    scores = scores @ transform if rotation == "varimax" and count >= 2 else scores
    r = correlation_matrix(x)
    overall, per_variable = kmo(r)
    chi2, df, pvalue = bartlett_sphericity(r, n)
    out = Output(
        "Análisis de componentes principales",
        f"{k} variables · n = {n} · {'matriz de correlaciones' if standardize else 'matriz de covarianzas'} · {count} componente(s) retenido(s)",
    )
    dropped_alert(out, frame, data)
    out.tiles(
        [
            ("Componentes retenidos", count),
            ("Varianza explicada", 100 * float(share[:count].sum())),
            ("KMO", overall),
            ("Bartlett p", P(pvalue)),
        ]
    )
    table = [
        [
            f"CP{i + 1}",
            float(values[i]),
            100 * float(share[i]),
            100 * float(share[: i + 1].sum()),
            good("●") if i < count else "",
        ]
        for i in range(k)
    ]
    out.table(
        "Valores propios y varianza explicada",
        ["Componente", "Valor propio", "% de la varianza", "% acumulado", "Retenido"],
        table,
        note=f"Regla usada: {dict(_PCA_RULES)[rule].lower()}.",
    )
    out.table(
        f"Cargas de los componentes ({rotation_text})"
        + (" = correlación de cada variable con el componente" if standardize else ""),
        ["Variable", *[f"CP{i + 1}" for i in range(count)], "Comunalidad"],
        [
            [names[j], *(float(loadings[j, i]) for i in range(count)), float((loadings[j, :count] ** 2).sum())]
            for j in range(k)
        ],
        note="Comunalidad: parte de la varianza de la variable que explican los componentes retenidos.",
    )
    contribution = 100 * (loadings[:, :count] ** 2) / np.maximum((loadings[:, :count] ** 2).sum(axis=0), 1e-12)
    out.table(
        "Contribución de cada variable a cada componente (%)",
        ["Variable", *[f"CP{i + 1}" for i in range(count)]],
        [[names[j], *(float(contribution[j, i]) for i in range(count))] for j in range(k)],
    )
    out.table(
        "Adecuación de la muestra",
        ["Medida", "Valor", "Lectura"],
        [
            ["KMO global", overall, _kmo_label(overall)],
            [
                f"Bartlett (χ², {df} gl)",
                chi2,
                "hay correlaciones que justifican reducir"
                if pvalue < p["alpha"]
                else "las variables casi no se correlacionan: poco que reducir",
            ],
            *[[f"MSA de {names[j]}", float(per_variable[j]), _kmo_label(float(per_variable[j]))] for j in range(k)],
        ],
        note="KMO y MSA deberían ser al menos 0.6 para que un análisis de componentes tenga sentido.",
    )
    scree = [{"name": "Valor propio", "x": np.arange(1, k + 1), "y": values, "markers": True}]
    if horn is not None:
        scree.append({"name": "Análisis paralelo (95 %)", "x": np.arange(1, k + 1), "y": horn, "dash": "dash"})
    out.chart(
        line_chart(
            "Gráfico de sedimentación",
            "Componente",
            "Valor propio",
            scree,
            hlines=[
                {"value": 1.0 if standardize else float(values.mean()), "label": "criterio de Kaiser", "dash": "dot"}
            ],
        )
    )
    if count >= 2:
        group = roles.get("group")
        labels = frame.loc[data.index, group].astype(str).to_numpy() if group else None
        series = []
        if labels is not None:
            for level in sorted(set(labels)):
                mask = labels == level
                series.append({"name": level, "x": scores[mask, 0], "y": scores[mask, 1]})
        else:
            series.append({"name": "Observaciones", "x": scores[:, 0], "y": scores[:, 1]})
        out.chart(
            scatter_chart(
                f"Puntuaciones: CP1 ({100 * share[0]:.1f} %) y CP2 ({100 * share[1]:.1f} %)",
                "CP1",
                "CP2",
                series,
                hlines=[{"value": 0.0}],
                vlines=[{"value": 0.0}],
                height=420,
            )
        )
        circle = np.linspace(0, 2 * np.pi, 100)
        arrows = [
            {"name": names[j], "x": np.array([0.0, loadings[j, 0]]), "y": np.array([0.0, loadings[j, 1]])}
            for j in range(k)
        ]
        out.chart(
            scatter_chart(
                "Círculo de correlaciones (cargas en CP1 y CP2)",
                "CP1",
                "CP2",
                [
                    {
                        "name": "Variables",
                        "x": loadings[:, 0],
                        "y": loadings[:, 1],
                        "text": names,
                        "labels": True,
                        "size": 9,
                    }
                ],
                lines=[*arrows, {"name": "Círculo unitario", "x": np.cos(circle), "y": np.sin(circle), "dash": "dot"}]
                if standardize
                else arrows,
                height=420,
            )
        )
    out.dataset(
        "puntuaciones",
        "Puntuaciones de los componentes",
        _result_frame(frame, data.index, **{f"CP{i + 1}": scores[:, i] for i in range(count)}),
    )
    return out.set(
        eigenvalues=[float(v) for v in values],
        variance_share=[float(v) for v in share],
        components=count,
        kmo=overall,
        bartlett_p=pvalue,
        loadings={names[j]: [float(loadings[j, i]) for i in range(count)] for j in range(k)},
    ).build()


_PCA_RULES = (
    ("kaiser", "Kaiser (valor propio > 1)"),
    ("parallel", "Análisis paralelo de Horn"),
    ("variance", "Varianza acumulada mínima"),
    ("fixed", "Número fijo"),
)


def _pca_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\n\n"
        + load_line(dataset)
        + f"X = df[{roles['vars']!r}].dropna()\n"
        + ("X = StandardScaler().fit_transform(X)\n" if p["standardize"] else "")
        + "pca = PCA().fit(X)\nprint(pca.explained_variance_, pca.explained_variance_ratio_)\nprint(pca.components_)\n"
    )


# ============================================================================ factor analysis
_ROTATIONS = (
    ("varimax", "Varimax (ortogonal)"),
    ("quartimax", "Quartimax (ortogonal)"),
    ("equamax", "Equamax (ortogonal)"),
    ("promax", "Promax (oblicua)"),
    ("oblimin", "Oblimin directa (oblicua)"),
    ("none", "Sin rotar"),
)


def _real(values: Any) -> np.ndarray:
    """A float array from a result that statsmodels may hand back as complex with a null imaginary part."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        if np.abs(array.imag).max(initial=0.0) > 1e-8:
            raise ComputationError(
                "La estimación factorial no convergió a una solución real; pruebe con menos factores."
            )
        array = array.real
    return array.astype(float)


def ml_chi_square(sample: np.ndarray, fitted: np.ndarray, n: int, k: int) -> tuple[float, int, float]:
    """Likelihood-ratio test that ``k`` factors are enough: (chi-square, df, p-value)."""
    p = sample.shape[0]
    df = ((p - k) ** 2 - (p + k)) // 2
    if df <= 0:
        return float("nan"), df, float("nan")
    discrepancy = (
        np.linalg.slogdet(fitted)[1] - np.linalg.slogdet(sample)[1] + np.trace(np.linalg.solve(fitted, sample)) - p
    )
    chi2 = (n - 1 - (2 * p + 5) / 6 - 2 * k / 3) * discrepancy
    return float(chi2), df, float(stats.chi2.sf(chi2, df))


def _factor(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names = roles["vars"]
    data = complete(frame, names, 20)
    n, k = data.shape
    factors = int(p["factors"])
    if factors >= k or ((k - factors) ** 2 - (k + factors)) < 0:
        raise InvalidInputError(
            f"Con {k} variables no se pueden estimar {factors} factor(es): reduzca el número de factores."
        )
    r = correlation_matrix(data.to_numpy(float))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = Factor(data, n_factor=factors, method=p["method"]).fit()
    except (np.linalg.LinAlgError, ValueError) as error:
        raise ComputationError(f"No se pudo estimar el modelo factorial: {error}") from None
    unrotated = _real(fitted.loadings).copy()
    oblique = p["rotation"] in ("promax", "oblimin")
    phi = np.eye(factors)
    rotation_text = "sin rotar"
    if p["rotation"] != "none" and factors >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted.rotate(p["rotation"])
        rotation_text = dict(_ROTATIONS)[p["rotation"]].split(" (")[0].lower()
        if oblique:
            transform = _real(fitted.rotation_matrix)
            phi = transform.T @ transform
            scale = np.sqrt(np.diag(phi))
            phi = phi / np.outer(scale, scale)
    loadings = _real(fitted.loadings)
    communalities, uniqueness = _real(fitted.communality), _real(fitted.uniqueness)
    overall, per_variable = kmo(r)
    chi2, df, pvalue = bartlett_sphericity(r, n)
    out = Output(
        f"Análisis factorial ({'máxima verosimilitud' if p['method'] == 'ml' else 'ejes principales'})",
        f"{k} variables · {factors} factor(es) · rotación {rotation_text} · n = {n}",
    )
    dropped_alert(out, frame, data)
    explained = (loadings**2).sum(axis=0)
    out.tiles(
        [
            ("Factores", factors),
            ("Varianza explicada", 100 * float(explained.sum() / k)),
            ("KMO", overall),
            ("Bartlett p", P(pvalue)),
        ]
    )
    out.table(
        f"Cargas factoriales ({rotation_text})",
        ["Variable", *[f"F{i + 1}" for i in range(factors)], "Comunalidad", "Unicidad"],
        [
            [
                names[j],
                *(
                    strong(float(loadings[j, i])) if abs(loadings[j, i]) >= p["cutoff"] else float(loadings[j, i])
                    for i in range(factors)
                ),
                float(communalities[j]),
                float(uniqueness[j]),
            ]
            for j in range(k)
        ],
        note=f"En negrita: |carga| ≥ {p['cutoff']:g}. Unicidad = 1 − comunalidad: la parte propia de cada variable que los factores no explican.",
    )
    out.table(
        "Varianza explicada por factor",
        ["Factor", "Suma de cuadrados de cargas", "% de la varianza total", "% acumulado"],
        [
            [f"F{i + 1}", float(explained[i]), 100 * float(explained[i] / k), 100 * float(explained[: i + 1].sum() / k)]
            for i in range(factors)
        ],
        note=(
            "Con rotación oblicua los factores están correlacionados, por lo que estos porcentajes se solapan."
            if oblique
            else None
        ),
    )
    if oblique and factors >= 2:
        out.table(
            "Correlaciones entre factores",
            ["", *[f"F{i + 1}" for i in range(factors)]],
            [[f"F{i + 1}", *(float(phi[i, j]) for j in range(factors))] for i in range(factors)],
        )
    if p["method"] == "ml":
        sample = np.cov(data.to_numpy(float), rowvar=False)
        sample = sample / np.sqrt(np.outer(np.diag(sample), np.diag(sample)))
        implied = unrotated @ unrotated.T + np.diag(uniqueness)  # the rotation does not change the implied correlations
        test = ml_chi_square(sample, implied, n, factors)
        if np.isfinite(test[0]):
            out.table(
                "Prueba de bondad de ajuste (máxima verosimilitud)",
                ["χ²", "gl", "Valor p", "Lectura"],
                [
                    [
                        test[0],
                        test[1],
                        P(test[2]),
                        "los factores no bastan: aumente su número"
                        if test[2] < p["alpha"]
                        else f"{factors} factor(es) reproducen bien la matriz de correlaciones",
                    ]
                ],
                note="H₀: el número de factores elegido es suficiente.",
            )
    out.table(
        "Adecuación de la muestra",
        ["Medida", "Valor", "Lectura"],
        [
            ["KMO global", overall, _kmo_label(overall)],
            [
                f"Bartlett (χ², {df} gl)",
                chi2,
                "las variables están correlacionadas" if pvalue < p["alpha"] else "casi no hay correlaciones",
            ],
        ],
    )
    out.chart(
        heatmap_chart(
            f"Cargas factoriales ({rotation_text})",
            loadings,
            [f"F{i + 1}" for i in range(factors)],
            names,
            annotate=True,
            midpoint=0.0,
            height=max(300, 34 * k + 120),
        )
    )
    scores = None
    if p["scores"] != "none" and factors >= 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = _real(fitted.factor_scoring(method=p["scores"]))
        out.dataset(
            "puntuaciones",
            "Puntuaciones factoriales",
            _result_frame(frame, data.index, **{f"F{i + 1}": scores[:, i] for i in range(scores.shape[1])}),
        )
    return out.set(
        loadings={names[j]: [float(loadings[j, i]) for i in range(factors)] for j in range(k)},
        communalities={names[j]: float(communalities[j]) for j in range(k)},
        variance_explained=100 * float(explained.sum() / k),
        kmo=overall,
    ).build()


def _factor_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from statsmodels.multivariate.factor import Factor\n\n"
        + load_line(dataset)
        + f"fa = Factor(df[{roles['vars']!r}].dropna(), n_factor={int(p['factors'])}, method={p['method']!r}).fit()\n"
        + (f"fa.rotate({p['rotation']!r})\n" if p["rotation"] != "none" else "")
        + "print(fa.loadings)\nprint(fa.uniqueness)\n"
    )


# ============================================================================ reliability (Cronbach's alpha, omega, item analysis)
def cronbach_alpha(items: np.ndarray) -> float:
    k = items.shape[1]
    return float(k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum() / items.sum(axis=1).var(ddof=1)))


def _reliability(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names = roles["vars"]
    reverse = roles.get("reverse", [])
    unknown = [c for c in reverse if c not in names]
    if unknown:
        raise InvalidInputError(f"Las columnas a invertir deben estar entre los ítems: «{unknown[0]}» no lo está.")
    if len(names) < 3:
        raise InvalidInputError("Un análisis de fiabilidad necesita al menos 3 ítems.")
    data = complete(frame, names, 10)
    for column in reverse:
        data[column] = data[column].max() + data[column].min() - data[column]
    items, n, k = data.to_numpy(float), len(data), len(names)
    alpha = cronbach_alpha(items)
    r = correlation_matrix(items)
    mean_r = float((r.sum() - k) / (k * (k - 1)))
    standardized = k * mean_r / (1 + (k - 1) * mean_r)
    total = items.sum(axis=1)
    rows = []
    for j in range(k):
        rest = np.delete(items, j, axis=1)
        corrected = float(np.corrcoef(items[:, j], total - items[:, j])[0, 1])
        rows.append(
            [
                names[j] + (" (invertido)" if names[j] in reverse else ""),
                float(items[:, j].mean()),
                float(items[:, j].std(ddof=1)),
                corrected,
                cronbach_alpha(rest) if rest.shape[1] > 1 else float("nan"),
            ]
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loading = np.asarray(Factor(data, n_factor=1, method="ml").fit().loadings, float)[:, 0]
    omega = float(loading.sum() ** 2 / (loading.sum() ** 2 + (1 - loading**2).sum()))
    odd, even = items[:, ::2].sum(axis=1), items[:, 1::2].sum(axis=1)
    split_r = float(np.corrcoef(odd, even)[0, 1])
    spearman_brown = 2 * split_r / (1 + split_r)
    sem = float(total.std(ddof=1) * np.sqrt(1 - alpha))
    out = Output(
        "Fiabilidad de una escala",
        f"{k} ítems · n = {n}" + (f" · {len(reverse)} ítem(s) invertido(s)" if reverse else ""),
    )
    dropped_alert(out, frame, data)
    out.tiles(
        [
            ("Alfa de Cronbach", alpha),
            ("Alfa estandarizado", standardized),
            ("Omega de McDonald", omega),
            ("Correlación media entre ítems", mean_r),
        ]
    )
    label = (
        "excelente"
        if alpha >= 0.9
        else "buena"
        if alpha >= 0.8
        else "aceptable"
        if alpha >= 0.7
        else "cuestionable"
        if alpha >= 0.6
        else "pobre"
    )
    out.alert(
        "success" if alpha >= 0.7 else "warning",
        f"Fiabilidad {label} (alfa = {alpha:.3f}). Referencia habitual: ≥ 0.7 para investigación, ≥ 0.9 para decisiones individuales.",
    )
    out.table(
        "Análisis de los ítems",
        ["Ítem", "Media", "Desv. est.", "Correlación ítem-total corregida", "Alfa si se elimina el ítem"],
        rows,
        note="Un ítem cuyo «alfa si se elimina» supera al alfa total, o con correlación ítem-total baja (< 0.3), empeora la escala.",
    )
    better = [r_[0] for r_ in rows if np.isfinite(r_[4]) and r_[4] > alpha + 0.005]
    if better:
        out.alert("info", "Quitar estos ítems aumentaría el alfa: " + ", ".join(better) + ".")
    negative = [r_[0] for r_ in rows if r_[3] < 0]
    if negative:
        out.alert(
            "warning",
            "Correlación ítem-total negativa en "
            + ", ".join(negative)
            + ": probablemente estén redactados al revés; use «Ítems a invertir».",
        )
    out.table(
        "Otras medidas",
        ["Medida", "Valor"],
        [
            ["Omega total (un factor, máxima verosimilitud)", omega],
            ["Dos mitades (impares/pares): correlación", split_r],
            ["Spearman-Brown (fiabilidad de la prueba completa)", float(spearman_brown)],
            ["Error estándar de medida", sem],
            ["Puntuación media de la escala", float(total.mean())],
            ["Desv. est. de la puntuación total", float(total.std(ddof=1))],
        ],
    )
    out.chart(
        heatmap_chart(
            "Correlaciones entre ítems", r, names, names, annotate=k <= 12, midpoint=0.0, height=max(320, 40 * k + 120)
        )
    )
    return out.set(
        alpha=alpha,
        standardized_alpha=standardized,
        omega=omega,
        mean_inter_item=mean_r,
        item_total={r_[0]: r_[3] for r_ in rows},
        split_half=float(spearman_brown),
    ).build()


def _reliability_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        load_line(dataset)
        + f"items = df[{roles['vars']!r}].dropna()\nk = items.shape[1]\nalpha = k / (k - 1) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1))\nprint(alpha)\n"
    )


# ============================================================================ canonical correlation
def _cca(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    xs, ys = roles["x_vars"], roles["y_vars"]
    if set(xs) & set(ys):
        raise InvalidInputError("Una columna no puede estar en los dos conjuntos.")
    data = complete(frame, [*xs, *ys], 10)
    n, px, py = len(data), len(xs), len(ys)
    if n <= px + py:
        raise InvalidInputError("Se necesitan más filas que variables.")
    xz = (data[xs] - data[xs].mean()) / data[xs].std(ddof=1)
    yz = (data[ys] - data[ys].mean()) / data[ys].std(ddof=1)
    model = CanCorr(yz, xz)
    correlations = np.asarray(model.cancorr, float)
    q = len(correlations)
    tests = model.corr_test().stats
    x_scores, y_scores = np.asarray(xz) @ np.asarray(model.x_cancoef), np.asarray(yz) @ np.asarray(model.y_cancoef)
    x_scores, y_scores = x_scores[:, :q], y_scores[:, :q]
    x_loadings = np.array([[np.corrcoef(xz.iloc[:, i], x_scores[:, j])[0, 1] for j in range(q)] for i in range(px)])
    y_loadings = np.array([[np.corrcoef(yz.iloc[:, i], y_scores[:, j])[0, 1] for j in range(q)] for i in range(py)])
    sign = np.sign([np.corrcoef(x_scores[:, j], y_scores[:, j])[0, 1] for j in range(q)])
    x_loadings, y_loadings = (
        x_loadings * sign,
        y_loadings,
    )  # the pair is oriented so that both variates correlate positively
    redundancy_x = (x_loadings**2).mean(axis=0) * correlations**2
    redundancy_y = (y_loadings**2).mean(axis=0) * correlations**2
    out = Output("Correlación canónica", f"{px} variables X y {py} variables Y · n = {n}")
    dropped_alert(out, frame, data)
    out.tiles(
        [
            ("Pares canónicos", q),
            ("Primera correlación", float(correlations[0])),
            ("Redundancia de Y dada X", 100 * float(redundancy_y.sum())),
            ("Redundancia de X dada Y", 100 * float(redundancy_x.sum())),
        ]
    )
    rows = []
    for i in range(q):
        rows.append(
            [
                i + 1,
                float(correlations[i]),
                float(correlations[i] ** 2),
                float(tests.iloc[i]["Wilks' lambda"]),
                float(tests.iloc[i]["Num DF"]),
                float(tests.iloc[i]["Den DF"]),
                float(tests.iloc[i]["F Value"]),
                P(float(tests.iloc[i]["Pr > F"])),
            ]
        )
    out.table(
        "Correlaciones canónicas y su significancia",
        ["Par", "Correlación", "R² (varianza compartida)", "Lambda de Wilks", "gl (num)", "gl (den)", "F", "Valor p"],
        rows,
        note="Cada prueba contrasta que ese par y todos los siguientes no tengan correlación.",
    )
    out.table(
        "Cargas de las variables X (correlación con su variable canónica)",
        ["Variable", *[f"U{i + 1}" for i in range(q)]],
        [[xs[i], *(float(x_loadings[i, j]) for j in range(q))] for i in range(px)],
    )
    out.table(
        "Cargas de las variables Y (correlación con su variable canónica)",
        ["Variable", *[f"V{i + 1}" for i in range(q)]],
        [[ys[i], *(float(y_loadings[i, j]) for j in range(q))] for i in range(py)],
    )
    out.table(
        "Índices de redundancia",
        ["Par", "Redundancia de Y dada X", "Redundancia de X dada Y"],
        [[i + 1, float(redundancy_y[i]), float(redundancy_x[i])] for i in range(q)],
        note="Redundancia: proporción de la varianza de un conjunto que explica el otro a través de ese par.",
    )
    out.chart(
        scatter_chart(
            f"Primer par canónico (r = {correlations[0]:.3f})",
            "U1 (variable canónica de X)",
            "V1 (variable canónica de Y)",
            [{"name": "Observaciones", "x": x_scores[:, 0] * sign[0], "y": y_scores[:, 0]}],
        )
    )
    out.chart(
        bar_chart(
            "Correlaciones canónicas",
            "Par canónico",
            "Correlación",
            [str(i + 1) for i in range(q)],
            [{"name": "Correlación", "y": correlations}],
        )
    )
    return out.set(
        correlations=[float(v) for v in correlations],
        wilks_p=[float(tests.iloc[i]["Pr > F"]) for i in range(q)],
        redundancy_y=float(redundancy_y.sum()),
    ).build()


def _cca_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from statsmodels.multivariate.cancorr import CanCorr\n\n"
        + load_line(dataset)
        + f"d = df[{[*roles['x_vars'], *roles['y_vars']]!r}].dropna()\nmodel = CanCorr(d[{roles['y_vars']!r}], d[{roles['x_vars']!r}])\nprint(model.cancorr)\nprint(model.corr_test().summary())\n"
    )


# ============================================================================ Mahalanobis distance and multivariate outliers
def _mahalanobis(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names, alpha = roles["vars"], p["alpha"]
    data = complete(frame, names, 10)
    n, k = data.shape
    if n <= k + 1:
        raise InvalidInputError(f"Se necesitan más de {k + 1} filas completas.")
    x = data.to_numpy(float)
    if p["method"] == "robust":
        estimator = MinCovDet(random_state=int(p["seed"])).fit(x)
        d2 = np.asarray(estimator.mahalanobis(x), float)
        centre = estimator.location_
    else:
        centre = x.mean(axis=0)
        d2 = np.einsum("ij,jk,ik->i", x - centre, np.linalg.pinv(np.cov(x, rowvar=False)), x - centre)
    cutoff = float(stats.chi2.ppf(1 - alpha, k))
    pvalues = stats.chi2.sf(d2, k)
    flagged = np.flatnonzero(d2 > cutoff)
    out = Output(
        "Distancia de Mahalanobis y atípicos multivariados",
        f"{'estimación robusta (MCD)' if p['method'] == 'robust' else 'media y covarianza clásicas'} · {k} variables · n = {n}",
    )
    dropped_alert(out, frame, data)
    out.tiles(
        [
            ("Observaciones", n),
            ("Atípicas", len(flagged)),
            ("Corte (χ²)", cutoff),
            ("Mayor distancia²", float(d2.max())),
        ]
    )
    out.alert(
        "warning" if len(flagged) else "success",
        f"{len(flagged)} observación(es) con D² > {cutoff:.2f} (χ² con {k} gl, α = {alpha:g})."
        if len(flagged)
        else "Ninguna observación supera el corte.",
    )
    order = np.argsort(d2)[::-1][:25]
    index = data.index
    out.table(
        "Observaciones más alejadas",
        ["Fila", "D²", "Valor p", "¿Atípica?", *names],
        [
            [
                int(index[i]) + 1,
                float(d2[i]),
                P(float(pvalues[i])),
                bad("sí") if d2[i] > cutoff else "no",
                *(float(x[i, j]) for j in range(min(k, 6))),
            ]
            for i in order
        ],
        note="Los números de fila empiezan en 1. Con la media y la covarianza clásicas los atípicos pueden «esconderse» a sí mismos (enmascaramiento): pruebe el método robusto.",
    )
    out.chart(
        scatter_chart(
            "Distancia de Mahalanobis² por observación",
            "Fila",
            "D²",
            [{"name": "D²", "x": index.to_numpy() + 1, "y": d2}],
            hlines=[{"value": cutoff, "label": f"χ² {100 * (1 - alpha):g} %", "dash": "dash"}],
        )
    )
    theoretical = stats.chi2.ppf((np.arange(1, n + 1) - 0.5) / n, k)
    out.chart(
        scatter_chart(
            "D² frente a los cuantiles de χ² (si los datos son normales, sigue la recta)",
            "Cuantil teórico de χ²",
            "D² ordenada",
            [{"name": "Datos", "x": theoretical, "y": np.sort(d2)}],
            lines=[
                {
                    "name": "Referencia",
                    "x": np.array([0.0, theoretical.max()]),
                    "y": np.array([0.0, theoretical.max()]),
                    "dash": "dash",
                }
            ],
        )
    )
    out.dataset(
        "distancias",
        "Distancia de Mahalanobis",
        _result_frame(frame, data.index, mahalanobis_d2=d2, valor_p=pvalues, atipica=(d2 > cutoff).astype(float)),
    )
    return out.set(flagged=int(len(flagged)), cutoff=cutoff, rows=[int(index[i]) + 1 for i in flagged[:50]]).build()


def _mahalanobis_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import numpy as np\nfrom scipy import stats\n\n"
        + load_line(dataset)
        + f"X = df[{roles['vars']!r}].dropna().to_numpy()\nd = X - X.mean(axis=0)\nd2 = np.einsum('ij,jk,ik->i', d, np.linalg.pinv(np.cov(X, rowvar=False)), d)\nprint(d2[d2 > stats.chi2.ppf(1 - {p['alpha']}, X.shape[1])])\n"
    )


# ============================================================================ correspondence analysis
def correspondence(table: np.ndarray) -> dict[str, np.ndarray]:
    """Simple correspondence analysis of a contingency table: coordinates, inertia and the quality of the representation."""
    total = table.sum()
    p = table / total
    r, c = p.sum(axis=1), p.sum(axis=0)
    standardized = (p - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    u, s, vt = np.linalg.svd(standardized, full_matrices=False)
    dims = min(table.shape) - 1
    u, s, v = u[:, :dims], s[:dims], vt.T[:, :dims]
    rows = (u * s) / np.sqrt(r)[:, None]
    cols = (v * s) / np.sqrt(c)[:, None]
    inertia = s**2
    row_inertia = r * ((p / r[:, None] - c) ** 2 / c).sum(axis=1)
    col_inertia = c * ((p / c[None, :] - r[:, None]) ** 2 / r[:, None]).sum(axis=0)
    return {
        "rows": rows,
        "cols": cols,
        "inertia": inertia,
        "row_mass": r,
        "col_mass": c,
        "row_inertia": row_inertia,
        "col_inertia": col_inertia,
        "total": np.array([inertia.sum()]),
    }


def _correspondence(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    row_name, col_name = roles["row"], roles["col"]
    if row_name == col_name:
        raise InvalidInputError("Elija dos columnas distintas.")
    if roles.get("count"):
        weights = pd.to_numeric(frame[roles["count"]], errors="coerce")
        data = pd.DataFrame({"r": frame[row_name], "c": frame[col_name], "w": weights}).dropna()
        table = data.pivot_table(index="r", columns="c", values="w", aggfunc="sum", fill_value=0.0)
    else:
        data = frame[[row_name, col_name]].dropna()
        table = pd.crosstab(data[row_name], data[col_name])
    if table.shape[0] < 3 or table.shape[1] < 3:
        raise InvalidInputError("El análisis de correspondencias necesita al menos 3 categorías en cada variable.")
    if table.shape[0] > 60 or table.shape[1] > 60:
        raise InvalidInputError("Se admiten hasta 60 categorías por variable.")
    counts = table.to_numpy(float)
    keep_rows, keep_cols = counts.sum(axis=1) > 0, counts.sum(axis=0) > 0
    table, counts = table.loc[keep_rows, keep_cols], counts[np.ix_(keep_rows, keep_cols)]
    n = counts.sum()
    result = correspondence(counts)
    inertia, dims = result["inertia"], len(result["inertia"])
    chi2 = float(n * inertia.sum())
    df = (counts.shape[0] - 1) * (counts.shape[1] - 1)
    out = Output(
        f"Análisis de correspondencias: {row_name} y {col_name}",
        f"{counts.shape[0]} × {counts.shape[1]} categorías · n = {int(n)}",
    )
    out.tiles(
        [
            ("Inercia total", float(inertia.sum())),
            ("χ² de independencia", chi2),
            ("Valor p", P(float(stats.chi2.sf(chi2, df)))),
            ("Dimensiones", dims),
        ]
    )
    out.table(
        "Inercia por dimensión",
        ["Dimensión", "Valor singular", "Inercia", "% de la inercia", "% acumulado", "χ²"],
        [
            [
                i + 1,
                float(np.sqrt(inertia[i])),
                float(inertia[i]),
                100 * float(inertia[i] / inertia.sum()),
                100 * float(inertia[: i + 1].sum() / inertia.sum()),
                float(n * inertia[i]),
            ]
            for i in range(dims)
        ],
        note="Inercia total = χ² / n: cuánto se apartan las filas del perfil promedio.",
    )
    shown = min(dims, 2)
    rows_coord, cols_coord = result["rows"][:, :shown], result["cols"][:, :shown]
    q_rows = 100 * (result["rows"][:, :shown] ** 2).sum(axis=1) / np.maximum((result["rows"] ** 2).sum(axis=1), 1e-12)
    q_cols = 100 * (result["cols"][:, :shown] ** 2).sum(axis=1) / np.maximum((result["cols"] ** 2).sum(axis=1), 1e-12)
    head = ["Categoría", "Masa", "Inercia", *[f"Dim {i + 1}" for i in range(shown)], "Calidad (%)"]
    out.table(
        f"Coordenadas de las filas ({row_name})",
        head,
        [
            [
                str(table.index[i]),
                float(result["row_mass"][i]),
                float(result["row_inertia"][i]),
                *(float(rows_coord[i, j]) for j in range(shown)),
                float(q_rows[i]),
            ]
            for i in range(len(table.index))
        ],
        note="Calidad: porcentaje de la distancia al centro que se ve en el mapa.",
    )
    out.table(
        f"Coordenadas de las columnas ({col_name})",
        head,
        [
            [
                str(table.columns[j]),
                float(result["col_mass"][j]),
                float(result["col_inertia"][j]),
                *(float(cols_coord[j, d]) for d in range(shown)),
                float(q_cols[j]),
            ]
            for j in range(len(table.columns))
        ],
    )
    if shown >= 2:
        share = 100 * inertia / inertia.sum()
        out.chart(
            scatter_chart(
                f"Mapa simétrico de correspondencias (dim 1: {share[0]:.1f} %, dim 2: {share[1]:.1f} %)",
                f"Dimensión 1 ({share[0]:.1f} %)",
                f"Dimensión 2 ({share[1]:.1f} %)",
                [
                    {
                        "name": row_name,
                        "x": rows_coord[:, 0],
                        "y": rows_coord[:, 1],
                        "text": [str(v) for v in table.index],
                        "labels": True,
                        "size": 10,
                    },
                    {
                        "name": col_name,
                        "x": cols_coord[:, 0],
                        "y": cols_coord[:, 1],
                        "text": [str(v) for v in table.columns],
                        "labels": True,
                        "size": 10,
                    },
                ],
                hlines=[{"value": 0.0}],
                vlines=[{"value": 0.0}],
                height=480,
            )
        )
    out.table(
        "Tabla de contingencia",
        [row_name, *[str(c) for c in table.columns]],
        [
            [str(table.index[i]), *(float(v) if roles.get("count") else int(v) for v in counts[i])]
            for i in range(len(table.index))
        ],
    )
    return out.set(
        inertia=[float(v) for v in inertia],
        total_inertia=float(inertia.sum()),
        chi2=chi2,
        row_coordinates=rows_coord.tolist(),
        column_coordinates=cols_coord.tolist(),
    ).build()


def _correspondence_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import numpy as np\nimport pandas as pd\n\n"
        + load_line(dataset)
        + f"N = pd.crosstab(df[{roles['row']!r}], df[{roles['col']!r}]).to_numpy(float)\nP = N / N.sum()\nr, c = P.sum(1), P.sum(0)\nu, s, vt = np.linalg.svd((P - np.outer(r, c)) / np.sqrt(np.outer(r, c)), full_matrices=False)\nprint(s ** 2)  # inercias; coordenadas: (u * s) / np.sqrt(r)[:, None]\n"
    )


# ============================================================================ multidimensional scaling
def classical_mds(distances: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Torgerson's classical scaling: (coordinates, all eigenvalues)."""
    n = len(distances)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (distances**2) @ j
    values, vectors = np.linalg.eigh(b)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    return vectors[:, :dims] * np.sqrt(np.maximum(values[:dims], 0)), values


def _mds(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    from scipy.spatial.distance import pdist, squareform

    names, dims = roles["vars"], int(p["dimensions"])
    data = complete(frame, names, 5)
    n = len(data)
    if n > 1500:
        raise InvalidInputError(
            "El escalamiento multidimensional admite hasta 1 500 filas (la matriz de distancias crece con n²)."
        )
    x = data.to_numpy(float)
    if p["standardize"]:
        x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    distances = squareform(pdist(x, metric=p["metric"]))
    out = Output(
        "Escalamiento multidimensional",
        f"{'clásico (métrico)' if p['method'] == 'classical' else 'no métrico (SMACOF)'} · distancia {p['metric']} · {dims} dimensión(es) · n = {n}",
    )
    dropped_alert(out, frame, data)
    if p["method"] == "classical":
        coordinates, eigenvalues = classical_mds(distances, dims)
        positive = eigenvalues[eigenvalues > 0]
        share = 100 * eigenvalues[:dims] / positive.sum()
        out.tiles([("Observaciones", n), ("Varianza reproducida", float(share.sum())), ("Dimensiones", dims)])
        out.table(
            "Valores propios",
            ["Dimensión", "Valor propio", "% de la varianza"],
            [
                [i + 1, float(eigenvalues[i]), float(100 * max(eigenvalues[i], 0) / positive.sum())]
                for i in range(min(8, n))
            ],
            note="Si hay valores propios negativos grandes, la matriz de distancias no es euclídea.",
        )
        fitted = squareform(pdist(coordinates))
        stress = float(np.sqrt(((fitted - distances) ** 2).sum() / (distances**2).sum()))
    else:
        model = MDS(
            n_components=dims,
            metric=False,
            dissimilarity="precomputed",
            random_state=int(p["seed"]),
            n_init=4,
            max_iter=300,
            normalized_stress="auto",
        )
        coordinates = model.fit_transform(distances)
        stress = float(model.stress_)
        out.tiles([("Observaciones", n), ("Estrés-1 de Kruskal", stress), ("Dimensiones", dims)])
        quality = (
            "excelente"
            if stress < 0.025
            else "buena"
            if stress < 0.05
            else "aceptable"
            if stress < 0.1
            else "pobre"
            if stress < 0.2
            else "mala"
        )
        out.alert(
            "success" if stress < 0.1 else "warning",
            f"Ajuste {quality} (estrés = {stress:.3f}): menor es mejor; por debajo de 0.1 el mapa es una buena representación.",
        )
    group = roles.get("group")
    labels = frame.loc[data.index, roles["label"]].astype(str).to_numpy() if roles.get("label") else None
    if group:
        categories = frame.loc[data.index, group].astype(str).to_numpy()
        series = [
            {
                "name": level,
                "x": coordinates[categories == level, 0],
                "y": coordinates[categories == level, 1],
                **({"text": labels[categories == level].tolist(), "labels": n <= 60} if labels is not None else {}),
            }
            for level in sorted(set(categories))
        ]
    else:
        series = [
            {
                "name": "Observaciones",
                "x": coordinates[:, 0],
                "y": coordinates[:, 1],
                **({"text": labels.tolist(), "labels": n <= 60} if labels is not None else {}),
            }
        ]
    if dims >= 2:
        out.chart(
            scatter_chart(
                "Configuración en las dos primeras dimensiones",
                "Dimensión 1",
                "Dimensión 2",
                series,
                hlines=[{"value": 0.0}],
                vlines=[{"value": 0.0}],
                height=440,
            )
        )
    out.table(
        "Ajuste",
        ["Medida", "Valor"],
        [["Estrés (diferencia relativa entre distancias reproducidas y originales)", stress]],
    )
    out.dataset(
        "coordenadas",
        "Coordenadas del mapa",
        _result_frame(frame, data.index, **{f"dim_{i + 1}": coordinates[:, i] for i in range(dims)}),
    )
    return out.set(stress=stress, n=n).build()


def _mds_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from scipy.spatial.distance import pdist, squareform\nfrom sklearn.manifold import MDS\n\n"
        + load_line(dataset)
        + f"X = df[{roles['vars']!r}].dropna()\nD = squareform(pdist(X, metric={p['metric']!r}))\nprint(MDS(n_components={int(p['dimensions'])}, dissimilarity='precomputed', metric={p['method'] == 'classical'}).fit_transform(D)[:5])\n"
    )


# ============================================================================ the catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


SPECS = (
    _spec("pca", "Componentes principales (PCA)", GROUP_REDUCE, "Resume muchas variables en pocos componentes: valores propios, cargas, KMO y Bartlett, análisis paralelo, gráficos y puntuaciones.", _pca, roles=(VARS, COLOR), params=(FieldSpec("standardize", "Estandarizar las variables (usar la matriz de correlaciones)", "boolean", True), FieldSpec("rule", "Cuántos componentes retener", "select", "kaiser", _PCA_RULES), FieldSpec("components", "Número de componentes", "integer", 2, minimum=1, maximum=50, visible_if=("rule", ("fixed",))), FieldSpec("variance", "Varianza acumulada mínima (%)", "number", 80, minimum=10, maximum=99, visible_if=("rule", ("variance",))), FieldSpec("rotation", "Rotación de las cargas", "select", "none", (("none", "Sin rotar"), ("varimax", "Varimax"))), ALPHA), outputs=(OutputSpec("puntuaciones", "Puntuaciones de los componentes"),), code=_pca_code, example={"roles": {"vars": ["y", "x1", "x2", "x3"], "group": "g"}, "params": {"rule": "parallel"}}),
    _spec("factor", "Análisis factorial", GROUP_REDUCE, "Factores latentes detrás de las variables: ejes principales o máxima verosimilitud, rotaciones ortogonales y oblicuas, bondad de ajuste y puntuaciones.", _factor, roles=(VARS,), params=(FieldSpec("factors", "Número de factores", "integer", 2, minimum=1, maximum=10), FieldSpec("method", "Método de extracción", "select", "pa", (("pa", "Ejes principales"), ("ml", "Máxima verosimilitud"))), FieldSpec("rotation", "Rotación", "select", "varimax", _ROTATIONS), FieldSpec("cutoff", "Resaltar cargas de al menos", "number", 0.4, minimum=0.1, maximum=0.9), FieldSpec("scores", "Puntuaciones factoriales", "select", "regression", (("regression", "Regresión (Thomson)"), ("bartlett", "Bartlett"), ("none", "No calcular"))), ALPHA), outputs=(OutputSpec("puntuaciones", "Puntuaciones factoriales"),), code=_factor_code, example={"roles": {"vars": ["r1", "r2", "r3", "c1", "c2", "c3"]}, "params": {"factors": 2}}),
    _spec("reliability", "Fiabilidad de una escala (alfa de Cronbach)", GROUP_SCALES, "Alfa, omega, análisis de ítems, dos mitades y qué ítems conviene quitar o invertir.", _reliability, roles=(RoleSpec("vars", "Ítems de la escala (3 o más)", "numeric", multiple=True, min_count=3), RoleSpec("reverse", "Ítems redactados al revés (opcional)", "numeric", multiple=True, optional=True)), code=_reliability_code, example={"roles": {"vars": ["r1", "r2", "r3"], "reverse": []}, "params": {}}),
    _spec("cca", "Correlación canónica", GROUP_SCALES, "Relación entre dos conjuntos de variables: pares canónicos, significancia (Wilks), cargas y redundancia.", _cca, roles=(RoleSpec("x_vars", "Conjunto X", "numeric", multiple=True), RoleSpec("y_vars", "Conjunto Y", "numeric", multiple=True)), params=(ALPHA,), code=_cca_code, example={"roles": {"x_vars": ["x1", "x2"], "y_vars": ["y", "x3"]}, "params": {}}),
    _spec("correspondence", "Análisis de correspondencias", GROUP_MAPS, "Mapa de la asociación entre dos variables categóricas: inercia, coordenadas, calidad y contribuciones.", _correspondence, roles=(RoleSpec("row", "Variable de las filas", "categorical"), RoleSpec("col", "Variable de las columnas", "categorical"), RoleSpec("count", "Frecuencias (opcional; vacío = contar filas)", "numeric", optional=True)), code=_correspondence_code, example={"roles": {"row": "g", "col": "cl"}, "params": {}}),
    _spec("mds", "Escalamiento multidimensional", GROUP_MAPS, "Mapa de las observaciones que conserva sus distancias: método clásico (Torgerson) o no métrico (SMACOF) con estrés de Kruskal.", _mds, roles=(VARS, COLOR, LABEL), params=(FieldSpec("method", "Método", "select", "classical", (("classical", "Clásico (métrico)"), ("nonmetric", "No métrico (SMACOF)"))), FieldSpec("metric", "Distancia", "select", "euclidean", (("euclidean", "Euclídea"), ("cityblock", "Manhattan"), ("cosine", "Coseno"), ("correlation", "Correlación"))), FieldSpec("dimensions", "Dimensiones", "integer", 2, minimum=2, maximum=5), FieldSpec("standardize", "Estandarizar las variables", "boolean", True), FieldSpec("seed", "Semilla (método no métrico)", "integer", 123, minimum=0, maximum=2**31 - 1)), outputs=(OutputSpec("coordenadas", "Coordenadas del mapa"),), code=_mds_code, example={"roles": {"vars": ["x1", "x2", "x3", "y"], "group": "g"}, "params": {"method": "classical"}}),
    _spec("mahalanobis", "Distancia de Mahalanobis y atípicos multivariados", GROUP_MAPS, "Observaciones que son raras en conjunto aunque cada variable parezca normal; métodos clásico y robusto (MCD).", _mahalanobis, roles=(VARS,), params=(FieldSpec("method", "Estimación", "select", "robust", (("classical", "Media y covarianza clásicas"), ("robust", "Robusta (determinante mínimo de covarianza)"))), FieldSpec("seed", "Semilla (robusta)", "integer", 123, minimum=0, maximum=2**31 - 1), ALPHA), outputs=(OutputSpec("distancias", "Distancia de Mahalanobis"),), code=_mahalanobis_code, example={"roles": {"vars": ["y", "x1", "x2"]}, "params": {}}),
)  # fmt: skip

registry.register(*SPECS)
