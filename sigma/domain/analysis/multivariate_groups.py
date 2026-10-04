"""Groups in multivariate data: hierarchical and k-means clustering, Gaussian mixtures, linear and quadratic discriminant analysis."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import cophenet, cut_tree, dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sigma.domain.analysis import registry
from sigma.domain.analysis.anova_advanced import box_m
from sigma.domain.analysis.charts import box_chart, line_chart, raw_chart, scatter_chart
from sigma.domain.analysis.multivariate import MAX_POINTS, VARS, _fix_signs, _result_frame, complete, dropped_alert
from sigma.domain.analysis.output import Output, P, good, strong
from sigma.domain.analysis.plots import level_order
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.analysis.support import load_line
from sigma.domain.errors import InvalidInputError

MODULE = "multivariado"
GROUP_CLUSTER = "Agrupamiento (clúster)"
GROUP_CLASSIFY = "Clasificación"
SEED = FieldSpec("seed", "Semilla", "integer", 123, minimum=0, maximum=2**31 - 1)
STANDARDIZE = FieldSpec(
    "standardize",
    "Estandarizar las variables",
    "boolean",
    True,
    help="Imprescindible cuando las variables tienen escalas distintas.",
)


def _prepare(
    frame: pd.DataFrame, names: list[str], standardize: bool, minimum: int = 10
) -> tuple[pd.DataFrame, np.ndarray]:
    data = complete(frame, names, minimum)
    values = data.to_numpy(float)
    if standardize:
        values = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
    return data, values


def _projection(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two first principal components of ``values`` for plotting: (scores, share of variance)."""
    centred = values - values.mean(axis=0)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    components = _fix_signs(vt.T[:, :2])
    return centred @ components, (s[:2] ** 2) / (s**2).sum()


def _profile(
    out: Output, data: pd.DataFrame, labels: np.ndarray, names: list[str], title: str = "Perfil de cada grupo (medias)"
) -> None:
    overall = data.mean()
    sd = data.std(ddof=1)
    groups = sorted(set(labels))
    rows = []
    for g in groups:
        part = data[labels == g]
        rows.append([f"Grupo {g + 1}", len(part), 100 * len(part) / len(data), *(float(part[c].mean()) for c in names)])
    out.table(title, ["Grupo", "n", "%", *names], rows, note="Medias en las unidades originales de cada variable.")
    z = [
        [f"Grupo {g + 1}", *(float((data[labels == g][c].mean() - overall[c]) / sd[c]) for c in names)] for g in groups
    ]
    out.table(
        "Diferencia de cada grupo respecto al promedio general (en desviaciones estándar)",
        ["Grupo", *names],
        [[row[0], *(strong(v) if abs(v) >= 0.5 else v for v in row[1:])] for row in z],
        note="En negrita: |diferencia| ≥ 0.5 desviaciones estándar, lo que define el carácter del grupo.",
    )


def _cluster_chart(
    out: Output,
    values: np.ndarray,
    labels: np.ndarray,
    title: str = "Grupos sobre los dos primeros componentes principales",
) -> None:
    if values.shape[1] < 2:
        return
    scores, share = _projection(values)
    shown = np.arange(len(labels))
    if len(shown) > MAX_POINTS:
        shown = np.sort(np.random.default_rng(0).choice(len(shown), MAX_POINTS, replace=False))
    series = [
        {"name": f"Grupo {g + 1}", "x": scores[shown][labels[shown] == g, 0], "y": scores[shown][labels[shown] == g, 1]}
        for g in sorted(set(labels))
    ]
    note = f" (muestra de {MAX_POINTS})" if len(shown) < len(labels) else ""
    out.chart(
        scatter_chart(
            f"{title} ({100 * share.sum():.0f} % de la varianza){note}",
            "Componente 1",
            "Componente 2",
            series,
            height=420,
        )
    )


def _silhouette(values: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    """Mean silhouette; large data sets are scored on a sample of 5 000 rows (the exact score is O(n²))."""
    n = len(values)
    return float(silhouette_score(values, labels, sample_size=5000 if n > 5000 else None, random_state=seed))


# ============================================================================ hierarchical clustering
_LINKAGES = (
    ("ward", "Ward (mínima varianza)"),
    ("complete", "Enlace completo"),
    ("average", "Enlace promedio (UPGMA)"),
    ("single", "Enlace simple"),
    ("centroid", "Centroide"),
)
_DISTANCES = (
    ("euclidean", "Euclídea"),
    ("cityblock", "Manhattan"),
    ("correlation", "1 − correlación"),
    ("cosine", "Coseno"),
)


def _dendrogram_chart(Z: np.ndarray, labels: list[str], leaves: int, cut_height: float | None) -> dict[str, Any]:
    n = len(labels)
    truncated = n > leaves
    tree = dendrogram(
        Z,
        no_plot=True,
        truncate_mode="lastp" if truncated else None,
        p=leaves if truncated else n,
        labels=labels if not truncated else None,
    )
    xs, ys = [], []
    for icoord, dcoord in zip(tree["icoord"], tree["dcoord"], strict=True):
        xs += [*icoord, None]
        ys += [*dcoord, None]
    positions = [5 + 10 * i for i in range(len(tree["ivl"]))]
    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "line": {"color": "#4f46e5", "width": 1.5},
            "hoverinfo": "skip",
            "showlegend": False,
        }
    ]
    layout: dict[str, Any] = {
        "xaxis": {
            "tickmode": "array",
            "tickvals": positions,
            "ticktext": [str(t) for t in tree["ivl"]],
            "tickangle": -60 if len(positions) > 12 else 0,
            "title": {"text": "Observaciones (o grupos agrupados)" if truncated else "Observaciones"},
        },
        "yaxis": {"title": {"text": "Distancia de enlace"}},
        "showlegend": False,
    }
    if cut_height is not None:
        layout["shapes"] = [
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "yref": "y",
                "y0": cut_height,
                "y1": cut_height,
                "line": {"dash": "dash", "color": "#ef4444"},
            }
        ]
    return raw_chart(traces, layout, "Dendrograma", height=460)


def _hierarchical(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names, method, metric = roles["vars"], p["linkage"], p["distance"]
    data, values = _prepare(frame, names, p["standardize"])
    n = len(data)
    if n > 3000:
        raise InvalidInputError(
            "El agrupamiento jerárquico admite hasta 3 000 filas (la matriz de distancias crece con n²)."
        )
    if method in ("ward", "centroid") and metric != "euclidean":
        raise InvalidInputError(f"El método «{dict(_LINKAGES)[method]}» solo funciona con distancia euclídea.")
    distances = pdist(values, metric=metric)
    Z = linkage(distances, method=method)
    coph = float(cophenet(Z, distances)[0])
    k_max = min(10, n - 1)
    square = squareform(distances)
    silhouettes = {}
    for k in range(2, k_max + 1):
        assignment = cut_tree(Z, n_clusters=k).ravel()
        if len(set(assignment)) > 1:
            silhouettes[k] = float(silhouette_score(square, assignment, metric="precomputed"))
    suggested = max(silhouettes, key=silhouettes.get) if silhouettes else 2
    k = int(p["clusters"]) if p["clusters"] else suggested
    labels = cut_tree(Z, n_clusters=k).ravel()
    heights = Z[:, 2]
    cut_height = float((heights[n - k - 1] + heights[n - k]) / 2) if k < n else None
    out = Output(
        "Agrupamiento jerárquico",
        f"{dict(_LINKAGES)[method].split(' (')[0].lower()} · distancia {dict(_DISTANCES)[metric].lower()} · {k} grupos · n = {n}",
    )
    dropped_alert(out, frame, data)
    out.tiles(
        [
            ("Grupos", k),
            ("Correlación cofenética", coph),
            ("Silueta media", silhouettes.get(k, float("nan"))),
            ("Grupos sugeridos (silueta)", suggested),
        ]
    )
    out.alert(
        "info",
        f"La correlación cofenética ({coph:.3f}) mide cuánto respeta el árbol las distancias originales: cerca de 1 es muy fiel; menos de 0.7 sugiere probar otro enlace.",
    )
    label_source = (
        frame.loc[data.index, roles["label"]].astype(str).tolist()
        if roles.get("label")
        else [str(i + 1) for i in data.index]
    )
    out.chart(_dendrogram_chart(Z, label_source, 40, cut_height))
    out.table(
        "Calidad según el número de grupos",
        ["Grupos", "Silueta media", "Altura del corte"],
        [
            [kk, s, float((heights[n - kk - 1] + heights[n - kk]) / 2) if kk < n else None]
            for kk, s in silhouettes.items()
        ],
        note="Silueta: de −1 a 1; más alto es mejor separación. Se sugiere el número que la maximiza.",
    )
    _profile(out, data, labels, names)
    _cluster_chart(out, values, labels)
    out.dataset("grupos", "Grupo de cada fila", _result_frame(frame, data.index, grupo=labels + 1.0))
    return out.set(
        clusters=k,
        cophenetic=coph,
        silhouette=silhouettes.get(k),
        suggested=suggested,
        sizes=[int((labels == g).sum()) for g in range(k)],
    ).build()


def _hierarchical_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from scipy.cluster.hierarchy import fcluster, linkage\nfrom scipy.stats import zscore\n\n"
        + load_line(dataset)
        + f"X = df[{roles['vars']!r}].dropna()\n"
        + ("X = X.apply(zscore)\n" if p["standardize"] else "")
        + f"Z = linkage(X, method={p['linkage']!r}, metric={p['distance']!r})\nprint(fcluster(Z, t={int(p['clusters'] or 3)}, criterion='maxclust'))\n"
    )


# ============================================================================ k-means
def _kmeans(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names = roles["vars"]
    data, values = _prepare(frame, names, p["standardize"])
    n = len(data)
    k_max = int(min(p["k_max"], n - 1))
    if k_max < 2:
        raise InvalidInputError("Hay muy pocas filas para agrupar.")
    seed, restarts = int(p["seed"]), int(p["restarts"])
    scan = {}
    for k in range(2, k_max + 1):
        model = KMeans(n_clusters=k, n_init=restarts, random_state=seed).fit(values)
        scan[k] = {
            "inertia": float(model.inertia_),
            "silhouette": _silhouette(values, model.labels_, seed),
            "ch": float(calinski_harabasz_score(values, model.labels_)),
            "db": float(davies_bouldin_score(values, model.labels_)),
        }
    suggested = max(scan, key=lambda kk: scan[kk]["silhouette"])
    k = int(p["clusters"]) if p["clusters"] else suggested
    if k >= n:
        raise InvalidInputError("Hay más grupos pedidos que filas.")
    final = KMeans(n_clusters=k, n_init=restarts, random_state=seed).fit(values)
    labels = final.labels_
    if k not in scan:
        scan[k] = {
            "inertia": float(final.inertia_),
            "silhouette": _silhouette(values, labels, seed),
            "ch": float(calinski_harabasz_score(values, labels)),
            "db": float(davies_bouldin_score(values, labels)),
        }
    out = Output("Agrupamiento por k-medias", f"{k} grupos · {restarts} inicios · n = {n}")
    dropped_alert(out, frame, data)
    out.tiles(
        [
            ("Grupos", k),
            ("Silueta media", scan[k]["silhouette"]),
            ("Inercia (suma de cuadrados dentro)", float(final.inertia_)),
            ("Grupos sugeridos (silueta)", suggested),
        ]
    )
    out.table(
        "Elección del número de grupos",
        ["Grupos", "Inercia", "Silueta", "Calinski-Harabasz", "Davies-Bouldin", "Elegido"],
        [
            [
                kk,
                scan[kk]["inertia"],
                scan[kk]["silhouette"],
                scan[kk]["ch"],
                scan[kk]["db"],
                good("●") if kk == k else "",
            ]
            for kk in sorted(scan)
        ],
        note="Silueta y Calinski-Harabasz: más alto es mejor. Davies-Bouldin e inercia: más bajo es mejor (la inercia siempre baja con k: busque el «codo»).",
    )
    ks = sorted(scan)
    out.chart(
        line_chart(
            "Codo: inercia según el número de grupos",
            "Grupos (k)",
            "Inercia",
            [{"name": "Inercia", "x": ks, "y": [scan[kk]["inertia"] for kk in ks], "markers": True}],
            vlines=[{"value": k, "label": "elegido"}],
            height=320,
        )
    )
    out.chart(
        line_chart(
            "Silueta media según el número de grupos",
            "Grupos (k)",
            "Silueta",
            [{"name": "Silueta", "x": ks, "y": [scan[kk]["silhouette"] for kk in ks], "markers": True}],
            vlines=[{"value": k, "label": "elegido"}],
            height=320,
        )
    )
    _profile(out, data, labels, names)
    _cluster_chart(out, values, labels)
    distance = np.linalg.norm(values - final.cluster_centers_[labels], axis=1)
    out.dataset(
        "grupos",
        "Grupo de cada fila",
        _result_frame(frame, data.index, grupo=labels + 1.0, distancia_al_centro=distance),
    )
    return out.set(
        clusters=k,
        silhouette=scan[k]["silhouette"],
        inertia=float(final.inertia_),
        suggested=suggested,
        sizes=[int((labels == g).sum()) for g in range(k)],
        centers=final.cluster_centers_.tolist(),
    ).build()


def _kmeans_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from sklearn.cluster import KMeans\nfrom sklearn.preprocessing import StandardScaler\n\n"
        + load_line(dataset)
        + f"X = df[{roles['vars']!r}].dropna()\n"
        + ("X = StandardScaler().fit_transform(X)\n" if p["standardize"] else "")
        + f"model = KMeans(n_clusters={int(p['clusters'] or 3)}, n_init={int(p['restarts'])}, random_state={int(p['seed'])}).fit(X)\nprint(model.labels_, model.inertia_)\n"
    )


# ============================================================================ Gaussian mixtures
_COVARIANCES = (
    ("full", "Completa (cada grupo con su matriz)"),
    ("tied", "Común a todos los grupos"),
    ("diag", "Diagonal"),
    ("spherical", "Esférica"),
)


def _mixture(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names = roles["vars"]
    data, values = _prepare(frame, names, p["standardize"], 30)
    n, d = values.shape
    kind, seed = p["covariance"], int(p["seed"])
    limit = n // (d + 2)  # fewer rows per component than this and the covariances are not estimable
    wanted = int(p["components"]) if p["components"] else None
    if wanted and wanted > limit:
        raise InvalidInputError(f"Con {n} filas y {d} variables se pueden ajustar como máximo {limit} componentes.")
    k_max = min(max(int(p["k_max"]), wanted or 1), limit)
    if k_max < 1:
        raise InvalidInputError("Hay muy pocas filas para ajustar una mezcla con esas variables.")
    fits = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(1, k_max + 1):
            fits[k] = GaussianMixture(
                n_components=k, covariance_type=kind, n_init=int(p["restarts"]), random_state=seed, max_iter=300
            ).fit(values)
    chosen = wanted or min(fits, key=lambda kk: fits[kk].bic(values))
    model = fits[chosen]
    probabilities = model.predict_proba(values)
    labels = probabilities.argmax(axis=1)
    certainty = probabilities.max(axis=1)
    entropy = float(-(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum() / n)
    # means back in the original units of each variable
    scale = data.std(ddof=1).to_numpy() if p["standardize"] else np.ones(d)
    centre = data.mean().to_numpy() if p["standardize"] else np.zeros(d)
    out = Output(
        "Mezcla de gaussianas (agrupamiento probabilístico)",
        f"{chosen} componente(s) · covarianza {dict(_COVARIANCES)[kind].split(' (')[0].lower()} · n = {n}",
    )
    dropped_alert(out, frame, data)
    out.tiles(
        [
            ("Componentes", chosen),
            ("BIC", float(model.bic(values))),
            ("Certeza media de la asignación", float(certainty.mean())),
            ("Convergió", "sí" if model.converged_ else "no"),
        ]
    )
    out.table(
        "Selección del número de componentes",
        ["Componentes", "BIC", "AIC", "Log-verosimilitud", "Elegido"],
        [
            [
                kk,
                float(fits[kk].bic(values)),
                float(fits[kk].aic(values)),
                float(fits[kk].score(values) * n),
                good("●") if kk == chosen else "",
            ]
            for kk in sorted(fits)
        ],
        note="BIC más bajo = mejor equilibrio entre ajuste y complejidad.",
    )
    ks = sorted(fits)
    out.chart(
        line_chart(
            "BIC y AIC según el número de componentes",
            "Componentes",
            "Criterio",
            [
                {"name": "BIC", "x": ks, "y": [float(fits[kk].bic(values)) for kk in ks], "markers": True},
                {
                    "name": "AIC",
                    "x": ks,
                    "y": [float(fits[kk].aic(values)) for kk in ks],
                    "markers": True,
                    "dash": "dash",
                },
            ],
            vlines=[{"value": chosen, "label": "elegido"}],
            height=320,
        )
    )
    out.table(
        "Pesos y medias de los componentes",
        ["Componente", "Peso", *names],
        [
            [f"Grupo {g + 1}", float(model.weights_[g]), *(float(v) for v in centre + model.means_[g] * scale)]
            for g in range(chosen)
        ],
    )
    _profile(out, data, labels, names, "Perfil de los grupos asignados (medias)")
    _cluster_chart(out, values, labels)
    columns = {
        "grupo": labels + 1.0,
        "certeza": certainty,
        **{f"prob_grupo_{g + 1}": probabilities[:, g] for g in range(chosen)},
    }
    out.dataset("grupos", "Grupo y probabilidad de cada fila", _result_frame(frame, data.index, **columns))
    return out.set(
        components=chosen,
        bic=float(model.bic(values)),
        certainty=float(certainty.mean()),
        entropy=entropy,
        weights=model.weights_.tolist(),
    ).build()


def _mixture_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from sklearn.mixture import GaussianMixture\nfrom sklearn.preprocessing import StandardScaler\n\n"
        + load_line(dataset)
        + f"X = df[{roles['vars']!r}].dropna()\n"
        + ("X = StandardScaler().fit_transform(X)\n" if p["standardize"] else "")
        + f"for k in range(1, {int(p['k_max'])} + 1):\n    print(k, GaussianMixture(k, covariance_type={p['covariance']!r}, random_state={int(p['seed'])}).fit(X).bic(X))\n"
    )


# ============================================================================ discriminant analysis
def _eigen_ratios(x: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigenvalues of W⁻¹B, the matching directions (canonical discriminant functions) and the pooled within covariance."""
    overall = x.mean(axis=0)
    within = np.zeros((x.shape[1], x.shape[1]))
    between = np.zeros_like(within)
    for g in np.unique(groups):
        part = x[groups == g]
        centre = part.mean(axis=0)
        within += (part - centre).T @ (part - centre)
        between += len(part) * np.outer(centre - overall, centre - overall)
    values, vectors = np.linalg.eig(np.linalg.solve(within, between))
    order = np.argsort(values.real)[::-1]
    return values.real[order], vectors.real[:, order], within / (len(x) - len(np.unique(groups)))


def _confusion(truth: np.ndarray, guess: np.ndarray, labels: list[str]) -> np.ndarray:
    return np.array([[int(((truth == a) & (guess == b)).sum()) for b in labels] for a in labels])


def _discriminant(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names, group_name, kind = roles["vars"], roles["group"], p["kind"]
    data = frame[[group_name, *names]].copy()
    data[names] = data[names].apply(pd.to_numeric, errors="coerce")
    data = data.dropna()
    classes = level_order(data[group_name])
    g, n, d = len(classes), len(data), len(names)
    if g < 2:
        raise InvalidInputError("Se necesitan al menos 2 grupos.")
    counts = data[group_name].value_counts()
    if counts.min() < 3:
        raise InvalidInputError("Cada grupo necesita al menos 3 observaciones.")
    reg = float(p["regularization"] or 0.0)
    if kind == "qda" and counts.min() <= d and not reg:
        raise InvalidInputError(
            f"El discriminante cuadrático necesita más de {d} observaciones en cada grupo (el menor tiene {int(counts.min())}); use regularización o el lineal."
        )
    x = data[names].to_numpy(float)
    if (x.std(axis=0, ddof=1) == 0).any():
        raise InvalidInputError("Alguna variable es constante.")
    y = data[group_name].astype(str).to_numpy()
    labels_text = [str(c) for c in classes]
    priors = None if p["priors"] == "proportional" else np.full(g, 1 / g)
    model = (
        LinearDiscriminantAnalysis(solver="svd", priors=priors)
        if kind == "lda"
        else QuadraticDiscriminantAnalysis(priors=priors, reg_param=reg)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, y)
        predicted = model.predict(x)
        folds = int(min(p["folds"], counts.min()))
        cross = cross_val_predict(
            clone(model), x, y, cv=StratifiedKFold(folds, shuffle=True, random_state=int(p["seed"]))
        )
    resub = float((predicted == y).mean())
    cv_accuracy = float((cross == y).mean())
    baseline = float(counts.max() / n)
    out = Output(
        f"Análisis discriminante {'lineal' if kind == 'lda' else 'cuadrático'}: {group_name}",
        f"{g} grupos · {d} variables · n = {n} · probabilidades previas {'iguales' if priors is not None else 'proporcionales'}",
    )
    if len(frame) > n:
        out.alert("info", f"Se omitieron {len(frame) - n} fila(s) con datos faltantes.")
    out.tiles(
        [
            ("Exactitud (resustitución)", resub),
            (f"Exactitud (validación cruzada, {folds} pliegues)", cv_accuracy),
            ("Línea base (clase mayoritaria)", baseline),
            ("Observaciones", n),
        ]
    )
    if resub - cv_accuracy > 0.1:
        out.alert(
            "warning",
            f"La exactitud baja de {resub:.1%} a {cv_accuracy:.1%} al validar: el modelo se sobreajusta; reduzca variables o use el lineal.",
        )
    if cv_accuracy <= baseline + 0.02:
        out.alert(
            "warning",
            "La exactitud validada no supera a predecir siempre la clase mayoritaria: las variables discriminan poco.",
        )
    for title, guess in (
        ("Clasificación por validación cruzada (real en filas, predicho en columnas)", cross),
        ("Clasificación con los mismos datos (resustitución)", predicted),
    ):
        matrix = _confusion(y, guess, labels_text)
        out.table(
            title,
            ["Real / predicho", *labels_text, "Total", "% correcto"],
            [
                [
                    labels_text[i],
                    *(int(v) for v in matrix[i]),
                    int(matrix[i].sum()),
                    100 * float(matrix[i, i] / max(matrix[i].sum(), 1)),
                ]
                for i in range(g)
            ],
        )
    matrix = _confusion(y, cross, labels_text)
    out.table(
        "Precisión y exhaustividad por grupo (validación cruzada)",
        ["Grupo", "Precisión", "Exhaustividad", "F1", "n"],
        [
            [
                labels_text[i],
                float(matrix[i, i] / max(matrix[:, i].sum(), 1)),
                float(matrix[i, i] / max(matrix[i].sum(), 1)),
                float(2 * matrix[i, i] / max(matrix[:, i].sum() + matrix[i].sum(), 1)),
                int(matrix[i].sum()),
            ]
            for i in range(g)
        ],
    )
    out.table(
        "Medias por grupo",
        ["Grupo", *names],
        [[labels_text[i], *(float(x[y == labels_text[i], j].mean()) for j in range(d))] for i in range(g)],
    )
    canonical: dict[str, Any] = {}
    if kind == "lda":
        eigenvalues, directions, pooled = _eigen_ratios(x, y)
        functions = min(g - 1, d)
        rows = []
        for i in range(functions):
            lam = float(np.prod(1 / (1 + eigenvalues[i:functions])))
            chi2 = -(n - 1 - (d + g) / 2) * np.log(lam)
            df = (d - i) * (g - 1 - i)
            rows.append(
                [
                    f"Función {i + 1}",
                    float(eigenvalues[i]),
                    100 * float(eigenvalues[i] / eigenvalues[:functions].sum()),
                    float(np.sqrt(eigenvalues[i] / (1 + eigenvalues[i]))),
                    lam,
                    float(chi2),
                    df,
                    P(float(stats.chi2.sf(chi2, df))),
                ]
            )
        out.table(
            "Funciones discriminantes canónicas",
            [
                "Función",
                "Valor propio",
                "% de la separación",
                "Correlación canónica",
                "Lambda de Wilks",
                "χ²",
                "gl",
                "Valor p",
            ],
            rows,
            note="Cada prueba contrasta que esa función y las siguientes no separen los grupos.",
        )
        raw = directions[:, :functions]
        raw = _fix_signs(raw / np.sqrt(np.einsum("ij,ik,kj->j", raw, pooled, raw)))  # unit variance within groups
        out.table(
            "Coeficientes estandarizados de las funciones",
            ["Variable", *[f"Función {i + 1}" for i in range(functions)]],
            [[names[j], *(float(raw[j, i] * np.sqrt(pooled[j, j])) for i in range(functions))] for j in range(d)],
            note="Un coeficiente grande en valor absoluto señala una variable que contribuye mucho a separar los grupos.",
        )
        scores = (x - x.mean(axis=0)) @ raw
        if functions >= 2:
            out.chart(
                scatter_chart(
                    "Grupos sobre las dos primeras funciones discriminantes",
                    "Función 1",
                    "Función 2",
                    [{"name": c, "x": scores[y == c, 0], "y": scores[y == c, 1]} for c in labels_text],
                    height=420,
                )
            )
        else:
            out.chart(
                box_chart(
                    "Puntuación de la función discriminante por grupo",
                    "Función 1",
                    [{"name": c, "values": scores[y == c, 0]} for c in labels_text],
                    points=True,
                )
            )
        canonical = {
            "eigenvalues": [float(v) for v in eigenvalues[:functions]],
            "wilks": float(np.prod(1 / (1 + eigenvalues[:functions]))),
        }
    if all(counts >= d + 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            box = box_m([x[y == c] for c in labels_text])
        if np.isfinite(box[0]):
            out.table(
                "Igualdad de matrices de covarianza (M de Box)",
                ["χ²", "gl", "Valor p", "Lectura"],
                [
                    [
                        box[0],
                        box[2],
                        P(box[1]),
                        "las covarianzas difieren: considere el discriminante cuadrático"
                        if box[1] < 0.001
                        else "supuesto del lineal razonable",
                    ]
                ],
                note="M de Box es muy sensible; solo se descarta el lineal con valores p muy pequeños.",
            )
    posterior = model.predict_proba(x)
    result = _result_frame(frame, data.index, **{f"prob_{c}": posterior[:, i] for i, c in enumerate(model.classes_)})
    result["grupo_predicho"] = pd.Series(predicted, index=data.index).reindex(frame.index)
    out.dataset("clasificacion", "Probabilidades de pertenencia", result)
    return out.set(
        accuracy=resub, cv_accuracy=cv_accuracy, baseline=baseline, groups=g, confusion=matrix.tolist(), **canonical
    ).build()


def _discriminant_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    cls = "LinearDiscriminantAnalysis" if p["kind"] == "lda" else "QuadraticDiscriminantAnalysis"
    return (
        f"from sklearn.discriminant_analysis import {cls}\nfrom sklearn.model_selection import cross_val_score\n\n"
        + load_line(dataset)
        + f"d = df[[{roles['group']!r}, *{roles['vars']!r}]].dropna()\nmodel = {cls}()\nprint(cross_val_score(model, d[{roles['vars']!r}], d[{roles['group']!r}], cv={int(p['folds'])}).mean())\n"
    )


# ============================================================================ the catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


SPECS = (
    _spec("hierarchical", "Clúster jerárquico y dendrograma", GROUP_CLUSTER, "Agrupa observaciones paso a paso (Ward, enlace completo, promedio…) y muestra el árbol, la silueta por número de grupos y el perfil de cada uno.", _hierarchical, roles=(VARS, RoleSpec("label", "Etiqueta de cada fila (opcional)", "any", optional=True)), params=(FieldSpec("linkage", "Método de enlace", "select", "ward", _LINKAGES), FieldSpec("distance", "Distancia", "select", "euclidean", _DISTANCES), FieldSpec("clusters", "Número de grupos (vacío = el de mayor silueta)", "integer", optional=True, minimum=2, maximum=20), STANDARDIZE), outputs=(OutputSpec("grupos", "Grupo de cada fila"),), code=_hierarchical_code, example={"roles": {"vars": ["y", "x1", "x2"]}, "params": {"clusters": 3}}),
    _spec("kmeans", "K-medias", GROUP_CLUSTER, "Agrupamiento por centroides con búsqueda del mejor k (codo, silueta, Calinski-Harabasz, Davies-Bouldin) y perfil de cada grupo.", _kmeans, heavy=True, roles=(VARS,), params=(FieldSpec("clusters", "Número de grupos (vacío = el de mayor silueta)", "integer", optional=True, minimum=2, maximum=30), FieldSpec("k_max", "Probar hasta k =", "integer", 8, minimum=3, maximum=20), FieldSpec("restarts", "Inicios aleatorios", "integer", 10, minimum=1, maximum=50), STANDARDIZE, SEED), outputs=(OutputSpec("grupos", "Grupo de cada fila"),), code=_kmeans_code, example={"roles": {"vars": ["y", "x1", "x2"]}, "params": {"k_max": 5}}),
    _spec("mixture", "Mezcla de gaussianas (clúster probabilístico)", GROUP_CLUSTER, "Cada fila recibe una probabilidad de pertenecer a cada grupo; el número de grupos se elige con BIC.", _mixture, heavy=True, roles=(VARS,), params=(FieldSpec("components", "Número de componentes (vacío = el de menor BIC)", "integer", optional=True, minimum=1, maximum=20), FieldSpec("k_max", "Probar hasta k =", "integer", 6, minimum=2, maximum=15), FieldSpec("covariance", "Forma de la covarianza", "select", "full", _COVARIANCES), FieldSpec("restarts", "Inicios aleatorios", "integer", 3, minimum=1, maximum=20), STANDARDIZE, SEED), outputs=(OutputSpec("grupos", "Grupo y probabilidad de cada fila"),), code=_mixture_code, example={"roles": {"vars": ["y", "x1", "x2"]}, "params": {"k_max": 3, "restarts": 2}}),
    _spec("discriminant", "Análisis discriminante (lineal y cuadrático)", GROUP_CLASSIFY, "Clasifica en grupos conocidos y explica qué variables los separan: validación cruzada, funciones canónicas, Wilks y M de Box.", _discriminant, roles=(RoleSpec("group", "Grupo (la categoría a predecir)", "categorical"), VARS), params=(FieldSpec("kind", "Tipo", "select", "lda", (("lda", "Lineal (covarianza común)"), ("qda", "Cuadrático (covarianza por grupo)"))), FieldSpec("priors", "Probabilidades previas", "select", "proportional", (("proportional", "Proporcionales al tamaño de cada grupo"), ("equal", "Iguales"))), FieldSpec("regularization", "Regularización (cuadrático)", "number", 0.0, minimum=0, maximum=1, visible_if=("kind", ("qda",))), FieldSpec("folds", "Pliegues de validación cruzada", "integer", 5, minimum=2, maximum=20), SEED), outputs=(OutputSpec("clasificacion", "Probabilidades de pertenencia"),), code=_discriminant_code, example={"roles": {"group": "g", "vars": ["y", "x1", "x2"]}, "params": {}}),
)  # fmt: skip

registry.register(*SPECS)
