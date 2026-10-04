"""Supervised learning models: trees, forests, boosting, nearest neighbours, SVM, naive Bayes, neural networks and penalized linear models."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import line_chart, raw_chart
from sigma.domain.analysis.ml_algorithms import ALGORITHMS, parse_layers
from sigma.domain.analysis.ml_core import (
    CLASSIFICATION,
    COMMON_PARAMS,
    FEATURES,
    MODULE,
    PREDICTIONS,
    TARGET,
    code_for,
    encoded_info,
    holdout_split,
    metric_table,
    run_supervised,
    search_values,
)
from sigma.domain.analysis.output import Output, good
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec

GROUP_TREES = "Árboles y conjuntos"
GROUP_OTHER = "Otros modelos"
PALETTE = ("#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#ec4899", "#64748b", "#84cc16")
CLASS_WEIGHT = FieldSpec(
    "class_weight",
    "Peso de las clases",
    "select",
    "none",
    (("none", "Sin ajustar"), ("balanced", "Equilibrado (clases raras pesan más)")),
    help="Solo clasificación. Útil cuando una clase es mucho más rara que las otras.",
)


def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    kw.setdefault("roles", (TARGET, FEATURES))
    kw.setdefault("outputs", (PREDICTIONS,))
    return AnalysisSpec(id_, label, group, MODULE, description, compute, heavy=True, **kw)


def _model_step(context: SimpleNamespace) -> Any:
    """The fitted estimator (inside the target scaler when the response is standardised)."""
    model = context.final.named_steps["model"]
    return getattr(model, "regressor_", model)


def _importance_by_variable(context: SimpleNamespace, values: np.ndarray) -> dict[str, float]:
    """Sum the importance of the dummies of one variable back into that variable."""
    totals: dict[str, float] = {}
    for (raw, _, _), value in zip(encoded_info(context.final, context.problem), values, strict=True):
        totals[raw] = totals.get(raw, 0.0) + float(value)
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def _impurity_table(out: Output, context: SimpleNamespace, values: np.ndarray, title: str) -> dict[str, float]:
    totals = _importance_by_variable(context, values)
    out.table(
        title,
        ["Variable", "Importancia", "% del total"],
        [[name, value, 100 * value / max(sum(totals.values()), 1e-12)] for name, value in totals.items()],
        note="Mide cuánto reduce cada variable la impureza (o el error) en los cortes del modelo. Tiende a favorecer variables con muchos valores distintos: contraste con la importancia por permutación.",
    )
    return totals


# ============================================================================ decision tree
def _tree_rules(model: Any, names: list[str], dummy: list[str | None], problem) -> list[list[Any]]:
    t = model.tree_
    rows: list[list[Any]] = []

    def condition(node: int, left: bool) -> str:
        i, thr = t.feature[node], t.threshold[node]
        if dummy[i] is not None:
            raw = names[i].split(" = ")[0]
            return f"{raw} {'≠' if left else '='} {dummy[i]}"
        return f"{names[i]} {'≤' if left else '>'} {thr:.4g}"

    def visit(node: int, path: list[str]) -> None:
        if t.children_left[node] == -1:
            counts = t.value[node][0]
            if problem.task == CLASSIFICATION:
                share = counts / counts.sum()
                top = int(share.argmax())
                rows.append(
                    [
                        " y ".join(path) or "(todas las filas)",
                        problem.labels[top],
                        int(t.n_node_samples[node]),
                        100 * float(share[top]),
                    ]
                )
            else:
                rows.append(
                    [
                        " y ".join(path) or "(todas las filas)",
                        float(counts[0]),
                        int(t.n_node_samples[node]),
                        float(np.sqrt(max(t.impurity[node], 0.0))),
                    ]
                )
            return
        visit(t.children_left[node], [*path, condition(node, True)])
        visit(t.children_right[node], [*path, condition(node, False)])

    visit(0, [])
    return rows


def _tree_chart(model: Any, names: list[str], dummy: list[str | None], problem) -> dict[str, Any] | None:
    t = model.tree_
    if t.node_count > 63:
        return None
    x: dict[int, float] = {}
    depth: dict[int, int] = {}
    counter = [0]

    def place(node: int, level: int) -> None:
        depth[node] = level
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:
            x[node] = float(counter[0])
            counter[0] += 1
        else:
            place(left, level + 1)
            place(right, level + 1)
            x[node] = (x[left] + x[right]) / 2

    place(0, 0)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    internal, leaves = (
        {"x": [], "y": [], "text": [], "hover": []},
        {"x": [], "y": [], "text": [], "hover": [], "color": []},
    )
    for node in range(t.node_count):
        left, right = t.children_left[node], t.children_right[node]
        samples = int(t.n_node_samples[node])
        if left != -1:
            for child in (left, right):
                edge_x += [x[node], x[child], None]
                edge_y += [-depth[node], -depth[child], None]
            i = t.feature[node]
            test = (
                f"{names[i].split(' = ')[0]} = {dummy[i]}?"
                if dummy[i] is not None
                else f"{names[i]} ≤ {t.threshold[node]:.4g}?"
            )
            internal["x"].append(x[node])
            internal["y"].append(-depth[node])
            internal["text"].append(test)
            internal["hover"].append(f"{test}<br>n = {samples}<br>sí → izquierda, no → derecha")
        else:
            counts = t.value[node][0]
            leaves["x"].append(x[node])
            leaves["y"].append(-depth[node])
            if problem.task == CLASSIFICATION:
                share = counts / counts.sum()
                top = int(share.argmax())
                leaves["text"].append(f"{problem.labels[top]}<br>n={samples}")
                leaves["hover"].append(f"{problem.labels[top]} ({100 * share[top]:.0f} %)<br>n = {samples}")
                leaves["color"].append(PALETTE[top % len(PALETTE)])
            else:
                leaves["text"].append(f"{float(counts[0]):.3g}<br>n={samples}")
                leaves["hover"].append(f"media {float(counts[0]):.4g}<br>n = {samples}")
                leaves["color"].append(PALETTE[0])
    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": edge_x,
            "y": edge_y,
            "line": {"color": "#94a3b8", "width": 1.2},
            "hoverinfo": "skip",
            "showlegend": False,
        },
        {
            "type": "scatter",
            "mode": "markers+text",
            "x": internal["x"],
            "y": internal["y"],
            "text": internal["text"],
            "textposition": "top center",
            "textfont": {"size": 10},
            "marker": {"size": 9, "color": "#64748b"},
            "hovertext": internal["hover"],
            "hoverinfo": "text",
            "name": "Corte",
            "showlegend": False,
        },
        {
            "type": "scatter",
            "mode": "markers+text",
            "x": leaves["x"],
            "y": leaves["y"],
            "text": leaves["text"],
            "textposition": "bottom center",
            "textfont": {"size": 10},
            "marker": {"size": 12, "color": leaves["color"], "symbol": "square"},
            "hovertext": leaves["hover"],
            "hoverinfo": "text",
            "name": "Hoja",
            "showlegend": False,
        },
    ]
    height = int(min(900, 140 + 110 * (max(depth.values()) + 1)))
    layout = {
        "xaxis": {"visible": False},
        "yaxis": {"visible": False, "range": [-(max(depth.values()) + 0.9), 0.6]},
        "showlegend": False,
    }
    return raw_chart(traces, layout, "Árbol de decisión (a la izquierda lo que cumple el corte)", height=height)


def _tree_resolve(params: dict[str, Any]):
    def resolve(problem, metric, seed):
        if params["prune"] != "cv":
            return {}, None
        # candidate pruning levels: the cost-complexity path of a deep tree on the training portion
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        from sigma.domain.analysis.ml_core import make_model

        algorithm = ALGORITHMS["tree"]
        train, _ = holdout_split(problem.y, problem.task, float(params["test_size"]), seed)
        grow = {
            "max_depth": int(params["max_depth"]),
            "min_samples_leaf": int(params["min_samples_leaf"]),
            "criterion": params["criterion"],
            "class_weight": params["class_weight"],
            "ccp_alpha": 0.0,
        }
        prep = make_model(problem, algorithm, grow, seed).named_steps["prep"]
        encoded = prep.fit_transform(problem.X.iloc[train])
        tree = (DecisionTreeClassifier if problem.task == CLASSIFICATION else DecisionTreeRegressor)(
            max_depth=params["max_depth"], min_samples_leaf=int(params["min_samples_leaf"]), random_state=seed
        )
        alphas = np.unique(np.round(tree.cost_complexity_pruning_path(encoded, problem.y[train]).ccp_alphas[:-1], 10))
        if len(alphas) > 25:
            alphas = np.unique(np.quantile(alphas, np.linspace(0, 1, 25)))
        if len(alphas) == 0:
            return {}, None
        chosen, rows = search_values(
            problem,
            algorithm,
            {**grow},
            "ccp_alpha",
            [float(a) for a in alphas],
            metric=metric,
            folds=int(params["folds"]),
            test_size=float(params["test_size"]),
            seed=seed,
            prefer="simple_high",
        )
        return {"ccp_alpha": float(chosen)}, rows

    return resolve


def _tree_extras(out: Output, ctx: SimpleNamespace) -> dict[str, Any]:
    model = _model_step(ctx)
    info = encoded_info(ctx.final, ctx.problem)
    names, dummy = [i[1] for i in info], [i[2] for i in info]
    depth, leaves = int(model.get_depth()), int(model.get_n_leaves())
    out.tiles(
        [("Profundidad del árbol", depth), ("Hojas", leaves), ("Poda (ccp_alpha)", float(ctx.settings["ccp_alpha"]))]
    )
    if ctx.info:
        label = metric_table(ctx.problem.task)[ctx.evaluation.metric][0].split(" (")[0]
        out.table(
            "Poda por costo-complejidad (validación cruzada)",
            ["ccp_alpha", f"{label} (media)", "Desv. est.", "Elegido"],
            [
                [float(a), m, sd, good("●") if abs(a - ctx.settings["ccp_alpha"]) < 1e-12 else ""]
                for a, m, sd in ctx.info
            ],
            note="Se eligió el árbol más simple cuyo desempeño queda a menos de una desviación estándar del mejor (regla de un error estándar).",
        )
    chart = _tree_chart(model, names, dummy, ctx.problem)
    if chart is not None:
        out.chart(chart)
    else:
        out.alert(
            "info", "El árbol tiene demasiados nodos para dibujarlo; lea las reglas de abajo o limite la profundidad."
        )
    rules = sorted(_tree_rules(model, names, dummy, ctx.problem), key=lambda r: -r[2])[:40]
    columns = (
        ["Regla", "Clase predicha", "Filas", "% de esa clase en la hoja"]
        if ctx.problem.task == CLASSIFICATION
        else ["Regla", "Valor predicho (media)", "Filas", "Desv. est. en la hoja"]
    )
    out.table(
        "Reglas de decisión (hojas más pobladas primero)",
        columns,
        rules,
        note="Cada fila es un camino del árbol hasta una hoja; sirve para explicar la predicción a quien no es estadístico.",
    )
    totals = _impurity_table(out, ctx, model.feature_importances_, "Importancia por impureza")
    return {"depth": depth, "leaves": leaves, "impurity_importance": totals}


def _tree(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    hyper = {
        "max_depth": int(p["max_depth"]),
        "min_samples_leaf": int(p["min_samples_leaf"]),
        "criterion": p["criterion"],
        "class_weight": p["class_weight"],
        "ccp_alpha": float(p["ccp_alpha"]),
    }
    return run_supervised(
        frame,
        roles,
        p,
        ALGORITHMS["tree"],
        hyper,
        title="Árbol de decisión (CART)",
        detail=f"profundidad máx. {hyper['max_depth']}",
        extras=_tree_extras,
        resolve=_tree_resolve(p),
    )


def _tree_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return code_for(
        roles,
        p,
        dataset,
        "from sklearn.tree import DecisionTreeClassifier  # DecisionTreeRegressor si el objetivo es numérico",
        f"DecisionTreeClassifier(max_depth={int(p['max_depth'])}, min_samples_leaf={int(p['min_samples_leaf'])}, ccp_alpha={float(p['ccp_alpha'])}, random_state={int(p['seed'])})",
        False,
    )


# ============================================================================ random forest
def _forest_extras(out: Output, ctx: SimpleNamespace) -> dict[str, Any]:
    model = _model_step(ctx)
    oob = float(model.oob_score_) if hasattr(model, "oob_score_") else float("nan")
    out.tiles(
        [
            ("Árboles", int(model.n_estimators)),
            ("OOB: " + ("exactitud" if ctx.problem.task == CLASSIFICATION else "R²"), oob),
        ]
    )
    out.alert(
        "info",
        "OOB (fuera de la bolsa) es una validación gratuita del bosque: cada árbol se prueba con las filas que no vio al construirlo.",
    )
    totals = _impurity_table(out, ctx, model.feature_importances_, "Importancia por impureza (promedio de los árboles)")
    return {"oob_score": oob, "impurity_importance": totals}


def _forest(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    hyper = {
        "n_estimators": int(p["n_estimators"]),
        "max_depth": int(p["max_depth"]) if p["max_depth"] else None,
        "min_samples_leaf": int(p["min_samples_leaf"]),
        "max_features": p["max_features"],
        "class_weight": p["class_weight"],
    }
    return run_supervised(
        frame,
        roles,
        p,
        ALGORITHMS["forest"],
        hyper,
        title="Bosque aleatorio",
        detail=f"{hyper['n_estimators']} árboles",
        extras=_forest_extras,
    )


def _forest_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return code_for(
        roles,
        p,
        dataset,
        "from sklearn.ensemble import RandomForestClassifier  # RandomForestRegressor si el objetivo es numérico",
        f"RandomForestClassifier(n_estimators={int(p['n_estimators'])}, max_depth={int(p['max_depth']) if p['max_depth'] else None}, min_samples_leaf={int(p['min_samples_leaf'])}, max_features={p['max_features']!r}, random_state={int(p['seed'])})",
        False,
    )


# ============================================================================ gradient boosting
def _boosting_extras(out: Output, ctx: SimpleNamespace) -> dict[str, Any]:
    model = _model_step(ctx)
    problem, ev = ctx.problem, ctx.evaluation
    used = int(model.n_iter_)
    out.tiles([("Iteraciones usadas", used), ("Tasa de aprendizaje", float(ctx.settings["learning_rate"]))])
    if len(ev.test_rows) == 0:
        out.alert(
            "info", "Reserve una fracción para prueba si quiere ver cómo cambia el error con el número de iteraciones."
        )
        return {"iterations": used}
    prep = ctx.final.named_steps["prep"]
    shown = (
        ev.train_rows
        if len(ev.train_rows) <= 4000
        else np.sort(np.random.default_rng(ev.seed).choice(ev.train_rows, 4000, replace=False))
    )
    sets = {
        "Entrenamiento": (prep.transform(problem.X.iloc[shown]), problem.y[shown]),
        "Prueba": (prep.transform(problem.X.iloc[ev.test_rows]), problem.y[ev.test_rows]),
    }
    curves: dict[str, list[float]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (x, y) in sets.items():
            if problem.task == CLASSIFICATION:
                curves[name] = [
                    float(log_loss(y, np.clip(p, 1e-15, 1), labels=list(range(problem.k))))
                    for p in model.staged_predict_proba(x)
                ]
            else:
                curves[name] = [float(np.sqrt(mean_squared_error(y, p))) for p in model.staged_predict(x)]
    iterations = list(range(1, len(curves["Prueba"]) + 1))
    best = int(np.argmin(curves["Prueba"])) + 1
    ylabel = "Pérdida logarítmica" if problem.task == CLASSIFICATION else "RMSE"
    out.chart(
        line_chart(
            f"{ylabel} según el número de iteraciones",
            "Iteraciones (árboles)",
            ylabel,
            [{"name": name, "x": iterations, "y": values} for name, values in curves.items()],
            vlines=[{"value": best, "label": f"mejor en prueba: {best}", "dash": "dash"}],
            height=340,
        )
    )
    if best < 0.7 * len(iterations):
        out.alert(
            "info",
            f"La prueba mejora hasta la iteración {best} y luego empeora: use menos iteraciones, una tasa menor o active la parada temprana.",
        )
    return {"iterations": used, "best_iteration": best}


def _boosting(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    hyper = {
        "max_iter": int(p["max_iter"]),
        "learning_rate": float(p["learning_rate"]),
        "max_depth": int(p["max_depth"]),
        "min_samples_leaf": int(p["min_samples_leaf"]),
        "l2_regularization": float(p["l2_regularization"]),
        "early_stopping": bool(p["early_stopping"]),
        "class_weight": p["class_weight"],
    }
    return run_supervised(
        frame,
        roles,
        p,
        ALGORITHMS["boosting"],
        hyper,
        title="Gradient boosting",
        detail=f"{hyper['max_iter']} iteraciones, tasa {hyper['learning_rate']:g}",
        extras=_boosting_extras,
    )


def _boosting_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return code_for(
        roles,
        p,
        dataset,
        "from sklearn.ensemble import HistGradientBoostingClassifier  # HistGradientBoostingRegressor si el objetivo es numérico",
        f"HistGradientBoostingClassifier(max_iter={int(p['max_iter'])}, learning_rate={float(p['learning_rate'])}, max_depth={int(p['max_depth'])}, random_state={int(p['seed'])})",
        False,
    )


# ============================================================================ k nearest neighbours
def _knn_resolve(params: dict[str, Any]):
    def resolve(problem, metric, seed):
        if not params["search"]:
            return {}, None
        limit = int(min(params["k_max"], max(len(problem.X) // 3, 2)))
        values = sorted({*range(1, min(limit, 10) + 1), *np.unique(np.linspace(1, limit, 12).astype(int)).tolist()})
        base = {"weights": params["weights"], "p": int(params["p"])}
        chosen, rows = search_values(
            problem,
            ALGORITHMS["knn"],
            base,
            "n_neighbors",
            values,
            metric=metric,
            folds=int(params["folds"]),
            test_size=float(params["test_size"]),
            seed=seed,
            prefer="simple_high",
        )
        return {"n_neighbors": int(chosen)}, rows

    return resolve


def _knn_extras(out: Output, ctx: SimpleNamespace) -> dict[str, Any] | None:
    if not ctx.info:
        return None
    label = metric_table(ctx.problem.task)[ctx.evaluation.metric][0].split(" (")[0]
    chosen = int(ctx.settings["n_neighbors"])
    out.table(
        "Búsqueda del número de vecinos (validación cruzada)",
        ["k", f"{label} (media)", "Desv. est.", "Elegido"],
        [[int(k), m, sd, good("●") if int(k) == chosen else ""] for k, m, sd in ctx.info],
        note="Pocos vecinos dan modelos muy flexibles (sobreajuste); muchos, modelos muy suaves. Se eligió el k más grande dentro de un error estándar del mejor.",
    )
    out.chart(
        line_chart(
            f"{label} según k",
            "Vecinos (k)",
            label,
            [
                {
                    "name": "Validación cruzada",
                    "x": [int(k) for k, _, _ in ctx.info],
                    "y": [m for _, m, _ in ctx.info],
                    "markers": True,
                }
            ],
            vlines=[{"value": chosen, "label": f"k = {chosen}"}],
            height=320,
        )
    )
    return {"n_neighbors": chosen}


def _knn(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    hyper = {"n_neighbors": int(p["n_neighbors"]), "weights": p["weights"], "p": int(p["p"])}
    return run_supervised(
        frame,
        roles,
        p,
        ALGORITHMS["knn"],
        hyper,
        title="k vecinos más cercanos",
        detail="variables estandarizadas",
        extras=_knn_extras,
        resolve=_knn_resolve(p),
    )


def _knn_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return code_for(
        roles,
        p,
        dataset,
        "from sklearn.neighbors import KNeighborsClassifier  # KNeighborsRegressor si el objetivo es numérico",
        f"KNeighborsClassifier(n_neighbors={int(p['n_neighbors'])}, weights={p['weights']!r}, p={int(p['p'])})",
        True,
    )


# ============================================================================ support vector machines
def _svm_extras(out: Output, ctx: SimpleNamespace) -> dict[str, Any]:
    model = _model_step(ctx)
    n_support = int(model.support_.shape[0])
    share = 100 * n_support / len(ctx.evaluation.train_rows)
    out.tiles([("Vectores de soporte", n_support), ("% de las filas de entrenamiento", share)])
    if share > 70:
        out.alert(
            "info",
            "Casi todas las filas son vectores de soporte: el modelo es muy flexible o los datos muy mezclados; pruebe un C menor o el núcleo lineal.",
        )
    if ctx.problem.task == CLASSIFICATION:
        out.alert(
            "info",
            "La SVM no da probabilidades: el AUC y las curvas se calculan con la distancia al hiperplano (que ordena igual que una probabilidad). La pérdida logarítmica no aplica.",
        )
    return {"support_vectors": n_support}


def _svm(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    hyper = {
        "kernel": p["kernel"],
        "C": float(p["C"]),
        "gamma": p["gamma"],
        "degree": int(p["degree"]),
        "epsilon": float(p["epsilon"]),
        "class_weight": p["class_weight"],
    }
    return run_supervised(
        frame,
        roles,
        p,
        ALGORITHMS["svm"],
        hyper,
        title="Máquina de vectores de soporte",
        detail=f"núcleo {p['kernel']}, C = {hyper['C']:g}",
        extras=_svm_extras,
    )


def _svm_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return code_for(
        roles,
        p,
        dataset,
        "from sklearn.svm import SVC  # SVR si el objetivo es numérico",
        f"SVC(kernel={p['kernel']!r}, C={float(p['C'])}, gamma={p['gamma']!r})",
        True,
    )


# ============================================================================ naive Bayes
def _bayes_extras(out: Output, ctx: SimpleNamespace) -> dict[str, Any]:
    model = _model_step(ctx)
    info = encoded_info(ctx.final, ctx.problem)
    out.table(
        "Probabilidades previas de cada clase",
        ["Clase", "Probabilidad previa"],
        [[c, float(pr)] for c, pr in zip(ctx.problem.labels, model.class_prior_, strict=True)],
    )
    shown = list(range(min(len(info), 10)))
    out.table(
        "Media de cada variable por clase (las primeras 10)",
        ["Clase", *[info[i][1] for i in shown]],
        [[c, *(float(model.theta_[k, i]) for i in shown)] for k, c in enumerate(ctx.problem.labels)],
        note="Naive Bayes supone que, dentro de cada clase, las variables son normales e independientes; funciona bien aun cuando no lo son del todo, pero sus probabilidades suelen ser extremas.",
    )
    return {}


def _bayes(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    return run_supervised(
        frame,
        roles,
        p,
        ALGORITHMS["naive_bayes"],
        {"var_smoothing": float(p["var_smoothing"])},
        title="Naive Bayes gaussiano",
        extras=_bayes_extras,
    )


def _bayes_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return code_for(
        roles,
        p,
        dataset,
        "from sklearn.naive_bayes import GaussianNB",
        f"GaussianNB(var_smoothing={float(p['var_smoothing'])})",
        False,
    )


# ============================================================================ neural network
def _mlp_extras(out: Output, ctx: SimpleNamespace) -> dict[str, Any]:
    model = _model_step(ctx)
    epochs = int(model.n_iter_)
    limit = int(ctx.settings["max_iter"])
    out.tiles(
        [
            ("Épocas realizadas", epochs),
            ("Pérdida final", float(model.loss_)),
            ("Capas ocultas", " → ".join(str(s) for s in model.hidden_layer_sizes)),
        ]
    )
    if epochs >= limit:
        out.alert(
            "warning",
            f"El entrenamiento llegó al máximo de {limit} épocas sin converger: aumente las épocas, la tasa de aprendizaje o estandarice mejor los datos.",
        )
    series = [
        {
            "name": "Pérdida de entrenamiento",
            "x": list(range(1, len(model.loss_curve_) + 1)),
            "y": [float(v) for v in model.loss_curve_],
        }
    ]
    scores = getattr(model, "validation_scores_", None)
    if scores is not None:
        out.chart(line_chart("Evolución del entrenamiento", "Época", "Pérdida de entrenamiento", series, height=300))
        out.chart(
            line_chart(
                "Puntaje en la validación interna (parada temprana)",
                "Época",
                "Puntaje",
                [{"name": "Validación", "x": list(range(1, len(scores) + 1)), "y": [float(v) for v in scores]}],
                height=300,
            )
        )
    else:
        out.chart(line_chart("Evolución del entrenamiento", "Época", "Pérdida", series, height=300))
    return {"epochs": epochs}


def _mlp(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    layers = parse_layers(p["hidden_layer_sizes"])
    hyper = {
        "hidden_layer_sizes": layers,
        "activation": p["activation"],
        "alpha": float(p["alpha"]),
        "learning_rate_init": float(p["learning_rate_init"]),
        "max_iter": int(p["max_iter"]),
        "early_stopping": bool(p["early_stopping"]),
    }
    return run_supervised(
        frame,
        roles,
        p,
        ALGORITHMS["mlp"],
        hyper,
        title="Red neuronal (perceptrón multicapa)",
        detail="capas " + "-".join(str(s) for s in layers),
        extras=_mlp_extras,
    )


def _mlp_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    layers = parse_layers(p["hidden_layer_sizes"])
    return code_for(
        roles,
        p,
        dataset,
        "from sklearn.neural_network import MLPClassifier  # MLPRegressor si el objetivo es numérico",
        f"MLPClassifier(hidden_layer_sizes={layers}, activation={p['activation']!r}, alpha={float(p['alpha'])}, max_iter={int(p['max_iter'])}, early_stopping={bool(p['early_stopping'])}, random_state={int(p['seed'])})",
        True,
    )


# ============================================================================ penalized linear models
def _penalized_extras(out: Output, ctx: SimpleNamespace) -> dict[str, Any]:
    model = _model_step(ctx)
    tuned = hasattr(model, "best_params_")
    estimator = model.best_estimator_ if tuned else model
    info = encoded_info(ctx.final, ctx.problem)
    coef = np.atleast_2d(np.asarray(estimator.coef_, float))
    classification = ctx.problem.task == CLASSIFICATION
    if classification and ctx.problem.k == 2:
        coef = coef if coef.shape[0] == 1 else coef[:1]
    strength = (1 / estimator.C) if classification else estimator.alpha
    nonzero = int((np.abs(coef).max(axis=0) > 1e-10).sum())
    out.tiles(
        [
            ("Intensidad de la penalización (λ)", float(strength)),
            ("Coeficientes distintos de cero", f"{nonzero} de {len(info)}"),
        ]
    )
    if tuned:
        out.alert(
            "info",
            f"La intensidad se eligió por validación cruzada interna dentro de cada pliegue (valor final: {strength:.4g}).",
        )
    order = np.argsort(-np.abs(coef).max(axis=0))[:30]
    columns = ["Variable (estandarizada)"]
    if classification and ctx.problem.k == 2:
        columns += [f"Coeficiente (a favor de «{ctx.problem.labels[1]}»)", "Razón de momios (odds ratio)"]
        rows = [[info[i][1], float(coef[0, i]), float(np.exp(coef[0, i]))] for i in order]
    elif classification:
        columns += [f"Coef. {c}" for c in ctx.problem.labels[:6]]
        rows = [[info[i][1], *(float(coef[c, i]) for c in range(min(ctx.problem.k, 6)))] for i in order]
    else:
        columns += ["Coeficiente"]
        rows = [[info[i][1], float(coef[0, i])] for i in order]
    out.table(
        "Coeficientes (las 30 variables con mayor efecto)",
        columns,
        rows,
        note="Las numéricas se estandarizan, así que los coeficientes son comparables entre sí. La penalización L1 deja en cero las que no aportan; la L2 las encoge todas.",
    )
    return {"nonzero": nonzero, "lambda": float(strength)}


def _penalized(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    strength = float(p["strength"])
    hyper = {
        "C": 1.0 / strength,
        "alpha": strength,
        "l1_ratio": float(p["l1_ratio"]),
        "tune": bool(p["tune"]),
        "class_weight": p["class_weight"],
    }
    kind = {0.0: "L2 (ridge)", 1.0: "L1 (lasso)"}.get(hyper["l1_ratio"], f"elastic net (L1 = {hyper['l1_ratio']:g})")
    return run_supervised(
        frame,
        roles,
        p,
        ALGORITHMS["penalized"],
        hyper,
        title="Modelo lineal regularizado",
        detail=f"penalización {kind}",
        extras=_penalized_extras,
    )


def _penalized_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return code_for(
        roles,
        p,
        dataset,
        "from sklearn.linear_model import LogisticRegression  # Ridge / ElasticNet si el objetivo es numérico",
        f"LogisticRegression(C={1 / float(p['strength'])}, l1_ratio={float(p['l1_ratio'])}, solver='saga', max_iter=2000)",
        True,
    )


# ============================================================================ the catalogue
_NUMBER = "number"
_L1 = (("0", "0 — L2 (ridge)"), ("0.5", "0.5 — elastic net"), ("1", "1 — L1 (lasso)"))

SPECS = (
    _spec("tree", "Árbol de decisión (CART)", GROUP_TREES, "Reglas «si… entonces…» que explican la predicción, con poda por validación cruzada.", _tree, code=_tree_code, params=(*COMMON_PARAMS, FieldSpec("max_depth", "Profundidad máxima", "integer", 4, minimum=1, maximum=20), FieldSpec("min_samples_leaf", "Mínimo de filas por hoja", "integer", 5, minimum=1, maximum=1000), FieldSpec("criterion", "Criterio de corte (clasificación)", "select", "gini", (("gini", "Gini"), ("entropy", "Entropía"))), FieldSpec("prune", "Poda", "select", "none", (("none", "Sin podar"), ("cv", "Por costo-complejidad con validación cruzada"))), FieldSpec("ccp_alpha", "Poda manual (ccp_alpha)", _NUMBER, 0.0, minimum=0, maximum=10, visible_if=("prune", ("none",))), CLASS_WEIGHT), example={"roles": {"target": "g", "features": ["y", "x1", "x2", "h"]}, "params": {"prune": "cv", "folds": 3}}),
    _spec("forest", "Bosque aleatorio", GROUP_TREES, "Cientos de árboles promediados: robusto, con validación OOB y dos medidas de importancia.", _forest, code=_forest_code, params=(*COMMON_PARAMS, FieldSpec("n_estimators", "Número de árboles", "integer", 200, minimum=10, maximum=2000), FieldSpec("max_depth", "Profundidad máxima (vacío = sin límite)", "integer", optional=True, minimum=1, maximum=50), FieldSpec("min_samples_leaf", "Mínimo de filas por hoja", "integer", 1, minimum=1, maximum=1000), FieldSpec("max_features", "Variables consideradas en cada corte", "select", "auto", (("auto", "Automático (raíz cuadrada o un tercio)"), ("sqrt", "Raíz cuadrada"), ("log2", "Logaritmo base 2"), ("third", "Un tercio"), ("all", "Todas"))), CLASS_WEIGHT), example={"roles": {"target": "y", "features": ["x1", "x2", "x3", "g"]}, "params": {"n_estimators": 50, "folds": 3}}),
    _spec("boosting", "Gradient boosting", GROUP_TREES, "Árboles pequeños añadidos uno tras otro, cada uno corrige a los anteriores; con la curva del error por iteración.", _boosting, code=_boosting_code, params=(*COMMON_PARAMS, FieldSpec("max_iter", "Iteraciones (árboles)", "integer", 150, minimum=10, maximum=2000), FieldSpec("learning_rate", "Tasa de aprendizaje", _NUMBER, 0.1, minimum=0.001, maximum=1), FieldSpec("max_depth", "Profundidad de cada árbol", "integer", 3, minimum=1, maximum=12), FieldSpec("min_samples_leaf", "Mínimo de filas por hoja", "integer", 20, minimum=1, maximum=1000), FieldSpec("l2_regularization", "Regularización L2", _NUMBER, 0.0, minimum=0, maximum=100), FieldSpec("early_stopping", "Parada temprana", "boolean", False), CLASS_WEIGHT), example={"roles": {"target": "b", "features": ["y", "x1", "x2", "g"]}, "params": {"max_iter": 40, "folds": 3}}),
    _spec("knn", "k vecinos más cercanos", GROUP_OTHER, "Predice con las filas más parecidas; elige k por validación cruzada si se pide.", _knn, code=_knn_code, params=(*COMMON_PARAMS, FieldSpec("n_neighbors", "Vecinos (k; se ignora si busca el mejor)", "integer", 5, minimum=1, maximum=500), FieldSpec("search", "Buscar el mejor k con validación cruzada", "boolean", False), FieldSpec("k_max", "Probar k hasta", "integer", 30, minimum=3, maximum=200), FieldSpec("weights", "Ponderación de los vecinos", "select", "uniform", (("uniform", "Iguales"), ("distance", "Por cercanía"))), FieldSpec("p", "Distancia", "select", "2", (("2", "Euclídea"), ("1", "Manhattan")))), example={"roles": {"target": "g", "features": ["y", "x1", "x2"]}, "params": {"search": True, "k_max": 12, "folds": 3}}),
    _spec("svm", "Máquina de vectores de soporte (SVM)", GROUP_OTHER, "Frontera de máximo margen con núcleos lineal, polinómico o radial; para clasificación y regresión.", _svm, code=_svm_code, params=(*COMMON_PARAMS, FieldSpec("kernel", "Núcleo", "select", "rbf", (("rbf", "Radial (RBF)"), ("linear", "Lineal"), ("poly", "Polinómico"))), FieldSpec("C", "Costo C", _NUMBER, 1.0, minimum=0.001, maximum=10000), FieldSpec("gamma", "Gamma", "select", "scale", (("scale", "Automática (escala)"), ("auto", "1 / variables"))), FieldSpec("degree", "Grado (polinómico)", "integer", 3, minimum=2, maximum=6, visible_if=("kernel", ("poly",))), FieldSpec("epsilon", "Margen ε (regresión)", _NUMBER, 0.1, minimum=0, maximum=100), CLASS_WEIGHT), example={"roles": {"target": "g", "features": ["y", "x1", "x2"]}, "params": {"folds": 3}}),
    _spec("naive_bayes", "Naive Bayes gaussiano", GROUP_OTHER, "Clasificador probabilístico muy rápido; buen punto de partida con pocos datos.", _bayes, code=_bayes_code, params=(*COMMON_PARAMS, FieldSpec("var_smoothing", "Suavizado de la varianza", _NUMBER, 1e-9, minimum=0, maximum=1)), example={"roles": {"target": "g", "features": ["y", "x1", "x2"]}, "params": {"folds": 3}}),
    _spec("mlp", "Red neuronal (perceptrón multicapa)", GROUP_OTHER, "Capas de neuronas entrenadas por descenso de gradiente, con curva de pérdida y parada temprana.", _mlp, code=_mlp_code, params=(*COMMON_PARAMS, FieldSpec("hidden_layer_sizes", "Neuronas de cada capa oculta (separadas por comas)", "text", "32, 16"), FieldSpec("activation", "Función de activación", "select", "relu", (("relu", "ReLU"), ("tanh", "Tangente hiperbólica"), ("logistic", "Logística"))), FieldSpec("alpha", "Regularización L2 (α)", _NUMBER, 1e-4, minimum=0, maximum=10), FieldSpec("learning_rate_init", "Tasa de aprendizaje", _NUMBER, 0.001, minimum=1e-5, maximum=1), FieldSpec("max_iter", "Épocas máximas", "integer", 300, minimum=10, maximum=5000), FieldSpec("early_stopping", "Parada temprana (reserva 10 % para validar)", "boolean", True)), example={"roles": {"target": "y", "features": ["x1", "x2", "x3"]}, "params": {"hidden_layer_sizes": "8", "max_iter": 80, "folds": 3}}),
    _spec("penalized", "Modelo lineal regularizado (ridge, lasso, elastic net)", GROUP_OTHER, "Regresión o logística penalizada: encoge los coeficientes, selecciona variables (L1) y da odds ratios.", _penalized, code=_penalized_code, params=(*COMMON_PARAMS, FieldSpec("l1_ratio", "Mezcla L1 / L2", "select", "0", _L1), FieldSpec("strength", "Intensidad de la penalización (λ)", _NUMBER, 1.0, minimum=0.0001, maximum=100000), FieldSpec("tune", "Elegir λ por validación cruzada interna", "boolean", True), CLASS_WEIGHT), example={"roles": {"target": "b", "features": ["y", "x1", "x2", "x3"]}, "params": {"l1_ratio": "1", "folds": 3}}),
)  # fmt: skip

registry.register(*SPECS)
