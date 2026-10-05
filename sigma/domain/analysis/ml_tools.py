"""Around the models: comparing them, tuning them, learning curves, partial dependence, variable selection, anomalies and the evaluation of predictions."""

from __future__ import annotations

import itertools
import time
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LassoCV, LogisticRegression, Ridge
from sklearn.metrics import matthews_corrcoef, roc_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import (
    bar_chart,
    box_chart,
    heatmap_chart,
    hist_chart,
    line_chart,
    raw_chart,
    scatter_chart,
)
from sigma.domain.analysis.ml_algorithms import ALGORITHMS, algorithm_options, get, parse_layers, tuning_space
from sigma.domain.analysis.ml_core import (
    CLASSIFICATION,
    COMMON_PARAMS,
    FEATURES,
    METRIC_OPTIONS,
    MODULE,
    PREDICTIONS,
    REGRESSION,
    SEED,
    TARGET,
    Pred,
    build_problem,
    choose_metric,
    class_report,
    class_scores,
    classification_scores,
    confusion_section,
    cross_validate,
    curves_section,
    delong_auc,
    encoded_info,
    evaluate,
    holdout_split,
    make_model,
    make_splits,
    metric_table,
    predict_all,
    regression_scores,
    regression_section,
    report,
    score_all,
    summarize,
)
from sigma.domain.analysis.multivariate import MAX_POINTS, VARS, _result_frame, complete, dropped_alert
from sigma.domain.analysis.multivariate_groups import _projection
from sigma.domain.analysis.output import Output, P, bad, good, muted, strong
from sigma.domain.analysis.plots import level_order
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.analysis.support import load_line
from sigma.domain.errors import InvalidInputError

GROUP_COMPARE = "Comparar y ajustar"
GROUP_EXPLAIN = "Explicar y evaluar"
GROUP_DATA = "Variables y datos"
MAX_CONFIGS = 60
TASK_PARAM = COMMON_PARAMS[0]
FOLDS = FieldSpec("folds", "Pliegues de validación cruzada", "integer", 5, minimum=2, maximum=20)
METRIC = FieldSpec("metric", "Métrica principal", "select", "auto", METRIC_OPTIONS)
MISSING = COMMON_PARAMS[4]
POSITIVE = COMMON_PARAMS[6]
TEST_SIZE = FieldSpec(
    "test_size", "Fracción reservada para prueba (0 = ninguna)", "number", 0.2, minimum=0, maximum=0.5
)
ALGORITHM = FieldSpec("algorithm", "Modelo", "select", "forest", algorithm_options())
MODEL_PARAMS = (TASK_PARAM, FOLDS, METRIC, MISSING, POSITIVE, SEED)


def _spec(id_: str, label: str, group: str, description: str, compute, *, heavy: bool = True, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, heavy=heavy, **kw)


def _problem(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any], algorithm=None, predict_new: bool = False):
    return build_problem(
        frame,
        roles["target"],
        roles["features"],
        task=p["task"],
        missing=p["missing"],
        predict_new=predict_new,
        positive=p.get("positive") or None,
        allowed=algorithm.tasks if algorithm else (CLASSIFICATION, REGRESSION),
    )


def _label(problem, metric: str) -> str:
    return metric_table(problem.task)[metric][0].split(" (")[0]


def value_text(value: Any) -> str:
    if value is None:
        return "sin límite"
    if isinstance(value, tuple):
        return "-".join(str(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def settings_text(settings: dict[str, Any]) -> str:
    return ", ".join(f"{k} = {value_text(v)}" for k, v in settings.items())


# ============================================================================ compare models
def _compare(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    problem = _problem(frame, roles, p)
    seed, metric = int(p["seed"]), choose_metric(problem, p["metric"])
    higher = metric_table(problem.task)[metric][1]
    chosen = [a for a in p["algorithms"] if a in ALGORITHMS]
    if not chosen:
        raise InvalidInputError("Elija al menos un modelo para comparar.")
    skipped, algorithms = [], []
    for key in chosen:
        algorithm = ALGORITHMS[key]
        if problem.task not in algorithm.tasks:
            skipped.append(f"{algorithm.label} (solo clasificación)")
        elif algorithm.max_rows and len(problem.X) > algorithm.max_rows:
            skipped.append(f"{algorithm.label} (demasiadas filas)")
        else:
            algorithms.append(algorithm)
    if not algorithms:
        raise InvalidInputError("Ninguno de los modelos elegidos sirve para estos datos.")
    train, test = holdout_split(problem.y, problem.task, float(p["test_size"]), seed)
    x_train, y_train = problem.X.iloc[train], problem.y[train]
    repeats = int(p["repeats"])
    splits = make_splits(y_train, problem.task, int(p["folds"]), seed, repeats)
    per_repeat = len(splits) // repeats
    results, held_out = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for algorithm in algorithms:

            def build(algorithm=algorithm):
                return make_model(problem, algorithm, {}, seed)

            results[algorithm.id] = cross_validate(
                problem, build, x_train, y_train, splits, folds_per_repeat=per_repeat, keep_train=False
            )
            if len(test):
                model = build().fit(x_train, y_train)
                held_out[algorithm.id] = score_all(
                    problem, problem.y[test], predict_all(model, problem.X.iloc[test], problem.task)
                )[metric]
        baseline = cross_validate(
            problem, None, x_train, y_train, splits, folds_per_repeat=per_repeat, keep_train=False
        )
    folds_total = len(splits)
    ratio = float(np.mean([len(t) / len(tr) for tr, t in splits]))
    scores = {a.id: np.array([r[metric] for r in results[a.id].test], float) for a in algorithms}
    scores["baseline"] = np.array([r[metric] for r in baseline.test], float)
    names = {a.id: a.label for a in algorithms} | {"baseline": "Línea base (clase más frecuente / media)"}
    order = sorted((k for k in scores if k != "baseline"), key=lambda k: np.nanmean(scores[k]), reverse=higher)
    best = order[0]
    label = metric_table(problem.task)[metric][0]
    out = Output(
        "Comparación de modelos",
        f"{label} · {len(algorithms)} modelo(s) · {int(p['folds'])} pliegues × {repeats} repetición(es) · n = {len(problem.X)}",
    )
    if skipped:
        out.alert("info", "No se incluyeron: " + "; ".join(skipped) + ".")
    out.tiles(
        [
            ("Mejor modelo", names[best]),
            (label.split(" (")[0], float(np.nanmean(scores[best]))),
            ("Línea base", float(np.nanmean(scores["baseline"]))),
        ]
    )
    rows, pvalues = [], {}
    for key in [*order, "baseline"]:
        values = scores[key]
        mean, sd = float(np.nanmean(values)), float(np.nanstd(values, ddof=1))
        if key in (best, "baseline"):
            diff = pvalue = None
        else:
            d = scores[best] - values if higher else values - scores[best]
            var = float(np.var(d, ddof=1))
            t = float(np.mean(d) / np.sqrt((1 / folds_total + ratio) * var)) if var > 0 else float("inf")
            diff, pvalue = float(np.mean(d)), P(float(2 * stats.t.sf(abs(t), folds_total - 1)))
            pvalues[key] = float(pvalue)
        seconds = float(np.mean(results[key].seconds)) if key in results else float(np.mean(baseline.seconds))
        row = [
            strong(names[key]) if key == best else names[key],
            mean,
            sd,
            float(np.nanmin(values)),
            float(np.nanmax(values)),
            diff,
            pvalue,
            seconds,
        ]
        if len(test):
            row.insert(5, held_out.get(key))
        rows.append(row)
    columns = [
        "Modelo",
        "Media",
        "Desv. est.",
        "Mínimo",
        "Máximo",
        "Diferencia con el mejor",
        "Valor p (t corregida)",
        "Tiempo por ajuste (s)",
    ]
    if len(test):
        columns.insert(5, "Prueba (reservada)")
    out.table(
        "Desempeño por modelo (validación cruzada)",
        columns,
        rows,
        note="Mismos pliegues para todos. La diferencia con el mejor se contrasta con la prueba t corregida de Nadeau y Bengio, que reconoce que los pliegues comparten datos; un valor p alto significa que no hay evidencia de que el mejor sea realmente mejor.",
    )
    close = [names[k] for k in order[1:] if pvalues.get(k, 0.0) > 0.05]
    if close:
        out.alert(
            "info",
            "No se distinguen estadísticamente del mejor: "
            + ", ".join(close)
            + ". Entre ellos conviene elegir el más simple, rápido o interpretable.",
        )
    out.chart(
        box_chart(
            f"{label.split(' (')[0]} en cada pliegue",
            label.split(" (")[0],
            [{"name": names[k], "values": scores[k]} for k in [*order, "baseline"]],
            points=True,
            height=420,
        )
    )
    if problem.task == CLASSIFICATION and problem.k == 2:
        series = []
        for key in order:
            scored = class_scores(results[key].oof, problem.k, problem.positive)
            if scored is None or np.isnan(scored).any():
                continue
            fpr, tpr, _ = roc_curve(y_train == problem.positive, scored)
            series.append({"name": names[key], "x": fpr, "y": tpr})
        if series:
            series.append({"name": "Azar", "x": [0, 1], "y": [0, 1], "dash": "dash"})
            out.chart(
                line_chart(
                    "Curvas ROC (predicciones de validación cruzada)",
                    "1 − especificidad",
                    "Sensibilidad",
                    series,
                    height=400,
                )
            )
    table = pd.DataFrame(
        [
            {"modelo": names[k], "pliegue": i + 1, **{m: r.get(m) for m in metric_table(problem.task)}}
            for k in [*order, "baseline"]
            for i, r in enumerate((results.get(k) or baseline).test)
        ]
    )
    out.dataset("pliegues", "Métricas por modelo y pliegue", table)
    return out.set(
        best=best,
        metric=metric,
        means={k: float(np.nanmean(v)) for k, v in scores.items()},
        sds={k: float(np.nanstd(v, ddof=1)) for k, v in scores.items()},
    ).build()


def _compare_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from sklearn.model_selection import cross_val_score\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
        "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\n\n"
        + load_line(dataset)
        + f"df = df.dropna(subset=[{roles['target']!r}])\nX = pd.get_dummies(df[{roles['features']!r}]).fillna(0)\ny = df[{roles['target']!r}]\n"
        "models = {'árbol': DecisionTreeClassifier(max_depth=5), 'bosque': RandomForestClassifier(200), 'logística': make_pipeline(StandardScaler(), LogisticRegression())}\n"
        f"for name, model in models.items():\n    scores = cross_val_score(model, X, y, cv={int(p['folds'])})\n    print(name, scores.mean(), scores.std())\n"
    )


# ============================================================================ tuning
def _literal(text: str, name: str) -> Any:
    low = text.lower()
    if low in ("none", "null", "ninguno", "sin"):
        return None
    if low in ("true", "verdadero"):
        return True
    if low in ("false", "falso"):
        return False
    if name == "hidden_layer_sizes":
        return parse_layers(text.replace("-", ","))
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def parse_grid(text: str, algorithm) -> dict[str, list[Any]]:
    """``max_depth = 3, 5, None; min_samples_leaf = 1, 10`` → ``{"max_depth": [3, 5, None], ...}``."""
    allowed = list(algorithm.defaults)
    space: dict[str, list[Any]] = {}
    for part in text.replace("\n", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise InvalidInputError("Escriba la rejilla como «ajuste = valor1, valor2; otro = valor1, valor2».")
        name, values = (s.strip() for s in part.split("=", 1))
        if name not in allowed:
            raise InvalidInputError(
                f"«{name}» no es un ajuste de «{algorithm.label}». Disponibles: {', '.join(allowed)}."
            )
        items = [v.strip() for v in values.split(",") if v.strip()]
        if not items or len(items) > 12:
            raise InvalidInputError(f"«{name}» necesita entre 1 y 12 valores.")
        space[name] = [_literal(v, name) for v in items]
    if not space:
        raise InvalidInputError("La rejilla está vacía.")
    return space


def _tuning(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    algorithm = get(p["algorithm"])
    problem = _problem(frame, roles, p, algorithm)
    seed, metric = int(p["seed"]), choose_metric(problem, p["metric"])
    higher = metric_table(problem.task)[metric][1]
    if algorithm.max_rows and len(problem.X) > algorithm.max_rows:
        raise InvalidInputError(f"Este modelo admite hasta {algorithm.max_rows} filas.")
    space = parse_grid(p["grid"], algorithm) if p["grid"] else tuning_space(algorithm, problem.task)
    names = list(space)
    grid = [dict(zip(names, combo, strict=True)) for combo in itertools.product(*space.values())]
    full = len(grid)
    limit = min(MAX_CONFIGS, int(p["n_iter"])) if p["search"] == "random" else MAX_CONFIGS
    sampled = full > limit
    if sampled:
        pick = np.random.default_rng(seed).choice(full, limit, replace=False)
        grid = [grid[i] for i in sorted(pick)]
    train, _ = holdout_split(problem.y, problem.task, float(p["test_size"]), seed)
    x_train, y_train = problem.X.iloc[train], problem.y[train]
    splits = make_splits(y_train, problem.task, int(p["folds"]), seed)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for config in grid:
            started = time.perf_counter()
            result = cross_validate(
                problem, lambda config=config: make_model(problem, algorithm, config, seed), x_train, y_train, splits
            )
            mean, sd = summarize(result.test, metric)
            rows.append(
                {
                    "config": config,
                    "mean": mean,
                    "sd": sd,
                    "train": summarize(result.train, metric)[0],
                    "seconds": time.perf_counter() - started,
                }
            )
    usable = [r for r in rows if np.isfinite(r["mean"])]
    if not usable:
        raise InvalidInputError("Ninguna combinación pudo evaluarse.")
    usable.sort(key=lambda r: r["mean"], reverse=higher)
    best = usable[0]
    label = _label(problem, metric)
    out = Output(
        f"Ajuste de hiperparámetros: {algorithm.label}",
        f"{'aleatoria' if p['search'] == 'random' else 'rejilla'} · {len(grid)} combinaciones × {len(splits)} pliegues · {label}",
    )
    if sampled:
        out.alert(
            "info", f"La rejilla completa tiene {full} combinaciones; se evaluó una muestra aleatoria de {len(grid)}."
        )
    out.tiles(
        [
            ("Mejor combinación", settings_text(best["config"])),
            (f"{label} (validación cruzada)", f"{best['mean']:.4g} ± {best['sd']:.2g}"),
            ("Combinaciones evaluadas", len(grid)),
        ]
    )
    simplest = [
        r
        for r in usable
        if (r["mean"] >= best["mean"] - best["sd"] if higher else r["mean"] <= best["mean"] + best["sd"])
    ]
    if len(simplest) > 1:
        out.alert(
            "info",
            f"{len(simplest) - 1} combinación(es) más quedan dentro de una desviación estándar de la mejor: la diferencia entre ellas puede ser ruido; prefiera la más simple.",
        )
    out.table(
        f"Las {min(20, len(usable))} mejores combinaciones",
        ["#", "Ajustes", f"{label} (media)", "Desv. est.", f"{label} en entrenamiento", "Tiempo (s)"],
        [
            [
                i + 1,
                settings_text(r["config"]),
                strong(r["mean"]) if i == 0 else r["mean"],
                r["sd"],
                r["train"],
                r["seconds"],
            ]
            for i, r in enumerate(usable[:20])
        ],
        note="Una gran distancia entre entrenamiento y validación indica sobreajuste de esa combinación.",
    )
    varied = [n for n in names if len({value_text(r["config"][n]) for r in usable}) > 1]
    for name in varied[:4]:
        levels = []
        for r in usable:
            text = value_text(r["config"][name])
            if text not in levels:
                levels.append(text)
        means = {lv: [r["mean"] for r in usable if value_text(r["config"][name]) == lv] for lv in levels}
        out.table(
            f"Efecto de «{name}» (promedio de las demás combinaciones)",
            [name, f"{label} (media)", "Mejor", "Combinaciones"],
            [[lv, float(np.mean(v)), float(max(v) if higher else min(v)), len(v)] for lv, v in means.items()],
        )
    if len(varied) == 1:
        name = varied[0]
        ordered = sorted({value_text(r["config"][name]): r for r in usable}.items(), key=lambda kv: str(kv[0]))
        numeric = all(
            isinstance(r["config"][name], int | float) and not isinstance(r["config"][name], bool) for r in usable
        )
        items = sorted(usable, key=lambda r: r["config"][name]) if numeric else [r for _, r in ordered]
        out.chart(
            line_chart(
                f"{label} según «{name}»",
                name,
                label,
                [
                    {
                        "name": "Validación cruzada",
                        "x": [value_text(r["config"][name]) if not numeric else r["config"][name] for r in items],
                        "y": [r["mean"] for r in items],
                        "markers": True,
                    }
                ],
                height=320,
            )
        )
    elif len(varied) >= 2:
        a, b = varied[:2]
        xs = list(dict.fromkeys(value_text(r["config"][a]) for r in usable))
        ys = list(dict.fromkeys(value_text(r["config"][b]) for r in usable))
        z = np.full((len(ys), len(xs)), np.nan)
        for r in usable:
            i, j = ys.index(value_text(r["config"][b])), xs.index(value_text(r["config"][a]))
            cell = z[i, j]
            z[i, j] = r["mean"] if np.isnan(cell) else (max(cell, r["mean"]) if higher else min(cell, r["mean"]))
        out.chart(
            heatmap_chart(
                f"{label}: «{a}» frente a «{b}» (la mejor de las demás combinaciones)",
                z,
                xs,
                ys,
                x_title=a,
                y_title=b,
                height=380,
            )
        )
    top = usable[:10]
    out.chart(
        bar_chart(
            f"{label} de las 10 mejores combinaciones",
            None,
            label,
            [f"#{i + 1}" for i in range(len(top))],
            [{"name": "Validación cruzada", "y": [r["mean"] for r in top]}],
            errors=[[r["sd"] for r in top]],
            height=320,
        )
    )
    out.heading("Modelo final con la mejor combinación")
    chosen = best["config"]
    settings = {**algorithm.defaults, **chosen}
    evaluation = evaluate(
        problem,
        lambda: make_model(problem, algorithm, chosen, seed),
        metric=metric,
        folds=int(p["folds"]),
        test_size=float(p["test_size"]),
        seed=seed,
    )
    final = report(frame, evaluation, algorithm, settings, p, title=out.build().title, detail=settings_text(chosen))
    final.blocks = out.build().blocks + final.blocks
    final.result.update(
        best={k: (list(v) if isinstance(v, tuple) else v) for k, v in chosen.items()},
        best_score=best["mean"],
        evaluated=len(grid),
        searched=names,
    )
    return final


def _tuning_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from sklearn.model_selection import GridSearchCV\nfrom sklearn.ensemble import RandomForestClassifier\n\n"
        + load_line(dataset)
        + f"df = df.dropna(subset=[{roles['target']!r}])\nX = pd.get_dummies(df[{roles['features']!r}]).fillna(0)\ny = df[{roles['target']!r}]\n"
        f"search = GridSearchCV(RandomForestClassifier(random_state={int(p['seed'])}), {{'max_depth': [3, 6, 10, None], 'min_samples_leaf': [1, 3, 10]}}, cv={int(p['folds'])})\n"
        "search.fit(X, y)\nprint(search.best_params_, search.best_score_)\n"
    )


# ============================================================================ learning curve
def _learning_curve(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    algorithm = get(p["algorithm"])
    problem = _problem(frame, roles, p, algorithm)
    seed, metric = int(p["seed"]), choose_metric(problem, p["metric"])
    higher = metric_table(problem.task)[metric][1]
    splits = make_splits(problem.y, problem.task, int(p["folds"]), seed)
    fractions = np.linspace(0.15, 1.0, int(p["points"]))
    rng = np.random.default_rng(seed)
    curve = {f: {"train": [], "test": []} for f in fractions}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train, test in splits:
            shuffled = rng.permutation(train)
            for f in fractions:
                size = max(int(round(f * len(train))), problem.k * 3 if problem.task == CLASSIFICATION else 10)
                rows = shuffled[:size]
                if problem.task == CLASSIFICATION and len(np.unique(problem.y[rows])) < problem.k:
                    continue
                model = make_model(problem, algorithm, {}, seed).fit(problem.X.iloc[rows], problem.y[rows])
                curve[f]["train"].append(
                    score_all(problem, problem.y[rows], predict_all(model, problem.X.iloc[rows], problem.task))[metric]
                )
                curve[f]["test"].append(
                    score_all(problem, problem.y[test], predict_all(model, problem.X.iloc[test], problem.task))[metric]
                )
    stat = {
        f: {k: (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in d.items()}
        for f, d in curve.items()
        if d["test"]
    }
    kept = [f for f in fractions if f in stat]
    if len(kept) < 3:
        raise InvalidInputError("Hay muy pocos datos para trazar la curva de aprendizaje.")
    label = _label(problem, metric)
    sizes = [int(round(f * np.mean([len(tr) for tr, _ in splits]))) for f in kept]
    train_mean = np.array([stat[f]["train"][0] for f in kept])
    test_mean = np.array([stat[f]["test"][0] for f in kept])
    test_sd = np.array([stat[f]["test"][1] for f in kept])
    train_sd = np.array([stat[f]["train"][1] for f in kept])
    out = Output(f"Curva de aprendizaje: {algorithm.label}", f"{label} · {len(splits)} pliegues · n = {len(problem.X)}")
    out.chart(
        line_chart(
            f"{label} según el tamaño del entrenamiento",
            "Filas de entrenamiento",
            label,
            [
                {"name": "Entrenamiento", "x": sizes, "y": train_mean, "markers": True},
                {"name": "Validación", "x": sizes, "y": test_mean, "markers": True},
            ],
            bands=[
                {
                    "name": "Entrenamiento ± 1 de",
                    "x": sizes,
                    "lower": train_mean - train_sd,
                    "upper": train_mean + train_sd,
                },
                {"name": "Validación ± 1 de", "x": sizes, "lower": test_mean - test_sd, "upper": test_mean + test_sd},
            ],
            height=400,
        )
    )
    gap = float(train_mean[-1] - test_mean[-1]) if higher else float(test_mean[-1] - train_mean[-1])
    slope = float(test_mean[-1] - test_mean[-2]) if higher else float(test_mean[-2] - test_mean[-1])
    noise = float(max(test_sd[-1], test_sd[-2], 1e-12))
    out.table(
        "Valores de la curva",
        [
            "Filas de entrenamiento",
            f"{label} (entrenamiento)",
            f"{label} (validación)",
            "Desv. est. (validación)",
            "Brecha",
        ],
        [
            [
                sizes[i],
                float(train_mean[i]),
                float(test_mean[i]),
                float(test_sd[i]),
                float(abs(train_mean[i] - test_mean[i])),
            ]
            for i in range(len(kept))
        ],
    )
    rows = []
    if slope > noise:
        rows.append(
            [
                "Más datos ayudarían",
                "La validación todavía mejora al final de la curva: conseguir más filas probablemente suba el desempeño.",
            ]
        )
    else:
        rows.append(["La curva se aplanó", "Más filas ya aportan poco; mejore las variables o el modelo."])
    scale = max(abs(float(test_mean[-1])), 1e-9)
    if gap > 0.15 * scale + noise:
        rows.append(
            [
                "Varianza alta (sobreajuste)",
                "Hay una brecha grande entre entrenamiento y validación: simplifique o regularice el modelo, o consiga más datos.",
            ]
        )
    elif gap <= 0.05 * scale + noise:
        rows.append(
            [
                "Sesgo posible",
                "Entrenamiento y validación se parecen: si el desempeño es insuficiente, el modelo es demasiado simple o faltan variables.",
            ]
        )
    out.table("Diagnóstico", ["Lectura", "Qué sugiere"], rows)
    return out.set(sizes=sizes, train=train_mean.tolist(), validation=test_mean.tolist(), gap=gap).build()


def _learning_curve_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import numpy as np\nfrom sklearn.model_selection import learning_curve\nfrom sklearn.ensemble import RandomForestClassifier\n\n"
        + load_line(dataset)
        + f"df = df.dropna(subset=[{roles['target']!r}])\nX = pd.get_dummies(df[{roles['features']!r}]).fillna(0)\ny = df[{roles['target']!r}]\n"
        f"sizes, train, test = learning_curve(RandomForestClassifier(random_state={int(p['seed'])}), X, y, cv={int(p['folds'])}, train_sizes=np.linspace(0.15, 1, {int(p['points'])}))\n"
        "print(sizes, train.mean(axis=1), test.mean(axis=1))\n"
    )


# ============================================================================ partial dependence
def _pdp(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    algorithm = get(p["algorithm"])
    problem = _problem(frame, roles, p, algorithm)
    seed, metric = int(p["seed"]), choose_metric(problem, p["metric"])
    focus = roles["focus"]
    missing = [f for f in focus if f not in problem.features]
    if missing:
        raise InvalidInputError(f"«{missing[0]}» debe estar entre las variables predictoras.")
    model = make_model(problem, algorithm, {}, seed)
    splits = make_splits(problem.y, problem.task, 3, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quality = cross_validate(
            problem, lambda: make_model(problem, algorithm, {}, seed), problem.X, problem.y, splits, keep_train=False
        )
        model.fit(problem.X, problem.y)
    cv_mean, cv_sd = summarize(quality.test, metric)
    label = _label(problem, metric)
    which = (
        f"probabilidad de «{problem.labels[problem.positive]}»"
        if problem.task == CLASSIFICATION
        else f"predicción de {problem.target}"
    )
    out = Output(
        f"Dependencia parcial: {algorithm.label}", f"{which} · {len(focus)} variable(s) · n = {len(problem.X)}"
    )
    out.tiles([(f"{label} del modelo (validación cruzada, 3 pliegues)", f"{cv_mean:.4g} ± {cv_sd:.2g}")])
    out.alert(
        "info",
        "La dependencia parcial promedia el efecto de la variable sobre todas las filas, dejando fijas las demás: describe lo que el modelo aprendió, no necesariamente una relación causal, y es poco fiable donde hay pocos datos o variables muy correlacionadas.",
    )

    def pick(values: Any) -> np.ndarray:
        """The response asked for: one slice for regression / binary (the probability of class 1), one per class otherwise."""
        array = np.asarray(values, float)
        if array.shape[0] > 1:
            return array[problem.positive]
        return 1 - array[0] if problem.task == CLASSIFICATION and problem.positive == 0 else array[0]

    rng = np.random.default_rng(seed)
    ice_n = min(int(p["ice_rows"]), len(problem.X))
    summary = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in focus:
            categorical = name in problem.categorical
            result = partial_dependence(
                model,
                problem.X,
                [name],
                categorical_features=[name] if categorical else None,
                kind="both" if (p["ice"] and not categorical) else "average",
                grid_resolution=int(p["grid"]),
            )
            grid = result["grid_values"][0]
            mean_curve = pick(result["average"])
            if categorical:
                out.chart(
                    bar_chart(
                        f"Dependencia parcial de «{name}»",
                        name,
                        which,
                        [str(g) for g in grid],
                        [{"name": "Promedio", "y": mean_curve}],
                        height=320,
                    )
                )
            else:
                traces = []
                if "individual" in result:
                    ind = pick(result["individual"])
                    for row in rng.choice(len(ind), ice_n, replace=False):
                        traces.append(
                            {
                                "type": "scatter",
                                "mode": "lines",
                                "x": grid,
                                "y": ind[row],
                                "line": {"color": "rgba(100,116,139,0.18)", "width": 1},
                                "hoverinfo": "skip",
                                "showlegend": False,
                            }
                        )
                traces.append(
                    {
                        "type": "scatter",
                        "mode": "lines+markers",
                        "x": grid,
                        "y": mean_curve,
                        "line": {"color": "#4f46e5", "width": 3},
                        "name": "Promedio (dependencia parcial)",
                    }
                )
                layout = {"xaxis": {"title": {"text": name}}, "yaxis": {"title": {"text": which}}, "showlegend": True}
                out.chart(
                    raw_chart(
                        traces,
                        layout,
                        f"Dependencia parcial de «{name}»"
                        + (" con curvas individuales (ICE)" if "individual" in result else ""),
                        height=380,
                    )
                )
            summary.append(
                [name, float(mean_curve.min()), float(mean_curve.max()), float(mean_curve.max() - mean_curve.min())]
            )
        if p["pair"] and len(focus) >= 2:
            a, b = focus[:2]
            if a in problem.categorical or b in problem.categorical:
                out.alert("info", "La superficie de dos variables solo se calcula para variables numéricas.")
            else:
                result = partial_dependence(
                    model, problem.X, [a, b], kind="average", grid_resolution=int(min(p["grid"], 20))
                )
                z = pick(result["average"])
                out.chart(
                    heatmap_chart(
                        f"Dependencia parcial conjunta de «{a}» y «{b}»",
                        z.T,
                        [f"{v:.3g}" for v in result["grid_values"][0]],
                        [f"{v:.3g}" for v in result["grid_values"][1]],
                        x_title=a,
                        y_title=b,
                        annotate=False,
                        height=420,
                    )
                )
    out.table(
        "Amplitud del efecto de cada variable",
        ["Variable", "Mínimo de la curva", "Máximo de la curva", "Amplitud"],
        sorted(summary, key=lambda r: -r[3]),
        note="Una curva casi plana significa que el modelo prácticamente no usa esa variable (promediando sobre las demás).",
    )
    return out.set(effects={r[0]: r[3] for r in summary}, cv=cv_mean).build()


def _pdp_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.inspection import PartialDependenceDisplay\n\n"
        + load_line(dataset)
        + f"df = df.dropna(subset=[{roles['target']!r}])\nX = df[{roles['features']!r}].select_dtypes('number').fillna(0)\ny = df[{roles['target']!r}]\n"
        f"model = RandomForestClassifier(random_state={int(p['seed'])}).fit(X, y)\nPartialDependenceDisplay.from_estimator(model, X, {roles['focus']!r}, kind='both')\n"
    )


# ============================================================================ variable selection
def _select_features(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    problem = _problem(frame, roles, p)
    seed = int(p["seed"])
    methods = p["methods"] or ["mutual_info", "anova", "forest", "lasso"]
    pipeline = make_model(problem, ALGORITHMS["penalized"], {}, seed)
    encoded = pipeline.named_steps["prep"].fit_transform(problem.X)
    info = encoded_info(pipeline, problem)
    raw = [i[0] for i in info]
    dummy_mask = np.array([i[2] is not None for i in info])
    variables = problem.features
    classification = problem.task == CLASSIFICATION

    def by_variable(values: np.ndarray, how=max) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for column, value in zip(raw, values, strict=True):
            grouped.setdefault(column, []).append(float(value))
        return {v: how(grouped[v]) for v in variables}

    scores: dict[str, dict[str, float]] = {}
    names = {
        "mutual_info": "Información mutua",
        "anova": "Prueba F (ANOVA / regresión)",
        "forest": "Bosque aleatorio (impureza)",
        "lasso": "Penalización L1",
        "rfe": "Eliminación recursiva (RFE)",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in methods:
            if method == "mutual_info":
                fn = mutual_info_classif if classification else mutual_info_regression
                scores[method] = by_variable(fn(encoded, problem.y, discrete_features=dummy_mask, random_state=seed))
            elif method == "anova":
                stat = (f_classif if classification else f_regression)(encoded, problem.y)[0]
                scores[method] = by_variable(np.nan_to_num(stat))
            elif method == "forest":
                forest = (RandomForestClassifier if classification else RandomForestRegressor)(
                    n_estimators=200, random_state=seed, n_jobs=1
                ).fit(encoded, problem.y)
                scores[method] = by_variable(forest.feature_importances_, how=sum)
            elif method == "lasso":
                if classification:
                    model = LogisticRegression(
                        l1_ratio=1.0, solver="saga", C=0.1, max_iter=3000, random_state=seed
                    ).fit(encoded, problem.y)
                    magnitude = np.abs(np.atleast_2d(model.coef_)).max(axis=0)
                else:
                    magnitude = np.abs(LassoCV(cv=5, random_state=seed).fit(encoded, problem.y).coef_)
                scores[method] = by_variable(magnitude)
            elif method == "rfe":
                estimator = LogisticRegression(max_iter=2000) if classification else Ridge()
                ranking = (
                    RFE(estimator, n_features_to_select=1, step=max(1, encoded.shape[1] // 40))
                    .fit(encoded, problem.y)
                    .ranking_
                )
                scores[method] = by_variable(1.0 / ranking)
    if not scores:
        raise InvalidInputError("Elija al menos un método.")
    ranks = {m: pd.Series(s).rank(ascending=False, method="min") for m, s in scores.items()}
    mean_rank = pd.concat(ranks, axis=1).mean(axis=1)
    keep = int(p["keep"]) if p["keep"] else max(1, int(np.ceil(len(variables) / 2)))
    keep = min(keep, len(variables))
    order = mean_rank.sort_values().index.tolist()
    chosen = order[:keep]
    out = Output(
        "Selección de variables", f"{len(variables)} variable(s) · {len(scores)} método(s) · se sugieren {keep}"
    )
    out.tiles(
        [
            ("Variables sugeridas", ", ".join(chosen) if len(chosen) <= 6 else f"{len(chosen)} de {len(variables)}"),
            ("Métodos", len(scores)),
        ]
    )
    normalized = {m: {v: (s[v] / max(max(s.values()), 1e-12)) for v in variables} for m, s in scores.items()}
    out.table(
        "Ranking de las variables",
        ["Variable", *[names[m] for m in scores], "Rango medio", "¿Sugerida?"],
        [
            [v, *(normalized[m][v] for m in scores), float(mean_rank[v]), good("sí") if v in chosen else muted("no")]
            for v in order
        ],
        note="Cada método se escala a 1 para su mejor variable. El rango medio promedia las posiciones de la variable en cada método; los métodos no coinciden siempre y eso es informativo: las que quedan arriba en todos son las más seguras.",
    )
    out.chart(
        heatmap_chart(
            "Importancia relativa por método",
            [[normalized[m][v] for m in scores] for v in order],
            [names[m] for m in scores],
            order,
            annotate=True,
            height=max(300, 26 * len(order) + 120),
        )
    )
    kept = pd.DataFrame(
        {
            "variable": order,
            "rango_medio": [float(mean_rank[v]) for v in order],
            "sugerida": [float(v in chosen) for v in order],
        }
    )
    out.dataset("seleccion", "Ranking de variables", kept)
    return out.set(selected=chosen, ranking=order, mean_rank={v: float(mean_rank[v]) for v in order}).build()


def _select_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from sklearn.feature_selection import mutual_info_classif\n\n"
        + load_line(dataset)
        + f"df = df.dropna(subset=[{roles['target']!r}])\nX = pd.get_dummies(df[{roles['features']!r}]).fillna(0)\ny = df[{roles['target']!r}]\n"
        "print(pd.Series(mutual_info_classif(X, y, random_state=0), index=X.columns).sort_values(ascending=False))\n"
    )


# ============================================================================ anomaly detection
_ANOMALY = (
    ("iforest", "Bosque de aislamiento (Isolation Forest)"),
    ("lof", "Factor de atipicidad local (LOF)"),
    ("ocsvm", "SVM de una clase"),
    ("elliptic", "Envolvente elíptica (covarianza robusta)"),
)


def _anomaly(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names = roles["vars"]
    data = complete(frame, names, 20)
    values = data.to_numpy(float)
    if p["standardize"]:
        values = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
    n, seed, contamination = len(values), int(p["seed"]), float(p["contamination"])
    method = p["method"]
    if method == "ocsvm" and n > 20_000:
        raise InvalidInputError("La SVM de una clase admite hasta 20 000 filas; use el bosque de aislamiento.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "iforest":
            model = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(values)
            score, flagged = -model.score_samples(values), model.predict(values) == -1
        elif method == "lof":
            model = LocalOutlierFactor(n_neighbors=int(min(p["neighbors"], n - 1)), contamination=contamination)
            flagged = model.fit_predict(values) == -1
            score = -model.negative_outlier_factor_
        elif method == "ocsvm":
            model = OneClassSVM(nu=contamination, gamma="scale").fit(values)
            score, flagged = -model.score_samples(values), model.predict(values) == -1
        else:
            model = EllipticEnvelope(contamination=contamination, random_state=seed).fit(values)
            score, flagged = -model.score_samples(values), model.predict(values) == -1
    count = int(flagged.sum())
    threshold = float(score[flagged].min()) if count else float(score.max())
    out = Output(
        f"Detección de anomalías: {dict(_ANOMALY)[method].split(' (')[0].lower()}",
        f"{len(names)} variables · n = {n} · contaminación esperada {contamination:.1%}",
    )
    dropped_alert(out, frame, data)
    out.tiles([("Observaciones", n), ("Marcadas como anómalas", count), ("Puntaje de corte", threshold)])
    out.alert(
        "info",
        "Sin etiquetas no hay verdad contra la cual validar: la «contaminación» es la fracción que usted espera que sea anómala y fija el corte. Revise las marcadas con criterio del negocio antes de excluirlas.",
    )
    out.chart(
        hist_chart(
            "Distribución del puntaje de anomalía (mayor = más raro)",
            "Puntaje",
            score,
            vlines=[{"value": threshold, "label": "corte", "dash": "dash"}],
            norm="count",
            height=320,
        )
    )
    if values.shape[1] >= 2:
        points, share = _projection(values)
        shown = (
            np.arange(n) if n <= MAX_POINTS else np.sort(np.random.default_rng(0).choice(n, MAX_POINTS, replace=False))
        )
        normal, odd = shown[~flagged[shown]], shown[flagged[shown]]
        out.chart(
            scatter_chart(
                f"Observaciones sobre los dos primeros componentes principales ({100 * share.sum():.0f} % de la varianza)",
                "Componente 1",
                "Componente 2",
                [
                    {"name": "Normales", "x": points[normal, 0], "y": points[normal, 1]},
                    {"name": "Anómalas", "x": points[odd, 0], "y": points[odd, 1]},
                ],
                height=420,
            )
        )
    top = np.argsort(-score)[:15]
    out.table(
        "Observaciones más anómalas",
        ["Fila", "Puntaje", "¿Marcada?", *names[:8]],
        [
            [
                int(data.index[i]) + 1,
                float(score[i]),
                bad("sí") if flagged[i] else muted("no"),
                *(float(data.iloc[i][c]) for c in names[:8]),
            ]
            for i in top
        ],
    )
    out.dataset(
        "anomalias",
        "Puntaje de anomalía de cada fila",
        _result_frame(frame, data.index, puntaje_anomalia=score, anomala=flagged.astype(float)),
    )
    return out.set(
        flagged=count, threshold=threshold, rows=[int(data.index[i]) + 1 for i in np.where(flagged)[0][:50]]
    ).build()


def _anomaly_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from sklearn.ensemble import IsolationForest\nfrom sklearn.preprocessing import StandardScaler\n\n"
        + load_line(dataset)
        + f"X = df[{roles['vars']!r}].dropna()\n"
        + ("X = StandardScaler().fit_transform(X)\n" if p["standardize"] else "")
        + f"model = IsolationForest(contamination={float(p['contamination'])}, random_state={int(p['seed'])}).fit(X)\nprint((model.predict(X) == -1).sum())\n"
    )


# ============================================================================ evaluate predictions
def _evaluate(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    actual_name, predicted_name, score_name = roles["actual"], roles["predicted"], roles.get("score")
    if actual_name == predicted_name:
        raise InvalidInputError("Elija dos columnas distintas: la real y la predicha.")
    columns = [actual_name, predicted_name, *([score_name] if score_name else [])]
    data = frame[columns].dropna()
    if len(data) < 10:
        raise InvalidInputError(f"Se necesitan al menos 10 filas completas (hay {len(data)}).")
    task = p["task"]
    if task == "auto":
        numeric = all(
            pd.api.types.is_numeric_dtype(data[c]) and not pd.api.types.is_bool_dtype(data[c])
            for c in (actual_name, predicted_name)
        )
        task = REGRESSION if numeric and data[actual_name].nunique() > 2 else CLASSIFICATION
    out = Output(
        "Evaluación de predicciones",
        f"real: {actual_name} · predicho: {predicted_name} · n = {len(data)}"
        + (f" · puntaje: {score_name}" if score_name else ""),
    )
    if len(frame) > len(data):
        out.alert("info", f"Se omitieron {len(frame) - len(data)} fila(s) con datos faltantes.")
    if task == REGRESSION:
        y = pd.to_numeric(data[actual_name], errors="coerce").to_numpy(float)
        guess = pd.to_numeric(data[predicted_name], errors="coerce").to_numpy(float)
        if np.isnan(y).any() or np.isnan(guess).any():
            raise InvalidInputError("Para regresión, el valor real y el predicho deben ser numéricos.")
        scores = regression_scores(y, Pred(guess))
        names = {"rmse": "RMSE", "mae": "MAE", "r2": "R²", "mape": "MAPE (%)", "medae": "Mediana del error absoluto"}
        out.tiles(
            [
                ("R²", scores["r2"]),
                ("RMSE", scores["rmse"]),
                ("MAE", scores["mae"]),
                ("Sesgo (error medio)", float(np.mean(y - guess))),
            ]
        )
        out.table("Métricas", ["Métrica", "Valor"], [[names[k], v] for k, v in scores.items() if np.isfinite(v)])
        regression_section(out, SimpleNamespace(), y, guess, "predicciones evaluadas")
        mean, diff = (y + guess) / 2, y - guess
        limit = 1.96 * float(np.std(diff, ddof=1))
        shown = (
            np.arange(len(y))
            if len(y) <= MAX_POINTS
            else np.sort(np.random.default_rng(0).choice(len(y), MAX_POINTS, replace=False))
        )
        out.chart(
            scatter_chart(
                "Gráfico de Bland-Altman (acuerdo entre lo real y lo predicho)",
                "Promedio de real y predicho",
                "Real − predicho",
                [{"name": "Observaciones", "x": mean[shown], "y": diff[shown]}],
                hlines=[
                    {"value": float(diff.mean()), "label": "sesgo"},
                    {"value": float(diff.mean()) + limit, "label": "+1.96 de", "dash": "dash"},
                    {"value": float(diff.mean()) - limit, "label": "−1.96 de", "dash": "dash"},
                ],
                height=360,
            ),
            caption="El 95 % de las diferencias debería caer entre las líneas punteadas.",
        )
        return out.set(task=task, **scores).build()
    labels = [str(v) for v in level_order(pd.concat([data[actual_name], data[predicted_name]], ignore_index=True))]
    code = {v: i for i, v in enumerate(labels)}
    y = data[actual_name].astype(str).map(code).to_numpy(int)
    guess = data[predicted_name].astype(str).map(code).to_numpy(int)
    k = len(labels)
    if k < 2:
        raise InvalidInputError("Hay una sola categoría: no hay nada que evaluar.")
    positive = labels.index(p["positive"]) if p.get("positive") in labels else k - 1
    if p.get("positive") and p["positive"] not in labels:
        raise InvalidInputError(f"«{p['positive']}» no es una categoría de los datos.")
    problem = SimpleNamespace(labels=labels, k=k, positive=positive, task=CLASSIFICATION)
    pred = Pred(guess)
    event_scores = None
    if score_name:
        if k != 2:
            raise InvalidInputError("El puntaje solo se usa con dos categorías.")
        event_scores = pd.to_numeric(data[score_name], errors="coerce").to_numpy(float)
        if np.isnan(event_scores).any():
            raise InvalidInputError("La columna del puntaje debe ser numérica.")
        pred = Pred(guess, event_scores if positive == 1 else -event_scores, False)
    result = classification_scores(y, pred, k, positive)
    result["mcc"] = float(matthews_corrcoef(y, guess))
    tiles = [
        ("Exactitud", result["accuracy"]),
        ("Exactitud balanceada", result["balanced_accuracy"]),
        ("Kappa de Cohen", result["kappa"]),
        ("Coeficiente de Matthews", result["mcc"]),
    ]
    if np.isfinite(result["auc"]):
        tiles.append(("AUC", result["auc"]))
    out.tiles(tiles)
    matrix = confusion_section(out, problem, y, guess, "Matriz de confusión")
    class_report(out, problem, matrix)
    info: dict[str, Any] = {}
    if score_name:
        info = curves_section(out, problem, y, pred, "puntajes evaluados")
        if k == 2:
            area, se, low, high = delong_auc(y == positive, event_scores)
            out.table(
                "AUC con intervalo de confianza (DeLong)",
                ["AUC", "Error estándar", "Límite inferior 95 %", "Límite superior 95 %"],
                [[area, se, low, high]],
            )
    values = {**result, **{k_: v for k_, v in info.items() if k_ != "auc_ci"}}
    return out.set(
        task=task, confusion=matrix.tolist(), **{k_: v for k_, v in values.items() if np.isfinite(v)}
    ).build()


def _evaluate_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score\n\n"
        + load_line(dataset)
        + f"d = df[[{roles['actual']!r}, {roles['predicted']!r}]].dropna()\nprint(confusion_matrix(d[{roles['actual']!r}], d[{roles['predicted']!r}]))\nprint(classification_report(d[{roles['actual']!r}], d[{roles['predicted']!r}]))\n"
    )


# ============================================================================ the catalogue
_NUMBER = "number"
_GRID_HELP = "Opcional. Una rejilla propia: «ajuste = valor1, valor2; otro = valor1, valor2» (hasta 12 valores por ajuste y 60 combinaciones). Vacío = la rejilla recomendada del modelo."
_GRID_REFERENCE = tuple((a.label, ", ".join(a.defaults)) for a in ALGORITHMS.values())

SPECS = (
    _spec("compare", "Comparar modelos", GROUP_COMPARE, "Varios modelos con los mismos pliegues: media y dispersión, prueba t corregida contra el mejor, ROC superpuestas y tiempos.", _compare, roles=(TARGET, FEATURES), params=(FieldSpec("algorithms", "Modelos a comparar", "multiselect", ["tree", "forest", "boosting", "knn", "penalized"], algorithm_options()), FieldSpec("repeats", "Repeticiones de la validación cruzada", "integer", 1, minimum=1, maximum=5, help="Más repeticiones dan comparaciones más estables."), TASK_PARAM, FOLDS, TEST_SIZE, METRIC, MISSING, POSITIVE, SEED), outputs=(OutputSpec("pliegues", "Métricas por modelo y pliegue"),), code=_compare_code, example={"roles": {"target": "b", "features": ["y", "x1", "x2", "g"]}, "params": {"algorithms": ["tree", "penalized", "knn"], "folds": 3}}),
    _spec("tuning", "Ajuste de hiperparámetros (rejilla o aleatorio)", GROUP_COMPARE, "Busca la mejor configuración de un modelo con validación cruzada: ranking, efecto de cada ajuste y modelo final evaluado.", _tuning, roles=(TARGET, FEATURES), params=(ALGORITHM, FieldSpec("search", "Tipo de búsqueda", "select", "grid", (("grid", "Rejilla completa"), ("random", "Aleatoria"))), FieldSpec("n_iter", "Combinaciones (búsqueda aleatoria)", "integer", 20, minimum=2, maximum=MAX_CONFIGS), FieldSpec("grid", "Rejilla propia", "text", optional=True, help=_GRID_HELP), TASK_PARAM, FOLDS, TEST_SIZE, METRIC, MISSING, POSITIVE, SEED), outputs=(PREDICTIONS,), code=_tuning_code, reference=_GRID_REFERENCE, example={"roles": {"target": "b", "features": ["y", "x1", "x2"]}, "params": {"algorithm": "tree", "folds": 3}}),
    _spec("learning_curve", "Curva de aprendizaje", GROUP_COMPARE, "Cómo cambia el desempeño con el tamaño del entrenamiento: diagnostica sesgo, varianza y si más datos ayudarían.", _learning_curve, roles=(TARGET, FEATURES), params=(ALGORITHM, FieldSpec("points", "Tamaños a probar", "integer", 6, minimum=3, maximum=12), *MODEL_PARAMS), code=_learning_curve_code, example={"roles": {"target": "y", "features": ["x1", "x2", "x3"]}, "params": {"algorithm": "tree", "folds": 3, "points": 4}}),
    _spec("pdp", "Dependencia parcial y curvas ICE", GROUP_EXPLAIN, "Cómo cambia la predicción al variar una o dos variables: promedio (PDP), curvas individuales (ICE) y superficie conjunta.", _pdp, roles=(TARGET, FEATURES, RoleSpec("focus", "Variables a explicar (entre las predictoras)", "any", multiple=True, min_count=1, max_count=6)), params=(ALGORITHM, FieldSpec("ice", "Mostrar curvas individuales (ICE)", "boolean", True), FieldSpec("ice_rows", "Curvas individuales a dibujar", "integer", 40, minimum=5, maximum=200), FieldSpec("pair", "Superficie conjunta de las dos primeras", "boolean", False), FieldSpec("grid", "Puntos de la malla", "integer", 25, minimum=5, maximum=60), TASK_PARAM, METRIC, MISSING, POSITIVE, SEED), code=_pdp_code, example={"roles": {"target": "b", "features": ["y", "x1", "x2", "g"], "focus": ["x1", "x2"]}, "params": {"algorithm": "boosting", "pair": True}}),
    _spec("evaluate", "Evaluar predicciones existentes", GROUP_EXPLAIN, "Desempeño de predicciones ya hechas (de este programa o de otro): matriz de confusión, ROC con IC de DeLong, o error y Bland-Altman.", _evaluate, heavy=False, roles=(RoleSpec("actual", "Valor real", "any"), RoleSpec("predicted", "Valor predicho", "any"), RoleSpec("score", "Puntaje o probabilidad del evento (opcional, solo 2 clases)", "numeric", optional=True)), params=(TASK_PARAM, POSITIVE), code=_evaluate_code, example={"roles": {"actual": "b", "predicted": "c1", "score": "pb"}, "params": {}}),
    _spec("select_features", "Selección de variables", GROUP_DATA, "Ordena las predictoras con varios métodos (información mutua, F, bosque, L1, RFE) y sugiere cuáles conservar.", _select_features, roles=(TARGET, FEATURES), params=(FieldSpec("methods", "Métodos", "multiselect", ["mutual_info", "anova", "forest", "lasso"], (("mutual_info", "Información mutua"), ("anova", "Prueba F (ANOVA / regresión)"), ("forest", "Bosque aleatorio (impureza)"), ("lasso", "Penalización L1"), ("rfe", "Eliminación recursiva (RFE)"))), FieldSpec("keep", "Variables a sugerir (vacío = la mitad)", "integer", optional=True, minimum=1, maximum=100), TASK_PARAM, MISSING, POSITIVE, SEED), outputs=(OutputSpec("seleccion", "Ranking de variables"),), code=_select_code, example={"roles": {"target": "b", "features": ["y", "x1", "x2", "x3", "g"]}, "params": {"methods": ["mutual_info", "anova", "forest", "lasso", "rfe"]}}),
    _spec("anomaly", "Detección de anomalías", GROUP_DATA, "Marca observaciones raras sin etiquetas: bosque de aislamiento, LOF, SVM de una clase o envolvente elíptica.", _anomaly, roles=(VARS,), params=(FieldSpec("method", "Método", "select", "iforest", _ANOMALY), FieldSpec("contamination", "Fracción esperada de anomalías", _NUMBER, 0.05, minimum=0.001, maximum=0.5), FieldSpec("neighbors", "Vecinos (LOF)", "integer", 20, minimum=3, maximum=200, visible_if=("method", ("lof",))), FieldSpec("standardize", "Estandarizar las variables", "boolean", True), SEED), outputs=(OutputSpec("anomalias", "Puntaje de anomalía de cada fila"),), code=_anomaly_code, example={"roles": {"vars": ["y", "x1", "x2"]}, "params": {"method": "lof"}}),
)  # fmt: skip

registry.register(*SPECS)
