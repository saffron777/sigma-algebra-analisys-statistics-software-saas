"""The machine-learning engine shared by every supervised analysis: data preparation, validation, metrics and the common report.

A supervised analysis only says which estimator to fit and what to add to the report; this module does the rest the same
way for all of them: it holds out a test set, scores the model with cross-validation on the rest (with a naive baseline
on the same folds), explains the errors (confusion matrix, ROC, residuals), ranks the variables by permutation
importance and offers the predictions as a dataset.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    log_loss,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sigma.domain.analysis.charts import bar_chart, heatmap_chart, line_chart, scatter_chart
from sigma.domain.analysis.output import AnalysisOutput, Output, good, muted, strong
from sigma.domain.analysis.spec import FieldSpec, OutputSpec, RoleSpec
from sigma.domain.errors import InvalidInputError

MODULE = "aprendizaje"
MAX_ROWS = 200_000
MAX_FEATURES = 200
MAX_LEVELS = 50
MAX_CLASSES = 30
MAX_DUMMIES = 600
MAX_POINTS = 4000
MIN_ROWS = 30

CLASSIFICATION, REGRESSION = "classification", "regression"
TASK_NAMES = {CLASSIFICATION: "clasificación", REGRESSION: "regresión"}

# metric -> (label, higher is better)
CLS_METRICS = {
    "accuracy": ("Exactitud", True),
    "balanced_accuracy": ("Exactitud balanceada", True),
    "f1_macro": ("F1 (promedio de clases)", True),
    "precision_macro": ("Precisión (promedio de clases)", True),
    "recall_macro": ("Exhaustividad (promedio de clases)", True),
    "auc": ("AUC (área bajo la curva ROC)", True),
    "log_loss": ("Pérdida logarítmica", False),
    "kappa": ("Kappa de Cohen", True),
}
REG_METRICS = {
    "rmse": ("RMSE (raíz del error cuadrático medio)", False),
    "mae": ("MAE (error absoluto medio)", False),
    "r2": ("R²", True),
    "mape": ("MAPE (error porcentual absoluto, %)", False),
    "medae": ("Mediana del error absoluto", False),
}
METRIC_OPTIONS = (
    ("auto", "Automática (exactitud o RMSE)"),
    *((k, f"{v[0]} (clasificación)") for k, v in CLS_METRICS.items() if k not in ("log_loss",)),
    *((k, f"{v[0].split(' (')[0]} (regresión)") for k, v in REG_METRICS.items() if k != "medae"),
)


@dataclass(frozen=True)
class Algorithm:
    """An estimator family: how to build it for a task, its default settings and the search space used for tuning."""

    id: str
    label: str
    build: Callable[[str, dict[str, Any], int], Any]
    defaults: dict[str, Any]
    space: dict[str, list[Any]]
    scaled: bool = False  # distance- or gradient-based models need standardised numeric inputs
    tasks: tuple[str, ...] = (CLASSIFICATION, REGRESSION)
    max_rows: int | None = None  # kernel methods cost more than O(n²)


# ============================================================================ the problem
@dataclass
class Problem:
    target: str
    task: str
    X: pd.DataFrame  # the labelled rows
    y: np.ndarray  # class codes 0..k-1 or the numeric target
    labels: list[str]  # class names
    numeric: list[str]
    categorical: list[str]
    index: pd.Index  # frame index of the labelled rows
    X_new: pd.DataFrame  # rows without a target value, to predict
    new_index: pd.Index
    dropped: int  # rows left out for lack of data
    positive: int = 1  # which class counts as the event in binary problems

    @property
    def k(self) -> int:
        return len(self.labels)

    @property
    def features(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    @property
    def y_names(self) -> np.ndarray:
        return np.array(self.labels)[self.y] if self.task == CLASSIFICATION else self.y


def detect_task(series: pd.Series, requested: str) -> str:
    if requested != "auto":
        return requested
    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    if not numeric:
        return CLASSIFICATION
    return CLASSIFICATION if series.dropna().nunique() <= 2 else REGRESSION


def _natural_order(values: list[Any]) -> list[Any]:
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def build_problem(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    *,
    task: str = "auto",
    missing: str = "impute",
    predict_new: bool = True,
    positive: str | None = None,
    allowed: tuple[str, ...] = (CLASSIFICATION, REGRESSION),
) -> Problem:
    if target in features:
        raise InvalidInputError(f"«{target}» es el objetivo: no puede ser también una variable predictora.")
    if len(features) > MAX_FEATURES:
        raise InvalidInputError(f"Se admiten como máximo {MAX_FEATURES} variables predictoras.")
    y_raw = frame[target]
    task = detect_task(y_raw, task)
    if task not in allowed:
        raise InvalidInputError(f"Este modelo solo sirve para {' y '.join(TASK_NAMES[t] for t in allowed)}.")
    numeric, categorical = [], []
    for name in features:
        column = frame[name]
        if pd.api.types.is_datetime64_any_dtype(column):
            raise InvalidInputError(f"«{name}» es una fecha: conviértala en números (año, mes…) con Transformar datos.")
        if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
            numeric.append(name)
        else:
            categorical.append(name)
    x = frame[features].copy()
    for name in numeric:
        x[name] = pd.to_numeric(x[name], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    dummies = 0
    for name in categorical:
        text = x[name].astype(object)
        x[name] = text.where(text.isna(), text.astype(str)).where(text.notna(), np.nan)
        levels = x[name].nunique()
        if levels > MAX_LEVELS:
            raise InvalidInputError(
                f"«{name}» tiene {levels} categorías (el máximo es {MAX_LEVELS}): quítela o agrupe sus categorías."
            )
        dummies += levels
    if dummies > MAX_DUMMIES:
        raise InvalidInputError("Las variables categóricas generan demasiadas columnas; elija menos o agrúpelas.")
    empty = [c for c in features if x[c].notna().sum() == 0]
    if empty:
        raise InvalidInputError(f"«{empty[0]}» no tiene ningún valor.")

    if task == REGRESSION:
        if not pd.api.types.is_numeric_dtype(y_raw) or pd.api.types.is_bool_dtype(y_raw):
            raise InvalidInputError(f"«{target}» no es numérica: elija clasificación.")
        y_all = pd.to_numeric(y_raw, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    else:
        y_all = y_raw.astype(object).where(y_raw.notna(), np.nan)
    has_target = y_all.notna()
    incomplete = x.isna().any(axis=1) if missing == "drop" else pd.Series(False, index=x.index)
    labelled = has_target & ~incomplete
    unlabeled = ~has_target & ~incomplete if predict_new else pd.Series(False, index=x.index)
    dropped = int((~labelled & ~unlabeled).sum())
    if labelled.sum() < MIN_ROWS:
        raise InvalidInputError(
            f"Se necesitan al menos {MIN_ROWS} filas con el objetivo y las variables completas (hay {int(labelled.sum())})."
        )
    if labelled.sum() > MAX_ROWS:
        raise InvalidInputError(f"Se admiten hasta {MAX_ROWS:,} filas; muestree el dataset antes.".replace(",", " "))

    labels: list[str] = []
    positive_index = 1
    if task == CLASSIFICATION:
        levels = _natural_order(list(pd.unique(y_all[labelled])))
        if len(levels) < 2:
            raise InvalidInputError(f"«{target}» tiene una sola categoría: no hay nada que clasificar.")
        if len(levels) > MAX_CLASSES:
            raise InvalidInputError(
                f"«{target}» tiene {len(levels)} valores distintos (máximo {MAX_CLASSES} clases): ¿es una regresión?"
            )
        labels = [str(v) for v in levels]
        code = {v: i for i, v in enumerate(levels)}
        y = y_all[labelled].map(code).to_numpy(int)
        counts = np.bincount(y, minlength=len(labels))
        if counts.min() < 2:
            raise InvalidInputError(f"La clase «{labels[int(counts.argmin())]}» tiene una sola observación.")
        if positive:
            if positive not in labels:
                raise InvalidInputError(f"«{positive}» no es una categoría de «{target}».")
            positive_index = labels.index(positive)
        else:
            positive_index = len(labels) - 1
    else:
        y = y_all[labelled].to_numpy(float)
        if np.std(y) == 0:
            raise InvalidInputError(f"«{target}» es constante: no hay nada que predecir.")
    return Problem(
        target=target,
        task=task,
        X=x[labelled],
        y=y,
        labels=labels,
        numeric=numeric,
        categorical=categorical,
        index=x.index[labelled],
        X_new=x[unlabeled],
        new_index=x.index[unlabeled],
        dropped=dropped,
        positive=positive_index,
    )


def preprocessor(problem: Problem, scale: bool) -> ColumnTransformer:
    """Median/mode imputation, optional standardisation and one-hot encoding — fitted inside every fold (no leakage)."""
    steps = []
    if problem.numeric:
        numeric = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
        steps.append(("num", make_pipeline(*numeric), problem.numeric))
    if problem.categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        steps.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), encoder), problem.categorical))
    return ColumnTransformer(steps, verbose_feature_names_out=False)


def make_model(problem: Problem, algorithm: Algorithm, hyper: dict[str, Any], seed: int) -> Pipeline:
    settings = {**algorithm.defaults, **hyper}
    return Pipeline(
        [
            ("prep", preprocessor(problem, algorithm.scaled)),
            ("model", algorithm.build(problem.task, settings, seed)),
        ]
    )


# ============================================================================ metrics
@dataclass
class Pred:
    values: np.ndarray  # class codes or numbers
    scores: np.ndarray | None = None  # probabilities (n, k) or decision values
    proba: bool = False


def predict_all(model: Any, x: pd.DataFrame, task: str) -> Pred:
    values = np.asarray(model.predict(x))
    if task == REGRESSION:
        return Pred(values.astype(float))
    if hasattr(model, "predict_proba"):
        return Pred(values.astype(int), np.asarray(model.predict_proba(x), float), True)
    if hasattr(model, "decision_function"):
        return Pred(values.astype(int), np.asarray(model.decision_function(x), float), False)
    return Pred(values.astype(int))


def class_scores(pred: Pred, k: int, positive: int) -> np.ndarray | None:
    """Binary: the score of the positive class (n,). Multiclass: (n, k). ``None`` when the model gives no scores."""
    if pred.scores is None:
        return None
    s = pred.scores
    if k == 2:
        if s.ndim == 2:
            return s[:, positive]
        return s if positive == 1 else -s
    return s if s.ndim == 2 else None


def auc_value(y: np.ndarray, scores: np.ndarray | None, k: int, positive: int) -> float:
    if scores is None:
        return float("nan")
    try:
        if k == 2:
            return float(roc_auc_score(y == positive, scores))
        present = [c for c in range(k) if 0 < (y == c).sum() < len(y)]
        return float(np.mean([roc_auc_score(y == c, scores[:, c]) for c in present])) if present else float("nan")
    except ValueError:
        return float("nan")


def delong_auc(event: np.ndarray, scores: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """AUC with the DeLong standard error and confidence interval: (auc, se, lower, upper)."""
    from scipy.stats import norm, rankdata

    event = np.asarray(event, bool)
    positive, negative = scores[event], scores[~event]
    m, n = len(positive), len(negative)
    if m < 2 or n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pooled = rankdata(np.concatenate([positive, negative]))
    area = float((pooled[:m].sum() - m * (m + 1) / 2) / (m * n))
    placement_pos = (pooled[:m] - rankdata(positive)) / n
    placement_neg = 1 - (pooled[m:] - rankdata(negative)) / m
    se = float(np.sqrt(placement_pos.var(ddof=1) / m + placement_neg.var(ddof=1) / n))
    z = float(norm.ppf(1 - alpha / 2))
    return area, se, max(0.0, area - z * se), min(1.0, area + z * se)


def classification_scores(y: np.ndarray, pred: Pred, k: int, positive: int) -> dict[str, float]:
    p = pred.values
    result: dict[str, float] = {
        "accuracy": float((p == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
    }
    precision, recall, f1 = [], [], []
    for c in range(k):
        tp = float(((p == c) & (y == c)).sum())
        pp, ap = float((p == c).sum()), float((y == c).sum())
        if ap == 0 and pp == 0:
            continue
        pr = tp / pp if pp else 0.0
        rc = tp / ap if ap else 0.0
        precision.append(pr)
        recall.append(rc)
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    result["precision_macro"], result["recall_macro"], result["f1_macro"] = (
        float(np.mean(v)) for v in (precision, recall, f1)
    )
    result["kappa"] = float(cohen_kappa_score(y, p))
    result["auc"] = auc_value(y, class_scores(pred, k, positive), k, positive)
    if pred.proba and pred.scores is not None:
        try:
            result["log_loss"] = float(log_loss(y, np.clip(pred.scores, 1e-15, 1), labels=list(range(k))))
        except ValueError:
            result["log_loss"] = float("nan")
    else:
        result["log_loss"] = float("nan")
    return result


def regression_scores(y: np.ndarray, pred: Pred) -> dict[str, float]:
    p = pred.values
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
        "mape": float(100 * np.mean(np.abs((y - p) / y))) if np.all(y != 0) else float("nan"),
        "medae": float(median_absolute_error(y, p)),
    }


def score_all(problem: Problem, y: np.ndarray, pred: Pred) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if problem.task == CLASSIFICATION:
            return classification_scores(y, pred, problem.k, problem.positive)
        return regression_scores(y, pred)


def metric_table(task: str) -> dict[str, tuple[str, bool]]:
    return CLS_METRICS if task == CLASSIFICATION else REG_METRICS


def choose_metric(problem: Problem, requested: str) -> str:
    """The primary metric: the user's, or accuracy (balanced when the classes are uneven) / RMSE."""
    table = metric_table(problem.task)
    if requested in table:
        return requested
    if requested != "auto" and requested in {**CLS_METRICS, **REG_METRICS}:
        raise InvalidInputError(
            f"La métrica «{requested}» no sirve para {TASK_NAMES[problem.task]}: elija otra o la automática."
        )
    if problem.task == REGRESSION:
        return "rmse"
    counts = np.bincount(problem.y, minlength=problem.k)
    return "accuracy" if counts.max() <= 1.5 * counts.min() else "balanced_accuracy"


def scorer_for(problem: Problem, metric: str, has_proba: bool) -> Any:
    """A scikit-learn scorer matching the metric (for permutation importance and searches)."""
    if problem.task == REGRESSION:
        return {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error",
            "medae": "neg_median_absolute_error",
        }[metric]
    if metric == "kappa":
        return make_scorer(cohen_kappa_score)
    if metric == "auc":
        return ("roc_auc" if problem.k == 2 else "roc_auc_ovr") if has_proba else "balanced_accuracy"
    if metric == "log_loss":
        return "neg_log_loss" if has_proba else "balanced_accuracy"
    return metric


# ============================================================================ validation
def make_splits(
    problem_y: np.ndarray, task: str, folds: int, seed: int, repeats: int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    zeros = np.zeros(len(problem_y))
    if task == CLASSIFICATION:
        counts = np.bincount(problem_y)
        folds = int(min(folds, counts.min()))
        if folds < 2:
            raise InvalidInputError("Alguna clase tiene muy pocas observaciones para validar de forma cruzada.")
        cv = (
            StratifiedKFold(folds, shuffle=True, random_state=seed)
            if repeats == 1
            else RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
        )
        return list(cv.split(zeros, problem_y))
    cv = (
        KFold(folds, shuffle=True, random_state=seed)
        if repeats == 1
        else RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    )
    return list(cv.split(zeros))


def holdout_split(y: np.ndarray, task: str, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(len(y))
    if test_size <= 0:
        return index, np.array([], dtype=int)
    stratify = y if task == CLASSIFICATION else None
    try:
        train, test = train_test_split(index, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError as error:
        raise InvalidInputError(
            f"No se pudo separar el conjunto de prueba ({error}). Use una fracción menor."
        ) from None
    return np.sort(train), np.sort(test)


@dataclass
class CVResult:
    test: list[dict[str, float]] = field(default_factory=list)
    train: list[dict[str, float]] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    oof: Pred | None = None  # out-of-fold predictions of the first repetition (aligned with the rows)


def _dummy(problem: Problem) -> Any:
    return DummyClassifier(strategy="prior") if problem.task == CLASSIFICATION else DummyRegressor(strategy="mean")


def cross_validate(
    problem: Problem,
    build: Callable[[], Any] | None,
    x: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    folds_per_repeat: int | None = None,
    keep_train: bool = True,
) -> CVResult:
    """Fit on each training fold and score the validation fold (``build=None`` is the naive baseline)."""
    result = CVResult()
    n = len(y)
    keep = folds_per_repeat or len(splits)
    values = np.zeros(n)
    scores = None
    zeros = np.zeros((n, 1))
    for number, (train, test) in enumerate(splits):
        start = time.perf_counter()
        if build is None:
            model = _dummy(problem).fit(zeros[train], y[train])
            xt, xtr = zeros[test], zeros[train]
        else:
            model = build().fit(x.iloc[train], y[train])
            xt, xtr = x.iloc[test], x.iloc[train]
        pred = predict_all(model, xt, problem.task)
        result.seconds.append(time.perf_counter() - start)
        result.test.append(score_all(problem, y[test], pred))
        if keep_train:
            result.train.append(score_all(problem, y[train], predict_all(model, xtr, problem.task)))
        if number < keep:
            values[test] = pred.values
            if pred.scores is not None:
                if scores is None:
                    scores = np.full((n, *pred.scores.shape[1:]), np.nan)
                scores[test] = pred.scores
            result.oof = Pred(values, scores, pred.proba)
    return result


def summarize(rows: list[dict[str, float]], metric: str) -> tuple[float, float]:
    """Mean and standard deviation of one metric over the folds (missing values skipped)."""
    v = np.array([r[metric] for r in rows], float)
    v = v[np.isfinite(v)]
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0) if len(v) else (float("nan"), float("nan"))


# ============================================================================ the evaluation of one model
@dataclass
class Evaluation:
    problem: Problem
    metric: str
    higher: bool
    train_rows: np.ndarray
    test_rows: np.ndarray
    cv: CVResult
    baseline: CVResult
    final: Any  # fitted on the training portion
    test_pred: Pred | None
    new_pred: Pred | None
    seed: int
    builder: Callable[[], Any]


def evaluate(
    problem: Problem,
    builder: Callable[[], Any],
    *,
    metric: str,
    folds: int,
    test_size: float,
    seed: int,
    repeats: int = 1,
) -> Evaluation:
    train, test = holdout_split(problem.y, problem.task, test_size, seed)
    x_train, y_train = problem.X.iloc[train], problem.y[train]
    splits = make_splits(y_train, problem.task, folds, seed, repeats)
    per_repeat = len(splits) // repeats
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = cross_validate(problem, builder, x_train, y_train, splits, folds_per_repeat=per_repeat)
        baseline = cross_validate(problem, None, x_train, y_train, splits, keep_train=False)
        final = builder().fit(x_train, y_train)
        test_pred = predict_all(final, problem.X.iloc[test], problem.task) if len(test) else None
        new_pred = None
        if len(problem.X_new):
            refit = builder().fit(problem.X, problem.y) if len(test) else final
            new_pred = predict_all(refit, problem.X_new, problem.task)
    higher = metric_table(problem.task)[metric][1]
    return Evaluation(problem, metric, higher, train, test, cv, baseline, final, test_pred, new_pred, seed, builder)


# ============================================================================ helpers for the algorithm-specific parts
def encoded_info(final: Pipeline, problem: Problem) -> list[tuple[str, str, str | None]]:
    """For every column the model sees: (original variable, readable name, category when it is a dummy)."""
    prep = final.named_steps["prep"]
    info: list[tuple[str, str, str | None]] = []
    for name, transformer, columns in prep.transformers_:
        if name == "num":
            info += [(c, c, None) for c in columns]
        elif name == "cat":
            encoder = transformer[-1]
            for column, categories in zip(columns, encoder.categories_, strict=True):
                info += [(column, f"{column} = {level}", str(level)) for level in categories]
    return info


def search_values(
    problem: Problem,
    algorithm: Algorithm,
    base: dict[str, Any],
    name: str,
    values: list[Any],
    *,
    metric: str,
    folds: int,
    test_size: float,
    seed: int,
    prefer: str = "simple_high",
) -> tuple[Any, list[tuple[Any, float, float]]]:
    """Cross-validate one setting over ``values`` on the training portion: (the value chosen by the one-standard-error
    rule, the table). ``prefer`` says which end is the simpler model: ``simple_high`` (larger value) or ``simple_low``."""
    train, _ = holdout_split(problem.y, problem.task, test_size, seed)
    x, y = problem.X.iloc[train], problem.y[train]
    splits = make_splits(y, problem.task, folds, seed)
    higher = metric_table(problem.task)[metric][1]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for value in values:
            result = cross_validate(
                problem,
                lambda value=value: make_model(problem, algorithm, {**base, name: value}, seed),
                x,
                y,
                splits,
                keep_train=False,
            )
            mean, sd = summarize(result.test, metric)
            rows.append((value, mean, sd))
    finite = [r for r in rows if np.isfinite(r[1])]
    if not finite:
        raise InvalidInputError("No se pudo evaluar ninguna de las alternativas.")
    best = max(finite, key=lambda r: r[1]) if higher else min(finite, key=lambda r: r[1])
    limit = best[1] - best[2] if higher else best[1] + best[2]
    acceptable = [r for r in finite if (r[1] >= limit if higher else r[1] <= limit)]
    pick = max(acceptable, key=lambda r: r[0]) if prefer == "simple_high" else min(acceptable, key=lambda r: r[0])
    return pick[0], rows


# ============================================================================ the common report
def _fmt_pm(mean: float, sd: float) -> str:
    return "—" if not np.isfinite(mean) else f"{mean:.4g} ± {sd:.2g}"


def performance_table(out: Output, ev: Evaluation) -> None:
    problem, table = ev.problem, metric_table(ev.problem.task)
    rows = []
    for key, (label, _) in table.items():
        cv_mean, cv_sd = summarize(ev.cv.test, key)
        if not np.isfinite(cv_mean):
            continue
        train_mean, _ = summarize(ev.cv.train, key)
        base_mean, _ = summarize(ev.baseline.test, key)
        test_value = None
        if ev.test_pred is not None:
            test_value = score_all(problem, problem.y[ev.test_rows], ev.test_pred)[key]
        name = strong(label) if key == ev.metric else label
        rows.append(
            [
                name,
                None if not np.isfinite(train_mean) else train_mean,
                cv_mean,
                cv_sd,
                None if test_value is None or not np.isfinite(test_value) else test_value,
                None if not np.isfinite(base_mean) else base_mean,
            ]
        )
    folds = len(ev.cv.test)
    out.table(
        "Desempeño del modelo",
        [
            "Métrica",
            "Entrenamiento",
            f"Validación cruzada ({folds} pliegues)",
            "Desv. est. entre pliegues",
            "Prueba (reservada)",
            "Línea base",
        ],
        rows,
        note="Entrenamiento: promedio sobre los mismos datos con que se ajusta cada pliegue (optimista). Validación cruzada: lo que cabe esperar con datos nuevos. "
        "Prueba: filas reservadas que el modelo nunca vio. Línea base: predecir siempre la clase más frecuente (o la media).",
    )


def baseline_alerts(out: Output, ev: Evaluation) -> None:
    cv_mean, cv_sd = summarize(ev.cv.test, ev.metric)
    base_mean, _ = summarize(ev.baseline.test, ev.metric)
    train_mean, _ = summarize(ev.cv.train, ev.metric)
    label = metric_table(ev.problem.task)[ev.metric][0].split(" (")[0]
    if np.isfinite(cv_mean) and np.isfinite(base_mean):
        gain = (cv_mean - base_mean) if ev.higher else (base_mean - cv_mean)
        if gain <= cv_sd:
            out.alert(
                "warning",
                f"El modelo no supera de forma clara a la línea base ({label}: {cv_mean:.4g} frente a {base_mean:.4g}, con desviación {cv_sd:.2g} entre pliegues): las variables aportan poco.",
            )
    if np.isfinite(train_mean) and np.isfinite(cv_mean):
        if (
            ev.higher
            and train_mean - cv_mean > 0.1 * max(abs(train_mean), 1e-9) + 2 * cv_sd
            and train_mean - cv_mean > 0.05
        ):
            out.alert(
                "warning",
                f"Posible sobreajuste: {label} {train_mean:.4g} en entrenamiento y {cv_mean:.4g} al validar. Simplifique el modelo o regularícelo.",
            )
        if not ev.higher and cv_mean - train_mean > 0.25 * abs(cv_mean) + 2 * cv_sd:
            out.alert(
                "warning",
                f"Posible sobreajuste: {label} {train_mean:.4g} en entrenamiento y {cv_mean:.4g} al validar. Simplifique el modelo o regularícelo.",
            )
    if ev.test_pred is not None:
        test_value = score_all(ev.problem, ev.problem.y[ev.test_rows], ev.test_pred)[ev.metric]
        if (
            np.isfinite(test_value)
            and np.isfinite(cv_mean)
            and abs(test_value - cv_mean) > 3 * max(cv_sd, 1e-9)
            and abs(test_value - cv_mean) > 0.05 * abs(cv_mean)
        ):
            out.alert(
                "info",
                f"En la partición de prueba {label} fue {test_value:.4g}, algo distinto de la validación cruzada ({cv_mean:.4g}): con pocas filas la prueba es ruidosa.",
            )


def confusion_section(out: Output, problem: Problem, y: np.ndarray, p: np.ndarray, title: str) -> np.ndarray:
    k = problem.k
    matrix = np.zeros((k, k), int)
    np.add.at(matrix, (y, p), 1)
    labels = problem.labels
    out.table(
        title,
        ["Real / predicho", *labels, "Total", "% correcto"],
        [
            [
                labels[i],
                *(int(v) for v in matrix[i]),
                int(matrix[i].sum()),
                100 * float(matrix[i, i] / max(matrix[i].sum(), 1)),
            ]
            for i in range(k)
        ],
    )
    if k <= 12:
        share = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        out.chart(
            heatmap_chart(
                "Matriz de confusión (% de cada clase real)",
                100 * share,
                labels,
                labels,
                x_title="Predicho",
                y_title="Real",
                midpoint=50,
                height=360 if k <= 5 else 460,
            )
        )
    return matrix


def class_report(out: Output, problem: Problem, matrix: np.ndarray) -> None:
    rows = []
    for i, name in enumerate(problem.labels):
        tp = float(matrix[i, i])
        precision = tp / matrix[:, i].sum() if matrix[:, i].sum() else float("nan")
        recall = tp / matrix[i].sum() if matrix[i].sum() else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append([name, precision, recall, f1, int(matrix[i].sum())])
    out.table(
        "Precisión y exhaustividad por clase (validación cruzada)",
        ["Clase", "Precisión", "Exhaustividad (sensibilidad)", "F1", "n"],
        rows,
        note="Precisión: de lo que el modelo llama de esta clase, qué parte lo es. Exhaustividad: de las que son de esta clase, qué parte encuentra.",
    )


def curves_section(out: Output, problem: Problem, y: np.ndarray, pred: Pred, source: str) -> dict[str, Any]:
    """ROC (and precision-recall for two classes) from the out-of-fold scores, with the usual thresholds."""
    scores = class_scores(pred, problem.k, problem.positive)
    info: dict[str, Any] = {}
    if scores is None or np.isnan(scores).any():
        return info
    if problem.k == 2:
        event = y == problem.positive
        fpr, tpr, thr = roc_curve(event, scores)
        area = float(roc_auc_score(event, scores))
        _, _, low, high = delong_auc(event, scores)
        info["auc"] = area
        info["auc_ci"] = [low, high]
        out.chart(
            line_chart(
                f"Curva ROC ({source}; AUC = {area:.3f}, IC 95 % {low:.3f}–{high:.3f}; evento: {problem.labels[problem.positive]})",
                "1 − especificidad (falsos positivos)",
                "Sensibilidad",
                [{"name": "Modelo", "x": fpr, "y": tpr}, {"name": "Azar", "x": [0, 1], "y": [0, 1], "dash": "dash"}],
                height=380,
            )
        )
        precision, recall, _ = precision_recall_curve(event, scores)
        average = float(np.sum(np.diff(recall[::-1]) * precision[::-1][1:])) if len(recall) > 1 else float("nan")
        prevalence = float(event.mean())
        out.chart(
            line_chart(
                f"Curva de precisión-exhaustividad (precisión promedio = {average:.3f})",
                "Exhaustividad",
                "Precisión",
                [{"name": "Modelo", "x": recall, "y": precision}],
                hlines=[{"value": prevalence, "label": f"azar ({prevalence:.2f})", "dash": "dash"}],
                height=340,
            )
        )
        info["average_precision"] = average
        finite = np.isfinite(thr)
        j = tpr - fpr
        best_j = int(np.argmax(np.where(finite, j, -np.inf)))
        pos, neg = float(event.sum()), float((~event).sum())
        tp_all = tpr * pos
        fp_all = fpr * neg
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(
                2 * tp_all + fp_all + (pos - tp_all) > 0, 2 * tp_all / (2 * tp_all + fp_all + (pos - tp_all)), 0.0
            )
            prec = np.where(tp_all + fp_all > 0, tp_all / (tp_all + fp_all), np.nan)
        best_f = int(np.argmax(np.where(finite, f1, -np.inf)))
        balanced = int(np.argmin(np.where(finite, np.abs(tpr - (1 - fpr)), np.inf)))
        rows = [
            [
                name,
                float(thr[i]),
                float(tpr[i]),
                float(1 - fpr[i]),
                None if np.isnan(prec[i]) else float(prec[i]),
                float(f1[i]),
            ]
            for name, i in (
                ("Máximo de Youden (sensibilidad + especificidad)", best_j),
                ("Máximo de F1", best_f),
                ("Sensibilidad = especificidad", balanced),
            )
        ]
        out.table(
            "Umbrales de decisión sugeridos",
            ["Criterio", "Umbral del puntaje", "Sensibilidad", "Especificidad", "Precisión", "F1"],
            rows,
            note="El umbral por omisión (probabilidad 0.5) no siempre es el mejor si los errores cuestan distinto o la clase es rara.",
        )
        info["youden_threshold"] = float(thr[best_j])
    else:
        series, rows = [], []
        for c, name in enumerate(problem.labels):
            event = y == c
            if event.all() or not event.any():
                continue
            fpr, tpr, _ = roc_curve(event, scores[:, c])
            area = float(roc_auc_score(event, scores[:, c]))
            series.append({"name": f"{name} (AUC {area:.3f})", "x": fpr, "y": tpr})
            rows.append([name, area])
        series.append({"name": "Azar", "x": [0, 1], "y": [0, 1], "dash": "dash"})
        out.chart(
            line_chart(
                f"Curvas ROC de cada clase contra las demás ({source})",
                "1 − especificidad",
                "Sensibilidad",
                series,
                height=400,
            )
        )
        out.table("AUC por clase (una contra las demás)", ["Clase", "AUC"], rows)
        info["auc"] = float(np.mean([r[1] for r in rows])) if rows else float("nan")
    return info


def regression_section(out: Output, problem: Problem, y: np.ndarray, pred: np.ndarray, source: str) -> None:
    residual = y - pred
    shown = np.arange(len(y))
    if len(shown) > MAX_POINTS:
        shown = np.sort(np.random.default_rng(0).choice(len(y), MAX_POINTS, replace=False))
    low, high = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
    out.chart(
        scatter_chart(
            f"Observado frente a predicho ({source})",
            "Predicho",
            "Observado",
            [{"name": "Observaciones", "x": pred[shown], "y": y[shown]}],
            lines=[{"name": "Predicción perfecta", "x": [low, high], "y": [low, high], "dash": "dash"}],
            height=400,
        )
    )
    out.chart(
        scatter_chart(
            f"Residuos frente a lo predicho ({source})",
            "Predicho",
            "Residuo (observado − predicho)",
            [{"name": "Residuos", "x": pred[shown], "y": residual[shown]}],
            hlines=[{"value": 0.0, "dash": "dash"}],
            height=340,
        )
    )
    q = np.quantile(np.abs(residual), [0.5, 0.9, 0.99])
    out.table(
        "Distribución del error de predicción",
        ["Medida", "Valor"],
        [
            ["Error medio (sesgo)", float(residual.mean())],
            ["Desviación de los errores", float(residual.std(ddof=1))],
            ["Mediana del error absoluto", float(q[0])],
            ["Percentil 90 del error absoluto", float(q[1])],
            ["Percentil 99 del error absoluto", float(q[2])],
            ["Mayor error absoluto", float(np.abs(residual).max())],
        ],
        note="El 90 % de las predicciones se equivoca menos que el percentil 90.",
    )


def importance_section(out: Output, ev: Evaluation, repeats: int = 5) -> dict[str, float]:
    """Permutation importance of the original variables on the held-out rows (or on the training rows without a test set)."""
    problem = ev.problem
    rows = ev.test_rows if len(ev.test_rows) else ev.train_rows
    x, y = problem.X.iloc[rows], problem.y[rows]
    if len(rows) > 5000:
        pick = np.sort(np.random.default_rng(ev.seed).choice(len(rows), 5000, replace=False))
        x, y = x.iloc[pick], y[pick]
    scorer = scorer_for(problem, ev.metric, hasattr(ev.final, "predict_proba"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(ev.final, x, y, scoring=scorer, n_repeats=repeats, random_state=ev.seed)
    order = np.argsort(result.importances_mean)[::-1]
    names = problem.features
    label = metric_table(problem.task)[ev.metric][0].split(" (")[0]
    where = (
        "las filas de prueba reservadas"
        if len(ev.test_rows)
        else "las filas de entrenamiento (sin conjunto de prueba la cifra es optimista)"
    )
    out.table(
        "Importancia de las variables (por permutación)",
        ["Variable", f"Pérdida de {label} al desordenarla", "Desv. est.", "¿Aporta?"],
        [
            [
                names[i],
                float(result.importances_mean[i]),
                float(result.importances_std[i]),
                good("sí")
                if result.importances_mean[i] > 2 * result.importances_std[i] and result.importances_mean[i] > 0
                else muted("dudoso"),
            ]
            for i in order
        ],
        note=f"Se desordena cada variable en {where} y se mide cuánto empeora el modelo: mientras más empeora, más depende de ella. Es válida para cualquier tipo de modelo y corrige el sesgo de la importancia por impureza.",
    )
    top = order[:25]
    out.chart(
        bar_chart(
            "Importancia por permutación",
            f"Pérdida de {label} al desordenar la variable",
            None,
            [names[i] for i in top][::-1],
            [{"name": "Importancia", "y": [float(result.importances_mean[i]) for i in top][::-1]}],
            horizontal=True,
            errors=[[float(result.importances_std[i]) for i in top][::-1]],
            height=max(260, 28 * len(top) + 90),
        )
    )
    return {names[i]: float(result.importances_mean[i]) for i in order}


def predictions_frame(frame: pd.DataFrame, ev: Evaluation) -> pd.DataFrame:
    """The data with the model's prediction for every row: out-of-fold, held out or new."""
    problem = ev.problem
    result = frame.copy()
    name = f"pred_{problem.target}"
    column = pd.Series(np.nan, index=frame.index, dtype=object if problem.task == CLASSIFICATION else float)
    group = pd.Series(None, index=frame.index, dtype=object)
    probabilities = (
        {f"prob_{c}": pd.Series(np.nan, index=frame.index, dtype=float) for c in problem.labels}
        if problem.task == CLASSIFICATION
        else {}
    )

    def fill(index: pd.Index, pred: Pred, label: str) -> None:
        values = (
            np.array(problem.labels, dtype=object)[pred.values.astype(int)]
            if problem.task == CLASSIFICATION
            else pred.values
        )
        column.loc[index] = values
        group.loc[index] = label
        if probabilities and pred.proba and pred.scores is not None:
            for c, key in enumerate(probabilities):
                probabilities[key].loc[index] = pred.scores[:, c]

    train_index = problem.index[ev.train_rows]
    if ev.cv.oof is not None:
        fill(train_index, ev.cv.oof, "entrenamiento (validación cruzada)")
    if ev.test_pred is not None:
        fill(problem.index[ev.test_rows], ev.test_pred, "prueba")
    if ev.new_pred is not None:
        fill(problem.new_index, ev.new_pred, "nueva (sin objetivo)")
    result[name] = column
    result["conjunto"] = group
    for key, series in probabilities.items():
        result[key] = series
    if problem.task == REGRESSION:
        observed = pd.to_numeric(frame[problem.target], errors="coerce")
        result[f"error_{problem.target}"] = observed - column.astype(float)
    return result


# ============================================================================ the whole analysis
TARGET = RoleSpec("target", "Variable a predecir (objetivo)", "any")
FEATURES = RoleSpec("features", "Variables predictoras", "any", multiple=True, min_count=1)
PREDICTIONS = OutputSpec("predicciones", "Predicciones de cada fila")
SEED = FieldSpec("seed", "Semilla", "integer", 123, minimum=0, maximum=2**31 - 1)
COMMON_PARAMS = (
    FieldSpec(
        "task",
        "Tipo de problema",
        "select",
        "auto",
        (
            ("auto", "Automático (categorías → clasificación, números → regresión)"),
            (CLASSIFICATION, "Clasificación"),
            (REGRESSION, "Regresión"),
        ),
        help="Una variable numérica con solo dos valores se trata como clasificación.",
    ),
    FieldSpec("folds", "Pliegues de validación cruzada", "integer", 5, minimum=2, maximum=20),
    FieldSpec(
        "test_size",
        "Fracción reservada para prueba (0 = ninguna)",
        "number",
        0.2,
        minimum=0,
        maximum=0.5,
        help="Filas que el modelo nunca ve, para una evaluación final honesta.",
    ),
    FieldSpec("metric", "Métrica principal", "select", "auto", METRIC_OPTIONS),
    FieldSpec(
        "missing",
        "Datos faltantes en las predictoras",
        "select",
        "impute",
        (("impute", "Imputar (mediana / moda) dentro de cada pliegue"), ("drop", "Quitar las filas incompletas")),
    ),
    FieldSpec("predict_new", "Predecir también las filas sin valor en el objetivo", "boolean", True),
    FieldSpec("positive", "Clase que cuenta como evento (binaria; vacío = la última)", "text", optional=True),
    SEED,
)


def run_supervised(
    frame: pd.DataFrame,
    roles: dict[str, Any],
    params: dict[str, Any],
    algorithm: Algorithm,
    hyper: dict[str, Any],
    *,
    title: str,
    detail: str = "",
    extras: Callable[[Output, SimpleNamespace], dict[str, Any] | None] | None = None,
    resolve: Callable[[Problem, str, int], tuple[dict[str, Any], Any]] | None = None,
) -> AnalysisOutput:
    """Fit and report one algorithm. ``resolve(problem, metric, seed)`` may pick settings by validation before the
    evaluation (a pruning level, a number of neighbours): it returns the settings to add and anything the extras show."""
    problem = build_problem(
        frame,
        roles["target"],
        roles["features"],
        task=params["task"],
        missing=params["missing"],
        predict_new=params["predict_new"],
        positive=params.get("positive") or None,
        allowed=algorithm.tasks,
    )
    if algorithm.max_rows and len(problem.X) > algorithm.max_rows:
        raise InvalidInputError(
            f"Este modelo admite hasta {algorithm.max_rows:,} filas (su costo crece más que proporcionalmente); muestree el dataset o use otro modelo.".replace(
                ",", " "
            )
        )
    seed = int(params["seed"])
    metric = choose_metric(problem, params["metric"])
    info = None
    if resolve is not None:
        update, info = resolve(problem, metric, seed)
        hyper = {**hyper, **update}
    settings = {**algorithm.defaults, **hyper}

    def builder() -> Pipeline:
        return make_model(problem, algorithm, hyper, seed)

    ev = evaluate(
        problem, builder, metric=metric, folds=int(params["folds"]), test_size=float(params["test_size"]), seed=seed
    )
    return report(frame, ev, algorithm, settings, params, title=title, detail=detail, extras=extras, info=info)


def report(
    frame: pd.DataFrame,
    ev: Evaluation,
    algorithm: Algorithm,
    settings: dict[str, Any],
    params: dict[str, Any],
    *,
    title: str,
    detail: str = "",
    extras: Callable[[Output, SimpleNamespace], dict[str, Any] | None] | None = None,
    info: Any = None,
) -> AnalysisOutput:
    problem = ev.problem
    n_train, n_test = len(ev.train_rows), len(ev.test_rows)
    subtitle = (
        f"{TASK_NAMES[problem.task]} · {problem.target} · {len(problem.features)} variable(s) · n = {len(problem.X)}"
        + (f" · {detail}" if detail else "")
    )
    out = Output(title, subtitle)
    if problem.dropped:
        out.alert("info", f"Se omitieron {problem.dropped} fila(s) sin objetivo o con datos incompletos.")
    if problem.task == CLASSIFICATION:
        counts = np.bincount(problem.y, minlength=problem.k)
        if counts.max() > 4 * counts.min():
            out.alert(
                "info",
                f"Las clases están desbalanceadas ({', '.join(f'{c}: {int(v)}' for c, v in zip(problem.labels, counts, strict=True))}): fíjese en la exactitud balanceada, la exhaustividad por clase y el AUC.",
            )
    label = metric_table(problem.task)[ev.metric][0]
    cv_mean, cv_sd = summarize(ev.cv.test, ev.metric)
    base_mean, _ = summarize(ev.baseline.test, ev.metric)
    tiles: list[tuple] = [
        (f"{label.split(' (')[0]} (validación cruzada)", f"{cv_mean:.4g} ± {cv_sd:.2g}"),
        ("Línea base", base_mean),
    ]
    test_value = None
    if ev.test_pred is not None:
        test_value = score_all(problem, problem.y[ev.test_rows], ev.test_pred)[ev.metric]
        tiles.append((f"{label.split(' (')[0]} (prueba)", test_value))
    tiles.append(("Filas de entrenamiento / prueba", f"{n_train} / {n_test}"))
    out.tiles(tiles)
    baseline_alerts(out, ev)
    performance_table(out, ev)
    result: dict[str, Any] = {
        "task": problem.task,
        "metric": ev.metric,
        "cv_mean": cv_mean,
        "cv_sd": cv_sd,
        "baseline": base_mean,
        "test": test_value,
        "n_train": n_train,
        "n_test": n_test,
        "scores": {key: summarize(ev.cv.test, key)[0] for key in metric_table(problem.task)},
    }
    oof = ev.cv.oof
    y_train = problem.y[ev.train_rows]
    if problem.task == CLASSIFICATION and oof is not None:
        matrix = confusion_section(
            out, problem, y_train, oof.values.astype(int), "Matriz de confusión (validación cruzada)"
        )
        class_report(out, problem, matrix)
        result.update(confusion=matrix.tolist(), **curves_section(out, problem, y_train, oof, "validación cruzada"))
        if ev.test_pred is not None:
            confusion_section(
                out,
                problem,
                problem.y[ev.test_rows],
                ev.test_pred.values.astype(int),
                "Matriz de confusión (prueba reservada)",
            )
    elif oof is not None:
        regression_section(out, problem, y_train, oof.values, "validación cruzada")
    result["importance"] = importance_section(out, ev)
    context = SimpleNamespace(
        problem=problem,
        evaluation=ev,
        final=ev.final,
        params=params,
        settings=settings,
        algorithm=algorithm,
        result=result,
        info=info,
    )
    if extras is not None:
        more = extras(out, context)
        if more:
            result.update(more)
    out.dataset("predicciones", "Predicciones de cada fila", predictions_frame(frame, ev))
    return out.set(**result).build()


# ============================================================================ code for the "see the code" button
def code_for(
    roles: dict[str, Any], params: dict[str, Any], dataset: str, imports: str, estimator: str, scaled: bool
) -> str:
    from sigma.domain.analysis.support import load_line

    features = roles["features"]
    return (
        "import numpy as np\nimport pandas as pd\n"
        "from sklearn.compose import make_column_transformer\nfrom sklearn.impute import SimpleImputer\n"
        "from sklearn.model_selection import cross_val_score, train_test_split\n"
        "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
        f"{imports}\n\n" + load_line(dataset) + f"target, features = {roles['target']!r}, {features!r}\n"
        "df = df.dropna(subset=[target])\n"
        "X, y = df[features], df[target]\n"
        "numeric = X.select_dtypes('number').columns.tolist()\n"
        "categorical = [c for c in features if c not in numeric]\n"
        "prep = make_column_transformer(\n"
        f"    (make_pipeline(SimpleImputer(strategy='median'){', StandardScaler()' if scaled else ''}), numeric),\n"
        "    (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse_output=False)), categorical),\n"
        ")\n"
        f"model = make_pipeline(prep, {estimator})\n"
        f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={float(params.get('test_size', 0.2))}, random_state={int(params.get('seed', 123))})\n"
        f"scores = cross_val_score(model, X_train, y_train, cv={int(params.get('folds', 5))})\n"
        "print(scores.mean(), scores.std())\n"
        "print(model.fit(X_train, y_train).score(X_test, y_test))\n"
    )
