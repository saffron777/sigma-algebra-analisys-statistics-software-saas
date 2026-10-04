"""The estimator families of the learning module: how each one is built, its default settings and its tuning space.

Settings use plain names of our own (``max_features="sqrt"``, ``strength``…) so the forms, the comparison and the
searches speak the same language; ``build`` translates them into scikit-learn estimators.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sigma.domain.analysis.ml_core import CLASSIFICATION, Algorithm
from sigma.domain.errors import InvalidInputError

SVM_MAX_ROWS = 20_000
MLP_MAX_UNITS = 512
MLP_MAX_LAYERS = 4


def parse_layers(text: str) -> tuple[int, ...]:
    """``"64, 32"`` → ``(64, 32)``, with sane limits."""
    try:
        sizes = tuple(int(part) for part in str(text).replace(";", ",").split(",") if part.strip())
    except ValueError:
        raise InvalidInputError(
            "Las capas ocultas se escriben como números separados por comas, por ejemplo 64, 32."
        ) from None
    if not sizes or any(s < 1 for s in sizes):
        raise InvalidInputError("Indique al menos una capa oculta con 1 neurona o más.")
    if len(sizes) > MLP_MAX_LAYERS or any(s > MLP_MAX_UNITS for s in sizes):
        raise InvalidInputError(f"Se admiten hasta {MLP_MAX_LAYERS} capas de {MLP_MAX_UNITS} neuronas.")
    return sizes


def _scaled_target(regressor: Any) -> TransformedTargetRegressor:
    """Standardise the response: epsilon-tubes and gradient descent expect a target of about unit scale."""
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def _weights(value: Any) -> Any:
    return None if value in (None, "", "none") else value


# ------------------------------------------------------------------ builders
def _tree(task: str, h: dict[str, Any], seed: int) -> Any:
    common = {
        "max_depth": h["max_depth"],
        "min_samples_leaf": int(h["min_samples_leaf"]),
        "ccp_alpha": float(h["ccp_alpha"]),
        "random_state": seed,
    }
    if task == CLASSIFICATION:
        return DecisionTreeClassifier(criterion=h["criterion"], class_weight=_weights(h["class_weight"]), **common)
    return DecisionTreeRegressor(**common)


def _features(value: Any, task: str) -> Any:
    if value == "auto":
        return "sqrt" if task == CLASSIFICATION else 0.33
    return {"sqrt": "sqrt", "log2": "log2", "third": 0.33, "all": 1.0}.get(value, value)


def _forest(task: str, h: dict[str, Any], seed: int) -> Any:
    common = {
        "n_estimators": int(h["n_estimators"]),
        "max_depth": h["max_depth"],
        "min_samples_leaf": int(h["min_samples_leaf"]),
        "max_features": _features(h["max_features"], task),
        "random_state": seed,
        "n_jobs": 1,
        "oob_score": True,
    }
    if task == CLASSIFICATION:
        return RandomForestClassifier(class_weight=_weights(h["class_weight"]), **common)
    return RandomForestRegressor(**common)


def _boosting(task: str, h: dict[str, Any], seed: int) -> Any:
    common = {
        "max_iter": int(h["max_iter"]),
        "learning_rate": float(h["learning_rate"]),
        "max_depth": h["max_depth"],
        "min_samples_leaf": int(h["min_samples_leaf"]),
        "l2_regularization": float(h["l2_regularization"]),
        "early_stopping": bool(h["early_stopping"]),
        "validation_fraction": 0.15,
        "n_iter_no_change": 15,
        "random_state": seed,
    }
    if task == CLASSIFICATION:
        return HistGradientBoostingClassifier(class_weight=_weights(h["class_weight"]), **common)
    return HistGradientBoostingRegressor(**common)


def _knn(task: str, h: dict[str, Any], seed: int) -> Any:
    common = {"n_neighbors": int(h["n_neighbors"]), "weights": h["weights"], "p": int(h["p"])}
    return KNeighborsClassifier(**common) if task == CLASSIFICATION else KNeighborsRegressor(**common)


def _svm(task: str, h: dict[str, Any], seed: int) -> Any:
    common = {"kernel": h["kernel"], "C": float(h["C"]), "gamma": h["gamma"], "degree": int(h["degree"])}
    if task == CLASSIFICATION:
        return SVC(class_weight=_weights(h["class_weight"]), random_state=seed, **common)
    return _scaled_target(SVR(epsilon=float(h["epsilon"]), **common))


def _bayes(task: str, h: dict[str, Any], seed: int) -> Any:
    return GaussianNB(var_smoothing=float(h["var_smoothing"]))


def _mlp(task: str, h: dict[str, Any], seed: int) -> Any:
    layers = h["hidden_layer_sizes"]
    layers = parse_layers(layers) if isinstance(layers, str) else tuple(layers)
    common = {
        "hidden_layer_sizes": layers,
        "activation": h["activation"],
        "alpha": float(h["alpha"]),
        "learning_rate_init": float(h["learning_rate_init"]),
        "max_iter": int(h["max_iter"]),
        "early_stopping": bool(h["early_stopping"]),
        "n_iter_no_change": 15,
        "random_state": seed,
    }
    return MLPClassifier(**common) if task == CLASSIFICATION else _scaled_target(MLPRegressor(**common))


def _penalized(task: str, h: dict[str, Any], seed: int) -> Any:
    l1 = float(h["l1_ratio"])
    if task == CLASSIFICATION:
        model = LogisticRegression(
            C=float(h["C"]),
            l1_ratio=l1,
            solver="saga" if l1 > 0 else "lbfgs",
            class_weight=_weights(h["class_weight"]),
            max_iter=2000,
            random_state=seed,
        )
        grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    else:
        model = (
            ElasticNet(alpha=float(h["alpha"]), l1_ratio=l1, max_iter=5000, random_state=seed)
            if l1 > 0
            else Ridge(alpha=float(h["alpha"]))
        )
        grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    return GridSearchCV(model, grid, cv=3) if h["tune"] else model


_NO_WEIGHTS = {"class_weight": None}

ALGORITHMS: dict[str, Algorithm] = {
    "tree": Algorithm(
        "tree", "Árbol de decisión (CART)", _tree,
        {"max_depth": 5, "min_samples_leaf": 5, "ccp_alpha": 0.0, "criterion": "gini", **_NO_WEIGHTS},
        {"max_depth": [2, 3, 4, 5, 6, 8, 12], "min_samples_leaf": [1, 3, 5, 10, 20]},
    ),
    "forest": Algorithm(
        "forest", "Bosque aleatorio", _forest,
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": "auto", **_NO_WEIGHTS},
        {"max_depth": [3, 6, 10, None], "min_samples_leaf": [1, 3, 10], "max_features": ["sqrt", "log2", "all"]},
    ),
    "boosting": Algorithm(
        "boosting", "Gradient boosting (árboles potenciados)", _boosting,
        {"max_iter": 200, "learning_rate": 0.1, "max_depth": 3, "min_samples_leaf": 20, "l2_regularization": 0.0, "early_stopping": False, **_NO_WEIGHTS},
        {"learning_rate": [0.03, 0.1, 0.3], "max_depth": [2, 3, 5], "max_iter": [100, 200, 400], "min_samples_leaf": [10, 20, 50]},
    ),
    "knn": Algorithm(
        "knn", "k vecinos más cercanos", _knn,
        {"n_neighbors": 5, "weights": "uniform", "p": 2},
        {"n_neighbors": [1, 3, 5, 7, 11, 15, 25], "weights": ["uniform", "distance"], "p": [1, 2]},
        scaled=True,
    ),
    "svm": Algorithm(
        "svm", "Máquina de vectores de soporte (SVM)", _svm,
        {"kernel": "rbf", "C": 1.0, "gamma": "scale", "degree": 3, "epsilon": 0.1, **_NO_WEIGHTS},
        {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1], "kernel": ["rbf", "linear"]},
        scaled=True,
        max_rows=SVM_MAX_ROWS,
    ),
    "naive_bayes": Algorithm(
        "naive_bayes", "Naive Bayes gaussiano", _bayes,
        {"var_smoothing": 1e-9},
        {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
        tasks=(CLASSIFICATION,),
    ),
    "mlp": Algorithm(
        "mlp", "Red neuronal (perceptrón multicapa)", _mlp,
        {"hidden_layer_sizes": (64, 32), "activation": "relu", "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 500, "early_stopping": True},
        {"hidden_layer_sizes": [(16,), (32,), (64,), (64, 32), (128, 64)], "alpha": [1e-5, 1e-4, 1e-3, 1e-2], "learning_rate_init": [1e-3, 1e-2]},
        scaled=True,
    ),
    "penalized": Algorithm(
        "penalized", "Modelo lineal regularizado (logística / ridge / elastic net)", _penalized,
        {"C": 1.0, "alpha": 1.0, "l1_ratio": 0.0, "tune": False, **_NO_WEIGHTS},
        {"C": [0.01, 0.1, 1, 10, 100], "alpha": [0.01, 0.1, 1, 10, 100], "l1_ratio": [0.0, 0.5, 1.0]},
        scaled=True,
    ),
}  # fmt: skip

# the settings each family's search varies for a task (a regression forest has no ``C`` and so on)
TUNABLE = {
    "penalized": {CLASSIFICATION: ("C", "l1_ratio"), "regression": ("alpha", "l1_ratio")},
}


def tuning_space(algorithm: Algorithm, task: str) -> dict[str, list[Any]]:
    if algorithm.id in TUNABLE:
        return {k: algorithm.space[k] for k in TUNABLE[algorithm.id][task]}
    return dict(algorithm.space)


def algorithm_options(task_filter: str | None = None) -> tuple[tuple[str, str], ...]:
    return tuple((a.id, a.label) for a in ALGORITHMS.values() if task_filter is None or task_filter in a.tasks)


def get(algorithm_id: str) -> Algorithm:
    try:
        return ALGORITHMS[algorithm_id]
    except KeyError:
        raise InvalidInputError("Modelo no reconocido.") from None
