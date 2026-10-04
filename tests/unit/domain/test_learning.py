"""Machine-learning analyses against scikit-learn itself, planted signals and the metrics' definitions."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sigma.domain.analysis.ml_algorithms import ALGORITHMS, parse_layers
from sigma.domain.analysis.ml_core import Pred, classification_scores, delong_auc, regression_scores
from sigma.domain.analysis.ml_tools import parse_grid
from sigma.domain.errors import InvalidInputError
from tests.analysis_data import offered, run, standard_frame

IRIS_COLS = ["sl", "sw", "pl", "pw"]


def ml(analysis_id, frame, roles=None, params=None):
    return run("aprendizaje", analysis_id, frame, roles, params)


def table(output, title_start):
    return next(b for b in output.blocks if b["type"] == "table" and b["title"].startswith(title_start))["rows"]


def value(cell):
    return getattr(cell, "value", cell)


@pytest.fixture(scope="module")
def iris():
    data = load_iris()
    frame = pd.DataFrame(data.data, columns=IRIS_COLS)
    frame["species"] = pd.Categorical.from_codes(data.target, data.target_names).astype(str)
    return frame


@pytest.fixture(scope="module")
def signal():
    """y = 3·x1 + small noise; x2, x3 are noise; ``g`` shifts y a little."""
    rng = np.random.default_rng(0)
    n = 300
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
            "g": rng.choice(["a", "b", "c"], n),
        }
    )
    frame["y"] = 3 * frame["x1"] + frame["g"].map({"a": 0.0, "b": 1.0, "c": -1.0}) + rng.normal(0, 0.3, n)
    frame["cls"] = np.where(frame["x1"] + rng.normal(0, 0.3, n) > 0, "up", "down")
    return frame


# ------------------------------------------------------------------ the validation is scikit-learn's
def test_cross_validated_score_equals_cross_val_score(iris):
    out = ml(
        "tree",
        iris,
        {"target": "species", "features": IRIS_COLS},
        {"test_size": 0, "folds": 5, "seed": 3, "max_depth": 3, "min_samples_leaf": 5},
    )
    reference = cross_val_score(
        DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=3),
        iris[IRIS_COLS],
        iris["species"],
        cv=StratifiedKFold(5, shuffle=True, random_state=3),
    )
    assert out.result["cv_mean"] == pytest.approx(reference.mean())
    assert out.result["cv_sd"] == pytest.approx(reference.std(ddof=1))
    assert out.result["baseline"] == pytest.approx(1 / 3)


def test_the_held_out_score_is_that_of_a_model_fitted_on_the_training_rows(iris):
    out = ml(
        "tree",
        iris,
        {"target": "species", "features": IRIS_COLS},
        {"test_size": 0.2, "folds": 4, "seed": 3, "max_depth": 3, "min_samples_leaf": 5},
    )
    train, test = train_test_split(np.arange(150), test_size=0.2, random_state=3, stratify=iris["species"])
    model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=3).fit(
        iris.iloc[train][IRIS_COLS], iris.iloc[train]["species"]
    )
    assert out.result["test"] == pytest.approx(model.score(iris.iloc[test][IRIS_COLS], iris.iloc[test]["species"]))
    assert (out.result["n_train"], out.result["n_test"]) == (120, 30)


def test_regression_uses_kfold_and_rmse(signal):
    cols = ["x1", "x2", "x3"]
    out = ml(
        "tree",
        signal,
        {"target": "y", "features": cols},
        {"test_size": 0, "folds": 4, "seed": 5, "max_depth": 4, "min_samples_leaf": 5},
    )
    reference = -cross_val_score(
        DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=5),
        signal[cols],
        signal["y"],
        cv=KFold(4, shuffle=True, random_state=5),
        scoring="neg_root_mean_squared_error",
    )
    assert out.result["metric"] == "rmse" and out.result["cv_mean"] == pytest.approx(reference.mean())
    assert out.result["baseline"] == pytest.approx(
        signal["y"].std(), rel=0.1
    )  # predicting the mean ≈ the standard deviation


# ------------------------------------------------------------------ planted signals
def test_forest_finds_the_real_predictor_and_its_oob_agrees_with_cross_validation(signal):
    out = ml(
        "forest",
        signal,
        {"target": "y", "features": ["x1", "x2", "x3", "g"]},
        {"n_estimators": 80, "folds": 4, "test_size": 0.2, "metric": "r2"},
    )
    importance = out.result["importance"]
    assert max(importance, key=importance.get) == "x1" and importance["x1"] > 5 * max(
        importance["x2"], importance["x3"]
    )
    assert importance["g"] > importance["x2"]
    assert out.result["cv_mean"] > 0.75 and out.result["test"] > 0.7
    assert abs(out.result["oob_score"] - out.result["cv_mean"]) < 0.08
    totals = out.result["impurity_importance"]
    assert max(totals, key=totals.get) == "x1" and sum(totals.values()) == pytest.approx(1)


def test_a_useless_predictor_is_reported_as_no_better_than_the_baseline():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"x": rng.normal(size=200), "z": rng.normal(size=200), "y": rng.choice(["a", "b"], 200)})
    out = ml("forest", frame, {"target": "y", "features": ["x", "z"]}, {"n_estimators": 40, "folds": 4})
    assert any(b["type"] == "alert" and "línea base" in b["message"] for b in out.blocks)
    assert out.result["scores"]["auc"] < 0.62


def test_penalized_logistic_zeroes_the_noise_and_reports_odds_ratios(signal):
    rng = np.random.default_rng(4)
    frame = signal[["x1", "cls"]].copy()
    for i in range(8):
        frame[f"n{i}"] = rng.normal(size=len(frame))
    out = ml(
        "penalized",
        frame,
        {"target": "cls", "features": ["x1", *[f"n{i}" for i in range(8)]]},
        {"l1_ratio": "1", "strength": 20, "tune": False, "folds": 4, "positive": "up"},
    )
    assert out.result["nonzero"] < 6 and out.result["scores"]["auc"] > 0.95
    rows = {r[0]: r for r in table(out, "Coeficientes")}
    assert rows["x1"][1] > 1 and rows["x1"][2] == pytest.approx(np.exp(rows["x1"][1])) and rows["x1"][2] > 3
    assert all(abs(rows[f"n{i}"][1]) < 0.15 for i in range(8))  # the noise is shrunk to (almost) nothing


def test_ridge_regression_recovers_the_coefficient(signal):
    out = ml(
        "penalized",
        signal,
        {"target": "y", "features": ["x1", "x2", "x3"]},
        {"strength": 0.1, "tune": False, "test_size": 0},
    )
    rows = {r[0]: r[1] for r in table(out, "Coeficientes")}
    assert rows["x1"] == pytest.approx(3 * signal["x1"].std(), rel=0.05)  # coefficient of the standardised variable
    assert abs(rows["x2"]) < 0.15


# ------------------------------------------------------------------ data preparation
def test_missing_values_are_imputed_inside_the_folds_or_the_rows_dropped(signal):
    gaps = signal.copy()
    gaps.loc[gaps.index[::7], "x2"] = np.nan
    cols = ["x1", "x2", "x3"]
    kept = ml("tree", gaps, {"target": "y", "features": cols}, {"missing": "impute", "test_size": 0})
    dropped = ml("tree", gaps, {"target": "y", "features": cols}, {"missing": "drop", "test_size": 0})
    assert kept.result["n_train"] == 300 and dropped.result["n_train"] == 300 - gaps["x2"].isna().sum()


def test_rows_without_a_target_get_a_prediction_and_the_rest_keep_their_group(signal):
    unlabeled = signal.copy()
    unlabeled.loc[unlabeled.index[:10], "cls"] = np.nan
    out = ml("forest", unlabeled, {"target": "cls", "features": ["x1", "x2"]}, {"n_estimators": 30, "folds": 3})
    predictions = offered(out, "predicciones")
    assert (predictions["conjunto"].iloc[:10] == "nueva (sin objetivo)").all()
    assert set(predictions["pred_cls"].iloc[:10]) <= {"up", "down"}
    assert predictions["conjunto"].iloc[10:].isin(["entrenamiento (validación cruzada)", "prueba"]).all()
    assert predictions[["prob_down", "prob_up"]].sum(axis=1).to_numpy() == pytest.approx(1)
    skipped = ml(
        "forest",
        unlabeled,
        {"target": "cls", "features": ["x1", "x2"]},
        {"n_estimators": 30, "folds": 3, "predict_new": False},
    )
    assert offered(skipped, "predicciones")["conjunto"].iloc[:10].isna().all()


def test_out_of_fold_predictions_are_honest_and_the_error_column_matches(signal):
    out = ml("tree", signal, {"target": "y", "features": ["x1", "x2"]}, {"test_size": 0.2, "folds": 4, "seed": 2})
    frame = offered(out, "predicciones")
    assert frame["error_y"].to_numpy() == pytest.approx((frame["y"] - frame["pred_y"]).to_numpy())
    train = frame[frame["conjunto"] == "entrenamiento (validación cruzada)"]
    assert np.sqrt(np.mean(train["error_y"] ** 2)) == pytest.approx(
        out.result["cv_mean"], rel=0.12
    )  # the pooled fold errors


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda f: f.assign(t=pd.date_range("2020", periods=len(f))), "fecha"),
        (lambda f: f.assign(g=[f"k{i}" for i in range(len(f))]), "categorías"),
        (lambda f: f.iloc[:20], "al menos"),
        (lambda f: f.assign(cls="same"), "una sola categoría"),
    ],
)
def test_bad_inputs_are_explained(signal, change, message):
    frame = change(signal)
    features = ["x1", "g"] + (["t"] if "t" in frame else [])
    with pytest.raises(InvalidInputError, match=message):
        ml("tree", frame, {"target": "cls", "features": features})


def test_a_class_with_one_row_and_a_target_among_the_predictors_are_refused(signal):
    lonely = signal.copy()
    lonely.loc[0, "cls"] = "rare"
    with pytest.raises(InvalidInputError, match="una sola observación"):
        ml("tree", lonely, {"target": "cls", "features": ["x1"]})
    with pytest.raises(InvalidInputError, match="objetivo"):
        ml("tree", signal, {"target": "y", "features": ["x1", "y"]})


# ------------------------------------------------------------------ metrics by their definitions
def test_classification_scores_match_scikit_learn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    proba = rng.dirichlet(np.ones(3), 300)
    guess = proba.argmax(axis=1)
    scores = classification_scores(y, Pred(guess, proba, True), 3, 2)
    assert scores["accuracy"] == pytest.approx(accuracy_score(y, guess))
    assert scores["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, guess))
    assert scores["f1_macro"] == pytest.approx(f1_score(y, guess, average="macro"))
    assert scores["precision_macro"] == pytest.approx(precision_score(y, guess, average="macro"))
    assert scores["recall_macro"] == pytest.approx(recall_score(y, guess, average="macro"))
    assert scores["kappa"] == pytest.approx(cohen_kappa_score(y, guess))
    assert scores["auc"] == pytest.approx(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    assert scores["log_loss"] == pytest.approx(log_loss(y, proba))
    binary = rng.integers(0, 2, 300)
    p1 = np.clip(0.3 * binary + rng.uniform(size=300) * 0.7, 0, 1)
    pair = Pred((p1 > 0.5).astype(int), np.column_stack([1 - p1, p1]), True)
    assert classification_scores(binary, pair, 2, 1)["auc"] == pytest.approx(roc_auc_score(binary, p1))
    assert classification_scores(binary, pair, 2, 0)["auc"] == pytest.approx(roc_auc_score(binary == 0, 1 - p1))


def test_decision_scores_give_an_auc_but_no_log_loss():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    decision = y * 1.5 + rng.normal(size=200)
    scores = classification_scores(y, Pred((decision > 0.7).astype(int), decision, False), 2, 1)
    assert scores["auc"] == pytest.approx(roc_auc_score(y, decision)) and np.isnan(scores["log_loss"])


def test_regression_scores_match_scikit_learn():
    rng = np.random.default_rng(2)
    y = rng.normal(10, 3, 200)
    guess = y + rng.normal(0.5, 1, 200)
    scores = regression_scores(y, Pred(guess))
    assert scores["rmse"] == pytest.approx(np.sqrt(mean_squared_error(y, guess)))
    assert scores["mae"] == pytest.approx(mean_absolute_error(y, guess))
    assert scores["r2"] == pytest.approx(r2_score(y, guess))
    assert scores["mape"] == pytest.approx(100 * np.mean(np.abs((y - guess) / y)))


@pytest.mark.parametrize("ties", [False, True])
def test_delong_matches_the_pairwise_definition(ties):
    rng = np.random.default_rng(5)
    event = rng.integers(0, 2, 120).astype(bool)
    scores = np.where(event, rng.normal(1, 1, 120), rng.normal(0, 1, 120))
    if ties:
        scores = np.round(scores)
    pos, neg = scores[event], scores[~event]
    psi = (pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])
    variance = psi.mean(axis=1).var(ddof=1) / len(pos) + psi.mean(axis=0).var(ddof=1) / len(neg)
    area, se, low, high = delong_auc(event, scores)
    assert area == pytest.approx(psi.mean()) and area == pytest.approx(roc_auc_score(event, scores))
    assert se == pytest.approx(np.sqrt(variance))
    assert low == pytest.approx(max(0, area - 1.959964 * se), abs=1e-5) and high <= 1


def test_the_roc_title_carries_the_delong_interval(signal):
    out = ml(
        "forest",
        signal,
        {"target": "cls", "features": ["x1", "x2"]},
        {"n_estimators": 30, "folds": 3, "positive": "up"},
    )
    low, high = out.result["auc_ci"]
    assert low < out.result["auc"] < high <= 1
    chart = next(b["spec"] for b in out.blocks if b["type"] == "chart" and b["spec"]["kind"] == "line")
    assert "IC 95" in chart["title"]


# ------------------------------------------------------------------ algorithm-specific results
def test_tree_rules_cover_every_row_and_the_pruned_tree_is_smaller(signal):
    cols = ["x1", "x2", "x3", "g"]
    full = ml(
        "tree", signal, {"target": "y", "features": cols}, {"max_depth": 6, "min_samples_leaf": 2, "test_size": 0}
    )
    pruned = ml(
        "tree",
        signal,
        {"target": "y", "features": cols},
        {"max_depth": 6, "min_samples_leaf": 2, "test_size": 0, "prune": "cv"},
    )
    assert pruned.result["leaves"] < full.result["leaves"]
    path = table(pruned, "Poda por costo-complejidad")
    assert sum(1 for r in path if value(r[3])) == 1


def test_tree_rules_partition_the_rows_and_start_from_the_strongest_variable(signal):
    out = ml(
        "tree",
        signal,
        {"target": "y", "features": ["x1", "x2", "x3"]},
        {"max_depth": 3, "min_samples_leaf": 5, "test_size": 0},
    )
    rules = table(out, "Reglas de decisión")
    assert len(rules) == out.result["leaves"] <= 8
    assert sum(r[2] for r in rules) == 300  # every row falls in exactly one leaf
    assert all(r[0].startswith("x1") for r in rules)  # the first cut of every path is on x1
    assert out.result["impurity_importance"]["x1"] > 0.9


def test_categorical_splits_are_written_as_equalities(signal):
    out = ml(
        "tree", signal, {"target": "y", "features": ["g"]}, {"max_depth": 2, "min_samples_leaf": 5, "test_size": 0}
    )
    assert any("g =" in r[0] or "g ≠" in r[0] for r in table(out, "Reglas de decisión"))


def test_the_tree_is_drawn_when_small_and_listed_as_rules_when_large(signal):
    small = ml("tree", signal, {"target": "y", "features": ["x1"]}, {"max_depth": 3, "test_size": 0})
    big = ml(
        "tree",
        signal,
        {"target": "y", "features": ["x1", "x2", "x3"]},
        {"max_depth": 12, "min_samples_leaf": 1, "test_size": 0},
    )
    assert any(b["type"] == "chart" and b["spec"]["kind"] == "raw" for b in small.blocks)
    assert not any(b["type"] == "chart" and b["spec"]["kind"] == "raw" for b in big.blocks)
    assert any(b["type"] == "alert" and "demasiados nodos" in b["message"] for b in big.blocks)


def test_boosting_curve_has_one_point_per_iteration_and_the_best_iteration(signal):
    out = ml(
        "boosting",
        signal,
        {"target": "y", "features": ["x1", "x2"]},
        {"max_iter": 60, "learning_rate": 0.3, "test_size": 0.25, "folds": 3},
    )
    chart = next(b["spec"] for b in out.blocks if b["type"] == "chart" and b["spec"]["kind"] == "line")
    assert len(chart["series"][0]["x"]) == 60 and 1 <= out.result["best_iteration"] <= 60
    assert out.result["cv_mean"] < 0.5 * out.result["baseline"]


def test_knn_search_picks_the_largest_k_within_one_standard_error(iris):
    out = ml(
        "knn",
        iris,
        {"target": "species", "features": IRIS_COLS},
        {"search": True, "k_max": 20, "folds": 5, "test_size": 0},
    )
    rows = [(int(r[0]), r[1], r[2]) for r in table(out, "Búsqueda del número de vecinos")]
    best = max(rows, key=lambda r: r[1])
    expected = max(k for k, mean, _ in rows if mean >= best[1] - best[2])
    assert out.result["n_neighbors"] == expected
    assert out.result["scores"]["accuracy"] > 0.93


def test_svm_mlp_and_bayes_work_and_bayes_is_classification_only(iris, signal):
    svm = ml("svm", iris, {"target": "species", "features": IRIS_COLS}, {"folds": 4})
    assert svm.result["scores"]["accuracy"] > 0.9 and np.isnan(svm.result["scores"].get("log_loss", np.nan))
    bayes = ml("naive_bayes", iris, {"target": "species", "features": IRIS_COLS}, {"folds": 4})
    assert bayes.result["scores"]["accuracy"] > 0.9
    net = ml(
        "mlp",
        signal,
        {"target": "y", "features": ["x1", "x2"]},
        {"hidden_layer_sizes": "16", "max_iter": 200, "folds": 3},
    )
    assert net.result["scores"]["r2"] > 0.7
    with pytest.raises(InvalidInputError, match="clasificación"):
        ml("naive_bayes", signal, {"target": "y", "features": ["x1"]})


def test_kernel_methods_refuse_huge_data_and_layers_are_validated():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=20_500), "b": rng.normal(size=20_500)})
    frame["y"] = (frame["a"] > 0).astype(int)
    with pytest.raises(InvalidInputError, match="admite hasta"):
        ml("svm", frame, {"target": "y", "features": ["a", "b"]}, {})
    assert parse_layers("64, 32") == (64, 32)
    for bad_text in ("", "a", "0", "9999", "1,2,3,4,5"):
        with pytest.raises(InvalidInputError):
            parse_layers(bad_text)


# ------------------------------------------------------------------ comparing and tuning
def test_compare_ranks_the_models_and_uses_the_corrected_t_test(iris):
    out = ml(
        "compare",
        iris,
        {"target": "species", "features": IRIS_COLS},
        {"algorithms": ["tree", "knn", "naive_bayes"], "folds": 5, "test_size": 0, "seed": 4},
    )
    rows = table(out, "Desempeño por modelo")
    means = [r[1] for r in rows[:-1]]
    assert means == sorted(means, reverse=True) and rows[-1][0].startswith("Línea base")
    folds = offered(out, "pliegues")
    best_name = value(rows[0][0])
    best = folds[folds["modelo"] == best_name]["accuracy"].to_numpy()
    other_name = rows[1][0]
    other = folds[folds["modelo"] == other_name]["accuracy"].to_numpy()
    splits = list(StratifiedKFold(5, shuffle=True, random_state=4).split(np.zeros(150), iris["species"]))
    ratio = np.mean([len(t) / len(tr) for tr, t in splits])
    d = best - other
    t = d.mean() / np.sqrt((1 / 5 + ratio) * d.var(ddof=1)) if d.var(ddof=1) > 0 else np.inf
    assert float(rows[1][-2]) == pytest.approx(2 * stats.t.sf(abs(t), 4)) and rows[1][-3] == pytest.approx(d.mean())
    assert out.result["best"] in {"tree", "knn", "naive_bayes"} and out.result["means"]["baseline"] == pytest.approx(
        1 / 3
    )


def test_compare_skips_what_does_not_apply(signal):
    out = ml(
        "compare",
        signal,
        {"target": "y", "features": ["x1", "x2"]},
        {"algorithms": ["tree", "naive_bayes"], "folds": 3},
    )
    assert any(b["type"] == "alert" and "Naive Bayes" in b["message"] for b in out.blocks)
    with pytest.raises(InvalidInputError):
        ml("compare", signal, {"target": "y", "features": ["x1"]}, {"algorithms": ["naive_bayes"]})


def test_tuning_finds_the_same_best_score_as_grid_search_cv(iris):
    out = ml(
        "tuning",
        iris,
        {"target": "species", "features": IRIS_COLS},
        {
            "algorithm": "tree",
            "grid": "max_depth = 1, 2, 4; min_samples_leaf = 1, 10",
            "folds": 5,
            "test_size": 0,
            "seed": 6,
        },
    )
    reference = GridSearchCV(
        DecisionTreeClassifier(random_state=6),
        {"max_depth": [1, 2, 4], "min_samples_leaf": [1, 10]},
        cv=StratifiedKFold(5, shuffle=True, random_state=6),
    ).fit(iris[IRIS_COLS], iris["species"])
    assert out.result["best_score"] == pytest.approx(reference.best_score_)
    assert out.result["evaluated"] == 6
    top = table(out, "Las 6 mejores combinaciones")
    assert top[0][2] == pytest.approx(reference.best_score_) or value(top[0][2]) == pytest.approx(reference.best_score_)


def test_random_search_samples_the_grid_and_a_default_space_exists_for_every_model(signal):
    out = ml(
        "tuning",
        signal,
        {"target": "cls", "features": ["x1", "x2"]},
        {"algorithm": "tree", "search": "random", "n_iter": 6, "folds": 3},
    )
    assert out.result["evaluated"] == 6
    for key, algorithm in ALGORITHMS.items():
        assert algorithm.space and set(algorithm.space) <= set(algorithm.defaults), key


def test_the_grid_text_is_parsed_and_checked():
    tree = ALGORITHMS["tree"]
    assert parse_grid("max_depth = 3, 5, None; criterion = gini, entropy", tree) == {
        "max_depth": [3, 5, None],
        "criterion": ["gini", "entropy"],
    }
    assert parse_grid("hidden_layer_sizes = 8, 16-8", ALGORITHMS["mlp"]) == {"hidden_layer_sizes": [(8,), (16, 8)]}
    assert parse_grid("alpha = 0.1, 1e-3", ALGORITHMS["mlp"]) == {"alpha": [0.1, 0.001]}
    for text, match in (
        ("depth = 3", "no es un ajuste"),
        ("max_depth 3", "Escriba la rejilla"),
        ("max_depth = ", "necesita"),
        (";", "vacía"),
    ):
        with pytest.raises(InvalidInputError, match=match):
            parse_grid(text, tree)


# ------------------------------------------------------------------ learning curve, partial dependence, selection
def test_learning_curve_ends_at_the_cross_validated_score_and_shows_the_gap(signal):
    cols = ["x1", "x2", "x3"]
    out = ml(
        "learning_curve",
        signal,
        {"target": "y", "features": cols},
        {"algorithm": "tree", "folds": 4, "points": 4, "seed": 5},
    )
    reference = -cross_val_score(
        DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=5),
        signal[cols],
        signal["y"],
        cv=KFold(4, shuffle=True, random_state=5),
        scoring="neg_root_mean_squared_error",
    )
    assert out.result["validation"][-1] == pytest.approx(reference.mean())
    assert (
        out.result["sizes"] == sorted(out.result["sizes"])
        and out.result["validation"][0] > out.result["validation"][-1]
    )  # RMSE falls with data
    assert out.result["train"][-1] < out.result["validation"][-1]
    assert table(out, "Diagnóstico")


def test_partial_dependence_equals_scikit_learns_and_ranks_the_real_variable(signal):
    cols = ["x1", "x2", "x3"]
    out = ml(
        "pdp",
        signal,
        {"target": "y", "features": cols, "focus": ["x1", "x2"]},
        {"algorithm": "forest", "grid": 25, "seed": 7, "ice": True},
    )
    model = RandomForestRegressor(n_estimators=200, max_features=0.33, random_state=7, n_jobs=1).fit(
        signal[cols], signal["y"]
    )
    expected = partial_dependence(model, signal[cols], ["x1"], grid_resolution=25, kind="average", method="brute")[
        "average"
    ][0]
    assert out.result["effects"]["x1"] == pytest.approx(expected.max() - expected.min())
    assert out.result["effects"]["x1"] > 4 * out.result["effects"]["x2"]
    ranking = table(out, "Amplitud del efecto")
    assert ranking[0][0] == "x1"


def test_partial_dependence_for_classes_and_categories(signal):
    out = ml(
        "pdp",
        signal,
        {"target": "cls", "features": ["x1", "x2", "g"], "focus": ["x1", "g"]},
        {"algorithm": "boosting", "positive": "up", "ice": False},
    )
    assert out.result["effects"]["x1"] > 0.5 and out.result["effects"]["g"] < 0.3
    flipped = ml(
        "pdp",
        signal,
        {"target": "cls", "features": ["x1", "x2", "g"], "focus": ["x1"]},
        {"algorithm": "boosting", "positive": "down", "ice": False},
    )
    assert flipped.result["effects"]["x1"] == pytest.approx(
        out.result["effects"]["x1"], rel=1e-6
    )  # P(down) = 1 − P(up)


def test_feature_selection_ranks_the_informative_variables_first():
    rng = np.random.default_rng(3)
    n = 400
    frame = pd.DataFrame({f"v{i}": rng.normal(size=n) for i in range(6)})
    frame["g"] = rng.choice(["a", "b"], n)
    frame["y"] = 2 * frame["v0"] - 1.5 * frame["v1"] + rng.normal(0, 0.5, n)
    out = ml(
        "select_features",
        frame,
        {"target": "y", "features": [*[f"v{i}" for i in range(6)], "g"]},
        {"methods": ["mutual_info", "anova", "forest", "lasso", "rfe"], "keep": 2},
    )
    assert set(out.result["selected"]) == {"v0", "v1"}
    assert set(out.result["ranking"][:2]) == {"v0", "v1"}
    ranking = offered(out, "seleccion")
    assert ranking["sugerida"].sum() == 2


# ------------------------------------------------------------------ anomalies and external predictions
@pytest.mark.parametrize("method", ["iforest", "lof", "elliptic"])
def test_anomaly_methods_find_planted_outliers(method):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(300, 3))
    x[:5] += 9
    out = ml(
        "anomaly",
        pd.DataFrame(x, columns=list("abc")),
        {"vars": list("abc")},
        {"method": method, "contamination": 0.05},
    )
    assert set(range(1, 6)) <= set(out.result["rows"])
    assert out.result["flagged"] == pytest.approx(15, abs=3)
    assert offered(out, "anomalias")["anomala"].sum() == out.result["flagged"]


def test_evaluate_matches_scikit_learn_for_classes(signal):
    out = ml("evaluate", standard_frame(), {"actual": "g", "predicted": "ps"}, {})
    frame = standard_frame()
    assert out.result["accuracy"] == pytest.approx(accuracy_score(frame["g"], frame["ps"]))
    assert out.result["kappa"] == pytest.approx(cohen_kappa_score(frame["g"], frame["ps"]))
    assert out.result["mcc"] == pytest.approx(matthews_corrcoef(frame["g"], frame["ps"]))
    assert np.array(out.result["confusion"]) == pytest.approx(confusion_matrix(frame["g"], frame["ps"]))


def test_evaluate_with_a_score_adds_roc_and_delong():
    frame = standard_frame()
    frame["pred_b"] = (frame["pb"] > 0.5).astype(int)
    out = ml("evaluate", frame, {"actual": "b", "predicted": "pred_b", "score": "pb"}, {"positive": "1"})
    assert out.result["auc"] == pytest.approx(roc_auc_score(frame["b"], frame["pb"]))
    area, se, low, high = delong_auc(frame["b"].to_numpy() == 1, frame["pb"].to_numpy())
    row = table(out, "AUC con intervalo")[0]
    assert row[:4] == pytest.approx([area, se, low, high])
    assert any(b["type"] == "table" and b["title"].startswith("Umbrales") for b in out.blocks)


def test_evaluate_regression_reports_bias_and_agreement():
    frame = standard_frame()
    out = ml("evaluate", frame, {"actual": "y", "predicted": "py"}, {})
    assert out.result["rmse"] == pytest.approx(np.sqrt(mean_squared_error(frame["y"], frame["py"])))
    assert out.result["r2"] == pytest.approx(r2_score(frame["y"], frame["py"]))
    bias = table(out, "Distribución del error")[0][1]
    assert bias == pytest.approx((frame["y"] - frame["py"]).mean()) and bias < 0  # the guess is biased upwards by 0.3
    assert any(b["type"] == "chart" and "Bland-Altman" in b["spec"]["title"] for b in out.blocks)
    with pytest.raises(InvalidInputError, match="una sola categoría"):
        ml("evaluate", frame.assign(z="x", w="x"), {"actual": "z", "predicted": "w"}, {})
    with pytest.raises(InvalidInputError, match="distintas"):
        ml("evaluate", frame, {"actual": "y", "predicted": "y"}, {})
