"""Multivariate analyses against numpy / scipy / scikit-learn / statsmodels and published results."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from statsmodels.multivariate.cancorr import CanCorr

from sigma.domain.analysis.multivariate import (
    bartlett_sphericity,
    classical_mds,
    correspondence,
    cronbach_alpha,
    kmo,
    ml_chi_square,
    parallel_analysis,
)
from sigma.domain.errors import InvalidInputError
from tests.analysis_data import offered, run, standard_frame


def mv(analysis_id, frame, roles=None, params=None):
    return run("multivariado", analysis_id, frame, roles, params)


def table(output, title_start):
    return next(b for b in output.blocks if b["type"] == "table" and b["title"].startswith(title_start))["rows"]


def value(cell):
    return getattr(cell, "value", cell)


@pytest.fixture(scope="module")
def iris():
    data = load_iris()
    frame = pd.DataFrame(data.data, columns=["sl", "sw", "pl", "pw"])
    frame["species"] = pd.Categorical.from_codes(data.target, data.target_names).astype(str)
    return frame


def blobs(centers, n_each=60, sd=0.5, seed=3):
    rng = np.random.default_rng(seed)
    parts = [rng.normal(c, sd, size=(n_each, len(c))) for c in centers]
    frame = pd.DataFrame(np.vstack(parts), columns=["a", "b", "c"][: len(centers[0])])
    frame["truth"] = np.repeat(np.arange(len(centers)), n_each)
    return frame


# ------------------------------------------------------------------ principal components
def test_pca_on_iris_matches_the_eigendecomposition(iris):
    cols = ["sl", "sw", "pl", "pw"]
    out = mv("pca", iris, {"vars": cols}, {"rule": "fixed", "components": 4})
    corr = np.corrcoef(iris[cols].to_numpy(), rowvar=False)
    values, vectors = np.linalg.eigh(corr)
    values, vectors = values[::-1], vectors[:, ::-1]
    assert values == pytest.approx(
        [2.91849, 0.91403, 0.14676, 0.02071], abs=1e-4
    )  # published for the iris correlations
    assert [r[1] for r in table(out, "Valores propios")] == pytest.approx(values)
    assert out.result["variance_share"] == pytest.approx(values / 4)
    loadings = np.array([row[1:5] for row in table(out, "Cargas")])
    assert np.abs(loadings) == pytest.approx(np.abs(vectors * np.sqrt(values)), abs=1e-9)
    z = (iris[cols] - iris[cols].mean()) / iris[cols].std(ddof=1)
    scores = offered(out, "puntuaciones")[["CP1", "CP2", "CP3", "CP4"]].to_numpy()
    assert np.abs(scores) == pytest.approx(np.abs(z.to_numpy() @ vectors), abs=1e-9)
    assert scores.var(axis=0, ddof=1) == pytest.approx(values)


def test_pca_retention_rules(iris):
    cols = ["sl", "sw", "pl", "pw"]
    assert mv("pca", iris, {"vars": cols}, {"rule": "kaiser"}).result["components"] == 1
    assert mv("pca", iris, {"vars": cols}, {"rule": "variance", "variance": 95}).result["components"] == 2
    assert mv("pca", iris, {"vars": cols}, {"rule": "fixed", "components": 3}).result["components"] == 3


def test_pca_varimax_keeps_the_communalities(iris):
    cols = ["sl", "sw", "pl", "pw"]
    plain = mv("pca", iris, {"vars": cols}, {"rule": "fixed", "components": 2, "rotation": "none"})
    rotated = mv("pca", iris, {"vars": cols}, {"rule": "fixed", "components": 2, "rotation": "varimax"})
    assert [r[-1] for r in table(rotated, "Cargas")] == pytest.approx([r[-1] for r in table(plain, "Cargas")])


def test_kmo_is_one_half_for_any_two_variables_and_bartlett_rejects_the_identity():
    r = np.array([[1, 0.6], [0.6, 1]])
    assert kmo(r)[0] == pytest.approx(0.5)
    chi2, df, p = bartlett_sphericity(np.eye(4), 100)
    assert chi2 == pytest.approx(0) and df == 6 and p == pytest.approx(1)
    r3 = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    chi2, df, p = bartlett_sphericity(r3, 100)
    assert chi2 == pytest.approx(-(99 - 11 / 6) * np.log(0.5)) and df == 3 and p < 1e-4
    assert 0.5 < kmo(r3)[0] < 1


def test_parallel_analysis_threshold_is_a_decreasing_set_of_eigenvalues():
    threshold = parallel_analysis(200, 6, reps=100)
    assert np.all(np.diff(threshold) <= 1e-12)
    assert threshold[0] > 1 > threshold[-1] and threshold.sum() >= 6 - 1e-9


# ------------------------------------------------------------------ factor analysis
@pytest.fixture(scope="module")
def two_factors():
    rng = np.random.default_rng(11)
    n = 1500
    f = rng.normal(size=(n, 2))
    lam = np.array([[0.8, 0], [0.7, 0], [0.6, 0], [0, 0.8], [0, 0.7], [0, 0.6]])
    x = f @ lam.T + rng.normal(size=(n, 6)) * np.sqrt(1 - (lam**2).sum(axis=1))
    return pd.DataFrame(x, columns=[f"v{i}" for i in range(1, 7)]), lam


@pytest.mark.parametrize("method", ["pa", "ml"])
def test_factor_analysis_recovers_the_loadings(two_factors, method):
    frame, lam = two_factors
    out = mv("factor", frame, {"vars": list(frame.columns)}, {"factors": 2, "method": method, "rotation": "varimax"})
    found = np.array([[value(c) for c in row[1:3]] for row in table(out, "Cargas")], float)
    order = [0, 1] if abs(found[0, 0]) > abs(found[0, 1]) else [1, 0]
    assert np.abs(found[:, order]) == pytest.approx(lam, abs=0.08)
    assert out.result["kmo"] > 0.6


def test_ml_chi_square_accepts_the_true_number_of_factors_and_rejects_fewer(two_factors):
    frame, _ = two_factors
    cols = list(frame.columns)
    right = mv("factor", frame, {"vars": cols}, {"factors": 2, "method": "ml"})
    wrong = mv("factor", frame, {"vars": cols}, {"factors": 1, "method": "ml"})
    chi_right, df_right, p_right = table(right, "Prueba de bondad")[0][:3]
    chi_wrong, df_wrong, p_wrong = table(wrong, "Prueba de bondad")[0][:3]
    assert df_right == 4 and df_wrong == 9
    assert p_right > 0.01 and p_wrong < 1e-6 and chi_wrong > chi_right


def test_ml_chi_square_is_zero_for_a_perfect_fit_and_has_the_textbook_degrees_of_freedom():
    sample = np.array([[1, 0.5, 0.4], [0.5, 1, 0.3], [0.4, 0.3, 1]])
    chi2, df, p = ml_chi_square(sample, sample, 200, 1)
    assert df == 0 and np.isnan(chi2)  # three variables and one factor: the model is just identified
    chi2, df, p = ml_chi_square(sample, sample, 200, 0)
    assert chi2 == pytest.approx(0, abs=1e-9) and df == 3 and p == pytest.approx(1)
    assert ml_chi_square(np.eye(6), np.eye(6), 100, 2)[1] == 4


# ------------------------------------------------------------------ reliability
def test_cronbach_alpha_matches_the_definition():
    rng = np.random.default_rng(5)
    base = rng.normal(size=300)
    items = np.column_stack([base + rng.normal(scale=s, size=300) for s in (0.6, 0.8, 1.0, 1.2)])
    k = items.shape[1]
    expected = k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum() / items.sum(axis=1).var(ddof=1))
    assert cronbach_alpha(items) == pytest.approx(expected)
    frame = pd.DataFrame(items, columns=list("abcd"))
    out = mv("reliability", frame, {"vars": list("abcd")})
    assert out.result["alpha"] == pytest.approx(expected)
    deleted = {r[0]: r[-1] for r in table(out, "Análisis de los ítems")}
    assert deleted["d"] == pytest.approx(cronbach_alpha(items[:, :3]))
    assert out.result["omega"] == pytest.approx(out.result["alpha"], abs=0.05)


def test_reliability_reverse_scored_item_is_flipped():
    rng = np.random.default_rng(6)
    base = rng.normal(size=200)
    frame = pd.DataFrame(
        {
            "a": base + rng.normal(size=200) * 0.5,
            "b": base + rng.normal(size=200) * 0.5,
            "c": -base + rng.normal(size=200) * 0.5,
        }
    )
    wrong = mv("reliability", frame, {"vars": ["a", "b", "c"]})
    right = mv("reliability", frame, {"vars": ["a", "b", "c"], "reverse": ["c"]})
    assert right.result["alpha"] > 0.8 > wrong.result["alpha"]


# ------------------------------------------------------------------ canonical correlation
def test_canonical_correlations_of_iris_match_statsmodels(iris):
    out = mv("cca", iris, {"x_vars": ["sl", "sw"], "y_vars": ["pl", "pw"]})
    reference = CanCorr(iris[["pl", "pw"]], iris[["sl", "sw"]])
    assert out.result["correlations"] == pytest.approx(list(reference.cancorr))
    assert out.result["correlations"][0] == pytest.approx(0.9408, abs=1e-3)
    assert out.result["wilks_p"][0] < 1e-10


# ------------------------------------------------------------------ correspondence analysis
SMOKING = np.array([[4, 2, 3, 2], [4, 3, 7, 4], [25, 10, 12, 4], [18, 24, 33, 13], [10, 6, 7, 2]], float)


def test_correspondence_on_the_greenacre_smoking_table():
    result = correspondence(SMOKING)
    assert result["inertia"] == pytest.approx([0.07476, 0.01002, 0.00041], abs=2e-5)
    assert result["total"][0] == pytest.approx(0.08519, abs=2e-5)
    chi2 = stats.chi2_contingency(SMOKING)[0]
    assert result["total"][0] * SMOKING.sum() == pytest.approx(chi2)
    assert result["row_inertia"].sum() == pytest.approx(result["total"][0])
    assert result["col_inertia"].sum() == pytest.approx(result["total"][0])
    assert np.sum(result["row_mass"] * result["rows"][:, 0] ** 2) == pytest.approx(
        result["inertia"][0]
    )  # standard property


def test_correspondence_spec_counts_a_categorical_pair_or_uses_frequencies():
    rows = ["SM", "JM", "SE", "JE", "SC"]
    cols = ["none", "light", "medium", "heavy"]
    long = pd.DataFrame(
        [(r, c, SMOKING[i, j]) for i, r in enumerate(rows) for j, c in enumerate(cols)], columns=["staff", "smoke", "n"]
    )
    out = mv("correspondence", long, {"row": "staff", "col": "smoke", "count": "n"})
    assert out.result["total_inertia"] == pytest.approx(0.08519, abs=2e-5)
    expanded = long.loc[long.index.repeat(long["n"].astype(int))].reset_index(drop=True)
    again = mv("correspondence", expanded, {"row": "staff", "col": "smoke"})
    assert again.result["total_inertia"] == pytest.approx(out.result["total_inertia"])


# ------------------------------------------------------------------ multidimensional scaling
def test_classical_mds_recovers_a_planar_configuration():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(30, 2))
    d = np.linalg.norm(points[:, None] - points[None], axis=2)
    coords, values = classical_mds(d, 2)
    assert pdist(coords) == pytest.approx(pdist(points))
    assert values[2:] == pytest.approx(0, abs=1e-9) and values[0] >= values[1] > 0


def test_mds_spec_reproduces_the_distances_and_offers_the_map():
    frame = standard_frame()
    cols = ["x1", "x2", "x3", "y"]
    out = mv("mds", frame, {"vars": cols}, {"method": "classical", "dimensions": 2})
    assert 0 < out.result["stress"] < 0.6
    coords = offered(out, "coordenadas")
    assert list(coords.columns[-2:]) == ["dim_1", "dim_2"] and coords["dim_1"].notna().all()


# ------------------------------------------------------------------ Mahalanobis
def test_mahalanobis_classical_matches_the_formula_and_flags_by_chi_square():
    frame = standard_frame()
    cols = ["y", "x1", "x2"]
    out = mv("mahalanobis", frame, {"vars": cols}, {"method": "classical", "alpha": 0.05})
    x = frame[cols].to_numpy(float)
    diff = x - x.mean(axis=0)
    d2 = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(np.cov(x, rowvar=False)), diff)
    result = offered(out, "distancias")
    assert result["mahalanobis_d2"].to_numpy() == pytest.approx(d2)
    cutoff = stats.chi2.ppf(0.95, 3)
    assert out.result["cutoff"] == pytest.approx(cutoff)
    assert out.result["flagged"] == int((d2 > cutoff).sum())
    assert result["valor_p"].to_numpy() == pytest.approx(stats.chi2.sf(d2, 3))


def test_robust_mahalanobis_finds_planted_outliers():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(200, 3))
    x[:6] += 7
    out = mv(
        "mahalanobis", pd.DataFrame(x, columns=list("abc")), {"vars": list("abc")}, {"method": "robust", "alpha": 0.001}
    )
    assert set(range(1, 7)) <= set(out.result["rows"])


# ------------------------------------------------------------------ hierarchical clustering
CENTERS = [(0, 0, 0), (4, 0, 0), (0, 4, 4)]


def test_hierarchical_matches_scipy_and_sklearn():
    frame = blobs(CENTERS)
    cols = ["a", "b", "c"]
    out = mv("hierarchical", frame, {"vars": cols}, {"linkage": "ward", "clusters": 3, "standardize": False})
    labels = offered(out, "grupos")["grupo"].to_numpy()
    reference = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(frame[cols])
    assert adjusted_rand_score(reference, labels) == pytest.approx(1)
    assert adjusted_rand_score(frame["truth"], labels) == pytest.approx(1)
    x = frame[cols].to_numpy()
    assert out.result["cophenetic"] == pytest.approx(cophenet(linkage(x, "ward"), pdist(x))[0])
    assert out.result["suggested"] == 3 and sorted(out.result["sizes"]) == [60, 60, 60]
    assert out.result["silhouette"] > 0.6


@pytest.mark.parametrize("method", ["complete", "average", "single"])
def test_hierarchical_other_linkages_cut_like_scipy(method):
    from scipy.cluster.hierarchy import fcluster

    frame = blobs(CENTERS, n_each=30)
    cols = ["a", "b", "c"]
    out = mv("hierarchical", frame, {"vars": cols}, {"linkage": method, "clusters": 3, "standardize": False})
    labels = offered(out, "grupos")["grupo"].to_numpy()
    expected = fcluster(linkage(frame[cols].to_numpy(), method), 3, criterion="maxclust")
    assert adjusted_rand_score(expected, labels) == pytest.approx(1)


def test_hierarchical_draws_the_dendrogram_and_truncates_large_trees():
    chart = next(
        b["spec"]
        for b in mv("hierarchical", blobs(CENTERS, n_each=10), {"vars": ["a", "b", "c"]}, {"clusters": 3}).blocks
        if b["type"] == "chart"
    )
    assert chart["kind"] == "raw" and len(chart["data"][0]["x"]) > 0
    assert len(chart["layout"]["xaxis"]["tickvals"]) == 30
    big = mv("hierarchical", blobs(CENTERS, n_each=100), {"vars": ["a", "b", "c"]}, {"clusters": 3})
    tree = next(b["spec"] for b in big.blocks if b["type"] == "chart")
    assert len(tree["layout"]["xaxis"]["tickvals"]) == 40  # truncated to the last 40 merges


def test_hierarchical_rejects_ward_with_a_non_euclidean_distance():
    with pytest.raises(InvalidInputError, match="euclídea"):
        mv("hierarchical", blobs(CENTERS), {"vars": ["a", "b", "c"]}, {"linkage": "ward", "distance": "cityblock"})


# ------------------------------------------------------------------ k-means
def test_kmeans_finds_the_blobs_and_matches_sklearn():
    frame = blobs(CENTERS)
    cols = ["a", "b", "c"]
    out = mv("kmeans", frame, {"vars": cols}, {"standardize": False, "k_max": 6, "restarts": 10, "seed": 5})
    assert out.result["suggested"] == 3 and out.result["clusters"] == 3
    labels = offered(out, "grupos")["grupo"].to_numpy()
    assert adjusted_rand_score(frame["truth"], labels) == pytest.approx(1)
    reference = KMeans(3, n_init=10, random_state=5).fit(frame[cols])
    assert out.result["inertia"] == pytest.approx(reference.inertia_, rel=1e-6)
    elbow = {r[0]: r[1] for r in table(out, "Elección del número")}
    assert all(elbow[k] > elbow[k + 1] for k in range(2, 6))  # inertia always falls with k
    assert out.result["silhouette"] > 0.6


def test_kmeans_respects_a_fixed_number_of_groups_and_validates():
    frame = blobs(CENTERS, n_each=20)
    out = mv("kmeans", frame, {"vars": ["a", "b", "c"]}, {"clusters": 4, "k_max": 3})
    assert out.result["clusters"] == 4 and len(out.result["sizes"]) == 4
    assert {r[0] for r in table(out, "Elección del número")} == {2, 3, 4}
    with pytest.raises(InvalidInputError):
        mv("kmeans", frame.head(12), {"vars": ["a", "b", "c"]}, {"clusters": 12})


# ------------------------------------------------------------------ Gaussian mixtures
def test_mixture_selects_the_true_components_and_weights():
    rng = np.random.default_rng(4)
    a = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 210)
    b = rng.multivariate_normal([5, 5], [[1, -0.5], [-0.5, 1]], 90)
    frame = pd.DataFrame(np.vstack([a, b]), columns=["a", "b"])
    out = mv("mixture", frame, {"vars": ["a", "b"]}, {"standardize": False, "k_max": 4, "restarts": 3, "seed": 1})
    assert out.result["components"] == 2
    assert sorted(out.result["weights"]) == pytest.approx([0.3, 0.7], abs=0.05)
    reference = GaussianMixture(2, n_init=3, random_state=1).fit(frame)
    assert out.result["bic"] == pytest.approx(reference.bic(frame), rel=1e-3)
    groups = offered(out, "grupos")
    assert adjusted_rand_score(np.repeat([0, 1], [210, 90]), groups["grupo"]) > 0.97
    assert groups[["prob_grupo_1", "prob_grupo_2"]].sum(axis=1).to_numpy() == pytest.approx(1)
    means = {tuple(round(float(value(c)), 0) for c in r[2:]) for r in table(out, "Pesos y medias")}
    assert means == {(0.0, 0.0), (5.0, 5.0)}


def test_mixture_rejects_more_components_than_the_data_can_support():
    frame = blobs(CENTERS, n_each=15)
    with pytest.raises(InvalidInputError, match="como máximo"):
        mv("mixture", frame, {"vars": ["a", "b", "c"]}, {"components": 20})


# ------------------------------------------------------------------ discriminant analysis
def test_lda_on_iris_reproduces_the_published_functions(iris):
    out = mv("discriminant", iris, {"group": "species", "vars": ["sl", "sw", "pl", "pw"]}, {"kind": "lda"})
    functions = table(out, "Funciones discriminantes")
    assert [value(r[1]) for r in functions] == pytest.approx([32.1919, 0.2854], abs=1e-3)
    assert [value(r[3]) for r in functions] == pytest.approx([0.9848, 0.4712], abs=1e-3)
    assert value(functions[0][4]) == pytest.approx(0.0234, abs=5e-4)
    assert functions[0][7] < 1e-30
    assert out.result["accuracy"] == pytest.approx(147 / 150)
    reference = LinearDiscriminantAnalysis().fit(iris[["sl", "sw", "pl", "pw"]], iris["species"])
    assert out.result["accuracy"] == pytest.approx(reference.score(iris[["sl", "sw", "pl", "pw"]], iris["species"]))
    assert out.result["cv_accuracy"] > 0.95 and out.result["baseline"] == pytest.approx(1 / 3)
    probabilities = offered(out, "clasificacion")
    assert probabilities[["prob_setosa", "prob_versicolor", "prob_virginica"]].sum(axis=1).to_numpy() == pytest.approx(
        1
    )
    assert (probabilities["grupo_predicho"] == iris["species"]).mean() == pytest.approx(147 / 150)


def test_lda_explained_separation_matches_sklearn(iris):
    cols = ["sl", "sw", "pl", "pw"]
    out = mv("discriminant", iris, {"group": "species", "vars": cols}, {"kind": "lda"})
    reference = LinearDiscriminantAnalysis(solver="eigen").fit(iris[cols], iris["species"])
    shares = [value(r[2]) / 100 for r in table(out, "Funciones discriminantes")]
    assert shares == pytest.approx(list(reference.explained_variance_ratio_[:2]), abs=1e-6)


def test_qda_and_equal_priors_run_and_report_box_m(iris):
    cols = ["sl", "sw", "pl", "pw"]
    qda = mv("discriminant", iris, {"group": "species", "vars": cols}, {"kind": "qda", "priors": "equal"})
    assert qda.result["accuracy"] >= 0.97
    box = table(qda, "Igualdad de matrices")[0]
    assert box[1] == 20 and box[2] < 1e-10  # the iris covariances are known to differ (Box M = 146.7 in SPSS)


def test_discriminant_validates_the_groups(iris):
    cols = ["sl", "sw", "pl", "pw"]
    tiny = iris.groupby("species").head(2)
    with pytest.raises(InvalidInputError, match="3 observaciones"):
        mv("discriminant", tiny, {"group": "species", "vars": cols}, {})
    few = iris.groupby("species").head(4)
    with pytest.raises(InvalidInputError, match="cuadrático"):
        mv("discriminant", few, {"group": "species", "vars": cols}, {"kind": "qda"})
    one = iris[iris["species"] == "setosa"]
    with pytest.raises(InvalidInputError):
        mv("discriminant", one, {"group": "species", "vars": cols}, {})


def test_poorly_separated_groups_are_flagged_as_no_better_than_the_majority():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x": rng.normal(size=120), "y": rng.normal(size=120), "g": rng.choice(["a", "b", "c"], 120)})
    out = mv("discriminant", frame, {"group": "g", "vars": ["x", "y"]}, {})
    assert any(b["type"] == "alert" and "mayoritaria" in b["message"] for b in out.blocks)
