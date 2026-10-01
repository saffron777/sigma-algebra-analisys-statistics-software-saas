import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from sigma.domain.errors import InvalidInputError
from sigma.domain.statistics.survey_estimation import estimate_survey, proportion_interval
from sigma.domain.statistics.survey_sampling import (
    INCLUSION_COLUMN,
    WEIGHT_COLUMN,
    allocate_strata,
    draw_sample,
    sample_size_mean,
    sample_size_proportion,
    z_value,
)
from sigma.domain.statistics.survey_tabulation import crosstab, likert_summary


@pytest.fixture
def frame():
    """Sampling frame: 1000 households in 4 districts (sizes 400/300/200/100) with 50 blocks each."""
    rng = np.random.default_rng(5)
    district = np.repeat(["Norte", "Sur", "Este", "Oeste"], [400, 300, 200, 100])
    return pd.DataFrame(
        {
            "id": np.arange(1000),
            "distrito": district,
            "manzana": np.arange(1000) // 20,
            "ingreso": rng.normal(1000, 200, 1000),
            "aprueba": rng.choice(["Sí", "No"], 1000, p=[0.55, 0.45]),
        }
    )


# ------------------------------------------------------------------ sample size
def test_z_values():
    assert z_value(0.95) == pytest.approx(1.959964, abs=1e-6)
    assert z_value(0.99) == pytest.approx(2.575829, abs=1e-6)


def test_classic_sample_size_385():
    result = sample_size_proportion(0.95, 0.05, 0.5)
    assert result.n_infinite == pytest.approx(384.146, abs=1e-3)
    assert result.n_final == 385


def test_finite_population_and_design_effect():
    assert sample_size_proportion(0.95, 0.05, 0.5, population=1000).n_final == 278
    assert sample_size_proportion(0.95, 0.05, 0.5, design_effect=1.5).n_final == math.ceil(384.146 * 1.5)
    assert sample_size_proportion(0.95, 0.05, 0.5, population=300, design_effect=3).n_final <= 300


def test_sample_size_for_mean():
    result = sample_size_mean(0.95, margin_of_error=2, std_dev=10)
    assert result.n_final == math.ceil((1.959964 * 10 / 2) ** 2)
    assert len(result.curve) > 5 and result.curve[0][1] > result.curve[-1][1]


@pytest.mark.parametrize(
    "call",
    [
        lambda: sample_size_proportion(0.95, 0, 0.5),
        lambda: sample_size_proportion(1.2, 0.05),
        lambda: sample_size_proportion(0.95, 0.05, 1.0),
        lambda: sample_size_proportion(0.95, 0.05, population=1),
        lambda: sample_size_mean(0.95, 1, 0),
        lambda: sample_size_proportion(0.95, 0.05, design_effect=0.5),
    ],
)
def test_sample_size_validation(call):
    with pytest.raises(InvalidInputError):
        call()


# ------------------------------------------------------------------ allocation and drawing
def test_proportional_and_equal_allocation(frame):
    plan = {s.stratum: s.sample_size for s in allocate_strata(frame, "distrito", 100, "proportional")}
    assert plan == {"Este": 20, "Norte": 40, "Oeste": 10, "Sur": 30}
    equal = [s.sample_size for s in allocate_strata(frame, "distrito", 100, "equal")]
    assert equal == [25, 25, 25, 25]


def test_neyman_allocation_follows_variability(frame):
    frame = frame.assign(ingreso=np.where(frame["distrito"] == "Oeste", frame["ingreso"] * 5, frame["ingreso"]))
    plan = {s.stratum: s.sample_size for s in allocate_strata(frame, "distrito", 100, "neyman", "ingreso")}
    assert sum(plan.values()) == 100
    assert plan["Oeste"] > 10  # more than its proportional share because it is far more variable
    with pytest.raises(InvalidInputError):
        allocate_strata(frame, "distrito", 100, "neyman")


def test_allocation_always_sums_to_total(frame):
    for total in (4, 7, 33, 999):
        assert sum(s.sample_size for s in allocate_strata(frame, "distrito", total)) == total


def test_srs_draw_is_reproducible_and_weighted(frame):
    a = draw_sample(frame, "srs", 100, seed=123)
    b = draw_sample(frame, "srs", 100, seed=123)
    assert a.sample["id"].tolist() == b.sample["id"].tolist()
    assert a.sample["id"].is_unique and len(a.sample) == 100
    assert a.sample[INCLUSION_COLUMN].iloc[0] == pytest.approx(0.1)
    assert a.sample[WEIGHT_COLUMN].sum() == pytest.approx(1000)
    assert draw_sample(frame, "srs", 100, seed=124).sample["id"].tolist() != a.sample["id"].tolist()


def test_unseeded_draw_reports_the_seed(frame):
    result = draw_sample(frame, "srs", 50)
    assert result.seed is not None
    assert draw_sample(frame, "srs", 50, seed=result.seed).sample["id"].tolist() == result.sample["id"].tolist()


def test_systematic_draw_spacing(frame):
    result = draw_sample(frame, "systematic", 100, seed=9)
    assert len(result.sample) == 100 and result.interval == pytest.approx(10.0)
    gaps = np.diff(result.sample["id"].to_numpy())
    assert set(gaps) <= {10}


def test_stratified_draw_respects_allocation(frame):
    result = draw_sample(frame, "stratified", 100, seed=1, strata_column="distrito")
    counts = result.sample["distrito"].value_counts().to_dict()
    assert counts == {"Norte": 40, "Sur": 30, "Este": 20, "Oeste": 10}
    assert result.sample[WEIGHT_COLUMN].sum() == pytest.approx(1000)  # weights reproduce N


def test_cluster_draw_takes_whole_clusters(frame):
    result = draw_sample(frame, "cluster", seed=3, cluster_column="manzana", n_clusters=5)
    assert result.sample["manzana"].nunique() == 5 and len(result.sample) == 100
    assert result.sample[INCLUSION_COLUMN].iloc[0] == pytest.approx(5 / 50)


def test_draw_validation(frame):
    with pytest.raises(InvalidInputError):
        draw_sample(frame, "srs", 5000)
    with pytest.raises(InvalidInputError):
        draw_sample(frame, "stratified", 50)
    with pytest.raises(InvalidInputError):
        draw_sample(frame, "cluster", cluster_column="manzana", n_clusters=500)
    with pytest.raises(InvalidInputError):
        draw_sample(frame.assign(peso_diseno=1), "srs", 10)
    with pytest.raises(InvalidInputError):
        draw_sample(frame, "magic", 10)


# ------------------------------------------------------------------ estimation
@pytest.mark.parametrize(
    ("method", "reference"),
    [
        ("wald", "normal"),
        ("wilson", "wilson"),
        ("agresti_coull", "agresti_coull"),
        ("clopper_pearson", "beta"),
    ],
)
@pytest.mark.parametrize(("x", "n"), [(55, 100), (3, 40), (0, 25), (180, 200)])
def test_proportion_intervals_match_statsmodels(method, reference, x, n):
    low, high = proportion_interval(x / n, n, 0.95, method)
    expected = proportion_confint(x, n, alpha=0.05, method=reference)
    assert (low, high) == pytest.approx(expected, abs=1e-6)


def test_srs_proportion_estimate(frame):
    sample = draw_sample(frame, "srs", 400, seed=2).sample
    result = estimate_survey(sample, "aprueba", "proportion", 0.95, ["wald"])
    yes = next(r for r in result.rows if r.label == "Sí")
    p = (sample["aprueba"] == "Sí").mean()
    assert yes.estimate == pytest.approx(p)
    assert yes.std_error == pytest.approx(math.sqrt(p * (1 - p) / 400))
    assert yes.margin_of_error == pytest.approx(1.959964 * math.sqrt(p * (1 - p) / 400), rel=1e-4)
    assert sum(r.estimate for r in result.rows) == pytest.approx(1.0)


def test_finite_population_correction_narrows_interval(frame):
    sample = draw_sample(frame, "srs", 400, seed=2).sample
    plain = estimate_survey(sample, "aprueba", methods=["wald"]).rows[0]
    corrected = estimate_survey(sample, "aprueba", methods=["wald"], population_size=1000).rows[0]
    assert corrected.margin_of_error < plain.margin_of_error
    fpc = math.sqrt((1000 - 400) / 999)
    assert corrected.std_error == pytest.approx(plain.std_error * fpc)


def test_mean_estimate_uses_t_interval(frame):
    sample = draw_sample(frame, "srs", 200, seed=4).sample
    result = estimate_survey(sample, "ingreso", "mean", 0.95)
    row = result.rows[0]
    values = sample["ingreso"]
    low, high = stats.t.interval(0.95, 199, loc=values.mean(), scale=stats.sem(values))
    assert (row.ci_low, row.ci_high) == pytest.approx((low, high))


def test_stratified_weighted_estimate_corrects_unequal_allocation(frame):
    """Equal allocation over-represents small districts; weights must restore the population mean."""
    sample = draw_sample(frame, "stratified", 200, seed=6, strata_column="distrito", allocation="equal").sample
    naive = sample["ingreso"].mean()
    weighted = estimate_survey(sample, "ingreso", "mean", weight_column=WEIGHT_COLUMN, strata_column="distrito")
    truth = frame["ingreso"].mean()
    assert weighted.rows[0].estimate == pytest.approx(np.average(sample["ingreso"], weights=sample[WEIGHT_COLUMN]))
    assert abs(weighted.rows[0].estimate - truth) <= abs(naive - truth) + 25
    assert weighted.design_effect is not None and weighted.effective_n > 0
    assert weighted.rows[0].ci_low < truth < weighted.rows[0].ci_high or weighted.rows[0].margin_of_error > 0


def test_cluster_design_has_larger_design_effect_with_correlated_clusters():
    rng = np.random.default_rng(8)
    clusters = np.repeat(np.arange(30), 20)
    cluster_effect = rng.normal(0, 1, 30)[clusters]
    frame = pd.DataFrame({"cluster": clusters, "valor": cluster_effect + rng.normal(0, 0.3, 600)})
    result = estimate_survey(frame, "valor", "mean", cluster_column="cluster")
    assert result.design_effect > 5  # strong intra-cluster correlation
    srs = estimate_survey(frame, "valor", "mean")
    assert result.rows[0].margin_of_error > srs.rows[0].margin_of_error * 2


def test_estimation_validation(frame):
    with pytest.raises(InvalidInputError):
        estimate_survey(frame, "ingreso", "proportion")  # too many categories
    with pytest.raises(InvalidInputError):
        estimate_survey(frame, "aprueba", "mean")  # not numeric
    with pytest.raises(InvalidInputError):
        estimate_survey(frame.head(300), "aprueba", population_size=100)
    with pytest.raises(InvalidInputError):
        estimate_survey(frame, "aprueba", methods=["magic"])


# ------------------------------------------------------------------ tabulation
def test_crosstab_matches_scipy(frame):
    result = crosstab(frame, "distrito", "aprueba")
    expected = stats.chi2_contingency(pd.crosstab(frame["distrito"], frame["aprueba"]), correction=False)
    assert result.chi2 == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.df == 3 and result.n == 1000
    assert np.array(result.row_percent).sum(axis=1) == pytest.approx([100] * 4)
    assert 0 <= result.cramers_v <= 1


def test_likert_summary_and_cronbach_alpha():
    rng = np.random.default_rng(2)
    base = rng.integers(1, 6, 300)
    frame = pd.DataFrame({f"p{i}": np.clip(base + rng.integers(-1, 2, 300), 1, 5) for i in range(1, 5)})
    result = likert_summary(frame, ["p1", "p2", "p3", "p4"])
    assert result.levels == ["1", "2", "3", "4", "5"]
    assert all(sum(row) == 300 for row in result.counts)
    assert 0.7 < result.cronbach_alpha < 1
    reference = _alpha(frame)
    assert result.cronbach_alpha == pytest.approx(reference)
    assert all(0 <= t <= 100 for t in result.top2box)


def _alpha(frame):
    k = frame.shape[1]
    return k / (k - 1) * (1 - frame.var(ddof=1).sum() / frame.sum(axis=1).var(ddof=1))
