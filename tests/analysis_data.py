"""The synthetic dataset every catalogue analysis is exercised on (its ``example`` names these columns)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def standard_frame(n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(10, 2, n)
    x2 = rng.normal(5, 1.5, n)
    x3 = rng.uniform(0, 10, n)
    group = rng.choice(["A", "B", "C"], n)
    shift = pd.Series(group).map({"A": 0.0, "B": 2.0, "C": -1.5}).to_numpy()
    y = 3 + 1.5 * x1 - 2 * x2 + shift + rng.normal(0, 2, n)
    probability = 1 / (1 + np.exp(-(0.4 * (x1 - 10) - 0.6 * (x2 - 5))))
    binary = (rng.uniform(size=n) < probability).astype(int)
    counts = rng.poisson(np.exp(0.2 + 0.08 * x1 - 0.05 * x2))
    # a monthly series with trend, yearly seasonality and AR(1) noise
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + rng.normal(0, 1)
    season = 3 * np.sin(2 * np.pi * np.arange(n) / 12)
    h = rng.choice(["si", "no"], n)
    # drawn last so the values above (and every earlier test) do not change
    with_gaps = x2.copy()
    with_gaps[rng.choice(n, n // 10, replace=False)] = np.nan
    with_outliers = x1.copy()
    with_outliers[rng.choice(n, 4, replace=False)] = [30.0, 35.0, -12.0, 40.0]
    # three raters scoring the same subjects 1-5, and three binary judgements
    latent = 0.4 * (x1 - 10) + rng.normal(0, 0.5, n)
    ratings = {f"r{i}": np.clip(np.round(3 + latent + rng.normal(0, 0.7, n)), 1, 5).astype(int) for i in (1, 2, 3)}
    judgements = {f"c{i}": (latent + rng.normal(0, 0.8, n) > 0).astype(int) for i in (1, 2, 3)}
    # twelve clusters, each shifting ``yc`` (a copy of y) by its own random amount
    cluster = rng.integers(0, 12, n)
    shifts = rng.normal(0, 2.5, 12)
    # four classrooms (1-4) inside each group g: a factor nested in another
    classroom = rng.integers(1, 5, n)
    classroom_effect = rng.normal(0, 1.5, (3, 4))
    group_code = pd.Series(group).map({"A": 0, "B": 1, "C": 2}).to_numpy()
    # a panel (n // 12 entities observed 12 times), a GARCH(1,1) return series and a two-regime series
    entity = np.arange(n) // 12
    entity_effect = rng.normal(0, 2.0, entity.max() + 1)[entity]
    returns, variance = np.zeros(n), np.full(n, 1.0)
    for i in range(1, n):
        variance[i] = 0.1 + 0.15 * returns[i - 1] ** 2 + 0.75 * variance[i - 1]
        returns[i] = np.sqrt(variance[i]) * rng.standard_t(8) * np.sqrt(6 / 8)
    regime = np.zeros(n, dtype=int)
    for i in range(1, n):
        stay = 0.95
        regime[i] = regime[i - 1] if rng.uniform() < stay else 1 - regime[i - 1]
    switching = np.where(regime == 0, 1.0, 5.0) + rng.normal(0, np.where(regime == 0, 0.7, 1.4))
    # a model's predictions, drawn last: ``g`` with 20 % of the labels scrambled, a score for ``b``, a guess of ``y``
    scrambled = rng.uniform(size=n) < 0.2
    predicted_group = np.where(scrambled, rng.choice(["A", "B", "C"], n), group)
    score_b = np.clip(probability + rng.normal(0, 0.15, n), 0.01, 0.99)
    predicted_y = y + rng.normal(0, 1.5, n) + 0.3
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "g": group,
            "h": h,
            "b": binary,
            "count": counts,
            "t": pd.date_range("2000-01-01", periods=n, freq="MS"),
            "s": 50 + 0.15 * np.arange(n) + season + noise,
            "m": with_gaps,  # x2 with ~10 % missing values
            "o": with_outliers,  # x1 with four extreme values
            **ratings,  # r1..r3: 1-5 scores of three raters
            **judgements,  # c1..c3: 0/1 judgements
            "cl": [f"c{c + 1:02d}" for c in cluster],  # 12 clusters
            "yc": y + shifts[cluster],  # y with a random cluster effect
            "aula": [str(c) for c in classroom],  # classrooms 1-4, repeated inside every g
            "ya": y + classroom_effect[group_code, classroom - 1],  # y with a classroom-within-group effect
            "ent": [f"E{e + 1:02d}" for e in entity],  # panel entities
            "per": np.arange(n) % 12,  # panel periods
            "yp": 1.5 * x1 + entity_effect + rng.normal(0, 1.0, n),  # panel response with an entity effect
            "ret": returns,  # GARCH(1,1) returns
            "rg": switching,  # a series with two regimes
            "ps": predicted_group,  # predicted group (≈ 87 % right)
            "pb": score_b,  # predicted probability that b = 1
            "py": predicted_y,  # predicted y (error sd 1.5, bias 0.3)
        }
    )


def lookup_frame() -> pd.DataFrame:
    """A reference table to merge with: it has ``y`` too (the suffix) and a group ``D`` the main data lacks."""
    return pd.DataFrame({"g": ["A", "B", "D"], "peso": [1.5, 2.5, 9.0], "y": [0.0, 0.0, 0.0]})


def more_rows() -> pd.DataFrame:
    return standard_frame(30, seed=3)


# what an ``example`` writes as "@name" where a call needs another dataset
OTHER_DATASETS = {"@lookup": lookup_frame, "@more": more_rows}


def other_frames(spec, params: dict | None) -> dict[str, pd.DataFrame]:
    from sigma.domain.analysis.spec import dataset_params

    given = params or {}
    return {n: OTHER_DATASETS[given[n]]() for n in dataset_params(spec) if given.get(n) in OTHER_DATASETS}


def run(
    module: str,
    analysis_id: str,
    frame: pd.DataFrame | None = None,
    roles: dict | None = None,
    params: dict | None = None,
):
    """Validate the inputs like the use case does and compute the analysis (the output, not a report).

    A "dataset" parameter written as ``"@lookup"`` / ``"@more"`` stands for those test frames.
    """
    from sigma.domain.analysis import registry
    from sigma.domain.analysis.spec import prepare_inputs

    spec = registry.get(module, analysis_id)
    data = frame if frame is not None else standard_frame()
    others = other_frames(spec, params)
    clean_roles, clean_params = prepare_inputs(spec, data, roles or {}, params or {}, others)
    return spec.compute(data, clean_roles, {**clean_params, **others})


def offered(output, key: str = "resultado") -> pd.DataFrame:
    return output.datasets[key].frame
