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
