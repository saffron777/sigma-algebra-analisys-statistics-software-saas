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
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "g": group,
            "h": rng.choice(["si", "no"], n),
            "b": binary,
            "count": counts,
            "t": pd.date_range("2000-01-01", periods=n, freq="MS"),
            "s": 50 + 0.15 * np.arange(n) + season + noise,
        }
    )
