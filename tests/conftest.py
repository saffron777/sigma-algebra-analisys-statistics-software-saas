"""Shared fixtures: deterministic synthetic data used across test modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sales_frame() -> pd.DataFrame:
    """200 rows: sales = 10 + 2*price_index + 3*ads - 1.5*competition + noise, plus factors."""
    rng = np.random.default_rng(42)
    n = 200
    price = rng.normal(50, 10, n)
    ads = rng.normal(20, 5, n)
    competition = rng.normal(10, 3, n)
    region = rng.choice(["Norte", "Sur", "Centro"], n)
    channel = rng.choice(["Online", "Tienda"], n)
    region_effect = pd.Series(region).map({"Norte": 0.0, "Sur": 4.0, "Centro": -3.0}).to_numpy()
    sales = 10 + 2 * price + 3 * ads - 1.5 * competition + region_effect + rng.normal(0, 5, n)
    return pd.DataFrame(
        {
            "ventas": sales,
            "precio": price,
            "publicidad": ads,
            "competencia": competition,
            "region": region,
            "canal": channel,
            "comprador": rng.choice([0, 1], n, p=[0.6, 0.4]),
        }
    )
