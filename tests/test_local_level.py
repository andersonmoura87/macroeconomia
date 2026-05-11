"""Nível local (statsmodels)."""

from __future__ import annotations

import numpy as np

from macroeconomia.models.time_series.local_level import fit_local_level


def test_fit_local_level_smoke() -> None:
    rng = np.random.default_rng(42)
    y = rng.standard_normal(80).cumsum()
    res, metrics = fit_local_level(y)
    assert "aic" in metrics and "bic" in metrics
    assert res.nobs == 80
    assert np.isfinite(res.llf)
