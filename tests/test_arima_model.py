"""ARIMA univariado."""

from __future__ import annotations

import math

import numpy as np
import pytest

from macroeconomia.models.time_series.arima_model import fit_arima_grid


def test_fit_arima_grid_selects_model() -> None:
    rng = np.random.default_rng(9)
    y = np.zeros(200, dtype=float)
    for t in range(1, y.size):
        y[t] = 0.45 * y[t - 1] + float(rng.normal(scale=0.5))
    res, metrics = fit_arima_grid(y, max_p=2, max_q=2, d=0, trend="n")
    assert "aic" in metrics
    assert math.isfinite(metrics["aic"])
    assert res is not None


def test_fit_arima_grid_too_short() -> None:
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        fit_arima_grid(y, max_p=3, max_q=3, d=0)


def test_volatility_package_exports() -> None:
    import macroeconomia.models.volatility as vol

    assert callable(vol.fit_garch_11)
