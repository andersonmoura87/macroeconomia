"""ARIMA automático (pmdarima) — requer extra ``[arima-auto]``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pmdarima")

from macroeconomia.models.time_series.arima_pmdarima import fit_arima_auto_pmdarima


def test_fit_arima_auto_pmdarima_smoke() -> None:
    rng = np.random.default_rng(7)
    y = rng.standard_normal(100).cumsum()
    model, metrics = fit_arima_auto_pmdarima(
        y,
        seasonal=False,
        max_p=2,
        max_q=2,
        max_d=1,
    )
    assert isinstance(metrics["order"], tuple)
    assert len(metrics["order"]) == 3
    assert "aic" in metrics
    assert hasattr(model, "predict")
