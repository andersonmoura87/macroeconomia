"""GARCH opcional (extra ``arch``)."""

from __future__ import annotations

import numpy as np
import pytest


def test_fit_garch_11_runs() -> None:
    pytest.importorskip("arch")
    from macroeconomia.models.volatility.garch_fit import fit_garch_11

    rng = np.random.default_rng(3)
    y = rng.normal(scale=0.01, size=500)
    res, metrics = fit_garch_11(y, rescale=True)
    assert res is not None
    assert isinstance(metrics, dict)
