"""Testes de métricas."""

from __future__ import annotations

import numpy as np
import pytest

from macroeconomia.evaluation.metrics import mae, rmse, smape


def test_metrics_basic() -> None:
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 1.9, 3.2])
    assert mae(y, p) == pytest.approx(np.mean(np.abs(y - p)))
    assert rmse(y, p) == pytest.approx(np.sqrt(np.mean((y - p) ** 2)))
    assert 0 <= smape(y, p) <= 1.5
