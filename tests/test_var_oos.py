"""Out-of-sample temporal para VAR."""

from __future__ import annotations

import math

import polars as pl

from macroeconomia.models.time_series.var_oos import (
    temporal_train_test_split,
    var_multistep_oos_metrics,
)


def test_temporal_split_shapes(synthetic_stat_panel: pl.DataFrame) -> None:
    train, test = temporal_train_test_split(synthetic_stat_panel, train_ratio=0.85)
    assert train.height + test.height == synthetic_stat_panel.height
    assert test.height >= 1


def test_oos_metrics_finite(synthetic_stat_panel: pl.DataFrame) -> None:
    train, test = temporal_train_test_split(synthetic_stat_panel, train_ratio=0.85)
    metrics = var_multistep_oos_metrics(
        train,
        test,
        ["d_ln_ipca", "d_selic"],
        maxlags=4,
        ic="aic",
    )
    assert "oos_rmse_mean" in metrics
    assert math.isfinite(metrics["oos_rmse_mean"])
    assert metrics["oos_rmse_mean"] >= 0.0
