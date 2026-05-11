"""Testes de VAR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from macroeconomia.models.time_series.var_model import fit_var, var_aic_table


def test_var_aic_table_shapes() -> None:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2000-01-01", periods=80, freq="ME")
    df = pd.DataFrame(rng.normal(size=(80, 2)), index=idx, columns=["a", "b"])
    tab = var_aic_table(df, maxlags=4)
    assert tab.shape[0] == 4
    assert {"lag", "aic", "bic", "hqic"} <= set(tab.columns)


def test_fit_var_on_synthetic(synthetic_stat_panel: pl.DataFrame) -> None:
    res = fit_var(synthetic_stat_panel, ["d_ln_ipca", "d_selic"], maxlags=4, ic="aic")
    assert res.k_ar >= 1


def test_var_aic_table_invalid_maxlags() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(20, 2)), columns=["a", "b"])
    with pytest.raises(ValueError):
        var_aic_table(df, maxlags=0)


def test_fit_var_ic_none(synthetic_stat_panel: pl.DataFrame) -> None:
    res = fit_var(synthetic_stat_panel, ["d_ln_ipca", "d_selic"], maxlags=3, ic=None)
    assert res.k_ar == 3


def test_fit_var_bad_ic(synthetic_stat_panel: pl.DataFrame) -> None:
    with pytest.raises(ValueError):
        fit_var(synthetic_stat_panel, ["d_ln_ipca", "d_selic"], maxlags=3, ic="xyz")


def test_fit_var_short_sample_raises(synthetic_stat_panel: pl.DataFrame) -> None:
    small = synthetic_stat_panel.head(10)
    with pytest.raises(ValueError):
        fit_var(small, ["d_ln_ipca", "d_selic"], maxlags=8, ic="aic")
