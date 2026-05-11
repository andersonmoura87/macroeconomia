"""Auditoria de painel pós-join."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from macroeconomia.processing.macro_panel import audit_bivariate_macro_panel


def test_rejects_duplicate_dates() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 1), date(2020, 1, 1)],
            "ipca_index": [1.0, 2.0],
            "selic_pct": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="duplicada"):
        audit_bivariate_macro_panel(df)


def test_rejects_nulls() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 1), date(2020, 2, 1)],
            "ipca_index": pl.Series([1.0, None], dtype=pl.Float64),
            "selic_pct": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="nulo"):
        audit_bivariate_macro_panel(df)


def test_rejects_empty() -> None:
    df = pl.DataFrame(
        schema={
            "date": pl.Date,
            "ipca_index": pl.Float64,
            "selic_pct": pl.Float64,
        }
    )
    with pytest.raises(ValueError, match="vazio"):
        audit_bivariate_macro_panel(df)
