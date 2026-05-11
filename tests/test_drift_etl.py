"""Alertas simples de deriva no painel bivariado."""

from __future__ import annotations

from datetime import date

import polars as pl

from macroeconomia.processing.drift_alerts import macro_panel_drift_warnings


def test_drift_empty_for_short_series() -> None:
    dates = pl.date_range(date(2024, 1, 1), date(2024, 6, 1), interval="1mo", eager=True)
    df = pl.DataFrame(
        {
            "date": dates,
            "ipca_index": [100.0 + i * 0.1 for i in range(len(dates))],
            "selic_pct": [1.0] * len(dates),
        }
    )
    assert macro_panel_drift_warnings(df) == []


def test_drift_warns_on_extreme_last() -> None:
    n_hist = 50
    dates = pl.date_range(date(2020, 1, 1), date(2024, 3, 1), interval="1mo", eager=True)[
        : n_hist + 1
    ]
    assert len(dates) == n_hist + 1
    ipca = [100 + i * 0.05 for i in range(n_hist)] + [50000.0]
    selic = [0.8 + (i % 3) * 0.05 for i in range(n_hist)] + [120.0]
    df = pl.DataFrame({"date": dates, "ipca_index": ipca, "selic_pct": selic})
    warns = macro_panel_drift_warnings(df)
    assert any("ipca_index" in w for w in warns)
    assert any("selic_pct" in w for w in warns)
