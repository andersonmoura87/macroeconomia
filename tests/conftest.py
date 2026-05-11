"""Fixtures compartilhadas."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from macroeconomia.ingestion.bcb_sgs import BcbSgsPoint
from macroeconomia.processing.macro_panel import validate_macro_bivariate
from macroeconomia.processing.transforms import with_logdiff_ipca_and_diff_selic


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _scalar_date(cell: date | object) -> date:
    """Normaliza entrada de série temporal Polars."""

    return cell if isinstance(cell, date) else cell.date()  # type: ignore[union-attr]


@pytest.fixture(scope="session")
def sample_ipca_selic_points() -> tuple[list[BcbSgsPoint], list[BcbSgsPoint]]:
    """40 meses alinhados (inner join com 40 linhas)."""

    dates = pl.date_range(date(2020, 1, 1), date(2035, 1, 1), interval="1mo", eager=True)[:40]
    ipca: list[BcbSgsPoint] = []
    selic: list[BcbSgsPoint] = []
    for i, cell in enumerate(dates.to_list()):
        d = _scalar_date(cell)
        ipca.append(BcbSgsPoint(ref_date=d, value=float(100.0 + i * 0.06)))
        selic.append(BcbSgsPoint(ref_date=d, value=float(0.8 + (i % 7) * 0.015)))
    return ipca, selic


@pytest.fixture(scope="session")
def synthetic_stat_panel() -> pl.DataFrame:
    """~120 meses estáveis para VAR / OOS (independente do fixture SGS)."""

    rng = np.random.default_rng(5)
    n = 120
    dates = pl.date_range(date(2008, 1, 1), date(2040, 1, 1), interval="1mo", eager=True)[:n]
    ipca = 100 + np.linspace(0, 12, num=n) + np.cumsum(rng.normal(0, 0.05, size=n)) * 0.08
    sel = 0.55 + np.sin(np.linspace(0, 5, num=n)) * 0.2 + rng.normal(0, 0.025, size=n)
    sel = np.clip(sel, 0.12, 3.8)
    raw = pl.DataFrame(
        {
            "date": dates,
            "ipca_index": ipca.astype(np.float64),
            "selic_pct": sel.astype(np.float64),
        }
    )
    return with_logdiff_ipca_and_diff_selic(validate_macro_bivariate(raw))
