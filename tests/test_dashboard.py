"""Testes do dashboard — ``data_loader`` e ``charts`` (sem Runtime Streamlit)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from dashboard.charts import macro_line_chart_figure
from dashboard.data_loader import (
    PanelResolutionError,
    PanelValidationError,
    load_manifest_for_parent,
    parquet_last_modified,
    resolve_panel_parquet_path,
    validate_panel_contract,
)

from macroeconomia.processing.macro_panel import MacroBivariateSchema


def _write_valid_dashboard_parquet(dest: Path) -> None:
    """Parquet válido segundo o mesmo contrato Pandera do painel."""

    dates = pl.date_range(date(2020, 1, 1), date(2024, 1, 1), interval="1mo", eager=True)
    n = len(dates)
    df = pl.DataFrame(
        {
            "date": dates,
            "ipca_index": (100 + np.linspace(0, 6, num=n)),
            "selic_pct": np.clip(0.7 + np.sin(np.linspace(0, 3, num=n)) * 0.2, 0.2, 2.5),
        }
    )
    validated = MacroBivariateSchema.validate(df)
    dest.parent.mkdir(parents=True, exist_ok=True)
    validated.write_parquet(dest)


def test_resolve_demo_forced(tmp_path: Path) -> None:
    demo_file = tmp_path / "demo" / "macro_ipca_selic.parquet"
    _write_valid_dashboard_parquet(demo_file)
    proc = tmp_path / "processed"
    proc.mkdir()
    settings = SimpleNamespace(
        demo_mode=True,
        demo_parquet_relative=demo_file,
        data_processed_dir=proc,
        dashboard_parquet_name="macro_ipca_selic.parquet",
    )
    meta = resolve_panel_parquet_path(settings)
    assert meta.source == "demo_forced"
    assert meta.parquet_path.resolve() == demo_file.resolve()


def test_resolve_fallback_to_demo_when_processed_missing(tmp_path: Path) -> None:
    demo_file = tmp_path / "demo" / "macro_ipca_selic.parquet"
    _write_valid_dashboard_parquet(demo_file)
    proc = tmp_path / "processed"
    proc.mkdir()
    settings = SimpleNamespace(
        demo_mode=False,
        demo_parquet_relative=demo_file,
        data_processed_dir=proc,
        dashboard_parquet_name="macro_ipca_selic.parquet",
    )
    meta = resolve_panel_parquet_path(settings)
    assert meta.source == "demo_fallback"


def test_resolve_missing_raises(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    dummy_demo = tmp_path / "nah.parquet"
    settings = SimpleNamespace(
        demo_mode=False,
        demo_parquet_relative=dummy_demo,
        data_processed_dir=proc,
        dashboard_parquet_name="macro_ipca_selic.parquet",
    )
    with pytest.raises(PanelResolutionError):
        resolve_panel_parquet_path(settings)


def test_validate_contract_rejects_extra_column(tmp_path: Path) -> None:
    demo_file = tmp_path / "bad.parquet"
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 1)],
            "ipca_index": [100.0],
            "selic_pct": [1.0],
            "junk": ["x"],
        }
    )
    demo_file.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(demo_file)
    raw = pl.read_parquet(demo_file)
    with pytest.raises(PanelValidationError):
        validate_panel_contract(raw)


def test_line_chart_returns_figure(tmp_path: Path) -> None:
    demo_file = tmp_path / "demo.parquet"
    _write_valid_dashboard_parquet(demo_file)
    df = pl.read_parquet(demo_file)
    fig = macro_line_chart_figure(df, "ipca_index")
    assert fig.layout.title.text


def test_parquet_mtime_iso(tmp_path: Path) -> None:
    demo_file = tmp_path / "d.parquet"
    _write_valid_dashboard_parquet(demo_file)
    iso = parquet_last_modified(demo_file)
    assert iso and "T" in iso


def test_manifest_optional(tmp_path: Path) -> None:
    assert load_manifest_for_parent(tmp_path) is None
