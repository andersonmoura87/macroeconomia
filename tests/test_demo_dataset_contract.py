"""Contrato Pandera do parquet demo versionado ("clone-and-run")."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandera")
import polars as pl


def test_demo_parquet_if_present(repo_root: Path) -> None:
    from macroeconomia.processing.macro_panel import MacroBivariateSchema

    p = repo_root / "data" / "demo" / "macro_ipca_selic.parquet"
    if not p.is_file():
        pytest.skip("Parquet demo nao encontrado — rode scripts/gen_demo_panel.py")
    df = pl.read_parquet(p)
    MacroBivariateSchema.validate(df)
