"""Testes de painel e transformações."""

from __future__ import annotations

from macroeconomia.processing.macro_panel import (
    build_macro_bivariate_panel,
    validate_macro_bivariate,
    write_parquet_partition,
)
from macroeconomia.processing.transforms import with_logdiff_ipca_and_diff_selic


def test_build_and_validate(sample_ipca_selic_points: tuple) -> None:
    ipca, selic = sample_ipca_selic_points
    raw = build_macro_bivariate_panel(ipca, selic)
    validated = validate_macro_bivariate(raw)
    assert validated.height == 40
    assert set(validated.columns) == {"date", "ipca_index", "selic_pct"}


def test_write_parquet_partition(tmp_path, sample_ipca_selic_points: tuple) -> None:
    ipca, selic = sample_ipca_selic_points
    raw = validate_macro_bivariate(build_macro_bivariate_panel(ipca, selic))
    out = tmp_path / "p.parquet"
    write_parquet_partition(raw, out)
    assert out.exists()


def test_stationary_transform(sample_ipca_selic_points: tuple) -> None:
    ipca, selic = sample_ipca_selic_points
    raw = validate_macro_bivariate(build_macro_bivariate_panel(ipca, selic))
    st = with_logdiff_ipca_and_diff_selic(raw)
    assert st.height == raw.height - 1
    assert "d_ln_ipca" in st.columns
