"""Transformações de estacionariedade comuns em VAR macro."""

from __future__ import annotations

import polars as pl


def with_logdiff_ipca_and_diff_selic(df: pl.DataFrame) -> pl.DataFrame:
    """Cria ``d_ln_ipca`` (:math:`\\Delta\\ln IPCA`) e ``d_selic`` (:math:`\\Delta i`)."""
    return (
        df.sort("date")
        .with_columns(
            pl.col("ipca_index").log().diff().alias("d_ln_ipca"),
            pl.col("selic_pct").diff().alias("d_selic"),
        )
        .drop_nulls()
    )
