"""Diagnósticos VAR (IRF, FEVD, Granger, estabilidade)."""

from __future__ import annotations

import polars as pl

from macroeconomia.models.time_series.var_diagnostics import (
    fevd_summary_text,
    granger_causality_markdown,
    irf_to_long_frame,
    var_is_stable,
    var_max_root_modulus,
)
from macroeconomia.models.time_series.var_model import fit_var
from macroeconomia.visualization.var_irf import irf_long_frame_line_chart


def test_irf_fevd_granger_on_synthetic(synthetic_stat_panel: pl.DataFrame) -> None:
    res = fit_var(synthetic_stat_panel, ["d_ln_ipca", "d_selic"], maxlags=4, ic="aic")
    mx = var_max_root_modulus(res)
    assert mx >= 0.0
    assert isinstance(var_is_stable(res), bool)

    long_df = irf_to_long_frame(res, periods=8, orthogonalized=True)
    assert long_df.height > 0
    fig = irf_long_frame_line_chart(long_df, title="t")
    assert fig.layout.title is not None

    md = granger_causality_markdown(res)
    assert "Granger" in md

    txt = fevd_summary_text(res, periods=6)
    assert len(txt) > 10
