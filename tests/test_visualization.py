"""Testes de exportação de figuras."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
from plotly.graph_objects import Figure

from macroeconomia.visualization.exporters import export_figure, macro_line_chart


def test_macro_line_chart_builds(tmp_path: Path) -> None:
    dr = pd.date_range("2020-01-01", periods=5, freq="MS")
    df = pl.DataFrame(
        {
            "date": [ts.date() for ts in dr],
            "a": [1.0, 2.0, 3.0, 2.5, 2.0],
            "b": [0.1, 0.2, 0.15, 0.18, 0.2],
        }
    )
    fig = macro_line_chart(df, y_columns=["a", "b"], title="t")
    assert isinstance(fig, Figure)
    stem = tmp_path / "fig"
    export_figure(fig, stem)
    assert stem.with_suffix(".html").exists()


def test_export_figure_png_warning_branch(tmp_path: Path) -> None:
    dr = pd.date_range("2020-01-01", periods=3, freq="MS")
    df = pl.DataFrame({"date": [ts.date() for ts in dr], "a": [1.0, 2.0, 3.0]})
    fig = macro_line_chart(df, y_columns=["a"], title="t")
    stem = tmp_path / "fig2"
    with patch.object(fig, "write_image", side_effect=RuntimeError("no kaleido")):
        export_figure(fig, stem)
    assert stem.with_suffix(".html").exists()
