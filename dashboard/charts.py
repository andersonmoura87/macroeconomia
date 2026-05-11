"""Gráficos Plotly reutilizáveis (desacoplados do Streamlit)."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import polars as pl


def macro_line_chart_figure(df: pl.DataFrame, column: str, *, title: str | None = None) -> Any:
    """Série temporal de uma coluna contra ``date``.

    Args:
        df: Deve incluir ``date`` e ``column``.
        column: Métrica a plotar (ex.: ``ipca_index``).
        title: Título opcional.

    Returns:
        Figura Plotly (`go.Figure`).
    """

    cols = df.select(["date", column]).sort("date")
    pdf = cols.to_pandas().set_index("date")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pdf.index.astype(str),
            y=pdf[column],
            mode="lines",
            name=column,
        )
    )
    fig.update_layout(
        title=title or column,
        template="plotly_white",
        xaxis_title="data",
        yaxis_title=column,
        hovermode="x unified",
        height=440,
        margin=dict(l=40, r=24, t=56, b=48),
    )
    return fig
