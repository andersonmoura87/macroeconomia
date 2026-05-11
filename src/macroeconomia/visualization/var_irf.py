"""Gráficos de função de impulso-resposta (IRF) a partir de tabelas longas."""

from __future__ import annotations

import plotly.express as px
import polars as pl
from plotly.graph_objects import Figure


def irf_long_frame_line_chart(df: pl.DataFrame, *, title: str) -> Figure:
    """Linhas de IRF com um painel por variável de resposta (``facet_row``)."""
    pdf = df.to_pandas()
    n_resp = int(pdf["response"].nunique())
    fig = px.line(
        pdf,
        x="step",
        y="value",
        color="shock",
        facet_row="response",
        title=title,
        height=max(420, 260 * n_resp),
    )
    fig.update_layout(template="plotly_white", margin=dict(t=60, l=40, r=40, b=40))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig
