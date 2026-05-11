"""Gráficos macro reproduzíveis e exportação de artefatos."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from loguru import logger


def macro_line_chart(
    panel: pl.DataFrame,
    *,
    y_columns: Sequence[str],
    title: str,
) -> go.Figure:
    """Séries temporais multi-eixo (normalização z-score apenas visual)."""
    pdf = panel.to_pandas()
    fig = go.Figure()
    for col in y_columns:
        fig.add_trace(
            go.Scatter(
                x=pdf["date"],
                y=pdf[col],
                mode="lines",
                name=col,
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
    )
    fig.update_xaxes(title_text="Data")
    return fig


def export_figure(fig: go.Figure, stem: Path, *, width: int = 1100, height: int = 600) -> None:
    """Exporta HTML interativo e PNG estático ao lado de ``stem``."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    html_path = stem.with_suffix(".html")
    png_path = stem.with_suffix(".png")
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        fig.write_image(str(png_path), width=width, height=height)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Export PNG indisponível (instale kaleido ou veja logs): {}", exc)
    logger.info("Figura HTML exportada em {}.", html_path)
