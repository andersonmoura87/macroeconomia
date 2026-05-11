"""Alertas simples de deriva de dados no painel bivariado (faixas históricas)."""

from __future__ import annotations

import polars as pl
from loguru import logger

# Mínimo de observações para estimar percentis com alguma estabilidade.
_MIN_HISTORY = 24


def macro_panel_drift_warnings(df: pl.DataFrame) -> list[str]:
    """Compara a **última** linha à faixa histórica (P0.5–P99.5) excluindo-a.

    Não bloqueia o ETL: retorna mensagens para ``logger.warning`` ou UI.

    Args:
        df: Painel validado com ``date``, ``ipca_index``, ``selic_pct``.

    Returns:
        Lista de mensagens acionáveis (vazia se sem alertas).
    """
    if df.height <= _MIN_HISTORY:
        return []
    cols = ("ipca_index", "selic_pct")
    for c in cols:
        if c not in df.columns:
            return []

    hist = df.slice(0, df.height - 1)
    last = df.tail(1)
    warnings: list[str] = []
    for col in cols:
        lo = float(hist.select(pl.col(col).quantile(0.005, interpolation="nearest")).item())
        hi = float(hist.select(pl.col(col).quantile(0.995, interpolation="nearest")).item())
        v_raw = last.get_column(col).item()
        try:
            v = float(v_raw)
        except (TypeError, ValueError):  # pragma: no cover
            continue
        if v < lo or v > hi:
            warnings.append(
                f"Drift dados: último `{col}` ({v:g}) fora da banda histórica "
                f"P0.5–P99.5 observada até a penúltima data ([{lo:g}, {hi:g}])."
            )
    return warnings


def log_macro_panel_drift_alerts(df: pl.DataFrame) -> None:
    """Registra alertas de deriva sem expor valores sensíveis além das séries públicas."""

    for msg in macro_panel_drift_warnings(df):
        logger.warning("{} [observabilidade_dados]", msg)
