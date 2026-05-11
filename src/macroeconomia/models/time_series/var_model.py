"""Estimação de Vetores Autorregressivos (VAR) com statsmodels."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray
from statsmodels.tsa.api import VAR as VARModel
from statsmodels.tsa.vector_ar.var_model import VARResults

from macroeconomia.config import get_settings


def _polars_to_pandas(endog: pl.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Converte colunas numéricas de Polars para ``pandas.DataFrame``."""
    pdf = endog.select(["date", *value_cols]).to_pandas()
    pdf = pdf.set_index("date").sort_index()
    return pdf.astype(np.float64)


def polars_endog_matrix(panel: pl.DataFrame, value_cols: list[str]) -> NDArray[np.float64]:
    """Matriz ``(T, k)`` das variáveis endógenas (sem índice temporal explícito)."""
    return np.asarray(_polars_to_pandas(panel, value_cols).values, dtype=np.float64)


def var_aic_table(endog: pd.DataFrame, maxlags: int) -> pd.DataFrame:
    """Tabela de critérios de informação por defasagem máxima.

    Args:
        endog: Matriz ``T × k`` de variáveis endógenas (sem NaN).
        maxlags: Defasagem máxima testada (>=1).

    Returns:
        DataFrame com colunas ``lag``, ``aic``, ``bic``, ``hqic``.
    """
    if maxlags < 1:
        raise ValueError("maxlags deve ser >= 1.")
    rows: list[dict[str, float | int]] = []
    for lag in range(1, maxlags + 1):
        model = VARModel(endog)
        res = model.fit(lag, trend="n")
        rows.append(
            {
                "lag": lag,
                "aic": float(res.aic),
                "bic": float(res.bic),
                "hqic": float(res.hqic),
            }
        )
    return pd.DataFrame(rows)


def fit_var(
    panel: pl.DataFrame,
    value_cols: list[str],
    *,
    maxlags: int = 8,
    ic: str | None = "aic",
) -> VARResults:
    """Estima um VAR multivariado em ``panel`` (colunas ``value_cols``).

    Usa ``trend="n"`` (sem constante explícita no VAR), adequado a séries já
    centradas / estacionarizadas (ex.: diferenças e variações log).

    Args:
        panel: DataFrame com coluna ``date`` e séries numéricas.
        value_cols: Nomes das colunas endógenas (ex.: ``["ipca_index","selic_pct"]``).
        maxlags: Defasagem máxima considerada na seleção por critério.
        ic: Critério ``"aic"``, ``"bic"``, ``"hqic"`` ou ``None`` para usar ``maxlags``.

    Returns:
        Resultado ``statsmodels`` (:class:`VARResults`).

    Raises:
        ValueError: ``ic`` inválido ou dados insuficientes.
    """
    settings = get_settings()
    np.random.seed(settings.seed)
    pdf = _polars_to_pandas(panel, value_cols)
    if pdf.shape[0] <= maxlags + 5:
        raise ValueError("Amostra curta demais para o VAR especificado.")

    model = VARModel(pdf)
    if ic is None:
        return cast(VARResults, model.fit(maxlags=maxlags, trend="n"))

    ic_norm = ic.lower()
    if ic_norm not in {"aic", "bic", "hqic"}:
        raise ValueError("ic deve ser 'aic', 'bic', 'hqic' ou None.")
    return cast(VARResults, model.fit(maxlags=maxlags, ic=ic_norm, trend="n"))
