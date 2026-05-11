"""Avaliação out-of-sample simples para VAR (corte temporal contíguo)."""

from __future__ import annotations

import numpy as np
import polars as pl

from macroeconomia.evaluation.metrics import mae, rmse
from macroeconomia.models.time_series.var_model import fit_var, polars_endog_matrix


def temporal_train_test_split(
    panel: pl.DataFrame,
    train_ratio: float = 0.85,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Divide o painel ordenado por ``date`` em treino e teste contíguos.

    Args:
        panel: Painel com coluna ``date`` (sem buracos esperados no eixo após join).
        train_ratio: Fração inicial alocada ao treino (ex.: ``0.85``).

    Returns:
        Tupla ``(train, test)`` com ``test`` sendo o sufixo temporal.

    Raises:
        ValueError: ``train_ratio`` inválido ou amostra curta demais para gerar teste.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio deve estar no intervalo aberto (0, 1).")
    ordered = panel.sort("date")
    n = ordered.height
    split = int(n * train_ratio)
    if split < 1 or split >= n:
        raise ValueError(
            "O corte temporal produziu treino ou teste vazio. "
            f"Ajuste train_ratio ou aumente a amostra (n={n})."
        )
    train = ordered.head(split)
    test = ordered.tail(n - split)
    return train, test


def var_multistep_oos_metrics(
    train: pl.DataFrame,
    test: pl.DataFrame,
    value_cols: list[str],
    *,
    maxlags: int,
    ic: str | None = "aic",
) -> dict[str, float]:
    """Erro de previsão multi-passos fora da amostra (VAR estimado só no treino).

    Ajusta o VAR em ``train`` e compara a trajetória prevista ``h=len(test)`` passos
    com os valores observados em ``test`` (sem atualização recursiva do modelo).

    Args:
        train: Janela inicial (inclui ``date`` e ``value_cols``).
        test: Janela final contígua, mesmas colunas.
        value_cols: Endógenas do VAR.
        maxlags: Defasagem máxima na seleção / estimação (mesma convenção de :func:`fit_var`).
        ic: Critério de informação ou ``None``.

    Returns:
        Dicionário com ``oos_rmse_mean``, ``oos_mae_mean`` e métricas por equação.
    """
    steps = test.height
    if steps == 0:
        raise ValueError("Conjunto de teste vazio.")
    res = fit_var(train, value_cols, maxlags=maxlags, ic=ic)
    lag = int(res.k_ar)
    if lag < 1:
        raise ValueError("VAR com k_ar < 1; não é possível prever OOS.")
    train_y = polars_endog_matrix(train.sort("date"), value_cols)
    y0 = train_y[-lag:, :]
    fc = np.asarray(res.forecast(y0, steps=steps), dtype=float)
    actual = test.sort("date").select(value_cols).to_numpy().astype(float)
    if fc.shape != actual.shape:
        raise ValueError(
            "Formato inesperado na previsão OOS: "
            f"fc={fc.shape}, y_test={actual.shape}. Verifique colunas e lags."
        )
    flat_a = actual.ravel()
    flat_f = fc.ravel()
    out: dict[str, float] = {
        "oos_rmse_mean": float(np.sqrt(np.mean((flat_a - flat_f) ** 2))),
        "oos_mae_mean": float(np.mean(np.abs(flat_a - flat_f))),
    }
    for j, name in enumerate(value_cols):
        out[f"oos_rmse__{name}"] = float(rmse(actual[:, j], fc[:, j]))
        out[f"oos_mae__{name}"] = float(mae(actual[:, j], fc[:, j]))
    return out
