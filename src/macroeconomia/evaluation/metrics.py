"""Métricas de erro de previsão (nível e escala-invariante)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def mae(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Erro absoluto médio."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: npt.NDArray[np.floating[Any]], y_pred: npt.NDArray[np.floating[Any]]) -> float:
    """Raiz do erro quadrático médio."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(
    y_true: npt.NDArray[np.floating[Any]],
    y_pred: npt.NDArray[np.floating[Any]],
    *,
    eps: float = 1e-9,
) -> float:
    """SMAPE (simétrico, limitado a [0, 1] na implementação comum)."""
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))
