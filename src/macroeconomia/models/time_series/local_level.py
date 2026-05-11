"""Modelo de espaço de estados: nível local (``UnobservedComponents``)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from statsmodels.tsa.statespace.structural import UnobservedComponents


def fit_local_level(
    y: NDArray[Any] | list[float],
    *,
    irregular: bool = True,
    stochastic_level: bool = True,
) -> tuple[Any, dict[str, float]]:
    """Estima componente de nível local + irregularidade gaussiana.

    Args:
        y: Série observada (nível ou log-nível).
        irregular: Inclui componente irregular (ruído de medição).
        stochastic_level: Nível local estocástico (random walk suavizado).

    Returns:
        ``(resultado_statsmodels, métricas)`` com ``aic``, ``bic``, ``hqic``.
    """
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size < 10:
        raise ValueError("Série curta demais para o modelo de nível local.")

    mod = UnobservedComponents(
        arr,
        level=True,
        stochastic_level=stochastic_level,
        irregular=irregular,
    )
    res = mod.fit(disp=False)
    metrics = {
        "aic": float(res.aic),
        "bic": float(res.bic),
        "hqic": float(res.hqic),
    }
    return cast(Any, res), metrics
