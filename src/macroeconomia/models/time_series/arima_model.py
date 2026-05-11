"""ARIMA univariado com busca em grade pequena por critério de informação."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from statsmodels.tsa.arima.model import ARIMA


def fit_arima_grid(
    y: NDArray[Any] | list[float],
    *,
    max_p: int = 3,
    max_q: int = 3,
    d: int = 0,
    trend: str = "n",
) -> tuple[Any, dict[str, float]]:
    """Ajusta ARIMA(p,d,q) variando ``p`` e ``q`` em grade e escolhe menor AIC.

    Args:
        y: Série univariada (nível ou já estacionarizada).
        max_p: AR máximo testado.
        max_q: MA máximo testado.
        d: Ordem de integração fixa ``d``.
        trend: Tendência do ARIMA (``\"n\"`` sem intercepto; ``\"c\"`` com constante).

    Returns:
        Tupla ``(resultado_statsmodels, métricas)`` com chaves ``aic``, ``bic``, ``hqic``.

    Raises:
        ValueError: Nenhuma combinação convergiu na grade.
    """
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size < max(max_p, max_q, d) + 5:
        raise ValueError("Série curta demais para a grade ARIMA especificada.")

    best_aic = float("inf")
    best_res: Any | None = None
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(arr, order=(p, d, q), trend=trend)
                res = model.fit(method_kwargs={"warn_convergence": False})
                aic = float(res.aic)
                if aic < best_aic:
                    best_aic = aic
                    best_res = res
            except Exception:
                continue

    if best_res is None:
        raise ValueError("Nenhuma especificação ARIMA convergiu na grade informada.")

    metrics = {
        "aic": float(best_res.aic),
        "bic": float(best_res.bic),
        "hqic": float(best_res.hqic),
    }
    return cast(Any, best_res), metrics
