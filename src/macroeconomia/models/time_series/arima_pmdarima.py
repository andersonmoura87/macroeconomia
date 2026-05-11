"""ARIMA automático via ``pmdarima`` (extra ``[arima-auto]``)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def fit_arima_auto_pmdarima(
    y: NDArray[Any] | list[float],
    *,
    seasonal: bool = False,
    m: int = 1,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
) -> tuple[Any, dict[str, float | int | tuple[int, ...]]]:
    """Executa ``auto_arima`` (stepwise) e retorna o modelo encaixado + métricas.

    Args:
        y: Série univariada.
        seasonal: Se ``True``, habilita componente sazonal (usa ``m``).
        m: Período sazonal (ex.: 12 para mensal anual).
        max_p, max_q, max_d: Limites da busca automática.

    Returns:
        Tupla ``(modelo_pmdarima, métricas)`` com ``order`` e critérios de informação.

    Raises:
        RuntimeError: Pacote ``pmdarima`` não instalado.
    """
    try:
        from pmdarima import auto_arima
    except ImportError as exc:  # pragma: no cover - extra opcional
        raise RuntimeError(
            "Instale o extra `macroeconomia[arima-auto]` (pacote `pmdarima`) "
            "para seleção automática de ARIMA."
        ) from exc

    arr = np.asarray(y, dtype=float).ravel()
    model = auto_arima(
        arr,
        seasonal=seasonal,
        m=m if seasonal else 1,
        max_p=max_p,
        max_q=max_q,
        max_d=max_d,
        suppress_warnings=True,
        stepwise=True,
        error_action="ignore",
    )
    order = tuple(int(x) for x in model.order)
    metrics: dict[str, float | int | tuple[int, ...]] = {
        "order": order,
        "aic": float(model.aic()),
        "bic": float(model.bic()),
    }
    if seasonal and hasattr(model, "seasonal_order"):
        so = getattr(model, "seasonal_order", None)
        if so is not None:
            metrics["seasonal_order"] = tuple(int(x) for x in so)
    return cast(Any, model), metrics
