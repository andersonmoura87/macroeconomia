"""GARCH(1,1) opcional via pacote ``arch`` (extra ``[volatility]``)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def fit_garch_11(
    returns: NDArray[Any] | list[float],
    *,
    rescale: bool = True,
) -> tuple[Any, dict[str, float]]:
    """Estima GARCH(1,1) com média zero nas inovações.

    Args:
        returns: Retornos ou inovações (ex.: diferenças log).
        rescale: Repassa ``rescale`` ao ``arch`` (recomendado para estabilidade numérica).

    Returns:
        ``(resultado_arch, dict)`` com AIC/BIC/loglik quando disponíveis.

    Raises:
        RuntimeError: Pacote ``arch`` não instalado.
    """
    try:
        from arch import arch_model
    except ImportError as exc:  # pragma: no cover - extra opcional
        raise RuntimeError(
            "Instale o extra `macroeconomia[volatility]` (pacote `arch`) para GARCH."
        ) from exc

    y = np.asarray(returns, dtype=float).ravel()
    model = arch_model(y, mean="Zero", vol="Garch", p=1, q=1, rescale=rescale)
    res = model.fit(disp="off")
    out: dict[str, float] = {}
    for key in ("aic", "bic", "loglikelihood"):
        val = getattr(res, key, None)
        if val is not None:
            out[key] = float(val)
    return cast(Any, res), out
