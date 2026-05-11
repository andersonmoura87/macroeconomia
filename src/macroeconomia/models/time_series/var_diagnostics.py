"""Diagnósticos econométricos para resultados VAR (statsmodels)."""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np
import polars as pl
from statsmodels.tsa.vector_ar.var_model import VARResults


def var_max_root_modulus(res: VARResults) -> float:
    """Maior módulo entre as raízes do polinômio AR adjunto (companion matrix)."""
    roots = np.asarray(res.roots, dtype=np.complex128)
    return float(np.max(np.abs(roots)))


def var_is_stable(res: VARResults, *, rtol: float = 1e-9) -> bool:
    """Processo VAR estacionário se todos os módulos das raízes são ``< 1`` (Lütkepohl)."""
    return bool(np.all(np.abs(res.roots) < 1.0 + rtol))


def granger_causality_markdown(res: VARResults, *, maxlag: int | None = None) -> str:
    """Resumo em Markdown dos testes de causalidade de Granger (no VAR).

    Args:
        res: Resultado VAR estimado.
        maxlag: Ignorado (mantido por compatibilidade); o teste usa a ordem estimada.
    """
    _ = maxlag
    lines: list[str] = ["## Causalidade de Granger (SSR)", ""]
    names = list(res.names)
    for caused in names:
        causing = [n for n in names if n != caused]
        causation: Any = res.test_causality(caused, causing)
        pval = float(causation.pvalue)
        lines.append(f"- **{caused}** causado(a) por `{causing}`: p-valor = {pval:.4g}")
    return "\n".join(lines)


def irf_to_long_frame(
    res: VARResults,
    *,
    periods: int = 12,
    orthogonalized: bool = True,
) -> pl.DataFrame:
    """Converte IRF (ortogonalizada ou não) em tabela longa para Plotly.

    Args:
        res: Resultado VAR estimado.
        periods: Horizonte ``H`` da IRF (passos ``0..H``).
        orthogonalized: Se ``True``, usa choque ortogonalizado (ordem Cholesky).
    """
    irf = res.irf(periods=periods)
    tensor = np.asarray(
        irf.orth_irfs if orthogonalized else irf.irfs,
        dtype=float,
    )
    names = list(res.names)
    rows: list[dict[str, float | int | str]] = []
    for step in range(tensor.shape[0]):
        for response_idx, response in enumerate(names):
            for shock_idx, shock in enumerate(names):
                rows.append(
                    {
                        "step": step,
                        "response": response,
                        "shock": shock,
                        "value": float(tensor[step, response_idx, shock_idx]),
                    }
                )
    return pl.DataFrame(rows)


def fevd_summary_text(res: VARResults, periods: int = 12) -> str:
    """Tabela textual da decomposição da variância da previsão (FEVD)."""
    fevd = res.fevd(periods)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fevd.summary()
    text = buf.getvalue().strip()
    if text:
        return text
    decomp = np.asarray(fevd.decomp, dtype=float)
    return f"FEVD (fallback numérico), shape={decomp.shape}"
