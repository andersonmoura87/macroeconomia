"""Teste de integração do pipeline VAR com pontos sintéticos (sem rede)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroeconomia.config import get_settings
from macroeconomia.ingestion.bcb_sgs import BcbSgsPoint
from macroeconomia.logging_config import configure_logging


def _monthly_points(n: int) -> tuple[list[BcbSgsPoint], list[BcbSgsPoint]]:
    rng = np.random.default_rng(7)
    dr = pd.date_range("2000-01-01", periods=n, freq="MS")
    ipca: list[BcbSgsPoint] = []
    selic: list[BcbSgsPoint] = []
    for i, ts in enumerate(dr):
        d = ts.date()
        ipca.append(BcbSgsPoint(d, 100.0 + 0.05 * i + 0.02 * (i % 4) + float(rng.normal(0, 0.05))))
        selic.append(BcbSgsPoint(d, 0.45 + 0.01 * (i % 5) + float(rng.normal(0, 0.02))))
    return ipca, selic


def test_var_pipeline_end_to_end(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "raw"
    proc = tmp_path / "proc"
    ext = tmp_path / "ext"
    art = tmp_path / "art"
    ml = tmp_path / "mlruns"
    monkeypatch.setenv("MACRO_DATA_RAW_DIR", str(raw))
    monkeypatch.setenv("MACRO_DATA_PROCESSED_DIR", str(proc))
    monkeypatch.setenv("MACRO_DATA_EXTERNAL_DIR", str(ext))
    monkeypatch.setenv("MACRO_ARTIFACTS_DIR", str(art))
    monkeypatch.setenv("MACRO_MLFLOW_TRACKING_URI", ml.as_uri())
    get_settings.cache_clear()

    configure_logging(level="ERROR")
    ipca, selic = _monthly_points(220)

    from macroeconomia.examples.var_inflation_pipeline import run_var_inflation_pipeline

    summary = run_var_inflation_pipeline(ipca_points=ipca, selic_points=selic)

    assert summary.exists()
    assert (proc / "macro_ipca_selic.parquet").exists()
    assert (art / "VAR_IPCA_SELIC_SUMMARY.md").exists()
    assert (art / "VAR_FEVD.txt").exists()
    text = summary.read_text(encoding="utf-8")
    assert "VAR IPCA" in text
    assert "Out-of-sample" in text

    get_settings.cache_clear()
