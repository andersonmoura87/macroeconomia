"""Smoke tests MLflow pyfunc (VAR / ARIMA) com tracking apenas em diretório local."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from macroeconomia.mlflow_lineage import numpy_series_sha256
from macroeconomia.models.time_series.arima_mlflow_pyfunc import log_arima_pyfunc_bundle
from macroeconomia.models.time_series.arima_model import fit_arima_grid
from macroeconomia.models.time_series.var_mlflow_pyfunc import log_var_pyfunc_bundle
from macroeconomia.models.time_series.var_model import fit_var


@pytest.fixture(autouse=True)
def _isolate_mlflow_experiments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracking limpo entre testes."""

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")


def test_numpy_series_sha256_stable() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    h1 = numpy_series_sha256(a)
    h2 = numpy_series_sha256(a)
    assert h1 == h2
    assert len(h1) == 64


def test_var_pyfunc_load_and_predict(synthetic_stat_panel: object) -> None:
    mlflow.set_experiment("smoke_var_pyfunc")
    with mlflow.start_run() as run:
        res = fit_var(
            synthetic_stat_panel,
            ["d_ln_ipca", "d_selic"],
            maxlags=4,
            ic="aic",
        )
        k_ar = int(res.k_ar)
        log_var_pyfunc_bundle(
            res,
            value_cols=["d_ln_ipca", "d_selic"],
            k_ar=k_ar,
            registered_model_name=None,
        )
        rid = run.info.run_id

    model = mlflow.pyfunc.load_model(f"runs:/{rid}/var_model_pyfunc")
    tail = synthetic_stat_panel.tail(k_ar)
    pdf = tail.select(["d_ln_ipca", "d_selic"]).to_pandas()
    out = model.predict(pdf)
    assert out.shape[0] == 1
    assert out.shape[1] == 2


def test_arima_pyfunc_load_and_predict() -> None:
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.standard_normal(120))
    res, _metrics = fit_arima_grid(y, max_p=2, max_q=2, d=0, trend="n")
    mlflow.set_experiment("smoke_arima_pyfunc")
    with mlflow.start_run() as run:
        log_arima_pyfunc_bundle(res, registered_model_name=None)
        rid = run.info.run_id

    model = mlflow.pyfunc.load_model(f"runs:/{rid}/arima_model_pyfunc")
    out = model.predict(pd.DataFrame({"history_stub": [0.0]}))
    assert out.shape == (1, 1)
    assert "forecast_d_ln_ipca_1step" in out.columns
