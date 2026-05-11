"""ARIMA statsmodels empacotado para MLflow pyfunc (previsão 1 passo)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from loguru import logger
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema


class ArimaForecastOneStep(mlflow.pyfunc.PythonModel):
    """Usa o estado de ``ARIMA.fit`` para ``forecast(steps=1)``."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        pth = Path(context.artifacts["arima_bundle"])
        blob = pickle.loads(pth.read_bytes())
        self._res: Any = blob["res_obj"]

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,  # noqa: ARG002
        model_input: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        fc = self._res.forecast(steps=1)
        val = float(fc[0]) if hasattr(fc, "__getitem__") else float(fc)
        return pd.DataFrame({"forecast_d_ln_ipca_1step": [val]})


def build_arima_signature() -> ModelSignature:
    """Stub de entrada (ignored) para compatibilidade com validação MLflow."""

    return ModelSignature(
        inputs=Schema([ColSpec("double", "history_stub")]),
        outputs=Schema([ColSpec("double", "forecast_d_ln_ipca_1step")]),
    )


def arima_fit_to_metadata_dict(
    res_obj: Any,
    metrics: dict[str, float],
    *,
    dataset_sha256: str,
    endog_series_id: str = "bcb_433_d_ln_ipca",
    max_p_grid: int | None = None,
    max_q_grid: int | None = None,
    d_fixed: int | None = None,
) -> dict[str, Any]:
    """Resumo para artefacto JSON."""

    model = getattr(res_obj, "model", None)
    raw_order = getattr(model, "order", (0, 0, 0)) if model is not None else (0, 0, 0)
    order_list = [int(x) for x in raw_order]
    while len(order_list) < 3:
        order_list.append(0)
    meta: dict[str, Any] = {
        "endog_series_id": endog_series_id,
        "arima_order_p_d_q": order_list[:3],
        "nobs_fit": getattr(res_obj, "nobs", None),
        "aic": metrics.get("aic"),
        "bic": metrics.get("bic"),
        "hqic": metrics.get("hqic"),
        "dataset_sha256": dataset_sha256,
    }
    if max_p_grid is not None:
        meta["grid_max_p"] = max_p_grid
    if max_q_grid is not None:
        meta["grid_max_q"] = max_q_grid
    if d_fixed is not None:
        meta["d_fixed"] = d_fixed
    return meta


def write_arima_metadata_json(dest: Path, payload: dict[str, Any]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def log_arima_pyfunc_bundle(
    res_obj: Any,
    *,
    artifact_rel_path: str = "arima_model_pyfunc",
    registered_model_name: str | None = None,
) -> str | None:
    """Persiste pyfunc; ``register_model`` apenas se ``registered_model_name`` estiver definido."""

    sig = build_arima_signature()
    blob = pickle.dumps({"res_obj": res_obj}, protocol=pickle.HIGHEST_PROTOCOL)

    with TemporaryDirectory() as tmp:
        bp = Path(tmp) / "arima_bundle.pkl"
        bp.write_bytes(blob)
        mlflow.pyfunc.log_model(
            python_model=ArimaForecastOneStep(),
            artifact_path=artifact_rel_path,
            artifacts={"arima_bundle": str(bp)},
            signature=sig,
            pip_requirements=["statsmodels>=0.14.0", "pandas>=2.2.0", "numpy>=1.26.0"],
            input_example=pd.DataFrame({"history_stub": [0.0]}),
        )

    run = mlflow.active_run()
    if run is None or registered_model_name is None:
        return None
    uri = f"runs:/{run.info.run_id}/{artifact_rel_path}"
    try:
        mv = mlflow.register_model(uri, registered_model_name)
        return f"models:/{mv.name}/{mv.version}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model Registry (ARIMA) opcional falhou: {}", exc)
        return None
