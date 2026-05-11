"""Wrapper MLflow pyfunc para VAR (statsmodels) + registro no Model Registry."""

from __future__ import annotations

import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import mlflow.pyfunc
import pandas as pd
from loguru import logger
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from statsmodels.tsa.vector_ar.var_model import VARResults


class VarForecastPyFuncModel(mlflow.pyfunc.PythonModel):
    """Um passo à frente de ``VARResults.forecast`` (requer últimas ``k_ar`` linhas endógenas)."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        bundle_path = context.artifacts["var_bundle"]
        with Path(bundle_path).open("rb") as f:
            bundle = pickle.load(f)
        self._res: VARResults = bundle["res"]
        self._value_cols: list[str] = bundle["value_cols"]
        self._k_ar: int = int(bundle["k_ar"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,  # noqa: ARG002
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        missing = set(self._value_cols) - set(model_input.columns)
        if missing:
            raise ValueError(f"Colunas faltando no input: {sorted(missing)}")
        arr = model_input[self._value_cols].astype(float).values
        if arr.shape[0] < self._k_ar:
            raise ValueError(
                f"Mínimo {self._k_ar} observações lag necessarias; obtido {arr.shape[0]}."
            )
        fc = self._res.forecast(arr[-self._k_ar :], steps=1)
        colnames = [f"forecast_{c}" for c in self._value_cols]
        return pd.DataFrame(fc, columns=colnames)


def build_var_forecast_signature(value_cols: list[str]) -> ModelSignature:
    """Assinatura explícita (shapes colunares) para o Registry."""

    in_cols = [ColSpec("double", n) for n in value_cols]
    out_cols = [ColSpec("double", f"forecast_{n}") for n in value_cols]
    return ModelSignature(inputs=Schema(in_cols), outputs=Schema(out_cols))


def log_var_pyfunc_bundle(
    res: VARResults,
    *,
    value_cols: list[str],
    k_ar: int,
    artifact_rel_path: str = "var_model_pyfunc",
    registered_model_name: str | None = "macroeconomia_var_ipca_selic_stationary",
) -> str | None:
    """Serializa VAR, faz ``pyfunc.log_model`` e registra opcionalmente.

    Args:
        res: Resultado VAR em séries já estacionarizadas.
        value_cols: Colunas endógenas encaixadas (ex.: d_ln_ipca, d_selic).
        k_ar: Ordem autorregressiva.
        artifact_rel_path: Caminho do artefato no run.
        registered_model_name: Nome no Model Registry ou ``None`` para não registrar.

    Returns:
        URI do modelo registado quando ``register_model`` roda sem erro;
        caso contrário ``None``.
    """

    sig = build_var_forecast_signature(value_cols)
    bundle = {"res": res, "value_cols": list(value_cols), "k_ar": int(k_ar)}

    with TemporaryDirectory() as tmp:
        pkl = Path(tmp) / "var_bundle.pkl"
        pkl.write_bytes(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
        mlflow.pyfunc.log_model(
            python_model=VarForecastPyFuncModel(),
            artifact_path=artifact_rel_path,
            artifacts={"var_bundle": str(pkl)},
            signature=sig,
            pip_requirements=["statsmodels>=0.14.0", "pandas>=2.2.0", "numpy>=1.26.0"],
        )

    run = mlflow.active_run()
    if run is None or registered_model_name is None:
        return None
    uri = f"runs:/{run.info.run_id}/{artifact_rel_path}"
    try:
        mv = mlflow.register_model(uri, registered_model_name)
        return f"models:/{mv.name}/{mv.version}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model Registry opcional indisponível: {}", exc)
        return None
