"""Exemplo ARIMA univariado em ``d_ln_ipca`` (após coleta BCB SGS 433)."""

from __future__ import annotations

from datetime import UTC, datetime

import mlflow
import polars as pl
from loguru import logger

from macroeconomia.config import get_settings
from macroeconomia.ingestion.series_types import observations_to_polars
from macroeconomia.ingestion.sources import BcbSgsSource
from macroeconomia.logging_config import configure_logging
from macroeconomia.mlflow_lineage import (
    numpy_series_sha256,
    resolve_git_commit_sha,
    sanitized_params,
)
from macroeconomia.models.time_series.arima_mlflow_pyfunc import (
    arima_fit_to_metadata_dict,
    log_arima_pyfunc_bundle,
    write_arima_metadata_json,
)
from macroeconomia.models.time_series.arima_model import fit_arima_grid
from macroeconomia.processing.transforms import with_logdiff_ipca_and_diff_selic
from macroeconomia.visualization.exporters import export_figure, macro_line_chart


def _panel_from_ipca_obs(df_ipca: pl.DataFrame) -> pl.DataFrame:
    """Monta painel mínimo com Selic constante sintética (somente para reaproveitar transform)."""
    # Requer colunas date, ipca_index, selic_pct para transform existente
    return df_ipca.with_columns(pl.lit(0.5).alias("selic_pct"))


def main() -> None:
    """CLI: ARIMA em inflação mensal (aprox. ``d_ln_ipca``)."""
    configure_logging()
    settings = get_settings()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    src = BcbSgsSource()
    obs = src.fetch("433")
    pdf = observations_to_polars(obs, "ipca_index").drop(["source", "series_id"])
    panel = _panel_from_ipca_obs(pdf)
    stat = with_logdiff_ipca_and_diff_selic(panel)
    y = stat["d_ln_ipca"].to_numpy()

    res, metrics = fit_arima_grid(y, max_p=3, max_q=3, d=0, trend="n")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name + "_arima")

    with mlflow.start_run(run_name=f"arima_ipca_{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}"):
        git_sha = resolve_git_commit_sha(cwd=settings.repo_root)
        dataset_hash = numpy_series_sha256(y)
        mlflow.set_tag("git_commit", git_sha)
        mlflow.set_tag("dataset_sha256", dataset_hash)
        mlflow.set_tag("pipeline", "arima_ipca")
        base_params = {
            "series": "bcb_433_d_ln_ipca",
            "seed": settings.seed,
            "git_commit": git_sha,
            "dataset_sha256": dataset_hash,
            "n_endog": len(y),
        }
        mlflow.log_params(sanitized_params(base_params))
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

        meta_path = settings.artifacts_dir / "arima_ipca_mlflow_metadata.json"
        write_arima_metadata_json(
            meta_path,
            arima_fit_to_metadata_dict(
                res,
                metrics,
                dataset_sha256=dataset_hash,
                max_p_grid=3,
                max_q_grid=3,
                d_fixed=0,
            ),
        )
        mlflow.log_artifact(str(meta_path))

        stem = settings.artifacts_dir / "arima_d_ln_ipca_fitted"
        fig = macro_line_chart(stat, y_columns=["d_ln_ipca"], title="d ln IPCA (mensal)")
        export_figure(fig, stem)
        mlflow.log_artifact(str(stem.with_suffix(".html")))

        reg_arima = (settings.mlflow_arima_registered_model_name or "").strip() or None
        try:
            log_arima_pyfunc_bundle(res, registered_model_name=reg_arima)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow pyfunc (ARIMA) ignorado: {}", exc)

        logger.info("ARIMA concluído. Métricas: {}", metrics)


if __name__ == "__main__":
    main()
