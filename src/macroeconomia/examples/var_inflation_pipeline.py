"""Pipeline exemplo: IPCA × Selic → painel → VAR estacionário → MLflow + artefatos.

Códigos SGS usados (BCB):
    * ``433`` — IPCA (número-índice com base fixa).
    * ``432`` — Taxa Selic (% a.m.).

Referências:
    Lütkepohl, H. *New Introduction to Multiple Time Series Analysis* (2005).
    Sims, C. A. (1980). Macroeconomics and Reality. *Econometrica*.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
from loguru import logger
from statsmodels.tsa.vector_ar.var_model import VARResults

from macroeconomia.config import get_settings
from macroeconomia.ingestion.bcb_sgs import (
    BcbSgsPoint,
    default_raw_path,
    fetch_bcb_sgs_series,
    save_raw_json,
)
from macroeconomia.logging_config import configure_logging
from macroeconomia.mlflow_lineage import resolve_git_commit_sha, sanitized_params
from macroeconomia.models.time_series.var_diagnostics import (
    fevd_summary_text,
    granger_causality_markdown,
    irf_to_long_frame,
    var_is_stable,
    var_max_root_modulus,
)
from macroeconomia.models.time_series.var_mlflow_pyfunc import log_var_pyfunc_bundle
from macroeconomia.models.time_series.var_model import fit_var
from macroeconomia.models.time_series.var_oos import (
    temporal_train_test_split,
    var_multistep_oos_metrics,
)
from macroeconomia.processing.drift_alerts import log_macro_panel_drift_alerts
from macroeconomia.processing.macro_panel import (
    build_macro_bivariate_panel,
    validate_macro_bivariate,
    write_parquet_partition,
)
from macroeconomia.processing.manifest import parquet_sha256, write_dataset_manifest
from macroeconomia.processing.transforms import with_logdiff_ipca_and_diff_selic
from macroeconomia.visualization.exporters import export_figure, macro_line_chart
from macroeconomia.visualization.var_irf import irf_long_frame_line_chart


def _residual_matrix_rmse(res: VARResults) -> float:
    """RMSE global dos resíduos multivariados (in-sample)."""
    arr = np.asarray(res.resid, dtype=float)
    return float(np.sqrt(np.nanmean(arr**2)))


def _write_summary_md(path: Path, *, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_var_inflation_pipeline(
    *,
    ipca_points: Sequence[BcbSgsPoint] | None = None,
    selic_points: Sequence[BcbSgsPoint] | None = None,
    ipca_code: int = 433,
    selic_code: int = 432,
) -> Path:
    """Executa o pipeline VAR (política monetária × inflação) ponta a ponta.

    Args:
        ipca_points: Pontos IPCA já carregados (pula download se ambos forem dados).
        selic_points: Pontos Selic já carregados.
        ipca_code: Código SGS usado ao baixar IPCA (padrão ``433``).
        selic_code: Código SGS usado ao baixar Selic (padrão ``432``).

    Returns:
        Caminho do arquivo Markdown de resumo gerado.
    """
    settings = get_settings()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    if ipca_points is None:
        ipca_points = fetch_bcb_sgs_series(ipca_code)
    if selic_points is None:
        selic_points = fetch_bcb_sgs_series(selic_code)

    ipca_list = list(ipca_points)
    selic_list = list(selic_points)
    save_raw_json(ipca_code, ipca_list, default_raw_path(ipca_code))
    save_raw_json(selic_code, selic_list, default_raw_path(selic_code))

    raw_panel = build_macro_bivariate_panel(ipca_list, selic_list)
    panel = validate_macro_bivariate(raw_panel)
    log_macro_panel_drift_alerts(panel)
    processed_path = settings.data_processed_dir / "macro_ipca_selic.parquet"
    write_parquet_partition(panel, processed_path)
    write_dataset_manifest(
        processed_path,
        extra={"pipeline": "var_ipca_selic", "ipca_code": ipca_code, "selic_code": selic_code},
    )

    stat_panel = with_logdiff_ipca_and_diff_selic(panel)
    value_cols = ["d_ln_ipca", "d_selic"]
    maxlags = 6

    oos_metrics: dict[str, float] = {}
    try:
        train_stat, test_stat = temporal_train_test_split(stat_panel, train_ratio=0.85)
        oos_metrics = var_multistep_oos_metrics(
            train_stat,
            test_stat,
            value_cols,
            maxlags=maxlags,
            ic="aic",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Avaliação OOS não executada: {}", exc)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"var_ipca_selic_{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}"):
        git_sha = resolve_git_commit_sha(cwd=settings.repo_root)
        dataset_hash = parquet_sha256(processed_path)
        mlflow.set_tag("git_commit", git_sha)
        mlflow.set_tag("dataset_sha256", dataset_hash)
        mlflow.set_tag("pipeline", "var_ipca_selic")
        base_params = {
            "ipca_sgs": ipca_code,
            "selic_sgs": selic_code,
            "maxlags": maxlags,
            "ic": "aic",
            "seed": settings.seed,
            "git_commit": git_sha,
            "dataset_sha256": dataset_hash,
        }
        mlflow.log_params(sanitized_params(base_params))
        mlflow.log_artifact(str(processed_path))

        res_typed = fit_var(stat_panel, value_cols, maxlags=maxlags, ic="aic")
        k_ar = int(res_typed.k_ar)
        reg_name = (settings.mlflow_registered_model_name or "").strip() or None
        mlflow.log_params(
            sanitized_params(
                {
                    "var_endog_nobs": stat_panel.height,
                    "var_endog_neqs": len(value_cols),
                    "var_k_ar": k_ar,
                }
            )
        )
        mlflow.log_metrics(
            {
                "var_lag_order": float(k_ar),
                "aic": float(res_typed.aic),
                "bic": float(res_typed.bic),
                "hqic": float(res_typed.hqic),
                "var_max_root_modulus": float(var_max_root_modulus(res_typed)),
                "var_stable": float(var_is_stable(res_typed)),
            }
        )
        if oos_metrics:
            mlflow.log_metrics({k: float(v) for k, v in oos_metrics.items()})

        try:
            rmse_res = _residual_matrix_rmse(res_typed)
            mlflow.log_metrics({"residual_rmse": rmse_res})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Não foi possível calcular RMSE dos resíduos: {}", exc)

        try:
            log_var_pyfunc_bundle(
                res_typed,
                value_cols=list(value_cols),
                k_ar=k_ar,
                registered_model_name=reg_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Registro pyfunc/MLflow Registry ignorado: {}", exc)

        stem = settings.artifacts_dir / "ipca_selic_levels"
        fig = macro_line_chart(panel, y_columns=["ipca_index", "selic_pct"], title="IPCA e Selic")
        export_figure(fig, stem)
        mlflow.log_artifact(str(stem.with_suffix(".html")))
        if stem.with_suffix(".png").exists():
            mlflow.log_artifact(str(stem.with_suffix(".png")))

        irf_stem = settings.artifacts_dir / "var_irf_orthogonal"
        try:
            irf_long = irf_to_long_frame(res_typed, periods=12, orthogonalized=True)
            irf_fig = irf_long_frame_line_chart(
                irf_long,
                title="IRF ortogonalizada (Cholesky) — VAR IPCA × Selic",
            )
            export_figure(irf_fig, irf_stem)
            mlflow.log_artifact(str(irf_stem.with_suffix(".html")))
            if irf_stem.with_suffix(".png").exists():
                mlflow.log_artifact(str(irf_stem.with_suffix(".png")))
        except np.linalg.LinAlgError as exc:  # pragma: no cover - depende da amostra
            logger.warning("IRF omitida (problema numérico na ortogonalização): {}", exc)
            irf_stem.with_suffix(".txt").write_text(
                f"IRF indisponível (LinAlgError): {exc}\n", encoding="utf-8"
            )
            mlflow.log_artifact(str(irf_stem.with_suffix(".txt")))

        fevd_path = settings.artifacts_dir / "VAR_FEVD.txt"
        try:
            fevd_path.write_text(fevd_summary_text(res_typed, periods=12), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            fevd_path.write_text(f"FEVD indisponível: {exc}\n", encoding="utf-8")
        mlflow.log_artifact(str(fevd_path))

        summary = settings.artifacts_dir / "VAR_IPCA_SELIC_SUMMARY.md"
        lines: list[str] = [
            "# VAR IPCA × Selic",
            "",
            f"- **Ordem estimada (lags):** {k_ar}",
            (
                f"- **AIC / BIC / HQIC:** {res_typed.aic:.4f} / "
                f"{res_typed.bic:.4f} / {res_typed.hqic:.4f}"
            ),
            f"- **Observações (nível):** {panel.height}",
            f"- **Observações (estacionárias):** {stat_panel.height}",
            f"- **Estável (raízes companion):** {var_is_stable(res_typed)} "
            f"(máx. |λ| = {var_max_root_modulus(res_typed):.4f})",
            f"- **Parquet:** `{processed_path}`",
            f"- **Figuras:** `{stem}.html`, `{stem}.png`, `{irf_stem}.html`",
            f"- **FEVD (texto):** `{fevd_path}`",
            "",
            "## Out-of-sample (corte temporal 85/15)",
        ]
        if oos_metrics:
            rmse_m = oos_metrics.get("oos_rmse_mean", float("nan"))
            mae_m = oos_metrics.get("oos_mae_mean", float("nan"))
            lines.extend(
                [
                    "",
                    f"- **RMSE médio (multi-passos):** {rmse_m:.6f}",
                    f"- **MAE médio (multi-passos):** {mae_m:.6f}",
                ]
            )
            for k, v in sorted(oos_metrics.items()):
                if k in {"oos_rmse_mean", "oos_mae_mean"}:
                    continue
                lines.append(f"- **{k}:** {v:.6f}")
        else:
            lines.append("_Métricas OOS indisponíveis (amostra ou corte)._")

        try:
            granger_md = granger_causality_markdown(res_typed)
        except Exception as exc:  # noqa: BLE001
            granger_md = f"_Causalidade de Granger indisponível: {exc}_"

        lines.extend(
            [
                "",
                granger_md,
                "",
                "Interpretação: `d_ln_ipca` aproxima inflação mensal em termos log; "
                "`d_selic` captura mudanças na taxa de juros. Valide diagnósticos "
                "dos resíduos e a ordem Cholesky antes de inferência estrutural sobre IRFs.",
            ]
        )
        _write_summary_md(summary, lines=lines)
        mlflow.log_artifact(str(summary))
        logger.info("Pipeline concluído. Resumo em {}.", summary)
        return summary


def main() -> None:
    """CLI: configura logging e executa o pipeline com download BCB."""
    configure_logging()
    run_var_inflation_pipeline()


if __name__ == "__main__":
    main()
