"""Resolução de caminhos, carregamento de Parquet e manifesto para o Streamlit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from macroeconomia.config import Settings
from macroeconomia.processing.macro_panel import MacroBivariateSchema
from macroeconomia.processing.manifest import read_dataset_manifest

try:
    from pandera.errors import SchemaError
except ImportError:  # pragma: no cover
    SchemaError = Exception  # type: ignore[misc, assignment]

ResolvedSource = Literal["demo_forced", "processed", "demo_fallback"]


@dataclass(frozen=True)
class PanelLoadMeta:
    """Metadados de origem dos dados."""

    parquet_path: Path
    source: ResolvedSource


class PanelResolutionError(RuntimeError):
    """Falha ao localizar dados para o dashboard."""


class PanelValidationError(RuntimeError):
    """Dados presentes mas fora do contrato esperado (Pandera/schema)."""


def resolve_panel_parquet_path(settings: Settings) -> PanelLoadMeta:
    """Resolve ``macro_ipca_selic.parquet`` processado ou fallback demo."""

    demo = settings.demo_parquet_relative
    proc = settings.data_processed_dir / settings.dashboard_parquet_name
    if settings.demo_mode:
        if not demo.is_file():
            raise PanelResolutionError(
                "MACRO_DEMO_MODE=true mas o arquivo demo não existe. Rode "
                "`python scripts/gen_demo_panel.py` ou desative DEMO_MODE."
            )
        return PanelLoadMeta(parquet_path=demo, source="demo_forced")
    if proc.is_file():
        return PanelLoadMeta(parquet_path=proc.resolve(), source="processed")
    if demo.is_file():
        return PanelLoadMeta(parquet_path=demo.resolve(), source="demo_fallback")
    raise PanelResolutionError(
        "Nenhum Parquet em data/processed/ nem data/demo/. "
        "Use o botão lateral 'Atualizar dados' ou gere o demo: "
        "`python scripts/gen_demo_panel.py`."
    )


def validate_panel_contract(df: pl.DataFrame) -> pl.DataFrame:
    """Valida contra o mesmo esquema Pandera que o pipeline de produção usa."""

    try:
        return MacroBivariateSchema.validate(df)
    except SchemaError as exc:
        raise PanelValidationError(str(exc)) from exc


def load_manifest_for_parent(parquet_path: Path) -> dict[str, Any] | None:
    """Último manifesto gravado pelo ETL ao lado do Parquet processado."""

    return read_dataset_manifest(parquet_path.parent)


def parquet_last_modified(parquet_path: Path) -> str | None:
    """ISO mtime como fallback quando não há manifesto."""

    try:
        ts = parquet_path.stat().st_mtime_ns
        from datetime import UTC, datetime

        return datetime.fromtimestamp(ts / 1e9, tz=UTC).isoformat()
    except OSError:
        return None
