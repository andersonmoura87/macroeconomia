"""Tipos neutros de domínio para séries temporais (multi-fonte)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl

from macroeconomia.ingestion.bcb_sgs import BcbSgsPoint


@dataclass(frozen=True, slots=True)
class SeriesObservation:
    """Uma observação de série com metadados mínimos de proveniência."""

    ref_date: date
    value: float
    source: str
    series_id: str


def bcb_points_to_observations(
    points: list[BcbSgsPoint],
    *,
    code: int,
) -> list[SeriesObservation]:
    """Converte pontos SGS legados em observações genéricas."""
    sid = str(code)
    return [
        SeriesObservation(ref_date=p.ref_date, value=p.value, source="bcb_sgs", series_id=sid)
        for p in points
    ]


def observations_to_polars(points: list[SeriesObservation], value_column: str) -> pl.DataFrame:
    """Monta um ``DataFrame`` longo Polars a partir de observações."""
    return pl.DataFrame(
        {
            "date": [p.ref_date for p in points],
            value_column: [p.value for p in points],
            "source": [p.source for p in points],
            "series_id": [p.series_id for p in points],
        }
    )


def observations_metadata(points: list[SeriesObservation]) -> dict[str, Any]:
    """Resumo leve para logging/MLflow."""
    if not points:
        return {"n": 0}
    src = {p.source for p in points}
    ids = {p.series_id for p in points}
    return {
        "n": len(points),
        "sources": sorted(src),
        "series_ids": sorted(ids),
        "start": min(p.ref_date for p in points).isoformat(),
        "end": max(p.ref_date for p in points).isoformat(),
    }
