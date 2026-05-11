"""Coleta automatizada de séries públicas (BCB SGS, extensível a FRED/IMF)."""

from macroeconomia.ingestion.bcb_sgs import fetch_bcb_sgs_series
from macroeconomia.ingestion.errors import (
    BcbSgsError,
    BcbSgsHttpError,
    BcbSgsNetworkError,
    BcbSgsPayloadError,
)
from macroeconomia.ingestion.series_types import (
    SeriesObservation,
    bcb_points_to_observations,
    observations_metadata,
    observations_to_polars,
)
from macroeconomia.ingestion.sources import (
    BcbSgsSource,
    FredSource,
    MacroSeriesSource,
    SourceKind,
    WbdataSource,
    fetch_series_observations,
    get_series_source,
)

__all__ = [
    "BcbSgsError",
    "BcbSgsHttpError",
    "BcbSgsNetworkError",
    "BcbSgsPayloadError",
    "BcbSgsSource",
    "FredSource",
    "MacroSeriesSource",
    "SourceKind",
    "WbdataSource",
    "SeriesObservation",
    "bcb_points_to_observations",
    "fetch_bcb_sgs_series",
    "fetch_series_observations",
    "get_series_source",
    "observations_metadata",
    "observations_to_polars",
]
