"""Fontes de séries temporais com interface comum (BCB, FRED, World Bank, extensível)."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Protocol, runtime_checkable

import httpx
import pandas as pd
from loguru import logger

from macroeconomia.ingestion.bcb_sgs import fetch_bcb_sgs_series
from macroeconomia.ingestion.series_types import (
    SeriesObservation,
    bcb_points_to_observations,
)


@runtime_checkable
class MacroSeriesSource(Protocol):
    """Contrato para provedores de séries (ingestão desacoplada de ETL/modelos)."""

    @property
    def name(self) -> str:
        """Identificador curto (ex.: ``bcb_sgs``)."""
        ...

    def fetch(self, series_id: str, **kwargs: Any) -> list[SeriesObservation]:
        """Baixa toda a história disponível para ``series_id`` (formato depende da fonte)."""
        ...


class BcbSgsSource:
    """Fonte SGS do BCB. ``series_id`` deve ser o código numérico (ex.: ``\"433\"``)."""

    @property
    def name(self) -> str:
        return "bcb_sgs"

    def fetch(self, series_id: str, **kwargs: Any) -> list[SeriesObservation]:
        code = int(series_id)
        client = kwargs.get("client")
        timeout = float(kwargs.get("timeout", 30.0))
        max_retries = int(kwargs.get("max_retries", 4))
        pts = fetch_bcb_sgs_series(
            code,
            client=client if isinstance(client, httpx.Client) else None,
            timeout=timeout,
            max_retries=max_retries,
        )
        obs = bcb_points_to_observations(list(pts), code=code)
        logger.info("Fonte {} série {}: {} obs.", self.name, series_id, len(obs))
        return obs


class FredSource:
    """Fonte FRED (St. Louis Fed). Requer extra ``data-sources`` e ``FRED_API_KEY``.

    ``series_id`` é o código FRED (ex.: ``\"CPIAUCSL\"``).
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "fred"

    def fetch(self, series_id: str, **kwargs: Any) -> list[SeriesObservation]:
        try:
            from fredapi import Fred
        except ImportError as exc:  # pragma: no cover - ambiente sem extra
            raise RuntimeError(
                "Instale o extra `macroeconomia[data-sources]` (pacote `fredapi`) "
                "para usar a fonte FRED."
            ) from exc

        key = self._api_key or kwargs.get("api_key")
        if not key:
            import os

            key = os.environ.get("FRED_API_KEY")
        if not key:
            raise RuntimeError(
                "Defina FRED_API_KEY no ambiente ou passe api_key=... ao construir FredSource."
            )

        import pandas as pd

        fred = Fred(api_key=str(key))
        s = fred.get_series(series_id)
        out: list[SeriesObservation] = []
        for ts, val in s.items():
            if pd.isna(val):
                continue
            dt = pd.Timestamp(ts).date()
            out.append(
                SeriesObservation(
                    ref_date=dt,
                    value=float(val),
                    source=self.name,
                    series_id=series_id,
                )
            )
        out.sort(key=lambda o: o.ref_date)
        logger.info("Fonte {} série {}: {} obs.", self.name, series_id, len(out))
        return out


def _parse_wb_date(raw: object) -> date:
    """Converte rótulos de data do World Bank (ano, ano-mês, ISO) em ``date``."""
    if isinstance(raw, date):
        return raw
    s = str(raw).strip()
    if not s:
        raise ValueError("Data World Bank vazia.")
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Data World Bank não reconhecida: {raw!r}")
    return ts.date()


class WbdataSource:
    """Indicadores World Bank via ``wbdata`` (extra ``data-sources``).

    ``series_id`` é o código do indicador (ex.: ``\"NY.GDP.MKTP.CD\"``). Por padrão
    ``country="BR"`` (ISO-3166 alpha-2); pode ser sobrescrito em ``fetch(..., country=\"US\")``.
    """

    def __init__(self, country: str = "BR") -> None:
        self._country = country

    @property
    def name(self) -> str:
        return "wbdata"

    def fetch(self, series_id: str, **kwargs: Any) -> list[SeriesObservation]:
        try:
            import wbdata as wb
        except ImportError as exc:  # pragma: no cover - ambiente sem extra
            raise RuntimeError(
                "Instale o extra `macroeconomia[data-sources]` (pacote `wbdata`) "
                "para usar a fonte World Bank."
            ) from exc

        country = str(kwargs.get("country", self._country))
        parse_dates = bool(kwargs.get("parse_dates", True))
        try:
            rows = wb.get_data(series_id, country=country, parse_dates=parse_dates)
        except TypeError:
            rows = wb.get_data(series_id, country=country)

        out: list[SeriesObservation] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            d_raw = row.get("date")
            if d_raw is None:
                d_raw = row.get("Date")
            v_raw = row.get("value")
            if v_raw is None:
                v_raw = row.get("Value")
            if d_raw is None or v_raw is None or v_raw == "":
                continue
            try:
                val = float(v_raw)
            except (TypeError, ValueError):
                continue
            if pd.isna(val):
                continue
            try:
                dt = _parse_wb_date(d_raw)
            except ValueError:
                continue
            out.append(
                SeriesObservation(
                    ref_date=dt,
                    value=val,
                    source=self.name,
                    series_id=series_id,
                )
            )
        out.sort(key=lambda o: o.ref_date)
        logger.info("Fonte {} série {}: {} obs.", self.name, series_id, len(out))
        return out


SourceKind = Literal["bcb_sgs", "fred", "wbdata"]


def get_series_source(kind: SourceKind, **kwargs: Any) -> MacroSeriesSource:
    """Factory de fontes suportadas.

    Args:
        kind: ``bcb_sgs``, ``fred`` ou ``wbdata``.
        kwargs: Para ``fred``, opcional ``api_key``. Para ``wbdata``, opcional ``country``.
    """
    if kind == "bcb_sgs":
        return BcbSgsSource()
    if kind == "fred":
        return FredSource(api_key=kwargs.get("api_key"))
    if kind == "wbdata":
        return WbdataSource(country=str(kwargs.get("country", "BR")))
    raise ValueError(f"Fonte desconhecida: {kind!r}")


def fetch_series_observations(
    kind: SourceKind,
    series_id: str,
    **kwargs: Any,
) -> list[SeriesObservation]:
    """Atalho: obtém fonte e executa ``fetch``."""
    return get_series_source(kind, **kwargs).fetch(series_id, **kwargs)
