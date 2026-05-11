"""Fontes de ingestão unificadas (BCB / FRED / World Bank)."""

from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import httpx
import pytest

from macroeconomia.ingestion.series_types import observations_metadata
from macroeconomia.ingestion.sources import (
    BcbSgsSource,
    FredSource,
    WbdataSource,
    fetch_series_observations,
    get_series_source,
)


def test_bcb_source_mocked_transport() -> None:
    payload = '[{"data":"01/01/2020","valor":"1,0"},{"data":"01/02/2020","valor":"2,0"}]'
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text=payload))
    client = httpx.Client(transport=transport)
    src = BcbSgsSource()
    obs = src.fetch("123", client=client)
    assert obs[0].source == "bcb_sgs"
    assert obs[0].series_id == "123"
    meta = observations_metadata(obs)
    assert meta["n"] == 2


def test_fetch_series_observations_bcb() -> None:
    payload = '[{"data":"01/01/2020","valor":"1,0"}]'
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text=payload))
    client = httpx.Client(transport=transport)
    obs = fetch_series_observations("bcb_sgs", "7", client=client)
    assert len(obs) == 1


def test_fred_fetch_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    src = FredSource(api_key=None)
    with pytest.raises(RuntimeError, match="FRED|fredapi|data-sources"):
        src.fetch("GDP")


def test_get_series_source_fred_factory() -> None:
    src = get_series_source("fred", api_key="dummy")
    assert src.name == "fred"


def test_get_series_source_wbdata_factory() -> None:
    src = get_series_source("wbdata", country="US")
    assert src.name == "wbdata"


def test_wbdata_source_mocked(monkeypatch: pytest.MonkeyPatch) -> None:

    def _get_data(
        _indicator: str,
        _country: str = "BR",
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        return [
            {"date": "2019", "value": "10"},
            {"date": "2020", "value": 20},
        ]

    fake = types.SimpleNamespace(get_data=_get_data)
    monkeypatch.setitem(sys.modules, "wbdata", fake)
    src = WbdataSource(country="BR")
    obs = src.fetch("NY.GDP.MKTP.CD")
    assert len(obs) == 2
    assert obs[0].value == 10.0
    assert obs[0].source == "wbdata"
    assert obs[1].ref_date.year == 2020


def test_fetch_series_observations_wbdata_mocked(monkeypatch: pytest.MonkeyPatch) -> None:

    def _gd(_i: str, _c: str = "BR", **_kw: object) -> list[dict[str, object]]:
        return [{"date": "2021", "value": 1.5}]

    fake = types.SimpleNamespace(get_data=_gd)
    monkeypatch.setitem(sys.modules, "wbdata", fake)
    obs = fetch_series_observations("wbdata", "NY.GDP.MKTP.CD")
    assert len(obs) == 1
    assert obs[0].value == 1.5


def test_wbdata_fetch_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> object:
        if name == "wbdata":
            raise ImportError("blocked wbdata")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    src = WbdataSource()
    with pytest.raises(RuntimeError, match="World Bank|wbdata"):
        src.fetch("NY.GDP.MKTP.CD")


def test_get_series_source_unknown() -> None:
    with pytest.raises(ValueError):
        get_series_source("invalid_kind")  # type: ignore[arg-type]


def test_observations_metadata_empty() -> None:
    from macroeconomia.ingestion.series_types import observations_metadata

    assert observations_metadata([]) == {"n": 0}
