"""Testes do cliente SGS."""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from macroeconomia.ingestion.bcb_sgs import (
    BcbSgsPoint,
    fetch_bcb_sgs_series,
    series_to_polars,
)
from macroeconomia.ingestion.errors import BcbSgsPayloadError


def test_parse_via_mocked_client() -> None:
    payload = '[{"data":"01/02/2020","valor":"1,23"},{"data":"01/03/2020","valor":"4"}]'
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=payload))
    client = httpx.Client(transport=transport)
    pts = fetch_bcb_sgs_series(999, client=client)
    assert pts == [
        BcbSgsPoint(ref_date=date(2020, 2, 1), value=1.23),
        BcbSgsPoint(ref_date=date(2020, 3, 1), value=4.0),
    ]


def test_series_to_polars() -> None:
    pts = [BcbSgsPoint(ref_date=date(2020, 1, 1), value=1.0)]
    df = series_to_polars(pts, "x")
    assert df.columns == ["date", "x"]
    assert df.height == 1


def test_invalid_json_raises() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="[]"))
    client = httpx.Client(transport=transport)
    assert fetch_bcb_sgs_series(1, client=client) == []

    bad = httpx.MockTransport(lambda request: httpx.Response(200, text='[{"data":"x"}]'))
    with httpx.Client(transport=bad) as c, pytest.raises(BcbSgsPayloadError):
        fetch_bcb_sgs_series(1, client=c)
