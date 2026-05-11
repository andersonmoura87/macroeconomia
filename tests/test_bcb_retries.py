"""Retries e classificação de erros na ingestão SGS."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from macroeconomia.ingestion.bcb_sgs import fetch_bcb_sgs_series
from macroeconomia.ingestion.errors import BcbSgsHttpError


def test_retries_on_503_then_success() -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] < 2:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(
            200,
            text='[{"data":"01/01/2020","valor":"1,0"},{"data":"01/02/2020","valor":"2,0"}]',
        )

    transport = httpx.MockTransport(handler)
    with patch("macroeconomia.ingestion.bcb_sgs.time.sleep", lambda *_args, **_kw: None):
        client = httpx.Client(transport=transport)
        pts = fetch_bcb_sgs_series(42, client=client, max_retries=4)
    assert len(pts) == 2
    assert state["n"] == 2


def test_http_404_fails_fast() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(404, text="nope"))
    client = httpx.Client(transport=transport)
    with pytest.raises(BcbSgsHttpError) as excinfo:
        fetch_bcb_sgs_series(1, client=client, max_retries=3)
    assert excinfo.value.status_code == 404
