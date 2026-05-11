"""Testes adicionais de persistência e ciclo de vida do cliente HTTP."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from macroeconomia.ingestion.bcb_sgs import (
    BcbSgsPoint,
    default_raw_path,
    fetch_bcb_sgs_series,
    save_raw_json,
)


def test_save_raw_json_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MACRO_DATA_RAW_DIR", str(tmp_path))
    from macroeconomia.config import get_settings

    get_settings.cache_clear()
    pts = [BcbSgsPoint(ref_date=date(2020, 1, 1), value=1.0)]
    out = tmp_path / "x.json"
    save_raw_json(1, pts, out)
    assert out.exists()


def test_default_raw_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MACRO_DATA_RAW_DIR", str(tmp_path))
    from macroeconomia.config import get_settings

    get_settings.cache_clear()
    p = default_raw_path(433)
    assert p.name == "bcb_sgs_433.json"


def test_fetch_closes_ephemeral_client() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = []
    mock_response.raise_for_status.return_value = None
    mock_client = MagicMock()
    mock_client.get.return_value = mock_response
    with patch("macroeconomia.ingestion.bcb_sgs.httpx.Client", return_value=mock_client):
        fetch_bcb_sgs_series(123)
    mock_client.close.assert_called_once()
