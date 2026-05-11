"""Testes de configuração."""

from __future__ import annotations

from macroeconomia.config import Settings, get_settings


def test_get_settings_cached(monkeypatch) -> None:
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    assert isinstance(s1, Settings)
    assert s1.seed >= 0
