"""Caminhos absolutos em configuração."""

from __future__ import annotations

from pathlib import Path

from macroeconomia.config import Settings


def test_settings_absolute_paths_preserved(tmp_path: Path) -> None:
    abs_raw = tmp_path / "abs_raw"
    abs_raw.mkdir()
    s = Settings(
        data_raw_dir=abs_raw,
        data_processed_dir=tmp_path / "proc",
        data_external_dir=tmp_path / "ext",
        artifacts_dir=tmp_path / "art",
    )
    assert s.data_raw_dir == abs_raw.resolve()
