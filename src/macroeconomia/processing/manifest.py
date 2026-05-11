"""Manifesto de artefatos diários (linhagem leve para o dashboard)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from macroeconomia.config import get_settings


def parquet_sha256(path: Path) -> str:
    """Hash SHA-256 do conteúdo do arquivo (para linhagem / MLflow)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset_manifest(
    parquet_path: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Grava ``daily_manifest.json`` ao lado do Parquet processado."""
    parquet_path = parquet_path.resolve()
    root = get_settings().repo_root
    payload: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "parquet_path": (
            str(parquet_path.relative_to(root))
            if parquet_path.is_relative_to(root)
            else str(parquet_path)
        ),
        "dataset_sha256": parquet_sha256(parquet_path),
    }
    if extra:
        payload["extra"] = extra
    out = parquet_path.parent / "daily_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def read_dataset_manifest(parquet_parent: Path) -> dict[str, Any] | None:
    """Lê manifesto se existir; retorna ``None`` se ausente."""
    candidate = parquet_parent.resolve() / "daily_manifest.json"
    if not candidate.is_file():
        return None
    try:
        return cast(dict[str, Any], json.loads(candidate.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return None
