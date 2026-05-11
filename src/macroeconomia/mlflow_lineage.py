"""Metadados de linhagem para MLflow (Git, hash de dataset); evita logging de segredos."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_PARAMS_SECRET_KEYS = frozenset(
    {
        "api_key",
        "password",
        "token",
        "secret",
        "fred_api_key",
        "aws_access_key_id",
        "aws_secret_access_key",
    },
)


def numpy_series_sha256(series: NDArray[Any] | list[float]) -> str:
    """SHA-256 da série NumPy (bytes float64 ordenados para linhagem de dataset."""

    arr = np.asarray(series, dtype=np.float64).ravel()
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    digest = hashlib.sha256(arr.tobytes())
    return digest.hexdigest()


def resolve_git_commit_sha(*, cwd: Path | None = None) -> str:
    """SHA curto ou longo do commit atual; fallback ``MACRO_GIT_COMMIT`` ou ``unknown``."""
    env_sha = (
        os.environ.get("MACRO_GIT_COMMIT")
        or os.environ.get("GITHUB_SHA")
        or os.environ.get("SOURCE_VERSION")
        or ""
    ).strip()
    if env_sha:
        return env_sha[:40]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        return out[:40] if out else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def sanitized_params(params: dict[str, Any]) -> dict[str, str]:
    """Converte parâmetros para strings seguras para ``mlflow.log_params``."""

    out: dict[str, str] = {}
    for k, v in params.items():
        lk = k.lower()
        if any(tok in lk for tok in _PARAMS_SECRET_KEYS):
            continue
        out[str(k)] = "" if v is None else str(v)
    return out
