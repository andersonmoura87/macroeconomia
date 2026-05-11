"""Cliente resiliente para a API pública SGS do Banco Central do Brasil.

A API oficial expõe séries por **código numérico** (ex.: ``433`` IPCA em nível
índice; ``432`` taxa Selic em **% ao mês**). Unidades e frequência dependem da
série escolhida; consulte o catálogo SGS antes de modelar.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl
from loguru import logger

from macroeconomia.config import get_settings
from macroeconomia.ingestion.errors import BcbSgsHttpError, BcbSgsNetworkError, BcbSgsPayloadError

BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

# HTTP: repetir em sobrecarga / manutenção transitória / rate limit.
_RETRIABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class BcbSgsPoint:
    """Um ponto de série temporal retornado pelo SGS."""

    ref_date: date
    value: float


def _parse_bcb_date(raw: str) -> date:
    """Converte string ``dd/MM/yyyy`` em :class:`date`."""
    return datetime.strptime(raw, "%d/%m/%Y").date()


def _parse_value(raw: str) -> float:
    """Converte valor numérico com vírgula decimal (padrão BCB)."""
    normalized = raw.replace(",", ".")
    return float(normalized)


def _parse_payload_rows(payload: Sequence[dict[str, Any]], code: int) -> list[BcbSgsPoint]:
    out: list[BcbSgsPoint] = []
    for row in payload:
        try:
            d = _parse_bcb_date(str(row["data"]))
            v = _parse_value(str(row["valor"]))
        except (KeyError, ValueError) as exc:
            raise BcbSgsPayloadError(
                f"Linha SGS inválida para código {code}: {row!r}. "
                "Esperado dict com chaves 'data' (dd/MM/yyyy) e 'valor' numérico."
            ) from exc
        out.append(BcbSgsPoint(ref_date=d, value=v))
    out.sort(key=lambda p: p.ref_date)
    return out


def fetch_bcb_sgs_series(
    code: int,
    *,
    client: httpx.Client | None = None,
    timeout: float = 30.0,
    max_retries: int = 4,
    backoff_base_seconds: float = 0.75,
) -> list[BcbSgsPoint]:
    """Baixa todos os pontos disponíveis de uma série SGS com retries exponenciais.

    Política de resiliência:

    * **Rede** (:class:`httpx.RequestError`): até ``max_retries`` tentativas com
      *backoff* exponencial ``backoff_base_seconds * 2**attempt``.
    * **HTTP 429 / 5xx** (lista em ``_RETRIABLE_STATUSES``): mesma política.
    * **HTTP 4xx** restantes: falha imediata (:class:`BcbSgsHttpError`).
    * **JSON inválido** ou payload que não seja lista: :class:`BcbSgsPayloadError`
      (sem retry).
    * **Linhas corrompidas** após JSON válido: :class:`BcbSgsPayloadError`.

    Args:
        code: Código numérico da série no SGS (ex.: ``433`` IPCA índice).
        client: Cliente HTTP opcional (útil para testes com transport mock).
        timeout: Timeout total em segundos quando ``client`` é ``None``.
        max_retries: Número máximo de **tentativas** (>=1). Ex.: ``4`` = 1 inicial
            + até 3 reenvios.
        backoff_base_seconds: Base multiplicada por ``2**attempt`` entre tentativas.

    Returns:
        Lista ordenada cronologicamente de observações.

    Raises:
        BcbSgsNetworkError: Falha de transporte após esgotar tentativas.
        BcbSgsHttpError: Status HTTP não retentável ou 4xx definitivo.
        BcbSgsPayloadError: Corpo JSON inválido ou linhas SGS inesperadas.
    """
    if max_retries < 1:
        raise ValueError("max_retries deve ser >= 1.")

    url = BCB_SGS_URL.format(code=code)
    own_client = client is None
    c = client or httpx.Client(timeout=timeout)

    try:
        for attempt in range(max_retries):
            try:
                response = c.get(url, params={"formato": "json"})
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in _RETRIABLE_STATUSES and attempt < max_retries - 1:
                    delay = backoff_base_seconds * (2**attempt)
                    logger.warning(
                        "SGS {} HTTP {} — nova tentativa em {:.2f}s ({}/{}).",
                        code,
                        status,
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)
                    continue
                raise BcbSgsHttpError(
                    f"HTTP {status} ao consultar SGS {code}: {exc.response.text[:200]!r}",
                    status_code=status,
                ) from exc
            except httpx.RequestError as exc:
                if attempt < max_retries - 1:
                    delay = backoff_base_seconds * (2**attempt)
                    logger.warning(
                        "SGS {} rede ({}) — nova tentativa em {:.2f}s ({}/{}).",
                        code,
                        type(exc).__name__,
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)
                    continue
                raise BcbSgsNetworkError(
                    f"Falha de rede ao consultar SGS {code} após {max_retries} tentativas: {exc}"
                ) from exc

            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise BcbSgsPayloadError(
                    f"JSON inválido na resposta SGS {code} (tamanho {len(response.text)})."
                ) from exc

            if not isinstance(payload, list):
                raise BcbSgsPayloadError(
                    f"Resposta SGS {code} não é uma lista JSON (tipo {type(payload).__name__})."
                )

            points = _parse_payload_rows(payload, code)
            logger.info("Série SGS {} carregada com {} observações.", code, len(points))
            return points
        raise BcbSgsNetworkError(
            f"SGS {code}: esgotadas {max_retries} tentativas sem concluir o download."
        )
    finally:
        if own_client:
            c.close()


def series_to_polars(points: Sequence[BcbSgsPoint], value_column: str) -> pl.DataFrame:
    """Converte pontos SGS em :class:`polars.DataFrame` longo."""
    return pl.DataFrame(
        {
            "date": [p.ref_date for p in points],
            value_column: [p.value for p in points],
        }
    )


def save_raw_json(code: int, points: Sequence[BcbSgsPoint], path: Path) -> None:
    """Persiste cópia bruta JSON-compatível (auditoria / reprodutibilidade)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [{"data": p.ref_date.isoformat(), "valor": p.value} for p in points]
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug("Série {} salva em {}.", code, path)


def default_raw_path(code: int) -> Path:
    """Caminho padrão para cache bruto no diretório configurado."""
    settings = get_settings()
    settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_raw_dir / f"bcb_sgs_{code}.json"
