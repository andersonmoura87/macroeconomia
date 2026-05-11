"""Painel macro bivariado (ex.: inflação × juros) com validação Pandera + Polars."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandera.polars as pa
import polars as pl
from loguru import logger

from macroeconomia.ingestion.bcb_sgs import BcbSgsPoint, series_to_polars


class MacroBivariateSchema(pa.DataFrameModel):
    """Esquema declarativo para painel com IPCA (nível) e Selic.

    Unidades esperadas (exemplo padrão SGS):

    * ``ipca_index``: número-índice do IPCA (nível, base fixa do SGS).
    * ``selic_pct``: taxa Selic em **percentual ao mês** (ex. série ``432``).
    """

    date: pl.Date = pa.Field(nullable=False, coerce=True)
    ipca_index: pl.Float64 = pa.Field(nullable=False)
    selic_pct: pl.Float64 = pa.Field(nullable=False)

    class Config:
        """Configuração Pandera."""

        strict = True
        coerce = True


def audit_bivariate_macro_panel(df: pl.DataFrame) -> None:
    """Checagens de qualidade **antes** do esquema Pandera (mensagens acionáveis).

    Política: **sem imputação automática** de buracos; nulos ou datas duplicadas
    após o *inner join* indicam desalinhamento ou falhas na origem.

    Raises:
        ValueError: Painel vazio, datas duplicadas ou valores nulos em colunas núcleo.
    """
    if df.height == 0:
        raise ValueError(
            "Painel vazio após inner join entre IPCA e Selic. "
            "Verifique sobreposição de datas nas séries SGS ou códigos incorretos."
        )
    n_unique = int(df.select(pl.col("date").n_unique()).item())
    if n_unique != df.height:
        raise ValueError(
            "Painel com `date` duplicada após inner join. "
            "Esperado exatamente uma linha por data. Agregue na origem ou remova duplicatas."
        )
    for col in ("ipca_index", "selic_pct"):
        nulls = int(df.select(pl.col(col).null_count()).item())
        if nulls:
            raise ValueError(
                f"A coluna `{col}` contém {nulls} valor(es) nulo(s) após o join. "
                "Não imputamos automaticamente: revise a série no SGS ou o alinhamento."
            )


def build_macro_bivariate_panel(
    ipca_points: Sequence[BcbSgsPoint],
    selic_points: Sequence[BcbSgsPoint],
) -> pl.DataFrame:
    """Alinha duas séries SGS por data (inner join) e renomeia colunas.

    Args:
        ipca_points: Série IPCA em nível índice (ex.: código SGS ``433``).
        selic_points: Série Selic em % a.m. (ex.: código SGS ``432``).

    Returns:
        DataFrame Polars com colunas ``date``, ``ipca_index``, ``selic_pct``.
        Uma linha por data comum às duas séries (sem interpolação).
    """
    ipca = series_to_polars(ipca_points, "ipca_index")
    selic = series_to_polars(selic_points, "selic_pct")
    merged = ipca.join(selic, on="date", how="inner").sort("date")
    logger.info("Painel bivariado: {} linhas após inner join.", merged.height)
    return merged


def validate_macro_bivariate(df: pl.DataFrame) -> pl.DataFrame:
    """Audita qualidade e valida ``df`` contra :class:`MacroBivariateSchema`.

    Returns:
        O mesmo DataFrame validado (possivelmente *coerced*).

    Raises:
        ValueError: Falhas de auditoria (painel vazio, duplicatas, nulos).
        pandera.errors.SchemaError: Dados fora do esquema.
    """
    audit_bivariate_macro_panel(df)
    return MacroBivariateSchema.validate(df)


def write_parquet_partition(df: pl.DataFrame, path: Path) -> None:
    """Escreve Parquet (formato analítico padrão do repositório)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.debug("Parquet escrito em {}.", path)
