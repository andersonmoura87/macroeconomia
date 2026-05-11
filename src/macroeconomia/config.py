"""Configuração centralizada via variáveis de ambiente (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Parâmetros globais do projeto (prefixo de env: ``MACRO_``).

    Attributes:
        seed: Semente para RNG (numpy, statsmodels, etc.).
        log_level: Nível mínimo de log.
        data_raw_dir: Diretório de dados brutos (relativo à raiz do repositório).
        data_processed_dir: Diretório de dados processados.
        data_external_dir: Dados externos auxiliares.
        artifacts_dir: Artefatos (figuras, relatórios exportados).
        mlflow_tracking_uri: URI do servidor MLflow.
        mlflow_experiment_name: Nome do experimento MLflow.
        mlflow_registered_model_name: Nome no Model Registry (VAR); vazio desativa registro.
        mlflow_arima_registered_model_name: Nome Registry ARIMA; vazio só artefactos.
        demo_mode: Força uso dos Parquets de demonstração no dashboard.
        dashboard_parquet_name: Nome do arquivo lido pelo dashboard (diretório processado).
        demo_parquet_relative: Caminho demo (relativo à raiz).
        dashboard_cache_ttl_seconds: TTL do ``st.cache_data`` (padrão 12 h).
    """

    model_config = SettingsConfigDict(
        env_prefix="MACRO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    seed: int = Field(default=42, ge=0, description="Semente para reprodutibilidade.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    data_external_dir: Path = Path("data/external")
    artifacts_dir: Path = Path("data/processed/artifacts")
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "macroeconomia"
    mlflow_registered_model_name: str = Field(
        default="macroeconomia_var_ipca_selic_stationary",
        description="Registry MLflow (VAR); string vazia desativa register_model.",
    )
    mlflow_arima_registered_model_name: str = Field(
        default="",
        description="Registry MLflow (ARIMA exemplo); vazio só loga pyfunc/metadata sem registar.",
    )
    demo_mode: bool = False
    dashboard_parquet_name: str = "macro_ipca_selic.parquet"
    demo_parquet_relative: Path = Path("data/demo/macro_ipca_selic.parquet")
    dashboard_cache_ttl_seconds: int = Field(default=43200, ge=60)

    @model_validator(mode="after")
    def _normalize_paths(self) -> Settings:
        """Resolve caminhos relativos a partir da raiz do repositório."""

        def resolve(p: Path) -> Path:
            return p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve()

        object.__setattr__(self, "data_raw_dir", resolve(self.data_raw_dir))
        object.__setattr__(self, "data_processed_dir", resolve(self.data_processed_dir))
        object.__setattr__(self, "data_external_dir", resolve(self.data_external_dir))
        object.__setattr__(self, "artifacts_dir", resolve(self.artifacts_dir))
        object.__setattr__(
            self,
            "demo_parquet_relative",
            resolve(self.demo_parquet_relative),
        )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def repo_root(self) -> Path:
        """Raiz do repositório (detectada a partir do pacote)."""
        return _REPO_ROOT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna instância única de :class:`Settings` (cacheada)."""
    return Settings()
