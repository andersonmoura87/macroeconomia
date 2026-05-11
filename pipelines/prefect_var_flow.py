"""Fluxo Prefect que encapsula o exemplo VAR (pode ser agendado no Prefect Server).

Não logue variáveis de ambiente completas nem chaves de API em ``print``/tasks:
use segredos Prefect ou mounts de volume com ``.env`` fora dos logs.
"""

from __future__ import annotations

from prefect import flow, task

from macroeconomia.examples.var_inflation_pipeline import main as run_var_pipeline


@task(name="run-var-pipeline", retries=1, retry_delay_seconds=30)
def run_pipeline_task() -> None:
    """Executa o pipeline VAR (idempotente em termos de artefatos)."""
    run_var_pipeline()


@flow(name="macro-var-ipca-selic")
def macro_var_flow() -> None:
    """Flow principal: política monetária × inflação (nível e VAR estacionário)."""
    run_pipeline_task()


if __name__ == "__main__":
    macro_var_flow()
