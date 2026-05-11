"""Orquestração Streamlit — lógica de dados em ``data_loader`` e gráficos em ``charts``.

Execução na raiz do repositório::

    streamlit run dashboard/streamlit_app.py

Variáveis úteis: ``MACRO_DEMO_MODE``, ``MACRO_GIT_COMMIT`` (CI/build).

TTL do ``st.cache_data``: 12 horas (mesmo valor padrão de ``MACRO_DASHBOARD_CACHE_TTL_SECONDS``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import streamlit as st

from dashboard.charts import macro_line_chart_figure
from dashboard.data_loader import (
    PanelResolutionError,
    PanelValidationError,
    load_manifest_for_parent,
    parquet_last_modified,
    resolve_panel_parquet_path,
)

_REPO_ROOT_MARKER = Path(__file__).resolve().parents[1]

_CACHE_TTL = timedelta(seconds=43200)


@st.cache_data(ttl=_CACHE_TTL, show_spinner="Carregando painel…", max_entries=8)
def load_dashboard_polars(parquet_abs: str) -> object:
    import polars as pl

    from dashboard.data_loader import validate_panel_contract

    p = Path(parquet_abs).resolve()
    df = validate_panel_contract(pl.read_parquet(p))
    return df


def sidebar_meta(manifest_dict: dict | None, parquet_path: Path) -> None:
    iso = None
    if manifest_dict:
        iso = manifest_dict.get("generated_at")
    iso = iso or parquet_last_modified(parquet_path)

    st.sidebar.caption("Operação")
    st.sidebar.markdown(f"**Última atualização (dados):**  \n{iso or '_desconhecida_'}")
    if manifest_dict and manifest_dict.get("dataset_sha256"):
        sha = str(manifest_dict["dataset_sha256"])
        suf = "" if len(sha) <= 16 else "…"
        st.sidebar.caption(f"SHA256 dataset: `{sha[:16]}{suf}`")
    git_sha = (
        os.environ.get("MACRO_GIT_COMMIT")
        or os.environ.get("GITHUB_SHA")
        or os.environ.get("SOURCE_VERSION")
    )
    if git_sha:
        git_sha = git_sha[:40]
    st.sidebar.markdown(f"**Git commit:**  \n`{git_sha or 'local/unset'}`")


def run_pipeline_blocking() -> tuple[int, str]:
    root = _REPO_ROOT_MARKER
    env = dict(os.environ)
    src = root / "src"
    sep = os.pathsep
    env["PYTHONPATH"] = f"{src}{sep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(src)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "macroeconomia.examples.var_inflation_pipeline"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        tail_stderr = "\n".join((proc.stderr or "").splitlines()[-22:])
        tail_stdout = "\n".join((proc.stdout or "").splitlines()[-22:])
        msg = tail_stderr + ("\n---\n" if tail_stderr and tail_stdout else "") + tail_stdout
        return proc.returncode, (msg.strip() or "(pipeline sem texto de saida)")
    except subprocess.TimeoutExpired:
        return 124, "Timeout ao executar o pipeline (>10 min)."
    except OSError as exc:
        return 1, f"Erro ao lançar o pipeline: {exc}"


def main() -> None:
    from macroeconomia.config import get_settings

    _settings = get_settings()
    diagnostic = st.sidebar.checkbox("Diagnóstico técnico", value=False)
    _ = _settings.dashboard_cache_ttl_seconds

    st.set_page_config(page_title="macroeconomia — diagnóstico diário", layout="wide")
    st.title("Indicadores macro — painel diário")

    refresh = st.sidebar.button(
        "Atualizar dados (pipeline VAR IPCA × Selic)",
        help="Roda ``python -m macroeconomia.examples.var_inflation_pipeline`` (precisa rede BCB).",
    )

    if refresh:
        with st.sidebar.status("Rodando VAR + MLflow…", expanded=diagnostic) as slot:
            rc, msg_full = run_pipeline_blocking()
            if diagnostic:
                slot.write("(Detalhes abaixo após rerun.)")
            st.session_state["_pipe_rc"] = rc
            st.session_state["_pipe_msg"] = msg_full[:6800]

        load_dashboard_polars.clear()
        st.rerun()

    fb_rc = st.session_state.pop("_pipe_rc", None)
    fb_msg = st.session_state.pop("_pipe_msg", None)
    if fb_rc is not None:
        if fb_rc == 0:
            st.sidebar.success("Pipeline concluído")
            st.success("Dados atualizados pelo pipeline VAR (IPCA × Selic).")
        else:
            st.sidebar.error(f"Pipeline saiu com codigo {fb_rc}")
            st.error(_cap(f"A atualização falhou (codigo {fb_rc}). " + ("Veja lateral.")))

        if diagnostic and fb_msg:
            with st.expander("Últimas linhas stderr/stdout"):
                st.code(fb_msg)
        elif fb_rc != 0 and fb_msg and not diagnostic:
            st.info(
                "Ative «Diagnóstico técnico» na lateral e rode novamente para ver stderr/stdout."
            )

    try:
        meta = resolve_panel_parquet_path(_settings)
    except PanelResolutionError as exc:
        st.sidebar.error("Sem dados válidos")
        st.error("**Sem dados disponíveis.**\n\n" + str(exc))
        st.stop()

    parquet_resolved = meta.parquet_path.resolve()
    manifest = load_manifest_for_parent(parquet_resolved)
    sidebar_meta(manifest, parquet_resolved)

    if diagnostic and manifest:
        with st.sidebar.expander("daily_manifest"):
            st.json(manifest)

    src_labels = {
        "demo_forced": "Demonstração",
        "processed": "Processado (ETL)",
        "demo_fallback": "Demonstração (fallback — sem processado)",
    }
    tag = src_labels.get(meta.source, meta.source)
    if meta.source != "processed":
        st.warning(
            f"**Origem:** {tag}. Para atualizar pela API do BCB, rode o botão lateral "
            "`Atualizar dados` (rede)."
        )

    try:
        df_any = load_dashboard_polars(str(parquet_resolved))
        df = df_any
        cols = [c for c in df.columns if c != "date"]
        tab_prev, tab_charts = st.tabs(["Planilha", "Séries"])
        with tab_prev:
            st.dataframe(df.tail(480).to_pandas(), use_container_width=True)
        with tab_charts:
            if cols:
                default_i = min(1, len(cols) - 1) if len(cols) > 1 else 0
                choice = st.selectbox("Variável", cols, index=default_i)
                fig = macro_line_chart_figure(df, choice, title=f"{choice} × tempo")
                st.plotly_chart(fig, use_container_width=True)

    except PanelValidationError as exc:
        st.error(
            "**Dados fora do contrato Pandera (ETL oficial).**\n\n"
            f"{_cap(str(exc), 780)} "
            "Execute `macro-var-example` ou gere `data/demo` com scripts/gen_demo_panel.py."
        )
        if diagnostic:
            st.code(traceback.format_exc())
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error("**Erro inesperado ao ler o parquet.**")
        if diagnostic:
            st.code(traceback.format_exc())
        else:
            st.code(_cap(str(exc), 600))


def _cap(s: str, n: int = 480) -> str:
    x = (s or "").strip()
    return x[:n] + (" …" if len(x) > n else "")


if __name__ == "__main__":
    main()
