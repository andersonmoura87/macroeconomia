"""Gera ``data/demo/macro_ipca_selic.parquet`` coerente com o esquema Pandera do projeto.

Uso::

    PYTHONPATH=src python scripts/gen_demo_panel.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(_REPO / "src"))

    import numpy as np
    import polars as pl

    from macroeconomia.config import get_settings
    from macroeconomia.processing.macro_panel import validate_macro_bivariate

    rng = np.random.default_rng(7)
    n = 60
    dates = pl.date_range(
        date(2019, 1, 1),
        date(2024, 12, 1),
        interval="1mo",
        eager=True,
    )[:n]
    base_ipca = 100.0 + np.cumsum(rng.standard_normal(n) * 0.35)
    selic = np.clip(0.5 + rng.standard_normal(n) * 0.15 + np.linspace(0, 0.4, num=n), 0.1, 2.5)

    df = pl.DataFrame(
        {
            "date": dates,
            "ipca_index": base_ipca,
            "selic_pct": selic,
        }
    )
    panel = validate_macro_bivariate(df)
    root = get_settings().repo_root
    demo_dir = root / "data" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    out = demo_dir / "macro_ipca_selic.parquet"
    panel.write_parquet(out)
    print(f"Escrito {out.relative_to(root)}")


if __name__ == "__main__":
    main()
