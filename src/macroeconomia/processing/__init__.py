"""ETL/ELT de alto desempenho com Polars, DuckDB e validação Pandera."""

from macroeconomia.processing.macro_panel import (
    MacroBivariateSchema,
    audit_bivariate_macro_panel,
    build_macro_bivariate_panel,
    validate_macro_bivariate,
)

__all__ = [
    "MacroBivariateSchema",
    "audit_bivariate_macro_panel",
    "build_macro_bivariate_panel",
    "validate_macro_bivariate",
]
