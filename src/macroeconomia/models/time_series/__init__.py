"""Modelos de séries temporais (VAR, VECM, ARIMA)."""

from macroeconomia.models.time_series.arima_model import fit_arima_grid
from macroeconomia.models.time_series.arima_pmdarima import fit_arima_auto_pmdarima
from macroeconomia.models.time_series.local_level import fit_local_level
from macroeconomia.models.time_series.var_diagnostics import (
    fevd_summary_text,
    granger_causality_markdown,
    irf_to_long_frame,
    var_is_stable,
    var_max_root_modulus,
)
from macroeconomia.models.time_series.var_model import fit_var, polars_endog_matrix, var_aic_table
from macroeconomia.models.time_series.var_oos import (
    temporal_train_test_split,
    var_multistep_oos_metrics,
)

__all__ = [
    "fevd_summary_text",
    "fit_arima_auto_pmdarima",
    "fit_arima_grid",
    "fit_local_level",
    "fit_var",
    "granger_causality_markdown",
    "irf_to_long_frame",
    "polars_endog_matrix",
    "temporal_train_test_split",
    "var_aic_table",
    "var_is_stable",
    "var_max_root_modulus",
    "var_multistep_oos_metrics",
]
