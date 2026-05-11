"""Modelos bayesianos (BVAR, hierárquicos) — extensão com numpyro/pymc."""

from macroeconomia.models.bayesian.gaussian_mean_numpyro import (
    sample_posterior_mean_known_sigma,
)

__all__ = ["sample_posterior_mean_known_sigma"]
