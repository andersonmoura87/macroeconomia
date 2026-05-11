"""Bayes mínimo (numpyro) — extra ``[bayesian]``; omitido na cobertura global."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("numpyro")

from macroeconomia.models.bayesian.gaussian_mean_numpyro import (
    sample_posterior_mean_known_sigma,
)


def test_sample_posterior_mean_known_sigma_smoke() -> None:
    rng = np.random.default_rng(0)
    true_mu = 2.5
    y = rng.normal(true_mu, 1.0, size=30)
    out = sample_posterior_mean_known_sigma(
        y,
        sigma=1.0,
        num_warmup=200,
        num_samples=300,
        rng_seed=1,
    )
    assert out["mu_samples"].shape[0] == 300
    assert abs(float(np.mean(out["mu_samples"])) - true_mu) < 1.0
