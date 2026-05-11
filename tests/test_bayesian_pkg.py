"""Metadados do subpacote bayesian (sem numpyro no runner padrão)."""

from __future__ import annotations


def test_bayesian_subpackage_exports() -> None:
    import macroeconomia.models.bayesian as bay

    assert hasattr(bay, "sample_posterior_mean_known_sigma")
    assert "sample_posterior_mean_known_sigma" in bay.__all__
