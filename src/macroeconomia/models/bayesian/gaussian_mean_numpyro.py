"""Inferência bayesiana mínima: média de uma Normal com variância conhecida (``numpyro``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _build_normal_mean_model(sigma: float) -> Callable[..., None]:
    """Constrói o modelo ``obs_i | mu ~ Normal(mu, sigma)`` com prior em ``mu``."""

    def model(obs: NDArray[Any]) -> None:
        import numpyro
        import numpyro.distributions as dist

        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
        with numpyro.plate("i", obs.shape[0]):
            numpyro.sample("y", dist.Normal(mu, float(sigma)), obs=obs)

    return model


def sample_posterior_mean_known_sigma(
    y: NDArray[Any] | list[float],
    sigma: float,
    *,
    num_warmup: int = 500,
    num_samples: int = 1000,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Amostra a posteriori de ``mu`` em ``y_i ~ Normal(mu, sigma)``.

    Requer extra ``[bayesian]`` (``numpyro``, ``jax``).

    Args:
        y: Observações univariadas.
        sigma: Desvio-padrão conhecido.
        num_warmup: Aquecimento do NUTS.
        num_samples: Amostras retidas.
        rng_seed: Semente JAX.

    Returns:
        Dicionário com ``mu_samples`` (``ndarray`` 1D).

    Raises:
        RuntimeError: Dependências bayesianas ausentes.
    """
    try:
        import jax.random as jrandom
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:  # pragma: no cover - extra opcional
        raise RuntimeError(
            "Instale o extra `macroeconomia[bayesian]` (pacotes `numpyro` e `jax`)."
        ) from exc

    arr = np.asarray(y, dtype=float).ravel()
    nuts = NUTS(_build_normal_mean_model(float(sigma)))
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples)
    rng_key = jrandom.PRNGKey(rng_seed)
    mcmc.run(rng_key, obs=arr)
    samples = mcmc.get_samples()["mu"]
    return {
        "mu_samples": np.asarray(samples, dtype=float),
        "num_samples": int(samples.shape[0]),
    }
