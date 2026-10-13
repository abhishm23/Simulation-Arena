"""Monte Carlo engine.

Classic random sampling with an optional Gaussian-copula correlation structure,
so "discount depth" and "win probability" can be made genuinely dependent
rather than drifting independently.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

from . import distributions as dist
from . import stats as st


def _corr_matrix(names: Sequence[str], correlations: Iterable[tuple[str, str, float]]) -> np.ndarray:
    k = len(names)
    R = np.eye(k)
    idx = {n: i for i, n in enumerate(names)}
    for a, b, rho in correlations or []:
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            rho = float(np.clip(rho, -0.99, 0.99))
            R[i, j] = R[j, i] = rho
    # nearest PSD via eigen-clipping (robust against user-entered matrices)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-8, None)
    R = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def draw(
    n: int,
    specs: Sequence[dict],
    rng: np.random.Generator,
    correlations: Iterable[tuple[str, str, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Return {name: array(n,)} sampled jointly from `specs`.

    `correlations` is an iterable of (name_a, name_b, rho).
    """
    specs = [dist.normalise_spec(s) for s in specs]
    names = [s["name"] for s in specs]
    corr = list(correlations or [])

    if corr:
        R = _corr_matrix(names, corr)
        L = np.linalg.cholesky(R)
        z = rng.standard_normal((n, len(names))) @ L.T
        u = dist.norm_cdf(z)
    else:
        u = rng.random((n, len(names)))

    out: dict[str, np.ndarray] = {}
    for j, s in enumerate(specs):
        out[s["name"]] = dist.transform(s, u[:, j], rng)
    return out


def evaluate(draws: dict[str, np.ndarray], fn: Callable[..., np.ndarray]) -> np.ndarray:
    """Apply a vectorised payoff/cost function across all draws."""
    return np.asarray(fn(**draws), dtype=float)


def run(
    n: int,
    specs: Sequence[dict],
    payoff: Callable[..., np.ndarray],
    rng: np.random.Generator,
    correlations: Iterable[tuple[str, str, float]] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    d = draw(n, specs, rng, correlations)
    return d, evaluate(d, payoff)


def summarize_output(y: np.ndarray, probs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    s = st.summarize(y, probs)
    s["p_loss"] = st.prob_below(y, 0.0)
    s["cvar_5"] = st.cvar(y, 0.05)
    return s
