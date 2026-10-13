"""Latin Hypercube Sampling (LHS).

Why LHS instead of plain Monte Carlo: plain random sampling clumps. LHS forces
every slice of every input to be visited exactly once, so 1,000 runs cover the
cost-volatility space far more evenly than 10,000 random runs would. On a
3-5 dimensional problem it typically converges several times faster.

Correlation handling uses the Iman-Conover rank transform, which imposes a
target rank correlation without disturbing the marginals or the stratification.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from . import distributions as dist


def _lhs_uniform(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    u = np.empty((n, k))
    for j in range(k):
        perm = rng.permutation(n)
        u[:, j] = (perm + rng.random(n)) / n
    return u


def _iman_conover(u: np.ndarray, corr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reorder columns of `u` so their rank correlation matches `corr`."""
    n, k = u.shape
    # normal scores of the ranks (van der Waerden)
    ranks = np.argsort(np.argsort(u, axis=0), axis=0)
    scores = dist.inv_norm_cdf((ranks + 0.5) / n)

    w, V = np.linalg.eigh(corr)
    w = np.clip(w, 1e-8, None)
    corr = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        return u
    S = scores @ L.T

    out = np.empty_like(u)
    for j in range(k):
        target = np.argsort(np.argsort(S[:, j]))
        out[:, j] = np.sort(u[:, j])[target]
    return out


def sample(
    n: int,
    dims: Sequence[dict],
    rng: np.random.Generator,
    correlations: Iterable[tuple[str, str, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Stratified sample. `dims` entries are distribution specs with a `name`."""
    dims = [dist.normalise_spec(d) for d in dims]
    names = [d["name"] for d in dims]
    u = _lhs_uniform(n, len(dims), rng)

    corr = list(correlations or [])
    if corr:
        k = len(names)
        idx = {nm: i for i, nm in enumerate(names)}
        R = np.eye(k)
        for a, b, rho in corr:
            if a in idx and b in idx:
                R[idx[a], idx[b]] = R[idx[b], idx[a]] = float(np.clip(rho, -0.99, 0.99))
        u = _iman_conover(u, R, rng)

    out: dict[str, np.ndarray] = {}
    for j, d in enumerate(dims):
        out[d["name"]] = dist.transform(d, u[:, j], rng)
    return out


def strata_coverage(u: np.ndarray, bins: int = 10) -> float:
    """Diagnostic: fraction of the 1-D grid actually visited (1.0 = perfect)."""
    n, k = u.shape
    if n == 0:
        return 0.0
    idx = np.clip((u * bins).astype(int), 0, bins - 1)
    covered = sum(len(np.unique(idx[:, j])) for j in range(k))
    return covered / (k * min(bins, n))
