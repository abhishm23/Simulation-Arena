"""Resampling / empirical bootstrap engine.

Used when the honest answer comes from *your own history* rather than from an
assumed distribution: "resample thousands of past multi-line transactions with
similar customer tier and deal structure".
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from . import stats as st


def resample_indices(n_rows: int, n_draws: int, rng: np.random.Generator,
                     strata: Sequence | None = None) -> np.ndarray:
    """(n_draws, n_rows) index matrix. With `strata`, resample within stratum."""
    if n_rows <= 0:
        return np.zeros((n_draws, 0), dtype=int)
    if strata is None:
        return rng.integers(0, n_rows, size=(n_draws, n_rows))
    strata = np.asarray(strata)
    idx = np.zeros((n_draws, n_rows), dtype=int)
    for s in np.unique(strata):
        pool = np.flatnonzero(strata == s)
        if pool.size == 0:
            continue
        idx[:, pool] = rng.choice(pool, size=(n_draws, pool.size), replace=True)
    return idx


def resample_rows(rows: Sequence, n_draws: int, rng: np.random.Generator,
                  strata: Sequence | None = None) -> list[np.ndarray]:
    idx = resample_indices(len(rows), n_draws, rng, strata)
    arr = np.asarray(rows, dtype=object)
    return [arr[ix] for ix in idx]


def cluster_resample(groups: Sequence, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Cluster (block) bootstrap: resample whole deals, not their line items.

    Correct when line items inside one deal are correlated - which they are.
    """
    uniq, inverse = np.unique(np.asarray(groups), return_inverse=True)
    chosen = rng.choice(len(uniq), size=(n_draws, len(uniq)), replace=True)
    return uniq[chosen]


def bootstrap_statistic(values, statistic: Callable[[np.ndarray], float],
                        n_draws: int, rng: np.random.Generator, alpha: float = 0.05):
    """Returns (point_estimate, (low, high)) percentile CI."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0, (0.0, 0.0)
    point = float(statistic(v))
    draws = np.empty(n_draws)
    for i in range(n_draws):
        draws[i] = statistic(v[rng.integers(0, v.size, v.size)])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, (float(lo), float(hi))


def joint_probability(mask_a: Sequence[bool], mask_b: Sequence[bool]) -> dict:
    """P(A), P(B), P(B|A), P(A&B) and the lift - the cherry-picking core."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    n = a.size
    if n == 0:
        return {"n": 0, "p_a": 0.0, "p_b": 0.0, "p_b_given_a": 0.0,
                "p_a_and_b": 0.0, "lift": 0.0, "phi": 0.0}
    p_a = float(a.mean())
    p_b = float(b.mean())
    p_ab = float((a & b).mean())
    p_bga = float(b[a].mean()) if a.sum() else 0.0
    denom = math.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b)) if 0 < p_a < 1 and 0 < p_b < 1 else 0.0
    phi = (p_ab - p_a * p_b) / denom if denom > 0 else 0.0
    return {
        "n": int(n),
        "p_a": p_a, "p_b": p_b,
        "p_a_and_b": p_ab,
        "p_b_given_a": p_bga,
        "lift": st.lift(p_bga, p_b),
        "phi": float(phi),
    }
