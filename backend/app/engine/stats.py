"""Summary statistics every use case needs, computed the same way everywhere."""
from __future__ import annotations

import numpy as np


def _as(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def summarize(x, probs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    a = _as(x)
    if a.size == 0:
        return {"n": 0}
    q = {f"p{int(p * 100)}": float(v) for p, v in zip(probs, np.quantile(a, probs))}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        **q,
    }


def mean(x) -> float:
    a = _as(x)
    return float(a.mean()) if a.size else 0.0


def prob_below(x, threshold) -> float:
    a = _as(x)
    return float((a < threshold).mean()) if a.size else 0.0


def prob_above(x, threshold) -> float:
    a = _as(x)
    return float((a > threshold).mean()) if a.size else 0.0


def cvar(x, alpha: float = 0.05) -> float:
    """Average of the worst `alpha` fraction of outcomes (tail risk)."""
    a = _as(x)
    if a.size == 0:
        return 0.0
    k = max(int(np.floor(a.size * alpha)), 1)
    return float(np.sort(a)[:k].mean())


def cvar_upper(x, alpha: float = 0.05) -> float:
    a = _as(x)
    if a.size == 0:
        return 0.0
    k = max(int(np.floor(a.size * alpha)), 1)
    return float(np.sort(a)[-k:].mean())


def histogram(x, bins: int = 40):
    a = _as(x)
    if a.size == 0:
        return [], []
    counts, edges = np.histogram(a, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return counts.tolist(), centres.tolist()


def correlation(x, y) -> float:
    a, b = _as(x), _as(y)
    if a.size < 2 or a.size != b.size:
        return 0.0
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def lift(p_event_given_condition: float, p_event: float) -> float:
    """How much more likely the event is under the condition (1.0 = no effect)."""
    if p_event <= 0:
        return 0.0
    return p_event_given_condition / p_event


def moving_average(x, window: int = 4):
    a = _as(x)
    if a.size == 0:
        return []
    w = max(1, min(window, a.size))
    cum = np.cumsum(np.insert(a, 0, 0.0))
    out = (cum[w:] - cum[:-w]) / w
    return out.tolist()


def weighted_choice(values, weights, rng, n: int = 1):
    v = np.asarray(values)
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    p = w / w.sum()
    return rng.choice(v, size=n, p=p)
