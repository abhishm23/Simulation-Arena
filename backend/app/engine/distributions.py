"""Distribution layer shared by Monte Carlo and Latin Hypercube sampling.

Deliberately dependency-free (no SciPy): every distribution is specified as a
plain dict, e.g. {"name": "cloud_cost", "dist": "normal", "mean": 1.0, "sd": 0.1}.

Two capabilities matter:
  * sample()   - plain random draws (Monte Carlo)
  * inv_cdf()  - inverse transform (required by Latin Hypercube stratification)

For exotic distributions without a closed-form inverse CDF we fall back to
rank-matched sampling, which preserves the stratification property of LHS.
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Normal CDF / inverse CDF (Acklam's rational approximation, ~1e-9 accurate)
# ---------------------------------------------------------------------------
_SQRT2PI = math.sqrt(2.0 * math.pi)

_a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]
_p_low = 0.02425
_p_high = 1.0 - _p_low


def norm_cdf(x):
    """Vectorised standard normal CDF."""
    x = np.asarray(x, dtype=float)
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def inv_norm_cdf(p):
    """Vectorised standard normal inverse CDF (quantile function)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    out = np.empty_like(p)

    low = p < _p_low
    mid = ~(low | (p > _p_high))
    high = ~(low | mid)

    if low.any():
        q = np.sqrt(-2.0 * np.log(p[low]))
        out[low] = (((((_c[0] * q + _c[1]) * q + _c[2]) * q + _c[3]) * q + _c[4]) * q + _c[5]) / \
                   ((((_d[0] * q + _d[1]) * q + _d[2]) * q + _d[3]) * q + 1.0)
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (((((_a[0] * r + _a[1]) * r + _a[2]) * r + _a[3]) * r + _a[4]) * r + _a[5]) * q / \
                   (((((_b[0] * r + _b[1]) * r + _b[2]) * r + _b[3]) * r + _b[4]) * r + 1.0)
    if high.any():
        q = np.sqrt(-2.0 * np.log(1.0 - p[high]))
        out[high] = -(((((_c[0] * q + _c[1]) * q + _c[2]) * q + _c[3]) * q + _c[4]) * q + _c[5]) / \
                    ((((_d[0] * q + _d[1]) * q + _d[2]) * q + _d[3]) * q + 1.0)
    return out


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------
def _get(spec: dict, key: str, default=None):
    return spec.get(key, default)


def normalise_spec(spec: dict) -> dict:
    """Accept friendly aliases (sd/stdev/sigma, low/min, high/max, mode/most_likely)."""
    if not isinstance(spec, dict):
        return {"name": "x", "dist": "constant", "value": float(spec)}
    s = dict(spec)
    s.setdefault("name", "x")
    s.setdefault("dist", "constant")
    s.setdefault("value", 0.0)

    if "sd" not in s:
        for alias in ("stdev", "sigma", "std", "stddev"):
            if alias in s:
                s["sd"] = s[alias]
                break
    # Bounds aliases only apply to bounded distributions - otherwise "a"/"b"
    # would be stolen from beta(a, b) and turn into clip bounds.
    if s["dist"] in ("uniform", "triangular", "pert"):
        if "low" not in s:
            for alias in ("min", "a", "lower"):
                if alias in s:
                    s["low"] = s[alias]
                    break
        if "high" not in s:
            for alias in ("max", "b", "upper"):
                if alias in s:
                    s["high"] = s[alias]
                    break
    if "mode" not in s:
        for alias in ("most_likely", "ml", "peak"):
            if alias in s:
                s["mode"] = s[alias]
                break
    return s


def _clip_to_bounds(x: np.ndarray, spec: dict) -> np.ndarray:
    lo, hi = spec.get("low"), spec.get("high")
    if lo is not None or hi is not None:
        x = np.clip(x, -np.inf if lo is None else lo, np.inf if hi is None else hi)
    return x


# ---------------------------------------------------------------------------
# Inverse CDF (returns None when no closed form -> caller rank-matches)
# ---------------------------------------------------------------------------
def inv_cdf(spec: dict, u: np.ndarray) -> np.ndarray | None:
    s = normalise_spec(spec)
    d = s["dist"]
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)

    if d in ("constant", "fixed"):
        return np.full_like(u, float(s.get("value", 0.0)))
    if d == "uniform":
        return s.get("low", 0.0) + (s.get("high", 1.0) - s.get("low", 0.0)) * u
    if d == "normal":
        return s.get("mean", 0.0) + s.get("sd", 1.0) * inv_norm_cdf(u)
    if d in ("lognormal", "log_normal"):
        return float(s.get("shift", 0.0)) + np.exp(s.get("mean", 0.0) + s.get("sd", 1.0) * inv_norm_cdf(u))
    if d == "triangular":
        lo, hi = float(s.get("low", 0.0)), float(s.get("high", 1.0))
        mode = s.get("mode")
        mode = (lo + hi) / 2.0 if mode is None else float(mode)
        fc = (mode - lo) / max(hi - lo, 1e-12)
        out = np.where(u < fc,
                       lo + np.sqrt(np.maximum(u * (hi - lo) * (mode - lo), 0.0)),
                       hi - np.sqrt(np.maximum((1.0 - u) * (hi - lo) * (hi - mode), 0.0)))
        return out
    if d == "exponential":
        return -np.log(1.0 - u) / max(float(s.get("rate", s.get("lambda", 1.0))), 1e-12)
    if d == "weibull":
        k = max(float(s.get("shape", 1.5)), 1e-9)
        lam = max(float(s.get("scale", 1.0)), 1e-9)
        return lam * (-np.log(np.clip(1.0 - u, 1e-12, 1.0))) ** (1.0 / k)
    if d == "empirical":
        vals = np.asarray(s.get("values", [0.0]), dtype=float)
        if vals.size == 0:
            return np.zeros_like(u)
        vs = np.sort(vals)
        idx = np.clip((u * vs.size).astype(int), 0, vs.size - 1)
        return vs[idx]
    if d == "pert":
        lo, hi = float(s.get("low", 0.0)), float(s.get("high", 1.0))
        mode = float(s.get("mode", (lo + hi) / 2.0))
        lam = float(s.get("lambda", 4.0))
        rng_ = hi - lo
        if rng_ <= 0:
            return np.full_like(u, lo)
        mu = (lo + lam * mode + hi) / (lam + 2.0)
        a = 1.0 + lam * (mu - lo) / rng_
        b = 1.0 + lam * (hi - mu) / rng_
        # PERT has no closed-form inverse; rank-match via beta approximation below
        return None if a <= 0 or b <= 0 else _beta_ppf_approx(u, a, b, lo, hi)
    return None  # beta, gamma, poisson, bernoulli -> rank-matched by caller


def _beta_ppf_approx(u, a, b, lo, hi):
    """Bisection inverse of the Beta CDF via the regularised incomplete beta
    function computed with a continued fraction (Lentz). Accurate enough for
    business simulation and free of external dependencies."""
    out = np.empty_like(u)
    for i, ui in enumerate(u):
        x_lo, x_hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (x_lo + x_hi)
            if _betainc(mid, a, b) < ui:
                x_lo = mid
            else:
                x_hi = mid
        out[i] = lo + (hi - lo) * 0.5 * (x_lo + x_hi)
    return out


def _betacf(a, b, x):
    tiny, eps, maxit = 1e-30, 3e-12, 200
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betainc(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log(1.0 - x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


# ---------------------------------------------------------------------------
# Direct sampling (used by Monte Carlo and as the rank-match source)
# ---------------------------------------------------------------------------
def sample(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    s = normalise_spec(spec)
    d = s["dist"]

    if d in ("constant", "fixed"):
        return np.full(n, float(s.get("value", 0.0)))
    if d == "uniform":
        return rng.uniform(s.get("low", 0.0), s.get("high", 1.0), n)
    if d == "normal":
        return rng.normal(s.get("mean", 0.0), s.get("sd", 1.0), n)
    if d in ("lognormal", "log_normal"):
        return float(s.get("shift", 0.0)) + rng.lognormal(s.get("mean", 0.0), s.get("sd", 1.0), n)
    if d == "triangular":
        return rng.triangular(s.get("low", 0.0), s.get("mode", 0.5), s.get("high", 1.0), n)
    if d == "exponential":
        return rng.exponential(1.0 / max(float(s.get("rate", s.get("lambda", 1.0))), 1e-12), n)
    if d == "weibull":
        return s.get("scale", 1.0) * rng.weibull(max(float(s.get("shape", 1.5)), 1e-9), n)
    if d == "beta":
        return rng.beta(max(float(s.get("a", 2.0)), 1e-6), max(float(s.get("b", 2.0)), 1e-6), n)
    if d == "gamma":
        return rng.gamma(max(float(s.get("shape", 2.0)), 1e-6), max(float(s.get("scale", 1.0)), 1e-9), n)
    if d == "poisson":
        return rng.poisson(max(float(s.get("mu", 1.0)), 0.0), n).astype(float)
    if d == "bernoulli":
        return (rng.random(n) < float(s.get("p", 0.5))).astype(float)
    if d == "empirical":
        vals = np.asarray(s.get("values", [0.0]), dtype=float)
        if vals.size == 0:
            return np.zeros(n)
        return rng.choice(vals, size=n, replace=True)
    if d == "pert":
        lo, hi = float(s.get("low", 0.0)), float(s.get("high", 1.0))
        mode = float(s.get("mode", (lo + hi) / 2.0))
        lam = float(s.get("lambda", 4.0))
        rng_ = hi - lo
        if rng_ <= 0:
            return np.full(n, lo)
        mu = (lo + lam * mode + hi) / (lam + 2.0)
        a = max(1.0 + lam * (mu - lo) / rng_, 1e-6)
        b = max(1.0 + lam * (hi - mu) / rng_, 1e-6)
        return lo + rng_ * rng.beta(a, b, n)
    raise ValueError(f"Unknown distribution: {d}")


def transform(spec: dict, u: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Uniform(0,1) -> distribution space. Rank-matches when no inverse CDF."""
    vals = inv_cdf(spec, u)
    if vals is None:
        n = len(u)
        pool = sample(spec, n, rng if rng is not None else np.random.default_rng(0))
        order = np.argsort(np.argsort(u))
        vals = np.sort(pool)[order]
    return _clip_to_bounds(vals, normalise_spec(spec))


def describe(spec: dict) -> str:
    """Plain-English description for the report's assumptions appendix."""
    s = normalise_spec(spec)
    d = s["dist"]
    if d in ("constant", "fixed"):
        return f"fixed at {s.get('value', 0):g}"
    if d == "uniform":
        return f"anywhere between {s.get('low', 0):g} and {s.get('high', 1):g} (all values equally likely)"
    if d == "normal":
        return f"bell curve centred on {s.get('mean', 0):g} with spread {s.get('sd', 1):g}"
    if d in ("lognormal", "log_normal"):
        return f"skewed-right curve around {math.exp(s.get('mean', 0)):g}"
    if d == "triangular":
        return f"between {s.get('low', 0):g} and {s.get('high', 1):g}, most likely {s.get('mode', 0.5):g}"
    if d == "pert":
        return f"between {s.get('low', 0):g} and {s.get('high', 1):g}, most likely {s.get('mode', 0):g}"
    if d == "empirical":
        return f"drawn from {len(s.get('values', []))} observed historical values"
    return f"{d} distribution"
