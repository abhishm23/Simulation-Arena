"""Shared helpers for use cases.

IMPORTANT CONTRACT for every use case author
--------------------------------------------
`ctx.rows`   : list of dicts keyed by ROLE NAME (never the raw column name).
               Mapping/coercion/fallbacks already happened upstream.
               May be empty -> the use case must still work from `ctx.params`.
`ctx.params` : dict of resolved parameter values (defaults already merged).
`ctx.rng(*tags)`     : deterministic RNG stream.
`ctx.say(text, ...)` : emit a plain-language log line for the live log panel.

Golden rule: every number you surface must be explainable in one sentence to
someone who has never heard of a distribution.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ..core.contracts import (
    ChartSpec, KPI, LogEvent, ResultTable, RunResult, StoryStep, Recommendation,
)


# ---------------------------------------------------------------------------
# Row access
# ---------------------------------------------------------------------------
def col(rows: Sequence[dict], role: str, default: Any = None) -> list:
    return [r.get(role, default) for r in rows]


def num(rows: Sequence[dict], role: str, default: float = 0.0) -> np.ndarray:
    out = []
    for r in rows:
        v = r.get(role, None)
        if v is None or v == "":
            out.append(float(default))
            continue
        try:
            f = float(v)
            out.append(default if (math.isnan(f) or math.isinf(f)) else f)
        except (TypeError, ValueError):
            out.append(float(default))
    return np.asarray(out, dtype=float)


def cat(rows: Sequence[dict], role: str, default: str = "") -> list[str]:
    out = []
    for r in rows:
        v = r.get(role, None)
        out.append(default if v is None else str(v))
    return out


def first(rows: Sequence[dict], role: str, default: Any = None) -> Any:
    for r in rows:
        v = r.get(role, None)
        if v is not None and v != "":
            return v
    return default


def n_rows(rows) -> int:
    return len(rows)


def has(rows: Sequence[dict], role: str) -> bool:
    """True when the role is actually populated (not just present as a key)."""
    return any(r.get(role) is not None and r.get(role) != "" for r in rows)


def p(keys: dict, key: str, default: Any = None) -> Any:
    v = keys.get(key, None)
    return default if v is None else v


def as_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def as_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def money(v, dp: int = 0, compact: bool = False) -> str:
    v = as_float(v)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if compact and a >= 1_000_000:
        return f"{sign}${a / 1_000_000:,.2f}M"
    if compact and a >= 1_000:
        return f"{sign}${a / 1_000:,.0f}k"
    return f"{sign}${a:,.{dp}f}"


def money_axis(v) -> str:
    a = abs(as_float(v))
    sign = "-" if v < 0 else ""
    if a >= 1_000_000:
        return f"{sign}{a / 1_000_000:.1f}M"
    if a >= 1_000:
        return f"{sign}{a / 1_000:.0f}k"
    return f"{sign}{a:.0f}"


def pct(v, dp: int = 1) -> str:
    """A fraction as a percentage, never rounding up to a certainty we lack.

    QA NEW-12: a deal whose chance of dying was 99.96% printed as "100.0% never
    closes at all" in the same sentence as "$102 you can bank on" - two claims
    that cannot both be true. Anything that is not actually 100% now reads
    ">99.9%", so the reader can see it is close to certain without being told it
    is certain.
    """
    x = min(as_float(v) * 100.0, 100.0)
    step = 10.0 ** -dp
    if x < 100.0 and x > 100.0 - step / 2.0:
        return f">{100.0 - step:.{dp}f}%"
    return f"{x:.{dp}f}%"


def pct_num(v, dp: int = 1) -> str:
    """For values already expressed in percent units (e.g. 18 means 18%)."""
    return f"{as_float(v):.{dp}f}%"


def pct_fine(v) -> str:
    """Percentage with adaptive precision. Takes a fraction (0.0042 -> "0.42%").

    QA UX-17: a column of small probabilities rounded to one decimal read
    "0.0%" in every row while the money column visibly moved, and a second pass
    at two decimals still flattened a column whose chances were all under
    0.01% to the same "<0.01%" in every row. Below one percent we now keep two
    *significant figures*, so a column of small probabilities moves with the
    money it explains. The same "never round up to a certainty" rule as `pct`
    applies at the top.

    Use this for every probability that lands in a table cell, so that no two
    columns of the same table round by different rules. `pct` stays the right
    choice for prose, where one decimal reads naturally.
    """
    x = min(as_float(v) * 100.0, 100.0)
    a = abs(x)
    if a == 0.0:
        return "0.0%"
    if a >= 100.0:
        return "100.0%"
    if a > 99.9:
        return ">99.9%"
    if a >= 1.0:
        return f"{x:.1f}%"
    if a < _PCT_FINE_FLOOR:
        return f"<{_PCT_FINE_FLOOR:g}%" if x > 0 else f">-{_PCT_FINE_FLOOR:g}%"
    # two significant figures: 0.42% -> "0.42%", 0.0019% -> "0.0019%"
    exp = int(math.floor(math.log10(a)))
    dp = min(6, max(2, 1 - exp))
    text = f"{x:.{dp}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{text}%"


#: QA UX-17: below this a probability is too small to print honestly, so we say
#: "less than" rather than invent precision. Expressed in percent units.
_PCT_FINE_FLOOR = 0.0001


def num_fmt(v, dp: int = 0) -> str:
    return f"{as_float(v):,.{dp}f}"


def count_noun(n, noun: str, dp: int = 0) -> str:
    """A number and its noun, agreeing: "1 quarter", "11 quarters".

    QA NEW-13: the age table read "This deal (1 quarters)". The old check asked
    whether the *raw* value was 1, but the value is rounded before it is
    printed - 1.1 quarters printed as "1" and still took the plural.
    """
    value = as_float(n)
    text = f"{value:,.{dp}f}"
    # judge the plural from the number the reader will actually see
    shown = round(value, dp)
    singular = abs(shown - 1.0) < 1e-9
    return f"{text} {noun if singular else noun + 's'}"


def r2(v) -> float:
    return round(float(v), 2)


def r4(v) -> float:
    return round(float(v), 4)


# ---------------------------------------------------------------------------
# Reading a percentage column that could be written either way
# ---------------------------------------------------------------------------
#: above this, a column of percentages is written as 40 rather than 0.40
_PCT_UNIT_CUTOFF = 1.0


def pct_unit(values) -> str:
    """Is this column written as '40' (percent) or '0.40' (fraction)?

    Decided ONCE for the whole column from its typical magnitude, never value
    by value. The old per-value test - `v / 100 if v > 1 else v` - reads a
    NEGATIVE margin of -8 as -800% and a small positive 0.5 as 50%, so a
    portfolio whose real margin is +9.3% was reported at -248% and the run
    aborted (QA NEW-02), and a 0.5% discount looked like 50%.
    """
    a = np.asarray(list(values) if not isinstance(values, np.ndarray) else values,
                   dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "fraction"
    peak = float(np.percentile(np.abs(a), 95)) if a.size >= 20 else float(np.abs(a).max())
    return "percent" if peak > _PCT_UNIT_CUTOFF else "fraction"


def pct_to_fraction(values, default: float = 0.0) -> np.ndarray:
    """Read a percentage column as fractions (0.40), whatever unit it uses."""
    a = np.asarray(values, dtype=float)
    out = a / 100.0 if pct_unit(a) == "percent" else a
    return np.where(np.isfinite(out), out, default)


def pct_unit_note(unit: str, what: str) -> str:
    """One plain-English line saying how we read the column, so a misread shows."""
    if unit == "percent":
        return (f"Your {what} column is written as percentages, so 40 means 40% "
                f"and 0.5 means half a percent.")
    return (f"Your {what} column is written as fractions, so 0.40 means 40% "
            f"and 40 would mean 4000%.")


# ---------------------------------------------------------------------------
# Win-probability curve
# ---------------------------------------------------------------------------
def logistic_win_prob(discount_pct, floor: float, ceiling: float,
                      d50: float, steepness: float) -> np.ndarray:
    """Discount -> win probability. S-curve: gentle at first, then steep.

    floor     : win probability at zero discount
    ceiling   : asymptotic maximum (you cannot buy 100% certainty)
    d50       : discount where you are halfway between floor and ceiling
    steepness : how abruptly persuasion turns into giveaway
    """
    d = np.asarray(discount_pct, dtype=float)
    s = max(float(steepness), 0.05)
    return floor + (ceiling - floor) / (1.0 + np.exp(-(d - d50) / s))


def fit_win_curve(discounts, wins, floor: float, ceiling: float,
                  iters: int = 600, lr: float = 0.05, l2: float = 1e-3) -> tuple[float, float]:
    """Fit (d50, steepness) to historical (discount, won) pairs by deterministic
    gradient descent on log-loss. Returns (d50, steepness)."""
    d = np.asarray(discounts, dtype=float)
    y = np.asarray(wins, dtype=float)
    if d.size < 8 or y.size != d.size or (0 < y.mean() < 1) is False:
        return 15.0, 3.0
    d50, s = 15.0, 3.0
    n = d.size
    for _ in range(iters):
        z = (d - d50) / max(s, 0.05)
        pr = floor + (ceiling - floor) / (1.0 + np.exp(-z))
        pr = np.clip(pr, 1e-6, 1 - 1e-6)
        err = (pr - y) / (pr * (1 - pr) + 1e-9)
        dr = (ceiling - floor) * np.exp(-z) / (1 + np.exp(-z)) ** 2
        g_d50 = float(np.mean(err * dr * (-1.0 / max(s, 0.05)))) + l2 * (d50 - 15.0)
        g_s = float(np.mean(err * dr * (-(d - d50) / max(s, 0.05) ** 2))) + l2 * (s - 3.0)
        d50 -= lr * g_d50
        s -= lr * g_s
        d50 = float(np.clip(d50, 0.0, 80.0))
        s = float(np.clip(s, 0.3, 25.0))
    return float(d50), float(s)


def binned_win_rates(discounts, wins, bins: int = 8) -> list[dict]:
    d = np.asarray(discounts, dtype=float)
    y = np.asarray(wins, dtype=float)
    if d.size == 0:
        return []
    edges = np.linspace(d.min(), d.max() + 1e-9, bins + 1)
    out = []
    for i in range(bins):
        m = (d >= edges[i]) & (d < edges[i + 1])
        if m.sum() >= 1:
            out.append({
                "lo": float(edges[i]), "hi": float(edges[i + 1]),
                "mid": float(0.5 * (edges[i] + edges[i + 1])),
                "n": int(m.sum()), "win_rate": float(y[m].mean()),
            })
    return out


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------
def new_result(ctx, use_case_id: str, title: str, model_used: str,
               n_sims: int) -> RunResult:
    return RunResult(
        run_id=ctx.run_id,
        use_case_id=use_case_id,
        use_case_title=title,
        seed=ctx.request.seed,
        status="succeeded",
        model_used=model_used,
        n_simulations=int(n_sims),
        log=ctx.narrator.events(),
    )


#: QA DF-13 / UX-13: a result page is unrenderable when a table emits one row
#: per entity. No result table may exceed this; the rest are aggregated or
#: left to the Excel download.
MAX_TABLE_ROWS = 200


def table(name: str, description: str, columns: Sequence[str],
          rows: Sequence[Sequence[Any]], highlight: Sequence[int] | None = None,
          note: str = "") -> ResultTable:
    """Build a result table, capping it at MAX_TABLE_ROWS.

    Over-long tables broke the results page (QA measured ~50,000 rows on one
    page). We keep the largest rows and say plainly how many were left out.
    """
    all_rows = [list(r) for r in rows]
    total = len(all_rows)
    kept = all_rows
    extra_note = ""
    if total > MAX_TABLE_ROWS:
        kept = all_rows[:MAX_TABLE_ROWS]
        extra_note = (f"Showing the first {MAX_TABLE_ROWS:,} of {total:,} rows - "
                      f"download the Excel for the full list.")
        note = f"{note} {extra_note}".strip()
        description = f"{description} {extra_note}".strip()
    # drop highlight rows that no longer exist after truncation
    hl = [int(h) for h in (highlight or []) if 0 <= int(h) < len(kept)]
    return ResultTable(
        name=name, description=description, columns=list(columns),
        rows=kept,
        highlight_rows=hl, highlight_note=note,
    )


def chart(cid: str, ctype: str, title: str, **kw) -> ChartSpec:
    return ChartSpec(id=cid, type=ctype, title=title, **kw)


def kpi(label: str, value: str, tone: str = "neutral", plain: str = "",
        unit: str | None = None, comparison: str | None = None) -> KPI:
    return KPI(label=label, value=value, tone=tone, plain=plain,
               unit=unit, comparison=comparison)


def story(title: str, detail: str, evidence: str | None = None) -> StoryStep:
    return StoryStep(title=title, detail=detail, evidence=evidence)


def risk_label(score: float) -> str:
    s = as_float(score)
    if s >= 75:
        return "High"
    if s >= 50:
        return "Moderate"
    if s >= 25:
        return "Low"
    return "Minimal"


def tone_for(higher_is_better: bool, value: float, benchmark: float) -> str:
    if value == benchmark:
        return "neutral"
    better = value > benchmark if higher_is_better else value < benchmark
    return "good" if better else "bad"


def linspace(lo: float, hi: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.asarray([lo], dtype=float)
    return np.linspace(lo, hi, n)


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s)).strip("_").lower()
