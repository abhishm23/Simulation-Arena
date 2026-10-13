"""USE CASE 1 - Deal Approval & Margin Balancing (Expected Profit Simulation).

Model: Monte Carlo (with an empirical win-probability curve)

The question a spreadsheet cannot answer: "the rep wants 25% off to guarantee
the win - does that actually make us money?" A spreadsheet shows the margin IF
the deal closes. This simulates 10,000 worlds for every discount level and
reports the expected profit = P(win) x margin dollars, which is the only number
that is comparable across discount levels.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

from ..core.contracts import ParamSpec, Recommendation, RoleSpec, UseCaseMeta
from ..core.registry import UseCase, register
from ..engine import distributions as dist
from ..engine import stats as st
from . import _helpers as H

META = UseCaseMeta(
    id="deal_approval",
    title="Deal Approval & Margin Balancing",
    subtitle="What discount actually maximises expected profit?",
    category="Deal Desk",
    icon="⚖️",
    models=["Monte Carlo"],
    business_question="How much discount can we give before we are buying wins with our own profit?",
    plain_summary=(
        "A discount does two opposite things at once: it raises your chance of "
        "winning and it shrinks the profit you keep when you do win. This test "
        "plays out thousands of possible outcomes for every discount level and "
        "shows the one where those two forces balance best."
    ),
    story_steps=[
        "Read the deal: size, cost and standard margin",
        "Build a win-probability curve from your history (or your assumption)",
        "Simulate thousands of outcomes at every discount level",
        "Find the discount with the highest expected profit",
        "Compare it with the discount being requested",
    ],
    data_granularity="deal",
    sample_dataset="deals.csv",
    input_roles=[
        RoleSpec(key="deal_id", label="Deal / quote ID", dtype="identifier", required=False,
                 aliases=["id", "opportunity", "opp_id", "quote", "deal"]),
        RoleSpec(key="customer", label="Customer name", dtype="categorical", required=False,
                 aliases=["account", "client", "customer_name"]),
        RoleSpec(key="deal_value", label="Deal value (list price)", dtype="numeric", required=False,
                 aliases=["amount", "value", "list_price", "revenue", "arr", "tcv"],
                 fallback=1_000_000,
                  help="What the deal is worth before any discount."),
        RoleSpec(key="cost", label="Cost to deliver", dtype="numeric", required=False,
                 aliases=["cogs", "cost_to_serve", "delivery_cost"],
                  help="Optional if you provide margin % instead."),
        RoleSpec(key="margin_pct", label="Standard margin (%)", dtype="numeric", required=False,
                 aliases=["margin", "gross_margin", "margin_percent"], fallback=0.40,
                  help="Margin at list price. Use 0.40 for 40%."),
        RoleSpec(key="discount_pct", label="Discount given (%)", dtype="numeric", required=False,
                 aliases=["discount", "discount_percent", "off"],
                  help="Historical discounts - used to learn the win curve."),
        RoleSpec(key="won", label="Won? (1/0)", dtype="numeric", required=False,
                 aliases=["win", "is_won", "outcome", "status_won"],
                  help="1 if the deal was won, 0 if lost. Used to learn the win curve."),
        # QA TM-23: these three columns ARE the UC1 business problem. They used to
        # be listed and then ignored, so the model could not answer the question
        # the data was actually about.
        RoleSpec(key="tier_accelerator_hit", label="Hit a consumption tier accelerator",
                 dtype="numeric", required=False,
                 aliases=["accelerator", "tier_accelerator", "tier_break", "overage_tier"],
                  help="1 if this deal crossed a consumption tier, 0 if not. Crossing a "
                      "tier deepens the price you effectively give away."),
        RoleSpec(key="burst_allowance_pct", label="Unmetered burst used", dtype="numeric",
                 required=False,
                 aliases=["burst", "burst_pct", "overage_pct", "unmetered"],
                  help="Share of the deal's value consumed as unmetered burst - usage you "
                      "still have to pay for but never billed."),
        RoleSpec(key="rollover_window_months", label="Rollover window", dtype="numeric",
                 required=False,
                 aliases=["rollover", "rollover_months", "carry_forward"],
                  help="How many months unused allowance can be carried forward. Rolled-"
                      "over revenue arrives later, so it is worth less today."),
    ],
    parameters=[
        ParamSpec(key="deal_value", label="Deal value (list price)", kind="number",
                  default=1_000_000, min=0, step=10_000, unit="$",
                  group="The deal",
                  help="Used when you have no data file. Value before discount."),
        ParamSpec(key="standard_margin_pct", label="Standard margin", kind="number",
                  default=40, min=0, max=95, step=1, unit="%", group="The deal",
                  help="Your margin at full price. 40 means 40%."),
        ParamSpec(key="requested_discount_pct", label="Discount the rep is asking for",
                  kind="number", default=25, min=0, max=90, step=1, unit="%",
                  group="The deal", help="We will compare this against the optimum."),
        # QA TM-23: the cost of a tier accelerator, and the rate used to value
        # revenue that rollover defers.
        ParamSpec(key="tier_accelerator_penalty_pct", label="Cost of a tier accelerator",
                  kind="number", default=0, min=0, max=50, step=1, unit="%",
                  group="Consumption",
                  help="How much of the deal's value you effectively give away once the "
                       "customer crosses a consumption tier. 0 switches it off."),
        ParamSpec(key="discount_rate_pct", label="Rate for valuing delayed revenue",
                  kind="number", default=8, min=0, max=30, step=1, unit="%",
                  group="Consumption",
                  help="Used only if your file has a rollover column: money that arrives "
                       "later is worth less than money today."),
        ParamSpec(key="win_prob_at_zero", label="Win chance at full price", kind="number",
                  default=35, min=1, max=99, step=1, unit="%", group="Buyer behaviour",
                  help="Your honest odds if you refuse any discount."),
        ParamSpec(key="win_prob_ceiling", label="Best win chance money can buy", kind="number",
                  default=95, min=5, max=99, step=1, unit="%", group="Buyer behaviour",
                  help="Even a giveaway price does not win every deal."),
        ParamSpec(key="win_curve_midpoint", label="Discount where you are halfway there",
                  kind="number", default=14.8, min=0, max=80, step=0.5, unit="%",
                  group="Buyer behaviour",
                  help="The discount at which the buyer is 50/50 swayed."),
        ParamSpec(key="win_curve_steepness", label="How sharply the buyer flips",
                  kind="number", default=3.0, min=0.3, max=25, step=0.5,
                  group="Buyer behaviour",
                  help="Low = buyer barely reacts. High = a small extra discount flips them."),
        ParamSpec(key="win_prob_uncertainty", label="How unsure are those odds", kind="number",
                  default=5, min=0, max=25, step=1, unit="pp", group="Uncertainty",
                  help="Spread around the win chance, in percentage points."),
        ParamSpec(key="cost_volatility_pct", label="Cost uncertainty", kind="number",
                  default=6, min=0, max=50, step=1, unit="%", group="Uncertainty",
                  help="How much your cost to deliver might move."),
        ParamSpec(key="margin_floor_pct", label="Margin floor", kind="number",
                  default=12, min=0, max=60, step=1, unit="%", group="Guardrails",
                  help="Company minimum margin. We flag discounts that breach it."),
        ParamSpec(key="max_discount_pct", label="Test discounts up to", kind="number",
                  default=40, min=5, max=90, step=5, unit="%", group="Guardrails", help="The largest discount to test. Nothing above this is considered.",),
        ParamSpec(key="n_simulations", label="Simulations per discount level", kind="integer",
                  default=10000, min=1000, max=200000, step=1000, group="Engine", help="How many possible worlds to play at each discount. More gives a steadier answer.",),
        # How close counts as "the same answer". This is a judgement, not a fact,
        # so it belongs to the user rather than to us (see QA DF-07).
        ParamSpec(key="band_tolerance_pct", label="How close counts as the same answer",
                  kind="number", default=1, min=0.1, max=25, step=0.5, unit="%",
                  group="Guardrails",
                  help="Two discounts are 'the same answer' when the expected profit they "
                       "differ by is under this share of the best one (and under the "
                       "simulation's own margin of error). Raise it for a wider, more "
                       "forgiving range of discounts you are happy to approve; lower it "
                       "for a tighter one."),
    ],
)


# ---------------------------------------------------------------------------
# The range where the answer is the same
# ---------------------------------------------------------------------------
_PCT_RE = re.compile(r"^-?\d+(?:\.\d+)?%$")

#: A percentage in a *probability* field must be 0-100. Other percentages in
#: this use case legitimately go negative or past 100 - a deal's margin can be
#: -85%, the average portfolio margin can be below zero, a discount is signed -
#: so the check is by LABEL, exactly as qa_retest.py does it. A blanket regex
#: over every string aborted the whole run on a perfectly ordinary negative
#: margin (QA NEW-02).
_PROBABILITY_LABEL = re.compile(
    r"chance|probab|odds|share of|risk score|% of deals|likely to|how often",
    re.I)


def _walk(obj):
    """Yield every leaf value inside a nested dict / list structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def _answer_band(grid, exp_profit, level_se, best_i: int,
                 floor_pct: float = 1.0) -> dict:
    """The discounts whose expected profit is the same as the best one.

    Every discount's expected profit is an average over `n_sim` simulated
    worlds, so it carries its own margin of error (`sd / sqrt(n_sim)`). The
    *gap* between two such averages carries the wobble of both, so a gap counts
    as "the same answer" when it is smaller than `2 x (SE_this + SE_best)`.
    A floor (1% of the best profit by default) stops a trivially small wobble
    from pretending to be certainty.

    The floor is a JUDGEMENT, not a fact, so it is exposed to the user as
    "how close counts as the same answer" - raise it for a wider, more
    forgiving band; lower it for a stricter one.

    The band is returned as the unbroken run of tested levels around the best
    one, so every discount quoted inside it really is within the tolerance.
    """
    grid = np.asarray(grid, dtype=float)
    exp_profit = np.asarray(exp_profit, dtype=float)
    level_se = np.asarray(level_se, dtype=float)
    peak = float(exp_profit[best_i])
    floor_tol = (max(float(floor_pct), 0.0) / 100.0) * peak if peak > 0 else 0.0
    tol = np.maximum(2.0 * (level_se + level_se[best_i]), floor_tol)
    inside = (peak - exp_profit) <= tol
    inside[best_i] = True                     # the best is always its own answer

    lo_i = hi_i = int(best_i)
    while lo_i > 0 and bool(inside[lo_i - 1]):
        lo_i -= 1
    while hi_i < exp_profit.size - 1 and bool(inside[hi_i + 1]):
        hi_i += 1

    beyond_i = hi_i + 1
    has_beyond = beyond_i < exp_profit.size
    return {
        "best_i": int(best_i),
        "peak": peak,
        "tol": tol,
        "inside": inside,
        "lo_i": lo_i,
        "hi_i": hi_i,
        "low": float(grid[lo_i]),
        "high": float(grid[hi_i]),
        "single": lo_i == hi_i,
        "covers_all": lo_i == 0 and hi_i == exp_profit.size - 1,
        "beyond_i": beyond_i,
        "has_beyond": has_beyond,
        "beyond_cost": (peak - float(exp_profit[beyond_i])) if has_beyond else 0.0,
        "extra_inside": [int(i) for i in range(exp_profit.size)
                         if inside[i] and not (lo_i <= i <= hi_i)],
    }


def _check_pct(label: str, value: Any, bad: list) -> None:
    """Record a percentage that cannot be what its label says it is."""
    if not isinstance(value, str):
        return
    text = value.strip()
    if not _PCT_RE.match(text):
        return
    n = float(text[:-1])
    if _PROBABILITY_LABEL.search(label or ""):
        if not (0.0 <= n <= 100.0):
            bad.append((label, text))
    elif abs(n) > 100.0:
        # Not a probability, so a negative figure is fine (a margin can be
        # -12%) - but nothing a margin, discount, rate or share can be is
        # bigger than 100% in size, so that is still a real defect.
        bad.append((label, text))


def _self_check(res, recs, grid, band) -> None:
    """Raise if anything the user will see breaks the correctness contract."""
    payload = res.model_dump(mode="json")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"the result contains a number that is not finite: {exc}") from exc
    # JSON mode turns a NaN into null, so inspect the raw values as well.
    for v in _walk(res.model_dump()):
        if isinstance(v, float) and not math.isfinite(v):
            raise AssertionError(f"the result contains a number that is not finite: {v!r}")
    for v in _walk(payload):
        if isinstance(v, float) and not math.isfinite(v):
            raise AssertionError(f"the result contains a number that is not finite: {v!r}")

    # Percentages: probability-labelled fields must be 0-100, everything else
    # only has to be a size a percentage can actually be (see _check_pct).
    bad_pct: list = []
    for k in res.kpis:
        _check_pct(k.label, k.value, bad_pct)
    for t in res.tables:
        for j, header in enumerate(t.columns):
            for row in t.rows:
                if j < len(row):
                    _check_pct(header, row[j], bad_pct)
    for c in res.charts:
        for s in c.series:
            if _PROBABILITY_LABEL.search(str(s.get("name", ""))):
                for v in (s.get("values") or []):
                    if isinstance(v, (int, float)) and not (0.0 <= float(v) <= 100.0):
                        bad_pct.append((f"chart '{c.id}'/{s.get('name')}", v))
    if bad_pct:
        raise AssertionError(
            "a percentage does not match its label: "
            + "; ".join(f"{lbl} = {val}" for lbl, val in bad_pct[:5]))

    for r in recs:
        for key in ("win_prob", "prob_breach_floor", "prob_loss"):
            p = float(r[key])
            if not (0.0 <= p <= 1.0):
                raise AssertionError(f"'{key}' is not a probability: {p}")

    for t in res.tables:
        ncol = len(t.columns)
        for i, row in enumerate(t.rows):
            if len(row) != ncol:
                raise AssertionError(
                    f"table '{t.name}' row {i + 1} has {len(row)} cells, "
                    f"but the table has {ncol} column headers")

    lo_i, hi_i, best_i = band["lo_i"], band["hi_i"], band["best_i"]
    if not (0 <= lo_i <= best_i <= hi_i < len(grid)):
        raise AssertionError("the range where the answer is the same is not well formed")
    if not (abs(float(grid[lo_i]) - band["low"]) < 1e-9
            and abs(float(grid[hi_i]) - band["high"]) < 1e-9):
        raise AssertionError("the range edges are not tested discount levels")
    for i in range(lo_i, hi_i + 1):
        if not band["inside"][i]:
            raise AssertionError(
                "the range quotes a discount that is outside the margin of error")


@register
class DealApprovalUseCase(UseCase):
    meta = META

    def run(self, req, ctx) -> "H.RunResult":
        P = ctx.params
        rows = ctx.rows
        rng = ctx.rng("deal_approval", "main")
        n_sim = int(H.p(P, "n_simulations", 10000))
        floor_pct = H.as_float(H.p(P, "margin_floor_pct", 12)) / 100.0

        # -- 1. read the deal ------------------------------------------------
        ctx.say("Reading the deal you gave us...", phase="read", progress=0.05)
        # QA DF-10 / UX-04: a value the user TYPED at Step 4 outranks the value
        # learned from their file. `explicit_params` holds the keys the user
        # actually set, so "typed" is distinguishable from "still the default".
        explicit = set(getattr(ctx.request, "explicit_params", None) or [])
        typed_value = "deal_value" in explicit
        typed_margin = "standard_margin_pct" in explicit
        if typed_value or typed_margin:
            ctx.say("You typed the deal size or the margin yourself, so we are using the "
                    "figures you entered rather than reading them from your file.",
                    phase="read", progress=0.06)

        has_data = H.has(rows, "deal_value") or H.has(rows, "cost") or H.has(rows, "margin_pct")

        margin_unit = "fraction"
        if has_data:
            values = H.num(rows, "deal_value", 0.0)
            values = np.where(values > 0, values, 0.0)
            if H.has(rows, "cost") and not H.has(rows, "margin_pct"):
                costs = H.num(rows, "cost", 0.0)
                margin_each = np.where(values > 0, 1.0 - costs / np.maximum(values, 1e-9), 0.0)
            else:
                m = H.num(rows, "margin_pct", 0.40)
                # QA NEW-02 / NEW-07: the unit is decided ONCE for the whole
                # column. A per-value `> 1.0` test read every negative margin as
                # a fraction (-8 -> -800%), which collapsed the size-weighted
                # average and aborted the run at SUMMARISING.
                margin_unit = H.pct_unit(m)
                margin_each = H.pct_to_fraction(m, 0.40)   # accept 40 or 0.40
            tot_value = float(values.sum()) if values.size else H.as_float(H.p(P, "deal_value", 1e6))
            w = values / max(tot_value, 1e-9)
            std_margin = float(np.sum(margin_each * w)) if values.size else 0.40
            n_deals = int((values > 0).sum()) or len(rows) or 1
            # A typed value wins over the file (DF-10). Override only what was typed.
            file_value, file_margin = tot_value, std_margin
            if typed_value:
                tot_value = H.as_float(H.p(P, "deal_value", tot_value))
            if typed_margin:
                std_margin = H.as_float(H.p(P, "standard_margin_pct", std_margin * 100.0)) / 100.0
            if typed_value or typed_margin:
                ctx.say(f"You typed {'the deal size' if typed_value else ''}"
                        f"{' and ' if typed_value and typed_margin else ''}"
                        f"{'the margin' if typed_margin else ''}, so we used your figure "
                        f"rather than the file's ({H.money(file_value, compact=True)} at "
                        f"{H.pct(file_margin)}).", phase="validate", level="warn")
            ctx.say(f"Found {n_deals} deal(s) worth {H.money(tot_value, compact=True)} "
                    f"at an average {H.pct(std_margin)} margin.", phase="read",
                    detail=f"weighted by deal size; total list value {tot_value:,.0f}; "
                           f"{H.pct_unit_note(margin_unit, 'margin')}")
        else:
            tot_value = H.as_float(H.p(P, "deal_value", 1_000_000))
            std_margin = H.as_float(H.p(P, "standard_margin_pct", 40)) / 100.0
            n_deals = 1
            ctx.say(f"Working from the numbers you typed: {H.money(tot_value, compact=True)} "
                    f"at {H.pct(std_margin)} margin.", phase="read")

        # QA NEW-02: UC1 exists to model margins collapsing below zero, so a
        # negative average margin is a normal input, not a broken one. Say so
        # plainly, and warn if the figure is so far out that we probably read
        # the column in the wrong unit.
        negative_margin = std_margin < 0.0
        if negative_margin:
            ctx.say(f"Your deals run at a negative average margin ({H.pct(std_margin)}): on "
                    f"average each deal costs more to deliver than it sells for, so every "
                    f"discount you test loses money. We will show you the least-bad option.",
                    phase="read", level="warn",
                    detail=H.pct_unit_note(margin_unit, "margin"))
        if abs(std_margin) > 1.0:
            ctx.say(f"The average margin works out at {H.pct(std_margin)}, which is outside "
                    f"the range a margin can be. Check that your margin column really holds "
                    f"a percentage and not a ratio or a cost. We have carried on with it so "
                    f"you still get an answer.", phase="validate", level="warn",
                    detail=H.pct_unit_note(margin_unit, "margin"))

        # QA NEW-09: keep the price-versus-margin cost separate from the
        # consumption extras added below. Adding the extras straight into the
        # one number we printed next to "9.4% standard margin" made the sentence
        # disagree with itself by ~3 points. Two named bases instead of one
        # silently-mixed one.
        goods_cost = tot_value * (1.0 - std_margin)
        cost_total = goods_cost
        ctx.say(f"Cost to deliver is about {H.money(goods_cost, compact=True)}.", phase="prepare",
                detail=f"list value {tot_value:,.0f} x (1 - {std_margin:.3f})")

        # --- QA TM-23: UC1's own business problem -----------------------------
        # Tier accelerators, unmetered bursts and quarterly rollover were in the
        # file but never used, so the model could not answer the question the
        # data was actually about. They are consumption effects, so they land on
        # cost (usage you still pay for) and on revenue you cannot book yet.
        def _mean_role(role: str, default: float) -> float:
            if not H.has(rows, role):
                return default
            arr = H.num(rows, role, np.nan)
            arr = arr[np.isfinite(arr)]
            return float(arr.mean()) if arr.size else default

        accel_share = _mean_role("tier_accelerator_hit", 0.0)
        if accel_share > 1.0:                     # accept 0/1 or a percentage
            accel_share /= 100.0
        burst_pct = _mean_role("burst_allowance_pct", 0.0)
        rollover_m = _mean_role("rollover_window_months", 0.0)
        if rollover_m > 24.0:                     # months, not years
            rollover_m = min(rollover_m, 12.0)

        # A tier accelerator is a step: once the customer crosses a consumption
        # tier the price you effectively give them deepens.
        accel_penalty = H.as_float(H.p(P, "tier_accelerator_penalty_pct", 0)) / 100.0
        disc_rate = H.as_float(H.p(P, "discount_rate_pct", 8))
        extra_cost = tot_value * (accel_share * accel_penalty + burst_pct / 100.0)
        # Rollover defers revenue, so the money is worth less when it lands.
        roll_share = min(max(rollover_m, 0.0) / 12.0, 1.0)
        deferred_cost = tot_value * roll_share * (disc_rate / 100.0)
        extra_cost += deferred_cost

        consumption_note = None
        # QA NEW-09: the margin that actually goes with the cost we print. With
        # no extras it is just the file's standard margin; with extras it is the
        # margin those extras leave behind. Never print a margin and a cost that
        # describe different cost bases.
        effective_margin = std_margin
        if extra_cost > 1.0:
            cost_total = goods_cost + extra_cost
            if tot_value:
                effective_margin = 1.0 - cost_total / tot_value
            parts = []
            if accel_share > 0 and accel_penalty > 0:
                parts.append(f"{H.pct(accel_share)} of deals hit a consumption tier "
                             f"accelerator, costing {H.pct(accel_penalty)} of value each")
            if burst_pct > 0:
                parts.append(f"{H.pct_num(burst_pct)} of value is consumed as unmetered "
                             f"burst you still pay for")
            if roll_share > 0:
                parts.append(f"{rollover_m:.0f} months of rollover defers revenue")
            consumption_note = "Consumption effects: " + "; ".join(parts) + "."
            ctx.say(f"Added the consumption effects from your file: {H.money(extra_cost, compact=True)} "
                    f"of extra cost to serve, taking the cost base to "
                    f"{H.money(cost_total, compact=True)}.", phase="prepare", level="warn",
                    detail=consumption_note)
        else:
            consumption_note = ("Consumption effects: none - your file has no tier "
                                "accelerator, burst or rollover columns, so this is a "
                                "straight price-versus-margin question.")

        # -- 2. win-probability curve ---------------------------------------
        ctx.say("Working out how much a discount really moves the buyer...", phase="prepare",
                progress=0.15)
        pf = H.as_float(H.p(P, "win_prob_at_zero", 35)) / 100.0
        pc = H.as_float(H.p(P, "win_prob_ceiling", 95)) / 100.0
        pc = max(pc, pf + 0.01)
        d50 = H.as_float(H.p(P, "win_curve_midpoint", 14.8))
        steep = H.as_float(H.p(P, "win_curve_steepness", 3.0))

        curve_source = "your assumption"
        if H.has(rows, "discount_pct") and H.has(rows, "won"):
            d_obs = H.num(rows, "discount_pct", 0.0)
            # Same whole-column unit rule as the margin (NEW-02): accept 18 or
            # 0.18, but never let a 0.5% discount read as 50%.
            d_obs = H.pct_to_fraction(d_obs, 0.0) * 100.0
            w_obs = H.num(rows, "won", 0.0)
            w_obs = np.where(w_obs > 0.5, 1.0, 0.0)
            if d_obs.size >= 8 and 0.0 < w_obs.mean() < 1.0:
                d50, steep = H.fit_win_curve(d_obs, w_obs, pf, pc)
                curve_source = f"{d_obs.size} of your past deals"
                ctx.say(f"Learned the buyer's behaviour from {int(d_obs.size)} past deals "
                        f"(they flip hardest around {d50:.1f}% discount).", phase="prepare",
                        detail=f"fitted logistic: d50={d50:.2f}, steepness={steep:.2f}")
            else:
                ctx.say("Not enough historical win/loss data to learn from - using your assumption.",
                        phase="prepare", level="warn")
        else:
            ctx.say("No historical win/loss columns mapped - using the curve you set.",
                    phase="prepare", level="warn")

        # -- 3. sweep discount levels ---------------------------------------
        max_d = H.as_float(H.p(P, "max_discount_pct", 40))
        grid = np.linspace(0.0, max_d, 17)
        ctx.say(f"Testing {len(grid)} discount levels, {n_sim:,} outcomes each "
                f"({len(grid) * n_sim:,} scenarios in total)...",
                phase="simulate", progress=0.25,
                detail="each scenario asks: do we win, and what do we keep if we do?")

        sd_win = H.as_float(H.p(P, "win_prob_uncertainty", 5)) / 100.0
        cv_cost = H.as_float(H.p(P, "cost_volatility_pct", 6)) / 100.0

        recs = []
        tick = ctx.narrator.tracker(len(grid), "Simulated {pct}% of the discount levels...",
                                    phase="simulate")
        for i, d in enumerate(grid):
            frac = d / 100.0
            revenue = tot_value * (1.0 - frac)
            p_win = float(H.logistic_win_prob(d, pf, pc, d50, steep))

            win_draws = np.clip(rng.normal(p_win, sd_win, n_sim), 0.0, 1.0)
            # abs(): the spread of a cost is a size, not a direction - a negative
            # average cost (a margin above 100%) must not flip the sign of it.
            cost_draws = np.clip(rng.normal(cost_total, abs(cost_total) * cv_cost, n_sim),
                                 0.0, None)
            margin_if_won = revenue - cost_draws
            won = rng.random(n_sim) < win_draws
            realised = np.where(won, margin_if_won, 0.0)

            eff_margin_pct = np.where(revenue > 0, margin_if_won / revenue, 0.0)
            s = st.summarize(realised)
            recs.append({
                "discount": float(d),
                "revenue": float(revenue),
                "margin_if_won": float(margin_if_won.mean()),
                "win_prob": float(win_draws.mean()),
                "expected_profit": float(realised.mean()),
                "p5": s.get("p5", 0.0),
                "p95": s.get("p95", 0.0),
                "prob_breach_floor": float((eff_margin_pct < floor_pct).mean()),
                "prob_loss": float((margin_if_won < 0).mean()),
                "realised": realised,
            })
            tick(i)

        # -- 4. optimum ------------------------------------------------------
        exp_profit = np.array([r["expected_profit"] for r in recs])
        best_i = int(np.argmax(exp_profit))
        best = recs[best_i]
        # QA NEW-02: when the portfolio's margin is negative there is no
        # discount that makes money. Saying "the range you can approve without
        # losing money" would then be a claim the numbers do not support, so the
        # wording and the tone follow the sign of the answer.
        all_loss = bool(float(exp_profit[best_i]) <= 0.0)
        ctx.say("All discount levels tested. Looking for the sweet spot...",
                phase="summarize", progress=0.9)

        req_disc = H.as_float(H.p(P, "requested_discount_pct", 25))
        j = int(np.argmin(np.abs(grid - req_disc)))
        asked = recs[j]

        delta = asked["expected_profit"] - best["expected_profit"]
        extra_win_pts = (asked["win_prob"] - best["win_prob"]) * 100.0

        # risk-adjusted optimum (maximise the bad-case, not the average)
        worst = np.array([r["p5"] for r in recs])
        safe_i = int(np.argmax(worst))

        # -- 4b. the range where the answer is the same ----------------------
        # Every discount's expected profit is an average over n_sim worlds, so
        # it carries its own margin of error. A discount is "the same answer"
        # as the best one when the gap between them is smaller than the wobble
        # of both averages put together (or 1% of the best profit, whichever
        # is larger) - that is where the rep has genuine room to negotiate.
        level_se = np.array(
            [float(np.std(r["realised"], ddof=1)) / math.sqrt(max(n_sim, 1)) for r in recs],
            dtype=float)
        # `band_floor_pct` is the user's judgement of how close counts as "the
        # same answer" - see DF-07 in the QA workbook.
        band_floor_pct = H.as_float(H.p(P, "band_tolerance_pct", 1.0))
        band = _answer_band(grid, exp_profit, level_se, best_i, band_floor_pct)
        band_lo, band_hi = band["low"], band["high"]
        band_tol = float(band["tol"][best_i])
        beyond_d = float(grid[band["beyond_i"]]) if band["has_beyond"] else None
        beyond_cost = float(band["beyond_cost"])
        req_inside = bool(band["inside"][j])

        if band["covers_all"]:
            band_phrase = f"{grid[0]:.0f}% to {grid[-1]:.0f}%"
            band_sentence = ("every discount we tested gives the same answer within the "
                             "model's own margin of error")
        elif band["single"]:
            band_phrase = f"{band_lo:.0f}%"
            band_sentence = ("only one discount stands out - the model cannot tell any "
                             "neighbouring discount apart from it")
        else:
            band_phrase = f"{band_lo:.0f}% to {band_hi:.0f}%"
            band_sentence = ("across this range the expected profit is the same within the "
                             "model's own margin of error")

        ctx.say(f"Checked how much the answer would wobble if we ran it again - the range "
                f"where the answer is the same is {band_phrase}.", phase="summarize",
                progress=0.95,
                detail=f"tolerance {band_tol:,.0f} on a peak of {band['peak']:,.0f}")

        # -- 5. assemble -----------------------------------------------------
        res = H.new_result(ctx, META.id, META.title,
                           "Monte Carlo simulation with an empirical win-probability curve",
                           len(grid) * n_sim)

        best_d = float(best["discount"])
        if all_loss:
            res.headline = (
                f"No discount makes money on these numbers: even the least-bad one "
                f"({best_d:.0f}%) loses {H.money(abs(best['expected_profit']), compact=True)}. "
                f"The deal is unprofitable before any discount is given - the fix is the cost "
                f"to deliver or the price, not the size of the concession."
            )
        elif band["covers_all"]:
            res.headline = (
                f"Every discount we tested ({grid[0]:.0f}% to {grid[-1]:.0f}%) gives the same "
                f"answer within the model's own margin of error - the best single number is "
                f"{best_d:.0f}%, but this is a judgement call, not a calculation."
            )
        else:
            if band["single"]:
                core = (f"Approve {band_lo:.0f}% - it is the only discount the model can tell "
                        f"apart from the rest")
            else:
                core = (f"Approve anything between {band_lo:.0f}% and {band_hi:.0f}% - "
                        f"{best_d:.0f}% is the best single number")
            if req_inside:
                tail = (f". The {req_disc:.0f}% the rep is asking for sits inside that range - "
                        f"it is a fine call.")
            elif req_disc > band_hi:
                tail = (f". The {req_disc:.0f}% the rep is asking for is past it - that costs "
                        f"about {H.money(abs(delta), compact=True)} of expected profit.")
            else:
                tail = (f". The {req_disc:.0f}% the rep is asking for is short of it - holding "
                        f"back that much leaves about {H.money(abs(delta), compact=True)} of "
                        f"expected profit unclaimed.")
            res.headline = core + tail

        res.method_plain = (
            "For each discount level we ran thousands of 'what-if' worlds. In each one the "
            "buyer's odds of saying yes wobble a little, and your delivery cost wobbles a little. "
            "We then ask the only question that matters: if you offered this discount again and "
            "again, how much profit would you bank on average? That is the expected profit - "
            "the chance of winning multiplied by the profit you keep. Because those worlds are "
            "a sample, we also work out how much the answer itself would wobble if we ran it "
            "again. Any discount whose expected profit is within twice that wobble of the best - "
            "or within 1% of the best profit, whichever is wider - is treated as giving the same "
            "answer. That is the range where you have genuine room to negotiate."
        )

        if band["has_beyond"]:
            beyond_kpi = H.kpi(
                "Where it starts to genuinely cost you", f"{beyond_d:.0f}%", "warn",
                f"At {beyond_d:.0f}% the expected profit falls "
                f"{H.money(beyond_cost, compact=True)} below the best. That gap is wider than "
                f"the model's own margin of error, so the loss is real.",
                comparison=f"costs about {H.money(beyond_cost, compact=True)}")
        else:
            beyond_kpi = H.kpi(
                "Where it starts to genuinely cost you", "Not reached", "neutral",
                "Even the deepest discount we tested is still within the model's own margin of "
                "error of the best, so we did not find a discount where the answer genuinely "
                "gets worse.")

        range_kpi_label = ("Discount range that loses the least" if all_loss
                           else "Discount range you can approve without losing money")
        res.kpis = [
            H.kpi("Best discount to offer", f"{best_d:.0f}%",
                  "warn" if all_loss else "good",
                  (f"No discount makes money here - {best_d:.0f}% loses the least."
                   if all_loss else
                   f"This is where the extra wins are still worth more than the margin given away."),
                  comparison=f"rep asked for {req_disc:.0f}%"),
            H.kpi("Expected profit at that discount", H.money(best["expected_profit"], compact=True),
                  "bad" if all_loss else "good",
                  ("This is a loss, not a profit: on these numbers the deal costs more to deliver "
                   "than it sells for." if all_loss else
                   "Average profit if you made this call many times over.")),
            H.kpi(range_kpi_label, band_phrase, "bad" if all_loss else "good",
                  (f"{band_sentence.capitalize()}. The model's own margin of error here is about "
                   f"{H.money(band_tol, compact=True)}, so treat this as the best single number - "
                   f"the next discount along already costs real money."
                   if band["single"] else
                   f"{band_sentence.capitalize()}. The model's own margin of error here is about "
                   f"{H.money(band_tol, compact=True)}, so anywhere in this range is a judgement "
                   f"call, not a calculation."),
                  comparison=f"rep asked for {req_disc:.0f}%"
                              + ("" if req_inside else " - outside the range")),
            beyond_kpi,
            H.kpi("Expected profit if you cave in", H.money(asked["expected_profit"], compact=True),
                  "good" if req_inside else ("bad" if delta < 0 else "good"),
                  f"What {req_disc:.0f}% actually earns you."
                  + (" It is inside the range where the answer is the same."
                     if req_inside else ""),
                  comparison=f"{H.money(delta, compact=True)} vs the optimum"),
            H.kpi("Profit given away", H.money(abs(delta), compact=True),
                  "neutral" if req_inside else ("bad" if delta < 0 else "neutral"),
                  ("Inside the model's own margin of error - this is not a real loss, it is the "
                   "same answer." if req_inside else
                   f"Buying {extra_win_pts:+.0f} points of win probability costs you this much.")),
            H.kpi("Odds of winning at the best discount", H.pct(best["win_prob"]),
                  "neutral", "The chance the deal lands."),
            H.kpi("Worst case (1-in-20)", H.money(best["p5"], compact=True), "warn",
                  "In an unlucky run of 20 deals, this is roughly the profit you would see."),
        ]

        res.charts = [
            H.chart(
                "profit_curve", "line", "Where profit peaks",
                subtitle=f"Everything from {band_phrase} gives the same answer within the "
                         f"model's margin of error; the peak is {best_d:.0f}%",
                x_label="Discount offered", y_label="Expected profit",
                x=[H.r2(r["discount"]) for r in recs],
                series=[
                    {"name": "Expected profit", "values": [H.r2(r["expected_profit"]) for r in recs],
                     "color": "#22c55e"},
                    {"name": "Profit if the deal closes",
                     "values": [H.r2(r["margin_if_won"]) for r in recs], "color": "#64748b",
                     "dash": True},
                    {"name": "The best answer, minus the model's margin of error",
                     "values": [H.r2(band["peak"] - band_tol) for _ in recs],
                     "color": "#f59e0b", "dash": True},
                ],
                annotations=[
                    {"x": H.r2(best_d), "y": H.r2(best["expected_profit"]),
                     "text": f"Best: {best_d:.0f}%"},
                    {"x": H.r2(band_lo), "y": H.r2(float(exp_profit[band["lo_i"]])),
                     "text": f"Same answer from {band_lo:.0f}%"},
                    {"x": H.r2(band_hi), "y": H.r2(float(exp_profit[band["hi_i"]])),
                     "text": f"Same answer to {band_hi:.0f}%"},
                ],
                plain_meaning="Grey is what you keep per won deal - it only falls. Green is what "
                              "you actually expect to earn - it rises, peaks, then falls. The "
                              "amber dashed line is the best profit minus the model's own margin "
                              f"of error: every discount whose green line stays above it is the "
                              f"same answer, which is the {band_phrase} range. Past that you are "
                              "genuinely paying for the win.",
            ),
            H.chart(
                "win_curve", "line", "What the extra discount buys you",
                subtitle=f"Curve learned from {curve_source}",
                x_label="Discount offered", y_label="Chance of winning",
                x=[H.r2(r["discount"]) for r in recs],
                series=[{"name": "Chance of winning",
                         "values": [H.r2(r["win_prob"] * 100) for r in recs], "color": "#3b82f6"}],
                plain_meaning="Past the knee of this curve, extra discount buys almost no extra "
                              "certainty - but keeps costing margin. That is the trap.",
            ),
            H.chart(
                "outcome_spread", "hist", "The range of outcomes at the best discount",
                subtitle=f"{n_sim:,} simulated worlds at {best['discount']:.0f}% discount",
                x_label="Profit from the deal", y_label="How often",
                x=[], series=[{"name": "Profit outcomes",
                               "values": [float(v) for v in
                                          best["realised"][::max(1, n_sim // 2000)]]}],
                plain_meaning="The cluster at zero is the deals you lose. The spread on the right "
                              "is the profit you make when you win.",
            ),
            H.chart(
                "floor_risk", "line", "Chance of breaking the margin floor",
                subtitle=f"Company floor is {H.pct(floor_pct)}",
                x_label="Discount offered", y_label="Chance margin falls below floor",
                x=[H.r2(r["discount"]) for r in recs],
                series=[{"name": "Chance of breaching the floor",
                         "values": [H.r2(r["prob_breach_floor"] * 100) for r in recs],
                         "color": "#ef4444"}],
                plain_meaning="Even when a discount looks profitable on average, it may still be "
                              "likely to break the company's minimum margin rule.",
            ),
            H.chart(
                "answer_range", "line", "The range where the answer is the same",
                subtitle=f"Under the amber line is the same answer as the best; the safe range "
                         f"is {band_phrase}",
                x_label="Discount offered", y_label="Shortfall against the best answer",
                x=[H.r2(r["discount"]) for r in recs],
                series=[
                    {"name": "How far below the best answer",
                     "values": [H.r2(band["peak"] - r["expected_profit"]) for r in recs],
                     "color": "#38bdf8"},
                    {"name": "The model's own margin of error",
                     "values": [H.r2(band_tol) for _ in recs], "color": "#f59e0b", "dash": True},
                ],
                annotations=(
                    [{"x": H.r2(band_lo), "y": 0.0, "text": f"Same answer from {band_lo:.0f}%"},
                     {"x": H.r2(band_hi), "y": 0.0, "text": f"Same answer to {band_hi:.0f}%"}]
                    + ([{"x": H.r2(beyond_d), "y": H.r2(beyond_cost),
                         "text": f"Costs {H.money(beyond_cost, compact=True)}"}]
                       if band["has_beyond"] else [])
                ),
                plain_meaning="The same numbers as the profit chart, read the other way up: how "
                              "much you give up by moving away from the best discount. While the "
                              "blue line stays under the amber line the difference is inside the "
                              "model's own margin of error, so it is the same answer. The moment "
                              "it crosses, the loss is real and it is worth arguing about.",
            ),
        ]

        res.tables = [
            H.table(
                "Every discount level tested",
                "Expected profit = chance of winning x profit kept. Highlighted row is the winner.",
                ["Discount", "Price after discount", "Profit if won", "Win chance",
                 "Expected profit", "Worst case", "Breaches floor?"],
                [[f"{r['discount']:.1f}%", H.money(r["revenue"]), H.money(r["margin_if_won"]),
                  H.pct(r["win_prob"]), H.money(r["expected_profit"]), H.money(r["p5"]),
                  f"{r['prob_breach_floor'] * 100:.0f}% chance"] for r in recs],
                highlight=[best_i],
                note=f"Row {best_i + 1} maximises expected profit.",
            ),
            H.table(
                "The decision in one line",
                "What changes if you approve the discount being asked for.",
                ["", f"Best ({best['discount']:.0f}%)", f"Asked ({req_disc:.0f}%)", "Difference"],
                [
                    ["Discount", f"{best['discount']:.1f}%", f"{asked['discount']:.1f}%",
                     f"{asked['discount'] - best['discount']:+.1f} pts"],
                    ["Win chance", H.pct(best["win_prob"]), H.pct(asked["win_prob"]),
                     f"{extra_win_pts:+.1f} pts"],
                    ["Profit if won", H.money(best["margin_if_won"]),
                     H.money(asked["margin_if_won"]),
                     H.money(asked["margin_if_won"] - best["margin_if_won"])],
                    ["Expected profit", H.money(best["expected_profit"]),
                     H.money(asked["expected_profit"]), H.money(delta)],
                    ["Range where the answer is the same", band_phrase,
                     "inside it" if req_inside else "outside it",
                     f"margin of error {H.money(band_tol, compact=True)}"],
                ],
            ),
            H.table(
                "The range where the answer is the same",
                "Every discount we tested, how far it falls short of the best, and whether that "
                "shortfall is real money or just the model's own margin of error.",
                ["Discount", "Expected profit", "Short of the best", "Inside the range?",
                 "What it means"],
                [[f"{r['discount']:.1f}%", H.money(r["expected_profit"]),
                  H.money(band["peak"] - r["expected_profit"]),
                  "Inside" if band["lo_i"] <= i <= band["hi_i"] else "Outside",
                  ("safe to approve" if band["lo_i"] <= i <= band["hi_i"]
                   else f"costs you {H.money(band['peak'] - r['expected_profit'], compact=True)}")]
                 for i, r in enumerate(recs)],
                highlight=list(range(band["lo_i"], band["hi_i"] + 1)),
                note=f"Highlighted rows are the range where the answer is the same "
                     f"({band_phrase}) - within about {H.money(band_tol, compact=True)} of the "
                     f"best.",
            ),
        ]

        # per-deal view when a file was supplied
        if has_data and n_deals > 1:
            v = H.num(rows, "deal_value", 0.0)
            ids = H.cat(rows, "deal_id", default="") or [f"Deal {i+1}" for i in range(len(rows))]
            names = H.cat(rows, "customer", default="")
            keep = np.argsort(-v)[:25]
            per = []
            for i in keep:
                if v[i] <= 0:
                    continue
                m_i = float(margin_each[i])
                c_i = v[i] * (1 - m_i)
                best_local, best_local_exp = 0.0, -np.inf
                for r in recs:
                    ep = float(H.logistic_win_prob(r["discount"], pf, pc, d50, steep)) * \
                         (v[i] * (1 - r["discount"] / 100.0) - c_i)
                    if ep > best_local_exp:
                        best_local_exp, best_local = ep, r["discount"]
                per.append([ids[i], names[i] if names[i] else "-", H.money(v[i]),
                            H.pct(m_i), f"{best_local:.0f}%", H.money(best_local_exp)])
            if per:
                res.tables.append(H.table(
                    f"Recommended discount per deal (top {len(per)})",
                    "The same logic applied to each deal individually.",
                    ["Deal", "Customer", "Value", "Margin", "Best discount", "Expected profit"],
                    per,
                ))

        # -- 6. story + recommendation --------------------------------------
        res.story = [
            H.story(
                "The two forces fighting each other",
                f"Every point of discount does two things: it makes the buyer a bit more likely "
                f"to say yes, and it takes {H.money(tot_value / 100.0, compact=True)} straight "
                f"off your profit. Early on the first effect wins. Later, the second one does.",
                f"Cost to deliver {H.money(cost_total, compact=True)} on "
                f"{H.money(tot_value, compact=True)} of list value",
            ),
            H.story(
                "What the buyer's behaviour looks like",
                f"They start around {H.pct(pf)} likely to buy at full price and never become "
                f"certain - the curve flattens near {H.pct(pc)}. Around "
                f"{d50:.0f}% discount is where they are most persuadable.",
                f"Curve source: {curve_source}",
            ),
            H.story(
                "The sweet spot",
                (f"{best_d:.0f}% discount loses the least: "
                 f"{H.money(abs(best['expected_profit']), compact=True)} on average, with a "
                 f"{H.pct(best['win_prob'])} chance of winning. There is no discount here that "
                 f"makes money - the deal is already unprofitable at full price."
                 if all_loss else
                 f"{best_d:.0f}% discount gives the highest expected profit at "
                 f"{H.money(best['expected_profit'], compact=True)}, with a "
                 f"{H.pct(best['win_prob'])} chance of winning."),
                f"peak of {len(grid)} levels tested",
            ),
            H.story(
                "The range where the answer is the same",
                (f"Every discount we tested gives the same answer within the model's own margin "
                 f"of error, so this is a judgement call rather than a calculation."
                 if band["covers_all"] else
                 (f"{best_d:.0f}% is the only discount that stands out; the model cannot tell "
                  f"any neighbouring discount apart from it. The next level along already costs "
                  f"{H.money(beyond_cost, compact=True)}."
                  if band["single"] else
                  f"{band_sentence.capitalize()}. {best_d:.0f}% is the best single number, but "
                  f"anything from {band_lo:.0f}% to {band_hi:.0f}% is a defensible call - that "
                  f"is where the rep has room to negotiate.")),
                f"margin of error {H.money(band_tol, compact=True)}; safe range {band_phrase}",
            ),
            H.story(
                "What the requested discount really costs",
                (f"The requested {req_disc:.0f}% sits inside the range where the answer is the "
                 f"same - it is a fine call, not a rejection. It gives up "
                 f"{H.money(abs(delta), compact=True)} against the very best single number, which "
                 f"is inside the model's own margin of error."
                 if req_inside else
                 (f"Moving from {band_hi:.0f}% to {req_disc:.0f}% buys "
                  f"{extra_win_pts:+.0f} points of win probability but costs "
                  f"{H.money(abs(delta), compact=True)} of expected profit - more than the model's "
                  f"margin of error, so the loss is real."
                  if req_disc > band_hi else
                  f"The requested {req_disc:.0f}% is below the safe range: holding back that much "
                  f"leaves {H.money(abs(delta), compact=True)} of expected profit unclaimed.")),
                f"{H.money(asked['expected_profit'], compact=True)} vs "
                f"{H.money(best['expected_profit'], compact=True)}",
            ),
        ]

        risk = 0.0
        if best["prob_breach_floor"] > 0:
            risk = min(100.0, 25.0 + best["prob_breach_floor"] * 150.0)
        if asked["prob_breach_floor"] > 0.5:
            risk = max(risk, 70.0)

        if all_loss:
            actions = [
                f"Do not approve a discount on these numbers. At the best level we tested "
                f"({best_d:.0f}%) the deal still loses "
                f"{H.money(abs(best['expected_profit']), compact=True)}.",
                f"Fix the deal first: the average margin in your file is {H.pct(std_margin)}, so "
                f"the cost to deliver has to come down (or the price go up) before any discount "
                f"makes sense.",
            ]
        elif band["covers_all"]:
            actions = [f"Any discount from {grid[0]:.0f}% to {grid[-1]:.0f}% is a defensible call "
                       f"here - choose on the relationship, not on the maths."]
        elif band["single"]:
            actions = [f"Approve {band_lo:.0f}% as the standard concession - it is the only "
                       f"discount the model can tell apart from the rest."]
        else:
            actions = [f"Approve any discount between {band_lo:.0f}% and {band_hi:.0f}% - they "
                       f"are the same answer within the model's own margin of error."]
        if req_inside:
            actions.append(
                f"The {req_disc:.0f}% being asked for sits inside that range, so it can be "
                f"approved without further analysis.")
        elif req_disc > band_hi:
            span = max(req_disc - band_hi, 1e-9)
            actions.append(
                f"Above {band_hi:.0f}% (the top of the safe range), require VP sign-off: the "
                f"{req_disc:.0f}% being asked for costs about "
                f"{H.money(abs(delta) / span, compact=True)} per point of discount.")
        else:
            actions.append(
                f"Below {band_lo:.0f}% you are leaving money on the table - the {req_disc:.0f}% "
                f"being asked for gives up {H.money(abs(delta), compact=True)} of expected "
                f"profit.")
        if asked["prob_breach_floor"] > 0.2:
            actions.append(
                f"At {req_disc:.0f}% there is a {asked['prob_breach_floor'] * 100:.0f}% chance of "
                f"breaching the {H.pct(floor_pct)} margin floor - add a volume or term commitment "
                f"to earn the deeper discount.")
        if safe_i != best_i:
            actions.append(
                f"If this quarter must not misfire, {recs[safe_i]['discount']:.0f}% is the safer "
                f"call: it maximises the bad case rather than the average.")

        res.recommendation = Recommendation(
            headline=res.headline,
            rationale=[
                (f"Any discount from {band_lo:.0f}% to {band_hi:.0f}% gives the same expected "
                 f"profit within the model's own margin of error, so there is real room to "
                 f"negotiate there."
                 if not band["single"] else
                 f"Only {best_d:.0f}% stands out from the rest - the model cannot tell any "
                 f"neighbouring discount apart from it."),
                f"Expected profit peaks at {best_d:.0f}% "
                f"({H.money(best['expected_profit'], compact=True)}).",
                (f"The requested {req_disc:.0f}% is inside that range - it is a fine call, and it "
                 f"gives up only {H.money(abs(delta), compact=True)} against the peak."
                 if req_inside else
                 f"The requested {req_disc:.0f}% delivers "
                 f"{H.money(asked['expected_profit'], compact=True)} - "
                 f"{H.money(abs(delta), compact=True)} less than the peak, which is more than the "
                 f"model's margin of error."),
                (f"Win probability only improves {extra_win_pts:+.0f} points for that giveaway."
                 if abs(extra_win_pts) >= 0.5 else
                 "Win probability barely moves for that giveaway."),
            ],
            actions=actions,
            risk_score=round(risk, 1),
            risk_label=H.risk_label(risk),
            confidence="High" if curve_source != "your assumption" else "Medium",
            tradeoff=(
                f"The {req_disc:.0f}% being asked for gives up "
                f"{H.money(abs(delta), compact=True)} against the very best single number, but "
                f"that is inside the model's own margin of error - it is the same answer, not a "
                f"real loss."
                if req_inside else
                f"You are trading {H.money(abs(delta), compact=True)} of expected profit for "
                f"{extra_win_pts:+.0f} points of win probability."),
        )

        res.assumptions = [
            # QA NEW-09: state the margin and the cost on the *same* basis. The
            # old line paired the file's standard margin with a cost that had the
            # consumption extras folded in, so the two disagreed by ~3 points.
            (f"List value {H.money(tot_value)}; cost to deliver {H.money(goods_cost)} at the "
             f"{H.pct(std_margin)} standard margin your file implies."
             if extra_cost <= 1.0 else
             f"List value {H.money(tot_value)}; cost to deliver {H.money(goods_cost)} at the "
             f"{H.pct(std_margin)} standard margin your file implies, plus "
             f"{H.money(extra_cost)} of consumption effects on top, giving a total cost base of "
             f"{H.money(cost_total)} and an effective {H.pct(effective_margin)} margin once "
             f"those effects are included."),
            # QA TM-23 - state the consumption effects either way so they are
            # auditable rather than silently absent.
            consumption_note,
            f"Win probability curve: from {H.pct(pf)} at full price toward a ceiling of "
            f"{H.pct(pc)}, halfway at {d50:.1f}% discount (source: {curve_source}).",
            f"Win probability uncertainty: plus or minus {sd_win * 100:.0f} points.",
            f"Cost volatility: plus or minus {cv_cost * 100:.0f}% of cost.",
            f"Margin floor: {H.pct(floor_pct)}.",
            f"Two discounts are treated as giving the same answer when the gap between their "
            f"expected profits is smaller than twice the model's own margin of error at both "
            f"discounts added together (about {H.money(band_tol, compact=True)} at the best one), "
            f"or smaller than 1% of the best profit - whichever is larger.",
            f"The range where the answer is the same is {band_phrase}, taken from the "
            f"{len(grid)} discount levels we tested.",
            f"{n_sim:,} simulations per discount level; seed {ctx.request.seed} "
            f"(re-running gives identical numbers).",
        ]

        if negative_margin:
            res.warnings.append(
                f"Your history runs at a negative average margin ({H.pct(std_margin)}): on "
                f"average each deal costs more to deliver than it sells for. Every discount "
                f"therefore loses money, and the result shows the least-bad option rather than "
                f"a good one. Treat this as a problem with the deal's cost or price, not with "
                f"the discount.")
        if abs(std_margin) > 1.0:
            res.warnings.append(
                f"The average margin works out at {H.pct(std_margin)}, which is not a margin "
                f"any business can have. Check that your margin column holds a percentage "
                f"(40 means 40%) and not a ratio or a cost.")
        if band["covers_all"]:
            res.warnings.append(
                "Every discount we tested is within the model's own margin of error of the best, "
                "so the model cannot single one out - treat this as a judgement call, not a "
                "calculation.")
        if band["extra_inside"]:
            res.warnings.append(
                f"Some discounts outside the {band_phrase} range are also within the model's "
                f"margin of error; the range shown is the unbroken run around the best discount.")
        if asked["prob_loss"] > 0.05:
            res.warnings.append(
                f"At {req_disc:.0f}% discount there is a {asked['prob_loss'] * 100:.0f}% chance the "
                f"deal actually costs you money to deliver even if you win it.")
        if not has_data:
            res.warnings.append(
                "No historical deal file was mapped, so the buyer-response curve is your "
                "assumption. Upload past deals to learn it from evidence.")

        ctx.narrator.done(f"Done. Best discount is {best_d:.0f}% "
                          f"({H.money(best['expected_profit'], compact=True)} expected profit); "
                          f"the range where the answer is the same is {band_phrase}.")
        res.log = ctx.narrator.events()
        _self_check(res, recs, grid, band)
        return res
