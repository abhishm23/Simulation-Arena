"""USE CASE 2 - Discount Leakage & Cherry-Picking: The Basket Sabotage.

Model: cluster bootstrap resampling + empirical joint probabilities.

The question a spreadsheet cannot answer: the buyer wants 35% off the software
and *promises* to keep the full-price hardware and the three-year support
contract that pays for it. A spreadsheet prices the whole basket and says the
deal is fine. This test asks a different question: in past deals where we gave a
discount this deep, how often did the offsetting lines quietly disappear - and
what does that do to the profit we actually bank?
"""
from __future__ import annotations

import math

import numpy as np

from ..core.contracts import ParamSpec, Recommendation, RoleSpec, UseCaseMeta
from ..core.registry import UseCase, register
from ..engine import bootstrap as bs
from ..engine import stats as st
from . import _helpers as H

#: The bundle rebuild holds an index matrix of `resamples x units`. These two
#: numbers bound it (QA NEW-04): the pool of units never exceeds
#: `_BOOTSTRAP_MAX_UNITS`, and `_BOOTSTRAP_MAX_CELLS` caps the matrix whatever
#: the user asks for, so asking for 20,000 rebuilds cannot allocate gigabytes.
_BOOTSTRAP_MAX_UNITS = 10_000
_BOOTSTRAP_MAX_CELLS = 40_000_000

META = UseCaseMeta(
    id="basket_sabotage",
    title="Discount Leakage & Cherry-Picking: The Basket Sabotage",
    subtitle="Will they keep the discount and quietly drop the rest of the basket?",
    category="Deal Desk",
    icon="\U0001F9FA",
    models=["Bootstrap resampling", "Empirical joint probability"],
    business_question=(
        "If we give the deep discount on the headline line, how likely is the buyer to "
        "quietly drop the profitable lines that were supposed to pay for it?"
    ),
    plain_summary=(
        "Sophisticated buyers ask for a bundle discount, promise to buy the rest of the "
        "basket at full price, then quietly drop or delay the parts that carry your "
        "margin. We go back through thousands of your past multi-line deals, find the "
        "ones that looked like this one, and measure how often the promise was kept. "
        "Then we rebuild this deal thousands of times over to show what it actually earns."
    ),
    story_steps=[
        "Read the quote line by line and find where the profit really sits",
        "Look up what happened in past deals that got a discount this deep",
        "Measure how strongly a deep discount on one line is linked to another line dying",
        "Rebuild this bundle thousands of times and price the damage",
        "Write the contract clause that stops the leak",
    ],
    data_granularity="line_item",
    sample_dataset="quote_lines.csv",
    input_roles=[
        RoleSpec(key="quote_id", label="Quote / deal ID", dtype="identifier", required=False,
                 aliases=["quote", "quote_number", "opportunity_id", "deal_id", "order_id"],
                  help="Groups the lines that belong to the same deal. Needed to resample "
                      "whole deals rather than loose lines."),
        RoleSpec(key="customer_tier", label="Customer tier / segment", dtype="categorical",
                 required=False,
                 aliases=["tier", "segment", "customer_segment", "account_tier"],
                  help="Lets us check whether Enterprise buyers behave differently."),
        RoleSpec(key="line_id", label="Line item ID", dtype="identifier", required=False,
                 aliases=["line", "line_number", "item", "sku"],
                  help="Optional - only used to label the lines in the report."),
        RoleSpec(key="product_line", label="Product line name", dtype="categorical",
                 required=False,
                 aliases=["product", "item_name", "description", "sku_name"],
                  help="Used as the category if you have no category column."),
        RoleSpec(key="category", label="Line category", dtype="categorical", required=False,
                 aliases=["product_category", "family", "bucket", "type", "line_type"],
                  help="e.g. Software / Hardware / Support / Services."),
        RoleSpec(key="list_price", label="List price of the line", dtype="numeric",
                 required=False,
                 aliases=["price", "amount", "value", "list_value", "extended_price"]),
        RoleSpec(key="cost", label="Cost of the line", dtype="numeric", required=False,
                 aliases=["cogs", "unit_cost", "cost_to_serve", "total_cost"]),
        RoleSpec(key="discount_pct", label="Discount on the line (%)", dtype="numeric",
                 required=False,
                 aliases=["discount", "discount_percent", "off", "concession"],
                  help="Use 35 for 35% or 0.35 for 35% - both work."),
        RoleSpec(key="cancelled_within_6m", label="Was the line cancelled? (1/0)",
                 dtype="numeric", required=False,
                 aliases=["cancelled", "canceled", "churned", "dropped", "is_cancelled"],
                  help="1 if the line was cancelled or dropped, 0 if it survived."),
        RoleSpec(key="months_to_cancel", label="Months until cancellation", dtype="numeric",
                 required=False,
                 aliases=["months_to_churn", "cancel_month", "tenure_months"],
                  help="Optional. Lets us re-cut 'cancelled' to a shorter horizon."),
    ],
    parameters=[
        # ---- the quote being reviewed -------------------------------------
        ParamSpec(key="software_list_price", label="Software list price", kind="number",
                  default=600_000, min=0, step=10_000, unit="$", group="The quote",
                  help="The line the buyer wants discounted."),
        ParamSpec(key="hardware_list_price", label="Hardware list price", kind="number",
                  default=300_000, min=0, step=10_000, unit="$", group="The quote", help="The list price of the hardware line, before any discount.",),
        ParamSpec(key="support_list_price", label="Support list price", kind="number",
                  default=250_000, min=0, step=10_000, unit="$", group="The quote", help="The list price of the support line, before any discount.",),
        ParamSpec(key="software_discount_pct", label="Software discount", kind="number",
                  default=35, min=0, max=90, step=1, unit="%", group="The quote",
                  help="The discount the buyer is asking for on the headline line."),
        ParamSpec(key="hardware_discount_pct", label="Hardware discount", kind="number",
                  default=5, min=0, max=90, step=1, unit="%", group="The quote", help="The discount the customer is asking for on hardware.",),
        ParamSpec(key="support_discount_pct", label="Support discount", kind="number",
                  default=10, min=0, max=90, step=1, unit="%", group="The quote", help="The discount the customer is asking for on the support line.",),
        ParamSpec(key="software_margin_pct", label="Software margin", kind="number",
                  default=85, min=0, max=99, step=1, unit="%", group="The quote",
                  help="Margin on this line before any discount."),
        ParamSpec(key="hardware_margin_pct", label="Hardware margin", kind="number",
                  default=18, min=0, max=99, step=1, unit="%", group="The quote", help="Your margin on hardware at list price.",),
        ParamSpec(key="support_margin_pct", label="Support margin", kind="number",
                  default=70, min=0, max=99, step=1, unit="%", group="The quote", help="Your margin on support at list price.",),
        # ---- how we judge risk --------------------------------------------
        ParamSpec(key="deep_discount_threshold_pct", label="What counts as a deep discount",
                  kind="number", default=30, min=0, max=90, step=1, unit="%",
                  group="How we judge risk",
                  help="Discounts above this are the ones we study."),
        ParamSpec(key="horizon_months", label="How long we watch the deal", kind="number",
                  default=6, min=1, max=36, step=1, unit="months",
                  group="How we judge risk",
                  help="A line that dies inside this window counts as abandoned."),
        ParamSpec(key="abandon_base_rate_pct", label="Normal abandonment rate",
                  kind="number", default=22, min=0, max=100, step=1, unit="%",
                  group="How we judge risk",
                  help="Only used when you bring no history: how often a line is "
                       "dropped anyway."),
        ParamSpec(key="deep_discount_abandon_rate_pct",
                  label="Abandonment after a deep discount", kind="number", default=64,
                  min=0, max=100, step=1, unit="%", group="How we judge risk",
                  help="Only used when you bring no history: how often the offsetting "
                       "line dies once a deep discount is given."),
        # ---- guardrails ----------------------------------------------------
        ParamSpec(key="bundle_margin_floor_pct", label="Bundle margin floor", kind="number",
                  default=35, min=0, max=95, step=1, unit="%", group="Guardrails",
                  help="The blended margin below which the bundle is not worth signing."),
        ParamSpec(key="min_support_attach_pct", label="Minimum support commitment",
                  kind="number", default=80, min=0, max=100, step=5, unit="%",
                  group="Guardrails",
                  help="Share of committed support hours that must still be live."),
        # ---- engine --------------------------------------------------------
        ParamSpec(key="n_resamples", label="Bundles to rebuild", kind="integer",
                  default=2000, min=200, max=20000, step=200, group="Engine",
                  help="How many times we replay your own past deals and re-price this "
                       "one. More replays = a steadier answer."),
    ],
)


# ---------------------------------------------------------------------------
# small private helpers (nothing here belongs in the shared _helpers module)
# ---------------------------------------------------------------------------
def _frac(v, default_pct: float) -> float:
    """Accept 18 or 0.18 and always return a fraction."""
    f = H.as_float(v, default_pct)
    return f / 100.0 if f > 1.0 else f


def _match_category(name: str, categories) -> str | None:
    """Best-effort match of a target line name ('Software') onto a history
    category ('software', 'Software License', ...). None when nothing fits."""
    if not categories:
        return None
    target = str(name).strip().lower()
    for c in categories:                                  # exact first
        if str(c).strip().lower() == target:
            return c
    for c in categories:                                  # then containment
        cl = str(c).strip().lower()
        if target in cl or cl in target:
            return c
    return None


def _safe_div(a, b, default: float = 0.0) -> float:
    return float(a) / float(b) if float(b) else default


@register
class BasketSabotageUseCase(UseCase):
    meta = META

    def run(self, req, ctx) -> "H.RunResult":
        P = ctx.params
        rows = ctx.rows or []

        # ===================================================================
        # 1. READ THE QUOTE UNDER REVIEW
        # ===================================================================
        ctx.say("Reading the quote you want us to stress-test...", phase="read",
                progress=0.04)

        threshold_pct = 100.0 * _frac(H.p(P, "deep_discount_threshold_pct", 30), 30.0)
        threshold = threshold_pct / 100.0
        horizon = max(1, int(H.as_float(H.p(P, "horizon_months", 6))))
        floor_pct = 100.0 * _frac(H.p(P, "bundle_margin_floor_pct", 35), 35.0)
        floor_frac = floor_pct / 100.0
        attach = 100.0 * _frac(H.p(P, "min_support_attach_pct", 80), 80.0)
        n_res = int(H.as_float(H.p(P, "n_resamples", 2000)))
        n_res = int(min(max(n_res, 100), 20000))

        base_rate = _frac(H.p(P, "abandon_base_rate_pct", 22), 22.0)
        deep_rate = _frac(H.p(P, "deep_discount_abandon_rate_pct", 64), 64.0)

        spec = [
            ("Software", H.p(P, "software_list_price", 600_000),
             H.p(P, "software_discount_pct", 35), H.p(P, "software_margin_pct", 85)),
            ("Hardware", H.p(P, "hardware_list_price", 300_000),
             H.p(P, "hardware_discount_pct", 5), H.p(P, "hardware_margin_pct", 18)),
            ("Support", H.p(P, "support_list_price", 250_000),
             H.p(P, "support_discount_pct", 10), H.p(P, "support_margin_pct", 70)),
        ]
        lines = []
        for name, lp, d, m in spec:
            list_price = max(0.0, H.as_float(lp, 0.0))
            disc = _frac(d, 0.0)
            margin_frac = _frac(m, 0.0)
            cost = list_price * (1.0 - margin_frac)
            revenue = list_price * (1.0 - disc)
            lines.append({
                "name": name,
                "list": list_price,
                "disc": disc,
                "disc_pct": disc * 100.0,
                "margin_frac": margin_frac,
                "cost": cost,
                "revenue": revenue,
                "profit": revenue - cost,
            })
        lines = [ln for ln in lines if ln["list"] > 0] or lines

        profit_vec = np.array([ln["profit"] for ln in lines], dtype=float)
        revenue_vec = np.array([ln["revenue"] for ln in lines], dtype=float)
        disc_vec = np.array([ln["disc_pct"] for ln in lines], dtype=float)
        full_profit = float(profit_vec.sum())
        full_revenue = float(revenue_vec.sum())
        full_margin_pct = _safe_div(full_profit, full_revenue)

        # the "hero" line = the one carrying the deepest discount (the ask)
        hero_i = int(np.argmax(disc_vec))
        hero = lines[hero_i]
        hero_name = hero["name"]

        ctx.say(
            f"The basket is worth {H.money(full_revenue, compact=True)} and carries "
            f"{H.money(full_profit, compact=True)} of profit "
            f"({H.pct(full_margin_pct)} margin) if the buyer keeps every line.",
            phase="read", progress=0.10,
            detail="; ".join(f"{l['name']} {H.money(l['revenue'])} rev / "
                             f"{H.money(l['profit'])} profit" for l in lines))
        ctx.say(
            f"The line being discounted hardest is {hero_name} at "
            f"{hero['disc_pct']:.0f}% off - that is "
            f"{max(0.0, hero['disc_pct'] - threshold_pct):.0f} points past your "
            f"{threshold_pct:.0f}% deep-discount line.",
            phase="read", progress=0.14)

        # ===================================================================
        # 2. READ THE HISTORY
        # ===================================================================
        cat_role = "category" if H.has(rows, "category") else (
            "product_line" if H.has(rows, "product_line") else None)
        has_cancel = H.has(rows, "cancelled_within_6m")
        has_disc = H.has(rows, "discount_pct")
        has_quote = H.has(rows, "quote_id")
        has_tier = H.has(rows, "customer_tier")
        has_months = H.has(rows, "months_to_cancel")
        history_ready = bool(cat_role and has_cancel and has_disc and rows)

        hist: dict = {
            "ready": False, "cats": [], "units": 0, "quotes": 0,
            "by_cat_disc": {}, "by_cat_cancel": {}, "base": {},
            "cond": {}, "lift": {}, "phi": {}, "n_deep": {},
            "tiers": [], "avg_months": None, "whole": True,
        }

        if not rows:
            ctx.say("No past quote lines were supplied, so we will use the rates you "
                    "typed in instead of learning from evidence.", phase="read",
                    level="warn", progress=0.18)
        elif not history_ready:
            missing = []
            if not cat_role:
                missing.append("a line category")
            if not has_disc:
                missing.append("the discount given")
            if not has_cancel:
                missing.append("whether the line was cancelled")
            ctx.say(f"We could not learn from your file - it is missing "
                    f"{' and '.join(missing)}. We will use the rates you typed in.",
                    phase="read", level="warn", progress=0.18)
        else:
            cats_raw = H.cat(rows, cat_role, default="")
            disc_raw = H.num(rows, "discount_pct", 0.0)
            disc_raw = np.where(disc_raw > 1.0, disc_raw, disc_raw * 100.0)
            canc_raw = np.where(H.num(rows, "cancelled_within_6m", 0.0) > 0.5, 1.0, 0.0)
            if has_months:
                mtc = H.num(rows, "months_to_cancel", 0.0)
                within = (mtc > 0) & (mtc <= float(horizon))
                canc_raw = canc_raw * np.where(mtc > 0, within, 1.0)
                c_months = mtc[(canc_raw > 0.5) & (mtc > 0)]
                hist["avg_months"] = float(c_months.mean()) if c_months.size else None
                if horizon < 6:
                    ctx.say(f"Re-cutting 'cancelled' to your {horizon}-month window...",
                            phase="read", progress=0.16,
                            detail=f"{int((canc_raw > 0.5).sum())} lines inside the window")
            elif horizon > 6:
                ctx.say("Your file only records 6-month cancellations, so we cannot look "
                        f"further out than that - results use a 6-month window.",
                        phase="read", level="warn", progress=0.16)

            if has_quote:
                qids = np.array(H.cat(rows, "quote_id", default=""), dtype=object)
            else:
                # no quote id: every line becomes its own "deal"
                qids = np.array([f"row-{i}" for i in range(len(rows))], dtype=object)
                hist["whole"] = False

            uniq, inv = np.unique(qids, return_inverse=True)
            n_units_all = int(uniq.size)
            hist["units_in_file"] = n_units_all
            # QA NEW-04: the bootstrap draws `n_res` resamples of every unit and
            # holds the index matrix (resamples x units). On a two-million-row
            # file that matrix alone is tens of gigabytes, which is what took the
            # whole application down. The pool of units the resampling works on
            # is therefore bounded - statistically 10,000 units is far more than
            # a proportion needs - and the reduction is stated below.
            pool_n = _bootstrap_pool(n_units_all, n_res)
            keep_rows = None
            if 0 < pool_n < n_units_all:
                stride = int(math.ceil(n_units_all / float(pool_n)))
                keep_units = np.arange(0, n_units_all, stride)[:pool_n]
                remap = np.full(n_units_all, -1, dtype=np.int64)
                remap[keep_units] = np.arange(keep_units.size)
                keep_rows = np.isin(inv, keep_units)
                inv = remap[inv[keep_rows]]
                if isinstance(cats_raw, list):
                    cats_raw = [cats_raw[i] for i in np.nonzero(keep_rows)[0]]
                else:
                    cats_raw = cats_raw[keep_rows]
                disc_raw = disc_raw[keep_rows]
                canc_raw = canc_raw[keep_rows]
                uniq = uniq[keep_units]
                hist["pooled"] = True
            n_units = int(uniq.size)
            hist["units"] = n_units
            hist["quotes"] = n_units

            cats = []
            for c in cats_raw:
                c = str(c).strip()
                if c and c not in cats:
                    cats.append(c)
            cats = sorted(cats)[:8]
            hist["cats"] = cats

            # per-unit discount / cancellation per category
            for c in cats:
                d_u = np.zeros(n_units, dtype=float)
                k_u = np.zeros(n_units, dtype=bool)
                seen = np.zeros(n_units, dtype=bool)
                for i, cval in enumerate(cats_raw):
                    if str(cval).strip() != c:
                        continue
                    u = int(inv[i])
                    if not seen[u]:
                        d_u[u] = float(disc_raw[i])
                        seen[u] = True
                    else:
                        d_u[u] = 0.5 * (d_u[u] + float(disc_raw[i]))
                    k_u[u] = k_u[u] or bool(canc_raw[i] > 0.5)
                hist["by_cat_disc"][c] = d_u
                hist["by_cat_cancel"][c] = k_u
                hist["base"][c] = float(k_u.mean()) if n_units else 0.0

            # conditional matrix: A deeply discounted -> B cancelled
            for a in cats:
                deep = hist["by_cat_disc"][a] > threshold_pct
                hist["n_deep"][a] = int(deep.sum())
                for b in cats:
                    jp = bs.joint_probability(deep, hist["by_cat_cancel"][b])
                    hist["cond"][(a, b)] = float(jp["p_b_given_a"])
                    hist["lift"][(a, b)] = float(jp["lift"])
                    hist["phi"][(a, b)] = float(jp["phi"])

            hist["ready"] = True
            ctx.say(
                f"Learning from {n_units:,} past "
                f"{'deals' if hist['whole'] else 'line items'} "
                f"across {len(cats)} categories "
                f"({', '.join(cats)})...", phase="read", progress=0.20,
                detail=f"deep discount = anything above {threshold_pct:.0f}%")
            if hist.get("pooled"):
                # rule 8: if the model reduces the data it must say so, in the
                # run log and in the result (this line goes to both).
                ctx.say(
                    f"Your file holds {n_units_all:,} past "
                    f"{'deals' if hist['whole'] else 'line items'}. Rebuilding every one "
                    f"of them thousands of times needs more memory than a browser-class "
                    f"machine has, so the rebuilding works on an evenly spread pool of "
                    f"{n_units:,} of them ({100.0 * n_units / max(1, n_units_all):.1f}%). "
                    f"Every number below comes from that pool.",
                    phase="read", level="warn", progress=0.22)

            if has_tier:
                tiers_raw = np.array(H.cat(rows, "customer_tier", default=""), dtype=object)
                if keep_rows is not None:
                    tiers_raw = tiers_raw[keep_rows]
                tier_of_unit = np.array([""] * n_units, dtype=object)
                tier_of_unit[inv] = tiers_raw
                hist["tier_of_unit"] = tier_of_unit

        # ===================================================================
        # 3. WHAT HISTORY SAYS ABOUT THIS EXACT QUOTE
        # ===================================================================
        ctx.say("Checking what happened in past deals that got a discount this deep...",
                phase="prepare", progress=0.26)

        hero_cat = _match_category(hero_name, hist["cats"]) if hist["ready"] else None
        line_cats: list[str | None] = []
        for ln in lines:
            line_cats.append(_match_category(ln["name"], hist["cats"])
                             if hist["ready"] else None)

        # A joint probability only means something if the lines were quoted
        # together, so without a quote id we cannot measure the cherry-pick lift.
        empirical = bool(hist["ready"] and hero_cat is not None and hist["whole"]
                         and hist["n_deep"].get(hero_cat, 0) >= 10)
        source = "your own history" if empirical else "the rates you typed in"

        # the two rates only separate once the ask actually crosses your limit
        hero_deep = bool(disc_vec[hero_i] > threshold_pct + 1e-9)

        p_line = np.zeros(len(lines), dtype=float)
        if empirical:
            pool_rate = float(np.mean([hist["base"][k] for k in hist["cats"]])) \
                if hist["cats"] else base_rate
            for i, c in enumerate(line_cats):
                if c is None:
                    p_line[i] = pool_rate
                elif hero_deep:
                    p_line[i] = hist["cond"].get((hero_cat, c), pool_rate)
                else:
                    p_line[i] = hist["base"].get(c, pool_rate)
        else:
            # No joint probability available: fall back to the ordinary rate for
            # the category from your file where we have one, otherwise the rate
            # you typed in; the deep-discount rate only applies once the ask
            # actually crosses your limit.
            for i, ln in enumerate(lines):
                c = line_cats[i]
                observed = hist["base"].get(c) if (hist["ready"] and c is not None) else None
                ordinary = base_rate if observed is None else float(observed)
                p_line[i] = ordinary if (i == hero_i or not hero_deep) else deep_rate

        # strongest off-diagonal pair (the classic cherry-pick signature)
        strongest = None
        if hist["ready"] and hist["whole"]:
            best = None
            for a in hist["cats"]:
                for b in hist["cats"]:
                    if a == b:
                        continue
                    if hist["n_deep"].get(a, 0) < 10:
                        continue
                    lf = hist["lift"][(a, b)]
                    if lf <= 1.0:
                        continue
                    if best is None or lf > best[2]:
                        best = (a, b, lf)
            if best is not None:
                strongest = {
                    "a": best[0], "b": best[1], "lift": best[2],
                    "cond": hist["cond"][(best[0], best[1])],
                    "base": hist["base"][best[1]],
                    "phi": hist["phi"][(best[0], best[1])],
                    "n": hist["n_deep"][best[0]],
                }
                ctx.say(
                    f"Found it: when {best[0]} is discounted past "
                    f"{threshold_pct:.0f}%, the {best[1]} line is cancelled "
                    f"{H.pct(strongest['cond'])} of the time - "
                    f"{strongest['lift']:.1f}x its normal rate.",
                    phase="prepare", progress=0.34,
                    detail=f"conditional {strongest['cond']:.3f} vs baseline "
                           f"{strongest['base']:.3f}, phi={strongest['phi']:.3f}, "
                           f"n={strongest['n']}")
            else:
                ctx.say("No category pair in your history shows a clear "
                        "discount-then-cancel pattern - we will fall back on your "
                        "assumptions.", phase="prepare", level="warn", progress=0.34)

        offset_idx = [i for i in range(len(lines)) if i != hero_i]
        # the offsetting line we headline is the one with the most profit exposed
        lead_i = offset_idx[int(np.argmax([profit_vec[i] * p_line[i] for i in offset_idx]))] \
            if offset_idx else hero_i
        lead_offset = lines[lead_i]["name"]
        lead_p = float(p_line[lead_i])
        # the "normal rate" we compare against comes from history when we have it
        lead_baseline = (hist["base"].get(line_cats[lead_i], base_rate)
                         if empirical and line_cats[lead_i] is not None else base_rate)
        p_any_offset = 1.0 - float(np.prod([1.0 - p_line[i] for i in offset_idx])) \
            if offset_idx else 0.0
        # with only one line on the quote there is no basket to sabotage
        no_offset = not offset_idx
        if no_offset:
            ctx.say("This quote has only one line, so there is no offsetting line for a "
                    "buyer to drop - there is nothing here to cherry-pick.",
                    phase="prepare", level="warn", progress=0.40)
        if empirical:
            deep_mask = hist["by_cat_disc"][hero_cat] > threshold_pct
            any_off = np.zeros(hist["units"], dtype=bool)
            for i in offset_idx:
                c = line_cats[i]
                if c is not None:
                    any_off = any_off | hist["by_cat_cancel"][c]
            if deep_mask.sum() >= 10:
                p_any_offset = float(any_off[deep_mask].mean())

        ctx.say(f"Chance the offsetting lines disappear: {H.pct(p_any_offset)} - "
                f"that is what the discount is really buying.", phase="prepare",
                progress=0.40, detail=f"source: {source}")

        # ===================================================================
        # 4. SIMULATE: REBUILD THIS BUNDLE THOUSANDS OF TIMES
        # ===================================================================
        rng = ctx.rng("basket_sabotage", "cluster")
        ctx.say(f"Rebuilding {n_res:,} bundles that look like this one and re-pricing "
                f"each of them...", phase="simulate", progress=0.45,
                detail="each bundle re-draws past deals, so the answer carries the "
                       "uncertainty of how much history we have")

        p_mat = np.zeros((n_res, len(lines)), dtype=float)
        if empirical:
            idx = _cluster_indices(hist, n_res, rng)
            deep = hist["by_cat_disc"][hero_cat][idx] > threshold_pct   # (R, units)
            den = deep.sum(axis=1)
            for i, c in enumerate(line_cats):
                if c is None:
                    p_mat[:, i] = base_rate
                    continue
                num = (deep & hist["by_cat_cancel"][c][idx]).sum(axis=1)
                p_mat[:, i] = np.where(den > 0, num / np.maximum(den, 1.0), p_line[i])
            for i, c in enumerate(line_cats):
                if c is None:
                    continue
                fallback = p_line[i]
                p_mat[:, i] = np.where(den >= 5, p_mat[:, i], fallback)
        else:
            # no history: draw the rate itself from a spread around your assumption
            n_eff = 25.0
            for i in range(len(lines)):
                p0 = float(np.clip(p_line[i], 1e-4, 1 - 1e-4))
                a = p0 * n_eff
                b = (1.0 - p0) * n_eff
                p_mat[:, i] = np.clip(rng.beta(a, b, n_res), 0.0, 1.0)

        tick = ctx.narrator.tracker(n_res, "Rebuilt {pct}% of the bundles...",
                                    phase="simulate")
        for k in range(0, n_res, max(1, n_res // 10)):
            tick(k)

        draw_rng = ctx.rng("basket_sabotage", "outcome")
        dropped = draw_rng.random((n_res, len(lines))) < p_mat
        keep = 1.0 - dropped
        realised_profit = (keep * profit_vec[None, :]).sum(axis=1)
        realised_revenue = (keep * revenue_vec[None, :]).sum(axis=1)
        realised_pct = np.where(realised_revenue > 0,
                                realised_profit / np.maximum(realised_revenue, 1e-9), 0.0)

        s_profit = st.summarize(realised_profit)
        expected_profit = float(s_profit.get("mean", full_profit))
        bad_case = float(s_profit.get("p5", expected_profit))
        worst_case = float(s_profit.get("min", expected_profit))
        # QA NEW-12: the row labelled "Most likely outcome" was showing the
        # *mean*. When a handful of catastrophic runs drag the average down the
        # mean can sit BELOW the 1-in-20 bad case, so the table read "Most
        # likely outcome $481k" directly above "A bad run (1 in 20) $489k" - a
        # bad run that is better than the likely one. The most likely outcome is
        # the middle of the distribution, and the middle is never below the 5th
        # percentile. The average is still the right number to plan on, so it
        # stays where it belongs - on the "plan for" line.
        likely_case = float(s_profit.get("p50", expected_profit))
        bad_case = min(bad_case, likely_case)
        leakage = max(0.0, full_profit - expected_profit)
        leakage_pct = _safe_div(leakage, full_profit)
        p_below_floor = float((realised_pct < floor_frac).mean())

        ctx.say("All bundles priced. Working out what this deal really earns...",
                phase="summarize", progress=0.92)

        # ===================================================================
        # 5. CHERRY-PICK RISK SCORE
        # ===================================================================
        # three transparent pieces, each 0-100, fixed weights:
        #   25% how far past your own deep-discount line the ask sits
        #   50% how likely the offsetting lines are to disappear
        #   25% how much of the bundle's profit sits in the line being cut
        hero_share = _safe_div(profit_vec[hero_i], full_profit)
        c_depth = 100.0 * float(np.clip(max(0.0, disc_vec[hero_i] - threshold_pct) / 10.0,
                                        0.0, 1.0))
        c_abandon = 100.0 * float(np.clip(p_any_offset, 0.0, 1.0))
        c_conc = 100.0 * float(np.clip(hero_share, 0.0, 1.0))
        if no_offset:
            # nothing to cherry-pick against: score the quote as it stands, which
            # is just the ordinary risk of the single line dying
            score = 0.0
        else:
            score = float(np.clip(0.25 * c_depth + 0.50 * c_abandon + 0.25 * c_conc,
                                  0.0, 100.0))
        label = H.risk_label(score)
        drivers = [("how far past your discount limit the ask sits", 0.25 * c_depth),
                   ("how likely the offsetting lines are to vanish", 0.50 * c_abandon),
                   ("how much of the profit sits in the discounted line", 0.25 * c_conc)]
        drivers.sort(key=lambda d: -d[1])
        dominant = drivers[0][0]

        # ===================================================================
        # 6. ASSEMBLE THE ANSWER
        # ===================================================================
        res = H.new_result(ctx, META.id, META.title,
                           "Cluster bootstrap resampling with empirical joint probabilities",
                           n_res)

        offset_names = ", ".join(lines[i]["name"] for i in offset_idx) or "the other lines"
        offset_phrase = (f"the {lead_offset} line" if len(offset_idx) <= 1 else
                         f"the {lead_offset} line or another offsetting line")
        # The expected shortfall (`leakage`) is profit lost to EVERY line that can
        # drop, not only the offsetting ones. When the history shows the
        # offsetting lines never went at the same time as a discount this deep,
        # p_any_offset rounds to "0.0%" - and saying a zero chance "costs" money
        # is a contradiction the reader cannot resolve. Below this threshold we
        # say plainly that the discount bought nothing on those lines, and name
        # where the shortfall actually comes from.
        offset_signal = p_any_offset >= 0.005
        if no_offset:
            res.headline = (
                f"Cherry-pick risk {score:.0f}/100 ({label}): this quote has no second line "
                f"for the buyer to drop, so there is nothing to cherry-pick - the only risk "
                f"is the {hero_name} line itself dying "
                f"({H.pct(float(p_line[hero_i]))} ordinary rate), worth "
                f"{H.money(leakage, compact=True)}."
            )
        elif hero_deep and offset_signal:
            res.headline = (
                f"Cherry-pick risk {score:.0f}/100 ({label}): give this discount and there is "
                f"a {H.pct(p_any_offset)} chance {offset_phrase} disappears, costing "
                f"{H.money(leakage, compact=True)} - {H.pct(leakage_pct)} of the deal's "
                f"profit."
            )
        elif hero_deep:
            res.headline = (
                f"Cherry-pick risk {score:.0f}/100 ({label}): the deep discount here buys "
                f"nothing on {offset_phrase} - in the history we have, that line never went "
                f"missing on a deal discounted this deeply. The "
                f"{H.money(leakage, compact=True)} of expected profit at risk "
                f"({H.pct(leakage_pct)} of the deal) is the ordinary chance of a line dying, "
                f"not cherry-picking."
            )
        elif offset_signal:
            res.headline = (
                f"Cherry-pick risk {score:.0f}/100 ({label}): the deepest ask here is inside "
                f"your {threshold_pct:.0f}% limit, so there is no cherry-pick signal - but "
                f"there is still a {H.pct(p_any_offset)} ordinary chance {offset_phrase} "
                f"disappears, costing {H.money(leakage, compact=True)}."
            )
        else:
            res.headline = (
                f"Cherry-pick risk {score:.0f}/100 ({label}): the deepest ask here is inside "
                f"your {threshold_pct:.0f}% limit, and in the history we have "
                f"{offset_phrase} never went missing on a deal like this, so there is no "
                f"cherry-pick signal at all. The {H.money(leakage, compact=True)} of expected "
                f"profit at risk ({H.pct(leakage_pct)} of the deal) is the ordinary chance of "
                f"a line dying."
            )
        res.method_plain = (
            f"We went back through your past multi-line deals and found the ones where a "
            f"line was discounted past {threshold_pct:.0f}%. In those deals we measured how "
            f"often the other lines were cancelled inside {horizon} months, and compared it "
            f"with how often those lines die anyway. That difference is the cherry-pick "
            f"signal. We then rebuilt this quote {n_res:,} times, each "
            f"time re-drawing whole past deals so the answer reflects how much history we "
            f"actually have, and re-priced the basket with whatever survived. The risk score "
            f"is three things you can check by hand: how far past your discount limit the "
            f"ask sits (25%), how likely the offsetting lines are to vanish (50%), and how "
            f"much of the profit sits in the line being cut (25%)."
        )

        # ---- KPIs ---------------------------------------------------------
        res.kpis = [
            H.kpi("Cherry-pick risk score", f"{score:.0f} / 100",
                  "bad" if score >= 75 else ("warn" if score >= 50 else "good"),
                  f"Built from three checkable pieces; the biggest one is {dominant}.",
                  comparison=f"{label} risk"),
            H.kpi("Profit likely to leak away", H.money(leakage, compact=True),
                  "bad" if leakage > 0 else "good",
                  "The profit this deal loses once you allow for how buyers actually behave.",
                  comparison=f"{H.pct(leakage_pct)} of the deal's profit"),
            H.kpi(f"Chance the {lead_offset} line is dropped", H.pct(lead_p),
                  "bad", f"How often the line that pays for the discount dies in the first "
                         f"{horizon} months.",
                  comparison=f"{st.lift(lead_p, lead_baseline):.1f}x "
                             f"the normal {H.pct(lead_baseline)}"),
            H.kpi("Chance any offsetting line dies", H.pct(p_any_offset), "bad",
                  "The odds the basket does not survive intact."),
            H.kpi(f"Profit exposed on the {lead_offset} line",
                  H.money(profit_vec[lead_i] * lead_p, compact=True), "bad",
                  f"{H.money(profit_vec[lead_i], compact=True)} of profit with a "
                  f"{H.pct(lead_p)} chance of walking."),
            H.kpi("Profit at risk", H.pct(leakage_pct), "warn",
                  "Share of the bundle's profit that is exposed to cherry-picking."),
            H.kpi("Profit if they keep everything", H.money(full_profit, compact=True),
                  "good", f"The spreadsheet number: {H.pct(full_margin_pct)} margin on "
                          f"{H.money(full_revenue, compact=True)}."),
            H.kpi("Profit you should actually plan for", H.money(expected_profit, compact=True),
                  "warn", "The honest number after cherry-picking.",
                  comparison=f"as low as "
                             f"{H.money(min(bad_case, expected_profit), compact=True)} "
                             f"in a bad run"),
        ]
        if empirical and strongest is not None:
            res.kpis.append(H.kpi(
                "How much more often the worst pair dies",
                f"{strongest['lift']:.1f}x", "bad",
                f"Deals with a deep {strongest['a']} discount lose the "
                f"{strongest['b']} line {strongest['lift']:.1f} times more often than usual.",
                comparison=f"{H.pct(strongest['cond'])} vs {H.pct(strongest['base'])}"))
        if p_below_floor > 0.005:
            res.kpis.append(H.kpi(
                f"Chance the bundle ends under your {floor_pct:.0f}% floor",
                H.pct(p_below_floor), "bad" if p_below_floor > 0.2 else "warn",
                "How often the surviving basket is not worth signing at all."))

        # ---- charts -------------------------------------------------------
        res.charts = [_discount_band_chart(hist, lines, line_cats, hero_cat, hero_i,
                                           threshold_pct, base_rate, deep_rate,
                                           empirical, offset_idx),
                      _heatmap_chart(hist, threshold_pct, base_rate, deep_rate,
                                     lines, line_cats),
                      _bundle_hist_chart(realised_profit, full_profit, n_res),
                      _line_margin_chart(lines, profit_vec, p_line)]

        # ---- tables -------------------------------------------------------
        res.tables = [
            H.table(
                "The quote, line by line",
                "Where the profit sits, and how exposed each line is.",
                ["Line", "List price", "Discount", "Price after discount", "Your margin",
                 "Profit", "Share of profit", "Chance it is dropped", "Profit at risk"],
                [[ln["name"], H.money(ln["list"]), H.pct_num(ln["disc_pct"]),
                  H.money(ln["revenue"]), H.pct(ln["margin_frac"]), H.money(ln["profit"]),
                  H.pct(_safe_div(ln["profit"], full_profit)),
                  H.pct(p_line[i]), H.money(ln["profit"] * p_line[i])]
                 for i, ln in enumerate(lines)],
                highlight=[hero_i],
                note=f"Row {hero_i + 1} carries the discount being asked for."),
            H.table(
                "What happens to each line when another line is discounted past "
                f"{threshold_pct:.0f}%",
                "Read across a row: 'when THIS line is deeply discounted, the chance THAT "
                "line dies'. A figure above 1.0 means the two events travel together.",
                ["Discounted line (A)", "Line that dies (B)", "Deals like this",
                 "Normal rate", "Rate after the discount", "Times more often", "Link strength"],
                _matrix_rows(hist, threshold_pct, base_rate, deep_rate, lines, line_cats,
                             empirical),
                highlight=_matrix_highlight(hist, lines, line_cats),
                note="Rows where the second line dies far more often than normal are the "
                     "cherry-pick signature." if empirical else
                     "No history was available - these are your assumptions, not evidence."),
            H.table(
                "What the bundle actually earns",
                f"Every one of the {n_res:,} rebuilt bundles, summarised.",
                ["Scenario", "Profit", "Margin", "Versus the plan"],
                [
                    ["If they keep everything", H.money(full_profit), H.pct(full_margin_pct),
                     "the number on the quote"],
                    ["Most likely outcome", H.money(likely_case),
                     H.pct(_safe_div(likely_case, full_revenue)),
                     H.money(likely_case - full_profit)],
                    ["A bad run (1 in 20)", H.money(bad_case),
                     H.pct(_safe_div(bad_case, full_revenue)), H.money(bad_case - full_profit)],
                    ["Worst bundle we saw", H.money(worst_case),
                     H.pct(_safe_div(worst_case, full_revenue)),
                     H.money(worst_case - full_profit)],
                ],
                highlight=[1], note="Row 2 is the number to plan with."),
        ]

        tier_tbl = _tier_table(hist, threshold_pct, line_cats, offset_idx, hero_cat)
        if tier_tbl is not None:
            res.tables.append(tier_tbl)

        # ---- story --------------------------------------------------------
        timing = ""
        if hist.get("avg_months"):
            timing = (f" And they do not do it immediately - when a line dies it happens "
                      f"around month {hist['avg_months']:.0f}.")
        res.story = [
            H.story(
                "What is actually on the table",
                f"The basket is worth {H.money(full_revenue, compact=True)} and makes "
                f"{H.money(full_profit, compact=True)} if nothing is dropped. But "
                f"{H.pct(hero_share)} of that profit sits in {hero_name} - the exact line "
                f"the buyer wants {hero['disc_pct']:.0f}% off, which is "
                f"{max(0.0, hero['disc_pct'] - threshold_pct):.0f} points past the discount "
                f"line you set yourself.",
                f"{H.money(hero['profit'], compact=True)} of "
                f"{H.money(full_profit, compact=True)} profit in {hero_name}"),
            H.story(
                "What history says happens next",
                (f"In {strongest['n']:,} past deals where {strongest['a']} was discounted "
                 f"past {threshold_pct:.0f}%, the {strongest['b']} line was cancelled "
                 f"{H.pct(strongest['cond'])} of the time against a normal rate of "
                 f"{H.pct(strongest['base'])} - {strongest['lift']:.1f} times more often."
                 if (empirical and strongest) else
                 f"You gave us no history to learn from, so we used your own numbers: "
                 f"{H.pct(base_rate)} of lines die anyway, and "
                 f"{H.pct(deep_rate)} die once a deep discount is given.") + timing,
                f"source: {source}"),
            H.story(
                "What the rebuild shows",
                f"Rebuilding this quote {n_res:,} times, the profit lands at "
                f"{H.money(expected_profit, compact=True)} on average, and in an unlucky "
                f"run of twenty it is "
                f"{H.money(min(bad_case, expected_profit), compact=True)} or worse. "
                f"There is a "
                f"{H.pct(p_below_floor)} chance the bundle that survives is worth less than "
                f"your {floor_pct:.0f}% margin floor.",
                f"{H.money(leakage, compact=True)} average shortfall"),
            H.story(
                "What to do about it",
                (f"The biggest driver of the score is {dominant}. Tie the discount to the "
                 f"behaviour: if the {lead_offset} line is not live at {attach:.0f}% of "
                 f"committed hours by month {horizon}, the {hero['disc_pct']:.0f}% "
                 f"{hero_name} discount comes off and {hero_name} reverts to standard "
                 f"pricing - worth up to "
                 f"{H.money(hero['list'] * hero['disc'], compact=True)} back."
                 if hero_deep else
                 f"The ask is inside your limit, so there is no cherry-pick to defend "
                 f"against - the {H.pct(p_any_offset)} chance of losing an offsetting line "
                 f"is just your ordinary run rate. Price the {lead_offset} line as part of "
                 f"one bundle with a single cancellation date so it cannot be peeled off "
                 f"later."),
                f"clawback up to {H.money(hero['list'] * hero['disc'], compact=True)}"
                if hero_deep else f"ordinary drop rate {H.pct(base_rate)}"),
        ]

        # ---- recommendation -----------------------------------------------
        clawback = hero["list"] * hero["disc"]
        soft_clawback = hero["list"] * max(0.0, hero["disc"] - threshold)
        clause = (
            f"Make the {hero['disc_pct']:.0f}% {hero_name} discount conditional: if the "
            f"{lead_offset} line is not live at {attach:.0f}% of committed hours by month "
            f"{horizon}, {hero_name} pricing reverts to the standard tier "
            f"(recovering up to {H.money(clawback, compact=True)})."
        )
        # Nothing to make conditional if the buyer is not actually asking for a deep
        # cut - at that point the lines are just dying at their ordinary rate.
        if no_offset:
            actions = [
                f"There is no basket here, so no cherry-pick clause is possible - price the "
                f"{hero_name} line on its own merits and keep the "
                f"{hero['disc_pct']:.0f}% discount inside your {threshold_pct:.0f}% limit.",
                f"The line can still die on its own: {H.pct(float(p_line[hero_i]))} of lines "
                f"like it do. Budget {H.money(leakage, compact=True)} for that.",
            ]
        elif hero_deep:
            actions = [clause]
        else:
            actions = [
                f"The deepest discount here is {hero['disc_pct']:.0f}% on {hero_name}, which "
                f"is inside the {threshold_pct:.0f}% limit you set - so the offsetting lines "
                f"are dropping at their ordinary rate of {H.pct(base_rate)}, not the "
                f"{H.pct(deep_rate)} rate that follows a giveaway. Approve on normal terms.",
                f"Budget {H.money(leakage, compact=True)} of ordinary line-drop leakage "
                f"anyway - roughly {H.pct(p_any_offset)} of deals lose an offsetting line "
                f"whatever discount you give.",
            ]
        if len(offset_idx) > 1:
            actions.append(
                f"Do not let them drop one offsetting line and keep the other: make the "
                f"discount conditional on the whole {offset_names} basket surviving, not "
                f"on any single line.")
        if hist.get("avg_months"):
            actions.append(
                f"Put the test at month {horizon}, not at signature - past buyers who walk "
                f"do it around month {hist['avg_months']:.0f}.")
        else:
            actions.append(f"Put the test at month {horizon}, not at signature.")
        if p_below_floor > 0.05:
            actions.append(
                f"There is a {H.pct(p_below_floor)} chance the surviving bundle sits under "
                f"your {floor_pct:.0f}% floor - require CF sign-off above "
                f"{threshold_pct:.0f}% on any single line.")
        if not empirical:
            actions.append("Upload past quote lines with discount and cancellation flags so "
                           "this number comes from evidence instead of your assumption.")

        res.recommendation = Recommendation(
            headline=(
                (f"No cherry-pick possible: this quote is a single {hero_name} line, so "
                 f"there is no offsetting line to drop. Approve the "
                 f"{hero['disc_pct']:.0f}% on its own merits.")
                if no_offset else
                f"Approve the {hero['disc_pct']:.0f}% only as a conditional discount - "
                f"expect to lose {H.money(leakage, compact=True)} if you do not tie it "
                f"to the {lead_offset} line." if hero_deep else
                f"No cherry-pick signal: the deepest ask ({hero['disc_pct']:.0f}% on "
                f"{hero_name}) is inside your {threshold_pct:.0f}% limit, so the other lines "
                f"are only dying at their ordinary {H.pct(base_rate)} rate - approve on "
                f"normal terms and budget {H.money(leakage, compact=True)} for that."),
            rationale=[
                f"{H.pct(hero_share)} of this deal's profit sits in {hero_name}, the line "
                f"being cut by {hero['disc_pct']:.0f}%.",
                f"Past deals like this one lose the offsetting lines "
                f"{H.pct(p_any_offset)} of the time ({source}).",
                f"Expected profit after cherry-picking is "
                f"{H.money(expected_profit, compact=True)}, not "
                f"{H.money(full_profit, compact=True)} - a "
                f"{H.money(leakage, compact=True)} gap.",
            ],
            actions=actions,
            risk_score=round(score, 1),
            risk_label=label,
            confidence="High" if empirical else "Low",
            tradeoff=(
                (f"With one line there is no cross-subsidy to defend, so the only question "
                 f"is whether {hero_name} at {hero['disc_pct']:.0f}% off clears your own "
                 f"margin floor - nothing else on the quote can be used against you.")
                if no_offset else
                f"Walking away from the {hero['disc_pct']:.0f}% risks the deal; giving "
                f"it unprotected costs about {H.money(leakage, compact=True)}. The "
                f"clause lets you keep the deal and cap the loss. If the buyer refuses "
                f"any condition, the mildest version is to cap {hero_name} at "
                f"{threshold_pct:.0f}%, which still protects "
                f"{H.money(soft_clawback, compact=True)}." if hero_deep else
                f"There is no discount to trade here, so the only lever is structural: if "
                f"the {lead_offset} line matters to the deal, price it as a bundle with a "
                f"single cancellation date rather than as separate lines the buyer can peel "
                f"off one at a time."),
        )

        # ---- assumptions / warnings ---------------------------------------
        quote_desc = ", ".join(
            "{} {} at {:.0f}% off ({} margin)".format(
                ln["name"], H.money(ln["list"]), ln["disc_pct"], H.pct(ln["margin_frac"]))
            for ln in lines)
        res.assumptions = [
            f"Quote under review: {quote_desc}.",
            f"Deep discount means anything past {threshold_pct:.0f}%; a line counts as "
            f"abandoned if it dies within {horizon} months.",
            f"Abandonment rates come from {source}: "
            + (f"{hist['units']:,} past {'deals' if hist['whole'] else 'line items'}, "
               f"{int(sum(hist['n_deep'].values())):,} deep-discount category observations."
               if empirical else
               f"{H.pct(base_rate)} normal, {H.pct(deep_rate)} after a deep discount."),
            "Risk score = 25% x how far past your discount limit the ask sits (10 points "
            "past = full marks) + 50% x chance an offsetting line dies + 25% x share of "
            "profit sitting in the discounted line.",
            f"A cancelled line is treated as fully lost: no revenue and no cost.",
            f"Bundle margin floor {floor_pct:.0f}%; minimum support commitment "
            f"{attach:.0f}% of hours.",
            f"We replayed {n_res:,} bundles built from your own past deals; seed "
            f"{ctx.request.seed} - re-running gives identical numbers.",
        ]
        if not hist["ready"]:
            res.warnings.append(
                "No usable history was mapped (we need a category, the discount given and "
                "whether the line was cancelled), so every probability here is your typed-in "
                f"assumption: {H.pct(base_rate)} normal abandonment, "
                f"{H.pct(deep_rate)} after a deep discount.")
        if rows and not hist.get("whole", True):
            res.warnings.append(
                "Your file has no quote ID, so we resampled line items instead of whole "
                "deals. Lines inside the same deal move together, so treat the spread as "
                "optimistic - and we cannot measure how one line's discount changes another "
                f"line's survival, so the {H.pct(deep_rate)} abandonment rate you typed in "
                "is doing the work. Add a quote ID column to get the real lift.")
        if empirical and min(hist["n_deep"].values() or [0]) < 30:
            res.warnings.append(
                "Some category pairs rest on fewer than 30 past deals, so those figures "
                "are rough. The headline pair is the one with enough history behind it.")
        if horizon > 6 and not has_months:
            res.warnings.append(
                f"You asked for a {horizon}-month view but the data only records 6-month "
                f"cancellations, so the window stayed at 6 months.")
        if p_below_floor > 0.2:
            res.warnings.append(
                f"In {H.pct(p_below_floor)} of the rebuilt bundles the surviving basket is "
                f"below your {floor_pct:.0f}% margin floor.")

        ctx.narrator.done(
            f"Done. Cherry-pick risk {score:.0f}/100 ({label}) - expect to lose "
            f"{H.money(leakage, compact=True)} unless the discount is made conditional.")
        res.log = ctx.narrator.events()
        return res


# ---------------------------------------------------------------------------
# private chart / table builders
# ---------------------------------------------------------------------------
def _bootstrap_pool(n_units: int, n_res: int) -> int:
    """How many units the bundle rebuild may work on.

    The index matrix is `n_res x pool`, so the pool is what keeps this bounded:
    `_BOOTSTRAP_MAX_CELLS` caps the matrix at a few hundred megabytes whatever
    the user asks for. Returns `n_units` unchanged when the file is small enough
    that nothing has to be left out.
    """
    if n_units <= 0:
        return 0
    by_cells = max(500, _BOOTSTRAP_MAX_CELLS // max(1, int(n_res)))
    return int(min(int(n_units), _BOOTSTRAP_MAX_UNITS, by_cells))


def _cluster_indices(hist: dict, n_res: int, rng) -> np.ndarray:
    """(n_res, n_units) row indices: whole deals resampled together.

    A cluster bootstrap draws whole units with replacement, so the drawn
    *positions* are exactly the indices the caller needs - there is no need to
    build an array of labels and map them back (which cost one Python call per
    cell). `int32` because the pool is bounded well below 2^31.
    """
    n_units = int(hist["units"])
    if n_units <= 0:
        return np.zeros((n_res, 0), dtype=np.int32)
    if hist.get("whole", True):
        drawn = rng.choice(n_units, size=(n_res, n_units), replace=True)
    else:
        drawn = bs.resample_indices(n_units, n_res, rng)
    return np.asarray(drawn, dtype=np.int32)


def _discount_band_chart(hist, lines, line_cats, hero_cat, hero_i, threshold_pct,
                         base_rate, deep_rate, empirical, offset_idx):
    edges = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 200.0]
    labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50%+"]
    series = []
    if empirical and hero_cat is not None:
        d = hist["by_cat_disc"][hero_cat]
        # one common set of bands for every series, so the bars line up
        keep = [b for b in range(len(labels))
                if ((d >= edges[b]) & (d < edges[b + 1])).sum() >= 5]
        x = [labels[b] for b in keep]
        for i in offset_idx:
            c = line_cats[i]
            if c is None:
                continue
            vals = []
            for b in keep:
                m = (d >= edges[b]) & (d < edges[b + 1])
                vals.append(H.r2(100.0 * float(hist["by_cat_cancel"][c][m].mean())))
            series.append({"name": f"{lines[i]['name']} cancelled", "values": vals,
                           "color": "#ef4444" if i == offset_idx[0] else "#f59e0b"})
    else:
        x = labels[:4] if threshold_pct <= 20 else labels
        for i in offset_idx:
            vals = [H.r2(100.0 * (deep_rate if edges[b] >= threshold_pct else base_rate))
                    for b in range(len(x))]
            series.append({"name": f"{lines[i]['name']} cancelled", "values": vals,
                           "color": "#ef4444" if i == offset_idx[0] else "#f59e0b"})
    # which band your stated limit actually sits in (only if that band is drawn)
    band_mark = next((labels[b] for b in range(len(labels))
                      if labels[b] in x and edges[b] <= threshold_pct < edges[b + 1]),
                     x[min(3, len(x) - 1)] if x else labels[3])
    return H.chart(
        "cancel_by_discount_band", "bar" if len(series) == 1 else "grouped_bar",
        f"How often the other lines die, by how deep the {lines[hero_i]['name']} "
        f"discount goes",
        subtitle=("Each bar is a group of past deals with a similar discount"
                  if empirical else
                  "No history supplied - this is your assumption, drawn as a step"),
        x_label=f"Discount on {lines[hero_i]['name']}",
        y_label="Share of deals where that line was cancelled (%)",
        x=x, series=series,
        annotations=[{"x": band_mark,
                      "y": max([max(s["values"]) for s in series]) if series else 0,
                      "text": f"your {threshold_pct:.0f}% limit"}],
        plain_meaning="If the bars jump once the discount passes your limit, buyers are not "
                      "just buying more - they are buying differently. That jump is the "
                      "cherry-pick signal.",
    )


def _heatmap_chart(hist, threshold_pct, base_rate, deep_rate, lines, line_cats):
    cats = list(hist["cats"]) if hist.get("cats") else [l["name"] for l in lines]
    if not cats:
        cats = [l["name"] for l in lines]
    x = cats
    if hist.get("ready"):
        # only rows with enough deep-discount history to say anything
        rows = [a for a in cats if hist["n_deep"].get(a, 0) >= 10]
    else:
        rows = cats
    series = []
    for a in rows:
        row = []
        for b in cats:
            if hist.get("ready"):
                row.append(H.r2(hist["lift"][(a, b)]))
            else:
                row.append(H.r2(deep_rate / base_rate if a != b else 1.0))
        series.append({"name": a, "values": row})
    return H.chart(
        "pair_lift_heatmap", "heatmap",
        "Which discount poisons which line",
        subtitle=f"Numbers are 'times more likely than usual' when the row's line is "
                 f"discounted past {threshold_pct:.0f}%"
                 + (f" - rows with too little history are left out"
                    if hist.get("ready") else ""),
        x_label="Line that gets cancelled", y_label="Line that gets the deep discount",
        x=x, series=series,
        plain_meaning="Anything above 1.0 means the two events travel together. The hottest "
                      "cell is the pair you should write a contract clause about.",
    )


def _bundle_hist_chart(realised, full_profit, n_res):
    step = max(1, realised.size // 1500)
    return H.chart(
        "bundle_margin_hist", "hist",
        "What this deal actually earns",
        subtitle=f"{n_res:,} rebuilt bundles - the dashed line is the number on the quote",
        x_label="Profit from the surviving bundle", y_label="How often",
        x=[], series=[{"name": "Profit", "values": [float(v) for v in realised[::step]]}],
        annotations=[{"x": H.r2(float(np.mean(realised))), "y": 0,
                      "text": f"most likely {H.money(float(np.mean(realised)), compact=True)}"},
                     {"x": H.r2(float(full_profit)), "y": 0,
                      "text": f"if nothing is dropped {H.money(full_profit, compact=True)}"}],
        plain_meaning="The gap between the tall bars and the 'if nothing is dropped' marker "
                      "is the cost of cherry-picking. The left tail is the bad case you are "
                      "signing up for.",
    )


def _line_margin_chart(lines, profit_vec, p_line):
    expected = profit_vec * (1.0 - p_line)
    return H.chart(
        "line_margin_bars", "grouped_bar", "Where the profit comes from - and what survives",
        subtitle="Grey is what the quote promises, orange is what you should plan for",
        x_label="Line", y_label="Profit ($)",
        x=[l["name"] for l in lines],
        series=[
            {"name": "If they keep everything", "values": [H.r2(v) for v in profit_vec],
             "color": "#64748b"},
            {"name": "What you should plan for", "values": [H.r2(v) for v in expected],
             "color": "#f97316"},
        ],
        plain_meaning="The orange bar is the profit you can actually bank once past buyer "
                      "behaviour is priced in. The difference is the leak.",
    )


def _matrix_rows(hist, threshold_pct, base_rate, deep_rate, lines, line_cats,
                 empirical):
    if not empirical:
        rows = []
        for a in lines:
            for b in lines:
                same = a["name"] == b["name"]
                cond = base_rate if same else deep_rate
                rows.append([a["name"], b["name"], "-", H.pct(base_rate), H.pct(cond),
                             f"{st.lift(cond, base_rate):.2f}x",
                             "assumed" if not same else "-"])
        return rows
    rows = []
    for a in hist["cats"]:
        for b in hist["cats"]:
            n = hist["n_deep"].get(a, 0)
            if n < 10:
                continue
            lift = hist["lift"][(a, b)]
            rows.append([a, b, f"{n:,}", H.pct(hist["base"][b]),
                         H.pct(hist["cond"][(a, b)]), f"{lift:.2f}x",
                         _link_word(hist["phi"][(a, b)])])
    return rows


def _link_word(phi: float) -> str:
    a = abs(phi)
    if a >= 0.30:
        return "strong"
    if a >= 0.15:
        return "clear"
    if a >= 0.05:
        return "weak"
    return "none"


def _matrix_highlight(hist, lines, line_cats):
    if not hist.get("ready"):
        return []
    out = []
    best = None
    k = 0
    for a in hist["cats"]:
        for b in hist["cats"]:
            if hist["n_deep"].get(a, 0) < 10:
                continue
            if a == b:
                continue
            if best is None or hist["lift"][(a, b)] > best[1]:
                best = (k, hist["lift"][(a, b)])
            k += 1
    if best is not None and best[1] > 1.0:
        out.append(best[0])
    return out


def _tier_table(hist, threshold_pct, line_cats, offset_idx, hero_cat):
    if not hist.get("ready") or hero_cat is None or "tier_of_unit" not in hist:
        return None
    tiers = hist["tier_of_unit"]
    deep = hist["by_cat_disc"][hero_cat] > threshold_pct
    rows = []
    for t in sorted({str(x) for x in tiers if str(x)}):
        m = np.array([str(x) == t for x in tiers])
        if m.sum() < 30 or (m & deep).sum() < 10:
            continue
        vals = []
        for i in offset_idx:
            c = line_cats[i]
            if c is None:
                continue
            vals.append(f"{100.0 * float(hist['by_cat_cancel'][c][m & deep].mean()):.0f}%")
        rows.append([t, f"{int(m.sum()):,}", f"{int((m & deep).sum()):,}",
                     ", ".join(vals) if vals else "-"])
    if not rows:
        return None
    return H.table(
        "Does the buyer's tier change the behaviour?",
        f"Same test, split by customer tier (only tiers with enough history are shown).",
        ["Customer tier", "Deals", "Deals with a deep discount", "Chance each offsetting "
         "line dies"],
        rows,
        note="Use the worst tier as your planning number if you do not know the buyer well.")
