"""USE CASE - Competitor Pricing: The Virtual War Room (Agent-Based Model).

The question a spreadsheet cannot answer: "a rival just cut 12% - do we match
everywhere?" A spreadsheet can price one account. It cannot tell you what
happens to the *market* when you and two rivals keep answering each other for
two years, or which accounts are worth defending and which are worth losing.

So we build a virtual market: 500 buyers with different price sensitivities,
your firm, Competitor A and Competitor B. Every quarter each buyer looks at the
three offers and picks one - the cheaper offer wins more often than not, but not
always. We then play that market out 100 times for each strategy and compare
what each one is actually worth.

Every conclusion on the page - the verdict labels, the colours on the KPI
cards, the headline, the highlighted row, the chart captions and the
recommendation - is *derived from the numbers the simulation produced*. Nothing
is decided in advance. If the data says matching every cut earns the most
money, the report says so.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..core.contracts import ParamSpec, Recommendation, RoleSpec, UseCaseMeta
from ..core.registry import UseCase, register
from ..engine import agent_based as ab
from . import _helpers as H

#: A price gap is worth about ten times more than habit to a mid-sensitivity
#: buyer: a 10% gap moves a typical account, a 2% gap does not.
PRICE_WEIGHT = 10.0

#: How decisive a buyer is, in "points of attraction". A buyer who prefers us
#: by 0.2 points picks us about three times in four; by 0.6 points, nine times
#: in ten. Lower = buyers always take the best offer.
CHOICE_TEMPERATURE = 0.2

#: Strategies we put through the war room, in display order.
STRATEGIES = [
    ("aggressive_match", "Match every cut",
     "Answer every rival cut on every account, immediately."),
    ("defensible_niche", "Defensible niche",
     "Hold list price, bundle priority support on the accounts worth keeping, and "
     "match only where the account commits multi-year."),
    ("hold_price", "Hold the line",
     "Do nothing. Keep list price everywhere and let the chips fall."),
]

#: What the recommendation is allowed to maximise. The first is the default.
OBJECTIVES = ("profit", "margin", "tier1_retention")
OBJECTIVE_WORD = {
    "profit": "profit",
    "margin": "profit margin",
    "tier1_retention": "top-tier retention",
}

#: Dead-band sweep: how far the rival's undercut is moved before we re-run the
#: market, and how many versions of the future each point is worth.
DEADBAND_MIN_PCT = 0.0
DEADBAND_MAX_PCT = 40.0
DEADBAND_STEP_PCT = 2.0
DEADBAND_REPLICATES = 12

#: A number the user must never see. Used by `_self_check`.
_JUNK = re.compile(r"\b(nan|inf|infinity)\b", re.IGNORECASE)

META = UseCaseMeta(
    id="competitor_warroom",
    title="Competitor Pricing: The Virtual War Room",
    subtitle="A rival is undercutting you by 12%. Should you match them everywhere?",
    category="Pricing Strategy",
    icon="⚔️",
    models=["Agent-Based Modeling"],
    business_question="If we match every competitor cut, what does that do to our price, our best accounts and our profit?",
    plain_summary=(
        "We build a virtual market out of your accounts, two rivals and your own "
        "firm, then let them fight. Every buyer looks at the three offers each "
        "quarter and picks one - the cheapest offer usually wins, but habit and a "
        "priority-support promise count too. We run that market 100 times for each "
        "strategy and show which one leaves you with the most profit, plus how far "
        "the rival has to move before that answer changes."
    ),
    story_steps=[
        "Read the account book: who buys what, how price-sensitive they are, and what they cost to serve",
        "Build the virtual market: your firm, Competitor A and Competitor B",
        "Play out many versions of the quarters ahead for each strategy",
        "Rank the three strategies on the measure you chose to maximise",
        "Test how far the rival's undercut can move before the ranking changes",
        "Recommend the strategy that actually scores best, and say so plainly",
    ],
    data_granularity="portfolio",
    sample_dataset="market_accounts.csv",
    input_roles=[
        RoleSpec(key="account_id", label="Account / customer ID", dtype="identifier",
                 required=False, aliases=["id", "account", "customer", "client", "name"]),
        RoleSpec(key="tier", label="Account tier / segment", dtype="categorical",
                 required=False, aliases=["segment", "band", "category", "class"],
                  help="Groups of accounts, e.g. Tier-1 / Tier-2 / Tier-3."),
        RoleSpec(key="annual_volume", label="Annual spend with you", dtype="numeric",
                 required=False, aliases=["volume", "spend", "revenue", "arr", "acv",
                                          "annual_spend", "amount"]),
        RoleSpec(key="price_sensitivity", label="Price sensitivity (0-1)", dtype="numeric",
                 required=False, aliases=["sensitivity", "elasticity", "price_elasticity"],
                  help="0 = never switches on price, 1 = always chases the cheapest."),
        RoleSpec(key="current_supplier", label="Who supplies them today", dtype="categorical",
                 required=False, aliases=["supplier", "incumbent", "vendor", "provider"]),
        RoleSpec(key="margin_pct", label="Margin on the account (%)", dtype="numeric",
                 required=False, aliases=["margin", "gross_margin", "gm"],
                 fallback=35, help="Use 35 for 35%."),
        RoleSpec(key="annual_support_cost", label="Annual cost to support them", dtype="numeric",
                 required=False, aliases=["support_cost", "cost_to_serve", "service_cost"]),
    ],
    parameters=[
        ParamSpec(key="n_buyers", label="Buyers in the virtual market", kind="integer",
                  default=500, min=50, max=5000, step=50, group="The market",
                  help="Only used when you have no account file."),
        ParamSpec(key="n_quarters", label="Quarters to play out", kind="integer",
                  default=8, min=2, max=24, step=1, group="The market", help="How many quarters into the future to play out. Eight is two years.",),
        ParamSpec(key="n_replicates", label="Versions of the future to play", kind="integer",
                  default=100, min=10, max=1000, step=10, group="The market",
                  help="More versions = a tighter range around each answer."),
        # QA TM-07: "per unit" was ambiguous - use the plainer phrasing.
        ParamSpec(key="baseline_asp", label="Your average price per account", kind="number",
                  default=100000, min=100, step=1000, unit="$", group="The market",
                  help="What you charge a typical account today, before any discount."),
        ParamSpec(key="our_margin_pct", label="Your margin at list price", kind="number",
                  default=35, min=0, max=90, step=1, unit="%", group="The market",
                  help="Used for accounts where you have no margin figure. 35 means 35%."),
        ParamSpec(key="fixed_cost_base", label="Fixed cost per year", kind="number",
                  default=5_000_000, min=0, step=100_000, unit="$", group="The market",
                  help="Overheads the profit has to cover before it counts."),
        # QA TM-18: "undercuts you" was ambiguous - against list price or street price?
        ParamSpec(key="competitor_undercut_pct", label="How far the rival undercuts you",
                  kind="number", default=12, min=0, max=60, step=1, unit="%",
                  group="The rival",
                  help="How much cheaper the rival's price is than YOUR list price - the "
                       "price you charge today, before any discount. 12 means they are "
                       "asking for 12% less than you currently charge."),
        ParamSpec(key="competitor_reaction_speed", label="How fast the rival counters",
                  kind="number", default=0.5, min=0.05, max=1.0, step=0.05,
                  group="The rival",
                  help="0.5 = they close half the gap each quarter. 1.0 = instantly."),
        ParamSpec(key="max_market_discount_pct", label="Deepest the whole market can go",
                  kind="number", default=20, min=0, max=60, step=1, unit="%",
                  group="The rival",
                  help="Nobody sells below this off list - it is where a price war stops."),
        ParamSpec(key="loyalty_bonus", label="Pull of staying put", kind="number",
                  default=0.35, min=0, max=2, step=0.05, group="Buyer behaviour",
                  help="How much buyers prefer the supplier they already use."),
        ParamSpec(key="switching_friction", label="Hassle of switching", kind="number",
                  default=0.25, min=0, max=2, step=0.05, group="Buyer behaviour",
                  help="The drag that stops a buyer moving even for a better price."),
        ParamSpec(key="sla_value_pct", label="Value buyers put on priority support",
                  kind="number", default=6, min=0, max=30, step=1, unit="%",
                  group="Buyer behaviour", help="How much buyers value a priority-support promise, counted as if it were a discount of this size.",),
        ParamSpec(key="sla_cost_pct", label="What priority support costs you",
                  kind="number", default=1.5, min=0, max=15, step=0.5, unit="%",
                  group="Buyer behaviour",
                  help="Points of margin you spend to deliver it."),
        ParamSpec(key="choice_temperature", label="How decisive buyers are", kind="number",
                  default=0.2, min=0.05, max=1.5, step=0.05, group="Buyer behaviour",
                  help="Low = buyers almost always take the best offer. High = they shop around "
                       "and even a weak offer wins sometimes."),
        ParamSpec(key="multiyear_commitment_share_pct", label="Top accounts willing to commit multi-year",
                  kind="number", default=40, min=0, max=100, step=5, unit="%",
                  group="Buyer behaviour",
                  help="Only these accounts earn a matched price."),
        # QA TM-12: no tooltip, and it was ambiguous which margin this guards.
        ParamSpec(key="min_margin_after_match_pct", label="Never match below this margin",
                  kind="number", default=12, min=0, max=50, step=1, unit="%",
                  group="Buyer behaviour",
                  help="The margin you keep on the deal AFTER you have matched the "
                       "rival's price - not your margin today. If matching would push the "
                       "margin on that account below this number, we refuse to match and "
                       "let the account go instead."),
        # QA TM-13: "top-tier" was never defined anywhere in the flow.
        # QA DF-06: make "defend only a subset" an explicit, visible choice.
        ParamSpec(key="defend_scope", label="Which accounts to defend", kind="choice",
                  default="top_two", group="Your answer",
                  choices=[{"value": "top_only", "label": "Only the top tier"},
                           {"value": "top_two", "label": "Top two tiers"},
                           {"value": "all", "label": "Every account"}],
                  help="Under the selective strategy, this is the set of accounts we "
                       "defend with a matched price and priority support. Accounts outside "
                       "it are allowed to walk to the rival - and the result shows what "
                       "that costs and saves."),
        ParamSpec(key="tier1_retention_target_pct", label="Top-tier accounts you must keep",
                  kind="number", default=88, min=0, max=100, step=1, unit="%",
                  group="Target",
                  help="'Top-tier' means your highest-margin accounts - the ones in the "
                       "first tier of your file. This is the share of them that must still "
                       "be yours at the end. We flag it in the result when the answer "
                       "falls short of this number."),
        ParamSpec(key="objective", label="What the recommendation should maximise",
                  kind="choice", default="profit",
                  choices=[
                      {"value": "profit", "label": "Profit over the whole period you set"},
                      {"value": "margin", "label": "Profit margin (profit before overheads as a share of revenue)"},
                      {"value": "tier1_retention", "label": "Keeping your top-tier accounts"},
                  ],
                  group="Target",
                  help="The strategy that scores best on this is the one we recommend."),
    ],
)


# ---------------------------------------------------------------------------
# Choice rule, vectorised
# ---------------------------------------------------------------------------
def _period_phrase(quarters: int) -> str:
    """Plain-English name for the horizon. Never hardcode 'two years' - the
    horizon is a parameter and the labels must follow it."""
    q = max(int(quarters), 1)
    if q % 4 == 0:
        years = q // 4
        return "1 year" if years == 1 else f"{years} years"
    if q == 1:
        return "1 quarter"
    return f"{q} quarters"


def _period_short(quarters: int) -> str:
    q = max(int(quarters), 1)
    if q % 4 == 0:
        years = q // 4
        return "1 yr" if years == 1 else f"{years} yrs"
    return f"{q} qtrs"


def _defend_mask(tier_code: np.ndarray, scope: str) -> np.ndarray:
    """Which accounts the selective strategy defends.

    QA DF-06: UC4's real question is "which accounts do I defend?", so the set
    is an explicit choice rather than a hardcoded tier cut-off.
    """
    if scope == "top_only":
        return tier_code == 0
    if scope == "all":
        return np.ones(tier_code.shape, dtype=bool)
    return tier_code <= 1                       # "top_two" - the default


def _choose(u: np.ndarray, rng: np.random.Generator, temperature: float) -> np.ndarray:
    """Probabilistic supplier choice, one draw per row.

    The most attractive offer wins most often but never always. Scored against
    a fixed yardstick instead of the spread of that buyer's own three offers:
    the engine's version divides by the spread, which quietly caps the leading
    supplier at about four wins in five however good its offer is; a fixed
    yardstick lets a genuinely better offer win nine times in ten.
    """
    t = max(float(temperature), 0.02)
    centred = u - u.max(axis=1, keepdims=True)
    w = np.exp(centred / t)
    p = w / w.sum(axis=1, keepdims=True)
    r = rng.random(u.shape[0])[:, None]
    return (r > np.cumsum(p, axis=1)).sum(axis=1)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
def _supplier_codes(values: list[str], rng) -> tuple[np.ndarray, list[str], str]:
    """Map 'who supplies them today' onto 0 = us, 1 = Competitor A, 2 = Competitor B."""
    distinct = []
    for v in values:
        if v not in distinct:
            distinct.append(v)
    us_name = ""
    for v in distinct:
        if str(v).strip().lower() in ("us", "we", "our firm", "ours", "self", "incumbent us"):
            us_name = v
            break
    note = ""
    if not us_name and distinct:
        # no obvious 'Us' in the file - assume the biggest supplier is you
        counts = {v: sum(1 for x in values if x == v) for v in distinct}
        us_name = max(counts, key=lambda k: counts[k])
        note = (f"We could not see an obvious 'Us' in your supplier column, so we "
                f"assumed '{us_name}' is you.")
    rivals = [v for v in distinct if v != us_name]
    if not rivals:
        rivals = ["Competitor A", "Competitor B"]
    out = np.zeros(len(values), dtype=int)
    for i, v in enumerate(values):
        if v == us_name:
            out[i] = 0
        else:
            j = rivals.index(v) if v in rivals else 0
            out[i] = 1 if j % 2 == 0 else 2
    labels = [us_name or "Us", "Competitor A", "Competitor B"]
    if len(rivals) == 1:
        labels = [us_name or "Us", rivals[0], rivals[0] + " "]
    return out, labels, note


def _build_population(rows, P, rng):
    """Buyers as `ab.Agent` objects plus the same facts as number columns."""
    n_gen = int(H.p(P, "n_buyers", 500))
    base_margin = H.as_float(H.p(P, "our_margin_pct", 35))

    if H.has(rows, "tier") or H.has(rows, "annual_volume") or H.has(rows, "price_sensitivity"):
        n = max(len(rows), 1)
        ids = H.cat(rows, "account_id", "") or [f"Account {i + 1}" for i in range(n)]
        tiers = H.cat(rows, "tier", "Tier-2")
        if not any(tiers):
            tiers = [f"Tier-2" for _ in range(n)]
        vol = H.num(rows, "annual_volume", 1_000_000.0)
        vol = np.where(vol > 0, vol, 1_000_000.0)
        sens = np.clip(H.num(rows, "price_sensitivity", 0.5), 0.02, 1.0)
        marg = H.num(rows, "margin_pct", base_margin)
        marg = np.where(marg > 1.0, marg / 100.0, marg)
        marg = np.where((marg <= 0) | (marg > 0.9), base_margin / 100.0, marg)
        support = H.num(rows, "annual_support_cost", 0.0)
        incumbent, supplier_labels, note = _supplier_codes(
            H.cat(rows, "current_supplier", "Competitor A"), rng)
        source = f"{n:,} accounts from your file"
        ctx_note = note
    else:
        n = max(n_gen, 1)
        # independent draws, then shaped into a population with the same
        # profile as the sample account book
        agents = ab.make_agents(n, {
            "u_tier": {"dist": "uniform", "low": 0.0, "high": 1.0},
            "z_volume": {"dist": "normal", "mean": 0.0, "sd": 1.0},
            "z_sens": {"dist": "normal", "mean": 0.0, "sd": 1.0},
            "z_margin": {"dist": "normal", "mean": 0.0, "sd": 1.0},
            "u_supplier": {"dist": "uniform", "low": 0.0, "high": 1.0},
            "z_support": {"dist": "normal", "mean": 0.0, "sd": 1.0},
        }, rng)
        u_tier = np.array([a["u_tier"] for a in agents])
        z_vol = np.array([a["z_volume"] for a in agents])
        z_sens = np.array([a["z_sens"] for a in agents])
        z_marg = np.array([a["z_margin"] for a in agents])
        u_sup = np.array([a["u_supplier"] for a in agents])
        z_sup = np.array([a["z_support"] for a in agents])

        tier_ix = np.where(u_tier < 0.18, 0, np.where(u_tier < 0.55, 1, 2))
        vol_mu = np.array([15.0, 13.8, 12.6])[tier_ix]
        sens_mu = np.array([0.25, 0.50, 0.78])[tier_ix]
        marg_mu = np.array([base_margin + 8.0, base_margin - 2.0, base_margin - 16.0])[tier_ix]
        support_rate = np.array([0.050, 0.035, 0.020])[tier_ix]
        supp_w = np.array([[0.55, 0.28, 0.17], [0.45, 0.30, 0.25], [0.30, 0.38, 0.32]])[tier_ix]

        tiers = [f"Tier-{int(t) + 1}" for t in tier_ix]
        ids = [f"Account {i + 1:04d}" for i in range(n)]
        vol = np.round(np.exp(vol_mu + 0.45 * z_vol), -3)
        vol = np.maximum(vol, 25_000.0)
        sens = np.clip(sens_mu + 0.10 * z_sens, 0.02, 0.99)
        marg = np.clip((marg_mu + 4.0 * z_marg) / 100.0, 0.04, 0.70)
        support = vol * support_rate * np.clip(1.0 + 0.12 * z_sup, 0.7, 1.4)
        incumbent = np.where(u_sup < supp_w[:, 0], 0,
                             np.where(u_sup < supp_w[:, 0] + supp_w[:, 1], 1, 2))
        supplier_labels = ["Us", "Competitor A", "Competitor B"]
        source = f"{n:,} simulated accounts"
        ctx_note = ""

    # tier codes: 0 = highest margin, 2 = lowest margin
    labels_order = []
    for t in tiers:
        if t not in labels_order:
            labels_order.append(t)
    means = {t: float(marg[[i for i, x in enumerate(tiers) if x == t]].mean())
             for t in labels_order}
    order = sorted(labels_order, key=lambda t: -means[t])
    code_of = {t: i for i, t in enumerate(order)}
    tier_code = np.array([code_of[t] for t in tiers], dtype=int)

    asp = H.as_float(H.p(P, "baseline_asp", 100_000))
    list_price = max(asp, 1.0)
    units = np.maximum(vol / 4.0, 1e-9) / list_price        # units bought per quarter
    cost = list_price * (1.0 - marg)                        # cost to serve one unit

    commit = H.as_float(H.p(P, "multiyear_commitment_share_pct", 40)) / 100.0
    u_commit = rng.random(n)
    want = np.where(tier_code == 0, commit,
                    np.where(tier_code == 1, commit * 0.6, 0.0))
    multiyear = u_commit < want

    agents = [ab.Agent(i, {
        "account_id": ids[i],
        "tier": tiers[i],
        "tier_code": int(tier_code[i]),
        "annual_volume": float(vol[i]),
        "price_sensitivity": float(sens[i]),
        "margin_pct": float(marg[i] * 100.0),
        "current_supplier": supplier_labels[incumbent[i]],
        "annual_support_cost": float(support[i]),
        "commits_multiyear": bool(multiyear[i]),
    }) for i in range(n)]

    cols = {
        "n": n,
        "ids": list(ids),
        "tier_labels": list(tiers),
        "tier_code": tier_code,
        "volume": vol,
        "sens": sens,
        "margin": marg,
        "support": support,
        "incumbent": incumbent,
        "units": units,
        "cost": cost,
        "multiyear": multiyear,
        "supplier_labels": supplier_labels,
        "tier_order": order,
    }
    return agents, cols, source, ctx_note


# ---------------------------------------------------------------------------
# One world
# ---------------------------------------------------------------------------
def _run_world(strategy: str, cols: dict, P, rng) -> dict:
    n = cols["n"]
    quarters = int(H.p(P, "n_quarters", 8))
    list_price = max(H.as_float(H.p(P, "baseline_asp", 100_000)), 1.0)
    undercut = H.as_float(H.p(P, "competitor_undercut_pct", 12)) / 100.0
    speed = float(np.clip(H.as_float(H.p(P, "competitor_reaction_speed", 0.5)), 0.05, 1.0))
    floor = list_price * (1.0 - H.as_float(H.p(P, "max_market_discount_pct", 20)) / 100.0)
    loyalty = H.as_float(H.p(P, "loyalty_bonus", 0.35))
    friction = H.as_float(H.p(P, "switching_friction", 0.25))
    sla_bonus = H.as_float(H.p(P, "sla_value_pct", 6)) / 100.0 * PRICE_WEIGHT * 0.5
    sla_cost = H.as_float(H.p(P, "sla_cost_pct", 1.5)) / 100.0 * list_price
    min_margin = H.as_float(H.p(P, "min_margin_after_match_pct", 12)) / 100.0
    fixed_q = H.as_float(H.p(P, "fixed_cost_base", 5_000_000)) / 4.0
    temperature = H.as_float(H.p(P, "choice_temperature", CHOICE_TEMPERATURE))
    # QA DF-06: which accounts the selective strategy actually defends.
    defend_scope = str(H.p(P, "defend_scope", "top_two"))

    tier_code = cols["tier_code"]
    sens = cols["sens"]
    units = cols["units"]
    cost = cols["cost"]
    incumbent = cols["incumbent"]
    multiyear = cols["multiyear"]
    t1_start = (tier_code == 0) & (incumbent == 0)

    supplier = incumbent.copy()
    our_prev = list_price
    comp_a = list_price
    comp_b = list_price

    history = []
    for _t in range(quarters):
        # -- the rivals move first -----------------------------------------
        target = max(our_prev * (1.0 - undercut), floor)
        comp_a = max(comp_a + speed * (target - comp_a), floor)
        comp_b = max(comp_b + speed * (target * 1.03 - comp_b), floor)

        # -- our answer ------------------------------------------------------
        our = np.full(n, list_price)
        sla = np.zeros(n, dtype=bool)
        if strategy == "aggressive_match":
            our = np.full(n, min(comp_a, list_price))
        elif strategy == "defensible_niche":
            # QA DF-06: "which accounts to defend" is the UC4 question, so the
            # set we defend is now an explicit choice rather than hardcoded.
            defend = _defend_mask(tier_code, defend_scope)
            sla = defend                            # priority support on what we defend
            u_us = (loyalty * (incumbent == 0) - friction * (incumbent != 0)
                    + sla_bonus * sla)
            gap_a = (comp_a - list_price) / list_price
            gap_b = (comp_b - list_price) / list_price
            u_rival = np.maximum(
                -gap_a * sens * PRICE_WEIGHT + loyalty * (incumbent == 1) - friction * (incumbent != 1),
                -gap_b * sens * PRICE_WEIGHT + loyalty * (incumbent == 2) - friction * (incumbent != 2),
            )
            at_risk = u_rival > u_us + 0.02
            price_ok = comp_a >= cost * (1.0 + min_margin)
            match = at_risk & multiyear & defend & price_ok
            our = np.where(match, min(comp_a, list_price), list_price)

        # -- every buyer picks a supplier ------------------------------------
        prices = np.stack([our, np.full(n, comp_a), np.full(n, comp_b)], axis=1)
        gap = (prices - list_price) / list_price
        u = -gap * sens[:, None] * PRICE_WEIGHT
        u += np.where(np.arange(3)[None, :] == incumbent[:, None], loyalty, -friction)
        u[:, 0] += sla_bonus * sla
        new_supplier = _choose(u, rng, temperature)
        churned = int(np.sum((supplier == 0) & (new_supplier != 0)))
        supplier = new_supplier

        # -- bookkeeping -------------------------------------------------------
        ours = supplier == 0
        q_units = float(units[ours].sum())
        revenue = float((our[ours] * units[ours]).sum())
        unit_cost = cost[ours] + sla_cost * sla[ours]
        gross = float(((our[ours] - unit_cost) * units[ours]).sum())
        support_q = float(cols["support"][ours].sum() / 4.0)
        ebitda = gross - support_q - fixed_q
        asp = revenue / q_units if q_units > 0 else list_price
        history.append({
            "asp": asp,
            "units": q_units,
            "revenue": revenue,
            "gross_margin": gross,
            "support": support_q,
            "ebitda": ebitda,
            "share": float(ours.mean()),
            "tier1_ret": float((supplier[t1_start] == 0).mean()) if t1_start.any() else 0.0,
            "tier1_share": float((supplier[tier_code == 0] == 0).mean()) if (tier_code == 0).any() else 0.0,
            "churned": churned,
            "comp_price": comp_a,
            "our_price": asp,
        })
        our_prev = asp

    return {"history": history, "supplier": supplier}


def _metrics(histories: list[dict], quarters: int) -> dict:
    keys = ["asp", "units", "revenue", "gross_margin", "ebitda", "share",
            "tier1_ret", "tier1_share", "churned"]
    out = {}
    for k in keys:
        arr = np.array([[h[k] for h in hist] for hist in histories], dtype=float)
        out[k] = {
            "mean": arr.mean(axis=0),
            "p5": np.percentile(arr, 5, axis=0),
            "p95": np.percentile(arr, 95, axis=0),
            "last": arr[:, -1],
            "total": arr.sum(axis=1),
        }
    return out


# ---------------------------------------------------------------------------
# Objective, ranking and comparison wording - all derived from the numbers
# ---------------------------------------------------------------------------
def _objective_by_rep(worlds: list[dict], objective: str) -> np.ndarray:
    """One score per simulated version of the future, for the chosen objective."""
    if objective == "tier1_retention":
        return np.asarray([w["history"][-1]["tier1_ret"] for w in worlds], dtype=float)
    if objective == "margin":
        out = []
        for w in worlds:
            revenue = sum(h["revenue"] for h in w["history"])
            gross = sum(h["gross_margin"] for h in w["history"])
            out.append(gross / revenue if revenue > 0 else 0.0)
        return np.asarray(out, dtype=float)
    return np.asarray([sum(h["ebitda"] for h in w["history"]) for w in worlds], dtype=float)


def _fmt_obj(value: float, objective: str) -> str:
    if objective == "profit":
        return H.money(value, compact=True)
    return H.pct(value)


def _fmt_gap(value: float, objective: str) -> str:
    """A gap between two scores, always positive and always carrying its unit."""
    if objective == "profit":
        return H.money(abs(value), compact=True)
    return f"{abs(value) * 100:.1f} points"


def _direction_word(delta: float, tol: float = 1e-9) -> str:
    if delta > tol:
        return "more"
    if delta < -tol:
        return "less"
    return "the same as"


def _vs_phrase(gap: float, objective: str, label: str) -> str:
    """'$6.2M more than holding firm' - or, on a dead heat, 'the same as ...'."""
    direction = _direction_word(gap)
    if direction == "the same as":
        return f"the same as {label}"
    return f"{_fmt_gap(gap, objective)} {direction} than {label}"


def _rank_tone(rank: int, n: int) -> str:
    """Good for the winner, bad for the loser, warn in between."""
    if rank <= 0:
        return "good"
    if rank >= n - 1:
        return "bad"
    return "warn"


def _sweep_points(current_pct: float) -> list[float]:
    """0-40% in 2-point steps, always including where the world is today."""
    pts = [round(float(x), 2) for x in
           np.arange(DEADBAND_MIN_PCT, DEADBAND_MAX_PCT + 1e-9, DEADBAND_STEP_PCT)]
    if not any(abs(p - current_pct) < 1e-9 for p in pts):
        pts.append(round(float(current_pct), 2))
        pts.sort()
    return pts


def _sweep_replicates(n: int, quarters: int) -> int:
    """Fewer versions per point when the market is big, so the sweep stays quick."""
    work = max(n * quarters, 1)
    if work <= 40_000:
        return DEADBAND_REPLICATES
    return max(6, int(DEADBAND_REPLICATES * 40_000 / work))


def _dead_band(cols: dict, P, ctx, current_undercut: float,
               objective: str, reps: int) -> list[dict]:
    """Re-run the market at a range of rival undercuts, one random stream per point.

    One-factor test: only the rival's undercut moves; every other assumption is
    held at its current value. The same random stream is reused inside a point so
    the three strategies are compared on the same version of the future.
    """
    out = []
    for u in _sweep_points(current_undercut):
        rng = ctx.rng("competitor_warroom", "deadband", f"{u:.2f}")
        P2 = dict(P)
        P2["competitor_undercut_pct"] = u
        scores = {}
        for key, _label, _desc in STRATEGIES:
            worlds = [_run_world(key, cols, P2, rng) for _ in range(reps)]
            scores[key] = float(_objective_by_rep(worlds, objective).mean())
        best = max(STRATEGIES, key=lambda s: scores[s[0]])[0]
        out.append({"undercut": u, "scores": scores, "best": best})
    return out


def _self_check(res, facts: dict) -> None:
    """Fail loudly if the conclusion and the numbers disagree.

    Guards the correctness contract: no broken numbers anywhere the user can
    see, tables as wide as their headers, highlight rows that exist and point at
    the recommendation, the headline's direction word matching the sign of the
    computed gap, and every probability-like figure inside 0-100%.
    """
    delta = float(facts["headline_delta"])
    assert math.isfinite(delta), f"the headline gap is not a number: {delta}"
    word = _direction_word(delta)
    assert word in res.headline, (
        f"the headline does not state the direction the numbers show "
        f"('{word}', signed gap {delta:,.6f}): {res.headline!r}")

    for key, value in facts["objective_values"].items():
        assert math.isfinite(float(value)), (
            f"the {facts['objective']} score for '{key}' is not a number: {value}")

    for name, prop in facts["proportions"].items():
        assert math.isfinite(float(prop)), f"{name} is not a number: {prop}"
        assert -1e-9 <= float(prop) <= 1.0 + 1e-9, (
            f"{name} should be a percentage between 0% and 100%, got {prop}")

    for t in res.tables:
        for row in t.rows:
            assert len(row) == len(t.columns), (
                f"table '{t.name}': a row has {len(row)} cells but there are "
                f"{len(t.columns)} column headers")
            for cell in row:
                assert not _JUNK.search(str(cell)), (
                    f"table '{t.name}' contains a broken number: {cell!r}")
        for i in t.highlight_rows:
            assert 0 <= int(i) < len(t.rows), (
                f"table '{t.name}': highlighted row {i} does not exist")

    strategies_table = res.tables[0]
    assert strategies_table.highlight_rows == [facts["best_row_index"]], (
        f"the highlighted row is {strategies_table.highlight_rows} but the "
        f"recommended strategy sits in row {facts['best_row_index']}")

    for c in res.charts:
        for s in c.series:
            for v in s.get("values", []):
                assert isinstance(v, (int, float)) and math.isfinite(float(v)), (
                    f"chart '{c.id}', series '{s.get('name')}': broken value {v!r}")
        for a in c.annotations:
            y = a.get("y")
            if isinstance(y, (int, float)):
                assert math.isfinite(float(y)), (
                    f"chart '{c.id}': broken annotation {a!r}")
        assert not _JUNK.search(c.plain_meaning or ""), (
            f"chart '{c.id}' caption contains a broken number")

    for k in res.kpis:
        assert k.tone in ("good", "bad", "warn", "neutral"), (
            f"KPI '{k.label}' has an unknown colour: {k.tone}")
        assert not _JUNK.search(k.value) and not _JUNK.search(k.plain or ""), (
            f"KPI '{k.label}' contains a broken number")


# ---------------------------------------------------------------------------
@register
class CompetitorWarRoomUseCase(UseCase):
    meta = META

    def run(self, req, ctx) -> "H.RunResult":
        P = ctx.params
        rows = ctx.rows
        rng = ctx.rng("competitor_warroom", "population")
        quarters = int(H.p(P, "n_quarters", 8))
        reps = int(H.p(P, "n_replicates", 100))
        list_price = max(H.as_float(H.p(P, "baseline_asp", 100_000)), 1.0)
        target_ret = H.as_float(H.p(P, "tier1_retention_target_pct", 88))
        objective = str(H.p(P, "objective", "profit") or "profit")
        if objective not in OBJECTIVES:
            objective = "profit"
        word = OBJECTIVE_WORD[objective]

        ctx.say("Reading the account book...", phase="read", progress=0.04)
        agents, cols, source, note = _build_population(rows, P, rng)
        n = cols["n"]
        if note:
            ctx.say(note, phase="validate", level="warn")
        ctx.say(f"Built the market from {source}.", phase="prepare", progress=0.12,
                detail=f"{n} buyers, {len(cols['tier_order'])} tiers, "
                       f"{H.money(float(cols['volume'].sum()), compact=True)} of annual spend")
        ctx.say(f"Three suppliers are in the room: you, "
                f"{cols['supplier_labels'][1]} and {cols['supplier_labels'][2]}.",
                phase="prepare", progress=0.16)

        # -- play the market out under each strategy -------------------------
        total_worlds = reps * len(STRATEGIES)
        results: dict[str, dict] = {}
        tick = ctx.narrator.tracker(total_worlds,
                                    "Running the war room... {pct}% of the worlds simulated",
                                    phase="simulate")
        done = 0
        for key, label, _desc in STRATEGIES:
            worlds = []
            for r in range(reps):
                w_rng = ctx.rng("competitor_warroom", "world", key, str(r))
                worlds.append(_run_world(key, cols, P, w_rng))
                done += 1
                tick(done)
            results[key] = {
                "label": label,
                "m": _metrics([w["history"] for w in worlds], quarters),
                # chance each account ends the two years with us
                "final_supplier": np.mean([(w["supplier"] == 0).astype(float)
                                           for w in worlds], axis=0),
                "worlds": worlds,
            }
            ctx.say(f"{label}: {reps} versions played out.", phase="simulate",
                    progress=0.2 + 0.55 * done / total_worlds,
                    detail=f"average profit {H.money(results[key]['m']['ebitda']['total'].mean(), compact=True)} "
                           f"over {quarters} quarters")

        ctx.say("Scoring the three strategies against each other...", phase="summarize",
                progress=0.78)

        # -- the numbers every conclusion is built from ----------------------
        summ: dict[str, dict] = {}
        for key, label, _desc in STRATEGIES:
            m = results[key]["m"]
            obj_rep = _objective_by_rep(results[key]["worlds"], objective)
            revenue = float(m["revenue"]["total"].mean())
            gross = float(m["gross_margin"]["total"].mean())
            summ[key] = {
                "label": label,
                "profit": float(m["ebitda"]["total"].mean()),
                "revenue": revenue,
                "gross": gross,
                "margin_rate": gross / revenue if revenue > 0 else 0.0,
                "asp_delta": float(m["asp"]["last"].mean() / list_price - 1.0),
                "share": float(m["share"]["last"].mean()),
                "ret": float(m["tier1_ret"]["last"].mean()),
                "churn": float(m["churned"]["total"].mean()),
                "obj": float(obj_rep.mean()),
                "obj_rep": obj_rep,
            }

        order = sorted((k for k, _l, _d in STRATEGIES), key=lambda k: -summ[k]["obj"])
        best_key, second_key = order[0], order[1]
        rank_of = {k: i for i, k in enumerate(order)}
        best_row_index = next(i for i, (k, _l, _d) in enumerate(STRATEGIES) if k == best_key)

        # is the gap between first and second bigger than the noise?
        gap_1_2 = summ[best_key]["obj"] - summ[second_key]["obj"]
        paired = summ[best_key]["obj_rep"] - summ[second_key]["obj_rep"]
        se = float(np.std(paired, ddof=1) / math.sqrt(len(paired))) if len(paired) > 1 else 0.0
        noise_floor = max(2.0 * se, 0.02 * max(abs(summ[best_key]["obj"]), 1e-9))
        ambiguous = abs(gap_1_2) <= noise_floor

        # -- verdict labels, tones and the headline, all from the ranking ----
        def verdict_for(key: str) -> str:
            if key == best_key:
                return "Recommended" if not ambiguous else "Recommended - a close call"
            if ambiguous and rank_of[key] == 1:
                return "Close second - a judgement call, not a calculation"
            shortfall = max(summ[best_key]["obj"] - summ[key]["obj"], 0.0)
            return (f"Not recommended - {_fmt_gap(shortfall, objective)} less "
                    f"{word} over {quarters} quarters")

        agg = summ["aggressive_match"]
        if best_key == "aggressive_match":
            headline_delta = summ[best_key]["obj"] - summ[second_key]["obj"]
            headline_dir = _direction_word(headline_delta)
            headline = (
                f"Match every cut - on these numbers it earns the most {word}: "
                f"{_fmt_obj(summ[best_key]['obj'], objective)} over {quarters} quarters, "
                f"{_vs_phrase(headline_delta, objective, summ[second_key]['label'])}."
            )
        else:
            headline_delta = agg["obj"] - summ[best_key]["obj"]
            headline_dir = _direction_word(headline_delta)
            headline = (
                f"{summ[best_key]['label']} earns the most {word}: "
                f"{_fmt_obj(summ[best_key]['obj'], objective)} over {quarters} quarters. "
                f"Matching every cut earns {_fmt_obj(agg['obj'], objective)}, "
                f"{_vs_phrase(headline_delta, objective, summ[best_key]['label'])}."
            )
        if ambiguous:
            headline += (f" The top two are close enough that this is a judgement call, "
                         f"not a calculation.")

        # -- dead-band sweep -------------------------------------------------
        ctx.say("Testing how far the rival's undercut can move before the answer changes...",
                phase="simulate", progress=0.84)
        current_undercut = H.as_float(H.p(P, "competitor_undercut_pct", 12))
        sweep_reps = _sweep_replicates(n, quarters)
        sweep = _dead_band(cols, P, ctx, current_undercut, objective, sweep_reps)
        sweep_points = [s["undercut"] for s in sweep]
        band = [s["undercut"] for s in sweep if s["best"] == best_key]
        lo = min(band) if band else None
        hi = max(band) if band else None
        covers_all = bool(band) and lo <= sweep_points[0] + 1e-9 and hi >= sweep_points[-1] - 1e-9
        ref_label = summ[best_key]["label"]

        def winner_outside(points_subset):
            for s in points_subset:
                if s["best"] != best_key:
                    return summ[s["best"]]["label"]
            return "another strategy"

        if not band:
            band_text = ("The shorter sweep could not reproduce the recommendation at any "
                         "undercut we tested, so the three strategies are too close to call.")
            breakeven_found, breakeven_pct = False, None
            breakeven_text = ("Not found - the top strategies are within noise of each other "
                              "everywhere we tested.")
        elif covers_all:
            band_text = (f"{ref_label} stays the best call everywhere we tested, from "
                         f"{lo:.0f}% to {hi:.0f}% undercut.")
            breakeven_found, breakeven_pct = False, None
            breakeven_text = (f"Not found between {sweep_points[0]:.0f}% and "
                              f"{sweep_points[-1]:.0f}% - the recommendation does not change "
                              f"anywhere in that range.")
        elif lo > sweep_points[0] + 1e-9:
            band_text = (f"{ref_label} is the best call once the rival undercuts by "
                         f"{lo:.0f}% or more; below that it is not.")
            breakeven_found, breakeven_pct = True, lo
            breakeven_text = (f"{ref_label} only becomes the better call once the rival "
                              f"undercuts by more than {lo:.0f}%; below that, "
                              f"{winner_outside(sweep[:sweep_points.index(lo)])} earns more.")
        else:
            band_text = (f"{ref_label} is the best call up to {hi:.0f}% undercut; above that "
                         f"it is not.")
            breakeven_found, breakeven_pct = True, hi
            after = sweep[sweep_points.index(hi) + 1:]
            breakeven_text = (f"{ref_label} stops being the better call once the rival "
                              f"undercuts by more than {hi:.0f}%; above that, "
                              f"{winner_outside(after)} earns more.")
        ctx.say(breakeven_text, phase="summarize", progress=0.9)

        # -- result ----------------------------------------------------------
        n_sims = (reps * quarters * len(STRATEGIES)
                  + sweep_reps * quarters * len(STRATEGIES) * len(sweep_points))
        res = H.new_result(ctx, META.id, META.title,
                           "Agent-based market simulation (virtual buyers, three suppliers)",
                           n_sims)

        res.headline = headline
        res.method_plain = (
            "We built a virtual market out of your accounts. Each buyer has a budget, a "
            "price sensitivity and a supplier they use today. Every quarter the rivals set "
            "a price, you answer with one of three strategies, and each buyer picks a "
            "supplier - the cheapest offer usually wins, but habit, the hassle of switching "
            f"and a priority-support promise all count. We then played all "
            f"{_period_phrase(quarters)} out {reps} times for each strategy, so every number "
            f"below is an average across "
            f"{reps} different versions of the future, with the likely range shown around it. "
            "We then ranked the three strategies on the measure you chose and re-ran the "
            "market at a range of rival undercuts to see how far the answer travels before "
            "it changes."
        )

        # -- KPI cards, coloured by the ranking ------------------------------
        def profit_kpi(key: str, label: str) -> "H.KPI":
            s = summ[key]
            rank = rank_of[key]
            if key == best_key:
                plain = (f"The best of the three on {word}: "
                         f"{_vs_phrase(gap_1_2, objective, summ[second_key]['label'])}.")
            else:
                gap = summ[best_key]["obj"] - s["obj"]
                plain = (f"{_fmt_gap(gap, objective)} less {word} over {quarters} quarters "
                         f"than {ref_label}.")
            return H.kpi(
                label, H.money(s["profit"], compact=True), _rank_tone(rank, len(STRATEGIES)),
                plain,
                comparison=f"{s['asp_delta'] * 100:+.1f}% price vs today, "
                           f"{H.pct(s['share'])} of accounts served",
            )

        best_s = summ[best_key]
        res.kpis = [
            H.kpi(
                "Best call on these numbers", best_s["label"], "good",
                f"It scores best on {word} across {reps} versions of the next {quarters} quarters.",
                comparison=(_vs_phrase(gap_1_2, objective, summ[second_key]['label'])
                            if not ambiguous else
                            f"too close to call against {summ[second_key]['label']}"),
            ),
            profit_kpi("aggressive_match", "Profit if you match every cut"),
            profit_kpi("defensible_niche", "Profit if you defend a niche"),
            profit_kpi("hold_price", "Profit if you hold the line"),
            H.kpi("Top-tier accounts you keep", H.pct(best_s["ret"]),
                  "good" if best_s["ret"] * 100 >= target_ret else "warn",
                  f"Of the top accounts you serve today, this many are still yours at the end "
                  f"under the best call.",
                  comparison=f"target {H.pct_num(target_ret)}"),
            H.kpi("Accounts lost to rivals", f"{best_s['churn']:,.0f}", "neutral",
                  f"Accounts that walked to a competitor across {reps} versions of the two "
                  f"years under the best call.",
                  comparison=f"{(summ['aggressive_match']['churn']):,.0f} if you match everything"),
            H.kpi("Break-even rival undercut",
                  f"{breakeven_pct:.0f}%" if breakeven_found else "none in 0-40%",
                  "warn" if breakeven_found else "neutral", breakeven_text),
            H.kpi("How far the rival can move before the answer changes",
                  f"{lo:.0f}%-{hi:.0f}% undercut" if band else "no range found", "neutral",
                  band_text + " Only the rival's undercut is moved here; everything else "
                              "stays as you set it."),
        ]

        # -- charts ----------------------------------------------------------
        qx = [f"Q{i + 1}" for i in range(quarters)]
        colours = {"aggressive_match": "#ef4444", "defensible_niche": "#22c55e",
                   "hold_price": "#64748b"}

        scale = 1e6 if objective == "profit" else 1.0

        def sweep_value(v: float) -> float:
            return H.r2(v / scale) if objective == "profit" else H.r2(v * 100.0)

        sweep_y_label = ("Profit over the horizon ($M)" if objective == "profit"
                         else f"{word.capitalize()} (%)")
        sweep_annotations = []
        if breakeven_found:
            pt = next(s for s in sweep if abs(s["undercut"] - breakeven_pct) < 1e-9)
            sweep_annotations.append({
                "x": breakeven_pct, "y": sweep_value(pt["scores"][best_key]),
                "text": f"best call changes at {breakeven_pct:.0f}%",
            })

        res.charts = [
            H.chart(
                "asp_paths", "line", "What happens to your price",
                subtitle=f"Average across {reps} versions of the future; the shaded pair is the likely range",
                x_label="Quarter", y_label="Average price you charge",
                x=qx,
                series=[
                    {"name": "Match every cut",
                     "values": [H.r2(v / 1000.0) for v in results["aggressive_match"]["m"]["asp"]["mean"]],
                     "color": colours["aggressive_match"]},
                    {"name": "Defensible niche",
                     "values": [H.r2(v / 1000.0) for v in results["defensible_niche"]["m"]["asp"]["mean"]],
                     "color": colours["defensible_niche"]},
                    {"name": "Hold the line",
                     "values": [H.r2(v / 1000.0) for v in results["hold_price"]["m"]["asp"]["mean"]],
                     "color": colours["hold_price"], "dash": True},
                    {"name": "Best call - likely low",
                     "values": [H.r2(v / 1000.0) for v in results[best_key]["m"]["asp"]["p5"]],
                     "color": "#86efac", "dash": True},
                    {"name": "Best call - likely high",
                     "values": [H.r2(v / 1000.0) for v in results[best_key]["m"]["asp"]["p95"]],
                     "color": "#86efac", "dash": True},
                ],
                annotations=[{"x": qx[-1],
                              "y": H.r2(results["aggressive_match"]["m"]["asp"]["mean"][-1] / 1000.0),
                              "text": f"matching ends {agg['asp_delta'] * 100:+.0f}% vs today"}],
                plain_meaning="Each line is the average price you charge each quarter under one "
                              "strategy. The band is the range of outcomes across the versions we "
                              "played for the best call. A line that ends lower means more of a "
                              "discount was given away.",
            ),
            H.chart(
                "ebitda_paths", "line", "What happens to profit",
                subtitle="Profit per quarter, after the cost to serve and overheads",
                x_label="Quarter", y_label="Profit per quarter",
                x=qx,
                series=[{"name": results[k]["label"],
                         "values": [H.r2(v / 1e6) for v in results[k]["m"]["ebitda"]["mean"]],
                         "color": colours[k]} for k, _l, _d in STRATEGIES],
                annotations=[{"x": qx[-1],
                              "y": H.r2(results[best_key]["m"]["ebitda"]["mean"][-1] / 1e6),
                              "text": f"best call: {best_s['label']}"}],
                plain_meaning="Profit per quarter under each strategy. The line that ends highest "
                              "is the one that earns the most; which line that is comes from the "
                              "simulation, not from a rule of thumb.",
            ),
            H.chart(
                "scoreboard", "grouped_bar", "The scoreboard",
                subtitle="The three strategies side by side",
                x_label="", y_label="",
                x=["Price vs today", "Top accounts kept",
                   f"Profit over {_period_phrase(quarters)}", "Accounts served"],
                series=[
                    {"name": "Match every cut",
                     "values": [H.r2(agg["asp_delta"] * 100), H.r2(agg["ret"] * 100),
                                H.r2(agg["profit"] / 1e6), H.r2(agg["share"] * 100)],
                     "color": colours["aggressive_match"]},
                    {"name": "Defensible niche",
                     "values": [H.r2(summ["defensible_niche"]["asp_delta"] * 100),
                                H.r2(summ["defensible_niche"]["ret"] * 100),
                                H.r2(summ["defensible_niche"]["profit"] / 1e6),
                                H.r2(summ["defensible_niche"]["share"] * 100)],
                     "color": colours["defensible_niche"]},
                    {"name": "Hold the line",
                     "values": [H.r2(summ["hold_price"]["asp_delta"] * 100),
                                H.r2(summ["hold_price"]["ret"] * 100),
                                H.r2(summ["hold_price"]["profit"] / 1e6),
                                H.r2(summ["hold_price"]["share"] * 100)],
                     "color": colours["hold_price"]},
                ],
                plain_meaning="Four measures, three strategies, all on one axis so the bars can be "
                              "compared within each group. Price is a change versus today, top "
                              "accounts kept and accounts served are percentages, and profit over "
                              "the whole period is in millions.",
            ),
            H.chart(
                "tier_share", "grouped_bar", "Where you win and where you let go",
                subtitle=f"Share of each tier served, under the best call ({best_s['label']})",
                x_label="Account tier", y_label="Share of accounts served by you",
                x=list(cols["tier_order"]),
                series=[
                    {"name": "Today",
                     "values": [H.r2(float((cols["incumbent"][cols["tier_code"] == ti] == 0).mean()) * 100)
                                for ti in range(len(cols["tier_order"]))],
                     "color": "#94a3b8"},
                    {"name": f"After {_period_phrase(quarters)}",
                     "values": [H.r2(float(results[best_key]["final_supplier"][cols["tier_code"] == ti].mean()) * 100)
                                for ti in range(len(cols["tier_order"]))],
                     "color": colours[best_key]},
                ],
                plain_meaning="How much of each tier you serve today against how much you serve "
                              f"after {_period_phrase(quarters)} under the best call. Tiers where "
                              "the second bar is much lower are the ones the strategy chose to "
                              "give up.",
            ),
            H.chart(
                "churn_paths", "line", "Accounts walking out of the door",
                subtitle="Accounts lost to a rival, quarter by quarter",
                x_label="Quarter", y_label="Accounts lost in that quarter",
                x=qx,
                series=[{"name": results[k]["label"],
                         "values": [H.r2(v) for v in results[k]["m"]["churned"]["mean"]],
                         "color": colours[k]} for k, _l, _d in STRATEGIES],
                plain_meaning="Accounts lost to a rival each quarter under each strategy. A "
                              "strategy that loses fewer accounts is not automatically better - "
                              "the profit chart is what decides that.",
            ),
            H.chart(
                "dead_band", "line", "Where the recommendation flips",
                subtitle=f"Only the rival's undercut moves; everything else stays where it is "
                         f"today. {sweep_reps} versions of the future at each point.",
                x_label="How far the rival undercuts you", y_label=sweep_y_label,
                x=[H.r2(s["undercut"]) for s in sweep],
                series=[{"name": results[k]["label"],
                         "values": [sweep_value(s["scores"][k]) for s in sweep],
                         "color": colours[k]} for k, _l, _d in STRATEGIES],
                annotations=sweep_annotations,
                plain_meaning=f"Each line is one strategy's {word} as the rival's undercut "
                              f"changes. Where the lines cross, the best call changes. "
                              f"{breakeven_text} This is a one-factor test: only the undercut "
                              f"moves, so treat the crossing point as a guide, not a promise.",
            ),
        ]

        # -- tables ----------------------------------------------------------
        res.tables = [
            H.table(
                "The three strategies compared",
                f"Every figure is an average of {reps} simulated versions of the next "
                f"{quarters} quarters. Ranked on {word}.",
                ["Strategy", "Price vs today", f"Revenue ({_period_short(quarters)})",
                 f"Profit ({_period_short(quarters)})",
                 "Profit margin", "Top-tier kept", "Accounts served", "Moves to a rival",
                 "Verdict"],
                [[results[k]["label"],
                  f"{summ[k]['asp_delta'] * 100:+.1f}%",
                  H.money(summ[k]["revenue"], compact=True),
                  H.money(summ[k]["profit"], compact=True),
                  H.pct(summ[k]["margin_rate"]),
                  H.pct(summ[k]["ret"]),
                  H.pct(summ[k]["share"]),
                  f"{summ[k]['churn']:,.0f}",
                  verdict_for(k)] for k, _l, _d in STRATEGIES],
                highlight=[best_row_index],
                note=f"Row {best_row_index + 1} is the best call on {word}.",
            ),
        ]

        # per-tier outcome for the best strategy
        tier_order = cols["tier_order"]
        sup_mean = results[best_key]["final_supplier"]        # P(with us) at the end
        sla_pct = H.as_float(H.p(P, "sla_cost_pct", 1.5)) / 100.0 * list_price
        sla_applies = (best_key == "defensible_niche")
        price_t = float(results[best_key]["m"]["asp"]["mean"][-1])
        tier_rows = []
        for ti, tname in enumerate(tier_order):
            m_t = cols["tier_code"] == ti
            m_start = m_t & (cols["incumbent"] == 0)
            m_rival = m_t & (cols["incumbent"] != 0)
            start_us = float(m_start.sum())
            kept = float(sup_mean[m_start].sum()) / max(start_us, 1e-9)
            won = float(sup_mean[m_rival].sum())
            units_t = cols["units"][m_t]
            cost_t = cols["cost"][m_t] + (sla_pct * (cols["tier_code"][m_t] <= 1)
                                          if sla_applies else 0.0)
            rev_t = float((price_t * units_t * sup_mean[m_t]).sum())
            gm_t = float(((price_t - cost_t) * units_t * sup_mean[m_t]).sum())
            sup_t = float((cols["support"][m_t] / 4.0 * sup_mean[m_t]).sum())
            tier_rows.append([
                tname,
                f"{int(m_t.sum())}",
                f"{start_us:,.0f}",
                f"{float(sup_mean[m_t].sum()):,.0f}",
                f"{min(max(kept, 0.0), 1.0) * 100:.0f}%",
                f"{won:,.0f}",
                H.money(rev_t * quarters, compact=True),
                H.money((gm_t - sup_t) * quarters, compact=True),
                H.money(float(cols["support"][m_t].sum()) / 4.0 * quarters / max(m_t.sum(), 1),
                        compact=True),
                "Defend" if float(cols["margin"][m_t].mean()) >= 0.30 else "Let go",
            ])
        res.tables.append(H.table(
            f"What happens to each tier under the best call ({best_s['label']})",
            "Where you defend, where you let go, and what each tier earns you.",
            ["Tier", "Accounts", "Yours today",
             f"Yours after {_period_short(quarters)}",
             "Kept (of yours today)", "Won from rivals",
             f"Revenue ({_period_short(quarters)})",
             f"Profit ({_period_short(quarters)})",
             "Support cost per account", "Call"],
            tier_rows,
            highlight=[i for i, r in enumerate(tier_rows) if r[-1] == "Defend"],
            note="Profit is after the cost to serve that account.",
        ))
        res.tables.append(H.table(
            f"Quarter by quarter - best call ({best_s['label']})",
            "Your price, the rival's price, and what that earns.",
            ["Quarter", "Your price", "Rival price", "Accounts served", "Revenue", "Profit"],
            [[qx[i],
              H.money(results[best_key]["m"]["asp"]["mean"][i]),
              H.money(float(np.mean([w["history"][i]["comp_price"]
                                     for w in results[best_key]["worlds"]]))),
              H.pct(results[best_key]["m"]["share"]["mean"][i]),
              H.money(results[best_key]["m"]["revenue"]["mean"][i], compact=True),
              H.money(results[best_key]["m"]["ebitda"]["mean"][i], compact=True)]
             for i in range(quarters)],
        ))
        res.tables.append(H.table(
            "How far the rival's undercut can move before the answer changes",
            f"A one-factor test: only the rival's undercut moves, {sweep_reps} versions of the "
            f"future at each point. The row for today's undercut is highlighted.",
            ["Rival undercut", "Match every cut", "Defensible niche", "Hold the line",
             "Best call", "Still the recommendation?"],
            [[f"{s['undercut']:.0f}%",
              _fmt_obj(s["scores"]["aggressive_match"], objective),
              _fmt_obj(s["scores"]["defensible_niche"], objective),
              _fmt_obj(s["scores"]["hold_price"], objective),
              results[s["best"]]["label"],
              "Yes" if s["best"] == best_key else "No"] for s in sweep],
            highlight=[i for i, s in enumerate(sweep)
                       if abs(s["undercut"] - current_undercut) < 1e-9],
            note=breakeven_text,
        ))

        # -- story, all of it read off the numbers ---------------------------
        strategy_stories = []
        for key, label, desc in STRATEGIES:
            s = summ[key]
            tag = ("the best call" if key == best_key
                   else f"ranked {rank_of[key] + 1} of {len(STRATEGIES)}")
            strategy_stories.append(H.story(
                f"What '{label}' does",
                f"{desc} Over {quarters} quarters it leaves your average price "
                f"{s['asp_delta'] * 100:+.1f}% versus today, serves {H.pct(s['share'])} of the "
                f"accounts, keeps {H.pct(s['ret'])} of the top tier and banks "
                f"{H.money(s['profit'], compact=True)} of profit - {tag} on {word}.",
                f"{H.money(s['profit'], compact=True)} of profit over {quarters} quarters",
            ))

        share_gap = (agg["share"] - best_s["share"]) * 100.0
        profit_gap = agg["profit"] - best_s["profit"]
        if best_key == "aggressive_match" and objective == "profit":
            trade_detail = (
                f"Matching every cut wins on both counts here: it serves the most accounts "
                f"({H.pct(agg['share'])}) and banks the most profit "
                f"({H.money(agg['profit'], compact=True)}). On these numbers there is no "
                f"trade-off to make.")
            trade_evidence = (f"{H.money(agg['profit'], compact=True)} of profit, "
                              f"{H.pct(agg['share'])} of accounts")
        elif best_key == "aggressive_match":
            profit_best_key = max((k for k, _l, _d in STRATEGIES),
                                  key=lambda k: summ[k]["profit"])
            short = abs(agg["profit"] - summ[profit_best_key]["profit"])
            trade_detail = (
                f"Matching every cut is the best call on {word} and serves the most accounts "
                f"({H.pct(agg['share'])}). The catch: it banks "
                f"{H.money(agg['profit'], compact=True)} of profit, "
                f"{H.money(short, compact=True)} less than "
                f"{summ[profit_best_key]['label']}.")
            trade_evidence = (f"{H.money(agg['profit'], compact=True)} of profit against "
                              f"{H.money(summ[profit_best_key]['profit'], compact=True)}")
        else:
            trade_detail = (
                f"Matching every cut wins {share_gap:+.1f} points of account share against "
                f"the best call, and {_fmt_gap(profit_gap, objective)} "
                f"{_direction_word(profit_gap)} {word} over {quarters} quarters. "
                + (f"That is {H.money(abs(profit_gap) / abs(share_gap), compact=True)} of "
                   f"profit for every extra point of share."
                   if abs(share_gap) > 1e-6 else
                   "It wins no extra share at all."))
            trade_evidence = (f"{share_gap:+.1f} points of share for "
                              f"{_fmt_gap(profit_gap, objective)} {_direction_word(profit_gap)}")
        strategy_stories.append(H.story("The trade in plain numbers", trade_detail,
                                        trade_evidence))

        if best_key == "aggressive_match":
            bottom_detail = (
                f"Match the rival's cuts. Doing so banks "
                f"{H.money(agg['profit'], compact=True)} of profit over {quarters} quarters, "
                f"{_vs_phrase(gap_1_2, objective, summ[second_key]['label'])}. "
                f"The accounts it keeps are worth more than the discount it gives away.")
        else:
            bottom_detail = (
                f"Do not match every cut. {ref_label} banks "
                f"{_fmt_obj(best_s['obj'], objective)} of {word} over {quarters} quarters, "
                f"and matching every cut gives up "
                f"{_fmt_gap(best_s['obj'] - agg['obj'], objective)} of it. Defend the accounts "
                f"worth keeping and let the price-chasers go.")
        if ambiguous:
            bottom_detail += (f" Be aware the top two are within the noise of the simulation, "
                              f"so treat this as a judgement call rather than a precise answer.")
        strategy_stories.append(H.story("The bottom line", bottom_detail,
                                        f"{_fmt_obj(best_s['obj'], objective)} for {ref_label} "
                                        f"vs {_fmt_obj(agg['obj'], objective)} for matching"))

        res.story = [
            H.story(
                "The market we built",
                f"We put {n:,} buyers in a room with you and two rivals. They are not all "
                f"alike: the most profitable accounts barely react to price, while the "
                f"cheapest ones chase it hard. Today you serve "
                f"{H.pct(float((cols['incumbent'] == 0).mean()))} of them.",
                f"{H.money(float(cols['volume'].sum()), compact=True)} of annual spend in play",
            ),
            *strategy_stories,
        ]

        # -- risk and recommendation, from the same ranking ------------------
        risk = 25.0
        if best_s["ret"] * 100 < target_ret:
            risk += 20.0
        if best_s["profit"] <= 0:
            risk += 30.0
        if ambiguous:
            risk += 15.0
        if abs(gap_1_2) <= 0.10 * max(abs(best_s["obj"]), 1e-9):
            risk += 10.0
        if band and (current_undercut < lo - 1e-9 or current_undercut > hi + 1e-9):
            risk += 15.0
        risk = float(np.clip(risk, 0.0, 100.0))

        min_margin_pct = H.as_float(H.p(P, "min_margin_after_match_pct", 12))
        if breakeven_found and breakeven_pct is not None:
            watch = (f"Re-check monthly: the answer flips once the rival undercuts by more "
                     f"than {breakeven_pct:.0f}%.")
        else:
            watch = (f"Re-check monthly: the answer held everywhere from "
                     f"{sweep_points[0]:.0f}% to {sweep_points[-1]:.0f}% undercut.")

        if best_key == "aggressive_match":
            if objective == "profit":
                rec_headline = ("Authorise matching the rival's cuts - on these numbers it earns "
                                "more profit than holding firm.")
            else:
                rec_headline = (f"Authorise matching the rival's cuts - on these numbers it is "
                                f"the best call on {word}.")
            actions = [
                "Tell sales to match the rival's cut on the accounts we want to keep.",
                f"Cap the match at {H.pct_num(min_margin_pct)} margin - below that, walk away.",
                "Keep the priority-support promise, but charge for it where the account will pay.",
                watch,
            ]
        elif best_key == "defensible_niche":
            rec_headline = ("Do not authorise blanket price matching. Match only where the "
                            "account commits multi-year, and bundle priority support instead."
                            if objective == "profit" else
                            f"Do not authorise blanket price matching. On these numbers "
                            f"{ref_label} is the best call on {word}.")
            actions = [
                "Decline blanket price matching. Tell sales the answer is no by default.",
                f"Authorise a matched price only where the account signs a multi-year "
                f"commitment and the deal still clears {H.pct_num(min_margin_pct)} margin.",
                "Bundle priority support on the top two tiers instead of cutting price.",
                "Let the cheapest, most price-sensitive accounts go, and redeploy the support "
                "cost that frees up.",
                watch,
            ]
        else:
            rec_headline = ("Hold list price everywhere. On these numbers both matching and "
                            "niche defence earn less."
                            if objective == "profit" else
                            f"Hold list price everywhere - on these numbers it is the best call "
                            f"on {word}.")
            actions = [
                "Hold list price and let the price-chasers go.",
                "Redeploy the support cost freed up by the accounts that leave.",
                "Protect the top tier with service, not discounts.",
                watch,
            ]

        rationale = [
            (f"{ref_label} scores best on {word}: {_fmt_obj(best_s['obj'], objective)} over "
             f"{quarters} quarters, against {_fmt_obj(agg['obj'], objective)} for matching every "
             f"cut."),
            (f"Matching every cut leaves your average price {agg['asp_delta'] * 100:+.1f}% "
             f"versus today and serves {H.pct(agg['share'])} of the accounts; {ref_label} "
             f"serves {H.pct(best_s['share'])} at {best_s['asp_delta'] * 100:+.1f}%."),
            breakeven_text,
        ]
        if ambiguous:
            rationale.append(
                f"The gap between the top two is inside the simulation's own noise "
                f"({_fmt_gap(gap_1_2, objective)}), so this is a judgement call.")

        res.recommendation = Recommendation(
            headline=rec_headline,
            rationale=rationale,
            actions=actions,
            risk_score=round(risk, 1),
            risk_label=H.risk_label(risk),
            confidence=("High" if rows and not ambiguous else "Medium"),
            tradeoff=(
                f"You are choosing {_fmt_obj(best_s['obj'], objective)} of {word} with "
                f"{ref_label} over {_fmt_obj(agg['obj'], objective)} with matching every cut - "
                f"a difference of {_fmt_gap(best_s['obj'] - agg['obj'], objective)}."
            ),
        )

        # -- assumptions and warnings ----------------------------------------
        roles_found = [r.label for r in META.input_roles if H.has(rows, r.key)]
        res.assumptions = [
            f"{source}; each buyer keeps buying the same volume each quarter.",
            f"List price {H.money(list_price)} per unit; margin "
            f"{H.pct(float(cols['margin'].mean()))} on average, so cost is "
            f"{H.pct(1 - float(cols['margin'].mean()))} of list.",
            f"Rivals undercut your previous average price by "
            f"{H.pct_num(H.as_float(H.p(P, 'competitor_undercut_pct', 12)))} and close "
            f"{H.as_float(H.p(P, 'competitor_reaction_speed', 0.5)) * 100:.0f}% of the gap "
            f"each quarter, down to a floor of "
            f"{H.pct_num(H.as_float(H.p(P, 'max_market_discount_pct', 20)))} off list.",
            f"Buyers weigh price about ten times more heavily than habit; a "
            f"{H.pct_num(H.as_float(H.p(P, 'sla_value_pct', 6)))} priority-support promise "
            f"counts like a discount of half that size, and costs "
            f"{H.pct_num(H.as_float(H.p(P, 'sla_cost_pct', 1.5)))} of margin.",
            f"{H.as_float(H.p(P, 'multiyear_commitment_share_pct', 40)):.0f}% of top-tier "
            f"accounts are willing to commit multi-year (60% of that rate in the mid tier).",
            f"Overheads of {H.money(H.as_float(H.p(P, 'fixed_cost_base', 5_000_000)), compact=True)} "
            f"a year are charged against profit.",
            f"The recommendation maximises {word}; change it and the answer may change with it.",
            f"Robustness check: the rival's undercut alone moves, from "
            f"{sweep_points[0]:.0f}% to {sweep_points[-1]:.0f}% in {DEADBAND_STEP_PCT:.0f}-point "
            f"steps, with {sweep_reps} versions of the future at each point. Every other "
            f"assumption is held where it is today.",
            f"{reps} independent versions of {quarters} quarters per strategy; seed "
            f"{ctx.request.seed} (re-running gives identical numbers).",
        ]
        if roles_found:
            res.assumptions.append(
                "Columns read from your file: " + ", ".join(roles_found) + ".")

        if not rows:
            res.warnings.append(
                "No account file was mapped, so the market is generated from your assumptions "
                "rather than your real account book. Upload the account list to make this "
                "specific to you.")
        if best_s["ret"] * 100 < target_ret:
            res.warnings.append(
                f"Top-tier retention comes out at {H.pct(best_s['ret'])} under the best call, "
                f"below your {H.pct_num(target_ret)} target. Consider extending priority "
                f"support or a deeper multi-year offer to that tier.")
        if best_s["profit"] <= 0:
            res.warnings.append(
                f"Even the best call leaves profit at or below zero "
                f"({H.money(best_s['profit'], compact=True)} over {quarters} quarters) - on "
                f"these numbers the market is barely worth serving at any of the three prices.")
        if agg["profit"] <= 0 < best_s["profit"]:
            res.warnings.append(
                "Matching every cut drives profit to zero or below - at that point you are "
                "paying for the privilege of serving these accounts.")
        if ambiguous:
            res.warnings.append(
                f"The top two strategies are within {_fmt_gap(gap_1_2, objective)} of each "
                f"other, which is inside the simulation's own noise. Read this as a judgement "
                f"call, not a calculation.")
        if not band:
            res.warnings.append(
                "We could not reproduce this recommendation at any rival undercut we tested, "
                "which usually means the strategies are too close to separate.")
        if note:
            res.warnings.append(note)

        # -- guard the invariants --------------------------------------------
        _self_check(res, {
            "objective": objective,
            "objective_values": {k: summ[k]["obj"] for k, _l, _d in STRATEGIES},
            "headline_delta": headline_delta,
            "best_row_index": best_row_index,
            "proportions": {
                "share of accounts served (best call)": best_s["share"],
                "top-tier retention (best call)": best_s["ret"],
                **{f"share of accounts served ({k})": summ[k]["share"]
                   for k, _l, _d in STRATEGIES},
                **{f"top-tier retention ({k})": summ[k]["ret"] for k, _l, _d in STRATEGIES},
            },
        })

        ctx.narrator.done(
            f"Done. {ref_label} scores best on {word}: "
            f"{_fmt_obj(best_s['obj'], objective)} over {quarters} quarters.")
        res.log = ctx.narrator.events()
        return res
