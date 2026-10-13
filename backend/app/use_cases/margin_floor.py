"""USE CASE 5 - Margin Protection & Floor Setting: Stress-Testing Cost Volatility.

Model: Latin Hypercube Sampling.

The question a spreadsheet cannot answer: you sign a three-year, fixed-price
deal at 15% margin today. Cloud, licences, contractors and freight do not stay
still. A spreadsheet shows one cost path - usually today's. LHS walks every
slice of every cost input at once and runs 1,000 complete multi-year cash flow
projections, so you can say "there is a X% chance this contract breaks our
margin floor before it ends" - and, more usefully, "this is the annual
escalator that makes that risk acceptable".
"""
from __future__ import annotations

import numpy as np

from ..core.contracts import ParamSpec, Recommendation, RoleSpec, UseCaseMeta
from ..core.registry import UseCase, register
from ..engine import latin_hypercube as lhs
from ..engine import stats as st
from . import _helpers as H

META = UseCaseMeta(
    id="margin_floor",
    title="Margin Protection & Floor Setting: Stress-Testing Cost Volatility",
    subtitle="Will a fixed-price contract still clear your margin floor in year 3?",
    category="Margin & Cost",
    icon="\U0001F6E1",
    models=["Latin Hypercube Sampling"],
    business_question=(
        "What is the chance this fixed-price contract breaks our mandatory margin floor, "
        "and what annual escalator makes that risk acceptable?"
    ),
    plain_summary=(
        "Your cost to deliver - hosting, licences, labour, freight, currency - does not "
        "stay flat over a multi-year contract. We lay out every possible path those costs "
        "could take, walk a thousand complete multi-year cash flows through them, and tell "
        "you the chance of ending up under your minimum margin. Then we solve for the "
        "annual price escalator that brings that risk back inside your tolerance."
    ),
    story_steps=[
        "Break the contract's cost into its moving parts",
        "Give every cost part and the currency a range of possible yearly moves",
        "Walk 1,000 complete multi-year cash flows through those ranges",
        "Find the year where the margin floor is most at risk",
        "Solve for the annual escalator that makes the risk acceptable",
    ],
    data_granularity="cost",
    sample_dataset="contract_costs.csv",
    input_roles=[
        RoleSpec(key="contract_id", label="Contract ID", dtype="identifier", required=False,
                 aliases=["id", "contract", "agreement_id", "deal_id"]),
        RoleSpec(key="customer", label="Customer name", dtype="categorical", required=False,
                 aliases=["account", "client", "customer_name"]),
        RoleSpec(key="contract_value", label="Contract value (revenue)", dtype="numeric",
                 required=False,
                 aliases=["value", "tcv", "arr", "revenue", "price", "amount"],
                 fallback=8_000_000,
                  help="What the customer pays per year, before any escalator."),
        RoleSpec(key="year", label="Year", dtype="numeric", required=False,
                 aliases=["fy", "period", "contract_year"],
                  help="Needed if you want us to learn cost drift from your own history."),
        RoleSpec(key="cost_component", label="Cost component", dtype="categorical",
                 required=False,
                 aliases=["component", "cost_type", "category", "cost_category", "bucket"],
                  help="e.g. Cloud Hosting, Licensing, Labor, Logistics."),
        RoleSpec(key="base_cost", label="Cost in that year", dtype="numeric", required=False,
                 aliases=["cost", "amount", "spend", "cogs", "value"]),
        RoleSpec(key="margin_pct", label="Margin achieved (%)", dtype="numeric",
                 required=False,
                 aliases=["margin", "gross_margin", "margin_percent"],
                  help="Optional. Used as the starting margin if supplied."),
        # QA DF-11: two UC3 edge cases were in the file but not modelled at all.
        RoleSpec(key="capacity_cliff_flag", label="Capacity cliff flag", dtype="numeric",
                 required=False,
                 aliases=["cliff", "capacity_cliff", "surge_flag", "step_change"],
                  help="1 if this record hits a step-change capacity surge, 0 if not. "
                      "Used to decide what share of the cost base takes the cliff."),
        RoleSpec(key="attrition_turnover_count", label="Staff leavers (count)",
                 dtype="numeric", required=False,
                 aliases=["turnover", "attrition", "leavers", "headcount_leavers"],
                  help="How many people left and had to be backfilled. Each leaver costs "
                      "a multiple of their salary to replace."),
    ],
    parameters=[
        ParamSpec(key="contract_value", label="Contract value (per year)", kind="number",
                  default=8_000_000, min=0, step=100_000, unit="$",
                  group="The contract",
                  help="What the customer pays. Fixed unless you add an escalator."),
        ParamSpec(key="contract_years", label="Contract length", kind="integer",
                  default=3, min=1, max=10, step=1, unit="years", group="The contract", help="How many years the fixed price runs for.",),
        ParamSpec(key="current_margin_pct", label="Margin you signed at", kind="number",
                  default=15, min=-20, max=90, step=1, unit="%", group="The contract",
                  help="Year-1 margin at today's costs. 15 means 15%."),
        ParamSpec(key="hosting_share_pct", label="Cloud hosting - share of cost",
                  kind="number", default=30, min=0, max=100, step=1, unit="%",
                  group="Cost mix", help="Share of your total cost to deliver that cloud hosting represents.",),
        ParamSpec(key="hosting_volatility_pct", label="Cloud hosting - volatility",
                  kind="number", default=22, min=0, max=80, step=1, unit="%",
                  group="Cost mix",
                  help="How far this cost line can swing in a year."),
        ParamSpec(key="licensing_share_pct", label="Third-party licensing - share of cost",
                  kind="number", default=25, min=0, max=100, step=1, unit="%",
                  group="Cost mix", help="Share of your total cost to deliver that third-party licensing represents.",),
        # QA TM-10: only the cloud-hosting slider had an explanation; the other
        # three did not. Same wording for all four.
        ParamSpec(key="licensing_volatility_pct", label="Third-party licensing - volatility",
                  kind="number", default=12, min=0, max=80, step=1, unit="%",
                  group="Cost mix",
                  help="How far this cost line can swing over the whole contract. Read it "
                       "as the total swing, not a yearly figure: 12 means the line ends up "
                       "somewhere within about 12% of what you planned."),
        ParamSpec(key="labor_share_pct", label="Labor - share of cost", kind="number",
                  default=30, min=0, max=100, step=1, unit="%", group="Cost mix", help="Share of your total cost to deliver that labour represents.",),
        ParamSpec(key="labor_volatility_pct", label="Labor - volatility", kind="number",
                  default=9, min=0, max=80, step=1, unit="%", group="Cost mix",
                  help="How far this cost line can swing over the whole contract. Read it "
                       "as the total swing, not a yearly figure: 9 means the line ends up "
                       "somewhere within about 9% of what you planned."),
        ParamSpec(key="logistics_share_pct", label="Logistics - share of cost",
                  kind="number", default=15, min=0, max=100, step=1, unit="%",
                  group="Cost mix", help="Share of your total cost to deliver that logistics represents.",),
        ParamSpec(key="logistics_volatility_pct", label="Logistics - volatility",
                  kind="number", default=18, min=0, max=80, step=1, unit="%",
                  group="Cost mix",
                  help="How far this cost line can swing over the whole contract. Read it "
                       "as the total swing, not a yearly figure: 18 means the line ends up "
                       "somewhere within about 18% of what you planned."),
        # QA DF-11 - the two UC3 edge cases QA found absent from the model.
        ParamSpec(key="capacity_cliff_pct", label="Capacity cliff - step jump in hosting",
                  kind="number", default=0, min=0, max=200, step=5, unit="%",
                  group="Edge cases",
                  help="A capacity surge is not a smooth drift - it is a step. Set this to "
                       "the one-off jump in hosting cost when you cross a capacity "
                       "threshold. 0 switches it off."),
        ParamSpec(key="capacity_cliff_year", label="Year the cliff hits", kind="integer",
                  default=3, min=1, max=10, step=1, group="Edge cases",
                  help="Which contract year the capacity step lands in. It stays in place "
                       "for every year after that - it does not fade."),
        ParamSpec(key="attrition_backfill_pct", label="Cost to backfill a leaver",
                  kind="number", default=0, min=0, max=300, step=5, unit="%",
                  group="Edge cases",
                  help="Replacing someone costs more than their salary - recruiting, "
                       "onboarding and lost productivity. Set it as a share of one "
                       "person's annual cost. 0 switches it off."),
        ParamSpec(key="labor_headcount", label="People on this contract", kind="integer",
                  default=10, min=1, max=5000, step=1, group="Edge cases",
                  help="Used to turn a number of leavers into a share of the labour cost."),
        ParamSpec(key="expected_turnovers", label="Leavers you expect per year",
                  kind="number", default=0, min=0, max=500, step=1, group="Edge cases",
                  help="Only used if your file has no leaver column."),
        ParamSpec(key="fx_volatility_pct", label="Currency volatility", kind="number",
                  default=6, min=0, max=60, step=1, unit="%", group="Cost mix",
                  help="How much exchange rates move your total cost."),
        # QA TM-09: the acronym was never expanded.
        ParamSpec(key="annual_cola_pct", label="Yearly price rise (cost-of-living uplift)",
                  kind="number", default=0, min=0, max=20, step=0.5, unit="%",
                  group="Guardrails",
                  help="How much the price you charge goes up each year. 0 means the price "
                       "is frozen for the whole contract, so every cost increase comes "
                       "straight out of your margin."),
        # QA TM-11: the one input every result is judged against had no explanation.
        ParamSpec(key="margin_floor_pct", label="Mandatory margin floor", kind="number",
                  default=12, min=-10, max=90, step=1, unit="%", group="Guardrails",
                  help="The lowest margin this deal is allowed to reach. If the simulation "
                       "says the margin drops below this number in any year, the deal "
                       "breaks your rule - every risk figure on the result page is measured "
                       "against it."),
        ParamSpec(key="tolerance_pct", label="Risk you are willing to accept",
                  kind="number", default=10, min=1, max=50, step=1, unit="%",
                  group="Guardrails",
                  help="The most chance of breaking the floor you will sign off on."),
        # QA TM-08: "NPV" was never expanded anywhere. Spell the concept out.
        ParamSpec(key="discount_rate_pct", label="Discount rate for future cash",
                  kind="number",
                  default=8, min=0, max=30, step=1, unit="%", group="Guardrails",
                  help="Money you will receive in later years is worth less than money "
                       "today. This is the yearly haircut we apply - 8 means money three "
                       "years out counts for about 80% of its face value."),
        ParamSpec(key="n_simulations", label="Cash flow projections", kind="integer",
                  default=1000, min=200, max=20000, step=100, group="Engine", help="How many complete cash-flow projections to walk through. More gives a steadier answer.",),
    ],
)

# component template: (label, share key, volatility key)
_TEMPLATE = [
    ("Cloud Hosting", "hosting_share_pct", "hosting_volatility_pct", 22.0),
    ("Third-party Licensing", "licensing_share_pct", "licensing_volatility_pct", 12.0),
    ("Labor", "labor_share_pct", "labor_volatility_pct", 9.0),
    ("Logistics", "logistics_share_pct", "logistics_volatility_pct", 18.0),
]

_MIN_VOL = 0.04          # a cost line never looks perfectly stable
_MAX_COMPS = 6
_MIN_STEPS = 4           # year-on-year steps needed before we trust an observed swing


def _frac(v, default_pct: float) -> float:
    f = H.as_float(v, default_pct)
    return f / 100.0 if f > 1.0 else f


def _vol_lookup(priors: dict, name: str) -> float:
    """Best matching typed-in volatility for a cost line read from a file."""
    n = str(name).strip().lower()
    for label, v in priors.items():
        l = label.lower()
        if n == l or n in l or l in n:
            return v
    return float(np.mean(list(priors.values()))) if priors else 0.12


@register
class MarginFloorUseCase(UseCase):
    meta = META

    def run(self, req, ctx) -> "H.RunResult":
        P = ctx.params
        rows = ctx.rows or []

        # ===================================================================
        # 1. READ THE CONTRACT
        # ===================================================================
        ctx.say("Reading the contract and how its cost is built up...", phase="read",
                progress=0.05)

        value = max(0.0, H.as_float(H.p(P, "contract_value", 8_000_000)))
        years = int(min(max(H.as_float(H.p(P, "contract_years", 3)), 1), 10))
        floor_pct = 100.0 * _frac(H.p(P, "margin_floor_pct", 12), 12.0)
        floor_frac = floor_pct / 100.0
        tol_pct = 100.0 * _frac(H.p(P, "tolerance_pct", 10), 10.0)
        tol_frac = tol_pct / 100.0
        cola = _frac(H.p(P, "annual_cola_pct", 0), 0.0)
        disc = _frac(H.p(P, "discount_rate_pct", 8), 8.0)
        n_sim = int(min(max(H.as_float(H.p(P, "n_simulations", 1000)), 100), 20000))
        fx_vol = _frac(H.p(P, "fx_volatility_pct", 6), 6.0)
        margin0 = _frac(H.p(P, "current_margin_pct", 15), 15.0)

        # ===================================================================
        # 2. COST MIX: FROM YOUR FILE IF POSSIBLE, ELSE FROM THE KNOBS
        # ===================================================================
        comps: list[dict] = []          # {name, share, drift, vol, base}
        data_note = "the cost mix you set"
        file_rows = 0
        thin: list[str] = []

        if H.has(rows, "cost_component") and H.has(rows, "base_cost"):
            names = H.cat(rows, "cost_component", default="")
            costs = H.num(rows, "base_cost", 0.0)
            vals = H.num(rows, "contract_value", 0.0)
            yrs = H.num(rows, "year", 0.0) if H.has(rows, "year") else np.zeros(len(rows))
            ok = (costs > 0) & (np.array([bool(str(n).strip()) for n in names]))
            if ok.sum() >= 4:
                file_rows = int(ok.sum())
                names = [str(n).strip() for n, o in zip(names, ok) if o]
                costs = costs[ok]
                vals = vals[ok]
                yrs = yrs[ok]
                # cost as a share of contract value, so contracts of different
                # sizes can be compared on the same footing
                scale = np.where(vals > 0, vals, 0.0)
                ratio = np.where(scale > 0, costs / np.maximum(scale, 1e-9), costs)
                if (scale <= 0).all():
                    ratio = costs / max(costs.sum(), 1e-9)

                uniq_names: list[str] = []
                for n in names:
                    if n not in uniq_names:
                        uniq_names.append(n)
                uniq_names = uniq_names[:_MAX_COMPS]

                uniq_years = sorted({float(y) for y in yrs if y > 0})
                cell: dict[tuple[str, float], list[float]] = {}
                for n, y, r in zip(names, yrs, ratio):
                    if n not in uniq_names:
                        continue
                    cell.setdefault((n, float(y)), []).append(float(r))

                # what the user typed in - used whenever the file cannot actually
                # show us how a cost line moves
                priors = {label: _frac(H.p(P, vk, dv), dv)
                          for label, _sk, vk, dv in _TEMPLATE}

                if len(uniq_years) >= 2:
                    for n in uniq_names:
                        series = [float(np.mean(cell[(n, y)]))
                                  for y in uniq_years if (n, y) in cell]
                        if len(series) < 2:
                            continue
                        g = [series[i + 1] / series[i] - 1.0
                             for i in range(len(series) - 1) if series[i] > 0]
                        base_ratio = series[0]
                        drift = float(np.clip(np.mean(g), -0.10, 0.35)) if g else 0.0
                        vol_obs = float(np.std(g, ddof=1)) if len(g) >= 2 else 0.0
                        # A swing measured off only a couple of year-steps is thin
                        # evidence, and a file that shows a perfectly smooth cost
                        # path is really telling us "no movement was recorded",
                        # not "there is no risk". In both cases we keep the
                        # volatility the user typed in.
                        learned = len(g) >= _MIN_STEPS and vol_obs >= 0.02
                        vol = (float(np.clip(vol_obs, _MIN_VOL, 0.60)) if learned
                               else _vol_lookup(priors, n))
                        if not learned:
                            thin.append(n)
                        comps.append({"name": n, "ratio": base_ratio, "drift": drift,
                                      "vol": vol})
                    data_note = (f"{len(uniq_names)} cost lines observed across "
                                 f"{len(uniq_years)} years in your file")
                    ctx.say(f"Learned the cost mix and its drift from your file: "
                            f"{len(uniq_names)} cost lines across {len(uniq_years)} years.",
                            phase="read", progress=0.14,
                            detail="; ".join(f"{c['name']} drift "
                                             f"{c['drift'] * 100:.1f}%/yr, swing "
                                             f"{c['vol'] * 100:.0f}%" for c in comps))
                    if thin:
                        ctx.say(f"Your file shows {len(thin)} cost line(s) moving too "
                                f"smoothly to measure a swing from ({', '.join(thin)}), "
                                f"so we kept the volatility you typed in for them.",
                                phase="read", level="warn", progress=0.15)
                else:
                    for n in uniq_names:
                        vals_n = [v for (nm, _y), v in cell.items() if nm == n]
                        if not vals_n:
                            continue
                        comps.append({"name": n, "ratio": float(np.mean(vals_n)),
                                      "drift": 0.0, "vol": 0.0})
                    data_note = f"{len(uniq_names)} cost lines from your file"
                    ctx.say(f"Took the cost mix from your file ({len(uniq_names)} lines), "
                            f"but it has only one year in it - so we could not learn how "
                            f"fast costs drift. We used your volatility settings.",
                            phase="read", level="warn", progress=0.14)
                    for c in comps:
                        c["vol"] = _vol_lookup(priors, c["name"])
                    thin = [c["name"] for c in comps]

        if not comps:
            for label, sk, vk, _d in _TEMPLATE:
                share = _frac(H.p(P, sk, 25), 25.0)
                vol = _frac(H.p(P, vk, 12), 12.0)
                if share <= 0:
                    continue
                comps.append({"name": label, "ratio": share, "drift": 0.0, "vol": vol})
            ctx.say(f"No cost file was mapped, so we used the cost mix you set: "
                    f"{', '.join(c['name'] for c in comps)}.", phase="read",
                    level="warn", progress=0.14)

        total_ratio = float(sum(c["ratio"] for c in comps)) or 1.0
        for c in comps:
            c["share"] = c["ratio"] / total_ratio

        margin_param = margin0
        value_param = value
        margin_from_file = False
        # QA DF-10 / UX-04: a value the user TYPED at Step 4 must win over the
        # value learned from their file. `explicit_params` holds the keys the
        # user actually set (filled by the API from the request), so we can tell
        # "the user asked for this" apart from "this is just the default".
        explicit = set(getattr(ctx.request, "explicit_params", None) or [])
        if H.has(rows, "margin_pct") and "current_margin_pct" not in explicit:
            m_obs = H.num(rows, "margin_pct", np.nan)
            m_obs = np.where(m_obs > 1.0, m_obs, m_obs * 100.0)
            m_obs = m_obs[np.isfinite(m_obs)]
            if m_obs.size:
                margin0 = float(np.clip(m_obs.mean(), -50.0, 95.0)) / 100.0
                margin_from_file = True
        # QA UX-04: the override used to be disclosed only in the collapsed
        # "things to keep an eye on" panel at the bottom of the result. Say it in
        # the live run log as well, at the moment it happens, exactly as UC1 does.
        if H.has(rows, "margin_pct") and "current_margin_pct" in explicit:
            ctx.say(
                f"You typed the starting margin, so we used your {H.pct(margin0)} rather "
                f"than the average in your file.", phase="read", level="warn")
        elif H.has(rows, "margin_pct") and margin_from_file:
            ctx.say(
                f"Your file reports an average starting margin of {H.pct(margin0)}, so we "
                f"used that rather than the default.", phase="read")

        value_from_file = False
        if H.has(rows, "contract_value"):
            v_obs = H.num(rows, "contract_value", 0.0)
            v_obs = v_obs[v_obs > 0]
            if v_obs.size:
                value = float(np.mean(v_obs))
                value_from_file = True

        total_cost0 = value * (1.0 - margin0)
        for c in comps:
            c["base"] = total_cost0 * c["share"]

        mix_desc = ", ".join("{} {}".format(c["name"], H.pct(c["share"])) for c in comps)
        ctx.say(f"Year 1: you bill {H.money(value, compact=True)} and it costs you "
                f"{H.money(total_cost0, compact=True)} - a {H.pct(margin0)} margin.",
                phase="read", progress=0.18, detail=f"cost mix: {mix_desc}")

        # ===================================================================
        # 3. LATIN HYPERCUBE: ONE DRAW = ONE COMPLETE MULTI-YEAR CASH FLOW
        # ===================================================================
        # Year 1 is priced on the costs you can already see; every year after that
        # picks up a fresh cost shock. (A one-year contract has no "after", so its
        # single year is the one that carries the shock.)
        shock_from = 0 if years <= 1 else 1
        growth_years = max(years - shock_from, 1)

        rng = ctx.rng("margin_floor", "lhs")
        n_c = len(comps)
        # A volatility figure is read as "how far this line can end up from plan by
        # the end of the contract", so it is spread evenly over the years that carry
        # a shock. Without this, simply signing for one more year would inflate the
        # risk on its own.
        spread = 1.0 / np.sqrt(float(growth_years))
        dims = []
        for ci, c in enumerate(comps):
            mu = float(np.log(max(1.0 + c["drift"], 0.2)))
            sd = float(c["vol"]) * spread
            for t in range(years):
                dims.append({"name": f"inf_{ci}_{t}", "dist": "lognormal",
                             "mean": mu, "sd": sd})
        fx_sd = float(fx_vol) * spread
        for t in range(years):
            dims.append({"name": f"fx_{t}", "dist": "lognormal",
                         "mean": 0.0, "sd": fx_sd})

        corr = []
        for t in range(years):
            for a in range(n_c):
                for b in range(a + 1, n_c):
                    corr.append((f"inf_{a}_{t}", f"inf_{b}_{t}", 0.30))

        ctx.say(f"Laying out {n_sim:,} complete {years}-year cash flows across every "
                f"possible path your costs could take...", phase="simulate", progress=0.28,
                detail=f"{len(dims)} cost and currency inputs sampled on an even grid, so "
                       f"no slice of any range is missed")

        draws = lhs.sample(n_sim, dims, rng, correlations=corr)
        tick = ctx.narrator.tracker(n_sim, "Walked {pct}% of the cash flows...",
                                    phase="simulate")
        for k in range(0, n_sim, max(1, n_sim // 10)):
            tick(k)

        inf = np.ones((n_sim, years, n_c))
        for ci in range(n_c):
            for t in range(years):
                inf[:, t, ci] = np.asarray(draws[f"inf_{ci}_{t}"], dtype=float)
        cum = np.ones_like(inf)                             # (n, years, comps)
        for t in range(shock_from, years):
            cum[:, t, :] = np.prod(inf[:, shock_from:t + 1, :], axis=1)
        fx = np.stack([np.asarray(draws[f"fx_{t}"], dtype=float) for t in range(years)],
                      axis=1)                               # (n, years)
        fx[:, :shock_from] = 1.0
        base_vec = np.array([c["base"] for c in comps], dtype=float)
        cost_comp = cum * base_vec[None, None, :]                   # (n, years, comps)

        # --- QA DF-11: the two edge cases QA found missing --------------------
        # (1) A capacity cliff is a STEP, not a drift: from the cliff year
        #     onwards, hosting cost jumps once and stays there.
        names = [str(c["name"]).lower() for c in comps]
        hi = next((i for i, nm in enumerate(names)
                   if "host" in nm or "cloud" in nm or "infra" in nm), 0)
        li = next((i for i, nm in enumerate(names)
                   if "lab" in nm or "staff" in nm or "people" in nm), None)

        cliff_pct = H.as_float(H.p(P, "capacity_cliff_pct", 0)) / 100.0
        cliff_year = int(H.p(P, "capacity_cliff_year", 3))
        cliff_share = 1.0
        if H.has(rows, "capacity_cliff_flag"):
            flags = H.num(rows, "capacity_cliff_flag", 0.0)
            flags = np.where(flags > 0.5, 1.0, 0.0)
            if flags.size:
                cliff_share = float(np.clip(flags.mean(), 0.0, 1.0))
        cliff_live = cliff_pct > 0

        # (2) Attrition: each leaver costs a multiple of one person's annual cost
        #     to backfill, so turnover is a cost multiplier on labour.
        backfill = H.as_float(H.p(P, "attrition_backfill_pct", 0)) / 100.0
        headcount = max(int(H.p(P, "labor_headcount", 10)), 1)
        turnovers = H.as_float(H.p(P, "expected_turnovers", 0))
        if H.has(rows, "attrition_turnover_count"):
            tv = H.num(rows, "attrition_turnover_count", 0.0)
            tv = tv[np.isfinite(tv)]
            if tv.size:
                turnovers = float(np.nansum(tv))
        attrition_live = backfill > 0 and turnovers > 0

        if cliff_live:
            step = 1.0 + cliff_pct * cliff_share
            for t in range(max(cliff_year, 1) - 1, years):
                cost_comp[:, t, hi] *= step
            ctx.say(f"Added a capacity cliff: hosting cost steps up "
                    f"{cliff_pct * cliff_share * 100:.0f}% in year "
                    f"{max(cliff_year, 1)} and stays there.", phase="simulate",
                    progress=0.55)
        if attrition_live and li is not None:
            mult = 1.0 + backfill * (turnovers / headcount)
            cost_comp[:, :, li] *= mult
            ctx.say(f"Added attrition: {turnovers:.0f} leavers on a team of "
                    f"{headcount}, each costing {backfill * 100:.0f}% of a salary to "
                    f"backfill, lifts labour cost {mult - 1:+.0%}.", phase="simulate",
                    progress=0.55)

        cost = cost_comp.sum(axis=2) * fx                           # (n, years)

        def margins_at(escalator: float) -> np.ndarray:
            rev = value * (1.0 + escalator) ** np.arange(years, dtype=float)
            return (rev[None, :] - cost) / np.maximum(rev[None, :], 1e-9)

        margin = margins_at(cola)                            # (n, years) as a fraction
        margin_pct = margin * 100.0

        # ===================================================================
        # 4. YEAR BY YEAR
        # ===================================================================
        ctx.say("Reading the results year by year...", phase="summarize", progress=0.80)

        year_rows = []
        exp_margin = np.zeros(years)
        p5 = np.zeros(years)
        p50 = np.zeros(years)
        p95 = np.zeros(years)
        p_breach = np.zeros(years)
        p_loss = np.zeros(years)
        for t in range(years):
            s = st.summarize(margin_pct[:, t])
            exp_margin[t] = s.get("mean", 0.0)
            p5[t] = s.get("p5", 0.0)
            p50[t] = s.get("p50", 0.0)
            p95[t] = s.get("p95", 0.0)
            p_breach[t] = st.prob_below(margin_pct[:, t], floor_pct)
            p_loss[t] = st.prob_below(margin_pct[:, t], 0.0)
            year_rows.append([
                f"Year {t + 1}",
                H.money(value * (1.0 + cola) ** t),
                H.money(float(cost[:, t].mean())),
                H.pct_num(exp_margin[t]),
                H.pct_num(p5[t]),
                H.pct_num(p95[t]),
                H.pct(p_breach[t]),
                H.pct(p_loss[t]),
            ])

        p_final = float(p_breach[-1])
        p_ever = float((margin_pct < floor_pct).any(axis=1).mean())
        p_any_loss = float((margin_pct < 0.0).any(axis=1).mean())
        worst_t = int(max(np.flatnonzero(p_breach >= p_breach.max() - 1e-12)))

        # NPV of margin, today's plan vs what the simulation says
        yrs_arr = np.arange(1, years + 1, dtype=float)
        disc_vec = 1.0 / (1.0 + disc) ** yrs_arr
        plan_npv = float(np.sum(value * margin0 * disc_vec))
        sim_npv = float(np.sum((value * (1.0 + cola) ** (yrs_arr - 1.0)
                                - cost.mean(axis=0)) * disc_vec))
        npv_at_risk = plan_npv - sim_npv

        # ===================================================================
        # 5. BREAKEVEN ESCALATOR
        # ===================================================================
        ctx.say("Solving for the annual escalator that brings the risk back inside your "
                "tolerance...", phase="summarize", progress=0.88)

        grid = [round(0.005 * i, 4) for i in range(0, 17)]          # 0% .. 8%
        found = None
        search_rows = []

        def _row(g: float) -> tuple[list, bool]:
            m = margins_at(g) * 100.0
            pb = float((m < floor_pct).any(axis=1).mean())
            last = float((m[:, -1] < floor_pct).mean())
            exp_last = float(m[:, -1].mean())
            npv_g = float(np.sum((value * (1.0 + g) ** (yrs_arr - 1.0)
                                  - cost.mean(axis=0)) * disc_vec))
            meets = last < tol_frac
            return ([H.pct_num(g * 100.0, 1), H.pct(last), H.pct(pb),
                     H.pct_num(exp_last), H.money(npv_g, compact=True),
                     "inside tolerance" if meets else "too risky"], meets)

        # A one-year deal has no year 2, and an escalator only bites from year 2
        # onwards - so for a one-year term there is nothing to solve for.
        escalator_bites = years > 1
        best_g = 0.0
        best_last_p = float("nan")
        if escalator_bites:
            for g in grid:
                row, meets = _row(g)
                search_rows.append(row)
                p_last = float(row[1].rstrip("%"))
                if np.isnan(best_last_p) or p_last < best_last_p:
                    best_last_p, best_g = p_last, g
                if meets and found is None:
                    found = g
            if found is None:                   # push past the search range
                for i in range(17, 41):
                    g = round(0.005 * i, 4)
                    row, meets = _row(g)
                    search_rows.append(row)
                    p_last = float(row[1].rstrip("%"))
                    if np.isnan(best_last_p) or p_last < best_last_p:
                        best_last_p, best_g = p_last, g
                    if meets:
                        found = g
                        break
        else:
            search_rows.append(_row(0.0)[0])
            ctx.say("An escalator cannot help a one-year deal - it only moves the price "
                    "from Year 2 onwards. The only lever here is the price you sign today.",
                    phase="summarize", level="warn", progress=0.94)

        breakeven = found
        # QA DF-05: this branch used to print ">0.0%", which is meaningless and
        # contradicted the note promising a row "inside tolerance". Say what is
        # actually true: nothing in the range works, here is the best it gets.
        if breakeven is None:
            top_g = round(0.005 * (len(search_rows) - 1), 4) if search_rows else 0.0
            if not escalator_bites:
                breakeven_label = "not available"
                breakeven_plain = ("An escalator only moves the price from Year 2 onwards, "
                                   "so it cannot help a one-year deal. The only lever is "
                                   "the price you sign today.")
            else:
                breakeven_label = f"none within {H.pct_num(top_g * 100.0)}"
                breakeven_plain = (
                    f"No annual escalator up to {H.pct_num(top_g * 100.0)} brings the risk "
                    f"inside your {tol_pct:.0f}% tolerance. Even at the best one we tested "
                    f"({H.pct_num(best_g * 100.0)}) the chance of breaking your floor is "
                    f"still {best_last_p:.0f}%. A bigger escalator is not the fix here - "
                    f"shorten the term, reset the margin you sign at, or add a "
                    f"renegotiation trigger.")
            ctx.say(breakeven_plain, phase="summarize", level="warn", progress=0.94)
        else:
            breakeven_label = H.pct_num(breakeven * 100.0, 1)
            breakeven_plain = (f"A {breakeven_label} annual escalator brings the risk "
                               f"inside your {tol_pct:.0f}% tolerance.")
            ctx.say(f"Found it: {breakeven_plain}", phase="summarize", progress=0.94)

        trigger_t = years - 1
        for t in range(years):
            if p_breach[t] > tol_frac:
                trigger_t = t
                break
        trigger_year = max(2, trigger_t + 1) if years >= 2 else 1

        # ===================================================================
        # 6. WHAT DRIVES THE DAMAGE
        # ===================================================================
        shares, extra = _driver_split(comps, growth_years, fx_vol)

        # ===================================================================
        # 7. ASSEMBLE
        # ===================================================================
        res = H.new_result(ctx, META.id, META.title,
                           "Latin Hypercube Sampling of multi-year cost inflation "
                           "and currency", n_sim)

        # QA UX-11: the template did not adapt at the extremes - a 100% result
        # was described as "better than even money", which understates certainty.
        if p_final >= 0.999:
            risk_phrase = "certain - it will definitely break"
        elif p_final >= 0.95:
            risk_phrase = "almost certain"
        elif p_final >= 0.5:
            risk_phrase = "better than even money"
        elif p_final >= 0.095:
            risk_phrase = "about {} in 10".format(int(round(p_final * 10.0)))
        elif p_final > 0.0:
            risk_phrase = "less than 1 in 10"
        else:
            risk_phrase = "essentially never"
        fix_clause = (
            f"a {breakeven_label} annual escalator is what brings that back "
            f"inside your {tol_pct:.0f}% tolerance" if breakeven is not None
            else (f"no escalator gets it back inside your {tol_pct:.0f}% tolerance - "
                  f"this one has to be restructured")
        )
        price_clause = (f"With no price escalator" if cola <= 1e-9
                        else f"At the {H.pct(cola)} escalator in this deal")
        res.headline = (
            f"{price_clause} there is a {H.pct(p_final)} chance "
            f"({risk_phrase}) this contract breaks your {floor_pct:.0f}% margin floor by "
            f"Year {years} - {fix_clause}."
        )
        res.method_plain = (
            f"We split your cost to deliver into its moving parts and gave each one - plus "
            f"the currency - a range of yearly moves, from the small ones to the nasty ones. "
            f"Instead of picking randomly, we walked {n_sim:,} complete {years}-year cash "
            f"flows through those ranges on an even grid, so every slice of every range gets "
            f"tested and nothing important is missed. Each run asks one question: at the end "
            f"of each year, is the margin still above your floor? We then repeat the whole "
            f"thing at every escalator from 0% upwards until the chance of breaking the floor "
            f"falls below the {tol_pct:.0f}% you said you could live with. "
            f"Costs are modelled with a 'bad news is more likely than good news' curve, so "
            f"surprise increases weigh more than surprise windfalls."
        )

        res.kpis = [
            H.kpi(f"Chance of breaking the {floor_pct:.0f}% floor by Year {years}",
                  H.pct(p_final), "bad" if p_final > tol_frac else "good",
                  f"The odds that the Year {years} margin comes in below your minimum - "
                  f"the final year is where cost increases have compounded the longest.",
                  comparison=f"your tolerance is {H.pct(tol_frac)}"),
            H.kpi("Chance of breaking it at any point", H.pct(p_ever),
                  "bad" if p_ever > tol_frac else "warn",
                  "The odds that at least one of the years comes in below the floor."),
            H.kpi("Breakeven annual escalator", breakeven_label,
                  "good" if breakeven is not None and breakeven <= 0.05 else "warn",
                  breakeven_plain,
                  comparison=f"you are asking for {H.pct(cola)}"),
            H.kpi(f"Expected margin in Year {worst_t + 1} (the worst year)",
                  H.pct_num(exp_margin[worst_t]),
                  "bad" if exp_margin[worst_t] < floor_pct else "warn",
                  f"Year {worst_t + 1} is where the floor is most at risk.",
                  comparison=f"you signed at {H.pct(margin0)}"),
            H.kpi("Margin at risk, in today's money", H.money(npv_at_risk, compact=True),
                  "bad" if npv_at_risk > 0 else "good",
                  "The profit you planned to book minus the profit this simulation says you "
                  "will actually book, in today's money.",
                  comparison=f"planned {H.money(plan_npv, compact=True)}"),
            H.kpi("Chance the contract loses money in a year", H.pct(p_any_loss),
                  "bad" if p_any_loss > 0.05 else "neutral",
                  "How often a year comes in below zero margin, not just below the floor."),
            H.kpi(f"Chance of breaking the floor in Year 1", H.pct(p_breach[0]),
                  "bad" if p_breach[0] > tol_frac else "neutral",
                  "Year 1 is the one year an escalator cannot protect - only a higher "
                  "starting price can."),
        ]
        if breakeven is not None and breakeven > cola + 1e-9:
            res.kpis.append(H.kpi(
                "Margin you recover at the breakeven escalator",
                H.pct_num(float(margins_at(breakeven)[:, -1].mean() * 100.0)
                          - exp_margin[-1]),
                "good", "How much final-year margin the escalator buys back."))

        # ---- charts -------------------------------------------------------
        res.charts = [
            H.chart(
                "margin_fan", "line", "Margin over the life of the contract",
                subtitle="The middle line is the most likely path; the band is the range",
                x_label="Contract year", y_label="Margin (%)",
                x=[f"Year {t + 1}" for t in range(years)],
                series=[
                    {"name": "A bad run (1 in 20)", "values": [H.r2(v) for v in p5],
                     "color": "#ef4444", "dash": True},
                    {"name": "Most likely", "values": [H.r2(v) for v in p50],
                     "color": "#22c55e"},
                    {"name": "A good run (1 in 20)", "values": [H.r2(v) for v in p95],
                     "color": "#3b82f6", "dash": True},
                    {"name": f"Your {floor_pct:.0f}% floor",
                     "values": [H.r2(floor_pct)] * years, "color": "#f59e0b"},
                ],
                annotations=[{"x": f"Year {worst_t + 1}", "y": H.r2(p50[worst_t]),
                              "text": f"weakest year: {H.pct(p_breach[worst_t])} chance "
                                      f"of a breach"}],
                plain_meaning="As long as the red line stays above the orange floor line you "
                              "are safe. The moment it crosses, this contract can break your "
                              "minimum margin rule."),
            H.chart(
                "worst_year_hist", "hist",
                f"Where the margin lands in Year {worst_t + 1}",
                subtitle=f"{n_sim:,} projections - the marker is your "
                         f"{floor_pct:.0f}% floor",
                x_label="Margin (%)", y_label="How often",
                x=[], series=[{"name": "Margin",
                               "values": [float(v) for v in
                                          margin_pct[:, worst_t][::max(1, n_sim // 2000)]]}],
                annotations=[{"x": H.r2(floor_pct), "y": 0, "text": "your floor"},
                             {"x": H.r2(p50[worst_t]), "y": 0, "text": "most likely"}],
                plain_meaning="Everything to the left of the floor marker is a world where "
                              "this contract has broken your minimum margin rule."),
            H.chart(
                "breach_by_year", "bar", "Chance of breaking the floor, year by year",
                subtitle=f"Your tolerance is {tol_pct:.0f}% - anything above that bar is "
                         f"not signable as-is",
                x_label="Contract year", y_label="Chance of breaking the floor (%)",
                x=[f"Year {t + 1}" for t in range(years)],
                series=[{"name": "Chance of breaking the floor",
                         "values": [H.r2(v * 100.0) for v in p_breach], "color": "#ef4444"}],
                annotations=[{"x": f"Year {trigger_t + 1}", "y": H.r2(p_breach[trigger_t] * 100.0),
                              "text": "first year over your tolerance"}],
                plain_meaning="Risk builds as the contract ages because cost increases "
                              "compound while a frozen price does not."),
            H.chart(
                "cost_drivers", "bar", "Which cost line does the damage",
                subtitle=f"Share of the uncertainty in your Year {years} cost",
                x_label="Cost line", y_label="Share of the wobble (%)",
                x=[c["name"] for c in comps] + ["Currency"],
                series=[{"name": "Share of the wobble",
                         "values": [H.r2(v * 100.0) for v in shares], "color": "#8b5cf6"}],
                plain_meaning="Fix the biggest bar first: that is where a cap, a fixed-price "
                              "commitment or a pass-through clause buys you the most "
                              "protection."),
        ]
        if breakeven is not None and breakeven > 1e-9:
            m_be = margins_at(breakeven) * 100.0
            res.charts.append(H.chart(
                "cola_effect", "grouped_bar", "What the escalator buys you",
                subtitle=f"Margin by year, frozen price versus a "
                         f"{H.pct_num(breakeven * 100.0, 1)} escalator",
                x_label="Contract year", y_label="Margin (%)",
                x=[f"Year {t + 1}" for t in range(years)],
                series=[
                    {"name": "Frozen price", "values": [H.r2(v) for v in p50],
                     "color": "#64748b"},
                    {"name": f"With a {H.pct_num(breakeven * 100.0, 1)} escalator",
                     "values": [H.r2(float(np.median(m_be[:, t]))) for t in range(years)],
                     "color": "#22c55e"},
                ],
                plain_meaning="The green bars stay above the floor. That is the whole point "
                              "of the escalator."))

        # ---- tables -------------------------------------------------------
        res.tables = [
            H.table(
                "Year by year",
                f"Every one of the {n_sim:,} projections, summarised one year at a time.",
                ["Year", "You bill", "Expected cost", "Expected margin",
                 "A bad run (1 in 20)", "A good run (1 in 20)", "Chance below the floor",
                 "Chance of a loss"],
                year_rows, highlight=[worst_t],
                note=f"Row {worst_t + 1} is the year most likely to break your floor."),
            H.table(
                "Searching for the breakeven escalator",
                f"The same {n_sim:,} projections re-run with a different yearly price "
                f"increase. We measure the risk in Year {years}, the year the cost "
                f"increases have compounded the longest.",
                ["Annual escalator", f"Chance of breaking the floor in Year {years}",
                 "Chance of breaking it at some point", f"Expected margin in Year {years}",
                 "Margin in today's money", "Verdict"],
                search_rows,
                highlight=([int(round(breakeven / 0.005))]
                           if breakeven is not None
                           else ([int(round(best_g / 0.005))]
                                 if escalator_bites and search_rows else [])),
                note=("The highlighted row is the first one marked 'inside tolerance' - "
                      "that is the escalator to ask for."
                      if breakeven is not None else
                      ("No row reaches 'inside tolerance'. The highlighted row is the best "
                       "we found, and it is still too risky - so the escalator is not the "
                       "lever that fixes this contract."
                       if escalator_bites else
                       "An escalator only moves the price from Year 2 onwards, so there is "
                       "nothing to solve for on a one-year deal."))),
            H.table(
                "What goes into the model",
                f"Cost mix from {data_note}.",
                ["Cost line", "Share of cost", "Year 1 cost", "Expected drift per year",
                 "Volatility", f"Share of the Year {years} wobble",
                 f"Expected extra cost by Year {years}"],
                [[c["name"], H.pct(c["share"]), H.money(c["base"]),
                  H.pct_num(c["drift"] * 100.0), H.pct_num(c["vol"] * 100.0),
                  H.pct(shares[i]), H.money(extra[i])]
                 for i, c in enumerate(comps)]
                + [["Currency", "-", "-", "0%", H.pct_num(fx_vol * 100.0),
                    H.pct(shares[-1]), H.money(extra[-1])]],
            ),
            H.table(
                "The bottom line",
                "What you planned versus what the simulation says, in today's money.",
                ["", "Planned (margin held flat)", "What the simulation says", "Difference"],
                [
                    ["Margin in Year 1", H.pct(margin0), H.pct_num(exp_margin[0]),
                     H.pct_num(exp_margin[0] - margin0 * 100.0)],
                    [f"Margin in Year {years}", H.pct(margin0), H.pct_num(exp_margin[-1]),
                     H.pct_num(exp_margin[-1] - margin0 * 100.0)],
                    ["Margin in today's money", H.money(plan_npv, compact=True),
                     H.money(sim_npv, compact=True), H.money(-npv_at_risk, compact=True)],
                    [f"Chance of breaking the floor in Year {years}", "0%", H.pct(p_final),
                     H.pct(p_final)],
                    ["Chance of breaking it at some point", "0%", H.pct(p_ever),
                     H.pct(p_ever)],
                ],
                highlight=[2], note="Row 3 is the value at risk from cost volatility."),

        ]

        # ---- story --------------------------------------------------------
        res.story = [
            H.story(
                "What you signed and what it costs",
                # QA UX-11: when the margin starts BELOW the floor this printed a
                # bare negative ("-62.0% of room"), which reads as nonsense.
                (f"You bill {H.money(value, compact=True)} and it costs you "
                 f"{H.money(total_cost0, compact=True)} in year 1, leaving a "
                 f"{H.pct(margin0)} margin - which is already BELOW your {floor_pct:.0f}% "
                 f"floor, so this deal is unsignable as it stands."
                 if margin0 * 100.0 - floor_pct < 0 else
                 f"You bill {H.money(value, compact=True)} and it costs you "
                 f"{H.money(total_cost0, compact=True)} in year 1, leaving a "
                 f"{H.pct(margin0)} margin. Your floor is {floor_pct:.0f}%, so there is "
                 f"{H.pct_num(margin0 * 100.0 - floor_pct)} of room before this deal "
                 f"becomes unsignable."),
                (f"{H.pct_num(abs(margin0 * 100.0 - floor_pct))} below the floor"
                 if margin0 * 100.0 - floor_pct < 0 else
                 f"{H.pct_num(margin0 * 100.0 - floor_pct)} of headroom")),
            H.story(
                "Why a fixed price is the problem",
                f"Costs do not sit still. The biggest mover is "
                f"{comps[int(np.argmax([c['vol'] for c in comps]))]['name']}, which can swing "
                f"{H.pct_num(max(c['vol'] for c in comps) * 100.0)} in a year, and the "
                f"increases compound - year 3 carries three years of them. A frozen price "
                f"carries none of that.",
                f"{H.pct_num(sum(c['drift'] * c['share'] for c in comps) * 100.0)} expected "
                f"cost drift per year"),
            H.story(
                "Where it breaks",
                f"Across {n_sim:,} complete cash flows, the Year {years} margin comes in "
                f"below {floor_pct:.0f}% in {H.pct(p_final)} of them, and at least one year "
                f"breaks it in {H.pct(p_ever)}. Year {worst_t + 1} is the weakest, with an "
                f"expected margin of {H.pct_num(exp_margin[worst_t])}. In today's money that "
                f"is {H.money(npv_at_risk, compact=True)} of planned profit at risk.",
                f"{H.pct(p_final)} chance of a breach in Year {years}"),
            H.story(
                "What fixes it",
                (f"A {breakeven_label} annual escalator pulls the Year {years} risk back "
                 f"inside your {tol_pct:.0f}% tolerance. Beyond the escalator, buy the right "
                 f"to reopen pricing: if, at the start of Year {trigger_year}, delivered "
                 f"margin is below {floor_pct:.0f}%, you renegotiate or re-scope."
                 if breakeven is not None else
                 f"No escalator up to 20% fixes this. The contract needs restructuring: a "
                 f"cost pass-through on {comps[int(np.argmax(shares[:n_c]))]['name']}, a "
                 f"shorter term, or a higher starting price."),
                f"escalator {breakeven_label}, trigger at Year {trigger_year}"),
        ]

        # ---- recommendation -----------------------------------------------
        risk_score = float(np.clip(100.0 * (p_ever / max(tol_frac * 3.0, 0.03)), 0.0, 100.0))
        label = H.risk_label(risk_score)

        if p_ever <= tol_frac and cola <= 1e-9:
            rec_head = (f"Approve as signed: only a {H.pct(p_ever)} chance of breaking the "
                        f"{floor_pct:.0f}% floor at any point, inside your {tol_pct:.0f}% "
                        f"tolerance.")
            actions = [
                f"Sign at {H.pct(margin0)} margin with no escalator - the numbers support it.",
                f"Review the cost mix every year: {comps[int(np.argmax(shares[:n_c]))]['name']} "
                f"is the line that would hurt first.",
            ]
        elif breakeven is not None:
            rec_head = (f"Do not sign a frozen price. Add a {breakeven_label} annual "
                        f"escalator plus a renegotiation trigger at Year {trigger_year}.")
            actions = [
                f"Add an annual cost-of-living escalator of {breakeven_label} to the price "
                f"- that alone brings the chance of breaking the floor in Year {years} from "
                f"{H.pct(p_final)} to "
                f"{H.pct(float((margins_at(breakeven)[:, -1] * 100.0 < floor_pct).mean()))}.",
                f"Add a renegotiation trigger: if delivered margin is below "
                f"{floor_pct:.0f}% at the start of Year {trigger_year}, you may reopen "
                f"pricing or re-scope the service.",
                f"An escalator cannot protect Year 1 (it starts from Year 2). If the "
                f"{H.pct(p_breach[0])} Year-1 risk is not acceptable either, raise the "
                f"starting price rather than relying on the escalator.",
                f"Cap or pass through the {comps[int(np.argmax(shares[:n_c]))]['name']} line "
                f"first: it drives {H.pct(float(np.max(shares[:n_c])))} of the cost "
                f"uncertainty.",
            ]
        else:
            rec_head = (f"Restructure before signing: a {H.pct(p_ever)} chance of breaking "
                        f"the floor cannot be fixed by an escalator alone.")
            actions = [
                f"Raise the starting margin or shorten the term - the gap between "
                f"{H.pct(margin0)} and your {floor_pct:.0f}% floor is too thin for this "
                f"level of cost volatility.",
                f"Move {comps[int(np.argmax(shares[:n_c]))]['name']} to a pass-through or a "
                f"capped index; it drives {H.pct(float(np.max(shares[:n_c])))} of the "
                f"uncertainty.",
                "Add a hard renegotiation trigger at the first year the margin would break "
                f"the floor (Year {trigger_year}).",
            ]

        res.recommendation = Recommendation(
            headline=rec_head,
            rationale=[
                f"Expected margin falls from {H.pct(margin0)} in Year 1 to "
                f"{H.pct_num(exp_margin[-1])} by Year {years}.",
                f"Chance of breaking the {floor_pct:.0f}% floor by Year {years}: "
                f"{H.pct(p_final)} against a tolerance of {H.pct(tol_frac)} "
                f"({H.pct(p_ever)} if you count every year).",
                f"Margin at risk, in today's money: {H.money(npv_at_risk, compact=True)} of the "
                f"{H.money(plan_npv, compact=True)} you planned to book.",
            ],
            actions=actions,
            risk_score=round(risk_score, 1),
            risk_label=label,
            confidence="High" if n_sim >= 1000 else "Medium",
            tradeoff=(f"Every point of escalator you win protects about "
                      f"{H.money(npv_at_risk / max(breakeven * 100.0, 1e-9) if breakeven else 0, compact=True)} "
                      f"of what that money is worth today - and every point you ask for is a point "
                      f"buyer will not wear the escalator, take the renegotiation trigger "
                      f"instead: it costs nothing upfront and gives you an exit."),
        )

        # ---- assumptions / warnings ---------------------------------------
        res.assumptions = [
            f"Contract: {H.money(value)} billed per year for {years} years, starting margin "
            f"{H.pct(margin0)}, so year-1 cost is {H.money(total_cost0)}.",
            "Cost mix: " + data_note + " - " + ", ".join(
                f"{c['name']} {H.pct(c['share'])} of cost, drift "
                f"{H.pct_num(c['drift'] * 100.0)} per year, volatility "
                f"{H.pct_num(c['vol'] * 100.0)}" for c in comps) + ".",
            f"Currency: a separate wobble of {H.pct_num(fx_vol * 100.0)} applied to the "
            f"whole cost base.",
            # QA DF-11 - state the edge cases explicitly so they are auditable.
            (f"Capacity cliff: hosting cost steps up "
             f"{cliff_pct * cliff_share * 100:.0f}% in year {max(cliff_year, 1)} and stays "
             f"there (applies to {H.pct(cliff_share)} of the cost base"
             + (", taken from your cliff flag column." if H.has(rows, "capacity_cliff_flag")
                else " - set by you, no cliff column in the file.")
             + ")" if cliff_live else
             "Capacity cliff: none modelled (set the step jump above 0% to include one)."),
            (f"Attrition: {turnovers:.0f} leavers on a team of {headcount}, each costing "
             f"{H.pct(backfill)} of a salary to backfill, lifting labour cost by "
             f"{backfill * (turnovers / headcount):+.0%}"
             + (" (from your leaver column)." if H.has(rows, "attrition_turnover_count")
                else " (set by you, no leaver column in the file).")
             if attrition_live else
             "Attrition: none modelled (set a backfill cost above 0% to include it)."),
            "Each cost line follows a 'bad news weighs more than good news' curve: the "
            "middle outcome is the drift, but surprise increases are larger than surprise "
            "windfalls.",
            f"Year 1 is priced on the cost you can see today; every year after that picks up "
            f"a fresh cost shock over {growth_years} year(s).",
            f"Each volatility figure is read as the total swing for that cost line across "
            f"the shocked years and spread evenly over them, so a "
            f"{H.pct_num(max(c['vol'] for c in comps) * 100.0)} figure means 'ends up within "
            f"about that much of plan by Year {years}', not 'that much every single year'. "
            f"Costs still compound - good and bad years stack up. Year-on-year volatility "
            f"learned from your file is floored at 4%, because we will not assume a cost "
            f"line is steadier than that. If your file holds fewer than {_MIN_STEPS - 1} "
            f"year-on-year steps, or shows an almost perfectly smooth path, we treat the "
            f"swing as unmeasured and keep the volatility you typed in instead.",
            f"Cost lines in the same year move together somewhat (they share the same "
            f"economy); years move independently of each other.",
            f"Margin floor {floor_pct:.0f}%, risk tolerance {tol_pct:.0f}%, discount rate "
            f"{H.pct(disc)} used to value future money, escalator in this run {H.pct(cola)}.",
            f"{n_sim:,} complete cash flows on an even grid; seed {ctx.request.seed} - "
            f"re-running gives identical numbers.",
        ]
        if cola <= 1e-9 and p_final > 0.15:
            # QA UX-11: adapt at the extremes (a certainty is not "even money").
            odds_txt = ("certain" if p_final >= 0.999 else
                        "almost certain" if p_final >= 0.95 else
                        "better than even money" if p_final >= 0.5 else
                        "1 in {}".format(int(round(1.0 / p_final)))
                        if p_final >= 0.02 else "small but real")
            res.warnings.append(
                f"With no escalator there is roughly a {odds_txt} chance of breaking your "
                f"{floor_pct:.0f}% floor by Year {years} - that is the whole risk of a "
                f"frozen price.")
        if p_any_loss > 0.02:
            res.warnings.append(
                f"There is a {H.pct(p_any_loss)} chance of an outright loss-making year, "
                f"not just a breach of the floor.")
        if breakeven is not None and breakeven > 0.08:
            res.warnings.append(
                f"The escalator this contract needs ({breakeven_label}) is above the 8% "
                f"range you asked us to search. Expect the buyer to refuse it - negotiate "
                f"the renegotiation trigger as the fallback.")
        if not H.has(rows, "cost_component"):
            res.warnings.append(
                "No cost file was mapped, so the cost mix and every volatility are your "
                "settings rather than observed history. Upload contract costs by component "
                "and year to learn them from evidence.")
        elif not H.has(rows, "year"):
            res.warnings.append(
                "Your cost file has no year column, so we could not learn how fast costs "
                "drift - we used zero drift and your volatility settings.")
        if thin:
            res.warnings.append(
                "Your file shows {} moving too smoothly to measure a real swing from, so "
                "we kept your volatility setting for {} rather than reading near-zero "
                "risk off a short history.".format(
                    ", ".join(thin), "it" if len(thin) == 1 else "them"))
        if margin_from_file and abs(margin0 - margin_param) >= 0.01:
            res.warnings.append(
                "Your file reports an average margin of {}, so that - not the {} you "
                "typed - sets year-1 cost. If you meant the figure you typed, clear the "
                "margin column from the mapping and run again.".format(
                    H.pct(margin0), H.pct(margin_param)))
        if (not margin_from_file) and H.has(rows, "margin_pct"):
            res.warnings.append(
                "You typed {} at Step 4, so that is the margin we used - it overrides the "
                "average your file reports.".format(H.pct(margin_param)))
        if value_from_file and abs(value - value_param) >= 1.0:
            res.warnings.append(
                "Contract value was read from your file ({} average), overriding the {} "
                "you typed.".format(H.money(value, compact=True),
                                    H.money(value_param, compact=True)))

        ctx.narrator.done(
            f"Done. {H.pct(p_final)} chance of breaking the floor by Year {years}; "
            + (f"you need a {breakeven_label} annual escalator." if breakeven is not None
               else "no escalator fixes it - this one needs restructuring."))
        res.log = ctx.narrator.events()
        return res


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------
def _driver_split(comps, years: int, fx_vol: float):
    """Split the uncertainty in the final-year cost across the cost lines.

    Returns (shares, extra_cost) where `shares` has one entry per component
    plus one for currency, and `extra_cost` is the expected extra dollars each
    one adds to the final year's bill.
    """
    n = len(comps)
    base = np.array([c["base"] for c in comps], dtype=float)
    vol = np.array([c["vol"] for c in comps], dtype=float)
    m = np.array([(1.0 + c["drift"]) ** years for c in comps], dtype=float) \
        * np.exp(0.5 * vol ** 2)
    var_c = (m ** 2) * (np.exp(vol ** 2) - 1.0)

    ef = float(np.exp(0.5 * fx_vol ** 2))
    var_f = float(np.exp(fx_vol ** 2) * (np.exp(fx_vol ** 2) - 1.0))
    ef2 = var_f + ef ** 2

    e_s = float(np.sum(base * m))
    var_s = float(np.sum(base ** 2 * var_c))
    comp_contrib = (base ** 2 * var_c) * ef2
    fx_contrib = (var_s + e_s ** 2) * var_f
    total = float(np.sum(comp_contrib) + fx_contrib)
    if total <= 0:
        shares = np.zeros(n + 1)
        shares[-1] = 1.0
    else:
        shares = np.concatenate([comp_contrib, [fx_contrib]]) / total

    extra = np.concatenate([base * (m - 1.0), [e_s * (ef - 1.0)]])
    return shares, extra
