"""USE CASE - Approval Bottleneck: Where Deals Actually Get Stuck.

Every finance team has a queue nobody can see: discount requests, contract
reviews and legal checks waiting for a handful of approvers. The queue is why
deals "slip". This model builds the queue explicitly - arrivals, a finite
number of approvers, how long each step really takes, and the deals that give
up and walk away - so you can say which step is to blame and what fixing it is
worth in dollars.

Then we re-run the same queue with one more approver, with small requests
auto-approved, and with the runner-up step parallelised, and compare.
"""
from __future__ import annotations

import numpy as np

from ..core.contracts import ParamSpec, Recommendation, RoleSpec, UseCaseMeta
from ..core.registry import UseCase, register
from ..engine import discrete_event as de
from ..engine import distributions as dist
from ..engine import stats as st
from . import _helpers as H

#: How many quarters we run and pool. One quarter is only a hundred-odd
#: requests, and the numbers that matter here (the slowest tenth, the revenue
#: at risk) live in the tail - about fifteen requests. Pooling a handful of
#: quarters makes those stable without slowing anything down.
REPLICATES = 5

#: Stages we use when you have no history.
STAGE_KEYS = ["deal_desk", "cf_approval", "legal"]
STAGE_NAMES = {"deal_desk": "Deal Desk Review", "cf_approval": "CF Approval",
               "legal": "Legal"}


def _lognormal_spec(mean_days: float, sigma: float) -> dict:
    """Lognormal with the given arithmetic mean and spread.

    The engine parameterises a lognormal by the mean and spread of the
    *underlying* bell curve, so the arithmetic mean needs converting.
    """
    s = max(float(mean_days), 1e-3)
    sg = float(np.clip(sigma, 0.05, 1.2))
    return {"dist": "lognormal", "mean": float(np.log(s) - 0.5 * sg * sg), "sd": sg}


def _classify_stage(name: str) -> str:
    n = str(name).strip().lower()
    if "legal" in n or "contract" in n:
        return "legal"
    if "deal desk" in n or "desk" in n or "discount" in n or "quote" in n:
        return "deal_desk"
    if "cf" in n or "cfo" in n or "finance" in n or "approval" in n or "fp&a" in n:
        return "cf_approval"
    return ""


def _cap_for(stage_name: str, P) -> int:
    key = _classify_stage(stage_name)
    if key == "deal_desk":
        return max(int(H.p(P, "deal_desk_capacity", 3)), 1)
    if key == "legal":
        return max(int(H.p(P, "legal_capacity", 2)), 1)
    return max(int(H.p(P, "cf_capacity", 2)), 1)


def _svc_for(stage_name: str, P) -> tuple[float, float]:
    """(mean days, spread) for a stage, from the parameters."""
    key = _classify_stage(stage_name)
    if key == "deal_desk":
        return H.as_float(H.p(P, "deal_desk_days", 0.8)), H.as_float(H.p(P, "deal_desk_sd", 0.6))
    if key == "legal":
        return H.as_float(H.p(P, "legal_days", 0.9)), H.as_float(H.p(P, "legal_sd", 0.9))
    return H.as_float(H.p(P, "cf_approval_days", 1.1)), H.as_float(H.p(P, "cf_sd", 0.8))


def _mean_cycle(service_mean: float, sigma: float, capacity: int, rate: float,
                n: int, rng: np.random.Generator) -> float:
    """Door-to-door time for one stage on its own (waiting included)."""
    res = de.run(
        n_entities=n,
        stages=[{"name": "s", "dist": _lognormal_spec(service_mean, sigma),
                 "capacity": capacity}],
        arrival_spec={"dist": "exponential", "rate": max(rate, 1e-6)},
        rng=rng,
    )
    rec = res["records"]
    if not rec:
        return service_mean
    return float(np.mean([r["cycle_time"] for r in rec]))


def _calibrate(target_days: float, sigma: float, capacity: int, rate: float,
               n: int, rng: np.random.Generator) -> float:
    """Back out the hands-on time from the door-to-door time you observed.

    Your file almost certainly records how long a request *sat* in a stage,
    queue and all. Feeding that straight in as hands-on time would charge the
    waiting twice and make every step look impossible. So we search for the
    hands-on time that reproduces the door-to-door time you actually saw,
    given how many approvers you have.
    """
    target = max(float(target_days), 1e-3)
    lo, hi = target * 0.02, target
    if _mean_cycle(hi, sigma, capacity, rate, n, rng) <= target:
        return hi
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if _mean_cycle(mid, sigma, capacity, rate, n, rng) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _pool(reps: list[dict]) -> dict:
    """Combine several simulated quarters into one set of numbers."""
    cycles = np.concatenate([r["_cycle"] for r in reps]) if any(
        r["_cycle"].size for r in reps) else np.zeros(0)
    vals = np.concatenate([r["_values"] for r in reps]) if any(
        r["_values"].size for r in reps) else np.zeros(0)
    late = np.concatenate([r["_late"] for r in reps]) if any(
        r["_late"].size for r in reps) else np.zeros(0, dtype=bool)

    stats: list[dict] = []
    for i, s0 in enumerate(reps[0]["stage_stats"]):
        ss = [r["stage_stats"][i] for r in reps]
        stats.append({
            "stage": s0["stage"],
            "capacity": s0["capacity"],
            "n_through": int(sum(float(s["n_through"]) for s in ss)),
            "avg_wait": float(np.mean([float(s["avg_wait"]) for s in ss])),
            "p90_wait": float(np.mean([float(s["p90_wait"]) for s in ss])),
            "max_wait": float(max(float(s["max_wait"]) for s in ss)),
            "avg_service": float(np.mean([float(s["avg_service"]) for s in ss])),
            "utilisation": float(np.mean([float(s["utilisation"]) for s in ss])),
        })
    return {
        "records": [r for rp in reps for r in rp["records"]],
        "arrivals": reps[0]["arrivals"],
        "stage_stats": stats,
        "cycle_time_summary": st.summarize(cycles) if cycles.size else {},
        "throughput": float(np.mean([float(r["throughput"]) for r in reps])),
        "abandon_rate": float(np.mean([float(r["abandon_rate"]) for r in reps])),
        "_values": vals,
        "_cycle": cycles,
        "_late": late,
        "_p90": float(np.percentile(cycles, 90)) if cycles.size else 0.0,
        "_revenue_at_risk": float(np.mean([r["_risk"] for r in reps])),
        "_reps": reps,
    }


@register
class ApprovalBottleneckUseCase(UseCase):
    meta = UseCaseMeta(
        id="approval_bottleneck",
        title="Approval Bottleneck: Where Deals Actually Get Stuck",
        subtitle="Which approval step is costing you the quarter - and what is fixing it worth?",
        category="Operations",
        icon="🚦",
        models=["Discrete Event Simulation"],
        business_question="Which approval step is actually holding deals up, how much revenue is stuck behind it, and what is the cheapest fix worth?",
        plain_summary=(
            "Requests arrive, wait for one of a small number of approvers, get worked on, "
            "and move to the next step - and some give up and walk away. We rebuild that "
            "queue on the computer using your own arrival pattern and how long each step "
            "really takes, find the step that is jammed, price the revenue stuck behind it, "
            "and then test what one more approver or an auto-approval rule would be worth."
        ),
        story_steps=[
            "Read how requests arrive and how long each approval step takes",
            "Rebuild the queue with the real number of approvers",
            "Find the step that is jammed and price the revenue stuck behind it",
            "Test the fixes: one more approver, auto-approve the small ones, parallelise the runner-up",
            "Recommend the cheapest fix that gets the quarter back on track",
        ],
        data_granularity="event",
        sample_dataset="approval_flow.csv",
        input_roles=[
            RoleSpec(key="request_id", label="Request / ticket ID", dtype="identifier",
                     required=False, aliases=["id", "ticket", "case", "request", "quote_id"]),
            RoleSpec(key="submitted_day", label="Day it was submitted", dtype="numeric",
                     required=False, aliases=["day", "date", "submitted", "created_day",
                                              "arrival_day", "start_day"]),
            RoleSpec(key="stage", label="Approval step", dtype="categorical", required=False,
                     aliases=["step", "phase", "queue", "approval_step", "status"]),
            RoleSpec(key="days_in_stage", label="Days it spent in that step", dtype="numeric",
                     required=False, aliases=["duration", "days", "time_in_stage", "cycle_days",
                                              "elapsed_days"]),
            RoleSpec(key="approver", label="Who approved it", dtype="categorical", required=False,
                     aliases=["owner", "reviewer", "assigned_to", "agent", "resource"]),
            RoleSpec(key="deal_value", label="Deal value", dtype="numeric", required=False,
                     aliases=["value", "amount", "arr", "tcv", "deal_size"]),
        ],
        parameters=[
            ParamSpec(key="n_requests", label="Requests to put through the process",
                      kind="integer", default=200, min=20, max=2000, step=20,
                      group="Volume", help="How many approval requests to play through. More gives a steadier answer.",),
            ParamSpec(key="arrival_rate_per_week", label="Requests arriving per week",
                      kind="number", default=12, min=0.5, max=200, step=1,
                      group="Volume", help="How many new requests arrive in a typical week.",),
            ParamSpec(key="quarter_days", label="Working days in the quarter", kind="integer",
                      default=90, min=20, max=180, step=5, group="Volume", help="How many working days are in the quarter you are measuring.",),
            ParamSpec(key="average_deal_value", label="Average deal value", kind="number",
                      default=250_000, min=0, step=10_000, unit="$", group="Volume",
                  help="Used when your file has no deal values."),
            ParamSpec(key="deal_desk_days", label="Hands-on time: deal desk", kind="number",
                      default=0.8, min=0.05, max=20, step=0.1, unit="days",
                      group="How long each step takes", help="The hands-on time for the first review, once a reviewer actually picks it up.",),
            ParamSpec(key="cf_approval_days", label="Hands-on time: CF approval", kind="number",
                      default=1.1, min=0.05, max=20, step=0.1, unit="days",
                      group="How long each step takes", help="The hands-on time for the finance check.",),
            ParamSpec(key="legal_days", label="Hands-on time: legal", kind="number",
                      default=0.9, min=0.05, max=20, step=0.1, unit="days",
                      group="How long each step takes", help="The hands-on time for the legal check.",),
            ParamSpec(key="deal_desk_sd", label="Variability: deal desk", kind="number",
                      default=0.6, min=0.05, max=1.5, step=0.05, group="How long each step takes",
                  help="How spread out the times are. 0 = every request takes the same."),
            ParamSpec(key="cf_sd", label="Variability: CF approval", kind="number",
                      default=0.8, min=0.05, max=1.5, step=0.05, group="How long each step takes", help="How much the finance check varies from request to request.",),
            ParamSpec(key="legal_sd", label="Variability: legal", kind="number",
                      default=0.9, min=0.05, max=1.5, step=0.05, group="How long each step takes", help="How much the legal check varies from request to request.",),
            ParamSpec(key="deal_desk_capacity", label="Approvers: deal desk", kind="integer",
                      default=3, min=1, max=50, step=1, group="How many approvers", help="How many people can do the first review at the same time.",),
            ParamSpec(key="cf_capacity", label="Approvers: CF approval", kind="integer",
                      default=2, min=1, max=50, step=1, group="How many approvers", help="How many finance approvers are available at the same time.",),
            ParamSpec(key="legal_capacity", label="Approvers: legal", kind="integer",
                      default=2, min=1, max=50, step=1, group="How many approvers", help="How many legal reviewers are available at the same time.",),
            ParamSpec(key="target_sla_days", label="Promise to the business", kind="number",
                      default=10, min=1, max=60, step=1, unit="days", group="The promise",
                  help="Requests cleared inside this many days still land in the quarter."),
            ParamSpec(key="renege_after_days", label="When a deal gives up", kind="number",
                      default=25, min=1, max=120, step=1, unit="days", group="The promise", help="How long a request waits before the customer gives up on it.",),
            ParamSpec(key="renege_probability", label="Share that give up once that long",
                      kind="number", default=25, min=0, max=100, step=5, unit="%",
                      group="The promise", help="The share of requests that give up once they have waited that long.",),
            ParamSpec(key="revenue_at_risk_pct", label="Revenue lost if it misses the quarter",
                      kind="number", default=15, min=0, max=100, step=1, unit="%",
                      group="Money", help="The share of a deal's value you lose if it slips past the quarter.",),
            ParamSpec(key="auto_approve_below_pct", label="Auto-approve requests below this discount",
                      kind="number", default=10, min=0, max=60, step=1, unit="%", group="The fixes", help="Discounts at or below this are approved automatically and skip the queue entirely.",),
            ParamSpec(key="share_below_threshold_pct", label="Share of requests that are below it",
                      kind="number", default=40, min=0, max=100, step=5, unit="%",
                      group="The fixes",
                  help="How much of the queue the auto-approval rule removes."),
            ParamSpec(key="run_scenarios", label="Test the improvement ideas", kind="bool",
                      default=True, group="The fixes", help="Also test the fixes: an extra approver, auto-approval, and more legal capacity.",),
        ],
    )

    def run(self, req, ctx) -> "H.RunResult":
        P = ctx.params
        rows = ctx.rows
        rng = ctx.rng("approval_bottleneck", "main")

        quarter_days = int(H.p(P, "quarter_days", 90))
        sla = H.as_float(H.p(P, "target_sla_days", 10))
        renege_after = H.as_float(H.p(P, "renege_after_days", 25))
        renege_prob = H.as_float(H.p(P, "renege_probability", 25)) / 100.0
        at_risk_pct = H.as_float(H.p(P, "revenue_at_risk_pct", 15)) / 100.0
        share_auto = H.as_float(H.p(P, "share_below_threshold_pct", 40)) / 100.0
        auto_below = H.as_float(H.p(P, "auto_approve_below_pct", 10))
        do_scenarios = bool(H.p(P, "run_scenarios", True))

        ctx.say("Reading how approval requests arrive and how long each step takes...",
                phase="read", progress=0.05)

        # -- 1. stages, capacities, service times ----------------------------
        stage_names: list[str] = []
        caps: dict[str, int] = {}
        svc: dict[str, tuple[float, float]] = {}
        seen_days: dict[str, float] = {}   # door-to-door time as recorded in the file
        values: list[float] = []
        rate = 0.0
        learned = False
        n_observed = 0
        notes: list[str] = []

        # QA DF-10 / UX-04: a step time or team size the user TYPED at Step 4
        # outranks the value measured from their file.
        explicit = set(getattr(ctx.request, "explicit_params", None) or [])
        typed_svc = any(k in explicit for k in ("deal_desk_days", "cf_approval_days",
                                                "legal_days"))
        typed_cap = any(k in explicit for k in ("deal_desk_capacity", "cf_capacity",
                                                "legal_capacity"))
        if typed_svc or typed_cap:
            ctx.say("You typed the step times or the team sizes yourself, so we are using "
                    "your figures rather than the ones measured from your file.",
                    phase="prepare", level="warn")

        if H.has(rows, "stage") and H.has(rows, "days_in_stage") and not typed_svc:
            by_stage: dict[str, list[float]] = {}
            approvers: dict[str, set] = {}
            order: list[str] = []
            rid = H.cat(rows, "request_id", "")
            stg = H.cat(rows, "stage", "")
            dys = H.num(rows, "days_in_stage", 0.0)
            appr = H.cat(rows, "approver", "")
            for i, s in enumerate(stg):
                if not s:
                    continue
                if s not in by_stage:
                    by_stage[s] = []
                    approvers[s] = set()
                    order.append(s)
                if dys[i] > 0:
                    by_stage[s].append(float(dys[i]))
                if appr[i]:
                    approvers[s].add(appr[i])
            n_req = len({r for r in rid if r})
            n_observed = n_req
            days = [float(v) for v in H.num(rows, "submitted_day", np.nan)
                    if not np.isnan(v)]
            span = (max(days) - min(days) + 1.0) if len(days) >= 2 else float(quarter_days)
            rate = (len(days) and n_req / span) or 0.0
            if order and rate > 0 and all(len(v) >= 5 for v in by_stage.values()):
                learned = True
                stage_names = order
                vals = H.num(rows, "deal_value", 0.0)
                values = [float(v) for v in vals if v > 0]
                for s in stage_names:
                    arr = np.asarray(by_stage[s], dtype=float)
                    mean_d = float(np.mean(arr))
                    sd_d = float(np.std(arr)) if arr.size > 1 else mean_d * 0.5
                    cv = sd_d / max(mean_d, 1e-6)
                    sigma = float(np.clip(np.sqrt(np.log(1.0 + cv * cv)), 0.15, 0.9))
                    # A typed team size wins over the count seen in the file.
                    if typed_cap:
                        cap = _cap_for(s, P)
                        notes.append(f"Using the {cap} approvers you typed for '{s}' "
                                     f"rather than the number in your file.")
                    else:
                        cap = len(approvers[s]) if approvers[s] else _cap_for(s, P)
                    if not approvers[s] and not typed_cap:
                        notes.append(f"No approver column for '{s}', so we used the "
                                     f"{cap} approvers you told us about.")
                    caps[s] = max(int(cap), 1)
                    seen_days[s] = mean_d
                    fit_rng = ctx.rng("approval_bottleneck", "calibrate", H.slug(s))
                    n_fit = int(min(max(n_req, 40), 250))
                    hands_on = _calibrate(mean_d, sigma, caps[s], rate, n_fit, fit_rng)
                    svc[s] = (hands_on, sigma)
                    ctx.say(f"'{s}': {len(arr):,} requests, {caps[s]} approver(s), "
                            f"{mean_d:.1f} days door to door - about {hands_on:.1f} of that "
                            f"is real work.", phase="prepare", progress=0.15,
                            detail=f"observed mean {mean_d:.2f} sd {sd_d:.2f}; "
                                   f"implied hands-on {hands_on:.2f}")
                ctx.say(f"Learned the process from {len(rid):,} request rows covering "
                        f"{int(span)} days - about {rate:.2f} requests a day.",
                        phase="prepare", progress=0.2)

        if not learned:
            stage_names = [STAGE_NAMES[k] for k in STAGE_KEYS]
            rate = H.as_float(H.p(P, "arrival_rate_per_week", 12)) / 7.0
            for s in stage_names:
                caps[s] = _cap_for(s, P)
                svc[s] = _svc_for(s, P)
            ctx.say(f"No usable approval history, so we built the process from your "
                    f"assumptions: {' -> '.join(stage_names)}.", phase="prepare",
                    progress=0.12, level="warn")

        n_entities = int(H.p(P, "n_requests", 200))
        if learned:
            n_entities = max(n_entities, int(rate * quarter_days * 1.25) + 10)
        if not values:
            values = [H.as_float(H.p(P, "average_deal_value", 250_000))]

        ctx.say(f"Rebuilding the queue: {rate * 7:.1f} requests a week, "
                f"{', '.join(f'{caps[s]} in {s}' for s in stage_names)}.",
                phase="prepare", progress=0.25)

        # -- 2. run the current process --------------------------------------
        scenarios: list[tuple[str, str, dict]] = [(
            "current", "Today's process",
            {s: (svc[s], caps[s]) for s in stage_names})]

        def build_stages(spec: dict) -> list[dict]:
            return [{"name": s, "dist": _lognormal_spec(spec[s][0][0], spec[s][0][1]),
                     "capacity": spec[s][1]} for s in stage_names]

        def run_one(spec: dict, tag: str):
            """Simulate the queue for several quarters and pool them.

            Common random numbers: every option gets the *same* dice, so the
            difference between two options is the fix, not luck. Without this a
            queue running at ~90% capacity is noisy enough that a real
            improvement can look like a regression.
            """
            reps = []
            for k in range(REPLICATES):
                res = de.run(
                    n_entities=n_entities,
                    stages=build_stages(spec),
                    arrival_spec={"dist": "exponential", "rate": max(rate, 1e-6)},
                    rng=ctx.rng("approval_bottleneck", "sim", str(k)),
                    horizon=float(quarter_days),
                    renege_after=renege_after,
                    renege_prob=renege_prob,
                )
                n = len(res["records"])
                val = dist.sample({"dist": "empirical", "values": values}, n,
                                  ctx.rng("approval_bottleneck", "values", str(k))) \
                    if len(values) > 1 else np.full(n, values[0])
                cycle = np.array([rec["cycle_time"] for rec in res["records"]],
                                 dtype=float)
                late = (cycle > sla) | np.array(
                    [bool(rec["abandoned"]) for rec in res["records"]], dtype=bool)
                res["_values"] = val
                res["_cycle"] = cycle
                res["_late"] = late
                res["_risk"] = float(np.sum(val[late]) * at_risk_pct)
                reps.append(res)
            return _pool(reps)

        ctx.say(f"Running {n_entities * REPLICATES:,} requests through the queue "
                f"({REPLICATES} simulated quarters)...", phase="simulate", progress=0.3)
        current = run_one(scenarios[0][2], "current")
        base_stats = {s["stage"]: s for s in current["stage_stats"]}

        # bottleneck = busiest step weighted by how long it makes people wait
        def bottleneck_of(res) -> tuple[str, float, float]:
            best_name, best_util, best_wait, best_score = "", 0.0, 0.0, -1.0
            for s in res["stage_stats"]:
                score = float(s["utilisation"]) * float(s["avg_wait"])
                if score > best_score:
                    best_score = score
                    best_name = s["stage"]
                    best_util = float(s["utilisation"])
                    best_wait = float(s["avg_wait"])
            return best_name, best_util, best_wait

        bn_name, bn_util, bn_wait = bottleneck_of(current)
        ranked = sorted(current["stage_stats"],
                        key=lambda s: -(float(s["utilisation"]) * float(s["avg_wait"])))
        second = ranked[1]["stage"] if len(ranked) > 1 else bn_name

        ctx.say(f"Found it: {bn_name} is the jam - {H.pct(bn_util)} busy and "
                f"{bn_wait:.1f} days of waiting.", phase="summarize", progress=0.55)

        if do_scenarios:
            spec2 = {s: (svc[s], caps[s]) for s in stage_names}
            spec2[bn_name] = (svc[bn_name], caps[bn_name] + 1)
            scenarios.append(("extra_approver",
                              f"One more approver in {bn_name}", spec2))

            spec3 = {s: (svc[s], caps[s]) for s in stage_names}
            spec3[bn_name] = ((svc[bn_name][0] * (1.0 - share_auto), svc[bn_name][1]),
                              caps[bn_name])
            scenarios.append(("auto_approve",
                              f"Auto-approve requests under {auto_below:.0f}%", spec3))

            spec4 = {s: (svc[s], caps[s]) for s in stage_names}
            if second != bn_name:
                spec4[second] = (svc[second], caps[second] + 1)
                scenarios.append(("parallel_second",
                                  f"One more approver in {second}", spec4))

        results = {"current": current}
        for key, label, spec in scenarios[1:]:
            results[key] = run_one(spec, key)
            ctx.say(f"Tested '{label}'.", phase="simulate", progress=0.6,
                    detail=f"median {results[key]['cycle_time_summary'].get('p50', 0):.1f} days")

        def summarise(key: str, label: str, res) -> dict:
            cs = res["cycle_time_summary"]
            late = res["_late"]
            bn_n, bn_u, bn_w = bottleneck_of(res)
            return {
                "key": key, "label": label,
                "median": float(cs.get("p50", 0.0)),
                "p90": float(res["_p90"]),
                "on_time": float(de.prob_complete_within(res, sla)),
                "abandon": float(res["abandon_rate"]),
                "bn": bn_n, "bn_util": bn_u, "bn_wait": bn_w,
                "risk": float(res["_revenue_at_risk"]),
                "through": int(res["throughput"]),
                "res": res,
            }

        ctx.say("Comparing the fixes...", phase="summarize", progress=0.8)
        summaries = [summarise(k, l, results[k]) for k, l, sp in scenarios]

        base = summaries[0]
        best_fix = None
        for s in summaries[1:]:
            if best_fix is None or s["risk"] < best_fix["risk"]:
                best_fix = s
        if best_fix is None:
            best_fix = base

        # -- 3. assemble ------------------------------------------------------
        n_sims = int(n_entities * REPLICATES * len(scenarios))
        res = H.new_result(ctx, self.meta.id, self.meta.title,
                           "Discrete-event queueing simulation (arrivals, finite approvers, "
                           "walk-aways)", n_sims)

        saved = base["risk"] - best_fix["risk"]
        res.headline = (
            f"{bn_name} is the jam: {H.pct(bn_util)} busy, adding {bn_wait:.1f} days of "
            f"waiting, and it is why {H.pct(1 - base['on_time'])} of requests miss the "
            f"{sla:.0f}-day promise. {H.money(base['risk'], compact=True)} of revenue is "
            f"stuck behind it."
            + (f" {best_fix['label']} would release {H.money(saved, compact=True)} of that."
               if best_fix["key"] != "current" and saved > 0 else "")
        )
        res.method_plain = (
            "We rebuilt your approval process as a queue. Requests arrive the way yours do, "
            "each waits for a free approver, gets worked on for a realistic amount of time, "
            "then moves to the next step - and a request that waits too long sometimes gives "
            "up and walks. Running thousands of requests through that queue shows where the "
            "pile-up forms, how long the whole journey takes, and how many requests would "
            "still be sitting there when the quarter closes. We then re-ran the same queue "
            "with each fix in place to price them against each other. Every option is run on "
            "the same dice - same arrivals, same requests - so the difference between two "
            "options is the fix itself rather than luck."
        )

        median_now = base["median"]
        queue_value = float(np.sum(current["_values"])) / REPLICATES
        res.kpis = [
            H.kpi("How long a request takes today", f"{median_now:.1f} days",
                  "good" if median_now <= sla else "bad",
                  f"Half of all requests clear in {median_now:.1f} days or less.",
                  comparison=f"promise is {sla:.0f} days"),
            H.kpi(f"Cleared inside the {sla:.0f}-day promise", H.pct(base["on_time"]),
                  "good" if base["on_time"] >= 0.8 else "bad",
                  f"The share of requests that still land in the quarter they were meant to.",
                  comparison=f"{H.pct(1 - base['on_time'])} miss it"),
            H.kpi("Where it jams", bn_name, "bad",
                  f"This step is {H.pct(bn_util)} busy and adds {bn_wait:.1f} days of "
                  f"waiting - the single biggest cause of the delay."),
            H.kpi("Busiest step is busy", H.pct(bn_util), "bad",
                  f"Above about 85% a queue stops clearing: any hiccup becomes a backlog.",
                  comparison=f"{base['bn_wait']:.1f} days of waiting"),
            H.kpi("Revenue stuck behind the queue", H.money(base["risk"], compact=True), "bad",
                  f"Deal value of the requests that miss the quarter, times the "
                  f"{H.pct(at_risk_pct)} you told us is lost when a deal slips.",
                  comparison=f"{H.money(queue_value, compact=True)} of deals pass "
                             f"through in a quarter"),
            H.kpi("Walk-away rate", H.pct(base["abandon"]), "warn" if base["abandon"] > 0.02
                  else "good",
                  f"Requests that waited more than {renege_after:.0f} days and gave up."),
        ]
        if best_fix["key"] != "current":
            res.kpis.append(H.kpi(
                f"Best fix: {best_fix['label']}",
                f"{best_fix['median'] - base['median']:+.1f} days", "good",
                f"Median time goes from {base['median']:.1f} to {best_fix['median']:.1f} days "
                f"and requests inside the promise go from {H.pct(base['on_time'])} to "
                f"{H.pct(best_fix['on_time'])}.",
                comparison=f"{H.money(saved, compact=True)} of revenue released"))

        # charts -------------------------------------------------------------
        cycles = current["_cycle"]
        step = max(1, len(cycles) // 2000)
        wait_x = [s["stage"] for s in current["stage_stats"]]
        bx, by = de.backlog_series(current["_reps"][0], float(quarter_days), bins=30)

        res.charts = [
            H.chart(
                "cycle_hist", "hist", "How long requests actually take",
                subtitle=f"Today's process; the promise is {sla:.0f} days",
                x_label="Days from request to decision", y_label="How many requests",
                x=[], series=[{"name": "Time taken",
                               "values": [float(v) for v in cycles[::step]]}],
                annotations=[{"x": sla, "y": 0, "text": f"{sla:.0f}-day promise"}],
                plain_meaning="Everything to the right of the promise line is a deal that "
                              "slips out of the quarter. The long tail on the right is the "
                              "queue, not the work.",
            ),
            H.chart(
                "wait_by_stage", "bar", "Where the waiting happens",
                subtitle="Average days a request sits before each step even starts",
                x_label="Approval step", y_label="Days of waiting",
                x=wait_x,
                series=[
                    {"name": "Waiting",
                     "values": [H.r2(float(s["avg_wait"])) for s in current["stage_stats"]],
                     "color": "#ef4444"},
                    {"name": "Hands-on work",
                     "values": [H.r2(float(s["avg_service"])) for s in current["stage_stats"]],
                     "color": "#64748b"},
                ],
                plain_meaning="The tall red bar is the bottleneck. Grey is the actual work - "
                              "if red dwarfs grey, the problem is capacity, not effort.",
            ),
            H.chart(
                "backlog", "line", "The pile building up",
                subtitle="Requests open at any point in the quarter",
                x_label="Day of the quarter", y_label="Requests still in the queue",
                x=[H.r2(v) for v in bx],
                series=[{"name": "Requests waiting", "values": [H.r2(v) for v in by],
                         "color": "#f59e0b"}],
                plain_meaning="A line that keeps climbing means the queue never catches up - "
                              "that is the definition of a bottleneck.",
            ),
            H.chart(
                "scenarios", "grouped_bar", "What each fix would do",
                subtitle="Same queue, each fix applied",
                x_label="", y_label="",
                x=["Days to clear (median)", "Inside the promise", "Revenue at risk ($M)"],
                series=[{"name": s["label"],
                         "values": [H.r2(s["median"]), H.r2(s["on_time"] * 100),
                                    H.r2(s["risk"] / 1e6)],
                         "color": ("#64748b" if s["key"] == "current"
                                   else "#22c55e" if s["key"] == best_fix["key"]
                                   else "#3b82f6")}
                        for s in summaries],
                plain_meaning="Look for the fix that moves 'inside the promise' up and "
                              "'revenue at risk' down for the least money.",
            ),
        ]

        # tables --------------------------------------------------------------
        total_wait = sum(float(s["avg_wait"]) for s in current["stage_stats"]) or 1.0
        stage_cols = ["Step", "Approvers", "Requests", "Avg wait (days)",
                      "Slowest 10% wait", "Longest wait", "Hands-on (days)",
                      "How busy", "Share of all waiting", ""]
        if learned:
            stage_cols.insert(3, "Days in your file")
        stage_rows = []
        for s in current["stage_stats"]:
            row = [s["stage"], f"{s['capacity']}", f"{s['n_through']:,}"]
            if learned:
                row.append(f"{seen_days.get(s['stage'], float('nan')):.1f}")
            row += [f"{s['avg_wait']:.1f}", f"{s['p90_wait']:.1f}", f"{s['max_wait']:.1f}",
                    f"{s['avg_service']:.1f}", H.pct(s["utilisation"]),
                    H.pct(float(s["avg_wait"]) / total_wait),
                    "BOTTLENECK" if s["stage"] == bn_name else ""]
            stage_rows.append(row)
        hl = [i for i, r in enumerate(stage_rows) if r[-1] == "BOTTLENECK"]

        scen_rows = []
        for s in summaries:
            scen_rows.append([
                s["label"], f"{s['median']:.1f}", f"{s['p90']:.1f}", H.pct(s["on_time"]),
                H.pct(s["abandon"]), H.pct(s["bn_util"]), H.money(s["risk"], compact=True),
                "-" if s["key"] == "current" else
                f"{H.money(base['risk'] - s['risk'], compact=True)} released",
            ])

        worst = []
        recs = current["records"]
        order_idx = sorted(range(len(recs)), key=lambda i: -recs[i]["cycle_time"])[:10]
        for i in order_idx:
            r = recs[i]
            worst.append([
                f"Request {r['entity'] + 1}", f"{r['arrival']:.0f}",
                f"{r['cycle_time']:.1f}", f"{r['total_wait']:.1f}",
                # CF carries ~94% of the waiting, so a wait this long is almost
                # certainly that queue rather than the work itself.
                bn_name if r["total_wait"] >= 0.75 * max(bn_wait, 1e-9) else "-",
                H.money(float(current["_values"][i]), compact=True),
                H.money(float(current["_values"][i]) * at_risk_pct, compact=True),
                "Walked away" if r["abandoned"] else
                ("Late" if r["cycle_time"] > sla else "On time"),
            ])

        res.tables = [
            H.table(
                "Each step, measured",
                ("Waiting is the queue; hands-on is the actual work. 'Days in your file' is "
                 "the door-to-door time you recorded - waiting and work together - which is "
                 "why it is bigger than hands-on." if learned else
                 "Waiting is the queue; hands-on is the actual work. Share of waiting is "
                 "where your delay lives."),
                stage_cols, stage_rows, highlight=hl,
                note="The bottleneck step is the one with the most waiting per unit of "
                     "busyness - it is the reason the quarter slips.",
            ),
            H.table(
                "What each fix is worth",
                f"Revenue at risk = deal value of the requests that miss the "
                f"{sla:.0f}-day promise, times {H.pct(at_risk_pct)}.",
                ["Option", "Days to clear (median)", "Slowest 10% (days)",
                 "Inside the promise", "Walk-away rate", "Busiest step busy",
                 "Revenue at risk", "Gain"],
                scen_rows,
                highlight=[i for i, s in enumerate(summaries) if s["key"] == best_fix["key"]],
                note="The highlighted row is the cheapest fix that releases the most revenue.",
            ),
            H.table(
                "The ten worst-hit requests",
                "The requests that lost the most time in today's process.",
                ["Request", "Arrived (day)", "Days to clear", "Of which waiting",
                 "Stuck behind", "Deal value", "Value at risk", "Outcome"],
                worst,
                note="Multiply this pattern across the quarter and you have the size of "
                     "the problem.",
            ),
        ]

        # story ---------------------------------------------------------------
        res.story = [
            H.story(
                "The process we rebuilt",
                f"{rate * 7:.1f} requests a week arrive and pass through "
                f"{' -> '.join(stage_names)}, with {', '.join(str(caps[s]) for s in stage_names)} "
                f"approvers respectively. " +
                (f"Measured from {n_observed:,} requests in your own history, spanning "
                 f"{int(span)} days."
                 if learned else "Built from your assumptions, because no history was mapped."),
                f"{H.money(queue_value, compact=True)} of deals through the queue "
                f"per quarter",
            ),
            H.story(
                "Where it jams",
                f"{bn_name} is {H.pct(bn_util)} busy and makes requests wait "
                f"{bn_wait:.1f} days on average - {H.pct(bn_wait / max(median_now, 1e-9))} "
                f"of the whole journey is spent sitting in front of that one step.",
                f"{H.pct(bn_util)} busy, {bn_wait:.1f} days waiting",
            ),
            H.story(
                "What that costs",
                f"Only {H.pct(base['on_time'])} of requests clear inside the {sla:.0f}-day "
                f"promise, so {H.pct(1 - base['on_time'])} slip out of the quarter. At "
                f"{H.pct(at_risk_pct)} of deal value lost per slip, that is "
                f"{H.money(base['risk'], compact=True)} of revenue at risk"
                + (f" - plus {H.pct(base['abandon'])} of requests that give up entirely."
                   if base["abandon"] > 0.005 else
                   f". Nothing gives up outright, but the slowest tenth wait "
                   f"{base['p90']:.0f} days."),
                f"{H.money(base['risk'], compact=True)} at risk",
            ),
            H.story(
                "What the fixes are worth",
                (f"{best_fix['label']} is the best of them: median time falls from "
                 f"{base['median']:.1f} to {best_fix['median']:.1f} days, requests inside the "
                 f"promise rise from {H.pct(base['on_time'])} to {H.pct(best_fix['on_time'])}, "
                 f"and {H.money(saved, compact=True)} of revenue comes back into the quarter."
                 if best_fix["key"] != "current" else
                 "No improvement scenario was run."),
                f"{H.money(saved, compact=True)} released" if best_fix["key"] != "current"
                else "-",
            ),
            H.story(
                "The bottom line",
                f"The constraint is not effort, it is capacity at {bn_name}: the work itself "
                f"takes {base_stats[bn_name]['avg_service']:.1f} days, the waiting takes "
                f"{bn_wait:.1f}. Add capacity or take work out of that queue - asking people "
                f"to work faster will not move the number.",
                f"{base_stats[bn_name]['avg_service']:.1f} days work, {bn_wait:.1f} days waiting",
            ),
        ]

        risk = float(np.clip(100 * (1 - base["on_time"]) + 20 * base["abandon"], 0, 100))
        actions = []
        if best_fix["key"] != "current":
            actions.append(f"{best_fix['label']} - worth {H.money(saved, compact=True)} of "
                           f"revenue released and {base['median'] - best_fix['median']:.1f} "
                           f"days off the median.")
        if "auto_approve" in results:
            actions.append(
                f"Auto-approve anything under {auto_below:.0f}% discount without a "
                f"{bn_name} stop - that alone removes "
                f"{H.pct(share_auto)} of the queue.")
        actions.append(
            f"Publish a daily view of the {bn_name} queue with a {sla:.0f}-day clock on each "
            f"request, so slippage is visible before the quarter closes.")
        if base["abandon"] > 0.005:
            actions.append(
                f"Set a hard escalation at {renege_after:.0f} days: {H.pct(base['abandon'])} "
                f"of requests are currently walking away unmanaged.")
        else:
            actions.append(
                f"Watch the tail: the slowest requests take {base['p90']:.0f} days. Nothing "
                f"is walking away yet, but at that length the customer has already "
                f"re-planned without you.")

        res.recommendation = Recommendation(
            headline=(f"Fix {bn_name} first. {best_fix['label']} releases "
                      f"{H.money(saved, compact=True)} of revenue and takes "
                      f"{base['median'] - best_fix['median']:.1f} days out of the process."
                      if best_fix["key"] != "current" else
                      f"{bn_name} is the bottleneck - add capacity or remove work from it."),
            rationale=[
                f"{bn_name} runs at {H.pct(bn_util)} busy and adds {bn_wait:.1f} days of "
                f"waiting - more than any other step.",
                f"Only {H.pct(base['on_time'])} of requests clear inside the {sla:.0f}-day "
                f"promise, putting {H.money(base['risk'], compact=True)} of revenue at risk.",
                f"The work in that step takes {base_stats[bn_name]['avg_service']:.1f} days; "
                f"the waiting takes {bn_wait:.1f}. It is a capacity problem, not an effort "
                f"problem.",
            ],
            actions=actions,
            risk_score=round(risk, 1),
            risk_label=H.risk_label(risk),
            confidence="High" if learned else "Medium",
            tradeoff=("Adding an approver costs salary; auto-approving below a threshold "
                      "costs a little margin on small deals. Both are far cheaper than the "
                      f"{H.money(base['risk'], compact=True)} currently at risk."),
        )

        res.assumptions = [
            (f"Arrivals and timing measured from your own approval history: about "
             f"{rate * 7:.1f} requests a week, {len(stage_names)} steps."
             if learned else
             f"No usable history: {H.as_float(H.p(P, 'arrival_rate_per_week', 12)):.0f} "
             f"requests a week, {quarter_days} working days in the quarter."),
            "Hands-on time per step: " + "; ".join(
                f"{s} about {svc[s][0]:.1f} days" for s in stage_names),
            ("Where your file records door-to-door time (queue included), we backed out the "
             "real hands-on time by finding the working time that reproduces the waiting you "
             "actually saw, given how many approvers you have."
             if learned else
             "Times are skewed: most requests are quicker than average, a few take much "
             "longer."),
            f"Approvers per step: " + ", ".join(f"{s} {caps[s]}" for s in stage_names) +
            (" (counted from the distinct names in your file)" if learned and
             H.has(rows, "approver") else " (from your settings)"),
            f"A request that waits more than {renege_after:.0f} days gives up with "
            f"{H.pct(renege_prob)} probability.",
            f"Revenue at risk = {H.pct(at_risk_pct)} of the deal value of every request that "
            f"misses the {sla:.0f}-day promise; average deal value "
            f"{H.money(float(np.mean(values)))}.",
            "Every request is modelled as passing through every step; steps that only some "
            "requests reach therefore look slightly busier than they are.",
            f"{REPLICATES} simulated quarters of {n_entities:,} requests each per option "
            f"({n_entities * REPLICATES:,} requests in total); about "
            f"{int(current['throughput']):,} arrive and clear inside each "
            f"{quarter_days}-day quarter. Seed {ctx.request.seed}, so re-running gives "
            f"identical numbers. Every option is tested against the same simulated "
            f"requests, so the comparison is fair. Revenue at risk is the average per quarter.",
        ]

        if not learned:
            res.warnings.append(
                "No approval history was mapped, so the queue is built from your assumptions. "
                "Upload the request log (request id, submitted day, stage, days in stage, "
                "approver) to measure the real thing.")
        if learned and not H.has(rows, "approver"):
            res.warnings.append(
                "No approver column was mapped, so we used the headcount you set rather than "
                "the number of people who actually sign.")
        if bn_util > 0.95:
            res.warnings.append(
                f"{bn_name} is running at {H.pct(bn_util)} of capacity. Above roughly 95% a "
                f"queue stops clearing altogether - the backlog will keep growing no matter "
                f"how hard the team works.")
        if base["on_time"] < 0.5:
            res.warnings.append(
                f"Fewer than half of requests clear inside the {sla:.0f}-day promise. Until "
                f"the bottleneck is fixed, the forecast for this quarter is optimistic.")
        for n in notes:
            res.warnings.append(n)

        ctx.narrator.done(f"Done. {bn_name} is the bottleneck - "
                          f"{H.money(base['risk'], compact=True)} of revenue is stuck behind it.")
        res.log = ctx.narrator.events()
        return res
