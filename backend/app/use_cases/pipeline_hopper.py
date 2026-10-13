"""USE CASE - Deal Aging & Multi-Quarter Sales Cycles: The Pipeline Hopper.

A pipeline is not a list of deals with dates on them. It is a board: deals hop
from stage to stage, they sit still for a quarter, and some of them quietly die
in legal. Sales forecasts the hop they *want* ("closes in Q2"); this model
forecasts the hop that history says actually happens.

Two things make this honest rather than decorative:

* the board is only allowed to end where your history says it ends. If nobody
  in your file was ever recorded as closed-won, we say "we cannot estimate a
  close rate from this" instead of quietly treating the first stage as the
  finish line, and we fall back to your own assumptions - out loud.
* the odds depend on how long a deal has been sitting still. A deal parked in
  Legal for a year is not priced the same as one that arrived last week.

We learn how deals moved from your own history (or from the assumptions below),
turn that into quarter-by-quarter odds, and then weight the money by those odds
- so the number you plan with is the one that has a real chance of arriving.
"""
from __future__ import annotations

import datetime as _dt
import math
import re

import numpy as np

from ..core.contracts import ParamSpec, Recommendation, RoleSpec, UseCaseMeta
from ..core.registry import UseCase, register
from ..engine import markov as mk
from . import _helpers as H

#: The board we use when you have no history. Last two entries are the endings.
FULL_PIPELINE = ["Initial RFP", "Technical Review", "CF Approval", "Legal",
                 "Closed Won", "Dead"]
SHORT_PIPELINE = ["Qualified", "Proposal", "Closed Won", "Dead"]

_WON_WORDS = ("won", "win", "close", "closed", "signed", "booked")
_DEAD_WORDS = ("dead", "lost", "loss", "churn", "dropped", "cancelled", "canceled",
               "stalled", "no decision", "nogo")

#: a quarter, for people whose age column counts days
_DAYS_PER_QUARTER = 91.0

_AGE_BUCKETS = ("young", "normal", "stale")

#: table names the consistency check knows how to interrogate
_STAGE_TABLE = "Quarter by quarter: where the deal is"
_MONEY_TABLE = "The money, weighted by the odds"

_PCT_CELL = re.compile(r"^[+-]?\d+(?:\.\d+)?%$")
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                 "%Y-%m-%d %H:%M", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m", "%b %Y", "%B %Y")


# ---------------------------------------------------------------------------
# Reading the board
# ---------------------------------------------------------------------------
def _is_won(name: str) -> bool:
    s = str(name).strip().lower()
    return any(w in s for w in _WON_WORDS) and not any(w in s for w in _DEAD_WORDS)


def _is_dead(name: str) -> bool:
    s = str(name).strip().lower()
    return any(w in s for w in _DEAD_WORDS)


def _terminal_indices(states) -> tuple[int | None, int | None]:
    """Where the board ends: (index of the closed-won stage, index of the dead one).

    Either may be ``None``. We never invent one. If the history has no
    closed-won stage, the honest answer is "your data cannot estimate a close
    rate", not "let us pretend the first stage is the finish line" - that
    mistake is what produced a negative close probability.
    """
    won = [i for i, s in enumerate(states) if _is_won(s)]
    dead = [i for i, s in enumerate(states) if _is_dead(s)]
    w = won[0] if won else None
    d = next((i for i in dead if i != w), None)
    return w, d


def _parse_date(s: str) -> tuple[int, int] | None:
    for fmt in _DATE_FORMATS:
        try:
            d = _dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
        return d.year, (d.month - 1) // 3 + 1
    return None


def _period_index(v) -> float | None:
    """'2024-Q4' -> 8100 ; 'Q3' -> 3 ; 7 -> 7 ; '2024-11-02' -> 8100."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    low = s.lower()
    if "-q" in low or low.startswith("q"):
        try:
            head, tail = low.split("-q") if "-q" in low else ("", low[1:])
            year = float(head) if head.strip() else 0.0
            return year * 4.0 + float(tail)
        except (ValueError, IndexError):
            return None
    if "q" in low and any(ch.isdigit() for ch in low):
        digits = "".join(ch for ch in low if ch.isdigit())
        try:
            return float(digits)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        pass
    d = _parse_date(s)
    if d is not None:
        return float(d[0] * 4 + d[1])
    return None


def _age_quarters(v, units: str, t: float | None = None) -> float | None:
    """Read one age value as a number of quarters.

    Plain numbers are quarters - or days when ``units`` says days, in which
    case we convert. '2024-Q4' / '2024-11-02' style values are turned into an
    age by subtracting them from the quarter the row describes (``t``).
    """
    if v is None or v == "":
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        f = None
    if f is not None:
        return f / _DAYS_PER_QUARTER if units == "days" else f
    idx = _period_index(s)
    if idx is None or t is None:
        return None
    return max(0.0, float(t) - float(idx))


def _order_states(states, order_values: dict[str, float]) -> list[str]:
    known = [s for s in states if s in order_values]
    unknown = [s for s in states if s not in order_values]
    known.sort(key=lambda s: order_values[s])
    return known + unknown


# ---------------------------------------------------------------------------
# How long a deal has been sitting changes its odds
# ---------------------------------------------------------------------------
def _age_bucket(age: float, stale: float) -> str:
    if age < stale * 0.5:
        return "young"
    if age < stale:
        return "normal"
    return "stale"


def _apply_aging(M: np.ndarray, states: list[str], boost: float,
                 dead_i: int | None) -> np.ndarray:
    """Make a deal that has been sitting too long more likely to slip or die.

    The share of each stage's weight that does *not* move the deal forward is
    multiplied by ``(1 + boost)``: some of the deals that would have advanced now
    sit still or die instead. ``boost`` is capped by the caller, so an ancient
    deal is never turned into a certain loss.
    """
    A = np.array(M, dtype=float)
    if boost <= 1e-12:
        return A
    for i, s in enumerate(states):
        if _is_won(s) or _is_dead(s):
            continue
        targets = [j for j in range(len(states))
                   if j != i and not _is_dead(states[j])]
        adv = sum(float(A[i, j]) for j in targets)
        if adv <= 1e-12:
            continue
        stay = float(A[i, i])
        die = float(A[i, dead_i]) if dead_i is not None else 0.0
        non = stay + die
        want = min(non * (1.0 + boost), 1.0)
        if non <= 1e-12:
            want = min(boost / (1.0 + boost) * 0.5, 0.5)
        grow = want - non
        if grow <= 1e-12:
            continue
        if non > 1e-12:
            A[i, i] += grow * stay / non
            if dead_i is not None:
                A[i, dead_i] += grow * die / non
        else:
            A[i, i] += grow * 0.5
            if dead_i is not None:
                A[i, dead_i] += grow * 0.5
            else:
                A[i, i] += grow * 0.5
        scale = max(1.0 - grow / adv, 0.0)
        for j in targets:
            A[i, j] *= scale
        row = A[i].sum()
        if row > 0:
            A[i] /= row
    return A


def _aged_matrix(base_M: np.ndarray, bucket_M: np.ndarray | None, states: list[str],
                 age: float, stale: float, mult: float, cap: float,
                 dead_i: int | None) -> np.ndarray:
    """The board a deal of this age should be judged against.

    When we have learned how deals of this age actually behave we use that.
    On top of it (or on top of the whole board when we have not) a deal past
    the stale mark gets progressively more likely to slip or die, up to the cap.

    QA DF-08 / TC-05: the extra risk used to be `min(mult * excess, cap)`, a
    straight line cut off by the cap. With the default stale mark of 4 quarters
    and a cap of 0.9 that line hit the ceiling at 10 quarters, so a deal 11
    quarters old and one 110 quarters old were given *identical* odds - the
    headline and every KPI came out byte for byte the same. The curve now
    approaches the cap instead of being clipped by it, so a longer wait is
    always a worse set of odds and the answer responds across the whole range
    without ever exceeding the cap the user set.
    """
    M = np.array(bucket_M if bucket_M is not None else base_M, dtype=float)
    if bucket_M is not None:
        excess = max(0.0, float(age) - stale) / max(stale, 1e-9)
    else:
        excess = float(age) / max(stale, 1e-9)
    if excess <= 0.0 or cap <= 0.0 or mult <= 0.0:
        return M
    # approaches `cap` from below, and for small excess is very close to the
    # `mult * excess` it replaces
    boost = cap * (1.0 - math.exp(-(mult * excess) / cap))
    if boost <= 1e-9:
        return M
    return _apply_aging(M, states, boost, dead_i)


def _age_matrices(records, ages, states, stale, smoothing, base_M,
                  min_moves: int = 25) -> tuple[dict, dict]:
    """Learn a separate hop pattern for young, settled and stale deals.

    A row with evidence in this band is built exactly the way the whole board
    builds it (observed moves plus the same small benefit of the doubt for
    paths nobody has taken yet), so a band never becomes more confident than
    the evidence warrants. A stage with *no* sightings in this band simply
    keeps the odds we learned from the full history. A band with too little
    evidence behind it is left out entirely, and the caller falls back to the
    whole board plus an ageing penalty.
    """
    k = len(states)
    idx = {s: i for i, s in enumerate(states)}
    by_entity: dict = {}
    for (eid, t, s), age in zip(records, ages):
        by_entity.setdefault(eid, []).append((t, s, float(age)))
    raw: dict[str, dict] = {b: {} for b in _AGE_BUCKETS}
    for seq in by_entity.values():
        seq = sorted(seq, key=lambda x: x[0])
        for (_, a, age_a), (_, b, _) in zip(seq, seq[1:]):
            bucket = _age_bucket(age_a, stale)
            raw[bucket][(a, b)] = raw[bucket].get((a, b), 0.0) + 1.0

    prior = float(smoothing)
    base = np.asarray(base_M, dtype=float)
    out: dict = {}
    moves: dict = {}
    for bucket, c in raw.items():
        moves[bucket] = int(sum(c.values()))
        if moves[bucket] < min_moves:
            out[bucket] = None
            continue
        counts = np.zeros((k, k))
        for (a, b), w in c.items():
            if a in idx and b in idx:
                counts[idx[a], idx[b]] += w
        B = np.array(base, dtype=float).copy()
        for i in range(k):
            obs = float(counts[i].sum())
            if obs <= 0.0:
                continue                      # nothing seen at this age: keep the board
            B[i] = (counts[i] + prior) / (obs + prior * k)
        out[bucket] = _force_absorbing(B, states)
    return out, moves


META = UseCaseMeta(
        id="pipeline_hopper",
        title="Deal Aging & Multi-Quarter Sales Cycles: The Pipeline Hopper",
        subtitle="Sales says it closes in Q2. What do the odds actually say?",
        category="Forecasting",
        icon="🕒",
        models=["Markov Chains"],
        business_question="What is the real chance this deal closes in the quarter sales promised - and how much cash should we commit against it?",
        plain_summary=(
            "Deals hop between stages; they do not march to the date on the forecast. "
            "We measure how your own deals have actually hopped - including the ones that "
            "sat still or died - and turn that into the chance of the deal being anywhere "
            "on the board in each coming quarter. We also allow for how long the deal has "
            "been sitting: the longer it is stuck, the more likely it is to slip or die. "
            "Then we weight the money by those odds, so you commit cash to what is likely "
            "to arrive, not to what sales hopes for."
        ),
        story_steps=[
            "Read the history of how deals have moved between stages, and how long they sat",
            "Measure the odds of every hop: forward, slip a quarter, or die",
            "Roll those odds forward quarter by quarter, allowing for the deal's age",
            "Weight the deal value by the odds - the number a CFO can plan with",
            "Compare it with the forecast sales committed to",
        ],
        data_granularity="transition",
        sample_dataset="pipeline_stages.csv",
        input_roles=[
            RoleSpec(key="deal_id", label="Deal / opportunity ID", dtype="identifier",
                     required=False, aliases=["id", "opportunity", "opp_id", "deal"]),
            RoleSpec(key="stage", label="Pipeline stage", dtype="categorical", required=False,
                     aliases=["phase", "step", "status", "stage_name"]),
            RoleSpec(key="stage_order", label="Stage order (1,2,3...)", dtype="numeric",
                     required=False, aliases=["order", "sequence", "stage_number", "rank"]),
            RoleSpec(key="quarter", label="Quarter of this sighting", dtype="categorical",
                     required=False, aliases=["qtr", "period", "date", "month", "snapshot"],
                  help="Accepts '2024-Q4', a plain period number, or a date."),
            RoleSpec(key="amount", label="Deal value", dtype="numeric", required=False,
                     aliases=["value", "arr", "tcv", "acv", "deal_value", "revenue"]),
            RoleSpec(key="entered_quarter", label="Quarter the deal entered", dtype="categorical",
                     required=False, aliases=["created_quarter", "start_quarter", "entry_quarter"],
                  help="Lets us work out how long each deal has been running."),
            RoleSpec(key="deal_age", label="How long the deal has been sitting (quarters)",
                     dtype="numeric", required=False,
                     aliases=["age", "age_quarters", "quarters_in_stage", "time_in_stage",
                              "stage_age", "tenure"],
                  help="Quarters the deal has been sitting in its stage. Days, dates or "
                          "'2024-Q4' also work - tell us the units below if they are days."),
            RoleSpec(key="days_in_stage", label="How long it has been sitting (days)",
                     dtype="numeric", required=False,
                     aliases=["age_days", "days_open", "days_since_entry", "days_since_created",
                              "stalled_days", "days_in_stage"],
                  help="The same idea counted in days - we convert to quarters for you."),
        ],
        parameters=[
            ParamSpec(key="target_deal_value", label="Value of the deal being forecast",
                      kind="number", default=3_000_000, min=0, step=100_000, unit="$",
                      group="The deal", help="The value of the deal you are forecasting.",),
            ParamSpec(key="starting_stage", label="Stage the deal is in now",
                      kind="choice", default="Initial RFP", group="The deal",
                      choices=[{"value": s, "label": s} for s in FULL_PIPELINE], help="The stage the deal is sitting in today. Any stage in your file can be used.",),
            ParamSpec(key="sales_forecast_quarter", label="Quarter sales promised it would close",
                      kind="integer", default=2, min=1, max=12, step=1, group="The deal",
                  help="1 = this quarter, 2 = next quarter, and so on."),
            # QA TM-15: plain label but no explanation, and "horizon" then appeared
            # undefined in the result.
            ParamSpec(key="horizon_quarters", label="Quarters to look ahead", kind="integer",
                      default=6, min=2, max=16, step=1, group="The deal",
                  help="How many quarters into the future the forecast runs. Six "
                           "quarters is about 18 months, which is far enough to see whether "
                           "a deal really slips or quietly dies."),
            # QA TM-14: the 2-12 range invited "why would a fiscal year have 12
            # quarters?" and nothing explained what it actually controls.
            ParamSpec(key="quarters_per_year", label="Quarters in your financial year",
                      kind="integer", default=4, min=2, max=12, step=1, group="The deal",
                  help="Almost everyone should leave this at 4. It only decides where "
                           "we draw the line for 'closes this financial year' versus 'closes "
                           "next financial year'. It does not change the odds themselves."),
            ParamSpec(key="pipeline_template", label="Which board to use", kind="choice",
                      default="enterprise", group="The board",
                      choices=[{"value": "enterprise", "label": "Enterprise: RFP, Technical "
                                "Review, CF Approval, Legal, then Won or Dead"},
                               {"value": "simple", "label": "Simple: Qualified, Proposal, "
                                "then Won or Dead"}],
                  help="Only used when your file cannot tell us how deals end."),
            # QA UX-16: these three must add up to 100 and nothing said so.
            ParamSpec(key="default_advance_prob", label="Chance of moving forward a quarter",
                      kind="number", default=45, min=1, max=99, step=1, unit="%",
                      group="The board", sum_group="board_odds",
                  help="Together with the two below, these three must add up to 100. "
                           "If they do not, we scale them for you and say so."),
            ParamSpec(key="default_slip_prob", label="Chance of sitting still a quarter",
                      kind="number", default=50, min=0, max=98, step=1, unit="%",
                      group="The board", sum_group="board_odds",
                  help="The share of deals that stay where they are this quarter."),
            ParamSpec(key="default_die_prob", label="Chance of dying at each step",
                      kind="number", default=5, min=0, max=50, step=1, unit="%",
                      group="The board", sum_group="board_odds",
                  help="The share of deals that are lost for good at each step."),
            ParamSpec(key="fast_track_pct", label="Deals that skip straight to won",
                      kind="number", default=12, min=0, max=60, step=1, unit="%",
                      group="The board",
                  help="Small deals that close almost immediately."),
            ParamSpec(key="use_deal_age", label="Allow for how long a deal has been sitting",
                      kind="bool", default=True, group="Deal aging",
                  help="On: a deal that has been parked for a while is judged more "
                           "likely to slip or die than one that just arrived."),
            ParamSpec(key="stale_age_quarters", label="How long before a deal counts as stale",
                      kind="integer", default=4, min=1, max=40, step=1, unit="quarters",
                      group="Deal aging",
                  help="Deals older than this get progressively more likely to slip or die."),
            ParamSpec(key="age_slip_multiplier", label="How much age raises the risk",
                      kind="number", default=0.25, min=0, max=5, step=0.05,
                      group="Deal aging",
                  help="Past the stale mark, every further stretch of the same length "
                           "makes a deal this much more likely to slip or die."),
            ParamSpec(key="max_age_effect", label="Cap on the extra risk from age",
                      kind="number", default=0.9, min=0, max=5, step=0.1, group="Deal aging",
                  help="Stops a very old deal from being treated as a certain loss - "
                           "the extra risk never rises above this."),
            ParamSpec(key="target_deal_age_quarters",
                      label="How long the deal you are forecasting has been sitting",
                      kind="number", default=0, min=0, max=60, step=1, unit="quarters",
                      group="Deal aging",
                  help="Leave at 0 to use the typical age of the deals in your file."),
            ParamSpec(key="age_units", label="Units in your deal-age column", kind="choice",
                      default="auto", group="Deal aging",
                      choices=[{"value": "auto", "label": "Work it out (quarters, or days if "
                                "you mapped a days column)"},
                               {"value": "quarters", "label": "Quarters"},
                               {"value": "days", "label": "Days (we convert to quarters)"}],
                  help="Only matters if your file has an age column."),
            # QA UX-14: this was capped at max=100 while the deal being forecast is
            # $3,000,000, so the control could not express a realistic figure.
            ParamSpec(key="cash_commitment_per_quarter", label="Cash already committed for that quarter",
                      kind="number", default=0, min=0, max=100_000_000, step=50_000,
                      unit="$", group="Money",
                  help="Optional: spending you have already signed up to on the strength "
                           "of this deal. Leave at 0 if you only want the gap against the "
                           "deal value."),
            # QA TM-17: direction was clear but there were no anchors.
            ParamSpec(key="smoothing", label="Caution on thin history", kind="number",
                      default=0.5, min=0.05, max=5, step=0.05, group="Engine",
                  help="How much we hold back from a stage move we have only seen a "
                           "few times. 0.5 is normal; go above 2 only for a very short "
                           "history, where a single deal would otherwise swing the answer."),
        ],
    )


@register
class PipelineHopperUseCase(UseCase):
    meta = META

    def run(self, req, ctx) -> "H.RunResult":
        P = ctx.params
        rows = ctx.rows

        horizon = int(H.p(P, "horizon_quarters", 6))
        qpy = int(H.p(P, "quarters_per_year", 4))
        forecast_q = int(H.p(P, "sales_forecast_quarter", 2))
        amount = max(0.0, H.as_float(H.p(P, "target_deal_value", 3_000_000)))
        start_stage = str(H.p(P, "starting_stage", "Initial RFP"))
        smoothing = H.as_float(H.p(P, "smoothing", 0.5))
        use_age = bool(H.p(P, "use_deal_age", True))
        stale = max(1.0, H.as_float(H.p(P, "stale_age_quarters", 4)))
        age_mult = max(0.0, H.as_float(H.p(P, "age_slip_multiplier", 0.6)))
        age_cap = max(0.0, H.as_float(H.p(P, "max_age_effect", 0.9)))
        age_units = str(H.p(P, "age_units", "auto") or "auto")
        age_param = max(0.0, H.as_float(H.p(P, "target_deal_age_quarters", 0)))

        ctx.say("Reading how your deals have moved through the pipeline...",
                phase="read", progress=0.05)

        # -- 1. learn the board (and how long each deal sat still) -----------
        learned = False
        n_moves = 0
        counts: dict = {}
        states: list[str] = []
        M_base = np.zeros((0, 0))
        bucket_mats: dict = {}
        bucket_moves: dict = {}
        typical_age = 0.0
        n_deals = 0

        if H.has(rows, "deal_id") and H.has(rows, "stage"):
            ids = H.cat(rows, "deal_id", "")
            stages = H.cat(rows, "stage", "")
            q_idx = [_period_index(v) for v in H.col(rows, "quarter", None)]
            use_q = bool(q_idx) and all(v is not None for v in q_idx)
            e_idx = [_period_index(v) for v in H.col(rows, "entered_quarter", None)]
            age_col = H.col(rows, "deal_age", None) if H.has(rows, "deal_age") else None
            days_col = (H.col(rows, "days_in_stage", None)
                        if H.has(rows, "days_in_stage") else None)
            units = age_units if age_units in ("quarters", "days") else (
                "days" if (days_col is not None and age_col is None) else "quarters")

            records: list[tuple] = []
            ages: list[float] = []
            counters: dict[str, int] = {}
            prev_stage: dict[str, str] = {}
            run_len: dict[str, int] = {}
            last_age: dict[str, float] = {}
            last_stage: dict[str, str] = {}
            for i, (d, s) in enumerate(zip(ids, stages)):
                if not s:
                    continue
                if use_q:
                    t = float(q_idx[i])
                else:
                    counters[d] = counters.get(d, 0) + 1
                    t = float(counters[d])
                age = None
                if age_col is not None and age_col[i] not in (None, ""):
                    age = _age_quarters(age_col[i], units, t)
                elif days_col is not None and days_col[i] not in (None, ""):
                    age = _age_quarters(days_col[i], "days", t)
                if age is None and use_q and e_idx[i] is not None:
                    age = max(0.0, t - float(e_idx[i]))
                if age is None:
                    run_len[d] = (run_len.get(d, 0) + 1) if prev_stage.get(d) == s else 0
                    age = float(run_len[d])
                prev_stage[d] = s
                age = max(0.0, float(age))
                records.append((d, t, s))
                ages.append(age)
                last_age[d] = age
                last_stage[d] = s

            n_deals = len({r[0] for r in records})
            if len(records) >= 20:
                states, M_base, counts = mk.from_event_log(records, smoothing=smoothing)
                n_moves = int(sum(counts.values()))
                learned = len(states) >= 3 and n_moves >= 10
            if learned:
                open_ages = [last_age[d] for d, s in last_stage.items()
                             if not (_is_won(s) or _is_dead(s))]
                pool = open_ages if open_ages else list(last_age.values())
                typical_age = float(np.median(pool)) if pool else 0.0
                bucket_mats, bucket_moves = _age_matrices(records, ages, states,
                                                          stale, smoothing, M_base)

        order_map: dict[str, float] = {}
        if learned and H.has(rows, "stage_order"):
            so = H.num(rows, "stage_order", np.nan)
            buckets: dict[str, list[float]] = {}
            for s, v in zip(H.cat(rows, "stage", ""), so):
                if s and not np.isnan(v):
                    buckets.setdefault(s, []).append(float(v))
            order_map = {s: float(np.median(v)) for s, v in buckets.items()}
            ordered = _order_states(list(states), order_map)
            if ordered != list(states) and M_base.size:
                perm = [states.index(s) for s in ordered]
                M_base = np.asarray(M_base)[np.ix_(perm, perm)]
                for key, bm in list(bucket_mats.items()):
                    if bm is not None:
                        bucket_mats[key] = np.asarray(bm)[np.ix_(perm, perm)]
                states = ordered
            ctx.say("Laid the stages out in the order you gave us.", phase="prepare",
                    progress=0.22, detail=" -> ".join(states))

        # -- 2. where does this board actually end? --------------------------
        w_idx, d_idx = _terminal_indices(states) if states else (None, None)
        history_no_won = bool(learned and w_idx is None)

        if learned and w_idx is not None:
            using_history = True
            ctx.say(f"Learned the hop pattern from {n_moves:,} real stage moves "
                    f"across {n_deals:,} deals.", phase="prepare", progress=0.2,
                    detail=f"{len(states)} stages, endings: "
                           f"{[s for s in states if _is_won(s) or _is_dead(s)]}")
            if not H.has(rows, "quarter"):
                ctx.say("No quarter column we could read, so we used the order the rows "
                        "appear in your file.", phase="validate", level="warn")
            if d_idx is None:
                ctx.say("Your file never records a deal as dead, so we cannot estimate "
                        "how often deals die - we will leave that blank rather than guess.",
                        phase="validate", level="warn")
        else:
            using_history = False
            template = str(H.p(P, "pipeline_template", "enterprise"))
            states = list(FULL_PIPELINE if template != "simple" else SHORT_PIPELINE)
            adv = H.as_float(H.p(P, "default_advance_prob", 45)) / 100.0
            slip = H.as_float(H.p(P, "default_slip_prob", 50)) / 100.0
            die = H.as_float(H.p(P, "default_die_prob", 5)) / 100.0
            # QA UX-16: the three must sum to 100. Scale them rather than
            # building a malformed matrix, and tell the user we did.
            trio = adv + slip + die
            if trio <= 0:
                adv, slip, die, trio = 0.45, 0.50, 0.05, 1.0
                ctx.say("Those three chances added up to nothing, so we used the "
                        "standard board instead.", phase="validate", level="warn")
            elif abs(trio - 1.0) > 0.005:
                ctx.say(f"Forward plus sitting still plus dying came to "
                        f"{trio * 100:.0f}%, not 100%, so we scaled all three to fit.",
                        phase="validate", level="warn")
                adv, slip, die = adv / trio, slip / trio, die / trio
            fast = H.as_float(H.p(P, "fast_track_pct", 12)) / 100.0
            M_base = _default_matrix(states, adv, slip, die, fast)
            w_idx, d_idx = _terminal_indices(states)
            bucket_mats, bucket_moves, counts = {}, {}, {}
            if history_no_won:
                ctx.say(
                    f"None of the {n_deals:,} deals in your file were ever recorded as "
                    f"closed-won, so there is no history to estimate a close rate from. "
                    f"We have switched to your assumption board instead - add closed-won "
                    f"rows to use your own evidence.", phase="validate", level="warn",
                    detail="no absorbing won state found in the learned board")
            else:
                ctx.say("No usable stage history, so we built the board from your "
                        "assumptions: " + " -> ".join(states) + ".",
                        phase="prepare", progress=0.2, level="warn")

        # a closed deal must never reopen
        M_base = _force_absorbing(M_base, states)
        for key, bm in list(bucket_mats.items()):
            bucket_mats[key] = _force_absorbing(bm, states) if bm is not None else None

        # QA DF-12 / UX-15: the stage picker only offered the standard template,
        # so a stage that exists in the user's own file ("Business Case") could
        # not be chosen. Accept anything the file contains, matching loosely.
        if start_stage not in states:
            low = str(start_stage).strip().lower()
            match = next((s for s in states if str(s).strip().lower() == low), None)
            if match is None:
                match = next((s for s in states
                              if low and low in str(s).strip().lower()), None)
            if match is not None:
                start_stage = match
            else:
                trans = [s for s in states if not (_is_won(s) or _is_dead(s))]
                start_stage = trans[0] if trans else states[0]
                ctx.say(f"That starting stage is not on this board - beginning at "
                        f"'{start_stage}' instead. The stages we found in your file are: "
                        f"{', '.join(states)}.", phase="validate", level="warn")

        # -- 3. the board this particular deal should be judged against ------
        target_age = max(0.0, age_param if age_param > 0 else typical_age)

        def matrix_for(age: float) -> np.ndarray:
            bm = bucket_mats.get(_age_bucket(age, stale)) if using_history else None
            return _force_absorbing(
                _aged_matrix(M_base, bm, states, age, stale, age_mult, age_cap, d_idx),
                states)

        M_aged = matrix_for(target_age) if use_age else np.array(M_base, dtype=float)
        aged_is_base = bool(np.allclose(M_aged, M_base))
        if use_age and using_history and not aged_is_base:
            bucket = _age_bucket(target_age, stale)
            ctx.say(f"Deals that sit still get riskier: this one has been going "
                    f"{target_age:.0f} quarters, so we priced it against "
                    f"{'the stale-deal pattern' if bucket == 'stale' else 'the ' + bucket + '-deal pattern'}.",
                    phase="prepare", progress=0.3,
                    detail=f"semi-Markov: age={target_age:.1f}q, stale={stale:.0f}q, "
                           f"bucket={bucket} ({bucket_moves.get(bucket, 0)} learned moves), "
                           f"boost cap={age_cap}")

        # -- 4. roll the odds forward ----------------------------------------
        ctx.say(f"Rolling the odds forward {horizon} quarters from '{start_stage}'...",
                phase="simulate", progress=0.35)
        long_h = max(horizon, 4 * qpy, 24)
        legal_i = next((i for i, s in enumerate(states)
                        if "legal" in str(s).lower() and i not in (w_idx, d_idx)), None)
        base = _roll(M_base, states, start_stage, horizon, long_h, amount,
                     forecast_q, qpy, w_idx, d_idx, legal_i)
        aged = base if aged_is_base else _roll(M_aged, states, start_stage, horizon,
                                               long_h, amount, forecast_q, qpy,
                                               w_idx, d_idx, legal_i)
        if base["clipped"] or aged["clipped"]:
            ctx.say("Some very small rounding drift had to be nudged back into range "
                    "before we showed you the odds.", phase="validate", level="warn")

        fq = aged["fq"]
        p_fq = aged["p_close"][fq - 1]
        source = "history" if using_history else "assumptions"
        wc = _word(p_fq)

        ctx.say(f"Sales says quarter {fq}. The odds from your {source} are "
                f"{H.pct(p_fq)} for closing in that quarter.",
                phase="summarize", progress=0.7)

        # -- 5. assemble ------------------------------------------------------
        n_sims = max(n_moves, horizon * 100)
        res = H.new_result(ctx, META.id, META.title,
                           "Markov chain on your own stage-to-stage history, with deal aging",
                           n_sims)

        age_sentence = ""
        if use_age and not aged_is_base:
            age_sentence = (
                f" That is after allowing for this deal having sat "
                f"{H.count_noun(target_age, 'quarter')} - "
                f"a deal that had just arrived would be at "
                f"{H.pct_fine(base['p_close'][fq - 1])}.")
        if history_no_won:
            res.headline = (
                f"None of the {n_deals:,} deals in your file were ever recorded as "
                f"closed-won, so your history cannot estimate a close rate for this deal. "
                f"We switched to your assumption board: on those assumptions there is a "
                f"{H.pct_fine(p_fq)} chance of closing by Q{fq}.")
        else:
            dead_bit = (f"{H.pct_fine(aged['p_dead'])} never closes at all"
                        if aged["p_dead"] is not None
                        else "an unknown share never closes at all")
            res.headline = (
                f"Sales says quarter {fq}; the odds from your {source} are "
                f"{H.pct_fine(p_fq)} for closing then - {H.pct_fine(aged['p_later'])} of the value "
                f"slips past that date, and {dead_bit}. Plan on "
                f"{H.money(aged['rw_forecast'], compact=True)} of this "
                f"{H.money(amount, compact=True)} deal, not the full amount."
                + age_sentence)
        res.method_plain = (
            ("Your file has no closed-won history, so the odds below come from your "
             "assumption board rather than your data. " if history_no_won else "")
            + "Every deal on your board hops from stage to stage. We counted how your own "
            "deals actually hopped - forward, staying put for a quarter, or dying - and "
            "turned those counts into odds. We also account for how long a deal has been "
            "sitting still: the older it gets, the more likely it is to slip or die, so a "
            "deal that has been parked for a year is priced differently from one that has "
            "just arrived. Then we rolled the odds forward one quarter at a time, which "
            "gives the chance of the deal being anywhere on the board in each future "
            "quarter. Money is then weighted by those odds: if there is a 1-in-8 chance of "
            "closing next quarter, only an eighth of the value deserves to be in next "
            "quarter's plan."
        )

        dead_kpi_value = H.pct_fine(aged["p_dead"]) if aged["p_dead"] is not None else "n/a"
        dead_kpi_plain = (
            "The deal never closes - it stalls or the customer walks."
            if aged["p_dead"] is not None else
            "Your file never records a deal as dead, so we cannot estimate this from "
            "your data.")
        legal_cmp = (f"{H.pct_fine(aged['p_legal_at_horizon'])} still stuck in Legal at the "
                     f"end of the horizon"
                     if aged["p_legal_at_horizon"] is not None else None)
        # QA NEW-13: the sub-label under this KPI changed wording depending on
        # whether the aged board happened to equal the plain one ("if we ignored
        # how long it has been sitting" vs "sales assumed 100%"), so the same
        # comparison read differently in two runs of the same model. One label,
        # always, whenever age is in play.
        age_cmp = (f"if we ignored how long it has been sitting: "
                   f"{H.pct_fine(base['p_close'][fq - 1])}"
                   if use_age else "age not taken into account")

        res.kpis = []
        if history_no_won:
            res.kpis.append(H.kpi(
                f"Chance it closes in Q{fq} - from your history", "n/a", "warn",
                f"None of the {n_deals:,} deals in your file were ever recorded as "
                f"closed-won, so there is nothing in your data to estimate a close rate "
                f"from.", comparison="cannot be estimated"))
            res.kpis.append(H.kpi(
                "What your history can bank on", "—", "warn",
                "With no closed-won deals on record, your history cannot put a reliable "
                "number on this deal. The figures below use your assumption board."))
        res.kpis += [
            H.kpi(f"Chance it closes in Q{fq}, as sales promised"
                  + (" - on your assumptions" if history_no_won else ""),
                  H.pct_fine(p_fq), "bad" if p_fq < 0.5 else "good",
                  f"{wc} chance the money actually lands in the quarter it was promised "
                  f"for.", comparison=age_cmp),
            H.kpi("Chance it slips past that quarter", H.pct_fine(aged["p_later"]), "warn",
                  "The chance the deal does close, but later than the date sales promised."),
            H.kpi("Chance it dies instead", dead_kpi_value,
                  "warn" if aged["p_dead"] is None else "bad", dead_kpi_plain,
                  comparison=legal_cmp),
            H.kpi(f"What you can bank on by Q{fq}", H.money(aged["rw_forecast"], compact=True),
                  "warn",
                  f"The deal value weighted by its real chance of arriving - "
                  f"{H.pct(aged['p_forecast'])} of {H.money(amount, compact=True)}."
                  + (" These are your assumptions, not your history." if history_no_won
                     else "")),
            H.kpi("Cash committed against odds", H.money(aged["gap"], compact=True), "bad",
                  f"You are planning to spend {H.money(aged['gap'], compact=True)} that has "
                  f"only a {H.pct(aged['p_forecast'])} chance of arriving.",
                  comparison=f"{H.money(amount, compact=True)} promised vs "
                             f"{H.money(aged['rw_forecast'], compact=True)} realistic"),
            H.kpi("Quarters until it really closes",
                  f"{aged['exp_q']:.1f}" if aged["p_win"] > 1e-9 else "n/a", "neutral",
                  f"On average, counted from today, for the deals that do close "
                  f"({aged['exp_answer']:.1f} quarters to a final yes or no).",
                  comparison=f"sales promised {fq}"),
            H.kpi("How long this deal has been sitting",
                  H.count_noun(target_age, "quarter"),
                  "neutral",
                  "The older a deal gets, the more likely it is to slip or die - we have "
                  "priced that in.",
                  comparison=f"we treat it as stale after {stale:.0f} quarters"),
        ]

        labels = [f"Q{n}" for n in aged["qs"]]
        excl = {states[w_idx]}
        if d_idx is not None:
            excl.add(states[d_idx])
        live = [s for s in states if s not in excl]
        palette = ["#3b82f6", "#8b5cf6", "#f59e0b", "#06b6d4", "#84cc16", "#e879f9"]

        dist_chart_series = []
        for i, s in enumerate(states):
            dist_chart_series.append({
                "name": s,
                "values": [H.r2(float(aged["dist"][q, i]) * 100.0)
                           for q in range(len(labels))],
                "color": ("#22c55e" if i == w_idx else "#ef4444" if i == d_idx
                          else palette[i % len(palette)]),
            })

        rw_cum = [H.r2(amount * aged["cum_by"][k] / 1e6) for k in range(len(labels))]

        close_series = [{"name": "Chance of closing",
                         "values": [H.r2(p * 100) for p in aged["p_close"]],
                         "color": "#3b82f6"}]
        if use_age and not aged_is_base:
            close_series.append({"name": "If we ignored how long it has been sitting",
                                 "values": [H.r2(p * 100) for p in base["p_close"]],
                                 "color": "#94a3b8", "dash": True})

        age_bands = [
            (f"Fresh (under {stale * 0.5:.0f} quarters)", 0.0, "#22c55e"),
            (f"Settled ({stale * 0.5:.0f}-{stale:.0f} quarters)", stale * 0.75, "#f59e0b"),
            (f"Stale (over {stale:.0f} quarters)", stale * 2.0, "#ef4444"),
        ]

        def roll_at(age: float) -> dict:
            m = matrix_for(age)
            if not use_age or np.allclose(m, M_base):
                return base
            return _roll(m, states, start_stage, horizon, long_h, amount,
                         forecast_q, qpy, w_idx, d_idx, legal_i)

        age_series = [{"name": label, "color": color,
                       "values": [H.r2(v * 100) for v in roll_at(rep)["cum_by"]]}
                      for label, rep, color in age_bands]

        res.charts = [
            H.chart(
                "stage_mix", "stacked_area", "Where the deal actually is, quarter by quarter",
                subtitle=f"Starting from '{start_stage}'. Green is won, red is dead.",
                x_label="Quarter", y_label="Chance of being there",
                x=labels, series=dist_chart_series,
                plain_meaning="Read the green band: that is the chance the deal is already "
                              "closed by each quarter. The grey blocks are the stages it is "
                              "still sitting in.",
            ),
            H.chart(
                "close_by_quarter", "bar", "When it actually closes vs when sales said",
                subtitle=f"Chance of closing in each quarter; sales promised Q{fq}",
                x_label="Quarter", y_label="Chance of closing in that quarter",
                x=labels, series=close_series,
                annotations=[{"x": f"Q{fq}", "y": H.r2(p_fq * 100),
                              "text": f"Sales promised this quarter: {H.pct(p_fq)}"}],
                plain_meaning="The tall bar is the quarter the deal is most likely to close "
                              "in. It is almost never the quarter on the forecast.",
            ),
            H.chart(
                "money_arriving", "line", "Money you can actually plan on",
                subtitle="Cumulative deal value weighted by its chance of arriving",
                x_label="Quarter", y_label="Value you can bank on ($M)",
                x=labels,
                series=[
                    {"name": "Weighted by the real odds",
                     "values": rw_cum, "color": "#22c55e"},
                    {"name": "What sales promised",
                     "values": [H.r2(amount / 1e6)] * len(labels), "color": "#ef4444",
                     "dash": True},
                ],
                plain_meaning="The gap between the green line and the red line is cash you "
                              "would be committing before it has arrived.",
            ),
        ]
        if use_age:
            res.charts.append(H.chart(
                "age_effect", "line", "What sitting still costs you",
                subtitle="Chance the deal has closed by each quarter, by how long it has "
                         "been sitting",
                x_label="Quarter", y_label="Chance it has closed by then",
                x=labels, series=age_series,
                plain_meaning="A deal that has been parked for a long time is less likely "
                              "to land in any given quarter than one that has just arrived. "
                              "That is why the older lines sit lower.",
            ))
        res.charts.append(H.chart(
            "funnel", "funnel", "How far deals get before they die",
            subtitle="Share of deals that ever reach each stage",
            x_label="Stage", y_label="Share of deals reaching it",
            x=list(live) + [states[w_idx]],
            series=[{"name": "Share of deals reaching this stage",
                     "values": _stage_reach(states, M_aged, w_idx, d_idx, live,
                                            counts if using_history else None, rows),
                     "color": "#8b5cf6"}],
            plain_meaning="Every step loses deals. The steepest drop is where the process "
                          "is actually costing you money.",
        ))

        dist_rows = []
        for k, q in enumerate(aged["qs"]):
            # quarter n sits at index n in `dist` (see the indexing contract in _roll)
            row = [labels[k]]
            row += [H.pct_fine(float(aged["dist"][q, i])) for i in range(len(states))]
            dist_rows.append(row)

        close_rows = []
        running = 0.0
        for k, q in enumerate(aged["qs"]):
            running += aged["rw_q"][k]
            dead_cell = (H.pct_fine(aged["cum_dead_by"][k])
                         if aged["cum_dead_by"] is not None else "—")
            close_rows.append([
                f"Q{q}", H.pct_fine(aged["p_close"][k]),
                H.money(aged["rw_q"][k], compact=True),
                H.money(running, compact=True),
                H.pct_fine(aged["cum_by"][k]),
                dead_cell,
                H.money(max(amount - running, 0.0), compact=True),
            ])

        age_rows = [[label, H.pct_fine(roll_at(rep)["cum_by"][fq - 1]),
                     H.pct_fine(max(1.0 - roll_at(rep)["p_win"], 0.0))]
                    for label, rep, _color in age_bands]
        age_rows.append([f"This deal ({H.count_noun(target_age, 'quarter')})",
                         H.pct_fine(aged["cum_by"][fq - 1]),
                         H.pct_fine(max(1.0 - aged["p_win"], 0.0))])

        res.tables = [
            H.table(
                _STAGE_TABLE,
                "The chance of the deal sitting in each stage at the end of every quarter.",
                ["Quarter"] + list(states), dist_rows,
                highlight=[],
                note=f"Read the '{states[w_idx]}' column to see how the chance of closing "
                     f"builds up.",
            ),
            H.table(
                _MONEY_TABLE,
                "What this deal is really worth to each quarter's plan.",
                ["Quarter", "Chance of closing then", "Value earned that quarter",
                 "Value banked by then", "Chance closed by then", "Chance dead by then",
                 "Still not banked"],
                close_rows,
                highlight=[fq - 1],
                note=f"Row {fq} is the quarter sales promised."
                     + (" 'Chance dead by then' is blank because your file never records a "
                        "deal as dead." if aged["cum_dead_by"] is None else ""),
            ),
        ]
        if use_age:
            res.tables.append(H.table(
                "What sitting still costs you",
                "The same deal judged at three ages, plus the age we actually used for it.",
                ["How long it has been sitting", f"Chance it closes by Q{fq}",
                 "Chance it never closes"],
                age_rows,
                highlight=[len(age_rows) - 1],
                note="The older a deal is, the lower it sits. The highlighted row is the "
                     "age we used for the forecast above.",
            ))

        # -- portfolio view ---------------------------------------------------
        if using_history and H.has(rows, "deal_id") and H.has(rows, "amount"):
            port = self._portfolio(rows, states, matrix_for, w_idx, d_idx, long_h)
            if port.rows:
                res.tables.append(port)
            else:
                ctx.say("Every deal in this file has already finished, so there is no open "
                        "pipeline left to re-price - here is how long your past deals "
                        "actually took instead.", phase="summarize", level="warn")
                res.tables.append(self._journey(rows, states, w_idx, d_idx))

        # -- story ------------------------------------------------------------
        if history_no_won:
            measured = (
                f"We read {n_moves:,} real stage moves, but not one deal in your file was "
                f"ever recorded as closed-won. Without a single win to learn from there is "
                f"no honest close rate to quote, so the odds below come from your own "
                f"assumption board instead.")
            measured_ev = f"{n_moves:,} moves, 0 recorded wins"
        elif using_history:
            measured = (
                f"We read {n_moves:,} real stage moves from your history and turned them "
                f"into odds: from each stage a deal moves forward, sits still for a "
                f"quarter, or dies.")
            measured_ev = f"{n_moves:,} observed moves"
        else:
            measured = ("You gave us no stage history, so the board below is built from "
                        "your own assumptions about how deals move.")
            measured_ev = "assumption-driven board"

        res.story = [
            H.story("What we measured", measured, measured_ev),
            H.story(
                "The forecast versus the odds",
                f"Sales has this deal closing in Q{fq}. The odds from your {source} put it at "
                f"{H.pct(p_fq)} for landing in that quarter - and "
                + (f"a {H.pct(aged['p_dead'])} chance of never landing at all."
                   if aged["p_dead"] is not None else
                   "your file does not record enough deaths to say how often they die."),
                f"{H.pct(p_fq)} vs 100%",
            ),
            H.story(
                "Where the time actually goes",
                f"Deals do not walk through the stages, they sit in them. Looking across "
                f"the whole horizon, {H.pct(aged['p_later'])} of the deals that close do so "
                f"later than Q{fq}, and {H.pct(aged['p_after_fy'])} close after this fiscal "
                f"year has ended."
                + (f" Even at the end of Q{horizon}, {H.pct(aged['p_legal_at_horizon'])} "
                   f"are still sitting in Legal." if aged["p_legal_at_horizon"] is not None
                   else ""),
                f"{H.pct(aged['p_legal_at_horizon'])} stuck in Legal"
                if aged["p_legal_at_horizon"] is not None
                else f"{H.pct(aged['p_later'])} slip",
            ),
            H.story(
                "Why the age of the deal matters",
                (f"This deal has been running {target_age:.0f} quarters, and we priced it "
                 f"that way. A deal that had only just arrived would show "
                 f"{H.pct(base['p_close'][fq - 1])} for Q{fq} instead of {H.pct(p_fq)}. "
                 f"Deals that sit still are the ones that quietly die, which is why age is "
                 f"part of the forecast rather than an afterthought."
                 if (use_age and not aged_is_base) else
                 f"We judged this deal at the same odds as every other deal, whatever its "
                 f"age: {H.pct(p_fq)} for Q{fq}. Turn on 'allow for how long a deal has "
                 f"been sitting' to price a deal that has been parked for a while as the "
                 f"bigger risk it usually is."),
                (f"{H.pct(p_fq)} at {target_age:.0f} quarters vs "
                 f"{H.pct(base['p_close'][fq - 1])} when fresh"
                 if (use_age and not aged_is_base) else
                 f"{H.pct(p_fq)} regardless of age"),
            ),
            H.story(
                "The number to plan with",
                f"Weighted by the odds, this deal is worth "
                f"{H.money(aged['rw_forecast'], compact=True)} to Q{fq}'s plan, not "
                f"{H.money(amount, compact=True)}. The difference, "
                f"{H.money(aged['gap'], compact=True)}, is money you would be committing "
                f"before it has any real chance of arriving.",
                f"{H.money(aged['gap'], compact=True)} of over-commitment",
            ),
            H.story(
                "What to do about it",
                (f"Your file cannot support a Q{fq} booking at all - there is no closed-won "
                 f"evidence behind it. " if history_no_won else "")
                + f"Plan the cash against {H.money(aged['rw_forecast'], compact=True)}, hold "
                f"the rest as a contingency, and put the deals that have been sitting the "
                f"longest in front of whoever owns the stage they are stuck in. A forecast "
                f"that says 'Q{fq}' with no odds attached is not a forecast - it is a hope.",
                f"plan on {H.pct(aged['p_forecast'])} of the value",
            ),
        ]

        gap_ratio = aged["gap"] / amount if amount > 0 else 0.0
        p_never = (aged["p_dead"] if aged["p_dead"] is not None
                   else max(1.0 - aged["p_win"], 0.0))
        risk = float(np.clip(25 + 70 * gap_ratio + 30 * p_never, 0, 100))
        committed = H.as_float(H.p(P, "cash_commitment_per_quarter", 0))
        rec_headline = (
            (f"Your history cannot support a Q{fq} booking. On your assumption board, book "
             f"{H.money(aged['rw_forecast'], compact=True)} rather than "
             f"{H.money(amount, compact=True)}.") if history_no_won else
            (f"Do not book {H.money(amount, compact=True)} for Q{fq}. Book "
             f"{H.money(aged['rw_forecast'], compact=True)} and hold the rest back."))
        res.recommendation = Recommendation(
            headline=rec_headline,
            rationale=[
                (f"Your file records no closed-won deals, so the close rate cannot be "
                 f"estimated from it - we used your assumption board instead."
                 if history_no_won else
                 f"The chance of closing in Q{fq} is {H.pct(p_fq)}, not 100%."),
                (f"{H.pct(aged['p_later'])} of the value arrives later than promised."
                 + (f" and {H.pct(aged['p_dead'])} never arrives at all."
                    if aged["p_dead"] is not None else "")),
                f"Planning on the full {H.money(amount, compact=True)} commits "
                f"{H.money(aged['gap'], compact=True)} of cash against a "
                f"{H.pct(aged['p_forecast'])} chance.",
                f"The deal has been running {target_age:.0f} quarters; at that age a deal "
                f"is judged {H.pct(base['p_close'][fq - 1])} likely to close by Q{fq} if it "
                f"were fresh, so sitting still is costing real odds."
                if (use_age and not aged_is_base) else
                f"Deal value {H.money(amount, compact=True)} is carried at full weight by "
                f"the forecast.",
            ],
            actions=[
                f"Put {H.money(aged['rw_forecast'], compact=True)} in the Q{fq} plan and "
                f"treat the remaining {H.money(aged['gap'], compact=True)} as contingent on "
                f"the deal actually reaching the next stage.",
                (f"You told us {H.money(committed, compact=True)} of spending is already "
                 f"committed for that quarter - move "
                 f"{H.money(max(committed - aged['rw_forecast'], 0), compact=True)} of it "
                 f"behind a stage gate." if committed > 0 else
                 "Attach a stage gate to any hiring, inventory or spend that depends on this "
                 "deal."),
                "Re-forecast the whole open pipeline the same way, not just this deal - the "
                "same optimism is almost certainly in the rest of the board.",
                (f"Put the deals sitting in Legal in front of the owner of that step: "
                 f"{H.pct(aged['p_legal_at_horizon'])} are still there at the end of "
                 f"Q{horizon}."
                 if aged["p_legal_at_horizon"] is not None
                 and aged["p_legal_at_horizon"] > 0.15 else
                 "Ask sales to forecast with odds attached, not dates."),
            ],
            risk_score=round(risk, 1),
            risk_label=H.risk_label(risk),
            confidence="High" if using_history else "Low",
            tradeoff=f"Booking less now means a smaller number in this quarter's plan - but it "
                     f"avoids the mid-year freeze that follows a promise that does not arrive.",
        )

        roles_used = [k for k in ("deal_id", "stage", "stage_order", "quarter", "amount",
                                  "entered_quarter", "deal_age", "days_in_stage")
                      if H.has(rows, k)]
        res.assumptions = [
            (f"Learned how deals hop between stages from {n_moves:,} real stage moves "
             f"(the {'quarter column' if H.has(rows, 'quarter') else 'row order'} told us "
             f"what happened first). Paths nobody has taken yet are given a small "
             f"benefit of the doubt rather than a flat zero chance."
             if using_history else
             (f"Your file records no closed-won deals, so the odds come from your "
              f"assumption board: " if history_no_won else "No usable history: board assumed as ")
             + f"{' -> '.join(states)} with "
             f"{H.pct_num(H.as_float(H.p(P, 'default_advance_prob', 45)))} forward, "
             f"{H.pct_num(H.as_float(H.p(P, 'default_slip_prob', 50)))} sitting still, "
             f"{H.pct_num(H.as_float(H.p(P, 'default_die_prob', 5)))} dying per step, and "
             f"{H.pct_num(H.as_float(H.p(P, 'fast_track_pct', 12)))} fast-track wins."),
            "A deal that is won or dead stays that way - we force those stages to be final "
            "even if the history is messy.",
            (f"We priced how long deals sit still: past {stale:.0f} quarters a deal gets "
             f"progressively more likely to slip or die, up to a cap of "
             f"{H.pct(age_cap)} extra risk. The deal being forecast is treated as "
             f"{target_age:.0f} quarters old."
             if use_age else
             "Deal age was switched off, so every deal is judged the same however long it "
             "has been sitting."),
            f"Deal value {H.money(amount)}, starting in '{start_stage}', "
            f"{horizon} quarters ahead, fiscal year = {qpy} quarters.",
            "Each quarter a deal moves at most one stage forward; it may sit still or die. "
            "Deal size does not change the odds.",
            "Stage behaviour is assumed stable: the pattern in your history continues.",
            (f"Columns of your file we actually used: {', '.join(roles_used)}."
             if roles_used else
             "No columns of your file were used - this run is entirely assumption-driven."),
        ]

        if history_no_won:
            res.warnings.append(
                f"None of the {n_deals:,} deals in your file were ever recorded as "
                f"closed-won, so there is no history to estimate a close rate from. Add "
                f"closed-won rows, or switch the board to your assumptions. We have "
                f"switched to your assumption board for the numbers shown here - the close "
                f"rate from your history is reported as 'cannot be estimated'.")
        if not learned:
            res.warnings.append(
                "No usable deal-stage history was mapped, so these odds are your assumptions, "
                "not evidence. Upload a file with deal id, stage and quarter to make this real.")
        if using_history and d_idx is None:
            res.warnings.append(
                "Your file never records a deal as dead, so we cannot estimate how often "
                "deals die. Treat the 'Chance dead by then' column as unknown rather than "
                "zero.")
        if p_fq < 0.35 and using_history:
            res.warnings.append(
                f"Sales says Q{fq}; the model says {H.pct(p_fq)}. Committing the "
                f"full {H.money(amount, compact=True)} on that date is the single biggest risk "
                f"in this plan.")
        if aged["p_dead"] is not None and aged["p_dead"] > 0.25:
            res.warnings.append(
                f"{H.pct(aged['p_dead'])} of deals at this stage never close at all - they "
                f"are not 'slipping', they are gone. Treat that part of the pipeline as zero.")
        if use_age and not aged_is_base and target_age > stale:
            res.warnings.append(
                f"This deal has been sitting {target_age:.0f} quarters - past the "
                f"{stale:.0f}-quarter mark where we start treating a deal as stale. Its "
                f"chance of closing is {H.pct(p_fq)} rather than "
                f"{H.pct(base['p_close'][fq - 1])} for a fresh deal.")
        if not H.has(rows, "quarter") and learned:
            res.warnings.append(
                "We could not read a quarter column, so the order of moves comes from the order "
                "of the rows in your file. Map a period/date column for a cleaner answer.")

        _self_check(res, _STAGE_TABLE, {
            _MONEY_TABLE: ["Value banked by then", "Chance closed by then",
                           "Chance dead by then"],
        })

        ctx.narrator.done(f"Done. Plan on {H.money(aged['rw_forecast'], compact=True)} of "
                          f"{H.money(amount, compact=True)} for Q{fq}, not the full amount.")
        res.log = ctx.narrator.events()
        return res

    # -- portfolio ---------------------------------------------------------
    def _journey(self, rows, states, w_idx, d_idx):
        """No open deals left: show how long finished deals actually took."""
        ids = H.cat(rows, "deal_id", "")
        stages = H.cat(rows, "stage", "")
        amounts = H.num(rows, "amount", 0.0)
        entered = H.cat(rows, "entered_quarter", "")
        q_idx = [_period_index(v) for v in H.col(rows, "quarter", None)]
        use_q = bool(q_idx) and all(v is not None for v in q_idx)

        agg: dict[str, dict] = {}
        for i, d in enumerate(ids):
            if not d:
                continue
            rec = agg.setdefault(d, {"amount": float(amounts[i]), "n": 0, "last": "",
                                     "t0": None, "t1": None, "entered": entered[i] or ""})
            rec["n"] += 1
            rec["last"] = stages[i]
            if use_q:
                rec["t0"] = q_idx[i] if rec["t0"] is None else min(rec["t0"], q_idx[i])
                rec["t1"] = q_idx[i] if rec["t1"] is None else max(rec["t1"], q_idx[i])
        items = sorted(agg.items(), key=lambda kv: -kv[1]["amount"])[:25]
        out = []
        for d, rec in items:
            span = (rec["t1"] - rec["t0"] + 1) if (use_q and rec["t0"] is not None) else rec["n"]
            if w_idx is not None and rec["last"] == states[w_idx]:
                outcome = "Won"
            elif d_idx is not None and rec["last"] == states[d_idx]:
                outcome = "Lost"
            else:
                outcome = "Still open"
            out.append([
                d, H.money(rec["amount"], compact=True), rec["entered"] or "-",
                rec["last"], f"{int(round(span))}", f"{rec['n']}", outcome,
            ])
        return H.table(
            f"How long your past deals actually took (top {len(out)} by value)",
            "Every deal in the file has finished, so this is the ageing evidence instead: "
            "how many quarters each deal really needed.",
            ["Deal", "Value", "Entered", "Ended in", "Quarters taken", "Stage visits",
             "Outcome"],
            out,
            note="Compare 'quarters taken' with the quarter sales usually promises.",
        )

    def _portfolio(self, rows, states, matrix_for, w_idx, d_idx, long_h):
        ids = H.cat(rows, "deal_id", "")
        stages = H.cat(rows, "stage", "")
        amounts = H.num(rows, "amount", 0.0)
        q_idx = [_period_index(v) for v in H.col(rows, "quarter", None)]
        use_q = bool(q_idx) and all(v is not None for v in q_idx)
        e_idx = [_period_index(v) for v in H.col(rows, "entered_quarter", None)]
        age_col = H.col(rows, "deal_age", None) if H.has(rows, "deal_age") else None

        last: dict[str, tuple] = {}
        for i, d in enumerate(ids):
            if not d or not stages[i]:
                continue
            t = float(q_idx[i]) if use_q else float(i)
            if d not in last or t >= last[d][0]:
                # keep the row index of the sighting that established this
                # deal's latest state, so its age is read from that row
                last[d] = (t, stages[i], float(amounts[i]), i)

        def deal_age(d: str, i: int, t: float) -> float:
            if age_col is not None and age_col[i] not in (None, ""):
                a = _age_quarters(age_col[i], "quarters", t)
                if a is not None:
                    return max(0.0, a)
            if use_q and e_idx[i] is not None:
                return max(0.0, t - float(e_idx[i]))
            return 0.0

        # QA NEW-08 / TC-05: this used to loop over every *row*, so a deal with
        # eleven stage sightings got eleven identical rows in a "top 25 deals"
        # table (five deals filled it), and each row took its age from whichever
        # sighting the loop happened to be on - so "Quarters sitting" was frozen
        # at one value per deal. One row per deal now, aged at its latest state.
        open_deals = []
        for d, (t, stage, amt, i) in last.items():
            if not d or not stage:
                continue
            if stage in states and (w_idx is None or stage != states[w_idx]) \
                    and (d_idx is None or stage != states[d_idx]):
                open_deals.append((d, stage, amt, deal_age(d, i, t)))
        open_deals.sort(key=lambda x: -x[2])
        open_deals = open_deals[:25]

        out = []
        for d, stage, amt, age in open_deals:
            M = matrix_for(age)
            fc = mk.forecast(M, states, stage, long_h)
            dist = np.asarray(fc["distribution"], dtype=float)
            cum = np.maximum.accumulate(np.clip(dist[:, w_idx], 0.0, 1.0))
            p1 = float(cum[1])
            p2 = float(cum[2])
            pw = float(np.clip(fc["absorption"].get(states[w_idx], 0.0), 0.0, 1.0))
            first = np.clip(np.diff(cum), 0.0, None)
            num = float(np.sum((np.arange(1, len(first) + 1)) * first))
            eq = num / pw if pw > 1e-9 else 0.0
            out.append([
                d, stage, H.money(amt, compact=True),
                H.pct_fine(p1), H.pct_fine(p2), H.pct_fine(pw),
                f"{eq:.1f}" if pw > 1e-9 else "n/a",
                H.money(amt * pw, compact=True),
                f"{age:.0f}",
            ])
        return H.table(
            f"Your open deals, weighted the same way (top {len(out)})",
            "Every open deal on the board, priced by the odds of it actually closing - "
            "each one judged at its own age.",
            ["Deal", "Stage now", "Value", "Closes next quarter", "Within 2 quarters",
             "Ever closes", "Expected quarter", "Value you can bank on",
             "Quarters sitting"],
            out,
            note="Value you can bank on = deal value x its chance of ever closing.",
        )


# ---------------------------------------------------------------------------
# Board building
# ---------------------------------------------------------------------------
def _default_matrix(states, adv, slip, die, fast) -> np.ndarray:
    """Sequential board: forward, sit still, or die - plus a fast-track win."""
    trans = [(i, s) for i, s in enumerate(states) if not (_is_won(s) or _is_dead(s))]
    won = [i for i, s in enumerate(states) if _is_won(s)]
    dead = [i for i, s in enumerate(states) if _is_dead(s)]
    dead_i = dead[0] if dead else len(states) - 1
    won_i = won[0] if won else len(states) - 1
    if not trans:
        return np.eye(len(states))
    edges = []
    for k, (i, name) in enumerate(trans):
        nxt = trans[k + 1][0] if k + 1 < len(trans) else won_i
        rest = 1.0 - fast if k == 0 else 1.0
        if k == 0:
            edges.append((name, states[won_i], fast))
        edges.append((name, states[nxt], adv * rest))
        edges.append((name, name, slip * rest))
        edges.append((name, states[dead_i], die * rest))
    edges.append((states[won_i], states[won_i], 1.0))
    edges.append((states[dead_i], states[dead_i], 1.0))
    return mk.build_matrix(states, edges)


def _force_absorbing(M: np.ndarray, states: list[str]) -> np.ndarray:
    """A won deal cannot reopen and a dead deal cannot come back."""
    M = np.array(M, dtype=float)
    if M.size == 0:
        return M
    for i, s in enumerate(states):
        if _is_won(s) or _is_dead(s):
            M[i, :] = 0.0
            M[i, i] = 1.0
    rs = M.sum(axis=1)
    for i in range(len(states)):
        if rs[i] <= 0:
            M[i, i] = 1.0
        else:
            M[i] /= rs[i]
    return M


# ---------------------------------------------------------------------------
# Rolling the board forward
# ---------------------------------------------------------------------------
def _roll(M, states, start_stage, horizon, long_h, amount, forecast_q, qpy,
          w_idx: int, d_idx: int | None, legal_i: int | None) -> dict:
    """Roll the board forward and return every number the page needs.

    ``cum`` is the *cumulative* chance the deal has closed - the probability of
    sitting in the closed-won stage, which can only rise because that stage is
    final. It is never a transient stage's mass.
    """
    fc = mk.forecast(M, states, start_stage, long_h)
    dist = np.asarray(fc["distribution"], dtype=float)
    clipped = False

    cum = np.clip(dist[:, w_idx], 0.0, 1.0)
    if np.any(np.diff(cum) < -1e-9):
        clipped = True
    cum = np.maximum.accumulate(cum)

    steps = np.diff(cum)
    if np.any(steps < -1e-12):
        clipped = True
    steps = np.clip(steps, 0.0, None)
    p_close_q = np.concatenate([[cum[0]], steps])
    p_close_q = np.clip(p_close_q, 0.0, 1.0)
    total = float(p_close_q.sum())
    if total > 1.0 + 1e-9:
        p_close_q = p_close_q / total
        clipped = True

    p_win = float(np.clip(fc["absorption"].get(states[w_idx], 0.0), 0.0, 1.0))
    p_dead = None
    cum_dead = None
    if d_idx is not None:
        p_dead = float(np.clip(fc["absorption"].get(states[d_idx], 0.0), 0.0, 1.0))
        cum_dead = np.maximum.accumulate(np.clip(dist[:, d_idx], 0.0, 1.0))

    # INDEXING CONTRACT: `dist[k]` is the state after k quarters, so dist[0] is
    # where the deal is right now and dist[n] is the end of quarter n. Quarter n
    # therefore reads index n - the previous code read index n-1, which reported
    # "chance closed by Q1" as the chance it had *already* closed before we
    # started, and made every quarter one step too pessimistic.
    qs = list(range(1, horizon + 1))
    p_close = [float(np.clip(p_close_q[n], 0.0, 1.0)) for n in qs]
    cum_by = [float(np.clip(cum[n], 0.0, 1.0)) for n in qs]
    cum_dead_by = ([float(np.clip(cum_dead[n], 0.0, 1.0)) for n in qs]
                   if cum_dead is not None else None)
    rw_q = [amount * p for p in p_close]

    fq = min(max(int(forecast_q), 1), horizon)
    p_forecast = float(np.clip(cum[fq], 0.0, 1.0))
    rw_forecast = amount * p_forecast
    gap = max(amount - rw_forecast, 0.0)
    p_later = float(max(p_win - p_forecast, 0.0))
    after_fy_i = min(qpy, long_h)
    p_after_fy = float(max(p_win - float(cum[after_fy_i]), 0.0))
    p_legal = (float(np.clip(dist[horizon, legal_i], 0.0, 1.0))
               if legal_i is not None else None)
    exp_q = _expected_close_quarter(M, states, start_stage, w_idx) if p_win > 1e-9 else 0.0

    return {
        "fc": fc, "dist": dist, "cum": cum, "cum_dead": cum_dead,
        "p_close_q": p_close_q, "p_win": p_win, "p_dead": p_dead, "qs": qs,
        "p_close": p_close, "cum_by": cum_by, "cum_dead_by": cum_dead_by,
        "rw_q": rw_q, "fq": fq, "p_forecast": p_forecast, "rw_forecast": rw_forecast,
        "gap": gap, "p_later": p_later, "p_after_fy": p_after_fy,
        "p_legal_at_horizon": p_legal, "exp_q": exp_q,
        "exp_answer": float(fc["expected_steps_overall"]) + 1.0,
        "clipped": clipped,
    }


def _stage_reach(states, M, w_idx, d_idx, live, counts, rows) -> list[float]:
    """Share of deals that ever reach each stage (from history, else modelled)."""
    if rows and counts:
        ids = H.cat(rows, "deal_id", "")
        stages = H.cat(rows, "stage", "")
        seen: dict[str, set] = {}
        for d, s in zip(ids, stages):
            if d:
                seen.setdefault(d, set()).add(s)
        total = max(len(seen), 1)
        out = [round(100.0 * sum(1 for v in seen.values() if s in v) / total, 1)
               for s in live]
        return out + [round(100.0 * sum(1 for v in seen.values() if states[w_idx] in v)
                            / total, 1)]
    # modelled: chance of ever reaching each stage before dying
    out = []
    k = len(states)
    A = np.array(M, dtype=float)
    reach = 1.0
    for s in live:
        out.append(round(float(100.0 * min(max(reach, 0.0), 1.0)), 1))
        i = states.index(s)
        adv = sum(A[i, j] for j in range(k)
                  if j != i and not _is_dead(states[j]))
        die = float(A[i, d_idx]) if (d_idx is not None and d_idx < k) else 0.0
        reach *= adv / max(adv + die, 1e-9)
    out.append(round(float(100.0 * min(max(reach, 0.0), 1.0)), 1))
    return out


def _expected_close_quarter(M: np.ndarray, states: list[str], start: str,
                            w_idx: int) -> float:
    """Average quarter number in which the deal closes, given that it closes.

    Tilt the chain towards the deals that do win first (each hop is reweighted by
    how likely winning still is from where it lands), then ask how many hops that
    leaves. Without the tilt, the deals that never close stretch the average out
    to something meaningless.
    """
    info = mk.classify(states, M)
    ti = [states.index(s) for s in info["transient"]]
    if not ti or start not in states:
        return 0.0
    b = np.array([float(mk.absorption_probabilities(M, states, s).get(states[w_idx], 0.0))
                  for s in states])
    bi = b[ti]
    Q = np.asarray(M, dtype=float)[np.ix_(ti, ti)]
    Qc = np.zeros_like(Q)
    for a in range(len(ti)):
        if bi[a] <= 1e-12:
            continue
        Qc[a, :] = Q[a, :] * bi / bi[a]
    try:
        t = np.linalg.solve(np.eye(len(ti)) - Qc, np.ones(len(ti)))
    except np.linalg.LinAlgError:
        return 0.0
    si = states.index(start)
    return float(t[ti.index(si)]) + 1.0 if si in ti else 1.0


# ---------------------------------------------------------------------------
# Consistency check - a table that fails this never reaches the user
# ---------------------------------------------------------------------------
def _num_cell(v) -> float | None:
    """Read a formatted cell ('$1.2M', '18.4%', '3') back as a number."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return None if v != v else float(v)
    if not isinstance(v, str):
        return None
    s = v.strip().replace(",", "")
    if s in ("", "—", "-", "–", "n/a", "N/A"):
        return None
    mult = 1.0
    if s.endswith("%"):
        s = s[:-1]
    elif s.endswith("M"):
        s = s[:-1]
        mult = 1e6
    elif s.endswith("k"):
        s = s[:-1]
        mult = 1e3
    if s.startswith("$"):
        s = s[1:]
    try:
        return float(s) * mult
    except ValueError:
        return None


def _self_check(res, stage_table_name: str, cumulative_columns: dict) -> None:
    """Assert the invariants before anything is shown to the user.

    A failure here is a bug in the model, not a bad input, so we raise loudly
    rather than emit a table whose numbers do not mean what they say.
    """
    problems: list[str] = []

    # 1. every table's rows line up with its headers
    for t in res.tables:
        ncol = len(t.columns)
        if ncol == 0 and t.rows:
            problems.append(f"table '{t.name}' has rows but no headers")
            continue
        for i, row in enumerate(t.rows):
            if len(row) != ncol:
                problems.append(f"table '{t.name}' row {i + 1} has {len(row)} cells "
                                f"but {ncol} headers")

    # 2. no probability outside 0-100, no negative money
    def _check_pct(where: str, value) -> None:
        if isinstance(value, str) and _PCT_CELL.match(value.strip()):
            f = float(value.strip().rstrip("%"))
            if f < 0.0 or f > 100.0:
                problems.append(f"{where} shows '{value}', outside 0-100%")

    def _check_negative(where: str, value) -> None:
        if isinstance(value, str) and value.strip().startswith("-$"):
            problems.append(f"{where} shows negative money ('{value}')")

    for k in res.kpis:
        _check_pct(f"KPI '{k.label}'", k.value)
        _check_negative(f"KPI '{k.label}'", k.value)
    for t in res.tables:
        for i, row in enumerate(t.rows):
            for j, cell in enumerate(row):
                _check_pct(f"table '{t.name}' row {i + 1} column {j + 1}", cell)
                _check_negative(f"table '{t.name}' row {i + 1} column {j + 1}", cell)

    # 3. running totals only go up
    for t in res.tables:
        for col in cumulative_columns.get(t.name, []):
            if col not in t.columns:
                continue
            j = t.columns.index(col)
            prev = None
            for i, row in enumerate(t.rows):
                v = _num_cell(row[j])
                if v is None:
                    continue
                if prev is not None and v < prev - 1e-6:
                    problems.append(f"table '{t.name}' column '{col}' falls at row "
                                    f"{i + 1}: {prev} -> {v}")
                prev = v

    # 4. the stage breakdown has to add up
    for t in res.tables:
        if t.name != stage_table_name:
            continue
        for i, row in enumerate(t.rows):
            vals = [v for v in (_num_cell(c) for c in row[1:]) if v is not None]
            if vals and abs(sum(vals) - 100.0) > 0.6:
                problems.append(f"table '{t.name}' row {i + 1} adds up to "
                                f"{sum(vals):.1f}%, not about 100%")

    # 5. charts may not carry impossible numbers
    for c in res.charts:
        for s in c.series:
            for v in s.get("values", []):
                if isinstance(v, (int, float)) and (v < 0 or v != v):
                    problems.append(f"chart '{c.id}' series '{s.get('name')}' holds an "
                                    f"impossible value ({v})")

    if res.recommendation is not None and res.recommendation.risk_score is not None:
        r = res.recommendation.risk_score
        if r < 0 or r > 100:
            problems.append(f"risk score {r} is outside 0-100")

    if problems:
        raise ValueError("pipeline_hopper internal consistency check failed: "
                         + "; ".join(problems[:8]))


def _word(p: float) -> str:
    if p < 0.10:
        return "A slim"
    if p < 0.25:
        return "Roughly a one-in-five"
    if p < 0.40:
        return "About a one-in-three"
    if p < 0.60:
        return "A coin-flip"
    return "A solid"
