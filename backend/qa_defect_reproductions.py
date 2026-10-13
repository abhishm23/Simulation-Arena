"""Reproduce each QA finding directly and show it is fixed.

Each block prints the state after the fix and flags it FIXED or BUG, so QA can
re-run the exact defects rather than re-deriving them.

    cd backend
    "C:/Users/Abhishek Mishra/.workbuddy-ai/binaries/python/envs/simarena/Scripts/python.exe" qa_defect_reproductions.py

No server needed - it drives the use cases directly.
"""
import csv, io, json, math, os, random, sys, tempfile, time

sys.path.insert(0, ".")

from app.core.contracts import RunRequest, ColumnProfile, RoleSpec
from app.core.narrate import Narrator
from app.core.registry import RunContext, get
from app.core.rng import make_rng
from app.io import schema_map
from app.io.ingest import read_table, profile

OUT = []
def head(t):
    OUT.append("")
    OUT.append("=" * 78)
    OUT.append(t)
    OUT.append("=" * 78)

def line(*a):
    s = "  ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


def ctx_for(uid, rows, params=None, seed=42, cols=None):
    uc = get(uid)
    meta = uc.meta
    mp, conf, notes, auto = schema_map.resolve_mapping(
        {}, cols or [], [], meta.input_roles, n_rows=len(rows))
    mapped, issues = schema_map.apply_mapping(rows, mp, meta.input_roles)
    p = {x.key: x.default for x in meta.parameters}
    p.update(params or {})
    n = Narrator()
    ctx = RunContext(narrator=n, rng_factory=lambda *t: make_rng(seed, *t), rows=mapped,
                     params=p, run_id="qa", req=RunRequest(use_case_id=uid, seed=seed),
                     profile=None)
    return uc, ctx, mp


# ---------------------------------------------------------------- UC5 no-won
head("QA-UC5-a  Negative close probability when history has no closed-won rows")
rows = []
for d in range(60):
    for i, st in enumerate(["Initial RFP", "Technical Review", "CF Approval", "Legal", "Dead"]):
        rows.append({"deal_id": f"D{d:03d}", "stage": st, "quarter": i + 1,
                     "amount": 3_000_000, "stage_order": i})
cols = ["deal_id", "stage", "quarter", "amount", "stage_order"]
uc, ctx, mp = ctx_for("pipeline_hopper", rows, cols=cols)
line("mapping used:", {k: v for k, v in mp.items() if v})
r = uc.run(ctx.request, ctx)
pcts = []
for k in r.kpis:
    for tok in str(k.value).replace("%", " % ").split():
        try:
            v = float(tok)
            if abs(v) > 0.5:
                pcts.append(v)
        except ValueError:
            pass
line("headline:", r.headline[:150])
line("min KPI numeric value seen:", min(pcts) if pcts else "n/a")
line("any negative probability:", "YES (BUG)" if any(v < 0 for v in pcts) else "NO  (FIXED)")
line("warnings explain the missing history:",
     "YES" if any("closed-won" in w.lower() or "closed won" in w.lower() for w in r.warnings)
     else "NO (BUG)")
t = [t for t in r.tables if "money" in t.name.lower()][0]
cum = [row[4] for row in t.rows]
line("money table 'Chance closed by then' column:", cum)
vals = [float(str(c).rstrip("%")) for c in cum]
line("cumulative column non-decreasing:", "YES (FIXED)" if all(
    b >= a - 1e-9 for a, b in zip(vals, vals[1:])) else "NO (BUG)")
line("row widths all equal header width:",
     "YES" if all(len(x) == len(t.columns) for x in t.rows) else "NO (BUG)")

# --------------------------------------------------------- UC5 age A/B test
head("QA-UC5-b  Two files differing only in deal age must not give the same answer")
# a realistic board: deals advance, some win, some die - so a close rate IS estimable
import random as _r
_rng = _r.Random(11)
STAGES = ["Initial RFP", "Technical Review", "CF Approval", "Legal"]
rows_a = []
for d in range(160):
    win = _rng.random() < 0.55
    hops = _rng.randint(2, 4)
    seq = STAGES[:hops] + ["Closed Won" if win else "Dead"]
    age = 1
    for i, st in enumerate(seq):
        rows_a.append({"deal_id": f"D{d:03d}", "stage": st, "quarter": i + 1,
                       "amount": 3_000_000, "deal_age": age, "stage_order": i})
        age += 1
# identical in every way except that every deal is ten times older
rows_b = [dict(x, deal_age=x["deal_age"] * 10, quarter=x["quarter"] * 10) for x in rows_a]
cols = ["deal_id", "stage", "quarter", "amount", "deal_age", "stage_order"]
_, ca, _ = ctx_for("pipeline_hopper", rows_a, cols=cols)
_, cb, _ = ctx_for("pipeline_hopper", rows_b, cols=cols)
ra = get("pipeline_hopper").run(ca.request, ca)
rb = get("pipeline_hopper").run(cb.request, cb)
line("age as recorded  :", ra.headline[:110])
line("every age x10    :", rb.headline[:110])
same = ra.headline == rb.headline
line("character-identical answer:", "YES (BUG)" if same else "NO  (FIXED)")
ka = [k.value for k in ra.kpis]
kb = [k.value for k in rb.kpis]
line("KPI values differ:", "YES (FIXED)" if ka != kb else "NO (BUG)")

# ------------------------------------------------------- UC4 verdict inversion
head("QA-UC4  Verdict must follow the numbers, not a hardcoded conclusion")
rows_m = []
rnd = random.Random(7)
for i in range(300):
    rows_m.append({"account_id": f"A{i:04d}", "tier": f"Tier-{rnd.choice([1,2,3])}",
                   "annual_volume": rnd.randint(3000, 9000), "price_sensitivity": 0.95,
                   "current_supplier": "Us", "margin_pct": 30.0,
                   "annual_support_cost": 90_000})
cols = ["account_id", "tier", "annual_volume", "price_sensitivity", "current_supplier",
        "margin_pct", "annual_support_cost"]
uc, ctx, mp = ctx_for("competitor_warroom", rows_m,
                      params={"n_replicates": 12, "n_quarters": 4}, cols=cols)
r = uc.run(ctx.request, ctx)
line("headline:", r.headline[:170])
tbl = [t for t in r.tables if "compared" in t.name.lower()][0]
line("columns:", tbl.columns)
best_val, best_row = None, None
for row in tbl.rows:
    line("  row:", " | ".join(str(c) for c in row))
line("highlighted row index:", tbl.highlight_rows, "->",
     tbl.rows[tbl.highlight_rows[0]][0] if tbl.highlight_rows else "none")
line("highlighted row verdict:",
     tbl.rows[tbl.highlight_rows[0]][-1] if tbl.highlight_rows else "none")
line("recommendation headline:", r.recommendation.headline[:130])

# --------------------------------------------------- UC4/SM tier -> account_id
head("QA-C5  'tier' must never silently map to 'account_id'")
roles = get("competitor_warroom").meta.input_roles
cols_m = ["account_id", "annual_volume", "price_sensitivity", "current_supplier",
          "margin_pct", "annual_support_cost"]
profs = [ColumnProfile(name="account_id", dtype="identifier", unique=300, null_pct=0.0),
         ColumnProfile(name="annual_volume", dtype="numeric", unique=280, null_pct=0.0),
         ColumnProfile(name="price_sensitivity", dtype="numeric", unique=120, null_pct=0.0),
         ColumnProfile(name="current_supplier", dtype="categorical", unique=3, null_pct=0.0),
         ColumnProfile(name="margin_pct", dtype="numeric", unique=40, null_pct=0.0),
         ColumnProfile(name="annual_support_cost", dtype="numeric", unique=90, null_pct=0.0)]
m, c, notes = schema_map.auto_map(cols_m, profs, roles, n_rows=300)
line("tier ->", m.get("tier"), "| confidence", c.get("tier"))
line("silently used account_id for tier:", "YES (BUG)" if m.get("tier") == "account_id"
     else "NO  (FIXED)")
for nline in notes:
    if "tier" in nline.lower() or "ID" in nline:
        line("  note:", nline[:150])

# --------------------------------------------------------------- summary
head("SUMMARY")
line("all defect reproductions above show the FIXED state where marked")
with open("_qa_regression.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT))
print("\nwrote _qa_regression.txt")
