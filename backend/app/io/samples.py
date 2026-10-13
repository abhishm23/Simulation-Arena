"""Built-in sample datasets.

Every file is generated from a FIXED seed, so the same bytes come out every
time - on your machine and on the reviewer's. They are also fake but *honest*:
each one contains a real, discoverable effect, so a simulation that looks for
that effect actually finds it instead of reporting noise.

Files land in `backend/data/samples/` and are plain CSV, readable by
`app.io.ingest.read_table`.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np

SEED = 20260101                      # fixed on purpose: never change it
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"


# ---------------------------------------------------------------------------
# small deterministic helpers
# ---------------------------------------------------------------------------
def _money(x: float, dp: int = 2) -> str:
    return f"{float(x):.{dp}f}"


def _round_to(x: float, step: float) -> float:
    return round(float(x) / step) * step


def _pct(x: float, dp: int = 1) -> str:
    return f"{float(x):.{dp}f}"


def _company(rng) -> str:
    first = ["Northwind", "Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne",
             "Cyberdyne", "Soylent", "Tyrell", "Aperture", "Massive", "Hooli",
             "Vehement", "Wonka", "Gringotts", "Duff", "Bluth", "Vandelay",
             "Prestige", "Ironwood", "Halcyon", "Meridian", "Kestrel", "Lumen",
             "Solstice", "Basalt", "Cobalt", "Verity", "Tessera"]
    second = ["Trading", "Industries", "Logistics", "Health", "Systems", "Group",
              "Holdings", "Partners", "Labs", "Foods", "Motors", "Media", "Energy",
              "Retail", "Bank", "Networks", "Chemicals", "Freight"]
    return f"{first[int(rng.integers(0, len(first)))]} {second[int(rng.integers(0, len(second)))]}"


_GOLDEN = 0.6180339887498949


def _quasi_uniform(i: int) -> float:
    """Deterministic, evenly-spread number in [0, 1).

    Used instead of a coin flip where the *rate* matters (how often a line gets
    cancelled, how often a deal is won): plain random draws wander a few points
    away from the rate we planted, this lands on it.
    """
    return (int(i) * _GOLDEN) % 1.0


def _quarter(idx: int) -> str:
    """0 -> 2024-Q1"""
    return f"{2024 + idx // 4}-Q{idx % 4 + 1}"


# ---------------------------------------------------------------------------
# 1. deals.csv  -  do deeper discounts actually buy the win?
# ---------------------------------------------------------------------------
def _gen_deals(rng, n: int = 450) -> tuple[list[str], list[list[Any]]]:
    header = ["deal_id", "customer", "segment", "region", "deal_value", "cost",
              "margin_pct", "discount_pct", "won"]
    segments = ["Enterprise", "Mid-Market", "SMB"]
    seg_w = [0.30, 0.38, 0.32]
    regions = ["North America", "EMEA", "APAC", "LATAM"]

    # win-curve shape per segment (Enterprise is a bit easier to win)
    floor = {"Enterprise": 0.38, "Mid-Market": 0.34, "SMB": 0.30}
    ceiling = {"Enterprise": 0.96, "Mid-Market": 0.95, "SMB": 0.93}
    d50 = {"Enterprise": 11.0, "Mid-Market": 12.5, "SMB": 14.0}

    rows = []
    band_counters: dict[int, int] = {}
    for i in range(n):
        seg = str(rng.choice(segments, p=seg_w))
        region = "" if rng.random() < 0.03 else str(rng.choice(regions))
        if seg == "Enterprise":
            mu, sd, m_base = 12.9, 0.55, 37.0
        elif seg == "Mid-Market":
            mu, sd, m_base = 12.0, 0.50, 41.0
        else:
            mu, sd, m_base = 11.3, 0.50, 45.0

        value = _round_to(float(np.exp(rng.normal(mu, sd))), 1000.0)
        value = max(value, 5000.0)
        margin = float(np.clip(rng.normal(m_base, 4.0), 15.0, 65.0))
        discount = float(np.clip(rng.normal(16.0 + (2.0 if seg == "SMB" else 0.0), 9.0),
                                 0.0, 45.0))
        p = floor[seg] + (ceiling[seg] - floor[seg]) / (
            1.0 + np.exp(-(discount - d50[seg]) / 4.0))
        band = int(min(discount // 5.0, 9.0))            # keeps the curve honest
        band_counters[band] = band_counters.get(band, 0) + 1
        won = 1 if _quasi_uniform(band_counters[band]) < p else 0
        cost = value * (1.0 - margin / 100.0)

        rows.append([
            f"D-{i + 1:04d}",
            _company(rng),
            seg,
            region,
            _money(value, 0),
            _money(cost),
            _pct(margin),
            _pct(discount),
            won,
        ])
    return header, rows


# ---------------------------------------------------------------------------
# 2. quote_lines.csv  -  does a deep software discount poison the support line?
# ---------------------------------------------------------------------------
def _gen_quote_lines(rng, n_quotes: int = 400) -> tuple[list[str], list[list[Any]]]:
    header = ["quote_id", "customer_tier", "line_id", "product_line", "category",
              "list_price", "cost", "discount_pct", "cancelled_within_6m",
              "months_to_cancel"]
    categories = ["Software", "Hardware", "Support", "Services"]
    lines = {
        "Software": ["Platform License", "Analytics Add-on", "Security Suite",
                     "Integration Pack"],
        "Hardware": ["Edge Appliance", "Server Rack", "Network Switch",
                     "Endpoint Devices"],
        "Support": ["24x7 Support", "Premium Support", "Standard Support",
                    "Managed Support"],
        "Services": ["Implementation", "Training", "Migration", "Consulting"],
    }
    price_band = {"Software": (45_000, 130_000), "Hardware": (18_000, 90_000),
                  "Support": (8_000, 32_000), "Services": (12_000, 65_000)}
    gross_margin = {"Software": 0.86, "Hardware": 0.24, "Support": 0.58,
                    "Services": 0.36}
    disc_mu = {"Software": 26.0, "Hardware": 12.0, "Support": 8.0, "Services": 14.0}
    disc_sd = {"Software": 12.0, "Hardware": 8.0, "Support": 6.0, "Services": 9.0}
    # the effect we are planting
    base_cancel = {"Software": 0.10, "Hardware": 0.18, "Support": 0.22,
                   "Services": 0.20}
    lift_cancel = {"Software": 0.06, "Hardware": 0.16, "Support": 0.42,
                   "Services": 0.10}
    tiers = ["Enterprise", "Mid-Market", "SMB"]

    rows = []
    counters: dict[tuple[str, bool], int] = {}
    for q in range(n_quotes):
        quote_id = f"Q-{q + 1:04d}"
        tier = str(rng.choice(tiers, p=[0.28, 0.40, 0.32]))
        tier_push = {"Enterprise": 3.0, "Mid-Market": 0.0, "SMB": -1.0}[tier]
        tier_cancel = {"Enterprise": -0.03, "Mid-Market": 0.0, "SMB": 0.03}[tier]

        order = [int(x) for x in rng.permutation(len(categories))]
        discounts: dict[str, float] = {}
        for ci in order:
            cat = categories[ci]
            mu = disc_mu[cat] + (tier_push if cat == "Software" else 0.0)
            discounts[cat] = float(np.clip(rng.normal(mu, disc_sd[cat]), 0.0, 50.0))
        deep_software = discounts["Software"] > 30.0

        for ci in order:
            cat = categories[ci]
            lo, hi = price_band[cat]
            rng_price = _round_to(float(rng.uniform(lo, hi)), 100.0)
            price = rng_price
            cost = price * (1.0 - gross_margin[cat]) * float(np.clip(
                rng.normal(1.0, 0.03), 0.9, 1.1))
            disc = discounts[cat]
            p = base_cancel[cat] + (lift_cancel[cat] if deep_software else 0.0) \
                + tier_cancel
            key = (cat, deep_software)
            counters[key] = counters.get(key, 0) + 1
            cancelled = 1 if _quasi_uniform(counters[key]) < float(np.clip(p, 0.0, 0.95)) else 0
            months = int(rng.integers(1, 7)) if cancelled else ""
            rows.append([
                quote_id, tier, f"{quote_id}-L{ci + 1}",
                lines[cat][int(rng.integers(0, len(lines[cat])))], cat,
                _money(price, 0), _money(cost), _pct(disc), cancelled, months,
            ])
    return header, rows


# ---------------------------------------------------------------------------
# 3. contract_costs.csv  -  fixed price, costs creeping up, margin squeezed
# ---------------------------------------------------------------------------
def _gen_contract_costs(rng, n_contracts: int = 30) -> tuple[list[str], list[list[Any]]]:
    header = ["contract_id", "customer", "contract_value", "year", "cost_component",
              "base_cost", "margin_pct"]
    components = ["Cloud Hosting", "Third-party Licensing", "Labor", "Logistics"]
    share = {"Cloud Hosting": 0.30, "Third-party Licensing": 0.22,
             "Labor": 0.38, "Logistics": 0.10}
    escalate = {"Cloud Hosting": 0.090, "Third-party Licensing": 0.060,
                "Labor": 0.035, "Logistics": 0.050}
    years = [2023, 2024, 2025, 2026]

    rows = []
    for i in range(n_contracts):
        contract_id = f"C-{i + 1:03d}"
        customer = _company(rng)
        value = _round_to(float(rng.uniform(1.8e6, 12.0e6)), 50_000.0)
        year = int(years[int(rng.integers(0, len(years)))])
        signed_margin = float(rng.uniform(0.34, 0.38))       # margin in year one
        base_total = value * (1.0 - signed_margin)
        total_cost = 0.0
        per_component: dict[str, float] = {}
        for comp in components:
            cost = base_total * share[comp]
            cost *= (1.0 + escalate[comp]) ** (year - 2023)   # costs creep up yearly
            cost *= float(np.clip(rng.normal(1.0, 0.03), 0.9, 1.1))
            per_component[comp] = cost
            total_cost += cost
        margin = (value - total_cost) / value * 100.0
        for comp in components:
            rows.append([
                contract_id, customer, _money(value, 0), year, comp,
                _money(per_component[comp]), _pct(margin),
            ])
    return header, rows


# ---------------------------------------------------------------------------
# 4. market_accounts.csv  -  which accounts can we actually defend?
# ---------------------------------------------------------------------------
def _gen_market_accounts(rng, n: int = 500) -> tuple[list[str], list[list[Any]]]:
    header = ["account_id", "tier", "annual_volume", "price_sensitivity",
              "current_supplier", "margin_pct", "annual_support_cost"]
    tiers = ["Tier-1", "Tier-2", "Tier-3"]
    tier_w = [0.18, 0.37, 0.45]
    vol_mu = {"Tier-1": 15.0, "Tier-2": 13.8, "Tier-3": 12.6}
    sens_mu = {"Tier-1": 0.25, "Tier-2": 0.50, "Tier-3": 0.78}
    margin_mu = {"Tier-1": 43.0, "Tier-2": 33.0, "Tier-3": 19.0}
    support_rate = {"Tier-1": 0.050, "Tier-2": 0.035, "Tier-3": 0.020}
    suppliers = ["Us", "Competitor A", "Competitor B"]
    supplier_w = {"Tier-1": [0.55, 0.28, 0.17],
                  "Tier-2": [0.45, 0.30, 0.25],
                  "Tier-3": [0.30, 0.38, 0.32]}

    rows = []
    for i in range(n):
        tier = str(rng.choice(tiers, p=tier_w))
        volume = _round_to(float(np.exp(rng.normal(vol_mu[tier], 0.45))), 1000.0)
        volume = max(volume, 25_000.0)
        sens = float(np.clip(rng.normal(sens_mu[tier], 0.10), 0.03, 0.98))
        margin = float(np.clip(rng.normal(margin_mu[tier], 4.0)
                               - 8.0 * (sens - sens_mu[tier]), 6.0, 58.0))
        support = volume * support_rate[tier] * float(np.clip(rng.normal(1.0, 0.12),
                                                              0.7, 1.4))
        supplier = str(rng.choice(suppliers, p=supplier_w[tier]))
        rows.append([
            f"A-{i + 1:04d}", tier, _money(volume, 0), f"{sens:.3f}", supplier,
            _pct(margin), _money(support),
        ])
    return header, rows


# ---------------------------------------------------------------------------
# 5. pipeline_stages.csv  -  where do deals actually stall and die?
# ---------------------------------------------------------------------------
def _gen_pipeline_stages(rng, n_deals: int = 420) -> tuple[list[str], list[list[Any]]]:
    header = ["deal_id", "stage", "stage_order", "quarter", "amount",
              "entered_quarter"]
    stage_order = {"Initial RFP": 1, "Technical Review": 2, "CF Approval": 3,
                   "Legal": 4, "Closed Won": 5, "Dead": 6}
    paths = {
        "fast_win": ["Initial RFP", "Closed Won"],
        "win": ["Initial RFP", "Technical Review", "CF Approval", "Legal",
                "Closed Won"],
        "long_win": ["Initial RFP", "Technical Review", "CF Approval", "Legal",
                     "Closed Won"],
        "legal_death": ["Initial RFP", "Technical Review", "CF Approval", "Legal",
                        "Dead"],
        "tech_death": ["Initial RFP", "Technical Review", "Dead"],
        "cf_death": ["Initial RFP", "Technical Review", "CF Approval", "Dead"],
    }
    outcomes = ["fast_win", "win", "long_win", "legal_death", "tech_death",
                "cf_death"]
    weights = [0.12, 0.38, 0.10, 0.18, 0.12, 0.10]      # ~60% win, ~18% die in Legal
    slip_p = 0.55                                        # deals slip a quarter, often

    rows = []
    for i in range(n_deals):
        deal_id = f"P-{i + 1:04d}"
        amount = _round_to(float(np.exp(rng.normal(12.3, 0.75))), 1000.0)
        amount = max(amount, 10_000.0)
        q0 = int(rng.integers(0, 6))                     # 2024-Q1 .. 2025-Q2
        entered = _quarter(q0)
        u = _quasi_uniform(i + 1)                        # exact outcome mix
        outcome = outcomes[int(np.searchsorted(np.cumsum(weights), u))]
        path = list(paths[outcome])
        max_slips = 0 if outcome == "fast_win" else 2     # a fast win does not slip
        forced = 2 if outcome == "long_win" else 0

        quarter = q0
        visits: list[str] = []
        slips_used = 0
        for k, stage in enumerate(path):
            visits.append(stage)
            if k >= len(path) - 1:
                continue
            want = 0
            if slips_used < forced:
                want = min(forced - slips_used, max_slips - slips_used)
            elif slips_used < max_slips and rng.random() < slip_p:
                want = 1
            for _ in range(want):
                visits.append(stage)                     # still stuck here next quarter
            slips_used += want
        for stage in visits:
            rows.append([deal_id, stage, stage_order[stage], _quarter(quarter),
                         _money(amount, 0), entered])
            quarter += 1
    return header, rows


# ---------------------------------------------------------------------------
# 6. approval_flow.csv  -  which approval step is the bottleneck?
# ---------------------------------------------------------------------------
def _gen_approval_flow(rng, horizon: int = 180,
                       rate: float = 1.35) -> tuple[list[str], list[list[Any]]]:
    header = ["request_id", "submitted_day", "stage", "days_in_stage", "approver",
              "deal_value"]
    stages = ["Deal Desk Review", "CF Approval", "Legal"]
    pools = {
        "Deal Desk Review": ["R. Mehta", "S. Alvarez", "J. Okafor", "L. Novak",
                             "M. Haddad", "T. Berg"],
        "CF Approval": ["A. Whitfield", "P. Raman", "D. Castellano",
                        "K. Sorensen"],                       # few approvers
        "Legal": ["N. Beaumont", "C. Iwu", "E. Lindqvist", "F. Moreau",
                  "H. Yamada"],
    }
    # service time (days): CF Approval is slowest and has the fewest approvers
    service_mu = {"Deal Desk Review": 2.0, "CF Approval": 2.6, "Legal": 2.6}
    free_day = {s: [0.0] * len(pools[s]) for s in stages}

    # deterministic arrivals
    arrivals: list[float] = []
    t = 0.0
    while True:
        t += float(rng.exponential(1.0 / rate))
        if t >= horizon:
            break
        arrivals.append(t)

    rows = []
    for i, t_arr in enumerate(arrivals):
        request_id = f"R-{i + 1:04d}"
        value = _round_to(float(np.exp(rng.normal(12.4, 0.7))), 1000.0)
        value = max(value, 20_000.0)
        path = stages if value >= 150_000 else stages[:2]     # small deals skip Legal
        clock = t_arr
        for stage in path:
            pool = pools[stage]
            free = free_day[stage]
            j = int(np.argmin(free))
            start = max(clock, free[j])
            mu = service_mu[stage]
            svc = float(np.clip(rng.lognormal(np.log(mu) - 0.5 * 0.45 ** 2, 0.45),
                                0.5, None))
            finish = start + svc
            free[j] = finish
            rows.append([
                request_id, int(t_arr // 1), stage,
                f"{finish - clock:.1f}", pool[j], _money(value, 0),
            ])
            clock = finish
    return header, rows


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SPECS: list[dict[str, Any]] = [
    {"name": "deals", "seed": 11, "title": "Past deals, discounts and outcomes",
     "use_case_id": "deal_approval",
     "blurb": "450 closed deals with the discount given and whether we won.",
     "gen": _gen_deals},
    {"name": "quote_lines", "seed": 22, "title": "Quotes and their line items",
     "use_case_id": "basket_sabotage",
     "blurb": "1,600 quote lines across Software, Hardware, Support and Services.",
     "gen": _gen_quote_lines},
    {"name": "contract_costs", "seed": 33, "title": "Contract cost components by year",
     "use_case_id": "margin_floor",
     "blurb": "30 contracts, 4 cost components each, with creeping costs.",
     "gen": _gen_contract_costs},
    {"name": "market_accounts", "seed": 44, "title": "Accounts and who supplies them",
     "use_case_id": "competitor_warroom",
     "blurb": "500 accounts with tier, volume, price sensitivity and current supplier.",
     "gen": _gen_market_accounts},
    {"name": "pipeline_stages", "seed": 55, "title": "Deals moving through the pipeline",
     "use_case_id": "pipeline_hopper",
     "blurb": "2,300+ stage visits for 400 deals, including the ones that died.",
     "gen": _gen_pipeline_stages},
    {"name": "approval_flow", "seed": 66, "title": "Approval requests by stage",
     "use_case_id": "approval_bottleneck",
     "blurb": "Every approval stop for 6 months of requests, with who signed.",
     "gen": _gen_approval_flow},
]


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def _meta(spec: dict[str, Any], path: Path) -> dict[str, Any]:
    n_rows, n_cols = 0, 0
    if path.exists():
        with open(path, "r", newline="", encoding="utf-8") as fh:
            for i, _row in enumerate(csv.reader(fh)):
                if i == 0:
                    n_cols = len(_row)
                else:
                    n_rows += 1
    return {
        "name": spec["name"],
        "title": spec["title"],
        "use_case_id": spec["use_case_id"],
        "rows": n_rows,
        "n_cols": n_cols,
        "path": str(path),
        "filename": path.name,
        "blurb": spec.get("blurb", ""),
    }


def sample_path(name: str) -> Path:
    stem = str(name).split(".")[0] if str(name).lower().endswith(".csv") else str(name)
    return SAMPLES_DIR / f"{stem}.csv"


def ensure_samples(force: bool = False) -> list[dict]:
    """Create every sample CSV that is missing (or all of them when force=True)."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for spec in SPECS:
        path = sample_path(spec["name"])
        if force or not path.exists():
            rng = np.random.default_rng(SEED + int(spec["seed"]))
            header, rows = spec["gen"](rng)
            _write_csv(path, header, rows)
        out.append(_meta(spec, path))
    return out


def list_samples() -> list[dict]:
    """[{name, title, use_case_id, rows, n_cols, path}] - generating if needed."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    if not all(sample_path(s["name"]).exists() for s in SPECS):
        ensure_samples()
    return [_meta(s, sample_path(s["name"])) for s in SPECS]


def load_sample(name: str) -> tuple[list[dict], list[str]]:
    """Load a built-in sample through the normal reader: (rows, columns)."""
    from .ingest import read_table
    return read_table(sample_path(name), f"{sample_path(name).name}")
