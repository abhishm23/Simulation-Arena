"""Self-contained HTML dashboard for a RunResult.

One file, zero external requests: inline CSS, inline SVG charts, no web fonts,
no scripts. It renders identically offline, forever, and prints cleanly.

Section order (the order an executive reads in):
    header -> what we did -> KPIs -> how we got there -> charts -> tables
    -> recommendation -> assumptions & warnings -> appendix (run log)
"""
from __future__ import annotations

import datetime as _dt
from html import escape as _escape
from typing import Any, Optional, Sequence
from urllib.parse import quote

from ..core.contracts import RunResult
from .brand import FAVICON_SVG, mark as brand_mark
from .charts import render_chart

TONE_COLOR = {
    "good": "#22c55e",
    "bad": "#ef4444",
    "warn": "#f59e0b",
    "warning": "#f59e0b",
    "neutral": "#3b82f6",
    "info": "#3b82f6",
}


# ---------------------------------------------------------------------------
# Defensive primitives
# ---------------------------------------------------------------------------
def esc(s: Any) -> str:
    if s is None:
        return ""
    return _escape(str(s), quote=True)


def text(s: Any, default: str = "") -> str:
    if s is None:
        return default
    s = str(s).strip()
    return s if s else default


def clip(s: Any, n: int) -> str:
    s = text(s)
    return s if len(s) <= n else s[: max(1, n - 1)].rstrip() + "\u2026"


def lines(items: Any) -> list[str]:
    """Coerce anything into a list of clean, non-empty strings."""
    if not items:
        return []
    if isinstance(items, str):
        items = [items]
    out: list[str] = []
    try:
        seq = list(items)
    except TypeError:
        return []
    for v in seq:
        s = text(v)
        if s:
            out.append(s)
    return out


def tone(t: Any) -> str:
    return TONE_COLOR.get(text(t, "neutral").lower(), TONE_COLOR["neutral"])


def stamp() -> str:
    try:
        return _dt.datetime.now().strftime("%d %b %Y at %H:%M")
    except Exception:
        return ""


def cell(v: Any) -> str:
    if v is None:
        return "&mdash;"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return "&mdash;"
        if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
            return f"{int(round(v)):,}"
        return f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    s = str(v).strip()
    return esc(s) if s else "&mdash;"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
:root{
  --bg:#0b1220; --panel:#0f172a; --panel2:#131f36; --line:#1e293b;
  --line2:#24344c; --text:#e2e8f0; --muted:#94a3b8; --dim:#64748b;
  --good:#22c55e; --bad:#ef4444; --warn:#f59e0b; --info:#3b82f6; --violet:#a78bfa;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; color:var(--text); background:#0b1220;
  background-image:radial-gradient(1100px 520px at 8% -8%, #172742 0%, rgba(11,18,32,0) 60%),
                   radial-gradient(900px 480px at 100% 0%, #14203a 0%, rgba(11,18,32,0) 55%);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
.wrap{max-width:1180px;margin:0 auto;padding:26px 20px 70px}
a{color:#7dd3fc}
h1,h2,h3,h4{margin:0;line-height:1.25;font-weight:650;letter-spacing:-0.01em}
p{margin:0 0 10px}
ul,ol{margin:0 0 0 20px;padding:0}
li{margin:0 0 7px}

.nav{position:sticky;top:0;z-index:20;margin:0 -20px 22px;padding:10px 20px;
  background:rgba(11,18,32,0.86);border-bottom:1px solid var(--line);
  backdrop-filter:blur(8px)}
.nav ol{list-style:none;display:flex;flex-wrap:wrap;gap:8px;margin:0;padding:0}
.nav a{display:inline-block;padding:5px 12px;border:1px solid var(--line2);border-radius:999px;
  color:var(--muted);text-decoration:none;font-size:12.5px;white-space:nowrap}
.nav a:hover{color:var(--text);border-color:#38506f}

.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);
  font-weight:600}
.card{background:linear-gradient(180deg,#121d33 0%, #0f172a 100%);
  border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:16px}
.card.tight{padding:14px 16px}
.grid{display:grid;gap:14px}
.kpis{grid-template-columns:repeat(auto-fit,minmax(215px,1fr))}
.two{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}

header.hero{margin-bottom:18px}
.brandbar{display:flex;align-items:center;gap:9px;margin-bottom:16px}
.brandbar svg{display:block;flex:none}
.brandname{font-size:15px;font-weight:700;letter-spacing:.2px;color:var(--text)}
.brandname i{font-style:normal;
  background:linear-gradient(135deg,#22c55e,#3b82f6);-webkit-background-clip:text;
  background-clip:text;color:transparent}
header.hero h1{font-size:clamp(22px,3.1vw,32px);margin:10px 0 12px}
.headline{font-size:clamp(20px,2.6vw,28px);font-weight:660;line-height:1.3;
  color:#f1f5f9;margin:0 0 14px;max-width:900px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.pill{display:inline-flex;align-items:center;gap:7px;background:#101c33;border:1px solid var(--line2);
  border-radius:999px;padding:5px 12px;font-size:12.5px;color:var(--muted)}
.pill b{color:var(--text);font-weight:600}
.pill.ok{border-color:rgba(34,197,94,.45);color:#bbf7d0}
.dot{width:7px;height:7px;border-radius:50%;background:var(--info);display:inline-block}

h2.section{font-size:19px;margin:30px 0 13px;display:flex;align-items:center;gap:10px}
h2.section .num{width:26px;height:26px;flex:0 0 26px;border-radius:8px;background:#16233d;
  border:1px solid var(--line2);display:grid;place-items:center;font-size:12.5px;color:var(--muted)}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 12px}

.kpi{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:13px;
  background:linear-gradient(180deg,#121d33,#0f172a);padding:15px 16px 14px}
.kpi:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--tone,#3b82f6)}
.kpi .label{font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);
  font-weight:600;margin-bottom:8px}
.kpi .value{font-size:27px;font-weight:680;letter-spacing:-0.02em;color:var(--tone,#e2e8f0);
  line-height:1.1;word-break:break-word}
.kpi .unit{font-size:14px;font-weight:500;color:var(--muted);margin-left:4px}
.kpi .plain{font-size:13px;color:#cbd5e1;margin-top:9px}
.kpi .cmp{display:inline-block;margin-top:10px;font-size:12px;color:#cbd5e1;background:#16233d;
  border:1px solid var(--line2);border-radius:7px;padding:3px 9px}

.story{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.step{border:1px solid var(--line);border-radius:13px;background:#101b30;padding:16px 17px}
.step .n{width:27px;height:27px;border-radius:50%;background:#1c2c48;border:1px solid var(--line2);
  display:grid;place-items:center;font-size:13px;font-weight:700;color:#93c5fd;margin-bottom:10px}
.step h3{font-size:15.5px;margin-bottom:7px}
.step p{font-size:13.5px;color:#cbd5e1;margin:0}
.step .ev{margin-top:11px;font-size:12px;color:var(--dim);border-top:1px dashed var(--line2);
  padding-top:9px}

.chart{margin-bottom:18px}
.chart h3{font-size:16.5px;margin-bottom:4px}
.chart .sub{margin-bottom:12px}
.chart svg{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:12px;
  background:#0f172a}
.meaning{margin-top:12px;border-left:3px solid var(--info);background:#101b30;border-radius:0 10px 10px 0;
  padding:11px 14px;font-size:13.5px;color:#cbd5e1}
.meaning b{display:block;font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--dim);margin-bottom:5px;font-weight:700}

.tblwrap{border:1px solid var(--line);border-radius:12px;overflow:auto;background:#0f172a}
table{width:100%;border-collapse:collapse;font-size:13px}
thead th{position:sticky;top:0;background:#16233d;color:#cbd5e1;text-align:left;font-weight:650;
  padding:10px 12px;border-bottom:1px solid var(--line2);white-space:nowrap;font-size:12px;
  letter-spacing:.03em;text-transform:uppercase}
tbody td{padding:9px 12px;border-bottom:1px solid #172236;color:#dbe3ee;vertical-align:top;
  overflow-wrap:anywhere}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:#131f36}
tr.hl{background:rgba(34,197,94,.10)!important}
tr.hl td:first-child{box-shadow:inset 3px 0 0 #22c55e}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.note{margin-top:10px;font-size:12.5px;color:var(--muted)}

.rec{border:1px solid rgba(34,197,94,.28);background:linear-gradient(180deg,#12251f,#0f172a)}
.rec h3{font-size:20px;margin:2px 0 14px;line-height:1.32;color:#f1f5f9}
.badge{display:inline-flex;align-items:center;gap:7px;border-radius:999px;padding:5px 12px;
  font-size:12.5px;font-weight:640;border:1px solid}
.meter{height:7px;border-radius:99px;background:#16233d;overflow:hidden;margin:12px 0 4px}
.meter i{display:block;height:100%;border-radius:99px}
.recgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-top:14px}
.callout{border-left:3px solid var(--warn);background:rgba(245,158,11,.08);border-radius:0 10px 10px 0;
  padding:12px 14px;font-size:13.5px;color:#fde68a;margin-top:14px}
.warnbox{border-left:3px solid var(--warn);background:rgba(245,158,11,.09);
  border-radius:0 10px 10px 0;padding:13px 16px;margin-bottom:12px}
.warnbox h4{font-size:13px;color:#fcd34d;margin-bottom:8px;letter-spacing:.03em;text-transform:uppercase}
.warnbox li{color:#fde68a;font-size:13.5px}

.log{font:12px/1.7 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#0d1628}
.log .row{display:grid;grid-template-columns:78px 108px 1fr;gap:10px;padding:7px 12px;
  border-bottom:1px solid #172236;align-items:start}
.log .row:last-child{border-bottom:none}
.log .row.warn{background:rgba(245,158,11,.07)}
.log .row.error{background:rgba(239,68,68,.08)}
.log .row.success{background:rgba(34,197,94,.07)}
.log .t{color:var(--dim)}
.log .ph{color:#93c5fd}
.log .msg{color:#dbe3ee;overflow-wrap:anywhere}
.log .det{color:#7c8ba1;grid-column:2/4;padding-top:2px;font-size:11.5px}
footer{margin-top:34px;padding-top:16px;border-top:1px solid var(--line);color:var(--dim);font-size:12px}

@media (max-width:640px){
  .wrap{padding:18px 13px 50px}
  .log .row{grid-template-columns:64px 1fr;}
  .log .det{grid-column:1/3}
  .kpi .value{font-size:23px}
}
@media print{
  body{background:#fff;color:#111827}
  .nav{display:none}
  .wrap{max-width:none;padding:0}
  .card,.kpi,.step,.chart svg,.tblwrap,.log{border-color:#cbd5e1;background:#fff!important;
    box-shadow:none;break-inside:avoid;page-break-inside:avoid}
  h1,h2,h3,.headline,.kpi .value,.rec h3{color:#0f172a!important}
  .sub,.kpi .label,.kpi .plain,.step p,.meaning,.note,footer{color:#374151!important}
  .step .n,.kpi .cmp,.log{background:#f3f4f6!important;color:#111827!important}
  .log .msg,.log .ph,.log .t,.log .det{color:#111827!important}
  .meaning,.callout,.warnbox{border-left-color:#111827;background:#f9fafb!important}
  .warnbox li,.warnbox h4,.callout{color:#111827!important}
  thead th{background:#f3f4f6!important;color:#111827!important}
  tbody td{color:#111827!important;border-bottom-color:#e5e7eb!important}
  tr.hl{background:#ecfdf5!important}
}
"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _nav(sections: Sequence[tuple[str, str]]) -> str:
    items = "".join(
        f'<li><a href="#{esc(anchor)}">{esc(label)}</a></li>' for anchor, label in sections
    )
    return f'<nav class="nav"><ol>{items}</ol></nav>'


def _header(r: RunResult) -> str:
    title = text(r.use_case_title) or text(r.use_case_id).replace("_", " ").title() or "Simulation report"
    headline = text(r.headline, "Here is what the simulation found.")
    pills = [
        f'<span class="pill">Seed <b>{esc(r.seed)}</b></span>',
        f'<span class="pill">Run <b>{esc(clip(r.run_id, 26))}</b></span>',
        f'<span class="pill">{esc(stamp())}</span>',
        '<span class="pill ok"><span class="dot" style="background:#22c55e"></span>'
        'Reproducible: same seed, same numbers</span>',
    ]
    if r.n_simulations:
        pills.append(f'<span class="pill">Scenarios tested <b>{r.n_simulations:,}</b></span>')
    if text(r.model_used):
        pills.append(f'<span class="pill">Method <b>{esc(clip(r.model_used, 60))}</b></span>')
    return (
        '<header class="hero">'
        f'<div class="brandbar">{brand_mark(34)}'
        '<span class="brandname">Simulation<i>Arena</i></span></div>'
        f'<div class="eyebrow">{esc(text(r.use_case_id).replace("_", " "))}</div>'
        f'<h1>{esc(title)}</h1>'
        f'<p class="headline">{esc(headline)}</p>'
        f'<div class="meta">{"".join(pills)}</div>'
        "</header>"
    )


def _what_we_did(r: RunResult) -> str:
    body = text(r.method_plain, "No method description was produced for this run.")
    return (
        '<h2 class="section" id="method"><span class="num">1</span>What we did</h2>'
        f'<div class="card"><p style="margin:0;font-size:15px;color:#dbe3ee">{esc(body)}</p></div>'
    )


def _kpis(r: RunResult) -> str:
    if not r.kpis:
        return ""
    cards = []
    for k in r.kpis[:12]:
        c = tone(getattr(k, "tone", "neutral"))
        unit = f'<span class="unit">{esc(k.unit)}</span>' if text(getattr(k, "unit", "")) else ""
        plain = (f'<div class="plain">{esc(clip(k.plain, 300))}</div>'
                 if text(getattr(k, "plain", "")) else "")
        cmp_ = (f'<div class="cmp">{esc(clip(k.comparison, 120))}</div>'
                if text(getattr(k, "comparison", "")) else "")
        cards.append(
            f'<div class="kpi" style="--tone:{c}">'
            f'<div class="label">{esc(clip(k.label, 90))}</div>'
            f'<div class="value">{esc(clip(k.value, 40))}{unit}</div>'
            f"{plain}{cmp_}</div>"
        )
    return (
        '<h2 class="section" id="kpis"><span class="num">2</span>The numbers that matter</h2>'
        f'<div class="grid kpis">{"".join(cards)}</div>'
    )


def _story(r: RunResult) -> str:
    if not r.story:
        return ""
    steps = []
    for i, s in enumerate(r.story[:10], start=1):
        ev = (f'<div class="ev">{esc(clip(s.evidence, 240))}</div>'
              if text(getattr(s, "evidence", "")) else "")
        steps.append(
            '<div class="step">'
            f'<div class="n">{i}</div>'
            f"<h3>{esc(clip(s.title, 120))}</h3>"
            f"<p>{esc(clip(s.detail, 700))}</p>{ev}</div>"
        )
    return (
        '<h2 class="section" id="story"><span class="num">3</span>How we got there</h2>'
        f'<div class="story">{"".join(steps)}</div>'
    )


def _charts(r: RunResult) -> str:
    if not r.charts:
        return ""
    blocks = []
    for spec in r.charts[:12]:
        svg = render_chart(spec)
        sub = (f'<p class="sub">{esc(clip(spec.subtitle, 220))}</p>'
               if text(getattr(spec, "subtitle", "")) else "")
        meaning = (f'<div class="meaning"><b>What this shows</b>'
                   f'{esc(clip(spec.plain_meaning, 600))}</div>'
                   if text(getattr(spec, "plain_meaning", "")) else "")
        blocks.append(
            '<div class="card chart">'
            f"<h3>{esc(clip(spec.title, 140))}</h3>{sub}{svg}{meaning}</div>"
        )
    return (
        '<h2 class="section" id="charts"><span class="num">4</span>Charts</h2>'
        + "".join(blocks)
    )


def _looks_numeric(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    s = str(v).strip()
    if not s:
        return False
    s = s.replace(",", "").replace("%", "").replace("$", "")
    try:
        float(s)
        return True
    except ValueError:
        return False


def _tables(r: RunResult) -> str:
    if not r.tables:
        return ""
    blocks = []
    for t in r.tables[:12]:
        cols = [text(c, f"Column {i + 1}") for i, c in enumerate(t.columns or [])]
        rows = list(t.rows or [])
        if not rows:
            blocks.append(
                '<div class="card"><h3 style="font-size:15.5px">'
                f'{esc(clip(t.name, 120))}</h3>'
                '<p class="sub" style="margin:8px 0 0">No rows were produced for this table.</p>'
                "</div>"
            )
            continue
        hl = set()
        for i in (t.highlight_rows or []):
            try:
                hl.add(int(i))
            except (TypeError, ValueError):
                continue
        ncols = max([len(cols)] + [len(rw) for rw in rows])
        cols = cols + [f"Column {i + 1}" for i in range(len(cols), ncols)]
        # right-align numeric columns
        numeric = [True] * ncols
        for rw in rows[:40]:
            for i in range(ncols):
                v = rw[i] if i < len(rw) else None
                if v is not None and str(v).strip() != "" and not _looks_numeric(v):
                    numeric[i] = False
        head = "".join(
            f'<th class="{"num" if numeric[i] else ""}">{esc(clip(c, 60))}</th>'
            for i, c in enumerate(cols)
        )
        body = []
        for ri, rw in enumerate(rows[:300]):
            tds = []
            for i in range(ncols):
                v = rw[i] if i < len(rw) else None
                cls = "num" if numeric[i] else ""
                tds.append(f'<td class="{cls}">{cell(v)}</td>')
            body.append(f'<tr class="{"hl" if ri in hl else ""}">{"".join(tds)}</tr>')
        desc = (f'<p class="sub">{esc(clip(t.description, 400))}</p>'
                if text(getattr(t, "description", "")) else "")
        note = (f'<p class="note">{esc(clip(t.highlight_note, 300))}</p>'
                if text(getattr(t, "highlight_note", "")) else "")
        extra = ""
        if len(rows) > 300:
            extra = (f'<p class="note">Showing the first 300 of {len(rows):,} rows.</p>')
        blocks.append(
            '<div class="card">'
            f'<h3 style="font-size:15.5px;margin-bottom:6px">{esc(clip(t.name, 120))}</h3>'
            f"{desc}"
            f'<div class="tblwrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>{note}{extra}</div>'
        )
    return (
        '<h2 class="section" id="tables"><span class="num">5</span>Tables</h2>'
        + "".join(blocks)
    )


def _recommendation(r: RunResult) -> str:
    rec = r.recommendation
    if rec is None:
        return ""
    headline = text(rec.headline, text(r.headline, "What we suggest doing next."))
    score = getattr(rec, "risk_score", None)
    try:
        score_f = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_f = None
    if score_f is None:
        score_f = 0.0
    score_f = max(0.0, min(100.0, score_f))
    if score_f >= 70:
        rc = TONE_COLOR["bad"]
    elif score_f >= 45:
        rc = TONE_COLOR["warn"]
    elif score_f >= 20:
        rc = "#eab308"
    else:
        rc = TONE_COLOR["good"]
    rlabel = text(getattr(rec, "risk_label", ""), "Not scored")
    conf = text(getattr(rec, "confidence", ""), "Medium")

    rat = "".join(f"<li>{esc(clip(x, 400))}</li>" for x in lines(rec.rationale))
    acts = "".join(f"<li>{esc(clip(x, 400))}</li>" for x in lines(rec.actions))

    badges = (
        f'<span class="badge" style="color:{rc};border-color:{rc}55;'
        f'background:{rc}14">Risk {esc(clip(rlabel, 24))}'
        + (f" &middot; {score_f:.0f}/100" if getattr(rec, "risk_score", None) is not None else "")
        + "</span>"
        f'<span class="badge" style="color:#93c5fd;border-color:#3b82f655;background:#3b82f614">'
        f"Confidence: {esc(clip(conf, 24))}</span>"
    )
    tradeoff = (f'<div class="callout"><b style="color:#fcd34d">The trade-off</b>'
                f'{esc(clip(rec.tradeoff, 500))}</div>'
                if text(getattr(rec, "tradeoff", "")) else "")
    return (
        '<h2 class="section" id="recommendation"><span class="num">6</span>What we recommend</h2>'
        '<div class="card rec">'
        f"<h3>{esc(clip(headline, 260))}</h3>"
        f'<div class="meta" style="margin-top:0">{badges}</div>'
        f'<div class="meter"><i style="width:{score_f:.0f}%;background:{rc}"></i></div>'
        '<div style="font-size:12px;color:#94a3b8">Risk of getting this wrong: '
        f"{score_f:.0f} out of 100</div>"
        '<div class="recgrid">'
        + (f'<div><div class="eyebrow" style="margin-bottom:8px">Why</div><ul>{rat}</ul></div>'
           if rat else "")
        + (f'<div><div class="eyebrow" style="margin-bottom:8px">What to do</div><ol>{acts}</ol></div>'
           if acts else "")
        + "</div>" + tradeoff + "</div>"
    )


def _assumptions(r: RunResult) -> str:
    al = lines(r.assumptions)
    wl = lines(r.warnings)
    out = ['<h2 class="section" id="assumptions"><span class="num">7</span>'
           "What we assumed &amp; what to watch</h2>"]
    if wl:
        li = "".join(f"<li>{esc(clip(x, 500))}</li>" for x in wl[:10])
        out.append('<div class="warnbox"><h4>Worth knowing before you act</h4>'
                   f"<ul>{li}</ul></div>")
    if al:
        li = "".join(f"<li>{esc(clip(x, 500))}</li>" for x in al[:20])
        out.append(f'<div class="card"><div class="eyebrow" style="margin-bottom:10px">'
                   f"Assumptions behind these numbers</div><ul>{li}</ul></div>")
    if not al and not wl:
        out.append('<div class="card"><p class="sub" style="margin:0">'
                   "No assumptions or caveats were recorded for this run.</p></div>")
    return "".join(out)


def _appendix(r: RunResult) -> str:
    out = ['<h2 class="section" id="appendix"><span class="num">8</span>'
           "Appendix: how this was computed</h2>"]
    al = lines(r.assumptions)
    if al:
        li = "".join(f"<li>{esc(clip(x, 500))}</li>" for x in al[:20])
        out.append('<div class="card tight"><div class="eyebrow" style="margin-bottom:9px">'
                   f"Assumptions, restated</div><ul style=\"font-size:13.5px;color:#cbd5e1\">{li}</ul></div>")
    events = list(r.log or [])
    if events:
        rows = []
        for e in events[:400]:
            t = getattr(e, "t", 0.0)
            try:
                ts = f"{float(t):.2f}s"
            except (TypeError, ValueError):
                ts = "0.00s"
            lvl = text(getattr(e, "level", "info"), "info").lower()
            det = text(getattr(e, "detail", ""))
            rows.append(
                f'<div class="row {esc(lvl)}">'
                f'<span class="t">{esc(ts)}</span>'
                f'<span class="ph">{esc(clip(getattr(e, "phase", "run"), 18))}</span>'
                f'<span class="msg">{esc(clip(getattr(e, "message", ""), 400))}</span>'
                + (f'<span class="det">{esc(clip(det, 400))}</span>' if det else "")
                + "</div>"
            )
        out.append(
            '<div class="card"><div class="eyebrow" style="margin-bottom:10px">'
            "Step-by-step run log</div>"
            f'<div class="log">{"".join(rows)}</div>'
            '<p class="note">Every line above was produced while the model ran. '
            "Re-running with the same seed replays it exactly.</p></div>"
        )
    else:
        out.append('<div class="card"><p class="sub" style="margin:0">'
                   "No run log was recorded.</p></div>")
    return "".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render(result: RunResult, path: str) -> str:
    """Write one self-contained HTML file and return the path."""
    r = result if result is not None else RunResult(run_id="unknown", use_case_id="unknown")
    title = text(r.use_case_title) or text(r.use_case_id).replace("_", " ").title() or "Simulation report"

    sections = [("method", "What we did"), ("kpis", "Numbers"), ("story", "How we got there"),
                ("charts", "Charts"), ("tables", "Tables"),
                ("recommendation", "Recommendation"), ("assumptions", "Assumptions"),
                ("appendix", "Appendix")]
    sections = [(a, b) for a, b in sections if _has_section(r, a)]

    parts = [
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(clip(title, 90))}</title>",
        # inline brand mark, same geometry as the app favicon (keeps the report offline)
        f'<link rel="icon" href="data:image/svg+xml,{quote(FAVICON_SVG)}">',
        f"<style>{CSS}</style>",
        "</head><body><div class=\"wrap\">",
        _nav(sections),
        _header(r),
        _what_we_did(r),
        _kpis(r),
        _story(r),
        _charts(r),
        _tables(r),
        _recommendation(r),
        _assumptions(r),
        _appendix(r),
        "<footer>Generated by Simulation Arena &middot; "
        f"seed {esc(r.seed)} &middot; run {esc(clip(r.run_id, 30))} &middot; {esc(stamp())}"
        "</footer>",
        "</div></body></html>",
    ]
    html = "".join(parts)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


def _has_section(r: RunResult, anchor: str) -> bool:
    if anchor == "method":
        return True
    if anchor == "kpis":
        return bool(r.kpis)
    if anchor == "story":
        return bool(r.story)
    if anchor == "charts":
        return bool(r.charts)
    if anchor == "tables":
        return bool(r.tables)
    if anchor == "recommendation":
        return r.recommendation is not None
    if anchor == "assumptions":
        return bool(lines(r.assumptions) or lines(r.warnings))
    return True
