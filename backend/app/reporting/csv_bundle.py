"""CSV bundle export: one .zip holding the whole run as flat files.

    kpis.csv, assumptions.csv, run_log.csv, one CSV per result table
    (slugified name) and summary.txt (headline + recommendation).

Stdlib only (zipfile + csv). UTF-8 with a BOM so Excel opens the files
cleanly on a Windows machine.
"""
from __future__ import annotations

import csv
import datetime as _dt
import io
import re
import zipfile
from typing import Any, Sequence

from ..core.contracts import RunResult


def text(v: Any, default: str = "") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def lines(items: Any) -> list[str]:
    if not items:
        return []
    if isinstance(items, str):
        items = [items]
    try:
        seq = list(items)
    except TypeError:
        return []
    return [text(v) for v in seq if text(v)]


def slug(s: Any, fallback: str = "table") -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text(s)).strip("_").lower()
    return s or fallback


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    w.writerow([text(h) for h in header])
    for r in rows:
        w.writerow(["" if v is None else text(v) for v in r])
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def render(result: RunResult, path: str) -> str:
    r = result if result is not None else RunResult(run_id="unknown", use_case_id="unknown")
    title = text(r.use_case_title) or text(r.use_case_id).replace("_", " ").title() or "Simulation report"

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # -- KPIs -----------------------------------------------------------
        kpi_rows = [[k.label, k.value, text(getattr(k, "plain", "")),
                     text(getattr(k, "comparison", "") or ""),
                     text(getattr(k, "tone", "neutral"))] for k in (r.kpis or [])]
        z.writestr("kpis.csv", _csv_bytes(
            ["Measure", "Value", "What it means", "Compared with", "Signal"], kpi_rows))

        # -- assumptions + warnings -----------------------------------------
        a_rows = [["Assumption", x] for x in lines(r.assumptions)]
        a_rows += [["Caveat", x] for x in lines(r.warnings)]
        z.writestr("assumptions.csv", _csv_bytes(["Type", "Detail"], a_rows))

        # -- run log ---------------------------------------------------------
        log_rows = []
        for e in (r.log or []):
            try:
                t = f"{float(getattr(e, 't', 0.0)):.3f}"
            except (TypeError, ValueError):
                t = "0.000"
            log_rows.append([t, text(getattr(e, "phase", "run")),
                             text(getattr(e, "level", "info")),
                             text(getattr(e, "message", "")),
                             text(getattr(e, "detail", "") or "")])
        z.writestr("run_log.csv", _csv_bytes(
            ["Seconds into the run", "Stage", "Type", "What happened", "Technical detail"],
            log_rows))

        # -- one CSV per table ----------------------------------------------
        names: set[str] = set()
        for i, t in enumerate(r.tables or []):
            base = slug(t.name, f"table_{i + 1}")
            name, k = base, 2
            while f"{name}.csv" in names:
                name = f"{base}_{k}"
                k += 1
            names.add(f"{name}.csv")
            cols = [text(c, f"Column {j + 1}") for j, c in enumerate(t.columns or [])]
            rows = list(t.rows or [])
            ncol = max([len(cols)] + [len(rw) for rw in rows]) if rows else len(cols)
            if rows:
                cols = (cols + [f"Column {j + 1}" for j in range(len(cols), ncol)])[:ncol]
                body = [(list(rw) + [""] * (ncol - len(rw)))[:ncol] for rw in rows]
            else:
                body = []
            z.writestr(f"{name}.csv", _csv_bytes(cols or ["Value"], body))

        # -- story -----------------------------------------------------------
        if r.story:
            z.writestr("story.csv", _csv_bytes(
                ["Step", "What happened", "Detail", "Evidence"],
                [[i, text(s.title), text(s.detail), text(getattr(s, "evidence", "") or "")]
                 for i, s in enumerate(r.story, start=1)]))

        # -- summary.txt ------------------------------------------------------
        z.writestr("summary.txt", _summary_text(r, title).encode("utf-8"))

    return path


def _summary_text(r: RunResult, title: str) -> str:
    out: list[str] = []
    out.append("=" * 78)
    out.append(title)
    out.append("=" * 78)
    out.append("")
    out.append(text(r.headline, "No headline was produced."))
    out.append("")
    out.append(f"Seed: {r.seed}   Run id: {r.run_id}   "
               f"Scenarios tested: {int(r.n_simulations):,}")
    try:
        out.append(f"Generated: {_dt.datetime.now().strftime('%d %b %Y %H:%M')}")
    except Exception:
        pass
    if text(r.model_used):
        out.append(f"Method: {r.model_used}")
    out.append("Same seed, same numbers - this run can be reproduced exactly.")
    out.append("")

    if text(r.method_plain):
        out.append("HOW THIS WORKS")
        out.append("-" * 78)
        out.append(_wrap(r.method_plain, 78))
        out.append("")

    if r.kpis:
        out.append("THE NUMBERS THAT MATTER")
        out.append("-" * 78)
        for k in r.kpis:
            out.append(f"* {k.label}: {k.value}")
            if text(getattr(k, "plain", "")):
                out.append(_wrap(f"    {k.plain}", 78, indent="    "))
            if text(getattr(k, "comparison", "") or ""):
                out.append(f"    Compared with: {k.comparison}")
        out.append("")

    if r.story:
        out.append("HOW WE GOT THERE")
        out.append("-" * 78)
        for i, s in enumerate(r.story, start=1):
            out.append(f"{i}. {s.title}")
            out.append(_wrap(s.detail, 78, indent="   "))
            if text(getattr(s, "evidence", "") or ""):
                out.append(f"   Evidence: {s.evidence}")
        out.append("")

    rec = r.recommendation
    if rec is not None:
        out.append("RECOMMENDATION")
        out.append("-" * 78)
        out.append(_wrap(text(rec.headline), 78))
        out.append("")
        if lines(rec.rationale):
            out.append("Why:")
            for x in lines(rec.rationale):
                out.append(_wrap(f"- {x}", 78, indent="  "))
        if lines(rec.actions):
            out.append("")
            out.append("What to do:")
            for i, x in enumerate(rec.actions, start=1):
                out.append(_wrap(f"{i}. {x}", 78, indent="   "))
        if rec.risk_score is not None or text(rec.risk_label) or text(rec.confidence):
            bits = []
            if text(rec.risk_label):
                bits.append(f"Risk: {rec.risk_label}")
            if rec.risk_score is not None:
                try:
                    bits.append(f"{float(rec.risk_score):.0f} out of 100")
                except (TypeError, ValueError):
                    pass
            if text(rec.confidence):
                bits.append(f"Confidence: {rec.confidence}")
            out.append("")
            out.append("  " + " | ".join(bits))
        if text(rec.tradeoff):
            out.append("")
            out.append("The trade-off:")
            out.append(_wrap(rec.tradeoff, 78, indent="  "))
        out.append("")

    if lines(r.warnings):
        out.append("WORTH KNOWING BEFORE YOU ACT")
        out.append("-" * 78)
        for x in lines(r.warnings):
            out.append(_wrap(f"- {x}", 78, indent="  "))
        out.append("")

    if lines(r.assumptions):
        out.append("ASSUMPTIONS")
        out.append("-" * 78)
        for x in lines(r.assumptions):
            out.append(_wrap(f"- {x}", 78, indent="  "))
        out.append("")

    if r.tables:
        out.append("TABLES IN THIS BUNDLE")
        out.append("-" * 78)
        for i, t in enumerate(r.tables):
            n = len(t.rows or [])
            out.append(f"- {slug(t.name, f'table_{i + 1}')}.csv : {t.name} ({n} rows)")
        out.append("")
    return "\n".join(out) + "\n"


def _wrap(s: Any, width: int = 78, indent: str = "") -> str:
    import textwrap

    s = text(s)
    if not s:
        return ""
    s = textwrap.fill(s, width=width, initial_indent=indent,
                      subsequent_indent=indent, break_long_words=True,
                      break_on_hyphens=False)
    return s
