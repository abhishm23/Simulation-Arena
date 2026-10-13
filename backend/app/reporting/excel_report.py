"""Excel workbook export for a RunResult (xlsxwriter).

Sheets: Summary, Recommendation, one per result table, Assumptions, Run log,
plus a Charts sheet with native Excel charts where the chart type maps
cleanly (line / bar / histogram). Anything that cannot be expressed natively
is skipped silently - the workbook is never the reason a run fails.
"""
from __future__ import annotations

import datetime as _dt
import io
import math
import re
from typing import Any, Optional, Sequence

from ..core.contracts import RunResult

MAX_CELL = 900
_INVALID = re.compile(r"[\[\]:*?/\\]")

TONE_FILL = {
    "good": "#DCFCE7",
    "bad": "#FEE2E2",
    "warn": "#FEF3C7",
    "warning": "#FEF3C7",
    "neutral": "#EFF6FF",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def text(v: Any, default: str = "") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def clip(v: Any, n: int = MAX_CELL) -> str:
    s = text(v)
    return s if len(s) <= n else s[: max(1, n - 3)] + "..."


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


def sheet_name(raw: Any, used: set[str]) -> str:
    base = _INVALID.sub(" ", text(raw, "Table")).strip() or "Table"
    base = re.sub(r"\s+", " ", base)[:31].strip()
    if not base:
        base = "Table"
    name, i = base, 2
    while name.lower() in used:
        suffix = f" {i}"
        name = base[: 31 - len(suffix)].rstrip() + suffix
        i += 1
    used.add(name.lower())
    return name


def is_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return v == v and v not in (float("inf"), float("-inf"))
    return False


def looks_money(v: Any) -> bool:
    s = text(v).lower()
    if not s:
        return False
    if s.startswith("$"):
        return True
    return any(w in s for w in ("$", "profit", "revenue", "cost", "price", "value",
                                "margin", "spend", "sales", "amount", "arr"))


def looks_pct(v: Any) -> bool:
    s = text(v)
    return s.endswith("%") or "%" in s


def stamp() -> str:
    try:
        return _dt.datetime.now().strftime("%d %b %Y %H:%M")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def render(result: RunResult, path: str) -> str:
    import xlsxwriter

    r = result if result is not None else RunResult(run_id="unknown", use_case_id="unknown")
    used: set[str] = set()
    wb = xlsxwriter.Workbook(path, {"constant_memory": False, "nan_inf_to_errors": True})

    f_title = wb.add_format({"bold": True, "font_size": 18, "font_color": "#0F172A"})
    f_head = wb.add_format({"bold": True, "font_size": 13, "font_color": "#0F172A",
                            "bottom": 1, "border_color": "#CBD5E1"})
    f_label = wb.add_format({"bold": True, "font_color": "#334155", "valign": "top"})
    f_wrap = wb.add_format({"text_wrap": True, "valign": "top", "font_color": "#0F172A"})
    f_wrap_wide = wb.add_format({"text_wrap": True, "valign": "top"})
    f_muted = wb.add_format({"font_color": "#64748B", "italic": True})
    f_hdr = wb.add_format({"bold": True, "bg_color": "#0F172A", "font_color": "#FFFFFF",
                           "border": 1, "border_color": "#334155", "text_wrap": True,
                           "valign": "vcenter"})
    f_cell = wb.add_format({"border": 1, "border_color": "#E2E8F0", "valign": "top"})
    f_cell_wrap = wb.add_format({"border": 1, "border_color": "#E2E8F0", "valign": "top",
                                 "text_wrap": True})
    f_hl = wb.add_format({"border": 1, "border_color": "#E2E8F0", "valign": "top",
                          "bg_color": "#DCFCE7", "bold": True})
    f_hl_wrap = wb.add_format({"border": 1, "border_color": "#E2E8F0", "valign": "top",
                               "bg_color": "#DCFCE7", "text_wrap": True, "bold": True})
    f_money = wb.add_format({"border": 1, "border_color": "#E2E8F0", "num_format": "$#,##0"})
    f_money2 = wb.add_format({"border": 1, "border_color": "#E2E8F0",
                              "num_format": "$#,##0.00"})
    f_pct = wb.add_format({"border": 1, "border_color": "#E2E8F0", "num_format": "0.0%"})
    f_num = wb.add_format({"border": 1, "border_color": "#E2E8F0", "num_format": "#,##0.00"})
    f_int = wb.add_format({"border": 1, "border_color": "#E2E8F0", "num_format": "#,##0"})
    f_hl_money = wb.add_format({"border": 1, "border_color": "#E2E8F0", "bg_color": "#DCFCE7",
                                "num_format": "$#,##0", "bold": True})
    f_hl_num = wb.add_format({"border": 1, "border_color": "#E2E8F0", "bg_color": "#DCFCE7",
                              "num_format": "#,##0.00", "bold": True})
    f_time = wb.add_format({"border": 1, "border_color": "#E2E8F0", "font_color": "#475569"})
    f_mono = wb.add_format({"border": 1, "border_color": "#E2E8F0", "valign": "top",
                            "text_wrap": True})

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    ws = wb.add_worksheet("Summary")
    used.add("summary")
    ws.set_column("A:A", 30)
    ws.set_column("B:B", 26)
    ws.set_column("C:C", 62)
    ws.set_column("D:D", 26)
    ws.hide_gridlines(2)

    title = text(r.use_case_title) or text(r.use_case_id).replace("_", " ").title() or "Simulation report"
    row = 0

    # brand mark in the top-left corner (skipped silently if it cannot be drawn)
    try:
        from .brand import png as _brand_png
        ws.insert_image(row, 0, "brand_mark.png",
                        {"image_data": io.BytesIO(_brand_png(128)),
                         "x_offset": 0, "y_offset": 0,
                         "x_scale": 0.22, "y_scale": 0.22})
        ws.write(row, 1, "Simulation Arena", wb.add_format(
            {"bold": True, "font_size": 11, "font_color": "#22C55E", "valign": "vcenter"}))
        ws.set_row(row, 30)
        row += 1
    except Exception:
        pass

    ws.write(row, 0, clip(title, 120), f_title)
    row += 1
    ws.write(row, 0, clip(r.headline, 400) or "No headline was produced.", f_wrap)
    ws.set_row(row, 34)
    row += 2

    ws.write(row, 0, "What we did", f_head)
    row += 1
    ws.merge_range(row, 0, row, 3, clip(r.method_plain, MAX_CELL)
                   or "No method description recorded.", f_wrap_wide)
    ws.set_row(row, 60)
    row += 2

    facts = [("Seed", r.seed), ("Run id", clip(r.run_id, 40)),
             ("Generated", stamp()),
             ("Scenarios tested", f"{int(r.n_simulations):,}" if r.n_simulations else "-"),
             ("Method", clip(r.model_used, 80) or "-"),
             ("Reproducible", "Same seed gives the same numbers")]
    for k, v in facts:
        ws.write(row, 0, k, f_label)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            ws.write_number(row, 1, float(v), f_int)
        else:
            ws.write(row, 1, v, f_wrap)
        row += 1
    row += 1

    # KPI block
    if r.kpis:
        ws.write(row, 0, "The numbers that matter", f_head)
        row += 1
        for c, h in enumerate(["Number", "Value", "What it means", "Compared with"]):
            ws.write(row, c, h, f_hdr)
        ws.set_row(row, 22)
        row += 1
        for k in r.kpis[:20]:
            fill = TONE_FILL.get(text(getattr(k, "tone", "neutral")).lower(), "#EFF6FF")
            fv = wb.add_format({"border": 1, "border_color": "#E2E8F0", "bg_color": fill,
                                "bold": True, "valign": "top"})
            fc = wb.add_format({"border": 1, "border_color": "#E2E8F0", "bg_color": fill,
                                "valign": "top", "text_wrap": True})
            ws.write(row, 0, clip(k.label, 120), fc)
            ws.write(row, 1, clip(k.value, 60), fv)
            ws.write(row, 2, clip(getattr(k, "plain", ""), MAX_CELL), fc)
            ws.write(row, 3, clip(getattr(k, "comparison", "") or "-", 200), fc)
            row += 1
    ws.freeze_panes(1, 0)

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    rec = r.recommendation
    if rec is not None:
        rs = wb.add_worksheet("Recommendation")
        used.add("recommendation")
        rs.set_column("A:A", 22)
        rs.set_column("B:B", 96)
        rs.hide_gridlines(2)
        rr = 0
        rs.write(rr, 0, "Recommendation", f_title)
        rr += 1
        rs.write(rr, 0, clip(rec.headline, MAX_CELL), f_wrap_wide)
        rs.merge_range(rr, 0, rr, 1, clip(rec.headline, MAX_CELL), f_wrap_wide)
        rs.set_row(rr, 34)
        rr += 2
        try:
            score = float(rec.risk_score) if rec.risk_score is not None else None
        except (TypeError, ValueError):
            score = None
        meta = [("Risk", text(rec.risk_label, "Not scored")),
                ("Risk score", f"{score:.0f} out of 100" if score is not None else "-"),
                ("Confidence", text(rec.confidence, "Medium"))]
        for k, v in meta:
            rs.write(rr, 0, k, f_label)
            rs.write(rr, 1, v, f_wrap)
            rr += 1
        rr += 1
        if lines(rec.rationale):
            rs.write(rr, 0, "Why", f_head)
            rr += 1
            for i, x in enumerate(lines(rec.rationale)[:15], start=1):
                rs.write(rr, 0, f"{i}.", f_label)
                rs.write(rr, 1, clip(x, MAX_CELL), f_wrap)
                rr += 1
            rr += 1
        if lines(rec.actions):
            rs.write(rr, 0, "What to do", f_head)
            rr += 1
            for i, x in enumerate(lines(rec.actions)[:15], start=1):
                rs.write(rr, 0, f"Action {i}", f_label)
                rs.write(rr, 1, clip(x, MAX_CELL), f_wrap)
                rr += 1
            rr += 1
        if text(rec.tradeoff):
            rs.write(rr, 0, "Trade-off", f_head)
            rr += 1
            rs.write(rr, 0, "", f_label)
            rs.write(rr, 1, clip(rec.tradeoff, MAX_CELL), f_wrap)

    # ------------------------------------------------------------------
    # Charts (native Excel charts where the type maps cleanly)
    # ------------------------------------------------------------------
    _add_charts(wb, r, used, f_hdr, f_cell)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    for t in (r.tables or [])[:20]:
        name = sheet_name(t.name, used)
        ts = wb.add_worksheet(name)
        ts.hide_gridlines(2)
        cols = [text(c, f"Column {i + 1}") for i, c in enumerate(t.columns or [])]
        rows = list(t.rows or [])
        ncols = max([len(cols)] + [len(rw) for rw in rows]) if rows else len(cols)
        ncols = max(ncols, 1)
        cols = (cols + [f"Column {i + 1}" for i in range(len(cols), ncols)])[:ncols]

        tr = 0
        if text(t.name):
            ts.write(tr, 0, clip(t.name, 120), f_head)
            tr += 1
        if text(getattr(t, "description", "")):
            ts.write(tr, 0, clip(t.description, MAX_CELL), f_muted)
            tr += 1
        if not rows:
            ts.write(tr, 0, "No rows were produced for this table.", f_muted)
            ts.set_column("A:A", 40)
            continue
        tr += 1
        head_row = tr
        for c, h in enumerate(cols):
            ts.write(head_row, c, clip(h, 80), f_hdr)
        ts.set_row(head_row, 24)

        hl = set()
        for i in (t.highlight_rows or []):
            try:
                hl.add(int(i))
            except (TypeError, ValueError):
                continue

        # decide per-column formats
        money_col = [looks_money(c) for c in cols]
        pct_col = [looks_pct(c) for c in cols]
        for c in range(ncols):
            sample = [rw[c] for rw in rows[:60] if c < len(rw)]
            if not money_col[c]:
                money_col[c] = sum(1 for v in sample if looks_money(v)) >= max(1, len(sample) // 2)
            if not pct_col[c]:
                pct_col[c] = sum(1 for v in sample if looks_pct(v)) >= max(1, len(sample) // 2)

        widths = [min(46, max(11, len(clip(c, 46)) + 3)) for c in cols]
        n_rows = len(rows)
        for ri, rw in enumerate(rows[:2000]):
            xr = head_row + 1 + ri
            highlighted = ri in hl
            for c in range(ncols):
                v = rw[c] if c < len(rw) else None
                if is_number(v):
                    if money_col[c]:
                        fmt = f_hl_money if highlighted else (
                            f_money2 if abs(float(v)) < 1000 else f_money)
                    elif pct_col[c]:
                        fmt = f_hl_num if highlighted else f_pct
                    elif float(v) == int(float(v)) and abs(float(v)) < 1e12:
                        fmt = f_hl_num if highlighted else f_int
                    else:
                        fmt = f_hl_num if highlighted else f_num
                    ts.write_number(xr, c, float(v), fmt)
                else:
                    s = clip(v, MAX_CELL) if v is not None else "-"
                    fmt = (f_hl_wrap if highlighted else f_cell_wrap) if len(s) > 40 else (
                        f_hl if highlighted else f_cell)
                    ts.write_string(xr, c, s if s else "-", fmt)
                    widths[c] = min(46, max(widths[c], min(len(s) + 2, 46)))
        for c, w in enumerate(widths):
            ts.set_column(c, c, w)
        last = head_row + min(n_rows, 2000)
        if n_rows > 0:
            try:
                ts.autofilter(head_row, 0, last, ncols - 1)
            except Exception:
                pass
        ts.freeze_panes(head_row + 1, 0)
        if text(getattr(t, "highlight_note", "")):
            ts.write(last + 2, 0, clip(t.highlight_note, MAX_CELL), f_muted)
        if n_rows > 2000:
            ts.write(last + 3, 0, f"Showing the first 2,000 of {n_rows:,} rows.", f_muted)

    # ------------------------------------------------------------------
    # Assumptions
    # ------------------------------------------------------------------
    al = lines(r.assumptions)
    wl = lines(r.warnings)
    if al or wl:
        a_s = wb.add_worksheet(sheet_name("Assumptions", used))
        a_s.set_column("A:A", 8)
        a_s.set_column("B:B", 110)
        a_s.hide_gridlines(2)
        ar = 0
        a_s.write(ar, 0, "Assumptions and caveats", f_head)
        ar += 2
        if wl:
            fw = wb.add_format({"text_wrap": True, "valign": "top", "bg_color": "#FEF3C7",
                                "font_color": "#78350F", "border": 1, "border_color": "#FDE68A"})
            a_s.write(ar, 0, "Watch", f_label)
            ar += 1
            for x in wl[:15]:
                a_s.write(ar, 0, "!", f_label)
                a_s.write(ar, 1, clip(x, MAX_CELL), fw)
                ar += 1
            ar += 1
        if al:
            a_s.write(ar, 0, "Assumed", f_label)
            ar += 1
            for i, x in enumerate(al[:30], start=1):
                a_s.write(ar, 0, i, f_label)
                a_s.write(ar, 1, clip(x, MAX_CELL), f_wrap)
                ar += 1

    # ------------------------------------------------------------------
    # Run log
    # ------------------------------------------------------------------
    events = list(r.log or [])
    if events:
        l_s = wb.add_worksheet(sheet_name("Run log", used))
        l_s.set_column("A:A", 12)
        l_s.set_column("B:B", 14)
        l_s.set_column("C:C", 90)
        l_s.set_column("D:D", 60)
        l_s.hide_gridlines(2)
        for c, h in enumerate(["Time", "Phase", "What happened", "Technical detail"]):
            l_s.write(0, c, h, f_hdr)
        l_s.set_row(0, 22)
        for i, e in enumerate(events[:2000], start=1):
            try:
                ts_ = f"{float(getattr(e, 't', 0.0)):.2f}s"
            except (TypeError, ValueError):
                ts_ = "0.00s"
            l_s.write(i, 0, ts_, f_time)
            l_s.write(i, 1, clip(getattr(e, "phase", "run"), 30), f_cell)
            l_s.write(i, 2, clip(getattr(e, "message", ""), MAX_CELL), f_mono)
            l_s.write(i, 3, clip(getattr(e, "detail", "") or "", MAX_CELL), f_mono)
        try:
            l_s.autofilter(0, 0, min(len(events), 2000), 3)
        except Exception:
            pass
        l_s.freeze_panes(1, 0)

    wb.close()
    return path


# ---------------------------------------------------------------------------
# Native Excel charts
# ---------------------------------------------------------------------------
def _add_charts(wb, r: RunResult, used: set[str], f_hdr, f_cell) -> None:
    specs = [s for s in (r.charts or [])]
    if not specs:
        return
    mappable = []
    for s in specs:
        ctype = text(getattr(s, "type", "")).lower()
        if ctype in ("line", "bar", "grouped_bar", "hist", "histogram", "stacked_area", "area"):
            mappable.append(s)
    if not mappable:
        return
    mappable = mappable[:3]

    try:
        data = wb.add_worksheet(sheet_name("Chart data", used))
        chartws = wb.add_worksheet(sheet_name("Charts", used))
    except Exception:
        return
    data.hide_gridlines(2)
    chartws.hide_gridlines(2)
    chartws.set_column("A:A", 2)
    chartws.write(0, 1, "Charts", wb.add_format({"bold": True, "font_size": 16}))

    cursor = 0          # column cursor on the data sheet
    row_cursor = 2      # where to drop the next chart image
    for spec in mappable:
        ctype = text(getattr(spec, "type", "")).lower()
        series = [s for s in (spec.series or []) if isinstance(s, dict)]
        if not series:
            continue
        try:
            if ctype in ("hist", "histogram"):
                vals = []
                for v in (series[0].get("values") or []):
                    try:
                        f = float(v)
                        if f == f and f not in (float("inf"), float("-inf")):
                            vals.append(f)
                    except (TypeError, ValueError):
                        continue
                if len(vals) < 2:
                    continue
                lo, hi = min(vals), max(vals)
                if hi <= lo:
                    hi = lo + 1.0
                nb = int(min(30, max(6, math.sqrt(len(vals)))))
                w = (hi - lo) / nb
                counts = [0] * nb
                for v in vals:
                    i = min(int((v - lo) / w), nb - 1)
                    counts[i] += 1
                cats = [f"{lo + w * (i + 0.5):,.2f}" for i in range(nb)]
                series_data = [{"name": text(series[0].get("name"), "Outcomes"),
                                "values": counts}]
            else:
                cats = [text(v, str(i + 1)) for i, v in enumerate(spec.x or [])]
                series_data = []
                for s in series[:6]:
                    vals = []
                    for v in (s.get("values") or []):
                        try:
                            f = float(v)
                        except (TypeError, ValueError):
                            f = None
                        vals.append(f)
                    series_data.append({"name": text(s.get("name"), "Series"),
                                        "values": vals})
                if not cats:
                    n = max([len(s["values"]) for s in series_data] + [1])
                    cats = [str(i + 1) for i in range(n)]
            if not series_data:
                continue

            n = max([len(s["values"]) for s in series_data] + [len(cats)])
            base_col = cursor
            # column 0 -> categories, then one column per series
            data.write(0, base_col, text(spec.title, "Chart"), f_hdr)
            for i in range(n):
                data.write(i + 1, base_col, cats[i] if i < len(cats) else str(i + 1), f_cell)
            for k, s in enumerate(series_data):
                c = base_col + 1 + k
                data.write(0, c, text(s["name"], f"Series {k + 1}"), f_hdr)
                for i in range(n):
                    v = s["values"][i] if i < len(s["values"]) else None
                    if v is None or v != v or v in (float("inf"), float("-inf")):
                        data.write_blank(i + 1, c, None, f_cell)
                    else:
                        data.write_number(i + 1, c, float(v), f_cell)
            data.set_column(base_col, base_col + len(series_data), 16)

            if ctype == "line":
                ch = wb.add_chart({"type": "line"})
            elif ctype in ("stacked_area", "area"):
                ch = wb.add_chart({"type": "area", "subtype": "stacked"})
            else:
                ch = wb.add_chart({"type": "column"})

            dq = "'Chart data'"
            for k, s in enumerate(series_data):
                c = base_col + 1 + k
                col = _col_letter(c)
                ch.add_series({
                    "name": [dq, 0, c],
                    "categories": [dq, 1, base_col, n, base_col],
                    "values": [dq, 1, c, n, c],
                })
            ch.set_title({"name": text(spec.title, "Chart")[:200]})
            ch.set_legend({"position": "bottom"})
            ch.set_size({"width": 720, "height": 380})
            ch.set_chartarea({"border": {"none": True}})
            try:
                if text(spec.x_label):
                    ch.set_x_axis({"name": text(spec.x_label)[:80]})
                if text(spec.y_label):
                    ch.set_y_axis({"name": text(spec.y_label)[:80]})
            except Exception:
                pass
            chartws.write(row_cursor, 1, text(spec.title, "Chart"))
            if text(spec.plain_meaning):
                chartws.write(row_cursor + 1, 1, clip(spec.plain_meaning, 600),
                              wb.add_format({"italic": True, "font_color": "#475569",
                                             "text_wrap": True}))
            chartws.insert_chart(row_cursor + 2, 1, ch)
            row_cursor += 22
            cursor += len(series_data) + 2
            if cursor > 40:
                break
        except Exception:
            continue


def _col_letter(i: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA (used only for building chart ranges)."""
    s = ""
    i += 1
    while i:
        i, rem = divmod(i - 1, 26)
        s = chr(65 + rem) + s
    return s
