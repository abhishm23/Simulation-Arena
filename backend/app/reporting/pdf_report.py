"""PDF export for a RunResult (reportlab platypus, A4).

Contents: title block, how this works, KPI card grid, the story, the first
three charts as images, every table truncated to 25 rows, the recommendation,
the assumptions, and a short appendix with the tail of the run log.

This module must never raise: a malformed or empty result produces a short,
honest PDF rather than losing the user's run.
"""
from __future__ import annotations

import datetime as _dt
import io
from html import escape as _xml_escape
from typing import Any, Optional, Sequence

from ..core.contracts import RunResult

MAX_ROWS = 25
MAX_COLS = 12
LOG_TAIL = 25

INK = "#0F172A"
BODY = "#1F2937"
MUTED = "#64748B"
ACCENT = "#0F172A"
RULE = "#CBD5E1"
TONE = {"good": "#15803D", "bad": "#B91C1C", "warn": "#B45309",
        "warning": "#B45309", "neutral": "#1D4ED8"}


def text(v: Any, default: str = "") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def clip(v: Any, n: int) -> str:
    s = text(v)
    return s if len(s) <= n else s[: max(1, n - 1)].rstrip() + "\u2026"


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


def cell_text(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return "-"
        if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
            return f"{int(round(v)):,}"
        return f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    s = str(v).strip()
    return s if s else "-"


def _p(s: Any) -> str:
    """Escape for reportlab's mini-HTML, and break pathologically long words."""
    s = clip(s, 4000)
    words = []
    for w in s.split(" "):
        if len(w) > 48:
            w = " ".join(w[i:i + 44] for i in range(0, len(w), 44))
        words.append(w)
    return _xml_escape(" ".join(words), quote=False)


def _png_size(data: bytes) -> tuple[int, int]:
    if len(data) > 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        try:
            return (int.from_bytes(data[16:20], "big"),
                    int.from_bytes(data[20:24], "big"))
        except Exception:
            return (900, 420)
    return (900, 420)


def _stamp() -> str:
    try:
        return _dt.datetime.now().strftime("%d %b %Y, %H:%M")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def render(result: RunResult, path: str) -> str:
    r = result if result is not None else RunResult(run_id="unknown", use_case_id="unknown")
    try:
        _build(r, path)
    except Exception:
        try:
            _build(r, path, lite=True)     # drop the heavy blocks and try again
        except Exception:
            try:
                _fallback(r, path)
            except Exception:
                pass
    return path


def _build(r: RunResult, path: str, lite: bool = False) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageTemplate,
                                    Paragraph, Spacer, Table, TableStyle)

    dark = colors.HexColor(ACCENT)
    rule = colors.HexColor(RULE)
    muted = colors.HexColor(MUTED)

    doc = BaseDocTemplate(path, pagesize=A4,
                          leftMargin=16 * mm, rightMargin=16 * mm,
                          topMargin=16 * mm, bottomMargin=18 * mm,
                          title=clip(text(r.use_case_title) or "Simulation report", 90),
                          author="Simulation Arena")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=_make_decorate(r))])

    W = doc.width
    title = text(r.use_case_title) or text(r.use_case_id).replace("_", " ").title() or "Simulation report"

    s_title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=19, leading=23,
                             textColor=colors.white, spaceAfter=4)
    s_head = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=13, leading=16,
                            textColor=colors.HexColor("#F1F5F9"), spaceAfter=3)
    s_h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.5, leading=15,
                          textColor=dark, spaceBefore=12, spaceAfter=5)
    s_body = ParagraphStyle("b", fontName="Helvetica", fontSize=9.6, leading=13.4,
                            textColor=colors.HexColor(BODY), alignment=TA_LEFT,
                            spaceAfter=5)
    s_small = ParagraphStyle("s", parent=s_body, fontSize=8.4, leading=11.4,
                             textColor=muted)
    s_kpi_v = ParagraphStyle("kv", fontName="Helvetica-Bold", fontSize=13.5, leading=15,
                             textColor=dark, spaceAfter=1)
    s_kpi_l = ParagraphStyle("kl", fontName="Helvetica-Bold", fontSize=7, leading=8.6,
                             textColor=muted, spaceAfter=2)
    s_kpi_p = ParagraphStyle("kp", fontName="Helvetica", fontSize=7.8, leading=10,
                             textColor=colors.HexColor(BODY))
    s_cell = ParagraphStyle("c", fontName="Helvetica", fontSize=7.6, leading=9.4,
                            textColor=colors.HexColor(BODY))
    s_cell_h = ParagraphStyle("ch", fontName="Helvetica-Bold", fontSize=7.6, leading=9.4,
                              textColor=colors.white)
    s_cap = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8.4, leading=11,
                           textColor=muted, spaceBefore=3, spaceAfter=8)
    s_mono = ParagraphStyle("m", fontName="Courier", fontSize=7.4, leading=9.6,
                            textColor=colors.HexColor(BODY))

    st: list[Any] = []

    # -- brand line ---------------------------------------------------------
    try:
        from .brand import png as _brand_png
        from reportlab.platypus import Image as _RlImage
        _icon = _RlImage(io.BytesIO(_brand_png(120)), width=22, height=22)
        _brand = Table(
            [[_icon, Paragraph(
                '<font color="#e2e8f0"><b>Simulation</b></font>'
                '<font color="#22c55e"><b> Arena</b></font>',
                ParagraphStyle("brand", fontName="Helvetica-Bold", fontSize=11.5,
                               leading=14, textColor=colors.HexColor(BODY)))]],
            colWidths=[27, W - 27])
        _brand.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        st.append(_brand)
    except Exception:
        pass  # the report must never fail because of a logo

    # -- title block --------------------------------------------------------
    banner = Table(
        [[Paragraph(_p(title), s_title)],
         [Paragraph(_p(text(r.headline, "Here is what the simulation found.")), s_head)]],
        colWidths=[W])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), dark),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEBELOW", (0, 0), (-1, 0), 0, colors.white),
    ]))
    st.append(banner)

    facts = [f"Seed {r.seed}", f"Run {clip(r.run_id, 18)}"]
    if r.n_simulations:
        facts.append(f"{int(r.n_simulations):,} scenarios tested")
    if text(r.model_used):
        facts.append(clip(r.model_used, 40))
    facts.append(_stamp())
    meta = Table([[Paragraph(_p("  |  ".join(facts)), s_small)]], colWidths=[W])
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.6, rule),
    ]))
    st.append(meta)
    st.append(Paragraph("Same seed, same numbers - this run can be reproduced exactly.",
                        s_small))
    st.append(Spacer(1, 6))

    # -- method -------------------------------------------------------------
    if text(r.method_plain):
        st.append(Paragraph("How this works", s_h2))
        st.append(Paragraph(_p(r.method_plain), s_body))

    # -- KPI grid -----------------------------------------------------------
    if r.kpis:
        st.append(Paragraph("The numbers that matter", s_h2))
        cards = []
        for k in r.kpis[:12]:
            c = colors.HexColor(TONE.get(text(getattr(k, "tone", "neutral")).lower(),
                                         TONE["neutral"]))
            unit = f" {k.unit}" if text(getattr(k, "unit", "")) else ""
            inner = [Paragraph(_p(clip(k.label, 90)).upper(), s_kpi_l),
                     Paragraph(f'<font color="#{c.hexval()[2:]}">'
                               f"{_p(clip(k.value, 34))}</font>{_p(unit)}", s_kpi_v)]
            if text(getattr(k, "plain", "")):
                inner.append(Paragraph(_p(clip(k.plain, 260)), s_kpi_p))
            if text(getattr(k, "comparison", "") or ""):
                inner.append(Paragraph(_p("vs " + clip(k.comparison, 90)), s_kpi_l))
            cards.append(inner)
        ncol = 3 if len(cards) >= 3 else max(1, len(cards))
        cw = W / ncol
        rows = [cards[i:i + ncol] for i in range(0, len(cards), ncol)]
        tbl_data = []
        style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("LEFTPADDING", (0, 0), (-1, -1), 7),
                 ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                 ("TOPPADDING", (0, 0), (-1, -1), 6),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                 ("BOX", (0, 0), (-1, -1), 0.5, rule),
                 ("INNERGRID", (0, 0), (-1, -1), 0.5, rule)]
        for ri, rw in enumerate(rows):
            tbl_data.append(rw + [""] * (ncol - len(rw)))
            for ci in range(ncol):
                k = ri * ncol + ci
                if k < len(r.kpis[:12]):
                    c = colors.HexColor(TONE.get(
                        text(getattr(r.kpis[k], "tone", "neutral")).lower(), TONE["neutral"]))
                    style.append(("LINEBEFORE", (ci, ri), (ci, ri), 2.6, c))
        kt = Table(tbl_data, colWidths=[cw] * ncol)
        kt.setStyle(TableStyle(style))
        st.append(kt)

    # -- story --------------------------------------------------------------
    if r.story:
        st.append(Paragraph("How we got there", s_h2))
        for i, s in enumerate(r.story[:8], start=1):
            st.append(Paragraph(f"<b>{i}. {_p(clip(s.title, 110))}</b>", s_body))
            st.append(Paragraph(_p(clip(s.detail, 1200)), s_body))
            if text(getattr(s, "evidence", "") or ""):
                st.append(Paragraph(_p("Evidence: " + clip(s.evidence, 240)), s_small))

    # -- charts -------------------------------------------------------------
    charts = list(r.charts or [])[:3]
    if charts:
        st.append(Paragraph("Charts", s_h2))
        from .charts import render_chart_png
        for spec in charts:
            try:
                st.append(Paragraph(f"<b>{_p(clip(spec.title, 120))}</b>", s_body))
                if text(getattr(spec, "subtitle", "")):
                    st.append(Paragraph(_p(clip(spec.subtitle, 220)), s_small))
                png = render_chart_png(spec)
                if png:
                    iw, ih = _png_size(png)
                    w = W
                    h = w * (ih / max(iw, 1))
                    max_h = 118 * (72 / 25.4) * 1.0
                    if h > max_h:
                        h = max_h
                        w = h * (iw / max(ih, 1))
                    st.append(Image(io.BytesIO(png), width=w, height=h))
                if text(getattr(spec, "plain_meaning", "")):
                    st.append(Paragraph(_p("What this shows: " + clip(spec.plain_meaning, 500)),
                                        s_cap))
                else:
                    st.append(Spacer(1, 6))
            except Exception:
                st.append(Paragraph(f"<i>{_p(clip(spec.title, 100))}</i> - this chart could "
                                    "not be drawn.", s_small))

    # -- tables -------------------------------------------------------------
    if r.tables and lite:
        st.append(Paragraph("Tables", s_h2))
        st.append(Paragraph("The tables for this run are wide; they are in the Excel "
                            "workbook and the CSV bundle.", s_small))
    elif r.tables:
        st.append(Paragraph("Tables", s_h2))
        for t in r.tables[:8]:
            st.append(Paragraph(f"<b>{_p(clip(t.name, 110))}</b>", s_body))
            if text(getattr(t, "description", "")):
                st.append(Paragraph(_p(clip(t.description, 400)), s_small))
            rows = list(t.rows or [])
            cols = [text(c, f"Column {i + 1}") for i, c in enumerate(t.columns or [])]
            if not rows:
                st.append(Paragraph("No rows were produced for this table.", s_small))
                st.append(Spacer(1, 6))
                continue
            ncol = max(len(cols), max(len(rw) for rw in rows))
            truncated_cols = False
            if ncol > MAX_COLS:
                ncol = MAX_COLS
                truncated_cols = True
            cols = (cols + [f"Column {i + 1}" for i in range(len(cols), ncol)])[:ncol]
            hl = set()
            for i in (t.highlight_rows or []):
                try:
                    hl.add(int(i))
                except (TypeError, ValueError):
                    continue
            shown = rows[:MAX_ROWS]
            data = [[Paragraph(_p(clip(c, 60)), s_cell_h) for c in cols]]
            for ri, rw in enumerate(shown):
                rws = [(rw[i] if i < len(rw) else None) for i in range(ncol)]
                data.append([Paragraph(_p(clip(cell_text(v), 120)), s_cell) for v in rws])
            cw = [W / ncol] * ncol
            tt = Table(data, colWidths=cw, repeatRows=1)
            tstyle = [
                ("BACKGROUND", (0, 0), (-1, 0), dark),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, rule),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
            for ri in range(1, len(data)):
                if (ri - 1) in hl:
                    tstyle.append(("BACKGROUND", (0, ri), (-1, ri),
                                   colors.HexColor("#DCFCE7")))
                elif ri % 2 == 0:
                    tstyle.append(("BACKGROUND", (0, ri), (-1, ri),
                                   colors.HexColor("#F8FAFC")))
            tt.setStyle(TableStyle(tstyle))
            st.append(tt)
            notes = []
            if len(rows) > MAX_ROWS:
                notes.append(f"Showing the first {MAX_ROWS} of {len(rows):,} rows.")
            if truncated_cols:
                notes.append(f"Showing the first {MAX_COLS} of "
                             f"{max(len(cols), max(len(rw) for rw in rows))} columns.")
            if text(getattr(t, "highlight_note", "")):
                notes.append(clip(t.highlight_note, 300))
            if notes:
                st.append(Paragraph(_p(" ".join(notes)), s_small))
            st.append(Spacer(1, 8))

    # -- recommendation -----------------------------------------------------
    rec = r.recommendation
    if rec is not None:
        st.append(Paragraph("What we recommend", s_h2))
        st.append(Paragraph(f"<b>{_p(clip(rec.headline, 300))}</b>", s_body))
        try:
            score = float(rec.risk_score) if rec.risk_score is not None else None
        except (TypeError, ValueError):
            score = None
        if score is not None:
            score = max(0.0, min(100.0, score))
            bar_col = "#B91C1C" if score >= 70 else ("#B45309" if score >= 45 else "#15803D")
            bt = Table([[Paragraph(
                f'<font color="white"><b>Risk {_p(clip(text(rec.risk_label, "Not scored"), 20))}'
                f"</b> &nbsp; {score:.0f} out of 100 &nbsp;|&nbsp; Confidence: "
                f"{_p(clip(text(rec.confidence, 'Medium'), 16))}</font>", s_small)]],
                colWidths=[W])
            bt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bar_col)),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            st.append(Spacer(1, 3))
            st.append(bt)
        elif text(rec.confidence):
            st.append(Paragraph(_p(f"Confidence: {rec.confidence}"), s_small))
        st.append(Spacer(1, 4))
        if lines(rec.rationale):
            st.append(Paragraph("<b>Why</b>", s_body))
            for x in lines(rec.rationale)[:12]:
                st.append(Paragraph(f"&bull; {_p(clip(x, 400))}", s_body))
        if lines(rec.actions):
            st.append(Paragraph("<b>What to do</b>", s_body))
            for i, x in enumerate(lines(rec.actions)[:12], start=1):
                st.append(Paragraph(f"<b>{i}.</b> {_p(clip(x, 400))}", s_body))
        if text(rec.tradeoff):
            st.append(Paragraph(f"<b>The trade-off:</b> {_p(clip(rec.tradeoff, 500))}",
                                s_body))

    # -- assumptions + warnings ---------------------------------------------
    wl, al = lines(r.warnings), lines(r.assumptions)
    if wl or al:
        st.append(Paragraph("What we assumed and what to watch", s_h2))
        if wl:
            st.append(Paragraph("<b>Worth knowing before you act</b>", s_body))
            for x in wl[:10]:
                # one table per line: a single tall table cannot split across pages
                wt = Table([[Paragraph("&bull; " + _p(clip(x, 900)), s_body)]],
                           colWidths=[W])
                wt.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#F59E0B")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                st.append(wt)
                st.append(Spacer(1, 4))
            st.append(Spacer(1, 4))
        if al:
            st.append(Paragraph("<b>Assumptions</b>", s_body))
            for x in al[:20]:
                st.append(Paragraph(f"&bull; {_p(clip(x, 400))}", s_body))

    # -- appendix -----------------------------------------------------------
    events = list(r.log or [])
    if events:
        st.append(Paragraph("Appendix: the last steps of the run", s_h2))
        st.append(Paragraph("Every line was produced while the model ran. Re-running with "
                            "the same seed replays it exactly.", s_small))
        tail = events[-LOG_TAIL:]
        data = [[Paragraph("<b><font color='white'>Time</font></b>", s_cell),
                 Paragraph("<b><font color='white'>Stage</font></b>", s_cell),
                 Paragraph("<b><font color='white'>What happened</font></b>", s_cell)]]
        for e in tail:
            try:
                ts_ = f"{float(getattr(e, 't', 0.0)):.2f}s"
            except (TypeError, ValueError):
                ts_ = "0.00s"
            data.append([Paragraph(_p(ts_), s_mono),
                         Paragraph(_p(clip(getattr(e, "phase", "run"), 14)), s_mono),
                         Paragraph(_p(clip(getattr(e, "message", ""), 220)), s_mono)])
        lt = Table(data, colWidths=[W * 0.13, W * 0.16, W * 0.71], repeatRows=1)
        lt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), dark),
            ("GRID", (0, 0), (-1, -1), 0.35, rule),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        st.append(lt)

    doc.build(st)


def _make_decorate(r: RunResult):
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    def decorate(canvas, doc):
        try:
            canvas.saveState()
            w, h = doc.pagesize
            canvas.setStrokeColor(colors.HexColor("#0F172A"))
            canvas.setLineWidth(1.1)
            canvas.line(doc.leftMargin, h - 10 * mm, w - doc.rightMargin, h - 10 * mm)
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(colors.HexColor("#64748B"))
            canvas.drawString(doc.leftMargin, h - 9 * mm,
                              clip(text(r.use_case_title) or "Simulation Arena", 60))
            canvas.drawRightString(w - doc.rightMargin, h - 9 * mm,
                                   f"seed {r.seed}")
            canvas.setLineWidth(0.5)
            canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
            canvas.line(doc.leftMargin, 12 * mm, w - doc.rightMargin, 12 * mm)
            canvas.drawString(doc.leftMargin, 8 * mm, "Simulation Arena")
            canvas.drawRightString(w - doc.rightMargin, 8 * mm, f"Page {canvas.getPageNumber()}")
            canvas.restoreState()
        except Exception:
            pass

    return decorate


def _fallback(r: RunResult, path: str) -> None:
    """Last resort: a two-page-ish PDF that at least carries the headline."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm)
    s = ParagraphStyle("x", fontName="Helvetica", fontSize=11, leading=15,
                       textColor=colors.HexColor(BODY))
    h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=16, leading=20,
                       textColor=colors.HexColor(INK))
    story = [Paragraph(_p(text(r.use_case_title) or "Simulation report"), h), Spacer(1, 10),
             Paragraph(_p(text(r.headline, "The report could not be rendered in full, "
                                           "but the run completed.")), s), Spacer(1, 8),
             Paragraph(_p(f"Seed {r.seed}  |  Run {r.run_id}"), s)]
    doc.build(story)
