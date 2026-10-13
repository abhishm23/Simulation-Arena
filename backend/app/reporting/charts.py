"""Chart rendering for reports.

Two outputs from the same :class:`ChartSpec`:

    render_chart(spec)      -> inline ``<svg ...>...</svg>`` string  (HTML / offline)
    render_chart_png(spec)  -> PNG bytes                              (PDF)

The SVG is hand-rolled: no JavaScript, no CDN, no web fonts, nothing that
makes a network request. That keeps the HTML artifact fully self-contained
and byte-deterministic for a given RunResult.

Golden rule: this module NEVER raises. A broken or unknown chart degrades to
something readable (or to a labelled placeholder) because a crash in reporting
would throw away the user's whole simulation run.
"""
from __future__ import annotations

import math
from html import escape
from typing import Any, Optional, Sequence

from ..core.contracts import ChartSpec

# ---------------------------------------------------------------------------
# Palette (dark, professional)
# ---------------------------------------------------------------------------
BG = "#0f172a"
PANEL = "#131f36"
GRID = "#1e293b"
AXIS = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
SERIES = [
    "#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#a78bfa",
    "#38bdf8", "#f472b6", "#facc15", "#2dd4bf", "#fb923c",
]
FONT = ("ui-sans-serif, -apple-system, 'Segoe UI', Roboto, Helvetica, "
        "Arial, sans-serif")
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

_PCT_WORDS = ("%", "chance", "share", "rate", "probability", "percent",
              "likelihood", "odds", "confidence", "proportion")
_MONEY_WORDS = ("$", "profit", "revenue", "cost", "price", "value", "margin",
                "spend", "sales", "dollar", "arr", "amount", "savings",
                "income", "cash", "budget", "fee", "discount value")

# 1x1 transparent PNG, used only if matplotlib is entirely unavailable.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
    b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _esc(s: Any) -> str:
    if s is None:
        return ""
    return escape(str(s), quote=True)


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _opt_num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _clip(s: Any, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: max(1, n - 1)] + "\u2026"


def _axis_kind(label: str) -> str:
    low = (label or "").lower()
    if any(w in low for w in _PCT_WORDS):
        return "pct"
    if any(w in low for w in _MONEY_WORDS):
        return "money"
    return "num"


def _fmt_tick(v: float, kind: str, as_fraction_pct: bool = False) -> str:
    """Smart, short axis labels: '$1.2M', '45%', '1,200'."""
    a = abs(v)
    sign = "-" if v < 0 else ""
    if kind == "money":
        if a >= 1e9:
            return f"{sign}${a / 1e9:.1f}B"
        if a >= 1e6:
            return f"{sign}${a / 1e6:.1f}M"
        if a >= 1e3:
            return f"{sign}${a / 1e3:.0f}k"
        if a >= 100:
            return f"{sign}${a:,.0f}"
        return f"{sign}${a:,.1f}"
    if kind == "pct":
        x = v * 100.0 if as_fraction_pct else v
        ax = abs(x)
        if ax >= 100:
            return f"{x:.0f}%"
        if ax >= 10:
            return f"{x:.0f}%"
        if ax >= 1:
            return f"{x:.1f}%"
        return f"{x:.2f}%"
    if a >= 1e9:
        return f"{sign}{a / 1e9:.1f}B"
    if a >= 1e6:
        return f"{sign}{a / 1e6:.1f}M"
    if a >= 1e4:
        return f"{sign}{a / 1e3:.0f}k"
    if a >= 100:
        return f"{sign}{a:,.0f}"
    if a >= 1:
        return f"{sign}{a:,.1f}"
    if a == 0:
        return "0"
    return f"{sign}{a:.2f}"


def _nice_ticks(lo: float, hi: float, target: int = 5) -> tuple[list[float], float, float]:
    if not (hi > lo):
        hi = lo + 1.0 if lo == hi else lo
        if hi == lo:
            hi = lo + 1.0
    span = hi - lo
    raw = span / max(1, target)
    if raw <= 0:
        raw = 1.0
    mag = 10.0 ** math.floor(math.log10(raw))
    step = mag * 10
    for m in (1, 2, 2.5, 5, 10):
        if raw <= mag * m + 1e-12:
            step = mag * m
            break
    start = math.floor(lo / step) * step
    end = math.ceil(hi / step) * step
    ticks: list[float] = []
    v = start
    guard = 0
    while v <= end + step * 1e-6 and guard < 60:
        ticks.append(round(v, 10))
        v += step
        guard += 1
    return ticks, min(start, lo), max(end, hi)


def _lerp_color(stops: Sequence[str], t: float) -> str:
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    n = len(stops) - 1
    pos = t * n
    i = min(int(pos), n - 1)
    f = pos - i

    def rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    a, b = rgb(stops[i]), rgb(stops[i + 1])
    return "#%02x%02x%02x" % (
        int(round(a[0] + (b[0] - a[0]) * f)),
        int(round(a[1] + (b[1] - a[1]) * f)),
        int(round(a[2] + (b[2] - a[2]) * f)),
    )


HEAT_STOPS = ["#0d1b33", "#1d4ed8", "#3b82f6", "#f59e0b", "#ef4444"]


# ---------------------------------------------------------------------------
# Tiny SVG builder
# ---------------------------------------------------------------------------
class _Box:
    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, x0: float, y0: float, x1: float, y1: float):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0


class _Svg:
    def __init__(self, w: int, h: int, pad: bool = True):
        self.w, self.h = w, h
        self.p: list[str] = []
        if pad:
            self.p.append(
                f'<rect x="0" y="0" width="{w}" height="{h}" rx="12" fill="{BG}"/>'
            )

    # -- primitives ---------------------------------------------------------
    def raw(self, s: str) -> None:
        self.p.append(s)

    def rect(self, x: float, y: float, w: float, h: float, fill: str, rx: float = 0,
             stroke: Optional[str] = None, sw: float = 1.0, opacity: Optional[float] = None):
        if w <= 0 or h <= 0:
            return
        op = f' opacity="{opacity:g}"' if opacity is not None else ""
        st = f' stroke="{stroke}" stroke-width="{sw:g}"' if stroke else ""
        self.p.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="{rx:g}" fill="{fill}"{st}{op}/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = GRID,
             sw: float = 1.0, dash: Optional[str] = None, opacity: Optional[float] = None):
        op = f' opacity="{opacity:g}"' if opacity is not None else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{sw:g}"{da}{op}/>'
        )

    def path(self, d: str, stroke: str, sw: float = 2.0, fill: str = "none",
             dash: Optional[str] = None, opacity: Optional[float] = None):
        op = f' opacity="{opacity:g}"' if opacity is not None else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}" '
            f'stroke-linejoin="round" stroke-linecap="round"{da}{op}/>'
        )

    def circle(self, cx: float, cy: float, r: float, fill: str,
               stroke: Optional[str] = None, sw: float = 1.5,
               opacity: Optional[float] = None):
        op = f' opacity="{opacity:g}"' if opacity is not None else ""
        st = f' stroke="{stroke}" stroke-width="{sw:g}"' if stroke else ""
        self.p.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:g}" fill="{fill}"{st}{op}/>'
        )

    def text(self, x: float, y: float, s: Any, size: float = 12, fill: str = TEXT,
             anchor: str = "start", weight: str = "", opacity: Optional[float] = None,
             mono: bool = False):
        if s is None or str(s) == "":
            return
        op = f' opacity="{opacity:g}"' if opacity is not None else ""
        wt = f' font-weight="{weight}"' if weight else ""
        fam = MONO if mono else FONT
        self.p.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{fam}" font-size="{size:g}" '
            f'fill="{fill}" text-anchor="{anchor}"{wt}{op}>{_esc(s)}</text>'
        )

    def out(self) -> str:
        return (
            f'<svg viewBox="0 0 {self.w} {self.h}" width="100%" height="auto" '
            f'preserveAspectRatio="xMidYMid meet" role="img" '
            f'font-family="{FONT}">' + "".join(self.p) + "</svg>"
        )


def _text_width(s: str, size: float) -> float:
    return len(s) * size * 0.56


# ---------------------------------------------------------------------------
# Spec -> tidy model
# ---------------------------------------------------------------------------
def _series(spec: ChartSpec) -> list[dict]:
    out: list[dict] = []
    for i, s in enumerate(spec.series or []):
        if not isinstance(s, dict):
            continue
        raw = s.get("values")
        if not isinstance(raw, (list, tuple)):
            raw = [] if raw is None else [raw]
        out.append({
            "name": str(s.get("name") or f"Series {i + 1}"),
            "values": [_opt_num(v) for v in raw],
            "color": str(s.get("color") or SERIES[i % len(SERIES)]),
            "dash": bool(s.get("dash")),
        })
    if not out:
        out = [{"name": "Values", "values": [], "color": SERIES[0], "dash": False}]
    return out


def _cats(spec: ChartSpec, n: int) -> list[str]:
    xs = list(spec.x or [])
    if len(xs) >= n:
        return [_label_of(v) for v in xs[:n]]
    return [_label_of(v) for v in xs] + [""] * (n - len(xs))


def _label_of(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v))}"
        return f"{v:g}"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    return str(v)


def _x_is_numeric(spec: ChartSpec) -> bool:
    xs = list(spec.x or [])
    if not xs:
        return False
    for v in xs:
        try:
            float(v)
        except (TypeError, ValueError):
            return False
    return True


def _values_flat(series: Sequence[dict]) -> list[float]:
    out: list[float] = []
    for s in series:
        for v in s["values"]:
            if v is not None:
                out.append(float(v))
    return out


# ---------------------------------------------------------------------------
# Legend + frame
# ---------------------------------------------------------------------------
def _pack_legend(items: Sequence[tuple[str, str]], width: float) -> list[list[tuple[str, str]]]:
    rows: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    cur_w = 0.0
    for name, color in items:
        w = 22 + _text_width(_clip(name, 38), 11.5)
        if cur and cur_w + w > width:
            rows.append(cur)
            cur, cur_w = [], 0.0
        cur.append((name, color))
        cur_w += w + 16
    if cur:
        rows.append(cur)
    return rows[:3]


def _frame(spec: ChartSpec, width: int, height: int, left: float, bottom: float,
           legend: bool = True, right: float = 24.0):
    ser = _series(spec)
    items = [(s["name"], s["color"]) for s in ser] if (legend and len(ser) > 0) else []
    if len(items) == 1 and items[0][0] in ("Values", "Series 1", ""):
        items = []
    rows = _pack_legend(items, width - left - right)
    legend_h = (len(rows) * 19 + 12) if rows else 8
    svg = _Svg(width, height)
    top = 14.0
    if rows:
        y = top + 13
        for row in rows:
            x = left
            for name, color in row:
                svg.rect(x, y - 8, 12, 10, color, rx=2)
                svg.text(x + 18, y, _clip(name, 38), 11.5, MUTED)
                x += 22 + _text_width(_clip(name, 38), 11.5) + 16
            y += 19
    box = _Box(left, top + legend_h, width - right, height - bottom)
    return svg, box, ser


def _axes(svg: _Svg, box: _Box, spec: ChartSpec, y_min: float, y_max: float,
          y_kind: str, y_frac_pct: bool = False, x_ticks: Optional[list[tuple[float, str]]] = None,
          vlines: Optional[list[float]] = None):
    """Horizontal grid + y axis labels + optional x tick labels + axis titles."""
    ticks, lo, hi = _nice_ticks(0.0 if (y_min >= 0) else y_min, y_max, 5)
    if not ticks:
        ticks, lo, hi = [0.0, 1.0], 0.0, 1.0

    def sy(v: float) -> float:
        span = (hi - lo) or 1.0
        return box.y1 - (v - lo) / span * box.h

    for t in ticks:
        y = sy(t)
        svg.line(box.x0, y, box.x1, y, GRID, 1.0)
        svg.text(box.x0 - 10, y + 4, _fmt_tick(t, y_kind, y_frac_pct), 11, MUTED,
                 anchor="end")
    # baseline / axis
    svg.line(box.x0, box.y1, box.x1, box.y1, AXIS, 1.2)
    svg.line(box.x0, box.y0, box.x0, box.y1, AXIS, 1.2)

    if vlines:
        for x in vlines:
            svg.line(x, box.y0, x, box.y1, GRID, 1.0, dash="3 4")

    if x_ticks:
        for x, lab in x_ticks:
            if lab == "":
                continue
            svg.text(x, box.y1 + 20, lab, 11, MUTED, anchor="middle")

    if spec.x_label:
        svg.text((box.x0 + box.x1) / 2, box.y1 + 48, _clip(spec.x_label, 70), 11.5,
                 MUTED, anchor="middle")
    if spec.y_label:
        cx = max(14.0, box.x0 - 62)
        svg.text(cx, (box.y0 + box.y1) / 2, _clip(spec.y_label, 40), 11.5, MUTED,
                 anchor="middle")
    return sy, lo, hi


def _cat_ticks(box: _Box, n: int, labels: Sequence[str], step: int = 1) -> list[tuple[float, str]]:
    if n <= 0:
        return []
    bw = box.w / n
    out = []
    for i in range(n):
        if i % step:
            continue
        out.append((box.x0 + bw * (i + 0.5), _clip(labels[i] if i < len(labels) else "", 12)))
    return out


def _thin(n: int, max_labels: int = 14) -> int:
    return 1 if n <= max_labels else int(math.ceil(n / max_labels))


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------
def _annotate(svg: _Svg, box: _Box, spec: ChartSpec, x_of, y_of, x_labels: Sequence[str],
              baseline_offset: float = 0.0, annotate_values_are_x: bool = False):
    """Draw labelled dots. Never raises, silently skips what it cannot place."""
    anns = list(spec.annotations or [])
    if not anns:
        return
    n = len(x_labels)
    for k, a in enumerate(anns[:4]):
        if not isinstance(a, dict):
            continue
        ax, ay = a.get("x"), a.get("y")
        text = _clip(a.get("text") or "", 42)
        av = _opt_num(ax)
        px: Optional[float] = None
        if av is None:
            # categorical annotation: match by label
            if n and str(ax) in [str(l) for l in x_labels]:
                idx = [str(l) for l in x_labels].index(str(ax))
                px = box.x0 + box.w * (idx + 0.5) / max(n, 1)
        elif annotate_values_are_x:
            px = x_of(av)
        else:
            if n:
                lab = [_opt_num(l) for l in x_labels]
                best, bd = 0, None
                for i, lv in enumerate(lab):
                    if lv is None:
                        continue
                    d = abs(lv - av)
                    if bd is None or d < bd:
                        bd, best = d, i
                px = box.x0 + box.w * (best + 0.5) / max(n, 1)
            else:
                px = x_of(av)
        if px is None:
            continue
        px = min(max(px, box.x0 + 2), box.x1 - 2)
        yv = _opt_num(ay)
        py = y_of(yv) if yv is not None else box.y1
        py = min(max(py, box.y0 + 2), box.y1 - 2)

        top_y = max(box.y0 + 14, py - 14)
        svg.line(px, py, px, top_y - 10, "#64748b", 1.0, dash="3 3", opacity=0.85)
        svg.circle(px, py, 4.6, "#0f172a", stroke="#e2e8f0", sw=1.6)
        svg.circle(px, py, 2.0, "#e2e8f0")
        if text:
            tw = _text_width(text, 11) + 12
            tx = min(max(px - tw / 2, box.x0), box.x1 - tw)
            ty = top_y - 4 - (k % 2) * 0
            svg.rect(tx, ty - 12, tw, 17, "#1e293b", rx=4, stroke="#334155", sw=1,
                     opacity=0.95)
            svg.text(tx + tw / 2, ty + 1, text, 11, TEXT, anchor="middle")


# ---------------------------------------------------------------------------
# Chart: line
# ---------------------------------------------------------------------------
def _chart_line(spec: ChartSpec) -> str:
    ser = _series(spec)
    xs = list(spec.x or [])
    n = max([len(s["values"]) for s in ser] + [0])
    numeric_x = _x_is_numeric(spec) and len(xs) >= n
    labels = _cats(spec, n)
    svg, box, ser = _frame(spec, 880, 430, 86, 62)

    vals = _values_flat(ser)
    y_max = max(vals) if vals else 1.0
    y_min = min(vals) if vals else 0.0
    if y_min > 0:
        y_min = 0.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.12
    y_max += pad
    if y_min < 0:
        y_min -= pad

    y_kind = _axis_kind(spec.y_label)
    frac_pct = y_kind == "pct" and y_max <= 1.5 and y_min >= -0.01

    if numeric_x:
        xn = [_num(v) for v in xs[:n]]
        xlo, xhi = min(xn), max(xn)
        if xhi <= xlo:
            xhi = xlo + 1.0

        def x_of(i: float) -> float:
            return box.x0 + (i - xlo) / (xhi - xlo) * box.w

        pts_x = [x_of(v) for v in xn]
        xticks, tlo, thi = _nice_ticks(xlo, xhi, 6)
        xk = _axis_kind(spec.x_label)
        xt = [(_safe_x(box, x_of(t)), _fmt_tick(t, xk)) for t in xticks]
    else:
        bw = box.w / max(n, 1)

        def x_of(i: float) -> float:
            return box.x0 + bw * (i + 0.5)

        pts_x = [x_of(i) for i in range(n)]
        xt = _cat_ticks(box, n, labels, _thin(n))

    x_ticks_for_axes = xt if not numeric_x else xt

    def _draw_axes():
        return _axes(svg, box, spec, y_min, y_max, y_kind, frac_pct, x_ticks_for_axes)

    sy, lo, hi = _draw_axes()

    for s in ser:
        d_seg: list[str] = []
        cur: list[tuple[float, float]] = []
        for i, v in enumerate(s["values"][:n]):
            if v is None or i >= len(pts_x):
                if len(cur) > 1:
                    d_seg.append(_poly(cur))
                cur = []
                continue
            cur.append((pts_x[i], sy(v)))
        if len(cur) > 1:
            d_seg.append(_poly(cur))
        elif len(cur) == 1:
            d_seg.append(f"M {cur[0][0]:.2f} {cur[0][1]:.2f} l 0.01 0")
        if not d_seg:
            continue
        dash = "6 4" if s["dash"] else None
        svg.path(" ".join(d_seg), s["color"], 2.4, dash=dash)
        for i, v in enumerate(s["values"][:n]):
            if v is not None and i < len(pts_x) and n <= 24:
                svg.circle(pts_x[i], sy(v), 2.8, BG, stroke=s["color"], sw=1.8)

    _annotate(svg, box, spec, x_of, sy, labels,
              annotate_values_are_x=bool(numeric_x))
    return svg.out()


def _poly(pts: Sequence[tuple[float, float]]) -> str:
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts)


def _safe_x(box: _Box, x: float) -> float:
    return min(max(x, box.x0), box.x1)


# ---------------------------------------------------------------------------
# Chart: bar / grouped_bar
# ---------------------------------------------------------------------------
def _chart_bar(spec: ChartSpec, grouped: bool = False) -> str:
    ser = _series(spec)
    n = max([len(s["values"]) for s in ser] + [0])
    if n == 0:
        n = len(spec.x or [])
    labels = _cats(spec, n)
    svg, box, ser = _frame(spec, 880, 430, 86, 62)

    vals = _values_flat(ser)
    y_max = max(vals) if vals else 1.0
    y_min = min(vals) if vals else 0.0
    if y_min > 0:
        y_min = 0.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_max += (y_max - y_min) * 0.10
    if y_min < 0:
        y_min -= abs(y_min) * 0.10

    y_kind = _axis_kind(spec.y_label)
    frac_pct = y_kind == "pct" and y_max <= 1.5 and y_min >= -0.01
    sy, lo, hi = _axes(svg, box, spec, y_min, y_max, y_kind, frac_pct,
                       _cat_ticks(box, n, labels, _thin(n)))

    nser = max(1, len(ser))
    group = box.w / max(n, 1)
    inner = group * 0.78
    bw = inner / nser if (grouped and nser > 1) else min(inner, 54)
    zero = sy(0.0)

    for i in range(n):
        gx = box.x0 + group * i + (group - inner) / 2
        for k, s in enumerate(ser):
            v = s["values"][i] if i < len(s["values"]) else None
            if v is None:
                continue
            if grouped and nser > 1:
                x = gx + bw * k
            else:
                x = gx + (inner - bw) / 2 if nser == 1 else gx + bw * k
            yv = sy(v)
            top, bot = (yv, zero) if v >= 0 else (zero, yv)
            svg.rect(x, top, max(bw - 2, 1), max(bot - top, 1), s["color"], rx=3,
                     opacity=0.92)

    def x_of(i: float) -> float:
        return box.x0 + group * (i + 0.5)

    _annotate(svg, box, spec, lambda v: v, sy, labels)
    return svg.out()


# ---------------------------------------------------------------------------
# Chart: hist (we bin the raw values ourselves)
# ---------------------------------------------------------------------------
def _chart_hist(spec: ChartSpec) -> str:
    ser = _series(spec)
    vals = [v for v in _values_flat(ser[:1])]
    svg, box, ser = _frame(spec, 880, 430, 86, 62)
    if not vals:
        svg.text(box.x0 + 20, box.y0 + 40, "No outcomes to plot.", 13, MUTED)
        return svg.out()

    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + 1.0
    nb = int(min(30, max(6, math.sqrt(len(vals)))))
    width = (hi - lo) / nb
    counts = [0] * nb
    for v in vals:
        i = int((v - lo) / width)
        if i >= nb:
            i = nb - 1
        counts[i] += 1
    y_max = max(counts) if counts else 1
    y_max += max(1.0, y_max * 0.10)
    y_kind = "num"
    x_kind = _axis_kind(spec.x_label)

    mids = [lo + width * (i + 0.5) for i in range(nb)]
    edges = [lo + width * i for i in range(nb + 1)]

    def sx(v: float) -> float:
        return box.x0 + (v - lo) / (hi - lo) * box.w

    xticks, tlo, thi = _nice_ticks(lo, hi, 6)
    sy, ylo, yhi = _axes(svg, box, spec, 0.0, y_max, y_kind, False,
                         [(_safe_x(box, sx(t)), _fmt_tick(t, x_kind)) for t in xticks])

    color = ser[0]["color"] if ser else SERIES[0]
    for i, c in enumerate(counts):
        x0, x1 = sx(edges[i]), sx(edges[i + 1])
        h = (c / (yhi or 1)) * box.h
        svg.rect(x0 + 0.8, box.y1 - h, max(x1 - x0 - 1.6, 1), h, color, rx=2,
                 opacity=0.88)

    _annotate(svg, box, spec, sx, sy, [], annotate_values_are_x=True)
    return svg.out()


# ---------------------------------------------------------------------------
# Chart: stacked_area
# ---------------------------------------------------------------------------
def _chart_stacked_area(spec: ChartSpec) -> str:
    ser = _series(spec)
    n = max([len(s["values"]) for s in ser] + [0])
    labels = _cats(spec, n)
    svg, box, ser = _frame(spec, 880, 430, 86, 62)

    totals = [0.0] * n
    for s in ser:
        for i in range(n):
            v = s["values"][i] if i < len(s["values"]) else None
            totals[i] += 0.0 if v is None else max(0.0, v)
    y_max = max(totals) if totals else 1.0
    if y_max <= 0:
        y_max = 1.0
    y_max += y_max * 0.06
    y_kind = _axis_kind(spec.y_label)
    frac_pct = y_kind == "pct" and y_max <= 1.5
    sy, lo, hi = _axes(svg, box, spec, 0.0, y_max, y_kind, frac_pct,
                       _cat_ticks(box, n, labels, _thin(n)))

    bw = box.w / max(n, 1)
    px = [box.x0 + bw * (i + 0.5) for i in range(n)]
    lower = [0.0] * n
    for s in ser:
        upper = [lower[i] + (0.0 if (i >= len(s["values"]) or s["values"][i] is None)
                             else max(0.0, s["values"][i])) for i in range(n)]
        pts = [(px[i], sy(upper[i])) for i in range(n)]
        back = [(px[i], sy(lower[i])) for i in reversed(range(n))]
        d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts + back) + " Z"
        svg.path(d, s["color"], 1.0, fill=s["color"], opacity=0.82)
        svg.path("M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts), s["color"], 1.6,
                 opacity=1.0)
        lower = upper
    return svg.out()


# ---------------------------------------------------------------------------
# Chart: heatmap  (one row per series, one column per x label)
# ---------------------------------------------------------------------------
def _chart_heatmap(spec: ChartSpec) -> str:
    ser = _series(spec)
    cols = list(spec.x or [])
    rows = [s["name"] for s in ser]
    n_r, n_c = len(ser), len(cols)
    if n_r == 0 or n_c == 0:
        return _chart_bar(spec)

    max_rl = max([len(_clip(r, 24)) for r in rows] + [4])
    left = min(210.0, max(80.0, max_rl * 6.6 + 18))
    height = int(min(760, max(300, 150 + n_r * min(46, max(24, 420 / max(n_r, 1))))))
    svg = _Svg(880, height)
    box = _Box(left, 26, 880 - 26, height - 74)

    flat = _values_flat(ser)
    lo = min(flat) if flat else 0.0
    hi = max(flat) if flat else 1.0
    if hi <= lo:
        hi = lo + 1.0

    cw = box.w / n_c
    ch = box.h / n_r
    show_text = n_c <= 14 and n_r <= 16
    for r, s in enumerate(ser):
        for c in range(n_c):
            v = s["values"][c] if c < len(s["values"]) else None
            x = box.x0 + cw * c
            y = box.y0 + ch * r
            if v is None:
                svg.rect(x + 1, y + 1, max(cw - 2, 1), max(ch - 2, 1), "#111c31", rx=3)
                continue
            t = (v - lo) / (hi - lo)
            svg.rect(x + 1, y + 1, max(cw - 2, 1), max(ch - 2, 1),
                     _lerp_color(HEAT_STOPS, t), rx=3)
            if show_text:
                a = abs(v)
                lab = f"{v:.2f}" if a < 10 else f"{v:,.0f}"
                fill = "#0f172a" if 0.35 < t < 0.78 else TEXT
                svg.text(x + cw / 2, y + ch / 2 + 4, lab, min(11.5, max(8.5, ch * 0.3)),
                         fill, anchor="middle")

    for r, name in enumerate(rows):
        svg.text(box.x0 - 10, box.y0 + ch * (r + 0.5) + 4, _clip(name, 24), 11.5, MUTED,
                 anchor="end")
    step = _thin(n_c, 14)
    for c in range(n_c):
        if c % step:
            continue
        svg.text(box.x0 + cw * (c + 0.5), box.y1 + 18, _clip(_label_of(cols[c]), 12), 11,
                 MUTED, anchor="middle")

    if spec.x_label:
        svg.text((box.x0 + box.x1) / 2, box.y1 + 44, _clip(spec.x_label, 70), 11.5, MUTED,
                 anchor="middle")
    if spec.y_label:
        svg.text(max(12.0, box.x0 - left + 14), box.y0 - 10, _clip(spec.y_label, 60), 11.5,
                 MUTED, anchor="start")
    # scale hint
    svg.text(box.x1, box.y0 - 10, f"low {_fmt_tick(lo, 'num')}  ->  high {_fmt_tick(hi, 'num')}",
             11, MUTED, anchor="end")
    return svg.out()


# ---------------------------------------------------------------------------
# Chart: scatter
# ---------------------------------------------------------------------------
def _chart_scatter(spec: ChartSpec) -> str:
    ser = _series(spec)
    if len(ser) >= 2 and not list(spec.x or []):
        xs = ser[0]["values"]
        ys = ser[1]["values"]
    else:
        xs = [_opt_num(v) for v in list(spec.x or [])]
        ys = ser[0]["values"] if ser else []
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    svg, box, ser = _frame(spec, 880, 430, 86, 62)
    if not pairs:
        svg.text(box.x0 + 20, box.y0 + 40, "No points to plot.", 13, MUTED)
        return svg.out()

    xlo = min(p[0] for p in pairs)
    xhi = max(p[0] for p in pairs)
    ylo = min(p[1] for p in pairs)
    yhi = max(p[1] for p in pairs)
    if xhi <= xlo:
        xhi = xlo + 1.0
    if yhi <= ylo:
        yhi = ylo + 1.0
    if ylo > 0:
        ylo = 0.0
    yhi += (yhi - ylo) * 0.08
    xk, yk = _axis_kind(spec.x_label), _axis_kind(spec.y_label)
    frac_pct = yk == "pct" and yhi <= 1.5

    def sx(v: float) -> float:
        return box.x0 + (v - xlo) / (xhi - xlo) * box.w

    xticks, _, _ = _nice_ticks(xlo, xhi, 6)
    sy, lo, hi = _axes(svg, box, spec, ylo, yhi, yk, frac_pct,
                       [(_safe_x(box, sx(t)), _fmt_tick(t, xk)) for t in xticks])
    color = (spec.series or [{}])[0].get("color") if spec.series else None
    color = color or SERIES[1]
    for x, y in pairs:
        svg.circle(sx(x), sy(y), 4.2, color, stroke=BG, sw=1.0, opacity=0.78)

    def x_of(v: float) -> float:
        return sx(v)

    _annotate(svg, box, spec, x_of, sy, [], annotate_values_are_x=True)
    return svg.out()


# ---------------------------------------------------------------------------
# Chart: funnel
# ---------------------------------------------------------------------------
def _chart_funnel(spec: ChartSpec) -> str:
    ser = _series(spec)
    labels = list(spec.x or [])
    values = ser[0]["values"] if ser else []
    n = max(len(labels), len(values))
    labels = [_label_of(v) for v in labels[:n]] + [""] * max(0, n - len(labels))
    if n == 0:
        n = 0
    height = int(min(700, max(240, 110 + n * 46)))
    left = min(240.0, max(90.0, max([len(_clip(l, 26)) for l in labels] + [4]) * 6.7 + 20))
    svg = _Svg(880, height)
    box = _Box(left + 60, 28, 880 - 70, height - 62)

    mv = max([v for v in values if v is not None] + [0.0]) or 1.0
    step = box.h / max(n, 1)
    bh = min(step * 0.72, 38)
    ykind = _axis_kind(spec.y_label)
    frac = ykind == "pct" and mv <= 1.5

    for i in range(n):
        v = values[i] if i < len(values) else None
        v = 0.0 if v is None else max(0.0, v)
        w = (v / mv) * box.w if mv > 0 else 0
        y = box.y0 + step * i + (step - bh) / 2
        cx = box.x0 + box.w / 2
        svg.rect(cx - w / 2, y, max(w, 2), bh, SERIES[4] if i == 0 else "#3b82f6", rx=4,
                 opacity=0.9)
        svg.text(left - 12, y + bh / 2 + 4, _clip(labels[i] or f"Step {i + 1}", 26), 12,
                 TEXT, anchor="end")
        svg.text(cx + w / 2 + 10, y + bh / 2 + 4, _fmt_tick(v, ykind, frac), 11.5, MUTED)
        if i + 1 < n:
            svg.line(cx, y + bh, cx, y + step, AXIS, 1.0, dash="3 4", opacity=0.6)

    if spec.y_label:
        svg.text(box.x0 + box.w / 2, height - 20, _clip(spec.y_label, 70), 11.5, MUTED,
                 anchor="middle")
    return svg.out()


# ---------------------------------------------------------------------------
# Chart: waterfall
# ---------------------------------------------------------------------------
def _chart_waterfall(spec: ChartSpec) -> str:
    ser = _series(spec)
    labels = _cats(spec, max([len(s["values"]) for s in ser[:1]] + [0]))
    values = ser[0]["values"] if ser else []
    n = len(values)
    svg, box, ser = _frame(spec, 880, 430, 86, 62, legend=False)
    if n == 0:
        svg.text(box.x0 + 20, box.y0 + 40, "Nothing to plot.", 13, MUTED)
        return svg.out()

    running, bars = 0.0, []
    for v in values:
        v = 0.0 if v is None else v
        bars.append((running, running + v, v))
        running += v
    lo = min([min(a, b) for a, b, _ in bars] + [0.0])
    hi = max([max(a, b) for a, b, _ in bars] + [0.0])
    if hi <= lo:
        hi = lo + 1.0
    hi += (hi - lo) * 0.08
    y_kind = _axis_kind(spec.y_label)
    frac = y_kind == "pct" and hi <= 1.5
    sy, ylo, yhi = _axes(svg, box, spec, lo, hi, y_kind, frac,
                         _cat_ticks(box, n, labels, _thin(n)))
    group = box.w / n
    bw = min(group * 0.66, 56)
    for i, (a, b, d) in enumerate(bars):
        x = box.x0 + group * i + (group - bw) / 2
        top, bot = (sy(b), sy(a)) if b >= a else (sy(a), sy(b))
        color = "#22c55e" if d >= 0 else "#ef4444"
        svg.rect(x, top, bw, max(bot - top, 1), color, rx=3, opacity=0.9)
        if i + 1 < len(bars):
            nxt = sy(b)
            svg.line(x + bw, nxt, box.x0 + group * (i + 1) + (group - bw) / 2, nxt,
                     AXIS, 1.0, dash="3 3", opacity=0.8)
    return svg.out()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def _degrade(spec: ChartSpec) -> str:
    """Unknown chart type: draw something sensible from the first series."""
    ser = _series(spec)
    vals = _values_flat(ser[:1])
    if len(vals) > 2:
        return _chart_line(spec)
    return _chart_bar(spec)


_RENDERERS = {
    "line": _chart_line,
    "bar": lambda s: _chart_bar(s, grouped=False),
    "grouped_bar": lambda s: _chart_bar(s, grouped=True),
    "hist": _chart_hist,
    "histogram": _chart_hist,
    "stacked_area": _chart_stacked_area,
    "area": _chart_stacked_area,
    "heatmap": _chart_heatmap,
    "scatter": _chart_scatter,
    "funnel": _chart_funnel,
    "waterfall": _chart_waterfall,
}


def render_chart(spec: ChartSpec) -> str:
    """Return an inline ``<svg>...</svg>`` string for a ChartSpec. Never raises."""
    try:
        if spec is None:
            raise ValueError("no chart")
        fn = _RENDERERS.get(str(getattr(spec, "type", "") or "").strip().lower())
        if fn is None:
            fn = _degrade
        return fn(spec)
    except Exception:
        try:
            title = _clip(getattr(spec, "title", "") or "Chart", 60)
            svg = _Svg(880, 120)
            svg.rect(0, 0, 880, 120, BG, rx=12)
            svg.text(24, 52, title, 14, TEXT, weight="600")
            svg.text(24, 78, "This chart could not be drawn from the data provided.",
                     12, MUTED)
            return svg.out()
        except Exception:
            return '<svg viewBox="0 0 400 60" width="100%" height="auto"></svg>'


# ---------------------------------------------------------------------------
# PNG (matplotlib, Agg) - used by the PDF
# ---------------------------------------------------------------------------
def _mpl_style(ax, fig):
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def render_chart_png(spec: ChartSpec) -> bytes:
    """Render a ChartSpec to PNG bytes with matplotlib (Agg). Never raises."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _TINY_PNG

    fig = None
    try:
        ctype = str(getattr(spec, "type", "") or "").strip().lower()
        ser = _series(spec)
        xs = list(spec.x or [])
        labels = [_label_of(v) for v in xs]
        n = max([len(s["values"]) for s in ser] + [len(xs)])

        if ctype in ("heatmap",) and ser and xs:
            n_r, n_c = len(ser), len(xs)
            h = min(9.0, max(3.2, 1.2 + n_r * 0.42))
            fig, ax = plt.subplots(figsize=(9.5, h), dpi=110)
            grid = [[(s["values"][c] if c < len(s["values"]) else None) for c in range(n_c)]
                    for s in ser]
            import numpy as np
            arr = np.array([[np.nan if v is None else v for v in row] for row in grid],
                           dtype=float)
            im = ax.imshow(arr, aspect="auto",
                           cmap=_mpl_heatmap_cmap(), interpolation="nearest")
            ax.set_xticks(range(n_c))
            ax.set_xticklabels([_clip(l, 12) for l in labels], rotation=0, color=MUTED)
            ax.set_yticks(range(n_r))
            ax.set_yticklabels([_clip(s["name"], 22) for s in ser], color=MUTED)
            if n_c <= 14 and n_r <= 16:
                for r in range(n_r):
                    for c in range(n_c):
                        v = arr[r, c]
                        if np.isnan(v):
                            continue
                        ax.text(c, r, f"{v:,.2f}" if abs(v) < 10 else f"{v:,.0f}",
                                ha="center", va="center", fontsize=7.5, color=TEXT)
            cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cb.ax.yaxis.set_tick_params(color=MUTED, labelsize=8)
            plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=MUTED)
            ax.set_facecolor(BG)
            fig.patch.set_facecolor(BG)
            if spec.x_label:
                ax.set_xlabel(spec.x_label, color=MUTED, fontsize=9)
        else:
            fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=110)
            _mpl_style(ax, fig)
            if ctype in ("hist", "histogram"):
                vals = _values_flat(ser[:1])
                if vals:
                    ax.hist(vals, bins=int(min(30, max(6, math.sqrt(len(vals))))),
                            color=SERIES[0], edgecolor=BG, linewidth=0.6)
                ax.set_ylabel(spec.y_label or "How often", color=MUTED, fontsize=9)
            elif ctype in ("funnel",):
                vals = [0.0 if v is None else v for v in (ser[0]["values"] if ser else [])]
                if vals:
                    ypos = list(range(len(vals)))[::-1]
                    ax.barh(ypos, vals, color=SERIES[4], height=0.6)
                    ax.set_yticks(ypos)
                    ax.set_yticklabels([_clip(l, 22) for l in labels[:len(vals)]],
                                       color=MUTED, fontsize=8)
                ax.set_xlabel(spec.y_label or "", color=MUTED, fontsize=9)
            elif ctype in ("stacked_area", "area"):
                import numpy as np
                base = None
                for s in ser:
                    arr = np.array([0.0 if v is None else v for v in s["values"][:n]],
                                   dtype=float)
                    if arr.size == 0:
                        continue
                    if base is None:
                        base = np.zeros_like(arr)
                    ax.fill_between(range(arr.size), base, base + arr,
                                    color=s["color"], alpha=0.85, label=s["name"])
                    base = base + arr
                ax.set_xlim(0, max(n - 1, 1))
                ax.set_xticks(range(n))
                ax.set_xticklabels([_clip(l, 12) for l in labels[:n]], color=MUTED,
                                   fontsize=8)
                ax.legend(fontsize=8, facecolor=BG, edgecolor=AXIS,
                          labelcolor=TEXT, framealpha=0.9)
            elif ctype in ("waterfall",):
                vals = [0.0 if v is None else v for v in (ser[0]["values"] if ser else [])]
                run = 0.0
                for i, v in enumerate(vals):
                    ax.bar(i, v, bottom=run, color=("#22c55e" if v >= 0 else "#ef4444"))
                    run += v
                ax.set_xticks(range(len(vals)))
                ax.set_xticklabels([_clip(l, 12) for l in labels[:len(vals)]],
                                   color=MUTED, fontsize=8)
            elif ctype in ("scatter",):
                pts = [(i, v) for i, v in enumerate(ser[0]["values"] if ser else [])
                       if v is not None]
                if pts:
                    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=28,
                               color=SERIES[1], alpha=0.8, edgecolors="none")
                ax.set_xticks(range(n))
                ax.set_xticklabels([_clip(l, 12) for l in labels[:n]], color=MUTED,
                                   fontsize=8)
            elif ctype in ("grouped_bar",) or (ctype == "bar" and len(ser) > 1):
                import numpy as np
                nser = max(1, len(ser))
                width = 0.8 / nser
                idx = list(range(max(n, 1)))
                for k, s in enumerate(ser):
                    vals = [0.0 if v is None else v for v in s["values"][:n]]
                    pos = [i + width * k - 0.4 + width / 2 for i in idx]
                    ax.bar(pos, vals, width=width * 0.94, color=s["color"],
                           label=_clip(s["name"], 30))
                ax.set_xticks(idx)
                ax.set_xticklabels([_clip(l, 12) for l in labels[:n]], color=MUTED,
                                   fontsize=8)
                ax.legend(fontsize=8, facecolor=BG, edgecolor=AXIS,
                          labelcolor=TEXT, framealpha=0.9)
            elif ctype in ("line", ""):
                for s in ser:
                    ys = [v for v in s["values"][:n]]
                    ax.plot(range(len(ys)), ys, color=s["color"],
                            linestyle="--" if s["dash"] else "-", linewidth=2.0,
                            label=_clip(s["name"], 30))
                ax.set_xticks(range(n))
                ax.set_xticklabels([_clip(l, 12) for l in labels[:n]], color=MUTED,
                                   fontsize=8)
                if len(ser) > 1:
                    ax.legend(fontsize=8, facecolor=BG, edgecolor=AXIS,
                              labelcolor=TEXT, framealpha=0.9)
            else:  # bar / unknown -> first series as bars
                vals = [0.0 if v is None else v for v in (ser[0]["values"] if ser else [])]
                ax.bar(range(len(vals)), vals, color=SERIES[0], width=0.7)
                ax.set_xticks(range(len(vals)))
                ax.set_xticklabels([_clip(l, 12) for l in labels[:len(vals)]],
                                   color=MUTED, fontsize=8)

            if spec.x_label:
                ax.set_xlabel(_clip(spec.x_label, 70), color=MUTED, fontsize=9)
            if spec.y_label and ctype not in ("funnel",):
                ax.set_ylabel(_clip(spec.y_label, 40), color=MUTED, fontsize=9)
            ax.set_title(_clip(spec.title, 90), color=TEXT, fontsize=12, pad=10)

        fig.tight_layout()
        buf = __import__("io").BytesIO()
        fig.savefig(buf, format="png", facecolor=BG, bbox_inches="tight")
        return buf.getvalue()
    except Exception:
        try:
            fig2, ax2 = plt.subplots(figsize=(9.5, 2.2), dpi=110)
            fig2.patch.set_facecolor(BG)
            ax2.set_facecolor(BG)
            ax2.axis("off")
            ax2.text(0.01, 0.6, _clip(getattr(spec, "title", "") or "Chart", 80),
                     color=TEXT, fontsize=12)
            ax2.text(0.01, 0.25, "This chart could not be drawn from the data provided.",
                     color=MUTED, fontsize=9)
            buf = __import__("io").BytesIO()
            fig2.savefig(buf, format="png", facecolor=BG, bbox_inches="tight")
            return buf.getvalue()
        except Exception:
            return _TINY_PNG
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass


def _mpl_heatmap_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("arena_heat",
                                             ["#0d1b33", "#1d4ed8", "#3b82f6",
                                              "#f59e0b", "#ef4444"])
