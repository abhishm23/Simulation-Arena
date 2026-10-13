"""The Simulation Arena brand mark, as inline SVG.

Single source of truth for every artifact (HTML report, PDF, Excel cover).
Geometry is identical to the app favicon, so print output and screen output
carry the same identity.

The mark: an arena (ring) holding a convergence fan - many possible futures
leaving one decision point (blue) and arriving at one answer (green).
"""
from __future__ import annotations

# --- geometry (64x64 viewBox) ---------------------------------------------
CX = CY = 32.0
DISC_R = 30.5
RING_R = 28.5
RING_W = 3.2

O = (18.0, 45.0)      # the decision you make today
D = (46.0, 19.0)      # the answer you get back
FAN_W = 3.2

DOT_DST_R = 4.4
DOT_DST_GLOW = 8.8
DOT_SRC_R = 3.4

_PATHS = (
    "M 18.00 45.00 Q 17.71 16.61 46.00 19.00",
    "M 18.00 45.00 Q 26.28 25.84 46.00 19.00",
    "M 18.00 45.00 Q 37.72 38.16 46.00 19.00",
    "M 18.00 45.00 Q 46.29 47.39 46.00 19.00",
)

_DEFS = (
    '<linearGradient id="saDisc" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0" stop-color="#1b2942"/>'
    '<stop offset="1" stop-color="#0a1120"/></linearGradient>'
    '<linearGradient id="saRing" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#22c55e"/>'
    '<stop offset="1" stop-color="#3b82f6"/></linearGradient>'
    '<linearGradient id="saFan" gradientUnits="userSpaceOnUse" '
    f'x1="{O[0]}" y1="{O[1]}" x2="{D[0]}" y2="{D[1]}">'
    '<stop offset="0" stop-color="#60a5fa"/>'
    '<stop offset="1" stop-color="#22c55e"/></linearGradient>'
)


def _body(defs_ids: str = "sa", prefix: str = "") -> str:
    """The mark's shapes. `prefix` namespaces gradient ids when several marks
    share one SVG document."""
    did = f"{prefix}{defs_ids}"
    paths = "".join(f'<path d="{d}"/>' for d in _PATHS)
    return (
        f'<circle cx="{CX}" cy="{CY}" r="{DISC_R}" fill="url(#{did}Disc)"/>'
        f'<circle cx="{CX}" cy="{CY}" r="{RING_R}" fill="none" '
        f'stroke="url(#{did}Ring)" stroke-width="{RING_W}"/>'
        f'<g fill="none" stroke="url(#{did}Fan)" stroke-width="{FAN_W}" '
        f'stroke-linecap="round">{paths}</g>'
        f'<circle cx="{D[0]}" cy="{D[1]}" r="8.8" fill="#22c55e" opacity="0.18"/>'
        f'<circle cx="{D[0]}" cy="{D[1]}" r="4.4" fill="#22c55e"/>'
        f'<circle cx="{O[0]}" cy="{O[1]}" r="3.4" fill="#60a5fa"/>'
    )


def _defs(prefix: str = "") -> str:
    did = f"{prefix}sa"
    return (
        _DEFS.replace('id="sa', f'id="{did}')
        .replace("url(#saDisc)", f"url(#{did}Disc)")
        .replace("url(#saRing)", f"url(#{did}Ring)")
        .replace("url(#saFan)", f"url(#{did}Fan)")
    )


FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">'
    f"<defs>{_defs()}</defs>{_body()}</svg>"
)


def mark(size: int = 40, prefix: str = "") -> str:
    """The mark on its own, at `size` pixels square."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
        f'width="{size}" height="{size}" role="img" aria-label="Simulation Arena">'
        f"<defs>{_defs(prefix)}</defs>{_body(prefix=prefix)}</svg>"
    )


def png(size: int = 256) -> bytes:
    """Raster version of the same mark, for PDFs (reportlab cannot read SVG).

    Drawn with numpy supersampling so the curves stay clean. Deterministic.
    """
    import io

    import numpy as np
    from PIL import Image, ImageDraw

    ss = 4 if size <= 256 else 2
    n = size * ss
    k = n / 64.0

    def gradient_disc():
        ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
        ux, uy = xs / k, ys / k
        dist = np.hypot(ux - CX, uy - CY)
        t = np.clip((uy - (CY - DISC_R)) / (2 * DISC_R), 0.0, 1.0)
        base = np.zeros((n, n, 3), dtype=np.float64)
        top, bot = np.array([27, 41, 66]), np.array([10, 17, 32])
        for i in range(3):
            base[:, :, i] = top[i] + (bot[i] - top[i]) * t
        edge = np.clip((DISC_R - dist) * k + 0.5, 0.0, 1.0)
        img = Image.new("RGBA", (n, n))
        img.paste(Image.fromarray(base.astype(np.uint8), "RGB"), (0, 0))
        img.putalpha(Image.fromarray((edge * 255).astype(np.uint8)))
        return img, ux, uy, dist

    canvas, ux, uy, dist = gradient_disc()

    inner, outer = RING_R - RING_W / 2, RING_R + RING_W / 2
    band = np.clip(np.minimum(dist - inner, outer - dist) * k + 0.5, 0.0, 1.0)
    s = np.clip(((ux - CX) + (uy - CY)) / (2 * DISC_R) + 0.5, 0.0, 1.0)
    green, blue = np.array([34, 197, 94]), np.array([59, 130, 246])
    ring_rgb = np.zeros((n, n, 3), dtype=np.float64)
    for i in range(3):
        ring_rgb[:, :, i] = green[i] + (blue[i] - green[i]) * s
    ring = Image.new("RGBA", (n, n))
    ring.paste(Image.fromarray(ring_rgb.astype(np.uint8), "RGB"), (0, 0))
    ring.putalpha(Image.fromarray((band * 255).astype(np.uint8)))
    canvas = Image.alpha_composite(canvas, ring)

    def qpoint(p0, c, p1, t):
        mt = 1 - t
        return (mt * mt * p0[0] + 2 * mt * t * c[0] + t * t * p1[0],
                mt * mt * p0[1] + 2 * mt * t * c[1] + t * t * p1[1])

    # rebuild the control points from the path definitions (kept in sync by hand)
    controls = [(17.71, 16.61), (26.28, 25.84), (37.72, 38.16), (46.29, 47.39)]
    lighter = np.array([96, 165, 250])
    fan = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(fan)
    r = FAN_W / 2.0 * k
    for c in controls:
        for j in range(700):
            tt = j / 699.0
            px, py = qpoint(O, c, D, tt)
            col = green + (lighter - green) * (1.0 - tt)
            col = tuple(int(round(v)) for v in col)
            d.ellipse([px * k - r, py * k - r, px * k + r, py * k + r], fill=col + (255,))
    canvas = Image.alpha_composite(canvas, fan)

    glow = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gr = 8.8 * k
    gd.ellipse([D[0] * k - gr, D[1] * k - gr, D[0] * k + gr, D[1] * k + gr],
               fill=tuple(int(v) for v in green) + (46,))
    canvas = Image.alpha_composite(canvas, glow)

    dots = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dots)
    for cx_, cy_, rr, col in ((D[0], D[1], DOT_DST_R, green),
                              (O[0], O[1], DOT_SRC_R, lighter)):
        rad = rr * k
        dd.ellipse([cx_ * k - rad, cy_ * k - rad, cx_ * k + rad, cy_ * k + rad],
                   fill=tuple(int(v) for v in col) + (255,))
    canvas = Image.alpha_composite(canvas, dots)

    out = canvas.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def lockup(width: int = 236) -> str:
    """Mark plus wordmark, for report headers."""
    height = int(width * 64 / 236)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 236 64" '
        f'width="{width}" height="{height}" role="img" aria-label="Simulation Arena">'
        f"<defs>{_defs()}</defs>{_body()}"
        '<text x="80" y="42" font-family="Inter, Segoe UI, system-ui, sans-serif" '
        'font-size="27" font-weight="700" letter-spacing="-0.4">'
        '<tspan fill="#e2e8f0">Simulation</tspan>'
        '<tspan fill="url(#saRing)"> Arena</tspan></text></svg>'
    )
