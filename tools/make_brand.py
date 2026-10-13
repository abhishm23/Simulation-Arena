"""Generate the Simulation Arena brand identity from a single geometry source.

The mark: an arena (ring) holding a convergence fan - many possible futures
leaving one decision point (blue dot, bottom-left) and arriving at one answer
(green dot, top-right). That is literally what the product does.

Outputs:
  frontend/public/favicon.svg          browser tab icon (SVG, modern browsers)
  frontend/public/favicon.ico          fallback for older browsers / Windows
  frontend/public/icon-192.png         PWA
  frontend/public/icon-512.png         PWA
  frontend/public/apple-touch-icon.png iOS home screen
  frontend/public/logo.svg             horizontal lockup for documents
  frontend/public/manifest.webmanifest
  frontend/src/components/BrandMark.jsx  inline React component (same geometry)

Deterministic: no randomness, no network.
"""
from __future__ import annotations

import math
import os
import pathlib

import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
PUB = ROOT / "frontend" / "public"
SRC = ROOT / "frontend" / "src" / "components"
PUB.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------
DISC_TOP = (27, 41, 66)      # #1b2942
DISC_BOT = (10, 17, 32)      # #0a1120
GREEN = (34, 197, 94)        # #22c55e
BLUE = (59, 130, 246)        # #3b82f6
BLUE_LIGHT = (96, 165, 250)  # #60a5fa
TEXT = (226, 232, 240)       # #e2e8f0

# --------------------------------------------------------------------------
# geometry (unit space, 64x64 viewBox)
# --------------------------------------------------------------------------
CX = CY = 32.0
DISC_R = 30.5
RING_R = 28.5
RING_W = 3.2

O = (18.0, 45.0)             # origin: your decision today
D = (46.0, 19.0)             # destination: the answer
FAN_W = 3.2
DEVIATIONS = (-10.5, -4.2, 4.2, 10.5)

DOT_DST_R = 4.4
DOT_DST_GLOW = 8.8
DOT_SRC_R = 3.4

# -- compact variant: for 16-32px, where fine strokes turn to mush ----------
# Fewer lines, much thicker, shorter dot overlap, ring pulled in so the mark
# breathes. Still obviously the same mark.
C_FAN_W = 3.6
C_DEVIATIONS = (-8.0, 0.0, 8.0)
C_DOT_DST_R = 3.8
C_DOT_SRC_R = 3.4
C_O = (21.0, 43.0)
C_D = (45.5, 21.5)


def _compact_controls():
    ox, oy = C_O
    dx, dy = C_D
    mx, my = 0.5 * (ox + dx), 0.5 * (oy + dy)
    vx, vy = dx - ox, dy - oy
    L = math.hypot(vx, vy)
    px, py = -vy / L, vx / L
    return [(mx + px * 2.0 * dev, my + py * 2.0 * dev) for dev in C_DEVIATIONS]


def control_points():
    """Quadratic-bezier control points that bend the fan by `deviation`."""
    ox, oy = O
    dx, dy = D
    mx, my = 0.5 * (ox + dx), 0.5 * (oy + dy)
    vx, vy = dx - ox, dy - oy
    L = math.hypot(vx, vy)
    px, py = -vy / L, vx / L                      # unit perpendicular
    out = []
    for dev in DEVIATIONS:
        # a quadratic curve peaks at half the control offset
        out.append((mx + px * 2.0 * dev, my + py * 2.0 * dev))
    return [(ox, oy), (dx, dy)], out


_, CONTROLS = control_points()
COMPACT_CONTROLS = _compact_controls()
COMPACT_PATHS = [
    f"M {C_O[0]:.2f} {C_O[1]:.2f} Q {c[0]:.2f} {c[1]:.2f} {C_D[0]:.2f} {C_D[1]:.2f}"
    for c in COMPACT_CONTROLS
]


def qpoint(p0, c, p1, t):
    mt = 1.0 - t
    return (mt * mt * p0[0] + 2 * mt * t * c[0] + t * t * p1[0],
            mt * mt * p0[1] + 2 * mt * t * c[1] + t * t * p1[1])


def path_d(c):
    return (f"M {O[0]:.2f} {O[1]:.2f} Q {c[0]:.2f} {c[1]:.2f} "
            f"{D[0]:.2f} {D[1]:.2f}")


PATHS = [path_d(c) for c in CONTROLS]

# --------------------------------------------------------------------------
# SVG
# --------------------------------------------------------------------------
DEFS = f"""  <defs>
    <linearGradient id="saDisc" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#1b2942"/>
      <stop offset="1" stop-color="#0a1120"/>
    </linearGradient>
    <linearGradient id="saRing" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#22c55e"/>
      <stop offset="1" stop-color="#3b82f6"/>
    </linearGradient>
    <linearGradient id="saFan" gradientUnits="userSpaceOnUse"
                    x1="{O[0]}" y1="{O[1]}" x2="{D[0]}" y2="{D[1]}">
      <stop offset="0" stop-color="#60a5fa"/>
      <stop offset="1" stop-color="#22c55e"/>
    </linearGradient>
  </defs>
"""

MARK_BODY = f"""  <circle cx="{CX}" cy="{CY}" r="{DISC_R}" fill="url(#saDisc)"/>
  <circle cx="{CX}" cy="{CY}" r="{RING_R}" fill="none"
          stroke="url(#saRing)" stroke-width="{RING_W}"/>
  <g fill="none" stroke="url(#saFan)" stroke-width="{FAN_W}" stroke-linecap="round">
{chr(10).join(f'    <path d="{p}"/>' for p in PATHS)}
  </g>
  <circle cx="{D[0]}" cy="{D[1]}" r="{DOT_DST_GLOW}" fill="#22c55e" opacity="0.18"/>
  <circle cx="{D[0]}" cy="{D[1]}" r="{DOT_DST_R}" fill="#22c55e"/>
  <circle cx="{O[0]}" cy="{O[1]}" r="{DOT_SRC_R}" fill="#60a5fa"/>
"""


COMPACT_BODY = f"""  <circle cx="{CX}" cy="{CY}" r="{DISC_R}" fill="url(#saDisc)"/>
  <circle cx="{CX}" cy="{CY}" r="{RING_R}" fill="none"
          stroke="url(#saRing)" stroke-width="{RING_W}"/>
  <g fill="none" stroke="url(#saCFan)" stroke-width="{C_FAN_W}" stroke-linecap="round">
{chr(10).join(f'    <path d="{p}"/>' for p in COMPACT_PATHS)}
  </g>
  <circle cx="{C_D[0]}" cy="{C_D[1]}" r="{C_DOT_DST_R}" fill="#22c55e"/>
  <circle cx="{C_O[0]}" cy="{C_O[1]}" r="{C_DOT_SRC_R}" fill="#60a5fa"/>
"""


def write_favicon_svg():
    """The tab icon uses the COMPACT geometry - it must survive 16px."""
    defs = (DEFS.replace("</defs>",
            '  <linearGradient id="saCFan" gradientUnits="userSpaceOnUse" '
            f'x1="{C_O[0]}" y1="{C_O[1]}" x2="{C_D[0]}" y2="{C_D[1]}">\n'
            '      <stop offset="0" stop-color="#60a5fa"/>\n'
            '      <stop offset="1" stop-color="#22c55e"/>\n'
            '    </linearGradient>\n  </defs>'))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
           f'width="64" height="64" role="img" aria-label="Simulation Arena">\n'
           f"{defs}{COMPACT_BODY}</svg>\n")
    (PUB / "favicon.svg").write_text(svg, encoding="utf-8")


def write_logo_svg():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 236 64" width="236" height="64"
     role="img" aria-label="Simulation Arena">
{DEFS}  <g transform="translate(0,0)">
{MARK_BODY}  </g>
  <text x="80" y="42" font-family="Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
        font-size="27" font-weight="700" letter-spacing="-0.4">
    <tspan fill="#e2e8f0">Simulation</tspan><tspan fill="url(#saRing)"> Arena</tspan>
  </text>
</svg>
"""
    (PUB / "logo.svg").write_text(svg, encoding="utf-8")


def write_brand_component():
    jsx = f'''// Autogenerated by tools/make_brand.py - do not edit by hand.
// Inline so the mark inherits CSS sizing and needs no network request.
import React from 'react';

const PATHS = {PATHS!r};

export default function BrandMark({{ size = 28, className = '', label = 'Simulation Arena' }}) {{
  return (
    <svg
      className={{'brand-mark ' + className}}
      width={{size}}
      height={{size}}
      viewBox="0 0 64 64"
      role="img"
      aria-label={{label}}
      focusable="false"
    >
      <defs>
        <linearGradient id="saDisc" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#1b2942" />
          <stop offset="1" stopColor="#0a1120" />
        </linearGradient>
        <linearGradient id="saRing" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#22c55e" />
          <stop offset="1" stopColor="#3b82f6" />
        </linearGradient>
        <linearGradient id="saFan" gradientUnits="userSpaceOnUse"
                        x1="{O[0]}" y1="{O[1]}" x2="{D[0]}" y2="{D[1]}">
          <stop offset="0" stopColor="#60a5fa" />
          <stop offset="1" stopColor="#22c55e" />
        </linearGradient>
      </defs>
      <circle cx="{CX}" cy="{CY}" r="{DISC_R}" fill="url(#saDisc)" />
      <circle cx="{CX}" cy="{CY}" r="{RING_R}" fill="none"
              stroke="url(#saRing)" strokeWidth="{RING_W}" />
      <g fill="none" stroke="url(#saFan)" strokeWidth="{FAN_W}" strokeLinecap="round">
        {{PATHS.map((d, i) => (
          <path key={{i}} d={{d}} />
        ))}}
      </g>
      <circle cx="{D[0]}" cy="{D[1]}" r="{DOT_DST_GLOW}" fill="#22c55e" opacity="0.18" />
      <circle cx="{D[0]}" cy="{D[1]}" r="{DOT_DST_R}" fill="#22c55e" />
      <circle cx="{O[0]}" cy="{O[1]}" r="{DOT_SRC_R}" fill="#60a5fa" />
    </svg>
  );
}}
'''
    (SRC / "BrandMark.jsx").write_text(jsx, encoding="utf-8")


# --------------------------------------------------------------------------
# raster (supersampled, then downscaled - crisp at every size)
# --------------------------------------------------------------------------
SS = 8                      # supersample factor
UNIT = 64
BASE = UNIT * SS           # 512


def lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def rgba_from(alpha, color):
    """alpha float array (H,W) -> RGBA PIL image with a flat colour."""
    h, w = alpha.shape
    a = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.new("RGBA", (w, h), color + (0,))
    img.putalpha(Image.fromarray(a))
    return img


def render_icon(size=BASE, compact=False):
    """compact=True uses the thickened small-size geometry (for ico/small png)."""
    if compact:
        return _render_compact(size)
    k = size / UNIT
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    ux, uy = xs / k, ys / k
    dist = np.hypot(ux - CX, uy - CY)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # --- disc: vertical gradient, antialiased circular edge -----------------
    t = np.clip((uy - (CY - DISC_R)) / (2 * DISC_R), 0.0, 1.0)
    base = np.zeros((size, size, 3), dtype=np.float64)
    for i in range(3):
        base[:, :, i] = DISC_TOP[i] + (DISC_BOT[i] - DISC_TOP[i]) * t
    edge = np.clip((DISC_R - dist) * k + 0.5, 0.0, 1.0)
    disc = Image.new("RGBA", (size, size))
    disc.paste(Image.fromarray(base.astype(np.uint8), "RGB"), (0, 0))
    disc.putalpha(Image.fromarray((edge * 255).astype(np.uint8)))
    canvas = Image.alpha_composite(canvas, disc)

    # --- ring: annulus, colour sweeps green -> blue -------------------------
    inner, outer = RING_R - RING_W / 2, RING_R + RING_W / 2
    band = np.clip(np.minimum(dist - inner, outer - dist) * k + 0.5, 0.0, 1.0)
    s = np.clip(((ux - CX) + (uy - CY)) / (2 * DISC_R) + 0.5, 0.0, 1.0)
    ring_rgb = np.zeros((size, size, 3), dtype=np.float64)
    for i in range(3):
        ring_rgb[:, :, i] = GREEN[i] + (BLUE[i] - GREEN[i]) * s
    ring = Image.new("RGBA", (size, size))
    ring.paste(Image.fromarray(ring_rgb.astype(np.uint8), "RGB"), (0, 0))
    ring.putalpha(Image.fromarray((band * 255).astype(np.uint8)))
    canvas = Image.alpha_composite(canvas, ring)

    # --- fan: stamp round dots along each bezier (gives round caps) ---------
    def stamp(img, cx, cy, r, color, alpha=1.0):
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = cx - r, cy - r, cx + r, cy + r
        if alpha >= 1.0:
            d.ellipse([x0, y0, x1, y1], fill=color + (255,))
        else:
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(layer).ellipse([x0, y0, x1, y1], fill=color + (255,))
            layer.putalpha(Image.fromarray(
                (np.asarray(layer.split()[3], dtype=np.float32) * alpha).astype(np.uint8)))
            img.paste(Image.alpha_composite(img, layer).split()[3], (0, 0))
            img.paste(layer, (0, 0), layer)

    fan_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    steps = 900
    for c in CONTROLS:
        rr = FAN_W / 2.0 * k
        for j in range(steps + 1):
            tt = j / steps
            px, py = qpoint(O, c, D, tt)
            # colour travels blue -> green along the path
            col = lerp(BLUE_LIGHT, GREEN, tt)
            stamp(fan_layer, px * k, py * k, rr, tuple(int(round(v)) for v in col))
    canvas = Image.alpha_composite(canvas, fan_layer)

    # --- answer dot: glow then solid ---------------------------------------
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    stamp(glow, D[0] * k, D[1] * k, DOT_DST_GLOW * k, GREEN, alpha=0.18)
    canvas = Image.alpha_composite(canvas, glow)

    dots = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    stamp(dots, D[0] * k, D[1] * k, DOT_DST_R * k, GREEN)
    stamp(dots, O[0] * k, O[1] * k, DOT_SRC_R * k, BLUE_LIGHT)
    canvas = Image.alpha_composite(canvas, dots)

    return canvas


def _render_compact(size: int):
    """Same idea as render_icon but with the bold small-size geometry."""
    from PIL import ImageDraw as _ID

    k = size / UNIT
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # disc
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    ux, uy = xs / k, ys / k
    dist = np.hypot(ux - CX, uy - CY)
    t = np.clip((uy - (CY - DISC_R)) / (2 * DISC_R), 0.0, 1.0)
    base = np.zeros((size, size, 3), dtype=np.float64)
    for i in range(3):
        base[:, :, i] = DISC_TOP[i] + (DISC_BOT[i] - DISC_TOP[i]) * t
    edge = np.clip((DISC_R - dist) * k + 0.5, 0.0, 1.0)
    disc = Image.new("RGBA", (size, size))
    disc.paste(Image.fromarray(base.astype(np.uint8), "RGB"), (0, 0))
    disc.putalpha(Image.fromarray((edge * 255).astype(np.uint8)))
    canvas = Image.alpha_composite(canvas, disc)

    # ring
    inner, outer = RING_R - RING_W / 2, RING_R + RING_W / 2
    band = np.clip(np.minimum(dist - inner, outer - dist) * k + 0.5, 0.0, 1.0)
    s = np.clip(((ux - CX) + (uy - CY)) / (2 * DISC_R) + 0.5, 0.0, 1.0)
    ring_rgb = np.zeros((size, size, 3), dtype=np.float64)
    for i in range(3):
        ring_rgb[:, :, i] = GREEN[i] + (BLUE[i] - GREEN[i]) * s
    ring = Image.new("RGBA", (size, size))
    ring.paste(Image.fromarray(ring_rgb.astype(np.uint8), "RGB"), (0, 0))
    ring.putalpha(Image.fromarray((band * 255).astype(np.uint8)))
    canvas = Image.alpha_composite(canvas, ring)

    # fan - thick, two strands
    fan = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dr = _ID.Draw(fan)
    rr = C_FAN_W / 2.0 * k
    for c in COMPACT_CONTROLS:
        for j in range(600):
            tt = j / 599.0
            px, py = qpoint(C_O, c, C_D, tt)
            col = tuple(int(round(v)) for v in
                        (BLUE_LIGHT[i] + (GREEN[i] - BLUE_LIGHT[i]) * tt for i in range(3)))
            dr.ellipse([px * k - rr, py * k - rr, px * k + rr, py * k + rr], fill=col + (255,))
    canvas = Image.alpha_composite(canvas, fan)

    dots = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dd = _ID.Draw(dots)
    for cx_, cy_, r_, col in ((C_D[0], C_D[1], C_DOT_DST_R, GREEN),
                              (C_O[0], C_O[1], C_DOT_SRC_R, BLUE_LIGHT)):
        rad = r_ * k
        dd.ellipse([cx_ * k - rad, cy_ * k - rad, cx_ * k + rad, cy_ * k + rad],
                   fill=tuple(int(v) for v in col) + (255,))
    return Image.alpha_composite(canvas, dots)


def write_rasters():
    big = render_icon(BASE)                      # detailed, for large sizes
    small = render_icon(BASE, compact=True)      # bold, for tiny sizes

    big.resize((512, 512), Image.LANCZOS).save(PUB / "icon-512.png")
    big.resize((192, 192), Image.LANCZOS).save(PUB / "icon-192.png")
    big.resize((180, 180), Image.LANCZOS).save(PUB / "apple-touch-icon.png")

    # the .ico carries real per-size bitmaps drawn from the COMPACT geometry,
    # so 16px is crisp instead of a downscaled blur.
    small.resize((16, 16), Image.LANCZOS).save(
        PUB / "favicon.ico", format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48)])
    imgs = [small.resize((16 + 8 * i, 16 + 8 * i), Image.LANCZOS) for i in range(7)]
    imgs[0].save(PUB / "favicon.ico", format="ICO",
                 append_images=imgs[1:], sizes=[im.size for im in imgs])

    big.resize((256, 256), Image.LANCZOS).save(PUB / "_preview.png")


def write_manifest():
    (PUB / "manifest.webmanifest").write_text(
        """{
  "name": "Simulation Arena",
  "short_name": "Sim Arena",
  "description": "Answer business questions with numbers, in plain English.",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0b1220",
  "theme_color": "#0b1220",
  "icons": [
    { "src": "/favicon.svg", "sizes": "any", "type": "image/svg+xml" },
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable" }
  ]
}
""", encoding="utf-8")


if __name__ == "__main__":
    write_favicon_svg()
    write_logo_svg()
    write_brand_component()
    write_rasters()
    write_manifest()
    print("brand written to", PUB)
    for f in sorted(PUB.iterdir()):
        print(f"  {f.name:<24} {f.stat().st_size:>8,} bytes")
    print("  components/BrandMark.jsx", (SRC / "BrandMark.jsx").stat().st_size, "bytes")
