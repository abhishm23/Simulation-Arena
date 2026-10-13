import React from 'react';

/**
 * Hand-rolled SVG charts. No charting library, no canvas: deterministic markup
 * that scales with the container via viewBox.
 *
 * Supported: line, bar, grouped_bar, hist, stacked_area, heatmap, funnel,
 * scatter, waterfall. Any unknown type falls back to a plain bar chart of the
 * first series, so a new chart type from the backend can never blank the page.
 */

const PALETTE = ['#22c55e', '#3b82f6', '#f59e0b', '#a78bfa', '#ef4444', '#38bdf8', '#f472b6'];

const W = 680;
const H = 360;
const PAD = { l: 64, r: 22, t: 20, b: 50 };

const BG = '#0f172a';
const GRID = '#1e293b';
const TEXT = '#e2e8f0';
const DIM = '#94a3b8';
const FAINT = '#64748b';

// ---------------------------------------------------------------------------
// numbers
// ---------------------------------------------------------------------------

function nums(list) {
  const out = [];
  (list || []).forEach((v) => {
    const n = Number(v);
    if (isFinite(n)) out.push(n);
  });
  return out;
}

function unitOf(label) {
  const s = String(label || '').toLowerCase();
  if (s.includes('%') || /chance|probab|rate|share|percent|margin %/.test(s)) return 'pct';
  if (
    s.includes('$') ||
    /profit|revenue|cost|price|value|margin|arr|spend|saving|cash|ebitda|discount \$|budget/.test(s)
  ) {
    return 'money';
  }
  return 'num';
}

function compact(n, dp) {
  const a = Math.abs(n);
  if (a >= 1e9) return `${(n / 1e9).toFixed(dp ?? 1)}B`;
  if (a >= 1e6) return `${(n / 1e6).toFixed(dp ?? 1)}M`;
  if (a >= 1e3) return `${(n / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`;
  if (a >= 100) return n.toFixed(0);
  if (a >= 1) return n.toFixed(dp ?? 1);
  if (a === 0) return '0';
  return n.toFixed(2);
}

/** "1.2M" -> "$1.2M" -> "45%" depending on what the axis is measuring. */
function fmt(v, unit, isFraction) {
  const n = Number(v);
  if (!isFinite(n)) return '';
  if (unit === 'pct') {
    const pct = isFraction ? n * 100 : n;
    const dp = Math.abs(pct) < 10 && !Number.isInteger(pct) ? 1 : 0;
    return `${pct.toFixed(dp)}%`;
  }
  if (unit === 'money') {
    const sign = n < 0 ? '-' : '';
    return `${sign}$${compact(Math.abs(n), Math.abs(n) < 10 ? 2 : 1)}`;
  }
  const sign = n < 0 ? '-' : '';
  return `${sign}${compact(Math.abs(n), Math.abs(n) < 10 ? 2 : 1)}`;
}

function niceTicks(min, max, count = 5) {
  if (!isFinite(min) || !isFinite(max)) return [0];
  let lo = min;
  let hi = max;
  if (lo === hi) {
    lo = lo === 0 ? 0 : lo * 0.9;
    hi = hi === 0 ? 1 : hi * 1.1;
  }
  const rawStep = (hi - lo) / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep || 1)));
  const norm = rawStep / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const start = Math.floor(lo / step) * step;
  const end = Math.ceil(hi / step) * step;
  const out = [];
  for (let v = start; v <= end + step * 1e-9; v += step) {
    out.push(Math.round(v * 1e9) / 1e9);
  }
  return out;
}

// ---------------------------------------------------------------------------
// scales / frame
// ---------------------------------------------------------------------------

function makeFrame(opts) {
  const p = Object.assign({ l: PAD.l }, opts.pad || {});
  const pad = { ...PAD, ...p };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const ymin = opts.ymin;
  const ymax = opts.ymax === opts.ymin ? opts.ymin + 1 : opts.ymax;
  const y = (v) => pad.t + ih * (1 - (Number(v) - ymin) / (ymax - ymin));
  return { pad, iw, ih, y, ymin, ymax };
}

function xLabelsOf(chart, n) {
  const xs = (chart && chart.x) || [];
  const out = [];
  for (let i = 0; i < n; i += 1) {
    const v = xs[i];
    out.push(v === undefined || v === null ? String(i + 1) : String(v));
  }
  return out;
}

function numericX(xs, n) {
  if (n === 0) return false;
  let count = 0;
  for (let i = 0; i < n; i += 1) {
    const v = Number(xs[i]);
    if (xs[i] !== '' && xs[i] != null && isFinite(v)) count += 1;
  }
  return count === n;
}

/** Evenly spaced labels, thinned so they never overlap. */
function thinLabels(labels, maxLabels) {
  const n = labels.length;
  if (n <= maxLabels) return labels.map((l, i) => ({ i, l }));
  const stride = Math.ceil(n / maxLabels);
  const out = [];
  for (let i = 0; i < n; i += stride) out.push({ i, l: labels[i] });
  if (out[out.length - 1].i !== n - 1) out.push({ i: n - 1, l: labels[n - 1] });
  return out;
}

function ellipsis(s, max) {
  const str = String(s);
  return str.length > max ? `${str.slice(0, max - 1)}…` : str;
}

// ---------------------------------------------------------------------------
// small svg pieces
// ---------------------------------------------------------------------------

function YAxis({ frame, unit, isFraction }) {
  const ticks = niceTicks(frame.ymin, frame.ymax, 5);
  return (
    <g>
      {ticks.map((t) => (
        <g key={t}>
          <line
            x1={frame.pad.l}
            x2={W - frame.pad.r}
            y1={frame.y(t)}
            y2={frame.y(t)}
            stroke={GRID}
            strokeWidth={1}
          />
          <text
            x={frame.pad.l - 9}
            y={frame.y(t) + 4}
            fill={FAINT}
            fontSize={11}
            textAnchor="end"
          >
            {fmt(t, unit, isFraction)}
          </text>
        </g>
      ))}
    </g>
  );
}

function XAxis({ frame, labels, mode, xs }) {
  const shown = thinLabels(labels, mode === 'numeric' ? 9 : 8);
  return (
    <g>
      <line
        x1={frame.pad.l}
        x2={W - frame.pad.r}
        y1={frame.pad.t + frame.ih}
        y2={frame.pad.t + frame.ih}
        stroke={GRID}
        strokeWidth={1}
      />
      {shown.map(({ i, l }) => {
        let cx;
        if (mode === 'numeric') {
          const xmin = Number(xs[0]);
          const xmax = Number(xs[xs.length - 1]);
          cx = frame.pad.l + frame.iw * ((Number(xs[i]) - xmin) / (xmax - xmin || 1));
        } else {
          cx = frame.pad.l + frame.iw * ((i + 0.5) / labels.length);
        }
        return (
          <text
            key={`${i}-${l}`}
            x={cx}
            y={frame.pad.t + frame.ih + 20}
            fill={FAINT}
            fontSize={11}
            textAnchor="middle"
          >
            <title>{String(l)}</title>
            {ellipsis(l, 14)}
          </text>
        );
      })}
    </g>
  );
}

function AxisTitles({ frame, xLabel, yLabel }) {
  return (
    <g>
      {xLabel && (
        <text
          x={frame.pad.l + frame.iw / 2}
          y={H - 8}
          fill={DIM}
          fontSize={11.5}
          textAnchor="middle"
        >
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text
          x={-(frame.pad.t + frame.ih / 2)}
          y={15}
          fill={DIM}
          fontSize={11.5}
          textAnchor="middle"
          transform="rotate(-90)"
        >
          {yLabel}
        </text>
      )}
    </g>
  );
}

function Annotations({ frame, annotations, labels, xs, mode, unit, isFraction }) {
  // QA NEW-11: every annotation used to be drawn at the same baseline with a
  // fixed 9px nudge, so when a band collapsed - several annotations landing on
  // the same point - their labels printed straight on top of each other and
  // only the last one could be read. Work out every position first, then stack
  // apart the ones that would collide.
  const placed = [];
  (annotations || []).forEach((a, i) => {
    let cx;
    let idx = -1;
    if (mode === 'numeric') {
      const xmin = Number(xs[0]);
      const xmax = Number(xs[xs.length - 1]);
      const v = Number(a.x);
      if (!isFinite(v)) return;
      cx = frame.pad.l + frame.iw * ((v - xmin) / (xmax - xmin || 1));
    } else {
      idx = labels.findIndex((l) => String(l) === String(a.x));
      if (idx < 0) idx = Number(a.x);
      if (!isFinite(idx) || idx < 0) return;
      cx = frame.pad.l + frame.iw * ((idx + 0.5) / labels.length);
    }
    const cy = frame.y(Number(a.y));
    if (!isFinite(cy)) return;
    placed.push({ key: i, text: String(a.text ?? ''), cx, cy, labelY: cy - 10 });
  });

  // QA NEW-11: the room one line of label actually needs. The chart is a
  // viewBox scaled up to the card, so a 11.5-unit font renders about 1.54
  // units per pixel - a 13-unit step was 20px against a 22px-tall label and
  // the two still clipped by 2px. 16 units clears it with a little air.
  const SLOT = 16;
  const halfWidth = (t) => Math.max(26, String(t).length * 3.4);
  placed.forEach((p) => {
    let slot = 0;
    const hits = (s) =>
      placed.some(
        (u) =>
          u !== p &&
          u.slot !== undefined &&
          Math.abs(u.cx - p.cx) < halfWidth(u.text) + halfWidth(p.text) &&
          Math.abs(u.labelY - (p.cy - 10 - s * SLOT)) < SLOT,
      );
    while (slot < 9 && hits(slot)) slot += 1;
    p.slot = slot;
    p.labelY = p.cy - 10 - slot * SLOT;
  });

  return (
    <g>
      {placed.map((a) => {
        const cx = a.cx;
        const cy = a.cy;
        const anchor = cx > W * 0.72 ? 'end' : 'start';
        const dx = anchor === 'end' ? -9 : 9;
        return (
          <g key={a.key}>
            <circle cx={cx} cy={cy} r={5.5} fill="none" stroke={TEXT} strokeWidth={1.6} />
            <circle cx={cx} cy={cy} r={2.6} fill={TEXT} />
            <text
              x={cx + dx}
              y={a.labelY}
              fill={TEXT}
              fontSize={11.5}
              fontWeight={700}
              textAnchor={anchor}
            >
              {a.text}
            </text>
            <title>{`${a.text} (${fmt(a.y, unit, isFraction)})`}</title>
          </g>
        );
      })}
    </g>
  );
}

// ---------------------------------------------------------------------------
// chart bodies
// ---------------------------------------------------------------------------

function LineBody({ chart, frame, unit, isFraction, labels, xs, mode, series, palette }) {
  const xAt = (i) =>
    mode === 'numeric'
      ? frame.pad.l +
        frame.iw *
          ((Number(xs[i]) - Number(xs[0])) /
            (Number(xs[xs.length - 1]) - Number(xs[0]) || 1))
      : frame.pad.l + frame.iw * ((i + 0.5) / labels.length);

  return (
    <g>
      {series.map((s, si) => {
        const vals = nums(s.values);
        const color = s.color || palette[si % palette.length];
        const d = vals
          .map((v, i) => `${i === 0 ? 'M' : 'L'}${xAt(i).toFixed(2)},${frame.y(v).toFixed(2)}`)
          .join(' ');
        return (
          <g key={si}>
            <path
              d={d}
              fill="none"
              stroke={color}
              strokeWidth={2.4}
              strokeDasharray={s.dash ? '6 4' : undefined}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {vals.map((v, i) => (
              <circle key={i} cx={xAt(i)} cy={frame.y(v)} r={7} fill="transparent">
                <title>{`${s.name}: ${fmt(v, unit, isFraction)} at ${labels[i]}`}</title>
              </circle>
            ))}
          </g>
        );
      })}
      <Annotations
        frame={frame}
        annotations={chart.annotations}
        labels={labels}
        xs={xs}
        mode={mode}
        unit={unit}
        isFraction={isFraction}
      />
    </g>
  );
}

function BarBody({ chart, frame, unit, isFraction, labels, series, palette }) {
  const n = labels.length;
  const groups = series.length || 1;
  const slot = frame.iw / Math.max(1, n);
  const bw = Math.max(3, (slot * 0.78) / groups);
  return (
    <g>
      {series.map((s, si) => {
        const color = s.color || palette[si % palette.length];
        return nums(s.values).map((v, i) => {
          const x0 = frame.pad.l + slot * i + (slot - bw * groups) / 2 + bw * si;
          const yTop = frame.y(Math.max(0, v));
          const yBase = frame.y(Math.min(0, v));
          return (
            <rect
              key={`${si}-${i}`}
              x={x0}
              y={Math.min(yTop, yBase)}
              width={bw}
              height={Math.max(1.5, Math.abs(yBase - yTop))}
              fill={color}
              rx={Math.min(3, bw / 3)}
            >
              <title>{`${s.name} — ${labels[i]}: ${fmt(v, unit, isFraction)}`}</title>
            </rect>
          );
        });
      })}
      <Annotations
        frame={frame}
        annotations={chart.annotations}
        labels={labels}
        xs={chart.x || []}
        mode="band"
        unit={unit}
        isFraction={isFraction}
      />
    </g>
  );
}

function StackedAreaBody({ frame, unit, isFraction, labels, series, palette }) {
  const n = labels.length;
  const cum = new Array(n).fill(0);
  const xAt = (i) => frame.pad.l + frame.iw * (n === 1 ? 0.5 : i / (n - 1));
  const layers = [];
  series.forEach((s, si) => {
    const vals = nums(s.values);
    const upper = [];
    for (let i = 0; i < n; i += 1) {
      cum[i] += vals[i] || 0;
      upper.push(cum[i]);
    }
    layers.push({ s, si, vals, upper });
  });

  return (
    <g>
      {layers.map(({ s, si, vals, upper }) => {
        const color = s.color || palette[si % palette.length];
        const top = upper.map((v, i) => `${xAt(i).toFixed(2)},${frame.y(v).toFixed(2)}`);
        const bottom = upper
          .map((v, i) => `${xAt(i).toFixed(2)},${frame.y(v - (vals[i] || 0)).toFixed(2)}`)
          .reverse();
        return (
          <path
            key={si}
            d={`M${top.join(' L')} L${bottom.join(' L')} Z`}
            fill={color}
            fillOpacity={0.62}
            stroke={color}
            strokeWidth={1.2}
          >
            <title>{s.name}</title>
          </path>
        );
      })}
      {labels.map((l, i) => (
        <circle key={i} cx={xAt(i)} cy={frame.y(cum[i])} r={7} fill="transparent">
          <title>{`${l}: ${fmt(cum[i], unit, isFraction)} in total`}</title>
        </circle>
      ))}
    </g>
  );
}

function histogram(values, bins = 26) {
  const clean = nums(values);
  if (!clean.length) return { counts: [], edges: [] };
  let lo = Math.min(...clean);
  let hi = Math.max(...clean);
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const width = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  clean.forEach((v) => {
    let idx = Math.floor((v - lo) / width);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    counts[idx] += 1;
  });
  const edges = [];
  for (let i = 0; i < bins; i += 1) edges.push(lo + width * (i + 0.5));
  return { counts, edges, lo, hi, width };
}

function HistBody({ frame, unit, isFraction, series }) {
  const all = [];
  series.forEach((s) => nums(s.values).forEach((v) => all.push(v)));
  const { counts, edges } = histogram(all, 26);
  const ymax = Math.max(1, ...counts);
  const inner = makeFrame({ ymin: 0, ymax });
  const n = counts.length;
  const slot = frame.iw / Math.max(1, n);
  const bw = Math.max(2, slot * 0.86);
  return (
    <g>
      {counts.map((c, i) => {
        const x0 = frame.pad.l + slot * i + (slot - bw) / 2;
        const yv = inner.y(c);
        return (
          <rect
            key={i}
            x={x0}
            y={yv}
            width={bw}
            height={Math.max(1, frame.pad.t + frame.ih - yv)}
            fill={series[0] && series[0].color ? series[0].color : PALETTE[0]}
            fillOpacity={0.85}
            rx={2}
          >
            <title>{`${fmt(edges[i], unit, isFraction)}: ${c} outcomes`}</title>
          </rect>
        );
      })}
      {counts.map((c, i) => (
        <text
          key={`t${i}`}
          x={frame.pad.l + frame.iw * ((i + 0.5) / n)}
          y={frame.pad.t + frame.ih + 20}
          fill={FAINT}
          fontSize={10}
          textAnchor="middle"
        >
          {i % Math.ceil(n / 8) === 0 ? fmt(edges[i], unit, isFraction) : ''}
        </text>
      ))}
    </g>
  );
}

const RAMP = [
  [0.0, [19, 32, 56]],
  [0.25, [29, 78, 216]],
  [0.5, [34, 197, 94]],
  [0.75, [245, 158, 11]],
  [1.0, [239, 68, 68]],
];

function rampColor(t) {
  const x = Math.min(1, Math.max(0, t));
  for (let i = 0; i < RAMP.length - 1; i += 1) {
    const [p0, c0] = RAMP[i];
    const [p1, c1] = RAMP[i + 1];
    if (x >= p0 && x <= p1) {
      const f = p1 === p0 ? 0 : (x - p0) / (p1 - p0);
      const c = c0.map((v, j) => Math.round(v + (c1[j] - v) * f));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return 'rgb(239,68,68)';
}

function HeatmapBody({ chart, series, labels, unit, isFraction }) {
  const pad = { l: 118, r: 22, t: 22, b: 46 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const rows = series.length || 1;
  const cols = Math.max(1, labels.length);
  const cw = iw / cols;
  const ch = ih / rows;

  let lo = Infinity;
  let hi = -Infinity;
  series.forEach((s) =>
    nums(s.values).forEach((v) => {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }),
  );
  if (!isFinite(lo) || !isFinite(hi)) {
    lo = 0;
    hi = 1;
  }
  if (lo === hi) hi = lo + 1;

  const cells = [];
  series.forEach((s, ri) => {
    const vals = nums(s.values);
    for (let i = 0; i < cols; i += 1) {
      const v = vals[i];
      if (v === undefined) continue;
      cells.push({ ri, i, v });
    }
  });

  return (
    <g>
      {cells.map(({ ri, i, v }) => {
        const t = (v - lo) / (hi - lo || 1);
        const x = pad.l + cw * i;
        const y = pad.t + ch * ri;
        return (
          <g key={`${ri}-${i}`}>
            <rect
              x={x + 1}
              y={y + 1}
              width={Math.max(1, cw - 2)}
              height={Math.max(1, ch - 2)}
              fill={rampColor(t)}
              rx={3}
            >
              <title>{`${series[ri].name} × ${labels[i]}: ${fmt(v, unit, isFraction)}`}</title>
            </rect>
            {cw > 46 && ch > 22 && (
              <text
                x={x + cw / 2}
                y={y + ch / 2 + 4}
                fill="#f8fafc"
                fontSize={10.5}
                fontWeight={700}
                textAnchor="middle"
                pointerEvents="none"
              >
                {fmt(v, unit, isFraction)}
              </text>
            )}
          </g>
        );
      })}
      {series.map((s, ri) => (
        <text
          key={`r${ri}`}
          x={pad.l - 9}
          y={pad.t + ch * ri + ch / 2 + 4}
          fill={DIM}
          fontSize={11}
          textAnchor="end"
        >
          <title>{String(s.name)}</title>
          {ellipsis(s.name, 16)}
        </text>
      ))}
      {labels.map((l, i) => (
        <text
          key={`c${i}`}
          x={pad.l + cw * i + cw / 2}
          y={pad.t + ih + 18}
          fill={FAINT}
          fontSize={10.5}
          textAnchor="middle"
        >
          {/* QA UX-18: truncated labels lost their meaning ("Third-party L..." is
              ambiguous between Licensing and Logistics). Keep the full text
              available as a native tooltip. */}
          {cw >= 34 || i % 2 === 0 ? <title>{String(l)}</title> : null}
          {cw < 34 && i % 2 === 1 ? '' : ellipsis(l, 12)}
        </text>
      ))}
      {chart.x_label && (
        <text x={pad.l + iw / 2} y={H - 8} fill={DIM} fontSize={11.5} textAnchor="middle">
          {chart.x_label}
        </text>
      )}
    </g>
  );
}

function FunnelBody({ series, labels, unit, isFraction }) {
  const pad = { l: 22, r: 22, t: 22, b: 34 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const vals = nums(series[0] ? series[0].values : []);
  const top = Math.max(...vals, 1e-9);
  const rowH = ih / Math.max(1, vals.length);
  const barH = Math.min(46, rowH * 0.72);
  const color = (series[0] && series[0].color) || PALETTE[3];
  return (
    <g>
      {vals.map((v, i) => {
        const w = Math.max(6, (iw * Math.max(0, v)) / top);
        const x = pad.l + (iw - w) / 2;
        const y = pad.t + rowH * i + (rowH - barH) / 2;
        return (
          <g key={i}>
            <rect x={x} y={y} width={w} height={barH} rx={5} fill={color} fillOpacity={0.55 + 0.4 * (v / top)}>
              <title>{`${labels[i]}: ${fmt(v, unit, isFraction)}`}</title>
            </rect>
            <text
              x={pad.l + 6}
              y={y + barH / 2 + 4}
              fill={TEXT}
              fontSize={11.5}
              fontWeight={600}
            >
              <title>{String(labels[i])}</title>
              {ellipsis(labels[i], 20)}
            </text>
            <text
              x={W - pad.r - 6}
              y={y + barH / 2 + 4}
              fill={DIM}
              fontSize={11.5}
              textAnchor="end"
            >
              {fmt(v, unit, isFraction)}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function ScatterBody({ frame, unit, isFraction, labels, series, palette, xs }) {
  const pts = [];
  series.forEach((s, si) => {
    const vals = s.values || [];
    vals.forEach((v, i) => {
      if (Array.isArray(v)) {
        const [px, py] = v;
        if (isFinite(Number(px)) && isFinite(Number(py))) {
          pts.push({ x: Number(px), y: Number(py), s: si });
        }
      } else if (isFinite(Number(v))) {
        const px = isFinite(Number(xs[i])) ? Number(xs[i]) : i;
        pts.push({ x: px, y: Number(v), s: si, label: labels[i] });
      }
    });
  });
  if (!pts.length) return null;
  const xsN = pts.map((p) => p.x);
  const xmin = Math.min(...xsN);
  const xmax = Math.max(...xsN);
  const xAt = (v) => frame.pad.l + frame.iw * ((v - xmin) / (xmax - xmin || 1));
  return (
    <g>
      {pts.map((p, i) => (
        <circle
          key={i}
          cx={xAt(p.x)}
          cy={frame.y(p.y)}
          r={4.2}
          fill={series[p.s].color || palette[p.s % palette.length]}
          fillOpacity={0.78}
        >
          <title>{`${series[p.s].name}${p.label ? ` — ${p.label}` : ''}: ${fmt(
            p.x,
            'num',
            false,
          )} → ${fmt(p.y, unit, isFraction)}`}</title>
        </circle>
      ))}
    </g>
  );
}

function WaterfallBody({ frame, unit, isFraction, labels, series, palette }) {
  const vals = nums(series[0] ? series[0].values : []);
  const slot = frame.iw / Math.max(1, vals.length);
  const bw = Math.max(4, slot * 0.62);
  let running = 0;
  const bars = [];
  vals.forEach((v, i) => {
    const from = running;
    running += v;
    bars.push({ i, from, to: running, delta: v });
  });
  return (
    <g>
      {bars.map((b) => {
        const yTop = frame.y(Math.max(b.from, b.to));
        const yBot = frame.y(Math.min(b.from, b.to));
        const x = frame.pad.l + slot * b.i + (slot - bw) / 2;
        const color = b.delta >= 0 ? PALETTE[0] : PALETTE[4];
        return (
          <g key={b.i}>
            <rect
              x={x}
              y={yTop}
              width={bw}
              height={Math.max(1.5, yBot - yTop)}
              rx={3}
              fill={color}
              fillOpacity={0.85}
            >
              <title>{`${labels[b.i]}: ${b.delta >= 0 ? '+' : ''}${fmt(
                b.delta,
                unit,
                isFraction,
              )} → ${fmt(b.to, unit, isFraction)}`}</title>
            </rect>
            {b.i > 0 && (
              <line
                x1={frame.pad.l + slot * (b.i - 1) + slot / 2}
                x2={x + bw / 2}
                y1={frame.y(bars[b.i - 1].to)}
                y2={frame.y(b.from)}
                stroke={FAINT}
                strokeDasharray="3 3"
              />
            )}
          </g>
        );
      })}
    </g>
  );
}

// ---------------------------------------------------------------------------
// component
// ---------------------------------------------------------------------------

export default function ChartView({ chart }) {
  const c = chart || {};
  const type = String(c.type || '').toLowerCase();
  const series = (c.series || []).filter((s) => s && Array.isArray(s.values));
  const palette = PALETTE;

  const n = Math.max(
    (c.x || []).length,
    ...series.map((s) => (s.values || []).length),
    0,
  );
  const labels = xLabelsOf(c, n);
  const xs = c.x || [];
  const mode = numericX(xs, n) && n > 0 && type !== 'heatmap' ? 'numeric' : 'band';

  const unit = unitOf(`${c.y_label || ''} ${c.title || ''}`);

  // Choose a y domain that suits the shape.
  let flat = [];
  if (type === 'stacked_area') {
    const sums = new Array(n).fill(0);
    series.forEach((s) => {
      nums(s.values).forEach((v, i) => {
        sums[i] += v;
      });
    });
    flat = sums;
  } else if (type === 'hist') {
    const all = [];
    series.forEach((s) => nums(s.values).forEach((v) => all.push(v)));
    const { counts } = histogram(all, 26);
    flat = counts;
  } else {
    series.forEach((s) => nums(s.values).forEach((v) => flat.push(v)));
    (c.annotations || []).forEach((a) => {
      const v = Number(a.y);
      if (isFinite(v)) flat.push(v);
    });
  }

  let ymin = flat.length ? Math.min(...flat) : 0;
  let ymax = flat.length ? Math.max(...flat) : 1;
  const isFraction = unit === 'pct' && Math.max(Math.abs(ymin), Math.abs(ymax)) <= 1.5;

  if (type === 'hist' || type === 'bar' || type === 'grouped_bar' || type === 'stacked_area') {
    ymin = Math.min(0, ymin);
    ymax = ymax * (ymax > 0 ? 1.08 : 1);
  } else if (type === 'waterfall') {
    let running = 0;
    let lo = 0;
    let hi = 0;
    nums(series[0] ? series[0].values : []).forEach((v) => {
      running += v;
      lo = Math.min(lo, running);
      hi = Math.max(hi, running);
    });
    ymin = Math.min(0, lo);
    ymax = hi * 1.08 || 1;
  } else {
    const pad = (ymax - ymin || 1) * 0.1;
    ymin -= pad;
    ymax += pad;
  }
  if (ymin === ymax) {
    ymin = Math.min(0, ymin);
    ymax = ymin + 1;
  }

  const frame = makeFrame({
    ymin,
    ymax,
    pad: type === 'funnel' ? { l: 22, r: 22, t: 22, b: 34 } : undefined,
  });

  let body = null;
  let showAxes = true;

  if (!n || !series.length) {
    body = (
      <text x={W / 2} y={H / 2} fill={FAINT} fontSize={13} textAnchor="middle">
        No numbers to draw for this chart.
      </text>
    );
    showAxes = false;
  } else if (type === 'heatmap') {
    body = <HeatmapBody chart={c} series={series} labels={labels} unit={unit} isFraction={isFraction} />;
    showAxes = false;
  } else if (type === 'funnel') {
    body = <FunnelBody series={series} labels={labels} unit={unit} isFraction={isFraction} />;
    showAxes = false;
  } else if (type === 'hist') {
    body = <HistBody frame={frame} unit={unit} isFraction={isFraction} series={series} />;
  } else if (type === 'stacked_area') {
    body = (
      <StackedAreaBody
        frame={frame}
        unit={unit}
        isFraction={isFraction}
        labels={labels}
        series={series}
        palette={palette}
      />
    );
  } else if (type === 'scatter') {
    body = (
      <ScatterBody
        frame={frame}
        unit={unit}
        isFraction={isFraction}
        labels={labels}
        series={series}
        palette={palette}
        xs={xs}
      />
    );
  } else if (type === 'waterfall') {
    body = (
      <WaterfallBody
        frame={frame}
        unit={unit}
        isFraction={isFraction}
        labels={labels}
        series={series}
        palette={palette}
      />
    );
  } else if (type === 'line') {
    body = (
      <LineBody
        chart={c}
        frame={frame}
        unit={unit}
        isFraction={isFraction}
        labels={labels}
        xs={xs}
        mode={mode}
        series={series}
        palette={palette}
      />
    );
  } else {
    // bar, grouped_bar, and anything we do not recognise yet.
    body = (
      <BarBody
        chart={c}
        frame={frame}
        unit={unit}
        isFraction={isFraction}
        labels={labels}
        series={series}
        palette={palette}
      />
    );
  }

  const multi = series.length > 1 && type !== 'heatmap';

  return (
    <div className="chart-card">
      <div className="chart-title">{c.title || 'Chart'}</div>
      {c.subtitle && <div className="chart-sub">{c.subtitle}</div>}

      <svg
        className="chart-svg"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={c.title || 'chart'}
        style={{ background: BG, borderRadius: 12 }}
      >
        {showAxes && (
          <>
            <YAxis frame={frame} unit={unit} isFraction={isFraction} />
            {/* QA UX-10: the histogram body draws its own bucket labels on this
                baseline; rendering the shared XAxis as well put two label sets
                in the same place and they collided into unreadable strings. */}
            {type === 'hist' ? null : (
              <XAxis frame={frame} labels={labels} mode={mode} xs={xs} />
            )}
            <AxisTitles frame={frame} xLabel={c.x_label} yLabel={c.y_label} />
          </>
        )}
        {body}
      </svg>

      {multi && (
        <div className="legend">
          {series.map((s, i) => (
            <span className="legend-item" key={i}>
              <span
                className="legend-swatch"
                style={{ background: s.color || palette[i % palette.length] }}
              />
              {s.name}
            </span>
          ))}
        </div>
      )}

      {c.plain_meaning && (
        <div className="chart-caption">
          <span className="chart-caption-label">What this means</span>
          {c.plain_meaning}
        </div>
      )}
    </div>
  );
}
