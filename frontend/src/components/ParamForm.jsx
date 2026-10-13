import React, { useEffect, useMemo, useState } from 'react';
import HelpTip from './HelpTip.jsx';

/** Each group gets one plain-English sentence so the screen reads like a story. */
const GROUP_LINES = {
  'the deal': 'First, the deal itself — what it is worth and what it costs you.',
  'buyer behaviour': 'How the buyer reacts to price. Use your judgement if you have no history.',
  'buyer behavior': 'How the buyer reacts to price. Use your judgement if you have no history.',
  uncertainty: 'How much things might move. Bigger numbers mean we test wilder futures.',
  guardrails: 'The rules you must not break. We flag anything that crosses them.',
  engine: 'How hard the computer works. More trials means a smoother answer, slightly slower.',
  costs: 'What it costs you to deliver, and how those costs might drift.',
  market: 'The market you are selling into, and who else is in it.',
  rivals: 'What your competitors are likely to do when you move.',
  pipeline: 'How deals move through your stages, and where they leak.',
  // QA TM-19: "board" is sales jargon. It is the word for the set of stages a
  // deal can be in, so the group line says exactly that.
  'the board': 'Your board — the set of stages a deal can be in, and the chance it '
               + 'moves on, sits still or dies at each one.',
  process: 'How your approval process works today, step by step.',
  team: 'Who is available to do the work, and how fast they work.',
  simulation: 'How hard the computer works. More trials means a smoother answer.',
  assumptions: 'The numbers behind the answer. Change any of them and re-run.',
};

function groupLine(name) {
  const key = String(name || '').trim().toLowerCase();
  if (GROUP_LINES[key]) return GROUP_LINES[key];
  return `Everything about ${String(name || 'this').toLowerCase()}.`;
}

function stepOf(p) {
  if (p.step) return Number(p.step);
  if (p.kind === 'integer') return 1;
  const span = (p.max ?? 100) - (p.min ?? 0);
  return span > 200 ? 1 : 0.1;
}

/**
 * What the "?" says when the sentence under the label has already been read.
 *
 * QA UX-06: the bubble used to repeat `p.help` word for word, which is the
 * sentence already printed right below the label. It now carries the things
 * the caption does not: what the control accepts, how it steps, what it falls
 * back to, and what the choices are. Returns '' when there is nothing extra to
 * say, and the "?" is then not rendered at all rather than opening an empty
 * bubble.
 */
function extraTip(p) {
  const bits = [];
  if (p.unit) bits.push(`Measured in ${p.unit}.`);
  // A bound the parameter does not have arrives as null, and Number(null) is
  // 0 - which is finite, so a parameter with no ceiling was described as
  // "Accepts 0 to 0". Treat "no bound given" as no bound.
  const bound = (v) => {
    if (v === null || v === undefined || v === '') return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };
  const lo = bound(p.min);
  const hi = bound(p.max);
  if (lo !== null && hi !== null) {
    bits.push(`Accepts ${lo} to ${hi}${p.step ? `, in steps of ${p.step}` : ''}.`);
  } else if (lo !== null) {
    bits.push(`Accepts ${lo} or more.`);
  } else if (hi !== null) {
    bits.push(`Accepts up to ${hi}.`);
  }
  if (p.default !== undefined && p.default !== null && p.default !== '') {
    bits.push(typeof p.default === 'boolean'
      ? `Left alone it is ${p.default ? 'on' : 'off'}.`
      : `Left alone it uses ${p.default}.`);
  }
  if (Array.isArray(p.choices) && p.choices.length) {
    bits.push(`Choose one of: ${p.choices
      .map((c) => (typeof c === 'string' ? c : c.label))
      .join(', ')}.`);
  }
  return bits.join(' ');
}

export function defaultParams(parameters) {
  const out = {};
  (parameters || []).forEach((p) => {
    out[p.key] = p.default;
  });
  return out;
}

/** QA DF-12 / UX-15: merge values found in the user's file into a dropdown's
 *  choices, keeping the built-in ones as a fallback. Never drop an option. */
function mergedParam(param, choiceOverrides) {
  const extra = choiceOverrides?.[param.key];
  if (!extra || !extra.length) return param;
  const base = param.choices || [];
  const seen = new Set(base.map((c) => String(c.value)));
  const merged = [...base];
  extra.forEach((v) => {
    if (!seen.has(String(v))) {
      seen.add(String(v));
      merged.push({ value: v, label: String(v) });
    }
  });
  return { ...param, choices: merged };
}

export default function ParamForm({ parameters, values, onChange, compact, choiceOverrides }) {
  // QA TM-16: params that must add up to a total now show the running total, so
  // the constraint is visible on screen instead of only enforced later.
  const sumTotals = useMemo(() => {
    const out = {};
    (parameters || []).forEach((p) => {
      if (!p.sum_group) return;
      const acc =
        out[p.sum_group] ||
        (out[p.sum_group] = { total: 0, target: Number(p.sum_target) || 100 });
      const v = Number(values[p.key]);
      if (Number.isFinite(v)) acc.total += v;
    });
    return out;
  }, [parameters, values]);

  const groups = useMemo(() => {
    const map = new Map();
    (parameters || []).forEach((p) => {
      const g = p.group || 'Assumptions';
      if (!map.has(g)) map.set(g, []);
      map.get(g).push(p);
    });
    return Array.from(map.entries());
  }, [parameters]);

  // The second argument tells the caller WHICH key the user actually touched, so
  // a value the user typed can outrank one learned from their file (QA DF-10).
  const set = (key, value) => onChange({ ...values, [key]: value }, key);

  if (!parameters || !parameters.length) {
    return <div className="empty-note">This simulation has no extra assumptions to set.</div>;
  }

  return (
    <div>
      {groups.map(([group, params]) => (
        <section className="param-group" key={group}>
          <div className="param-group-head">
            <div className="param-group-name">{group}</div>
          </div>
          <p className="param-group-line">{groupLine(group)}</p>

          {params.map((p) => (
            <div className="param" key={p.key}>
              {/* QA UX-08: the unit used to repeat on its own line under the label
                  while the value box sat detached. It now appears once, beside
                  the number it belongs to (see the Control below). */}
              <div className="param-label">
                {p.label}
                {/* QA UX-06: the bubble carries the acceptance detail, never a
                    copy of the sentence printed underneath. */}
                {extraTip(p) && (
                  <HelpTip label={p.label} text={extraTip(p)} />
                )}
              </div>
              {p.help && <p className="param-help">{p.help}</p>}
              <Control
                param={mergedParam(p, choiceOverrides)}
                value={values[p.key]}
                onChange={(v) => set(p.key, v)}
              />
              {p.sum_group && sumTotals[p.sum_group] ? (
                <p className="param-help">
                  {sumTotals[p.sum_group].total.toFixed(0)}% of{' '}
                  {sumTotals[p.sum_group].target.toFixed(0)}%{' '}
                  {Math.abs(sumTotals[p.sum_group].total -
                            sumTotals[p.sum_group].target) < 0.5
                    ? '— that adds up.'
                    : '— we will scale these to fit.'}
                </p>
              ) : null}
            </div>
          ))}
        </section>
      ))}

      {!compact && (
        <div className="btn-row" style={{ marginTop: 20 }}>
          <button
            type="button"
            className="btn btn-sm btn-ghost"
            onClick={() => onChange(defaultParams(parameters), null)}
          >
            Reset to the usual values
          </button>
          <span className="muted">
            Nothing here is permanent — change a number and press Run again.
          </span>
        </div>
      )}
    </div>
  );
}

function Control({ param, value, onChange }) {
  const kind = param.kind || 'number';

  if (kind === 'bool') {
    const on = Boolean(value);
    return (
      <div className="param-control">
        <button
          type="button"
          className={`toggle${on ? ' is-on' : ''}`}
          role="switch"
          aria-checked={on}
          aria-label={param.label}
          onClick={() => onChange(!on)}
        />
        <span className="toggle-word">{on ? 'Yes' : 'No'}</span>
      </div>
    );
  }

  if (kind === 'choice') {
    const choices = param.choices || [];
    return (
      <div className="param-control">
        <div className="seg">
          {choices.map((c) => {
            const active = c.value === value;
            return (
              <button
                type="button"
                key={String(c.value)}
                className={active ? 'is-on' : ''}
                aria-pressed={active}
                onClick={() => onChange(c.value)}
              >
                {c.label ?? String(c.value)}
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  if (kind === 'integer') {
    const min = param.min ?? 0;
    const max = param.max ?? 1000000;
    const step = stepOf(param);
    const num = clampNum(value, min, max, Math.round(Number(param.default ?? min)));
    return (
      <div className="param-control">
        <div className="stepper">
          <button
            type="button"
            aria-label={`Decrease ${param.label}`}
            disabled={num - step < min}
            onClick={() => onChange(Math.max(min, num - step))}
          >
            −
          </button>
          <input
            type="number"
            value={num}
            min={min}
            max={max}
            step={step}
            onChange={(e) => onChange(clampNum(e.target.value, min, max, num))}
          />
          <button
            type="button"
            aria-label={`Increase ${param.label}`}
            disabled={num + step > max}
            onClick={() => onChange(Math.min(max, num + step))}
          >
            +
          </button>
        </div>
        {param.unit && <span className="unit">{param.unit}</span>}
        <span className="muted">
          between {fmt(min)} and {fmt(max)}
        </span>
      </div>
    );
  }

  // number: slider + typed box
  const min = param.min ?? 0;
  const max = param.max ?? Math.max(Number(param.default ?? 1) * 2, 100);
  const step = stepOf(param);
  const num = clampNum(value, min, max, Number(param.default ?? min));

  return (
    <div className="param-control">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={num}
        aria-label={`${param.label} slider`}
        onChange={(e) => onChange(clampNum(e.target.value, min, max, num))}
      />
      <input
        className="input num-box"
        type="number"
        min={min}
        max={max}
        step={step}
        value={num}
        aria-label={`${param.label} value`}
        onChange={(e) => onChange(clampNum(e.target.value, min, max, num))}
      />
      {param.unit && <span className="unit">{param.unit}</span>}
    </div>
  );
}

function clampNum(raw, min, max, fallback) {
  const n = Number(raw);
  if (!isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function fmt(n) {
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

/** Used by App to build a fresh parameter set whenever the use case changes. */
export function useParamDefaults(parameters) {
  const [values, setValues] = useState(() => defaultParams(parameters));
  useEffect(() => {
    setValues(defaultParams(parameters));
  }, [parameters]);
  return [values, setValues];
}
