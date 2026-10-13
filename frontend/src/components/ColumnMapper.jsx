import React, { useCallback, useEffect, useState } from 'react';
import { automap } from '../api.js';
import HelpTip from './HelpTip.jsx';

const UNMAPPED = '__none__';

/** Confidence number -> the two words the user actually reads. */
function chipFor(confidence) {
  const c = Number(confidence);
  if (!isFinite(c) || c <= 0) return null;
  if (c >= 0.8) return { cls: 'badge-accent', text: 'matched automatically' };
  if (c >= 0.5) return { cls: 'badge-amber', text: 'please check' };
  return { cls: 'badge-plain', text: 'best guess — please check' };
}

/**
 * What the "?" says when the sentence under the label has already been read.
 *
 * QA UX-06: the bubble used to repeat `role.help` word for word - the same
 * sentence printed on the line below it. It now says the things that line does
 * not: whether the column is required, what kind of column we are looking for,
 * and the other column names we will accept for it. Returns '' when there is
 * nothing extra, and the "?" is then not rendered.
 */
function roleTip(role) {
  const bits = [
    role.required
      ? 'This one is needed - without it there is no answer.'
      : 'Optional - leave it as "not in my file" if you do not have it.',
  ];
  const kinds = {
    identifier: 'a column that names each deal or order',
    categorical: 'a column of categories, such as a stage or a region',
    numeric: 'a column of numbers',
    datetime: 'a column of dates',
    duration: 'a column of time spans',
  };
  if (role.dtype && kinds[role.dtype]) {
    bits.push(`We look for ${kinds[role.dtype]}.`);
  } else if (role.dtype) {
    bits.push(`We look for a ${role.dtype} column.`);
  }
  if (Array.isArray(role.aliases) && role.aliases.length) {
    bits.push(`Column names we also accept: ${role.aliases.join(', ')}.`);
  }
  return bits.join(' ');
}

export default function ColumnMapper({
  useCase,
  datasetId,
  mapping,
  onMapping,
  onAuto,
}) {
  const [loading, setLoading] = useState(false);
  const [notes, setNotes] = useState([]);
  const [error, setError] = useState(null);
  const [errorDetail, setErrorDetail] = useState('');

  const roles = (useCase && useCase.input_roles) || [];

  const runGuess = useCallback(() => {
    if (!datasetId || !useCase) return undefined;
    let alive = true;
    setLoading(true);
    setError(null);
    setErrorDetail('');
    automap(useCase.id, datasetId)
      .then((res) => {
        if (!alive) return;
        const auto = (res && res.mapping) || {};
        const next = {};
        roles.forEach((r) => {
          next[r.key] = auto[r.key] == null ? null : auto[r.key];
        });
        // Only fill roles the user has not chosen themselves yet.
        const merged = {};
        roles.forEach((r) => {
          const current = (mapping || {})[r.key];
          merged[r.key] = current === undefined ? next[r.key] : current;
        });
        onMapping(merged);
        setNotes((res && res.notes) || []);
        if (onAuto) onAuto(res || {});
      })
      .catch((err) => {
        if (!alive) return;
        setError(err.message);
        setErrorDetail(err.detail || '');
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, useCase && useCase.id]);

  useEffect(runGuess, [runGuess]);
  const retry = runGuess;

  if (!useCase) return null;

  const columns = (useCase._columns || []);
  const requiredMissing = roles.filter(
    (r) => r.required && !mapping[r.key],
  );

  function setRole(roleKey, value) {
    onMapping({ ...mapping, [roleKey]: value === UNMAPPED ? null : value });
  }

  return (
    <>
      {loading && (
        <div className="empty-note" style={{ marginBottom: 16 }}>
          <span className="spinner" /> Matching your columns to what the simulation needs…
        </div>
      )}

      {error && (
        <div className="error-card" style={{ marginBottom: 16 }}>
          <div className="error-title">We could not guess your columns</div>
          <div className="error-msg">
            {error} You can still pick each one yourself below — nothing is lost.
          </div>
          <div className="btn-row" style={{ marginTop: 12 }}>
            <button type="button" className="btn btn-sm" onClick={retry}>
              Try the guess again
            </button>
          </div>
          {errorDetail && (
            <details style={{ marginTop: 12 }}>
              <summary className="muted" style={{ cursor: 'pointer' }}>
                Technical detail (for your IT team)
              </summary>
              <pre className="trace">{String(errorDetail)}</pre>
            </details>
          )}
        </div>
      )}

      {!!notes.length && (
        <div className="story-preview" style={{ marginTop: 0, marginBottom: 20 }}>
          <div className="section-title">What we spotted</div>
          <ul style={{ margin: '10px 0 0', paddingLeft: 20, fontSize: 13.5, lineHeight: 1.7 }}>
            {notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </div>
      )}

      <p className="section-sub">
        Each line below is one piece of information the simulation needs. Choose which of your
        columns holds it — or leave it blank and we will use a sensible default.
      </p>

      {!!roles.length && !columns.length && (
        <div className="warn-block" style={{ marginBottom: 16 }}>
          <div className="warn-title">Your file did not give us any columns to match</div>
          <p style={{ margin: '0 0 8px', fontSize: 13, lineHeight: 1.65 }}>
            That usually means the first row was not a header row, or the file is empty. You have
            three options, and none of them lose your work so far:
          </p>
          <ul>
            <li>Go back a step and pick another file — one with column names on the top row.</li>
            <li>
              Carry on: every one of the {roles.length} pieces of information below falls back to
              a sensible default, and you can set those yourself on the assumptions screen.
            </li>
            <li>Run it with no data at all, entirely on the numbers you type in.</li>
          </ul>
        </div>
      )}

      <div>
        {roles.map((role) => {
          const value = mapping[role.key] == null ? UNMAPPED : mapping[role.key];
          const chip = chipFor((useCase._confidence || {})[role.key]);
          const missing = role.required && value === UNMAPPED;
          return (
            <div className="map-row" key={role.key}>
              <div>
                <div className="map-label">
                  {role.label}
                  {role.required ? (
                    <span className="badge badge-plain">needed</span>
                  ) : (
                    <span className="badge">optional</span>
                  )}
                  {chip && <span className={`badge ${chip.cls}`}>{chip.text}</span>}
                  {/* QA UX-06: the bubble carries the acceptance detail, never a
                      copy of the sentence printed underneath. */}
                  {roleTip(role) && (
                    <HelpTip label={role.label} text={roleTip(role)} />
                  )}
                </div>
                {role.help && <p className="map-help">{role.help}</p>}
                {missing && (
                  <p className="map-note">
                    ⚠️ We could not find this in your file. That is OK — {fallbackWords(role)}.
                  </p>
                )}
              </div>

              <div>
                <select
                  className="input"
                  value={value}
                  onChange={(e) => setRole(role.key, e.target.value)}
                  aria-label={`Column for ${role.label}`}
                >
                  <option value={UNMAPPED}>— not in my file —</option>
                  {columns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                  {value !== UNMAPPED && !columns.includes(value) && (
                    <option value={value}>{value}</option>
                  )}
                </select>
                <p className="map-fallback">
                  {role.required
                    ? 'If you leave this blank we use your standard value instead.'
                    : 'Optional — leave blank and we simply work without it.'}
                </p>
              </div>
            </div>
          );
        })}

        {!roles.length && (
          <div className="empty-note">
            This simulation does not need any columns from your file. Continue to the
            assumptions.
          </div>
        )}
      </div>

      {requiredMissing.length > 0 && (
        <div className="warn-block" style={{ marginTop: 16 }}>
          <div className="warn-title">
            {requiredMissing.length === 1
              ? 'One thing we need is not matched to a column'
              : `${requiredMissing.length} things we need are not matched to a column`}
          </div>
          <p style={{ margin: '0 0 8px', fontSize: 13, lineHeight: 1.65 }}>
            This is not a dead end — pick a different column above, or leave it blank and we will
            use your standard value. Either way the run log will state which one we used.
          </p>
          <ul>
            {requiredMissing.map((r) => (
              <li key={r.key}>
                “{r.label}” — {fallbackWords(r)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </>
  );
}

function fallbackWords(role) {
  const fb = role.fallback;
  if (fb == null) return 'we will use a sensible default value instead';
  if (typeof fb === 'object') {
    if (fb.derive) return `we will work it out from your other columns (${fb.derive})`;
    return 'we will use a sensible default value instead';
  }
  return `we will use ${fb} as the standard value`;
}
