import React from 'react';
import ChartView from './ChartView.jsx';
import DownloadBar from './DownloadBar.jsx';
import KpiCard from './KpiCard.jsx';
import StorySteps from './StorySteps.jsx';
import HelpTip from './HelpTip.jsx';
import { MODEL_TIP, modelPlain } from '../models.js';
import { humanCount } from '../format.js';

const RISK_CLASS = {
  low: 'badge-accent',
  moderate: 'badge-amber',
  high: 'badge-red',
  critical: 'badge-red',
};

/** A finished run that produced nothing a person could read. */
function emptyResult(r) {
  return !(
    (r.kpis || []).length ||
    (r.charts || []).length ||
    (r.tables || []).length ||
    (r.story || []).length ||
    r.recommendation
  );
}

export default function ResultsView({ result, seed, onSeedChange, onRerun, uploadSeconds }) {
  const r = result || {};
  const readSeconds = Number(uploadSeconds) || 0;
  const failed = r.status === 'failed';
  const rec = r.recommendation || null;
  const riskKey = String(rec && (rec.risk_label || '')).toLowerCase();

  return (
    <>
      <div className="hero">
        <div className="hero-meta">
          <span className="badge badge-accent">{r.use_case_title || 'Simulation'}</span>
          <span className="badge badge-plain">
            Reproducible — seed {r.seed ?? seed ?? 42}
          </span>
          {r.n_simulations ? (
            <span className="badge badge-plain">
              {humanCount(r.n_simulations)} scenarios tested
            </span>
          ) : null}
          {/* QA UX-24: the badge used to advertise only the simulation time while
              the upload - 94-99.98% of the real wait - was excluded. Show both,
              and the total, so the number reflects what the user actually waited. */}
          {r.runtime_ms ? (
            <span className="badge badge-plain">
              the simulation itself took {(r.runtime_ms / 1000).toFixed(1)}s
            </span>
          ) : null}
          {readSeconds > 0.05 ? (
            <span className="badge badge-plain">
              your file took {readSeconds.toFixed(1)}s to read
            </span>
          ) : null}
          {readSeconds > 0.05 && r.runtime_ms ? (
            <span className="badge badge-plain">
              {(readSeconds + r.runtime_ms / 1000).toFixed(1)}s end to end
            </span>
          ) : null}
        </div>
        <h1 className="hero-headline">{r.headline}</h1>
        {r.model_used && (
          <p className="inline-note">
            Worked out using <strong style={{ color: 'var(--text)' }}>{r.model_used}</strong> — in
            plain terms, {modelPlain(r.model_used)}.{' '}
            <HelpTip label="the model name" text={MODEL_TIP} />
          </p>
        )}
      </div>

      {failed && (
        <div className="error-card" style={{ marginTop: 20 }}>
          <div className="error-title">This run did not finish</div>
          <div className="error-msg">
            {(r.warnings || [])[0] || 'Something went wrong while running the simulation.'}
          </div>
          <p className="muted" style={{ marginTop: 10 }}>
            Nothing you did caused this, and none of your settings were changed. Press “Run it
            again” below — your assumptions and column matches are exactly as you left them.
          </p>
        </div>
      )}

      {!failed && emptyResult(r) && (
        <div className="empty-note" style={{ marginTop: 20, textAlign: 'left' }}>
          <strong style={{ color: 'var(--text)' }}>
            The run finished, but there is nothing to show.
          </strong>
          <p style={{ marginTop: 8 }}>
            That normally means the data could not support an answer. Try adding a column match on
            step 3, changing an assumption on step 4, or running with no data at all so the model
            works purely from your typed-in numbers.
          </p>
        </div>
      )}

      {r.method_plain && (
        <section className="card">
          <div className="section-title">
            How this works{' '}
            <HelpTip
              label="how this works"
              text="The method, described without the maths. You do not need to understand the model to trust the answer — but you are entitled to know what it did."
            />
          </div>
          <p className="section-sub" style={{ marginBottom: 0, fontSize: 14.5, color: 'var(--text-dim)' }}>
            {r.method_plain}
          </p>
        </section>
      )}

      {!!(r.kpis || []).length && (
        <section className="card">
          <div className="section-title">
            The numbers that matter{' '}
            <HelpTip
              label="these numbers"
              text="Each card is one number, what it means in a sentence, and how it compares with the alternative. Green is good, red is a warning."
            />
          </div>
          <div className="kpi-grid">
            {r.kpis.map((k, i) => (
              <KpiCard key={i} kpi={k} />
            ))}
          </div>
        </section>
      )}

      {!!(r.story || []).length && (
        <section className="card">
          <div className="section-title">The story behind the answer</div>
          <p className="section-sub">
            Read these in order — each one builds on the last, and each ends with the number
            that backs it up.
          </p>
          <StorySteps steps={r.story} />
        </section>
      )}

      {!!(r.charts || []).length && (
        <section>
          <div className="section-title" style={{ marginBottom: 14 }}>
            The picture
          </div>
          {r.charts.map((c, i) => (
            <ChartView key={(c && c.id) || i} chart={c} />
          ))}
        </section>
      )}

      {!!(r.tables || []).length && (
        <section className="card">
          <div className="section-title">The detail, row by row</div>
          {r.tables.map((t, i) => (
            <ResultTable key={i} table={t} />
          ))}
        </section>
      )}

      {rec && (
        <section className="rec-card" style={{ marginTop: 20 }}>
          <div className="hero-meta" style={{ marginBottom: 12 }}>
            <span className="badge badge-accent">What we recommend</span>
            {rec.risk_label && (
              <span className={`badge ${RISK_CLASS[riskKey] || 'badge-plain'}`}>
                Risk: {rec.risk_label}
                {typeof rec.risk_score === 'number' ? ` (${Math.round(rec.risk_score)}/100)` : ''}
              </span>
            )}
            {rec.confidence && (
              <span className="badge badge-plain">
                How sure we are: {rec.confidence}
                <HelpTip
                  label="how sure we are"
                  text="How much the answer would have to be wrong before the recommendation changed. It is not a probability that the recommendation is correct — it is a measure of how firmly your own numbers point this way."
                />
              </span>
            )}
          </div>
          <h2 className="rec-headline">{rec.headline}</h2>

          <div className="rec-grid">
            {!!(rec.rationale || []).length && (
              <div>
                <div className="section-sub" style={{ marginBottom: 8 }}>
                  Why
                </div>
                <ul className="rec-list">
                  {rec.rationale.map((x, i) => (
                    <li key={i}>{x}</li>
                  ))}
                </ul>
              </div>
            )}
            {!!(rec.actions || []).length && (
              <div>
                <div className="section-sub" style={{ marginBottom: 8 }}>
                  What to do next
                </div>
                <ol className="rec-actions">
                  {rec.actions.map((a, i) => (
                    <li key={i}>
                      <span>{a}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>

          {rec.tradeoff && (
            <div className="tradeoff">
              <strong>The trade-off: </strong>
              {rec.tradeoff}
            </div>
          )}
        </section>
      )}

      {!!(r.assumptions || []).length && (
        <section className="card">
          <div className="section-title">
            What we assumed{' '}
            <HelpTip
              label="assumptions"
              text="Every answer rests on assumptions. These are the ones we used. Change any of them on the assumptions screen and run again."
            />
          </div>
          <ul className="plain-list">
            {r.assumptions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </section>
      )}

      {!!(r.warnings || []).length && !failed && (
        <div className="warn-block" style={{ marginTop: 20 }}>
          <div className="warn-title">Things to keep an eye on</div>
          <ul>
            {r.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      <div style={{ marginTop: 20 }}>
        <DownloadBar
          runId={r.run_id}
          seed={r.seed ?? seed}
          onSeedChange={onSeedChange}
          onRerun={onRerun}
        />
      </div>

      <details className="appendix" style={{ marginTop: 20 }}>
        <summary>Appendix — the full run log ({r.log ? r.log.length : 0} steps)</summary>
        <div className="appendix-body">
          <div className="terminal" style={{ height: 300, marginTop: 14 }}>
            {(r.log || []).map((ev, i) => (
              <div className={`log-line is-${ev.level || 'info'}`} key={i}>
                <span className="log-t">{Number(ev.t || 0).toFixed(1)}s</span>
                <span className="log-phase">{ev.phase || 'run'}</span>
                <span className="log-msg">{ev.message}</span>
              </div>
            ))}
            {!(r.log || []).length && <div className="log-line"><span className="log-msg">No log was recorded.</span></div>}
          </div>
          <p className="muted" style={{ marginTop: 12 }}>
            Run {r.run_id}. Same seed ({r.seed ?? seed}) gives exactly the same numbers again —
            that is what makes this reproducible.
          </p>
        </div>
      </details>
    </>
  );
}

function ResultTable({ table }) {
  const t = table || {};
  const highlight = new Set(t.highlight_rows || []);
  return (
    <div style={{ marginTop: 18 }}>
      <div className="section-title" style={{ fontSize: 15 }}>
        {t.name}
      </div>
      {t.description && <p className="section-sub" style={{ marginBottom: 0 }}>{t.description}</p>}
      <div className="table-scroll">
        <table className="grid">
          <thead>
            <tr>
              {(t.columns || []).map((c, i) => (
                <th key={i}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(t.rows || []).map((row, ri) => (
              <tr key={ri} className={highlight.has(ri) ? 'is-highlight' : ''}>
                {(row || []).map((cell, ci) => (
                  <td key={ci}>{cell === null || cell === undefined ? '' : String(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {t.highlight_note && <div className="table-note">{t.highlight_note}</div>}
    </div>
  );
}
