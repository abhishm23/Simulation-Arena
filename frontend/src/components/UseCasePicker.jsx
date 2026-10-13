import React from 'react';
import HelpTip from './HelpTip.jsx';
import BrandMark from './BrandMark.jsx';
import { MODEL_TIP, modelListPlain } from '../models.js';

export default function UseCasePicker({ useCases, selectedId, onSelect, loading, error, onRetry }) {
  if (loading) {
    return (
      <div className="brand-splash">
        <BrandMark size={96} className="rail-logo" />
        <div className="empty-note" style={{ padding: 0, border: 0, background: 'none' }}>
          <span className="spinner" /> Loading the simulations we can run for you…
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-card">
        <div className="error-title">We could not load the list of simulations</div>
        <div className="error-msg">{error}</div>
        <div className="btn-row" style={{ marginTop: 14 }}>
          {onRetry && (
            <button type="button" className="btn btn-sm btn-primary" onClick={onRetry}>
              Try again
            </button>
          )}
          <span className="muted">
            Nothing is lost — the list is read from the engine each time this screen opens.
          </span>
        </div>
      </div>
    );
  }

  if (!useCases.length) {
    return (
      <div className="empty-note" style={{ textAlign: 'left' }}>
        <strong style={{ color: 'var(--text)' }}>There are no simulations to choose from.</strong>
        <p style={{ marginTop: 8 }}>
          The engine is running but reported no questions it can answer. That usually means the
          installation is incomplete rather than anything you did — press “Try again”, and if it
          stays empty show this screen to whoever set the app up.
        </p>
        {onRetry && (
          <button type="button" className="btn btn-sm" style={{ marginTop: 12 }} onClick={onRetry}>
            Try again
          </button>
        )}
      </div>
    );
  }

  return (
    <>
      <p className="section-sub">
        Each card is one question. The blue badges name the technique we use to answer it —{' '}
        <HelpTip label="the blue badges" text={MODEL_TIP} />
      </p>
      <div className="uc-grid">
        {useCases.map((uc) => {
          const selected = uc.id === selectedId;
          return (
            <button
              type="button"
              key={uc.id}
              className={`uc-card${selected ? ' is-selected' : ''}`}
              onClick={() => onSelect(uc.id)}
              aria-pressed={selected}
            >
              {selected && <span className="uc-check">✓</span>}
              <div className="uc-top">
                <span className="uc-icon" aria-hidden="true">
                  {uc.icon || '📊'}
                </span>
                <span className="uc-head">
                  <span className="uc-title" title={uc.title}>
                    {uc.title}
                  </span>
                  <span className="uc-sub">{uc.subtitle}</span>
                </span>
              </div>

              <p className="uc-question">{uc.business_question}</p>
              <p className="uc-summary">{uc.plain_summary}</p>

              <div className="badges">
                {(uc.models || []).map((m) => (
                  <span className="badge badge-model" key={m}>
                    {m}
                  </span>
                ))}
                {uc.category && <span className="badge">{uc.category}</span>}
              </div>
            </button>
          );
        })}
      </div>

      {selectedId && (
        <Preview useCase={useCases.find((u) => u.id === selectedId)} />
      )}
    </>
  );
}

function Preview({ useCase }) {
  if (!useCase) return null;

  return (
    <div className="story-preview">
      <div className="section-title">
        What this will do{' '}
        <HelpTip
          label="what this will do"
          text="These are the steps the simulation walks through, in order. Nothing here needs you to understand maths — the model does the counting."
        />
      </div>
      <p className="section-sub">
        It works like this: we use {modelListPlain(useCase.models)}.
      </p>
      <ol>
        {(useCase.story_steps || []).map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ol>

      <div className="divider" />

      <div className="badges" style={{ marginBottom: 10 }}>
        <span className="badge badge-accent">
          {(useCase.input_roles || []).length} pieces of information needed
        </span>
        <span className="badge badge-plain">
          {(useCase.parameters || []).length} assumptions you can adjust
        </span>
        {useCase.sample_dataset && (
          <span className="badge badge-plain">
            Example file ready: {useCase.sample_dataset}
          </span>
        )}
      </div>

      <p className="inline-note">
        You can run this with your own file, with our example data, or with no data at all —
        just the numbers you type in on the assumptions screen.
      </p>
    </div>
  );
}
