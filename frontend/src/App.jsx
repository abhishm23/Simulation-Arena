import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { getUseCases, distinctValues } from './api.js';
import ColumnMapper from './components/ColumnMapper.jsx';
import DataStep from './components/DataStep.jsx';
import ParamForm, { defaultParams } from './components/ParamForm.jsx';
import ResultsView from './components/ResultsView.jsx';
import RunPanel from './components/RunPanel.jsx';
import UseCasePicker from './components/UseCasePicker.jsx';
import HelpTip from './components/HelpTip.jsx';
import BrandMark from './components/BrandMark.jsx';
import { MODEL_TIP, granularityPlain, modelListPlain } from './models.js';
import { humanCount } from './format.js';

const STEPS = [
  { id: 1, label: 'Choose your problem', hint: 'Pick the question you need answered' },
  { id: 2, label: 'Give us your data', hint: 'A file, an example, a paste — or nothing' },
  { id: 3, label: 'Match your columns', hint: 'Tell us which column means what' },
  { id: 4, label: 'Set the assumptions', hint: 'The numbers behind the answer' },
  { id: 5, label: 'Run it', hint: 'Watch it work, then read the answer' },
];

const OUTPUT_FORMATS = ['html', 'xlsx', 'csv', 'pdf'];

export default function App() {
  const [step, setStep] = useState(1);
  const [useCases, setUseCases] = useState([]);
  const [catalogError, setCatalogError] = useState(null);
  const [loadingCatalog, setLoadingCatalog] = useState(true);

  const [selectedId, setSelectedId] = useState(null);
  const [profile, setProfile] = useState(null);
  const [dataNote, setDataNote] = useState('');
  const [mode, setMode] = useState('data'); // 'data' | 'manual'
  const [roleMapping, setRoleMapping] = useState({});
  const [autoInfo, setAutoInfo] = useState({ mapping: {}, confidence: {}, notes: [] });
  const [params, setParams] = useState({});
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState(null);
  const [autoStart, setAutoStart] = useState(0);

  const useCase = useMemo(
    () => useCases.find((u) => u.id === selectedId) || null,
    [useCases, selectedId],
  );
  const datasetId = profile ? profile.dataset_id : null;
  const columns = useMemo(
    () => (profile && profile.columns ? profile.columns.map((c) => c.name) : []),
    [profile],
  );

  const loadCatalog = useCallback(() => {
    setLoadingCatalog(true);
    setCatalogError(null);
    getUseCases()
      .then((list) => setUseCases(Array.isArray(list) ? list : []))
      .catch((err) => setCatalogError(err.message))
      .finally(() => setLoadingCatalog(false));
  }, []);

  useEffect(loadCatalog, [loadCatalog]);

  // True while a file is being read, so the wizard can guard its buttons (UX-02).
  const [dataBusy, setDataBusy] = useState(false);
  // How long the file took to read, for the end-to-end timing badge (UX-24).
  const [uploadSeconds, setUploadSeconds] = useState(0);

  // QA DF-12 / UX-15: dropdowns must offer the stages actually present in the
  // user's file, not only the ones in the standard template.
  const [choiceOverrides, setChoiceOverrides] = useState({});
  useEffect(() => {
    let cancelled = false;
    setChoiceOverrides({});
    if (!useCase || !datasetId) return undefined;
    const stageCol = roleMapping.stage || autoInfo.mapping?.stage;
    const hasStageParam = (useCase.parameters || []).some((p) => p.key === 'starting_stage');
    if (!stageCol || !hasStageParam) return undefined;
    distinctValues(datasetId, stageCol, 40)
      .then((r) => {
        const vals = r?.values || [];
        if (!cancelled && vals.length) {
          setChoiceOverrides({ starting_stage: vals });
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [useCase, datasetId, roleMapping, autoInfo.mapping]);

  // Which Step-4 values the user actually typed (QA DF-10 / UX-04). A typed value
  // must outrank one learned from the uploaded file, so we send the list of
  // touched keys with the run.
  const [touchedParams, setTouchedParams] = useState({});

  const handleParamsChange = useCallback((next, changedKey) => {
    setParams(next);
    setTouchedParams((prev) => (changedKey === null ? {} : { ...prev, [changedKey]: true }));
  }, []);

  // A different simulation means different roles and different assumptions.
  useEffect(() => {
    if (!useCase) return;
    setParams(defaultParams(useCase.parameters));
    setTouchedParams({});
    setRoleMapping({});
    setAutoInfo({ mapping: {}, confidence: {}, notes: [] });
    setResult(null);
  }, [useCase]);

  // Escape steps back one screen — but never while the user is typing in a
  // field (Escape there means "leave this field alone"), and never off the
  // results page, where there is nothing to go back to.
  useEffect(() => {
    function onKey(e) {
      if (e.key !== 'Escape') return;
      const el = e.target;
      const tag = el && el.tagName ? el.tagName.toLowerCase() : '';
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
      if (el && el.isContentEditable) return;
      if (result) return;
      setStep((s) => (s > 1 ? s - 1 : s));
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [result]);

  // QA UX-22: the card already carries the business question and a plain-English
  // summary, so choosing one can go straight to the next step - Back is always
  // there if it was the wrong card.
  function pickUseCase(id) {
    setSelectedId(id);
    setResult(null);
    setStep(2);
  }

  // `readSeconds` is how long the file took to read, so the report can show a
  // true end-to-end time rather than only the simulation (QA UX-24).
  function handleProfile(next, note, readSeconds) {
    setProfile(next);
    setUploadSeconds(typeof readSeconds === 'number' ? readSeconds : 0);
    setDataNote(note || '');
    setMode(next ? 'data' : 'manual');
    // QA UX-22 / UX-20: once the file is read, take the user straight to the
    // column check. That removes a click and puts the mapping review on the
    // main path instead of behind an optional button.
    if (next) setStep(3);
    else setStep(4);
    setRoleMapping({});
    setAutoInfo({ mapping: {}, confidence: {}, notes: [] });
    setResult(null);
  }

  function goManual() {
    setMode('manual');
    setProfile(null);
    setResult(null);
    setStep(4);
  }

  const mappingNeeded = mode === 'data' && !!datasetId;
  // Why a step is locked, in the user's own terms rather than a grey button.
  const lockedReason = !useCase ? 'Choose a question on step 1 first.' : null;

  function next() {
    if (step === 1 && !selectedId) return;
    if (step === 2) {
      if (mappingNeeded) setStep(3);
      else setStep(4);
      return;
    }
    setStep((s) => Math.min(5, s + 1));
  }

  function back() {
    if (step === 4 && !mappingNeeded) {
      setStep(2);
      return;
    }
    setStep((s) => Math.max(1, s - 1));
  }

  const ucForMapper = useMemo(() => {
    if (!useCase) return null;
    return { ...useCase, _columns: columns, _confidence: autoInfo.confidence || {} };
  }, [useCase, columns, autoInfo]);

  return (
    <div className="app">
      <a className="skip-link" href="#step-body">
        Skip to the main step
      </a>
      <aside className="rail">
        <div className="rail-brand">
          <BrandMark size={38} className="rail-logo" />
          <span>
            <div className="rail-title">Simulation Arena</div>
            <div className="rail-sub">Business answers, in plain English</div>
          </span>
        </div>

        <ol className="rail-steps">
          {STEPS.map((s) => {
            // QA UX-21: step 5 still showed a plain "5" on a finished report.
            // The last step is done once the run has produced a result.
            const done = step > s.id || (s.id === STEPS.length && !!result);
            const active = step === s.id;
            const locked = s.id > 1 && !useCase;
            return (
              <li key={s.id}>
                <button
                  type="button"
                  className={`rail-step${active ? ' is-active' : ''}${done ? ' is-done' : ''}${
                    locked ? ' is-locked' : ''
                  }`}
                  disabled={locked}
                  aria-current={active ? 'step' : undefined}
                  title={locked ? lockedReason : `Go to: ${s.label}`}
                  onClick={() => setStep(s.id)}
                >
                  <span className="rail-num">{done ? '✓' : s.id}</span>
                  <span>
                    <div className="rail-label">{s.label}</div>
                    <div className="rail-hint">{locked ? lockedReason : s.hint}</div>
                  </span>
                </button>
              </li>
            );
          })}
        </ol>

        {lockedReason && (
          <p className="rail-note">
            <span aria-hidden="true">🔒</span> Steps 2–5 unlock as soon as you pick a question.
          </p>
        )}

        <div className="rail-foot">
          Everything runs on this machine. Nothing is sent anywhere, and the same seed
          always gives the same answer.
        </div>
      </aside>

      <main className="main" id="step-body">
        {step === 1 && (
          <section className="card">
            <div className="step-head">
              <div className="step-kicker">Step 1 of 5</div>
              <h1 className="step-title">What do you need to decide?</h1>
              <p className="step-lede">
                Pick the question that sounds most like the one on your desk. Each one runs a
                different kind of test on your numbers — you do not need to know how it works,
                only which question you are asking.
              </p>
            </div>
            <UseCasePicker
              useCases={useCases}
              selectedId={selectedId}
              onSelect={pickUseCase}
              loading={loadingCatalog}
              error={catalogError}
              onRetry={loadCatalog}
            />
            {/* QA UX-22: choosing a card now moves on by itself, so this
                separate "Use this one" step is gone (-1 interaction). */}
            <div className="btn-row" style={{ marginTop: 24 }}>
              {!selectedId && (
                <span className="muted">
                  Choose a card above to continue — press Tab to move between them and Enter
                  to choose. Selecting one takes you to the next step straight away.
                </span>
              )}
            </div>
          </section>
        )}

        {step === 2 && useCase && (
          <section className="card">
            <div className="step-head">
              <div className="step-kicker">Step 2 of 5</div>
              <h1 className="step-title">Give us your data</h1>
              <p className="step-lede">
                For <strong className="wrap-any">{useCase.title}</strong> we need{' '}
                {granularityPlain(useCase.data_granularity)}. Any of these four will do, and you
                can skip this entirely.
              </p>
            </div>
            <DataStep
              profile={profile}
              onProfile={handleProfile}
              onManual={goManual}
              datasetId={datasetId}
              onBusyChange={setDataBusy}
            />
            {dataNote && (
              <p className="inline-note" style={{ marginTop: 14 }}>
                {dataNote}
              </p>
            )}
            <div className="btn-row" style={{ marginTop: 24 }}>
              <button type="button" className="btn btn-ghost" onClick={back}>
                ← Back
              </button>
              {/* QA UX-02: this stayed enabled through a 19-minute read. */}
              <button
                type="button"
                className="btn btn-primary"
                onClick={next}
                disabled={dataBusy}
                title={dataBusy ? 'Wait until your file has finished reading.' : undefined}
              >
                {dataBusy
                  ? 'Reading your file…'
                  : mappingNeeded
                    ? 'Match my columns →'
                    : 'Set the assumptions →'}
              </button>
              <span className="muted">
                {mode === 'manual'
                  ? 'No data — we will use the numbers you type.'
                  : 'You can change or remove this data later.'}
              </span>
            </div>
            {mode !== 'manual' && !datasetId && (
              <div className="empty-note" style={{ marginTop: 16, textAlign: 'left' }}>
                <strong style={{ color: 'var(--text)' }}>No data chosen yet.</strong>
                <p style={{ marginTop: 8 }}>
                  That is allowed — press “Set the assumptions” and the simulation will run on
                  the numbers you type in. Or pick a file, an example, or a paste above.
                </p>
              </div>
            )}
          </section>
        )}

        {step === 3 && useCase && (
          <section className="card">
            <div className="step-head">
              <div className="step-kicker">Step 3 of 5</div>
              <h1 className="step-title">Match your columns</h1>
              <p className="step-lede">
                We have guessed which of your columns means what. Check our guesses — the
                simulation works on these meanings, not on column names, so a wrong match
                gives a wrong answer.
              </p>
            </div>
            <ColumnMapper
              useCase={ucForMapper}
              datasetId={datasetId}
              mapping={roleMapping}
              onMapping={setRoleMapping}
              onAuto={(res) =>
                setAutoInfo({
                  mapping: res.mapping || {},
                  confidence: res.confidence || {},
                  notes: res.notes || [],
                })
              }
            />
            {/* QA UX-05: ColumnMapper used to draw its own primary button
                ("That looks right — continue") *and* this footer drew one
                ("Set the assumptions →"). Two primaries, both advancing, on the
                same screen. There is now exactly one, and its label says both
                things. */}
            <div className="btn-row" style={{ marginTop: 24 }}>
              <button type="button" className="btn btn-ghost" onClick={back}>
                ← Back
              </button>
              <button type="button" className="btn btn-primary" onClick={next}>
                That looks right — set the assumptions →
              </button>
              <span className="muted">
                You can come back and change this at any time before you run.
              </span>
            </div>
          </section>
        )}

        {step === 4 && useCase && (
          <section className="card">
            <div className="step-head">
              <div className="step-kicker">Step 4 of 5</div>
              <h1 className="step-title">Set the assumptions</h1>
              <p className="step-lede">
                These are the numbers the simulation uses where it has no history to learn
                from. They come pre-filled with sensible values — change any of them and the
                answer changes with them.
              </p>
              {/* QA TM-01: the model name was explained only at Step 5. Say it
                  here too, where the user is still deciding. */}
              {!!(useCase.models || []).length && (
                <p className="inline-note" style={{ marginTop: 10 }}>
                  How it works, in plain terms: this run uses{' '}
                  {modelListPlain(useCase.models)}.
                </p>
              )}
            </div>
            <ParamForm
              parameters={useCase.parameters}
              values={params}
              onChange={handleParamsChange}
              choiceOverrides={choiceOverrides}
            />
            <div className="btn-row" style={{ marginTop: 24 }}>
              <button type="button" className="btn btn-ghost" onClick={back}>
                ← Back
              </button>
              {/* QA UX-05 / UX-22: there were two near-synonymous primary buttons
                  ("Ready - run it" then "Run the simulation") and it was unclear
                  whether the first already started the run. One button, and it
                  genuinely starts the run. */}
              <button
                type="button"
                className="btn btn-primary btn-big"
                onClick={() => {
                  setAutoStart((n) => n + 1);
                  setStep(5);
                }}
              >
                Run the simulation →
              </button>
            </div>
          </section>
        )}

        {step === 5 && useCase && !result && (
          <section className="card">
            <div className="step-head">
              <div className="step-kicker">Step 5 of 5</div>
              <h1 className="step-title">Run it</h1>
              <p className="step-lede">
                It starts on its own — just watch. Every step is written out as it happens,
                so you can see exactly how the answer was reached. No black box, and no
                second button to press.
              </p>
            </div>

            <div className="story-preview" style={{ marginTop: 0, marginBottom: 22 }}>
              <div className="badges">
                <span className="badge badge-accent wrap-any" title={useCase.title}>
                  {useCase.title}
                </span>
                {(useCase.models || []).map((m) => (
                  <span className="badge badge-model" key={m}>
                    {m}
                  </span>
                ))}
                <HelpTip label="the model names in the blue badges" text={MODEL_TIP} />
                <span className="badge badge-plain">
                  {mode === 'data' && profile
                    ? `${humanCount(profile.n_rows)} rows of your data`
                    : 'your typed-in numbers'}
                </span>
                <span className="badge badge-plain">seed {seed}</span>
              </div>
              <p className="section-sub" style={{ margin: '12px 0 0' }}>
                In plain terms, this run uses {modelListPlain(useCase.models)}.
              </p>
              <div className="btn-row" style={{ marginTop: 14 }}>
                <button
                  type="button"
                  className="btn btn-sm btn-ghost"
                  onClick={() => setStep(4)}
                >
                  Change the assumptions
                </button>
                {mode === 'data' && datasetId && (
                  <button
                    type="button"
                    className="btn btn-sm btn-ghost"
                    onClick={() => setStep(3)}
                  >
                    Change the column matches
                  </button>
                )}
                <button
                  type="button"
                  className="btn btn-sm btn-ghost"
                  onClick={() => setStep(2)}
                >
                  Change the data
                </button>
              </div>
            </div>

            <RunPanel
              useCase={useCase}
              datasetId={datasetId}
              roleMapping={roleMapping}
              parameters={params}
              explicitParams={Object.keys(touchedParams)}
              seed={seed}
              outputFormats={OUTPUT_FORMATS}
              onResult={(res) => setResult(res)}
              resetKey={`${useCase.id}|${datasetId}|${seed}|${mode}`}
              autoStart={autoStart}
              onRunStart={() => setResult(null)}
            />

            <div className="btn-row" style={{ marginTop: 24 }}>
              <button type="button" className="btn btn-ghost" onClick={back}>
                ← Back
              </button>
              <span className="muted">
                Seed{' '}
                <HelpTip
                  label="seed"
                  text="The seed is the starting point for the random testing. Keep it the same and you get the same numbers every time; change it to see how much of the answer is luck."
                />{' '}
                {seed} — reproducible. Press Escape to step back a screen.
              </span>
            </div>
          </section>
        )}

        {step === 5 && result && (
          <div>
            <div className="btn-row" style={{ marginBottom: 18 }}>
              <button type="button" className="btn btn-sm" onClick={() => setStep(4)}>
                ← Change the assumptions and run again
              </button>
              <button
                type="button"
                className="btn btn-sm btn-ghost"
                onClick={() => {
                  setResult(null);
                  setStep(1);
                }}
              >
                Start a different question
              </button>
            </div>
            <ResultsView
              result={result}
              uploadSeconds={uploadSeconds}
              seed={seed}
              onSeedChange={(s) => {
                setSeed(s);
                setStep(5);
              }}
              onRerun={() => {
                setResult(null);
                setStep(5);
                setAutoStart((n) => n + 1);
              }}
            />
          </div>
        )}
      </main>
    </div>
  );
}
