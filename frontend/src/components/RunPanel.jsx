import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRun, getRunResult, openRunStream } from '../api.js';
import { humanElapsed } from '../format.js';
import HelpTip from './HelpTip.jsx';

const PHASE_LABEL = {
  read: 'reading',
  validate: 'checking',
  map: 'matching',
  prepare: 'preparing',
  simulate: 'simulating',
  summarize: 'summarising',
  report: 'reporting',
  run: 'running',
};

const STATUS_WORD = {
  queued: 'Waiting to start',
  running: 'Running',
  succeeded: 'Finished',
  failed: 'Could not finish',
};

/** How long the run may go quiet before we say so out loud. */
const QUIET_AFTER_S = 2;

/**
 * QA SP-05: how often we ask the engine directly how the run ended.
 *
 * The live log stream is a convenience, not the only way to learn the answer.
 * The engine closes that stream on its own clock, and a dropped connection
 * closes it sooner; before this, either one left the page spinning forever
 * because the only thing that could finish the run was the stream. Asking for
 * the result directly costs one small request every few seconds against a
 * local engine, and it means a stream that ends early still ends with the
 * answer on screen.
 */
const RESULT_POLL_MS = 4000;

/**
 * QA SP-05: how many times in a row the engine may be unreachable before we
 * stop pretending we are still waiting.
 *
 * A lost *log stream* must not end the run — the engine is still working and
 * the answer is one request away. But if the result request itself cannot
 * reach the engine five times running, the engine really has gone, and saying
 * so beats a page that spins for ever.
 */
const MAX_DEAD_POLLS = 5;

/**
 * Step 5: press the button, watch it think.
 *
 * The whole point of this screen is that the user is never looking at a black
 * box: every line the engine emits is shown, with a phase chip and elapsed
 * time. QA's follow-up finding was the other half of that promise — a long
 * silent stretch must never look like a freeze, so the elapsed clock ticks
 * every half second and a "still working" pulse appears after two quiet
 * seconds.
 */
export default function RunPanel({
  useCase,
  datasetId,
  roleMapping,
  parameters,
  explicitParams,
  seed,
  outputFormats,
  onResult,
  onError,
  onRunStart,
  resetKey,
  autoStart = 0,
}) {
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [log, setLog] = useState([]);
  const [failure, setFailure] = useState(null);
  //: set when the live log is gone but the run is not - see the note in the UI.
  const [streamNote, setStreamNote] = useState(null);
  const [runId, setRunId] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const [quietFor, setQuietFor] = useState(0);
  const logRef = useRef(null);
  const streamRef = useRef(null);
  const startedAt = useRef(null);
  const lastEventAt = useRef(0);
  const tickRef = useRef(null);
  const pollRef = useRef(null);
  //: set the moment the run reaches a terminal state, by stream or by poll -
  //: whichever gets there first. Stops the other one settling it twice.
  const settledRef = useRef(false);
  //: how many result requests in a row could not reach the engine at all.
  const deadPollsRef = useRef(0);

  const closeStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
  }, []);

  useEffect(() => () => closeStream(), [closeStream]);

  // A new run configuration means the old log no longer describes anything.
  useEffect(() => {
    closeStream();
    settledRef.current = false;
    deadPollsRef.current = 0;
    setLog([]);
    setProgress(0);
    setStatus('idle');
    setFailure(null);
    setStreamNote(null);
    setRunId(null);
    setElapsed(0);
    setQuietFor(0);
  }, [resetKey, closeStream]);

  useEffect(() => {
    if (!logRef.current) return;
    logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  // "Run again with a different seed" bumps this counter from the results page.
  useEffect(() => {
    if (autoStart > 0) run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart]);

  function run() {
    closeStream();
    settledRef.current = false;
    deadPollsRef.current = 0;
    setLog([]);
    setProgress(0);
    setFailure(null);
    setStreamNote(null);
    setStatus('queued');
    setElapsed(0);
    setQuietFor(0);
    startedAt.current = Date.now();
    lastEventAt.current = Date.now();

    const body = {
      use_case_id: useCase.id,
      dataset_id: datasetId || null,
      role_mapping: roleMapping || {},
      parameters: parameters || {},
      // Keys the user actually typed at Step 4 - these outrank values learned
      // from the uploaded file (QA DF-10 / UX-04).
      explicit_params: explicitParams || [],
      seed: Number(seed) || 42,
      output_formats: outputFormats && outputFormats.length ? outputFormats : ['html'],
    };

    // The clock is what keeps the screen alive: it changes every half second
    // whether or not the engine has anything new to say.
    tickRef.current = setInterval(() => {
      if (startedAt.current) setElapsed((Date.now() - startedAt.current) / 1000);
      setQuietFor((Date.now() - lastEventAt.current) / 1000);
    }, 500);

    createRun(body)
      .then((res) => {
        const id = res && res.run_id;
        if (!id) throw new Error('The engine did not give us a run to watch.');
        setRunId(id);
        if (onRunStart) onRunStart(id);

        // QA SP-05: the fallback. If the stream ends before the run does - the
        // engine's own stream clock, a proxy, a dropped socket - this is what
        // turns "spinning forever" into the answer. It also settles the run if
        // the stream's own `result` event was missed.
        //
        // It is started *before* the stream is opened, on purpose: the live log
        // is the nice-to-have and the answer is the point, so nothing that
        // happens to the log may stop us asking for the answer.
        pollRef.current = setInterval(async () => {
          if (settledRef.current) return;
          try {
            const payload = await getRunResult(id);
            if (settledRef.current || !payload || !payload.run_id) return;
            if (payload.status === 'running' || payload.status === 'queued') {
              deadPollsRef.current = 0;
              return;
            }
            settledRef.current = true;
            closeStream();
            const failed = payload.status === 'failed';
            setStatus(failed ? 'failed' : 'succeeded');
            setProgress(1);
            onResult(payload);
          } catch (err) {
            if (err && err.status === 404) {
              settledRef.current = true;
              closeStream();
              setStatus('failed');
              setFailure({ message: err.message, trace: '' });
              return;
            }
            // A real answer that came back unreadable is not the same as an
            // engine that is not there. Only a request that never reached the
            // engine counts towards giving up.
            if (err && err.status) return;
            deadPollsRef.current += 1;
            if (deadPollsRef.current >= MAX_DEAD_POLLS) {
              settledRef.current = true;
              closeStream();
              setStatus('failed');
              setFailure({
                message:
                  'We lost touch with the simulation engine and could not get this run back. '
                  + 'The engine may have stopped. Start it again and press “Run it again” — '
                  + 'nothing you set up is lost.',
                trace: '',
              });
            }
          }
        }, RESULT_POLL_MS);

        streamRef.current = openRunStream(
          id,
          (name, data) => {
            lastEventAt.current = Date.now();
            setQuietFor(0);
            if (name === 'log') {
              setLog((prev) => [...prev, data]);
              if (typeof data.progress === 'number' && isFinite(data.progress)) {
                setProgress((p) => Math.max(p, Math.min(1, data.progress)));
              }
            } else if (name === 'status') {
              setStatus(data.status || 'running');
            } else if (name === 'result') {
              settledRef.current = true;
              setStatus('succeeded');
              setProgress(1);
              closeStream();
              onResult(data);
            } else if (name === 'error') {
              settledRef.current = true;
              setStatus('failed');
              closeStream();
              setFailure(data);
              if (onError) onError(data);
            }
          },
          (msg) => {
            // QA SP-05: losing the *log* is not losing the *run*. The engine is
            // still working; the stream was only the running commentary. Say
            // what was lost, keep the clock and the result poll going, and let
            // the answer land in its own time.
            setStreamNote(msg);
          },
        );
      })
      .catch((err) => {
        setStatus('failed');
        setFailure({
          message: err.message || 'We could not start this run.',
          trace: err.detail || '',
        });
      });
  }
  const running = status === 'queued' || status === 'running';
  const quiet = running && quietFor > QUIET_AFTER_S;
  const lastEvent = log.length ? log[log.length - 1] : null;
  const pct = Math.round(Math.min(1, Math.max(0, progress)) * 100);
  // Before the first log line we genuinely do not know how far along we are,
  // so the bar animates rather than showing a confident "0%".
  const indeterminate = running && pct === 0 && !log.length;

  return (
    <>
      <div className="run-hero">
        <span
          className={`status-chip${running ? ' is-running' : ''}${
            status === 'succeeded' ? ' is-succeeded' : ''
          }${status === 'failed' ? ' is-failed' : ''}`}
        >
          <span className="dot" />
          {STATUS_WORD[status] || 'Not started yet'}
        </span>

        {running && (
          <span className="badge badge-model">
            Running: {(useCase.models || []).join(' + ') || 'the simulation'}
          </span>
        )}
        {running && lastEvent && (
          <span className="badge badge-plain">
            now: {PHASE_LABEL[lastEvent.phase] || lastEvent.phase || 'working'}
          </span>
        )}
        {runId && <span className="badge badge-plain">run {runId.slice(0, 8)}</span>}
      </div>

      <div
        className={`progress${indeterminate ? ' is-indeterminate' : ''}`}
        role="progressbar"
        aria-label="How far through the simulation we are"
        {...(indeterminate
          ? { 'aria-valuetext': 'Starting up' }
          : { 'aria-valuenow': pct, 'aria-valuemin': 0, 'aria-valuemax': 100 })}
      >
        <div
          className="progress-fill"
          style={indeterminate ? undefined : { width: `${pct}%` }}
        />
      </div>
      <div className="progress-meta">
        <span>{indeterminate ? 'starting up…' : `${pct}% done`}</span>
        <span>{elapsed ? `${humanElapsed(elapsed)} elapsed` : 'not started'}</span>
      </div>

      {quiet && (
        <p className="up-stalled">
          <span className="pulse-dot" aria-hidden="true" /> Still working — no news for{' '}
          {Math.round(quietFor)} second{Math.round(quietFor) === 1 ? '' : 's'}. The engine is
          busy on a big step and will report the moment it is done. Nothing is stuck, and the
          clock above keeps counting.
        </p>
      )}

      {/* QA SP-05: the live log can go without the run going with it. */}
      {streamNote && running && (
        <p className="inline-note" style={{ marginTop: 12 }}>
          <span aria-hidden="true">⚠</span> {streamNote}
        </p>
      )}

      <div className="btn-row" style={{ margin: '22px 0 18px' }}>
        <button
          type="button"
          className="btn btn-primary btn-big"
          disabled={running}
          onClick={run}
        >
          {running ? <span className="spinner" /> : <span aria-hidden="true">▶</span>}
          {running ? 'Running…' : status === 'idle' ? 'Run the simulation' : 'Run it again'}
        </button>
        <span className="muted">
          Same seed means the same numbers, every time.{' '}
          <HelpTip
            label="seed"
            text="A seed is the starting point for the random testing. Using the same seed gives you exactly the same numbers again, so you can show someone else and they will see what you saw."
          />
        </span>
      </div>

      <div className="section-title">
        Live log{' '}
        <HelpTip
          label="the live log"
          text="Every step the model takes, written as it happens. If something looks odd, this is where you will see why."
        />
      </div>
      <p className="section-sub">
        Nothing is hidden: this is the model talking through what it is doing, in order.
      </p>

      <div className="terminal" ref={logRef}>
        {!log.length && !failure && (
          <div className="log-line">
            <span className="log-t">—</span>
            <span className="log-phase">ready</span>
            <span className="log-msg">
              {running ? 'Starting…' : 'Press “Run the simulation” to begin.'}
            </span>
          </div>
        )}
        {log.map((ev, i) => (
          <LogLine key={i} event={ev} />
        ))}
        {running && log.length > 0 && (
          <div className="log-line is-pending">
            <span className="log-t">{elapsed.toFixed(1)}s</span>
            <span className="log-phase">{quiet ? 'waiting' : 'working'}</span>
            <span className="log-msg">
              <span className="pulse-dot" aria-hidden="true" />{' '}
              {quiet ? 'still working…' : 'working…'}
            </span>
          </div>
        )}
        {failure && (
          <div className="log-line is-error">
            <span className="log-t">—</span>
            <span className="log-phase">error</span>
            <span className="log-msg">{failure.message}</span>
          </div>
        )}
      </div>

      {failure && <FailureCard failure={failure} onRetry={run} />}
    </>
  );
}

function LogLine({ event }) {
  const level = event.level || 'info';
  const t = Number(event.t || 0);
  return (
    <>
      <div className={`log-line is-${level}`}>
        <span className="log-t">{t.toFixed(1)}s</span>
        <span className="log-phase">{PHASE_LABEL[event.phase] || event.phase || 'run'}</span>
        <span className="log-msg">{event.message}</span>
      </div>
      {event.detail && <div className="log-detail">{event.detail}</div>}
    </>
  );
}

function FailureCard({ failure, onRetry }) {
  return (
    <div className="error-card" style={{ marginTop: 18 }}>
      <div className="error-title">We could not finish this run</div>
      <div className="error-msg">{failure.message}</div>
      <div className="btn-row" style={{ marginTop: 14 }}>
        {onRetry && (
          <button type="button" className="btn btn-sm btn-primary" onClick={onRetry}>
            Try this run again
          </button>
        )}
        <span className="muted">
          Your assumptions and column matches are exactly as you left them — nothing to re-enter.
        </span>
      </div>
      {failure.trace && (
        <details style={{ marginTop: 12 }}>
          <summary className="muted" style={{ cursor: 'pointer' }}>
            Technical detail (for your IT team)
          </summary>
          <pre className="trace">{failure.trace}</pre>
        </details>
      )}
    </div>
  );
}
