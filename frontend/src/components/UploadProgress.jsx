import React, { useEffect, useRef } from 'react';
import {
  clampPct,
  humanBytes,
  humanCount,
  humanElapsed,
  humanEtaLeft,
  humanRate,
} from '../format.js';
import HelpTip from './HelpTip.jsx';

/**
 * The progress panel for a long upload — the headline fix for QA's H2/C2.
 *
 * QA's finding was that 47 of 48 screen samples over a 20-minute wait were
 * byte-identical. So this panel is built to move: the bar advances from the
 * engine's own `pct`, the row counter and throughput change as rows land, the
 * ETA counts down, the elapsed clock ticks every second (the parent feeds it),
 * and when the engine goes quiet the panel says so rather than sitting still.
 *
 * Everything the engine says arrives as `job`, the payload of
 * `GET /api/datasets/jobs/{job_id}`. Its `message` is already plain English and
 * is shown verbatim — we never rewrite it and never invent a number.
 */

const TERMINAL = {
  ready: { word: 'Ready', cls: 'is-succeeded' },
  failed: { word: 'Could not finish', cls: 'is-failed' },
  rejected: { word: 'Not accepted', cls: 'is-failed' },
  cancelled: { word: 'Cancelled', cls: 'is-cancelled' },
  queued: { word: 'Waiting to start', cls: 'is-running' },
  reading: { word: 'Reading your file', cls: 'is-running' },
  profiling: { word: 'Working out your columns', cls: 'is-running' },
  sampling: { word: 'Taking a fair sample', cls: 'is-running' },
  sending: { word: 'Sending your file', cls: 'is-running' },
};

export default function UploadProgress({
  job,
  fileName,
  elapsed = 0,
  stalled = false,
  stalledFor = 0,
  cancelling = false,
  onCancel,
  onRetry,
  onReset,
}) {
  const status = String((job && job.status) || 'queued');
  const panelRef = useRef(null);
  const done = status === 'ready';
  const failed = status === 'failed';
  const rejected = status === 'rejected';
  const cancelled = status === 'cancelled';
  const running = !done && !failed && !rejected && !cancelled;

  const pct = clampPct(job && job.pct);
  const rowsRead = Number((job && job.rows_read) || 0);
  const rowsTotal = Number((job && job.rows_total) || 0) || null;
  const bytesRead = Number((job && job.bytes_read) || 0);
  const bytesTotal = Number((job && job.bytes_total) || 0) || null;
  const rate = humanRate(job && job.rate_rows_s);
  const eta = running ? humanEtaLeft(job && job.eta_s) : null;
  const queuePosition = Number((job && job.queue_position) || 0) || null;

  // Move the reader to the panel when a job starts, so a keyboard user is not
  // left at the file picker while the work happens somewhere else on screen.
  useEffect(() => {
    if (panelRef.current && running && panelRef.current.focus) {
      panelRef.current.focus({ preventScroll: true });
    }
    // Only on the first appearance of the panel, not on every progress tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const chip = TERMINAL[status] || TERMINAL.queued;
  const pctLabel = pct === null ? 'working' : `${Math.round(pct)}%`;
  const technicalDetail = (job && (job.trace || job.error)) || '';

  return (
    <section
      className={`up-panel${failed || rejected ? ' is-failed' : ''}${
        done ? ' is-done' : ''
      }${cancelled ? ' is-cancelled' : ''}`}
      ref={panelRef}
      tabIndex={-1}
      role="region"
      aria-label="Progress of the file you are uploading"
      data-job-id={(job && job.job_id) || ''}
      data-job-status={status}
    >
      <div className="up-head">
        <span className={`status-chip ${chip.cls}`}>
          <span className="dot" />
          {chip.word}
        </span>
        {fileName && (
          <span className="badge badge-plain up-file" title={fileName}>
            {fileName}
          </span>
        )}
        {running && queuePosition && queuePosition > 1 && (
          <span className="badge badge-amber">
            You are {queuePosition}
            {ordinalSuffix(queuePosition)} in line
          </span>
        )}
        <span className="up-clock" aria-hidden="true">
          {running ? humanElapsed(elapsed) : ''}
        </span>
      </div>

      <div
        className={`progress${pct === null && running ? ' is-indeterminate' : ''}`}
        role="progressbar"
        aria-label="How much of your file we have read"
        {...(pct === null
          ? { 'aria-valuetext': 'Reading — the total is not known yet' }
          : { 'aria-valuenow': Math.round(pct), 'aria-valuemin': 0, 'aria-valuemax': 100 })}
      >
        <div
          className="progress-fill"
          style={pct === null ? undefined : { width: `${pct}%` }}
        />
      </div>

      {/* Live counters belong to a live job. Once the job has stopped they
          describe a file that is no longer being read, so they come off the
          screen rather than sitting there contradicting the status chip. */}
      {(running || done) && (
        <div className="up-meta">
          <span className="up-meta-strong">{pctLabel}</span>
          {rowsRead > 0 && (
            <span>
              <strong>{humanCount(rowsRead)}</strong> rows read
              {rowsTotal && rowsTotal > rowsRead ? ` of ${humanCount(rowsTotal)}` : ''}
            </span>
          )}
          {rate && <span>{rate}</span>}
          {eta && <span>{eta}</span>}
          {running && !rowsRead && !rate && <span>just started — no numbers yet</span>}
        </div>
      )}

      {/* The engine zeroes `bytes_read` once it moves on to working out the
          columns, so a "0 bytes — 98%" line would contradict itself. Only the
          byte counter is shown while it is still climbing; otherwise we show
          the size of the file, which stays true. */}
      {bytesTotal && (running || done) ? (
        <div className="up-bytes">
          {bytesRead > 0
            ? `${humanBytes(bytesRead)} of ${humanBytes(bytesTotal)} read`
            : `${humanBytes(bytesTotal)} file`}
        </div>
      ) : null}

      <p className="up-message" aria-live="polite">
        {(job && job.message) || 'Getting ready…'}
      </p>

      {stalled && running && (
        <p className="up-stalled">
          <span className="pulse-dot" aria-hidden="true" /> Still working — the engine has not
          reported anything for {Math.max(1, Math.round(stalledFor))} second
          {Math.round(stalledFor) === 1 ? '' : 's'}. Nothing is stuck; big files go quiet while a
          chunk is being read.
        </p>
      )}

      {done && (
        <p className="up-note up-note-good">
          Your file is ready. We are moving you on to the column matches.
        </p>
      )}

      {cancelled && (
        <p className="up-note">
          Pick a file whenever you are ready — you can also carry on with the example data, or
          with no data at all and just the numbers you type in.
        </p>
      )}

      {rejected && (
        <div className="up-note up-note-warn">
          <strong>This file is bigger than we can read.</strong> Nothing was uploaded and nothing
          was saved. Choose a different file whenever you are ready — or carry on with the example
          data, or with no data at all.
        </div>
      )}

      {failed && (
        <p className="up-note up-note-warn">
          The read stopped before it finished. Your file was not changed and nothing was saved.
        </p>
      )}

      {running && !stalled && (
        <p className="up-hint">
          You can leave this running — but if you would rather start again, cancelling is safe.{' '}
          <HelpTip
            label="cancelling"
            text="Cancelling stops the reader at its next chunk. The temporary copy of your file is deleted and nothing is added to the app. Your original file on disk is never touched."
          />
        </p>
      )}

      <div className="btn-row" style={{ marginTop: 16 }}>
        {running && onCancel && (
          <button type="button" className="btn btn-sm" onClick={onCancel} disabled={cancelling}>
            {cancelling ? <span className="spinner" /> : <span aria-hidden="true">✕</span>}
            {cancelling ? 'Stopping…' : 'Cancel this upload'}
          </button>
        )}
        {!running && onRetry && (failed || rejected || cancelled) && (
          <button type="button" className="btn btn-primary btn-sm" onClick={onRetry}>
            Try this file again
          </button>
        )}
        {!running && onReset && (
          <button type="button" className="btn btn-sm" onClick={onReset}>
            Choose a different file
          </button>
        )}
        {running && (
          <span className="muted">
            Nothing is sent anywhere — this all happens on this machine.
          </span>
        )}
      </div>

      {!!technicalDetail && (
        <details className="up-detail">
          <summary>Technical detail (for your IT team)</summary>
          <pre className="trace">{String(technicalDetail)}</pre>
        </details>
      )}
    </section>
  );
}

function ordinalSuffix(n) {
  if (n % 100 >= 11 && n % 100 <= 13) return 'th';
  return ['th', 'st', 'nd', 'rd'][n % 10] || 'th';
}
