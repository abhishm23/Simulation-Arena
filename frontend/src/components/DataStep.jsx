import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  cancelJob,
  getJob,
  getLimits,
  getSamples,
  isTerminal,
  loadSample,
  openJobStream,
  pasteDataset,
  startUpload,
} from '../api.js';
import { humanBytes, humanCount } from '../format.js';
import HelpTip from './HelpTip.jsx';
import UploadProgress from './UploadProgress.jsx';

const TABS = [
  { id: 'upload', label: 'Upload a file' },
  { id: 'sample', label: 'Use an example' },
  { id: 'paste', label: 'Paste a table' },
  { id: 'manual', label: 'Just answer a few questions' },
];

/** How long the reader may go quiet before we say so out loud. */
const STALL_AFTER_S = 2;
/** How many times we will quietly reopen a dropped live connection. */
const MAX_RECONNECTS = 30;

/**
 * Split pasted text (CSV or TSV, and anything Excel puts on the clipboard)
 * into rows of objects. Handles quoted fields and doubled quotes.
 */
export function parseDelimited(text) {
  const raw = String(text || '').replace(/\r\n?/g, '\n').trim();
  if (!raw) return { rows: [], columns: [] };

  const lines = raw.split('\n').filter((l) => l.trim() !== '');
  if (!lines.length) return { rows: [], columns: [] };

  const delim = pickDelimiter(lines[0]);
  const header = splitLine(lines[0], delim);

  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cells = splitLine(lines[i], delim);
    if (!cells.some((c) => c !== '')) continue;
    const obj = {};
    header.forEach((h, j) => {
      obj[h] = cells[j] === undefined ? '' : cells[j];
    });
    rows.push(obj);
  }
  return { rows, columns: header };
}

function pickDelimiter(firstLine) {
  const outside = stripQuoted(firstLine);
  const counts = { '\t': 0, ',': 0, ';': 0, '|': 0 };
  for (const ch of outside) if (ch in counts) counts[ch] += 1;
  let best = ',';
  for (const ch of Object.keys(counts)) if (counts[ch] > counts[best]) best = ch;
  return best;
}

function stripQuoted(line) {
  let out = '';
  let inQ = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      inQ = !inQ;
      continue;
    }
    if (!inQ) out += ch;
  }
  return out;
}

function splitLine(line, delim) {
  const cells = [];
  let cur = '';
  let inQ = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (inQ && line[i + 1] === '"') {
        cur += '"';
        i += 1;
      } else {
        inQ = !inQ;
      }
      continue;
    }
    if (ch === delim && !inQ) {
      cells.push(cur.trim());
      cur = '';
      continue;
    }
    cur += ch;
  }
  cells.push(cur.trim());
  return cells;
}

/** The refusal a person can act on, worded the same way the engine words it. */
function tooBigMessage(size, limit) {
  return (
    `That file is ${humanBytes(size)}, and the limit is ${humanBytes(limit)}. ` +
    `Save it as Parquet or gzip the CSV and it will be far smaller — then try again.`
  );
}

/** What we say once a file has been read. */
function noteFor(profile, fallbackName) {
  const rows = Number(profile.n_rows || 0);
  const cols = profile.n_cols || (profile.columns || []).length;
  const name = profile.source_name || fallbackName || 'your file';
  const sampled = (profile.warnings || []).some((w) => /sampl/i.test(String(w)));
  const base = `Read ${humanCount(rows)} rows and ${cols} columns from ${name}.`;
  return sampled
    ? `${base} Your file was bigger than we work with, so we ran on a fair sample of it — see the note below.`
    : base;
}

export default function DataStep({
  onBusyChange, profile, onProfile, onManual, datasetId }) {
  const [tab, setTab] = useState(profile || datasetId ? 'upload' : 'sample');

  // -- ceilings ------------------------------------------------------------
  const [limits, setLimits] = useState(null);
  const [limitsError, setLimitsError] = useState(null);
  const [limitsLoaded, setLimitsLoaded] = useState(false);

  // -- the current reading job --------------------------------------------
  const [job, setJob] = useState(null);
  const [jobName, setJobName] = useState('');
  const [sending, setSending] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [stalledFor, setStalledFor] = useState(0);
  const [streamNote, setStreamNote] = useState(null);

  // -- other async work ----------------------------------------------------
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [samples, setSamples] = useState([]);
  const [samplesError, setSamplesError] = useState(null);
  const [samplesLoaded, setSamplesLoaded] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [pasteText, setPasteText] = useState('');
  const [pastedName, setPastedName] = useState('Pasted table');

  const fileInput = useRef(null);
  const streamRef = useRef(null);
  const pollRef = useRef(null);
  const tickRef = useRef(null);
  const reconnectRef = useRef(0);
  const lastEventRef = useRef(0);
  const startedAtRef = useRef(0);
  const statusRef = useRef('');
  const jobIdRef = useRef('');
  const fileRef = useRef(null);
  const dropRef = useRef(null);
  //: true only while we actually want to be listening. Without it, closing the
  //: stream on purpose would fire the browser's own error event and we would
  //: cheerfully start polling a job we had already finished with.
  const watchingRef = useRef(false);

  // "In flight" is narrower than "we have a job object": once a job reaches a
  // terminal state the panel must switch from *watching* it to *reporting* it,
  // which is what puts the retry and start-again buttons on screen.
  const inFlight = sending || (!!job && !isTerminal(job.status));
  const pastePreview = useMemo(() => parseDelimited(pasteText), [pasteText]);

  // QA UX-24: the result page used to show only the simulation time, hiding the
  // upload - which was 94-99.98% of the real wait. Hand the read time over too
  // so the report can show a true end-to-end total.
  const deliverProfile = useCallback(
    (prof, note) => {
      const ms = startedAtRef.current ? Date.now() - startedAtRef.current : 0;
      onProfile(prof, note, ms / 1000);
    },
    [onProfile],
  );

  // QA UX-02: the wizard's primary button used to stay clickable for the whole
  // read. Tell the parent when we are busy so it can guard the button.
  useEffect(() => {
    if (onBusyChange) onBusyChange(inFlight);
  }, [inFlight, onBusyChange]);

  // -- housekeeping --------------------------------------------------------
  const stopWatching = useCallback(() => {
    watchingRef.current = false;
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

  useEffect(() => stopWatching, [stopWatching]);

  useEffect(() => {
    let alive = true;
    getLimits()
      .then((res) => alive && setLimits(res || null))
      .catch((err) => alive && setLimitsError(err.message))
      .finally(() => alive && setLimitsLoaded(true));
    return () => {
      alive = false;
    };
  }, []);

  const loadSamples = useCallback(() => {
    setSamplesError(null);
    setSamplesLoaded(false);
    getSamples()
      .then((list) => setSamples(Array.isArray(list) ? list : []))
      .catch((err) => setSamplesError(err.message))
      .finally(() => setSamplesLoaded(true));
  }, []);

  useEffect(() => {
    loadSamples();
  }, [loadSamples]);

  // One clock for the whole panel. This is what guarantees the screen changes
  // at least once a second even if the engine is quiet for a moment.
  const startTicker = useCallback(() => {
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = setInterval(() => {
      setElapsed((Date.now() - startedAtRef.current) / 1000);
      setStalledFor((Date.now() - lastEventRef.current) / 1000);
    }, 1000);
  }, []);

  const touch = useCallback(() => {
    lastEventRef.current = Date.now();
    setStalledFor(0);
  }, []);

  const finishJob = useCallback(
    (payload, trace) => {
      stopWatching();
      statusRef.current = String(payload.status || '');
      setJob(trace ? { ...payload, trace } : payload);
      setSending(false);
      setCancelling(false);
    },
    [stopWatching],
  );

  // -- watching a job ------------------------------------------------------
  const pollJob = useCallback(
    (jobId, name) => {
      if (pollRef.current) return;
      const tick = () => {
        getJob(jobId)
          .then((snap) => {
            touch();
            setJob(snap);
            statusRef.current = String(snap.status || '');
            if (isTerminal(snap.status)) {
              // `done` carries the profile; the live stream is the only place
              // it comes from, so keep trying to reopen it until we see it.
              if (snap.status !== 'ready') stopWatching();
            }
          })
          .catch(() => {
            /* the engine may be busy; keep the clock running and try again */
          });
        if (
          !streamRef.current &&
          reconnectRef.current < MAX_RECONNECTS &&
          !(isTerminal(statusRef.current) && statusRef.current !== 'ready')
        ) {
          reconnectRef.current += 1;
          watchStream(jobId, name);
        }
      };
      pollRef.current = setInterval(tick, 1500);
      tick();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [touch, stopWatching],
  );

  const watchStream = useCallback(
    (jobId, name) => {
      watchingRef.current = true;
      streamRef.current = openJobStream(
        jobId,
        (eventName, data) => {
          touch();
          if (eventName === 'progress') {
            statusRef.current = String(data.status || '');
            setJob(data);
            return;
          }
          if (eventName === 'done') {
            stopWatching();
            statusRef.current = 'ready';
            const next = data.profile || null;
            if (!next) {
              setJob(null);
              setError({
                message:
                  'Your file was read, but the description of it did not come back. Please upload it again.',
                detail: JSON.stringify(data, null, 2),
              });
              return;
            }
            setJob(null);
            setStreamNote(null);
            deliverProfile(next, noteFor(next, name));
            return;
          }
          if (eventName === 'error') {
            // The engine also reports "cancelled" and "rejected" as an error
            // event, so ask the job what actually happened instead of guessing
            // from the wording.
            getJob(jobId)
              .then((snap) => finishJob(snap, data.trace || ''))
              .catch(() =>
                finishJob(
                  {
                    job_id: jobId,
                    status: 'failed',
                    message: data.message || 'We could not read that file.',
                  },
                  data.trace || '',
                ),
              );
          }
        },
        (msg) => {
          if (!watchingRef.current) return;
          streamRef.current = null;
          setStreamNote(msg);
          pollJob(jobId, name);
        },
      );
    },
    [finishJob, onProfile, pollJob, stopWatching, touch],
  );

  // -- starting work -------------------------------------------------------
  function handleFiles(files) {
    const file = files && files[0];
    if (!file) return;
    fileRef.current = file;
    setError(null);
    setStreamNote(null);
    setJobName(file.name);
    reconnectRef.current = 0;
    startedAtRef.current = Date.now();
    lastEventRef.current = Date.now();
    setElapsed(0);
    setStalledFor(0);
    startTicker();

    // Refuse an oversized file *before* it goes anywhere near the engine.
    const limit = limits && Number(limits.max_file_bytes);
    if (limit && file.size > limit) {
      stopWatching();
      statusRef.current = 'rejected';
      setSending(false);
      setJob({
        job_id: null,
        status: 'rejected',
        message: tooBigMessage(file.size, limit),
        bytes_total: file.size,
        bytes_read: 0,
        pct: null,
      });
      return;
    }

    setJob(null);
    setSending(true);
    startUpload(file)
      .then((res) => {
        if (!res || !res.job_id) {
          throw Object.assign(
            new Error('The engine accepted the file but did not give us a job to watch.'),
            { detail: JSON.stringify(res || {}, null, 2) },
          );
        }
        if (isTerminal(res.status)) {
          // e.g. a file the engine refused on size: a finished job, not a task.
          finishJob(res, '');
          return;
        }
        setSending(false);
        statusRef.current = String(res.status || 'queued');
        jobIdRef.current = res.job_id;
        setJob(res);
        watchStream(res.job_id, file.name);
      })
      .catch((err) => {
        stopWatching();
        setSending(false);
        setJob(null);
        setError({
          message: err.message || 'We could not read that file.',
          detail: err.detail || '',
        });
      });
  }

  function handleSample(name) {
    setBusy(true);
    setError(null);
    startedAtRef.current = Date.now();
    lastEventRef.current = Date.now();
    setElapsed(0);
    startTicker();
    loadSample(name)
      .then((prof) => {
        stopWatching();
        deliverProfile(prof, noteFor(prof, name));
      })
      .catch((err) => setError({ message: err.message, detail: err.detail || '' }))
      .finally(() => {
        setBusy(false);
        stopWatching();
      });
  }

  function handlePaste() {
    const { rows, columns } = parseDelimited(pasteText);
    if (!rows.length || !columns.length) {
      setError({
        message:
          'We could not see a table in there. Paste rows with a header line on top — the first row should be your column names.',
        detail: '',
      });
      return;
    }
    const name = pastedName || 'Pasted table';
    setBusy(true);
    setError(null);
    setStreamNote(null);
    reconnectRef.current = 0;
    startedAtRef.current = Date.now();
    lastEventRef.current = Date.now();
    setElapsed(0);
    setStalledFor(0);
    startTicker();
    pasteDataset(name, rows)
      .then((res) => {
        // Two shapes come back from this call: a small paste is profiled on the
        // spot and answers with a DatasetProfile; a big one becomes a job.
        if (res && res.job_id) {
          setJobName(name);
          statusRef.current = String(res.status || 'queued');
          jobIdRef.current = res.job_id;
          setJob(res);
          watchStream(res.job_id, name);
          return;
        }
        if (res && res.dataset_id) {
          stopWatching();
          deliverProfile(res, noteFor(res, name));
          return;
        }
        throw Object.assign(new Error('The engine did not send back your table.'), {
          detail: JSON.stringify(res || {}, null, 2),
        });
      })
      .catch((err) => {
        stopWatching();
        setJob(null);
        setError({ message: err.message, detail: err.detail || '' });
      })
      .finally(() => setBusy(false));
  }

  function handleCancel() {
    const id = (job && job.job_id) || jobIdRef.current;
    setCancelling(true);
    if (!id) {
      // Nothing has reached the engine yet (still sending, or refused here).
      stopWatching();
      setSending(false);
      setJob({
        job_id: null,
        status: 'cancelled',
        message: 'You cancelled this upload. Nothing was saved.',
        pct: null,
      });
      setCancelling(false);
      return;
    }
    cancelJob(id)
      .then((res) => {
        finishJob({
          ...(job || {}),
          job_id: id,
          status: (res && res.status) || 'cancelled',
          pct: null,
          eta_s: null,
          queue_position: null,
          message:
            (job && job.status === 'queued'
              ? 'You cancelled this upload before it started. '
              : 'You cancelled this upload. ') + 'Nothing was saved.',
        });
      })
      .catch((err) => {
        setCancelling(false);
        setError({
          message: `We could not stop that upload: ${err.message}`,
          detail: err.detail || '',
        });
      });
  }

  function backToPicker(note) {
    stopWatching();
    setJob(null);
    setSending(false);
    setCancelling(false);
    setError(null);
    setStreamNote(null);
    setJobName('');
    setElapsed(0);
    setStalledFor(0);
    statusRef.current = '';
    jobIdRef.current = '';
    fileRef.current = null;
    if (fileInput.current) fileInput.current.value = '';
    // Send the keyboard user back to where the file is chosen, so cancelling
    // does not strand them on a panel that has just disappeared.
    if (dropRef.current && dropRef.current.focus) {
      dropRef.current.focus({ preventScroll: false });
    }
    if (note) setError({ message: note, detail: '' });
  }

  function retryLastFile() {
    const file = fileRef.current;
    if (!file) {
      backToPicker(null);
      return;
    }
    handleFiles([file]);
  }

  const limitsLine = limits
    ? `Up to ${limits.max_file_human || humanBytes(limits.max_file_bytes)} per file`
    : limitsLoaded && limitsError
      ? 'CSV, Excel or Parquet'
      : 'Checking what this machine can take…';

  return (
    <>
      <div className="tabs" role="tablist" aria-label="Where your data comes from">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            id={`tab-${t.id}`}
            aria-selected={tab === t.id}
            aria-controls={`panel-${t.id}`}
            className={`tab${tab === t.id ? ' is-active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'upload' && (
        <div id="panel-upload" role="tabpanel" aria-labelledby="tab-upload">
          <p className="tab-note">
            Drag a file in, or click to pick one. We read CSV, Excel (.xlsx / .xls) and
            Parquet. Your file stays on this machine.
          </p>

          <div
            className={`dropzone${dragOver ? ' is-over' : ''}${inFlight ? ' is-busy' : ''}`}
            role="button"
            tabIndex={0}
            ref={dropRef}
            aria-label="Choose a file to upload"
            aria-disabled={inFlight}
            onClick={() => !inFlight && fileInput.current && fileInput.current.click()}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                if (!inFlight && fileInput.current) fileInput.current.click();
              }
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              if (!inFlight) handleFiles(e.dataTransfer.files);
            }}
          >
            <div className="dropzone-icon" aria-hidden="true">
              📄
            </div>
            {/* QA UX-23: "CSV, Excel or Parquet - with a header row on top" was a
                dangling fragment with no verb. Give it a full sentence. */}
            <div className="dropzone-title">
              {inFlight ? 'Reading your file…' : 'Drop your file here'}
            </div>
            <div className="dropzone-sub">
              {inFlight ? (
                <span className="spinner" />
              ) : (
                <>
                  We read CSV, Excel and Parquet files — just put the column
                  headings in the top row.
                  <br />
                  <strong>{limitsLine}</strong>
                </>
              )}
            </div>
            {!inFlight && (
              <div className="dropzone-limits">
                {limits && (
                  <>
                    <span className="badge badge-plain">
                      Up to {limits.max_working_rows_human || `${humanCount(limits.max_working_rows)} rows`} per file
                    </span>
                    <span className="badge badge-plain">
                      {limits.max_concurrent_uploads === 1
                        ? 'One big file at a time'
                        : `${limits.max_concurrent_uploads} big files at a time`}
                    </span>
                    <span className="badge badge-plain">
                      Give up after {Math.round((limits.parse_timeout_s || 900) / 60)} minutes
                    </span>
                  </>
                )}
                {!limits && limitsLoaded && limitsError && (
                  <span className="badge badge-amber">
                    We could not check the size limits
                    <HelpTip
                      label="the size limits"
                      text="We normally ask the engine how big a file it can take so we can warn you before an upload starts. That check did not come back — you can still try, and the engine will refuse anything too big."
                    />
                  </span>
                )}
              </div>
            )}
          </div>
          <input
            ref={fileInput}
            type="file"
            accept=".csv,.tsv,.txt,.xlsx,.xls,.parquet"
            style={{ display: 'none' }}
            aria-hidden="true"
            tabIndex={-1}
            onChange={(e) => {
              handleFiles(e.target.files);
              e.target.value = '';
            }}
          />
        </div>
      )}

      {tab === 'sample' && (
        <div id="panel-sample" role="tabpanel" aria-labelledby="tab-sample">
          <p className="tab-note">
            Not sure what to bring? Start with one of these — they are made-up files that
            look like the real thing, so you can see the whole flow before using your own.
          </p>
          {samplesError && (
            <div className="error-card" style={{ marginBottom: 14 }}>
              <div className="error-title">We could not list the example files</div>
              <div className="error-msg">{samplesError}</div>
              <button
                type="button"
                className="btn btn-sm"
                style={{ marginTop: 12 }}
                onClick={loadSamples}
              >
                Try again
              </button>
            </div>
          )}
          <div className="sample-list">
            {samples.map((s) => (
              <button
                type="button"
                key={s.name}
                className="sample-row"
                disabled={busy}
                onClick={() => handleSample(s.name)}
              >
                <span className="sample-main">
                  <span className="sample-title">{s.title || s.name}</span>
                  <span className="sample-blurb">{s.blurb || ''}</span>
                </span>
                <span className="badge badge-plain">
                  {humanCount(s.rows)} rows
                </span>
                <span className="btn btn-sm btn-primary" aria-hidden="true">
                  {busy ? 'Loading…' : 'Use this'}
                </span>
              </button>
            ))}
            {!samples.length && !samplesError && !samplesLoaded && (
              <div className="empty-note">
                <span className="spinner" /> Looking for the example files…
              </div>
            )}
            {!samples.length && !samplesError && samplesLoaded && (
              <div className="empty-note">
                <strong style={{ color: 'var(--text)' }}>
                  There are no example files on this machine.
                </strong>
                <p style={{ marginTop: 8 }}>
                  That is not a problem — upload your own file, paste a table, or carry on with
                  the numbers you type in. Nothing else in the app depends on these.
                </p>
              </div>
            )}
          </div>
          {busy && (
            <p className="muted" style={{ marginTop: 12 }}>
              <span className="spinner" /> Loading the example and reading its columns…
            </p>
          )}
        </div>
      )}

      {tab === 'paste' && (
        <div id="panel-paste" role="tabpanel" aria-labelledby="tab-paste">
          <p className="tab-note">
            Copy a block of cells out of Excel or a spreadsheet and paste it below. Keep the
            header row on top — that is how we learn your column names.
          </p>
          <label className="param-label" htmlFor="paste-name">
            What shall we call this?
          </label>
          <input
            id="paste-name"
            className="input"
            style={{ marginTop: 8, marginBottom: 12 }}
            value={pastedName}
            onChange={(e) => setPastedName(e.target.value)}
          />
          <textarea
            className="input"
            rows={10}
            aria-label="Paste your table here"
            placeholder={'deal_id,deal_value,margin_pct,discount_pct,won\nD-1001,250000,42,18,1\nD-1002,90000,38,25,0'}
            value={pasteText}
            onChange={(e) => setPasteText(e.target.value)}
          />
          <div className="btn-row" style={{ marginTop: 14 }}>
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy || !pasteText.trim()}
              onClick={handlePaste}
            >
              {busy ? <span className="spinner" /> : null}
              {busy ? 'Reading…' : 'Read this table'}
            </button>
            <span className="muted">
              {pastePreview.rows.length > 0
                ? `${humanCount(pastePreview.rows.length)} rows and ${humanCount(
                    pastePreview.columns.length,
                  )} columns detected`
                : 'Nothing pasted yet'}
            </span>
          </div>
          {pastePreview.rows.length === 0 && !pasteText.trim() && (
            <div className="empty-note" style={{ marginTop: 14, textAlign: 'left' }}>
              Paste anything from a spreadsheet — a header row, then your rows. Very large
              pastes are read in the background just like a file, and you will see the same
              progress bar.
            </div>
          )}
        </div>
      )}

      {tab === 'manual' && (
        <div id="panel-manual" role="tabpanel" aria-labelledby="tab-manual">
          <div className="empty-note" style={{ textAlign: 'left' }}>
            <strong style={{ color: 'var(--text)' }}>
              No file? That is fine — we will use the numbers you enter.
            </strong>
            <p style={{ marginTop: 10 }}>
              Skip the data completely. The simulation still works: it runs entirely on the
              assumptions you set on the next screen. You will get the same charts, the same
              plain-English answer and the same downloads — just built from your judgement
              instead of your history.
            </p>
          </div>
          <div className="btn-row" style={{ marginTop: 16 }}>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => {
                onProfile(null, 'Using the numbers you type in — no data file needed.');
                onManual();
              }}
            >
              Continue with no data
            </button>
            {(profile || datasetId) && (
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() => {
                  onProfile(null, 'Switched to the numbers you type in.');
                  onManual();
                }}
              >
                Drop the file I loaded and use my numbers instead
              </button>
            )}
          </div>
        </div>
      )}

      {inFlight && (
        <UploadProgress
          job={
            sending
              ? {
                  status: 'sending',
                  message: `Sending “${jobName}” to the reader on this machine…`,
                  pct: null,
                  rows_read: 0,
                }
              : job
          }
          fileName={jobName}
          elapsed={elapsed}
          stalled={stalledFor > STALL_AFTER_S}
          stalledFor={stalledFor}
          cancelling={cancelling}
          onCancel={handleCancel}
        />
      )}

      {streamNote && inFlight && (
        <p className="inline-note" style={{ marginTop: 10 }}>
          {streamNote}
        </p>
      )}

      {!inFlight && job && (
        <UploadProgress
          job={job}
          fileName={jobName}
          elapsed={elapsed}
          stalled={false}
          cancelling={false}
          onRetry={fileRef.current ? retryLastFile : undefined}
          onReset={() => backToPicker(null)}
        />
      )}

      {error && (
        <div className="error-card" style={{ marginTop: 16 }}>
          <div className="error-title">We could not use that</div>
          <div className="error-msg">{error.message}</div>
          <div className="btn-row" style={{ marginTop: 14 }}>
            {fileRef.current && !inFlight && (
              <button type="button" className="btn btn-sm btn-primary" onClick={retryLastFile}>
                Try this file again
              </button>
            )}
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => backToPicker(null)}
            >
              Start again
            </button>
          </div>
          {error.detail && (
            <details style={{ marginTop: 12 }}>
              <summary className="muted" style={{ cursor: 'pointer' }}>
                Technical detail (for your IT team)
              </summary>
              <pre className="trace">{String(error.detail)}</pre>
            </details>
          )}
        </div>
      )}

      {!job && profile && <DatasetSummary profile={profile} />}
    </>
  );
}

/** The plain-English names for the kinds of column we detect. */
const DTYPE_WORDS = {
  numeric: 'numbers',
  integer: 'whole numbers',
  float: 'numbers',
  categorical: 'categories',
  category: 'categories',
  text: 'free text',
  string: 'free text',
  identifier: 'an ID or reference',
  id: 'an ID or reference',
  datetime: 'dates and times',
  date: 'dates',
  boolean: 'yes / no',
  bool: 'yes / no',
};

function dtypeWords(dtype) {
  const key = String(dtype || '').toLowerCase().trim();
  return DTYPE_WORDS[key] || 'values';
}

function DatasetSummary({ profile }) {
  const rows = Number(profile.n_rows || 0);
  const cols = profile.n_cols || (profile.columns || []).length;
  const warnings = profile.warnings || [];
  const sampling = warnings.filter((w) => /sampl|representative/i.test(String(w)));
  const otherWarnings = warnings.filter((w) => !sampling.includes(w));

  return (
    <div className="ds-summary">
      <div className="ds-headline">
        <span aria-hidden="true">✅</span>
        {`Read ${humanCount(rows)} rows and ${cols} columns`}
      </div>
      <p className="section-sub" style={{ marginTop: 6, marginBottom: 0 }}>
        From <strong style={{ color: 'var(--text)' }}>{profile.source_name}</strong>. Here is
        what we found in each column — the next step asks you to match them up.
      </p>

      {!!sampling.length && (
        <div className="warn-block is-prominent">
          <div className="warn-title">
            Read this first — we did not use every row
          </div>
          <ul>
            {sampling.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
          <p className="warn-after">
            For the kinds of question this app answers, a fair sample this size gives the same
            answer as the whole file — to within the noise you would see anyway.
          </p>
        </div>
      )}

      <div className="cols-wrap">
        {(profile.columns || []).map((c) => (
          <span className="col-chip" key={c.name} title={c.name}>
            <strong>{c.name}</strong>
            <span className="dtype">{dtypeWords(c.dtype)}</span>
          </span>
        ))}
        {!(profile.columns || []).length && (
          <span className="muted">No columns found.</span>
        )}
      </div>

      {!!otherWarnings.length && (
        <div className="warn-block">
          <div className="warn-title">
            {otherWarnings.length} thing{otherWarnings.length === 1 ? '' : 's'} worth knowing
            about this file
          </div>
          <ul>
            {otherWarnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
