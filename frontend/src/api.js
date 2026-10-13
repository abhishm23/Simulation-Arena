// Thin wrapper over the Simulation Arena HTTP API.
//
// Every function here maps 1:1 onto a row of the contract table in BRIEFING.md
// and QA_FIX_BRIEF.md. The backend never returns HTML errors - it returns JSON
// with a plain-English `detail` - so `apiError` pulls that message out for the
// user to read. If the backend is simply not there, the message says so in a
// sentence instead of surfacing "Failed to fetch".

const BASE = '/api';

//: The one sentence for "there is no engine answering", used both when fetch
//: itself fails and when the dev proxy answers 500 because nothing is upstream.
const UNREACHABLE =
  'We could not reach the simulation engine. It may not be running yet — start it, then try again. Your data never left this machine.';

export class ApiError extends Error {
  constructor(message, status, payload, path) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
    //: A short machine-ish line for the collapsible "technical detail".
    this.detail = payload
      ? typeof payload === 'string'
        ? payload
        : JSON.stringify(payload, null, 2)
      : status
        ? `HTTP ${status} from ${BASE}${path || ''}`
        : 'The request never reached the engine (the connection failed).';
  }
}

function detailOf(payload) {
  if (!payload) return null;
  if (typeof payload === 'string') return payload;
  if (typeof payload.detail === 'string') return payload.detail;
  if (Array.isArray(payload.detail)) {
    return payload.detail
      .map((d) => (typeof d === 'string' ? d : d && d.msg) || '')
      .filter(Boolean)
      .join('; ');
  }
  if (typeof payload.message === 'string') return payload.message;
  if (typeof payload.error === 'string') return payload.error;
  return null;
}

/** The sentence a person reads when a request comes back non-2xx. */
function friendlyForStatus(status) {
  if (status === 404) {
    return 'We could not find that on the simulation engine. It may have been cleared - try again from the start.';
  }
  if (status === 413) {
    return 'That file was too big for the engine to accept. Save it as Parquet or gzip the CSV and try again.';
  }
  if (status === 422) {
    return 'The engine could not understand that request. Try again - if it keeps happening, the technical detail below will help your IT team.';
  }
  if (status >= 500) {
    return 'The simulation engine hit a problem and stopped. Nothing you did caused it. Try again, and if it keeps happening show the technical detail below to your IT team.';
  }
  return `The engine could not do that (error ${status}). Please try again.`;
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(BASE + path, options);
  } catch (err) {
    throw new ApiError(UNREACHABLE, 0, null, path);
  }

  const text = await res.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }

  if (!res.ok) {
    const fromEngine = detailOf(payload);
    // A 5xx with no JSON body is almost always "nothing is listening" rather
    // than "the engine crashed mid-request" - say the useful one.
    const message =
      fromEngine || (res.status >= 500 && !payload ? UNREACHABLE : friendlyForStatus(res.status));
    throw new ApiError(message, res.status, payload, path);
  }
  return payload;
}

function json(path, method, body) {
  return request(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

// -- catalogue -------------------------------------------------------------
export const getHealth = () => request('/health');
export const getUseCases = () => request('/usecases');
export const getUseCase = (id) => request(`/usecases/${encodeURIComponent(id)}`);

// -- ceilings --------------------------------------------------------------
/**
 * The limits currently in force, so the drop zone can say
 * "Up to 2 GB" *before* someone wastes a 3 GB upload.
 */
export const getLimits = () => request('/limits');

// -- datasets --------------------------------------------------------------
/**
 * Send a file. The engine answers immediately (202) with
 * `{job_id, status}` and reads it in the background; watch it with
 * `openJobStream`. Two exceptions worth knowing:
 *
 * - a file over the size limit comes straight back as a *terminal* snapshot
 *   (`status: "rejected"`) with the plain-English reason already in `message`;
 * - `?sync=true` is the old blocking call, used by scripts and tests only.
 */
export function startUpload(file, { sync = false } = {}) {
  const form = new FormData();
  form.append('file', file);
  return request(`/datasets/upload${sync ? '?sync=true' : ''}`, {
    method: 'POST',
    body: form,
  });
}

/**
 * Paste a table. A small paste comes back as a DatasetProfile (it has
 * `dataset_id`); a large one comes back as `{job_id}` - the caller must
 * handle both shapes.
 */
export const pasteDataset = (name, rows) => json('/datasets/paste', 'POST', { name, rows });

export const loadSample = (name) => json(`/datasets/sample/${encodeURIComponent(name)}`, 'POST');
export const getSamples = () => request('/samples');
export const previewDataset = (id, n = 20) =>
  request(`/datasets/${encodeURIComponent(id)}/preview?n=${n}`);
// QA DF-12 / UX-15: let a dropdown offer the values actually in the user's file.
export const distinctValues = (id, column, limit = 50) =>
  request(`/datasets/${encodeURIComponent(id)}/distinct?column=${encodeURIComponent(column)}&limit=${limit}`);

/** One reading job's progress payload. */
export const getJob = (jobId) => request(`/datasets/jobs/${encodeURIComponent(jobId)}`);

/** Stop a reading job. The engine answers `{status: "cancelled"}`. */
export const cancelJob = (jobId) =>
  json(`/datasets/jobs/${encodeURIComponent(jobId)}/cancel`, 'POST');

export const jobStreamUrl = (jobId) =>
  `${BASE}/datasets/jobs/${encodeURIComponent(jobId)}/stream`;

//: The statuses that mean "nothing more will happen".
export const TERMINAL_STATUSES = ['ready', 'failed', 'rejected', 'cancelled'];

export const isTerminal = (status) => TERMINAL_STATUSES.includes(String(status || ''));

/**
 * Watch a reading job over server-sent events.
 *
 * `onEvent(name, data)` receives `progress` repeatedly, then `done` or `error`.
 * `onProblem(message)` fires only when the connection itself dies - the stream
 * is not retried for us, because the caller has a polling fallback.
 *
 * One subtlety: the engine sends a *named* event called `error`, and the
 * browser also fires its own `error` event for a broken connection. They land
 * on the same listener, so we tell them apart by whether `data` is present.
 */
export function openJobStream(jobId, onEvent, onProblem) {
  const src = new EventSource(jobStreamUrl(jobId));
  // QA NEW-03: the server closes the stream on purpose once it has sent a
  // terminal event, and the browser then fires its own `error`. Reporting
  // "we lost the connection" at that point blames the network for something
  // the engine already explained, and sends the user to retry a run that will
  // fail identically. Only a close we did not ask for is a problem.
  let settled = false;

  ['progress', 'done', 'error'].forEach((name) => {
    src.addEventListener(name, (ev) => {
      const raw = ev && typeof ev.data === 'string' ? ev.data : '';
      if (!raw) return; // a connection-level error, not an engine message
      let data = null;
      try {
        data = JSON.parse(raw);
      } catch {
        data = null;
      }
      if (name === 'done' || name === 'error') settled = true;
      onEvent(name, data || {});
    });
  });

  // Connection-level failures only: a retryable blip is not worth telling the
  // user about, but a dead stream is, because we then fall back to polling.
  src.addEventListener('error', (ev) => {
    const raw = ev && typeof ev.data === 'string' ? ev.data : '';
    if (raw || settled) return;
    if (src.readyState === EventSource.CLOSED && onProblem) {
      onProblem(
        'We lost the live connection to the file reader. We will keep checking on it in the background.',
      );
    }
  });

  return src;
}

// -- mapping ---------------------------------------------------------------
export const automap = (useCaseId, datasetId) =>
  json(`/usecases/${encodeURIComponent(useCaseId)}/automap`, 'POST', { dataset_id: datasetId });

// -- runs ------------------------------------------------------------------
export const createRun = (body) => json('/runs', 'POST', body);
export const getRunResult = (runId) => request(`/runs/${encodeURIComponent(runId)}/result`);
export const getRunLog = (runId) => request(`/runs/${encodeURIComponent(runId)}/log`);

export const artifactUrl = (runId, fmt) =>
  `${BASE}/runs/${encodeURIComponent(runId)}/artifact/${encodeURIComponent(fmt)}`;

/**
 * Open the live log stream for a run.
 *
 * Returns the EventSource so the caller can close it. `onEvent` receives
 * (name, data) for `log` | `status` | `result` | `error`.
 */
export function openRunStream(runId, onEvent, onProblem) {
  const src = new EventSource(`${BASE}/runs/${encodeURIComponent(runId)}/stream`);
  const names = ['log', 'status', 'result', 'error'];
  // QA NEW-03: a run that fails a model check ends with the server closing the
  // stream, which the browser reports as `error`. Saying "we lost the
  // connection to the run" there hides the real reason (shown one line above in
  // the live log) and invites a retry that will fail the same way.
  let settled = false;
  names.forEach((name) => {
    src.addEventListener(name, (ev) => {
      const raw = ev && typeof ev.data === 'string' ? ev.data : '';
      if (!raw) return;
      let data = null;
      try {
        data = JSON.parse(raw);
      } catch {
        data = null;
      }
      if (name === 'result' || name === 'error') settled = true;
      onEvent(name, data || {});
    });
  });
  src.onerror = () => {
    // EventSource retries on its own; only surface a problem if it is closing
    // and the run has not already told us how it ended.
    if (settled) return;
    if (src.readyState === EventSource.CLOSED && onProblem) {
      // QA SP-05: this used to say "we lost the connection to the run - please
      // try again", and the caller then marked the run as failed. But the
      // stream is only the running commentary: losing it says nothing about
      // the run, which is still going on the engine and is one request away
      // from its answer. Name what was actually lost, and say what happens
      // next - the caller keeps asking for the result.
      onProblem(
        'We lost the live log for this run. The engine is still working on it — we will keep asking, and the answer will appear here the moment it is ready.',
      );
    }
  };
  return src;
}
