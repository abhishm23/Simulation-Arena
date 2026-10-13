// Turning raw numbers into sentences a person can read.
//
// Every long job on this screen reports machine numbers (rows, bytes, rows a
// second, seconds left). QA's complaint was that none of it was shown at all —
// the second complaint would be showing it as `1.2e+06` or `Infinity`. These
// helpers are the only place those numbers become text, so every one of them
// has a defined answer for null, NaN, Infinity and negatives.

/** 1250000 -> "1,250,000". Never "NaN". */
export function humanCount(n) {
  const v = Number(n);
  if (!isFinite(v) || v < 0) return '0';
  return Math.round(v).toLocaleString('en-US');
}

/**
 * Rows a second, as the user should read it: "~2.2M rows/s", "~450 rows/s".
 * Returns null when there is nothing honest to say (0, null, NaN, Infinity) —
 * the caller hides the line rather than printing a fake number.
 */
export function humanRate(rate) {
  const v = Number(rate);
  if (!isFinite(v) || v <= 0) return null;
  if (v >= 1e6) return `~${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}M rows/s`;
  if (v >= 1e4) return `~${Math.round(v / 1e3)}k rows/s`;
  if (v >= 1e3) return `~${(v / 1e3).toFixed(1)}k rows/s`;
  return `~${Math.round(v).toLocaleString('en-US')} rows/s`;
}

/**
 * Seconds remaining in words. Anything that is not a finite, non-negative
 * number returns null — the user must never see "Infinity" or "-3 seconds".
 * `null` is *not* zero: it means "the engine does not know yet", which is a
 * different sentence from "almost done".
 */
function humanEta(seconds) {
  if (seconds === null || seconds === undefined || seconds === '') return null;
  const v = Number(seconds);
  if (!isFinite(v) || v < 0) return null;
  if (v < 1) return 'less than a second';
  if (v < 90) {
    const s = Math.round(v);
    return `${s} second${s === 1 ? '' : 's'}`;
  }
  if (v < 3600) {
    const m = Math.floor(v / 60);
    const s = Math.round(v - m * 60);
    if (s === 0 || s === 60) return `${m === 1 ? '1 minute' : `${m} minutes`}`;
    return `${m} minute${m === 1 ? '' : 's'} ${s} second${s === 1 ? '' : 's'}`;
  }
  const h = Math.floor(v / 3600);
  const m = Math.round((v - h * 3600) / 60);
  return m === 0 ? `${h} hour${h === 1 ? '' : 's'}` : `${h}h ${m}m`;
}

/**
 * "about 12 seconds left" — the full phrase, or null when the engine has not
 * told us yet. Null means the caller hides the line rather than printing
 * "almost done" over a job that has only just started.
 */
export function humanEtaLeft(seconds) {
  const word = humanEta(seconds);
  if (word === null) return null;
  if (word === 'less than a second') return 'almost done';
  return `about ${word} left`;
}

/** 2147483648 -> "2.0 GB". */
export function humanBytes(n) {
  const v = Number(n);
  if (!isFinite(v) || v <= 0) return '0 bytes';
  const units = [
    ['GB', 1024 ** 3],
    ['MB', 1024 ** 2],
    ['KB', 1024],
  ];
  for (const [unit, div] of units) {
    if (v >= div) return `${(v / div).toFixed(1)} ${unit}`;
  }
  return `${Math.round(v)} bytes`;
}

/** A percentage that is always 0–100, or null when the total is unknown. */
export function clampPct(pct) {
  if (pct === null || pct === undefined || pct === '') return null;
  const v = Number(pct);
  if (!isFinite(v)) return null;
  return Math.min(100, Math.max(0, v));
}

/** Elapsed seconds -> "3.4s" / "1m 12s". Used by the ticking readouts. */
export function humanElapsed(seconds) {
  const v = Number(seconds);
  if (!isFinite(v) || v < 0) return '0.0s';
  if (v < 60) return `${v.toFixed(1)}s`;
  const m = Math.floor(v / 60);
  const s = Math.round(v - m * 60);
  return `${m}m ${s}s`;
}
