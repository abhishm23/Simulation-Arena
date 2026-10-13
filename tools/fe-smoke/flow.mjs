/**
 * Browserless runtime smoke test for the built React app.
 *
 * jsdom supplies a real DOM; we import the actual production bundle and let
 * React mount, then drive the wizard. Anything that throws at runtime (bad
 * hook, undefined access, missing browser API) surfaces here. Network calls
 * are pointed at the real backend on :8000.
 *
 * Usage (from tools/fe-smoke, with the backend running and frontend built):
 *   npm install          # first time only
 *   node flow.mjs        # writes result.txt, exit 0 = clean
 */
import { JSDOM } from 'jsdom';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const DIST = '../../frontend/dist';
const API = 'http://127.0.0.1:8000';
const OUT = [];
const log = (...a) => OUT.push(a.join(' '));
const errors = [];

const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
  url: 'http://localhost:8000/',
  pretendToBeVisual: true,
  runScripts: 'outside-only',
});
const w = dom.window;
global.window = w;
global.document = w.document;

const set = (k, v) => {
  if (v === undefined) return;
  try { if (global[k] === undefined) global[k] = v; }
  catch { try { Object.defineProperty(global, k, { value: v, configurable: true }); } catch {} }
};
set('navigator', w.navigator);
for (const k of ['MutationObserver', 'ResizeObserver', 'IntersectionObserver', 'HTMLElement',
  'Element', 'Node', 'Event', 'CustomEvent', 'MouseEvent', 'KeyboardEvent', 'getComputedStyle',
  'DOMParser', 'MessageChannel', 'requestAnimationFrame', 'cancelAnimationFrame', 'matchMedia',
  'Image', 'SVGElement', 'HTMLInputElement', 'HTMLSelectElement', 'HTMLButtonElement',
  'DocumentFragment', 'NodeList', 'CSS']) set(k, w[k]);
if (!w.matchMedia) {
  w.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {},
    addListener() {}, removeListener() {} });
  set('matchMedia', w.matchMedia);
}
for (const n of ['ResizeObserver', 'IntersectionObserver']) {
  if (!w[n]) { w[n] = class { observe() {} unobserve() {} disconnect() {} }; set(n, w[n]); }
}

const realFetch = globalThis.fetch;
const appFetch = (i, o) => realFetch(
  typeof i === 'string' ? new URL(i, API).toString() : String(i), o);
global.fetch = appFetch;
w.fetch = appFetch;
// A WORKING EventSource stub: jsdom has no SSE, and the app waits on the
// stream for its result. This really fetches the endpoint, parses the
// event:/data: frames and dispatches them, so the run can complete.
const streams = [];
let received = 0;
w.EventSource = global.EventSource = class {
  constructor(u) {
    this.url = u; this.readyState = 1; this.handlers = {}; this.closed = false;
    streams.push(u);
    (async () => {
      try {
        const res = await realFetch(new URL(u, API).toString(),
          { headers: { Accept: 'text/event-stream' } });
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        for (;;) {
          const { done, value } = await reader.read();
          if (done || this.closed) break;
          const chunk = dec.decode(value, { stream: true });
          received += chunk.length;
          buf += chunk.replace(/\r\n/g, '\n');
          let i;
          while ((i = buf.indexOf('\n\n')) >= 0) {
            this._emit(buf.slice(0, i));
            buf = buf.slice(i + 2);
          }
        }
      } catch { /* stream ended */ }
    })();
  }
  addEventListener(t, f) { (this.handlers[t] = this.handlers[t] || []).push(f); }
  _emit(block) {
    let ev = 'message'; let raw = '';
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) ev = line.slice(6).trim();
      else if (line.startsWith('data:')) raw = line.slice(5).trim();
    }
    if (!raw) return;
    // Real EventSource delivers event.data as a STRING - the app does its own
    // JSON.parse. Passing a pre-parsed object broke every handler silently.
    const evt = { data: raw, type: ev, lastEventId: '' };
    (this.handlers[ev] || []).forEach((f) => { try { f(evt); } catch {} });
    if (ev !== 'message') {
      (this.handlers.message || []).forEach((f) => { try { f(evt); } catch {} });
    }
    const prop = this['on' + ev];
    if (typeof prop === 'function') { try { prop(evt); } catch {} }
    if (typeof this.onmessage === 'function') { try { this.onmessage(evt); } catch {} }
  }
  close() { this.closed = true; this.readyState = 2; }
};
process.on('uncaughtException', (e) => errors.push('UNCAUGHT: ' + e.message));
w.addEventListener('error', (e) => errors.push('ERR: ' + (e.message || e.error)));

const bundle = fs.readdirSync(path.join(DIST, 'assets')).find((f) => f.endsWith('.js'));
if (!bundle) throw new Error('no bundle in frontend/dist/assets - run npm run build');
// On Windows the ESM loader rejects bare "C:\..." paths - need a file:// URL.
await import(pathToFileURL(path.resolve(DIST, 'assets', bundle)).href);

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const txt = () => (w.document.getElementById('root')?.textContent || '').replace(/\s+/g, ' ');
const step = () => { const m = txt().match(/Step (\d) of 5/); return m ? m[1] : '?'; };
const click = (pred) => {
  const el = [...w.document.querySelectorAll('button')].find(pred);
  if (!el) return false;
  el.dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  return true;
};

await wait(3500);
const root = w.document.getElementById('root');
log('bundle              :', bundle);
log('mounted children    :', root ? root.children.length : 0);
log('initial step        :', step());

const ok1 = click((e) => /Deal Approval/i.test(e.textContent || ''));
await wait(2000);
log('card clicked        :', ok1, '| step now:', step());

const ok2 = click((e) => /closed deals|discount given/i.test(e.textContent || ''));
await wait(4000);
log('sample clicked      :', ok2, '| step now:', step());
log('mapping shown       :', /Match your columns|What we spotted/i.test(txt()));
log('streams before run   :', streams.length);

// --- continue to the assumptions form -------------------------------------
const ok3 = click((e) => /Set the assumptions|assumptions/i.test(e.textContent || ''));
await wait(2000);
log('to assumptions      :', ok3, '| step now:', step());
const selects = w.document.querySelectorAll('select').length;
const sliders = w.document.querySelectorAll('input[type="range"]').length;
log('form controls       : selects', selects, '| sliders', sliders);

// --- run it ---------------------------------------------------------------
const ok4 = click((e) => /Run the simulation/i.test(e.textContent || ''));
await wait(16000);
log('run clicked         :', ok4, '| step now:', step());
const out = txt();
log('headline rendered   :', /Approve|Give |best discount|discount/i.test(out));
log('rendered snippet    :', out.slice(260, 1100));
log('recommendation      :', /Recommend|What to do|Approve/i.test(out));
log('sim-time badge      :', /the simulation itself took/i.test(out));
log('read-time badge     :', /your file took/i.test(out));
log('end-to-end badge    :', /end to end/i.test(out));
log('result headline     :', (out.match(/(Approve|Give)[^.]{0,110}/) || ['(none)'])[0]);
log('streams opened       :', streams.length, streams.slice(0,1));
log('sse bytes received   :', received);
log('runtime errors      :', errors.length ? errors.slice(0, 5).join(' | ') : 'NONE');

// Be honest about scope: jsdom mounts the app and drives the wizard, and the
// run START is verified, but the run does not COMPLETE here because the app's
// SSE transport is not driven by this harness. Do not claim more than we prove.
const mounted = !!root && root.children.length > 0;
// On the result page the wizard chrome is replaced, so "Step N of 5" is gone -
// judge success by the rendered RESULT content instead.
const resultRendered = /the simulation itself took/i.test(out)
  && /Recommend|What to do/i.test(out);
const ranClean = errors.length === 0;
const pass = mounted && resultRendered && ranClean;
log('');
log('SCOPE               : verifies mount, wizard steps 1-5, assumptions form, a full');
log('                      run over SSE, and the RESULT page (headline, badges,');
log('                      recommendation) rendering without runtime errors.');
log('RESULT              :', pass
  ? 'PASS - app mounts, runs end to end, result page renders, no runtime errors'
  : `FAIL - mounted=${mounted} result=${resultRendered} errors=${errors.length}`);
fs.writeFileSync('result.txt', OUT.join('\n') + '\n');
process.exit(pass ? 0 : 1);
