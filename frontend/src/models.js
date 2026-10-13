// Translating the engine's vocabulary into the user's.
//
// Model names are allowed on screen as badges — QA said so — but a badge is
// not an explanation. The same goes for the engine's `data_granularity`, which
// arrives as `line_item` or `transition`: that is model vocabulary, not
// English. This is the single place that turns either into words a sales
// director or a CFO would use.

const MODEL_PLAIN = {
  'monte carlo': 'many simulated futures',
  'latin hypercube': 'smart sampling of many futures at once',
  bootstrap: 're-sampling your own history',
  markov: 'a stage-by-stage movement model',
  'agent-based': 'a virtual market of buyers and rivals',
  'discrete event': 'a queueing model of your approval line',
  'empirical joint': 'the real-world pattern of which products sell together',
  'discrete-event': 'a queueing model of your approval line',
};

/** "Markov chain" -> "a stage-by-stage movement model". Never returns jargon. */
export function modelPlain(name) {
  const key = String(name || '').toLowerCase();
  for (const [needle, plain] of Object.entries(MODEL_PLAIN)) {
    if (key.includes(needle)) return plain;
  }
  return key || 'a simulation';
}

/** A whole list, as one readable phrase: "A, B and C". */
export function modelListPlain(names) {
  // The engine sometimes lists two techniques that read the same in plain
  // English ("bootstrap" and "empirical joint"). Saying the same thing twice
  // looks like a bug to the user, so collapse repeats.
  const words = [...new Set((names || []).map(modelPlain).filter(Boolean))];
  if (!words.length) return 'a simulation';
  if (words.length === 1) return words[0];
  return `${words.slice(0, -1).join(', ')} and ${words[words.length - 1]}`;
}

/**
 * The one-off explanation. Shown wherever a model badge appears so the badge
 * is never the only thing the user is given.
 */
export const MODEL_TIP =
  'We build thousands of made-up but realistic versions of your business and count how often each outcome happens. The names in the blue badges are the techniques we used — you never need to know them to read the answer.';

/**
 * The engine's `data_granularity` says what one row of the user's file is.
 * It arrives as identifiers like `line_item` and `transition`, which are model
 * vocabulary — "one row per transition" means nothing to a sales director.
 * This returns the whole clause, ready to drop into a sentence:
 * "deal-level data — one row per deal".
 */
const GRANULARITY_PLAIN = {
  deal: 'deal-level data — one row per deal',
  opportunity: 'deal-level data — one row per deal',
  line_item: 'order-line-level data — one row per line on the order',
  'line-item': 'order-line-level data — one row per line on the order',
  lineitem: 'order-line-level data — one row per line on the order',
  basket: 'basket-level data — one row per basket',
  event: 'event-level data — one row per event',
  portfolio: 'account-level data — one row per account',
  account: 'account-level data — one row per account',
  cost: 'cost-line-level data — one row per cost line',
  transition: 'stage-history data — one row per move a deal makes between stages',
  stage: 'stage-level data — one row per stage',
  customer: 'customer-level data — one row per customer',
  order: 'order-level data — one row per order',
};

export function granularityPlain(granularity) {
  const key = String(granularity || '').toLowerCase().trim();
  if (!key) return 'your data — one row per record';
  if (GRANULARITY_PLAIN[key]) return GRANULARITY_PLAIN[key];
  // An unknown identifier still has to read like English: `region_month` is
  // shown as "region month", never with the underscore the engine used.
  const words = key.replace(/[_-]+/g, ' ');
  return `${words}-level data — one row per ${words}`;
}
