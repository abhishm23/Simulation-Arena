import React, { useState } from 'react';
import { artifactUrl } from '../api.js';
import HelpTip from './HelpTip.jsx';

const FORMATS = [
  { key: 'html', label: 'Web page', icon: '🌐', hint: 'One file you can open in any browser and send on.' },
  { key: 'xlsx', label: 'Excel', icon: '📊', hint: 'A workbook with a tab per table, ready to slice yourself.' },
  { key: 'csv', label: 'Data (CSV)', icon: '🗂️', hint: 'A zip of plain CSV files, one per table.' },
  { key: 'pdf', label: 'PDF', icon: '📕', hint: 'A print-ready document for a pack or a meeting.' },
];

export default function DownloadBar({ runId, seed, onSeedChange, onRerun }) {
  const [busy, setBusy] = useState(null);
  const [seedValue, setSeedValue] = useState(String(seed ?? 42));

  function download(fmt) {
    setBusy(fmt);
    // The browser follows the server's Content-Disposition, so a plain link is
    // enough - but we check it really exists first so we can say so politely.
    fetch(artifactUrl(runId, fmt), { method: 'HEAD' })
      .then((res) => {
        if (!res.ok) throw new Error('missing');
        const a = document.createElement('a');
        a.href = artifactUrl(runId, fmt);
        a.download = '';
        document.body.appendChild(a);
        a.click();
        a.remove();
      })
      .catch(() => {
        window.open(artifactUrl(runId, fmt), '_blank', 'noopener');
      })
      .finally(() => setBusy(null));
  }

  return (
    <div className="download-bar">
      {FORMATS.map((f) => (
        <button
          type="button"
          key={f.key}
          className="btn btn-sm"
          disabled={busy === f.key}
          onClick={() => download(f.key)}
          title={f.hint}
        >
          <span aria-hidden="true">{f.icon}</span>
          {busy === f.key ? 'Preparing…' : f.label}
        </button>
      ))}

      <div className="seed-row">
        <span className="muted">
          Seed{' '}
          <HelpTip
            label="seed"
            text="The seed is the starting point for the random testing. Same seed = same numbers, so anyone can reproduce what you see here. Change it to see how much the answer depends on luck."
          />
        </span>
        <span className="stepper seed-input">
          <button
            type="button"
            aria-label="Previous seed"
            onClick={() => setSeedValue(String(Math.max(1, Number(seedValue || 42) - 1)))}
          >
            −
          </button>
          <input
            type="number"
            min={1}
            value={seedValue}
            onChange={(e) => setSeedValue(e.target.value)}
          />
          <button
            type="button"
            aria-label="Next seed"
            onClick={() => setSeedValue(String(Number(seedValue || 42) + 1))}
          >
            +
          </button>
        </span>
        <button
          type="button"
          className="btn btn-sm btn-primary"
          onClick={() => {
            const next = Number(seedValue);
            if (onSeedChange) onSeedChange(isFinite(next) && next > 0 ? next : 42);
            if (onRerun) onRerun();
          }}
        >
          Run again with this seed
        </button>
      </div>
    </div>
  );
}
