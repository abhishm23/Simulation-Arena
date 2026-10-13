import React from 'react';

export default function KpiCard({ kpi }) {
  const tone = (kpi.tone || 'neutral').toLowerCase();
  return (
    <div className={`kpi tone-${tone}`}>
      <div className="kpi-label">{kpi.label}</div>
      <div className="kpi-value">
        {kpi.value}
        {kpi.unit && kpi.unit !== '%' && (
          <span style={{ fontSize: 15, color: 'var(--text-faint)', marginLeft: 4 }}>
            {kpi.unit}
          </span>
        )}
      </div>
      {kpi.plain && <div className="kpi-plain">{kpi.plain}</div>}
      {kpi.comparison && <div className="kpi-comp">{kpi.comparison}</div>}
    </div>
  );
}
