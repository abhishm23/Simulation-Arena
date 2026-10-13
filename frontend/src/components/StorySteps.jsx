import React from 'react';

export default function StorySteps({ steps }) {
  if (!steps || !steps.length) return null;
  return (
    <div>
      {steps.map((s, i) => (
        <div className="story-card" key={i}>
          <span className="story-num">{i + 1}</span>
          <div className="story-body">
            <div className="story-title">{s.title}</div>
            <div className="story-detail">{s.detail}</div>
            {s.evidence && <span className="story-evidence">{s.evidence}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}
