import React, { useEffect, useId, useRef, useState } from 'react';

/**
 * A "?" that explains a setting in more detail than the line under it.
 *
 * Opens on hover and on click, and closes on Escape, on blur, or on any click
 * outside. `placement="below"` flips the bubble for tips that sit near the top
 * of the viewport.
 *
 * QA UX-06: three things were wrong here.
 *   * `aria-controls` pointed at an id that only existed while the bubble was
 *     open, so at rest it referred to nothing. The bubble is now always in the
 *     document and simply hidden when closed, so the reference always resolves.
 *   * Hover and click fought each other: hovering set the bubble open, so the
 *     click that followed toggled it straight back shut and the control looked
 *     broken to anyone who clicked it. Hover and "pinned" are now separate, and
 *     a click pins the bubble open instead of closing it.
 *   * The bubble repeated the sentence already printed under the label. It now
 *     carries the acceptance detail (range, step, default, choices) instead -
 *     see `extraTip` in ParamForm, which only renders this control when there
 *     is something extra to say.
 */
export default function HelpTip({ text, placement = 'above', label }) {
  const [hovered, setHovered] = useState(false);
  const [pinned, setPinned] = useState(false);
  const open = hovered || pinned;
  const wrapRef = useRef(null);
  const btnRef = useRef(null);
  const tipId = useId();

  useEffect(() => {
    if (!pinned) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') {
        setPinned(false);
        if (btnRef.current) btnRef.current.focus();
      }
    };
    const onClick = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setPinned(false);
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('mousedown', onClick);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('mousedown', onClick);
    };
  }, [pinned]);

  if (!text) return null;

  return (
    <span
      ref={wrapRef}
      className={`tip-wrap${open ? ' is-open' : ''}`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        ref={btnRef}
        type="button"
        className="tip-btn"
        aria-label={label ? `More about ${label}` : 'More about this setting'}
        aria-expanded={open}
        aria-controls={tipId}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          // a click pins it open; the hover state alone never closes it again
          setPinned((v) => !v);
        }}
        onBlur={() => setPinned(false)}
      >
        ?
      </button>
      <span
        id={tipId}
        role="tooltip"
        hidden={!open}
        className={`tip-pop${placement === 'below' ? ' is-below' : ''}`}
      >
        {text}
      </span>
    </span>
  );
}
