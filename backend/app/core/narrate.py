"""The narration bus.

The whole point of this product: the user is never staring at a black box.
Every meaningful step emits a plain-language event. Events go to:
  * the live log stream (SSE -> front-end),
  * the final RunResult.log,
  * the exported report's "how this was computed" appendix.

Writers should use the verbs a business user would use ("Testing 10,000
price outcomes") and never the internal names ("drawing from beta(2,5)").
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

from .contracts import LogEvent

PHASES = ("read", "validate", "map", "prepare", "simulate", "summarize", "report")


class Narrator:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._events: list[LogEvent] = []
        self._listeners: list[queue.Queue] = []
        self._lock = threading.Lock()

    # -- subscription -------------------------------------------------------
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._listeners.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._listeners:
                self._listeners.remove(q)

    # -- emission -----------------------------------------------------------
    def say(
        self,
        message: str,
        phase: str = "run",
        level: str = "info",
        detail: Optional[str] = None,
        progress: Optional[float] = None,
    ) -> None:
        ev = LogEvent(
            t=round(time.perf_counter() - self._t0, 3),
            phase=phase,
            level=level,
            message=message,
            detail=detail,
            progress=None if progress is None else round(float(progress), 3),
        )
        with self._lock:
            self._events.append(ev)
            listeners = list(self._listeners)
        for q in listeners:
            try:
                q.put_nowait(ev)
            except Exception:
                pass

    # convenience -----------------------------------------------------------
    def step(self, message: str, detail: Optional[str] = None, progress: Optional[float] = None):
        self.say(message, "simulate", "info", detail, progress)

    def warn(self, message: str, detail: Optional[str] = None):
        self.say(message, "validate", "warn", detail)

    def done(self, message: str):
        self.say(message, "summarize", "success", None, 1.0)

    def fail(self, message: str, detail: Optional[str] = None):
        self.say(message, "summarize", "error", detail)

    # -- progress helper ----------------------------------------------------
    def tracker(self, total: int, template: str, phase: str = "simulate") -> Callable[[int], None]:
        """Emits a message every ~10% of `total` iterations."""
        next_mark = [0.1]

        def tick(i: int) -> None:
            frac = (i + 1) / max(total, 1)
            if frac >= next_mark[0] or i + 1 == total:
                next_mark[0] = min(1.0, frac + 0.1)
                self.say(template.format(pct=int(frac * 100), i=i + 1, n=total),
                         phase=phase, progress=frac)
        return tick

    # -- retrieval ----------------------------------------------------------
    def events(self) -> list[LogEvent]:
        with self._lock:
            return list(self._events)
