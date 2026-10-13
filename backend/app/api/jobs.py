"""Reading a file in the background, with progress the user can actually watch.

This module is the answer to the two worst QA findings:

* **C2 - one big upload took the whole product down.** Nothing here ever runs on
  the event loop. Every read happens on a worker thread (polars releases the GIL,
  so the loop stays genuinely free), large files are admitted one at a time
  behind a semaphore, and a file is measured with `ingest.probe()` and refused
  *before* a single byte is parsed. A watchdog enforces a parse deadline and a
  process-memory ceiling keeps the machine from swapping.
* **H2 - no progress, no row counter, no ETA.** Every job keeps a live progress
  record (rows read, bytes read, percentage, rows a second, seconds left) and a
  plain-English sentence a layman understands, e.g. "Reading your file - 1,250,000
  rows so far, about 12 seconds left".

The shapes are fixed by `QA_FIX_BRIEF.md`:

    POST /api/datasets/upload                 -> 202 {"job_id", "status"}
    GET  /api/datasets/jobs/{job_id}          -> the progress payload
    GET  /api/datasets/jobs/{job_id}/stream   -> SSE `progress` ... then
                                                 `done` or `error`
    POST /api/datasets/jobs/{job_id}/cancel   -> {"status": "cancelled"}
    GET  /api/limits                          -> the ceilings in force

`status` is one of queued | reading | profiling | sampling | ready | failed |
rejected | cancelled. Small files (under `LARGE_FILE_BYTES`) bypass the queue
completely so they stay instant; only large reads wait in line.

Terminal events carry the outcome
---------------------------------
Both closing SSE events name the state they ended in, so a client does not have
to ask again to tell a cancellation from a refusal from a crash:

    done   {"dataset_id", "profile", "status": "ready"}
    error  {"message", "trace", "status": "failed" | "rejected" | "cancelled"}

Progress invariants (enforced, not merely intended)
---------------------------------------------------
* `status` is always one of the eight documented words - anything else is a bug
  and `Job.update` raises rather than emitting it.
* `bytes_read` and `rows_read` never decrease for a job.
* `eta_s` is `null` or a positive number while work is going on; `0.0` is only
  ever published for a job that has finished.
* `pct` is `null` (unknown) or a non-decreasing 0-100. During the profiling
  phase there is no honest percentage, so `pct` is `null` and the UI shows an
  indeterminate bar instead of sitting at "98% - almost done" for eight seconds.
"""
from __future__ import annotations

import ctypes
import math
import multiprocessing
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from ..io import ingest
from . import store

# ---------------------------------------------------------------------------
# Statuses (the exact vocabulary in the contract)
# ---------------------------------------------------------------------------
QUEUED = "queued"
READING = "reading"
PROFILING = "profiling"
SAMPLING = "sampling"
READY = "ready"
FAILED = "failed"
REJECTED = "rejected"
CANCELLED = "cancelled"

TERMINAL = frozenset({READY, FAILED, REJECTED, CANCELLED})

#: every status a job is allowed to be in - the exact contract vocabulary
VALID_STATUSES = frozenset({QUEUED, READING, PROFILING, SAMPLING,
                            READY, FAILED, REJECTED, CANCELLED})

#: what the user is told while the columns are being worked out. There is no
#: honest percentage for this step, so the UI goes indeterminate rather than
#: pretending to be nearly finished.
PROFILING_PHASE_TEXT = "Working out what each column holds"
PROFILING_MESSAGE = ("We have your rows - now working out what each column "
                     "holds. This last step has no honest percentage, so the "
                     "bar pauses here.")

# ---------------------------------------------------------------------------
# Limits - module level on purpose, so tests and /api/limits read them live
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name)
        return int(raw) if raw not in (None, "") else int(default)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        return float(raw) if raw not in (None, "") else float(default)
    except Exception:
        return float(default)


#: how long an SSE listener may sit on one job before we hang up. The env
#: override exists so the timeout-then-retry journey (QA SP-05) can be driven
#: end to end in seconds instead of fifteen minutes; production never sets it.
MAX_STREAM_SECONDS = _env_float("SIMARENA_MAX_STREAM_SECONDS", 900.0)


class ProgressRegression(ValueError):
    """A progress update tried to move a counter backwards, or used a status
    outside the contract. It is always a bug in this module, never bad data, so
    it is raised loudly instead of being written to the job record."""


#: refuse anything bigger than this *before* reading it.
#: QA NEW-14: the product requirement is gigabyte-scale data - a 4-5 GB export
#: with tens of millions of rows. The old 2 GiB ceiling refused exactly the
#: files the tool exists for. The reader streams, so the file's *size* is not
#: what costs memory - the working set below is - and a 5 GB CSV reads in well
#: under a minute with the child reader. 6 GiB leaves headroom over "4-5 GB".
MAX_FILE_BYTES = _env_int("SIMARENA_MAX_FILE_BYTES", 6 * 1024 ** 3)
#: The working set. Above this many rows we keep an evenly spread sample and say
#: so, in the run log and in the result. This is the *only* row cap in the
#: product - what is read is what the run uses - so there is exactly one
#: reduction to disclose (QA NEW-04 / NEW-05). A use case works on `list[dict]`
#: and one row of ~17 columns costs about half a kilobyte of Python objects, so
#: 250k rows is roughly 120 MB for the rows a run holds, plus the mapped copy.
#: At 2,000,000 the run path alone held ~1 GB of dicts inside the server process
#: on top of the dataset cache and the export buffers, which is how one upload
#: reached 7.5 GB of RSS and took the whole application down with it.
#: QA NEW-10 / TC-02: the sample now moves in WHOLE business records (see
#: `ingest._Sampler`), so raising the file ceiling does not cost the deal-level
#: dependency the basket use case exists to measure.
MAX_WORKING_ROWS = _env_int("SIMARENA_MAX_WORKING_ROWS", 250_000)
#: how many *large* reads may run at once; small files never queue
MAX_CONCURRENT_UPLOADS = max(1, _env_int("SIMARENA_MAX_CONCURRENT_UPLOADS", 1))
#: give up on a read that runs longer than this
PARSE_TIMEOUT_S = _env_float("SIMARENA_PARSE_TIMEOUT_S", 900.0)
#: never let the process working set go past this
MEMORY_CEILING_BYTES = _env_int("SIMARENA_MEMORY_CEILING_BYTES",
                                int(3.5 * 1024 ** 3))
#: a file at least this big takes its turn in the queue
LARGE_FILE_BYTES = _env_int("SIMARENA_LARGE_FILE_BYTES", 200 * 1024 ** 2)
#: how many finished jobs we keep so a client can still ask what happened
MAX_RETAINED_JOBS = max(8, _env_int("SIMARENA_MAX_RETAINED_JOBS", 64))
#: a paste bigger than this becomes a background job too
PASTE_ASYNC_ROWS = max(1, _env_int("SIMARENA_PASTE_ASYNC_ROWS", 20_000))
#: a file at least this big is read AND profiled in its own process. Polars
#: releases the GIL for its own work, but turning a million Python dicts into a
#: frame is one long call that holds it - long enough to stop the server
#: answering. A separate process cannot freeze the event loop at all, and it can
#: be killed outright when the user cancels.
CHILD_INGEST_BYTES = _env_int("SIMARENA_CHILD_INGEST_BYTES", 64 * 1024 ** 2)


def ingest_limits() -> ingest.IngestLimits:
    """The ceilings, read live so a test can change them."""
    return ingest.IngestLimits(
        max_file_bytes=int(MAX_FILE_BYTES),
        max_working_rows=int(MAX_WORKING_ROWS),
        parse_timeout_s=float(PARSE_TIMEOUT_S),
    )


def limits_payload() -> dict[str, Any]:
    """Exactly the body `GET /api/limits` promises."""
    return {
        "max_file_bytes": int(MAX_FILE_BYTES),
        "max_working_rows": int(MAX_WORKING_ROWS),
        "max_concurrent_uploads": int(MAX_CONCURRENT_UPLOADS),
        "parse_timeout_s": float(PARSE_TIMEOUT_S),
        "max_file_human": _human_bytes(MAX_FILE_BYTES),
        "max_working_rows_human": f"{int(MAX_WORKING_ROWS):,} rows",
        "large_file_threshold_bytes": int(LARGE_FILE_BYTES),
        "memory_ceiling_bytes": int(MEMORY_CEILING_BYTES),
    }


# ---------------------------------------------------------------------------
# Plain-English formatting (the user never sees a raw number)
# ---------------------------------------------------------------------------
def _human_bytes(n: Any) -> str:
    try:
        value = float(n or 0)
    except Exception:
        return "0 bytes"
    for unit, div in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if value >= div:
            return f"{value / div:.1f} {unit}"
    return f"{int(value)} bytes"


def _human_count(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return "0"


def _human_duration(seconds: Any) -> str:
    """'12 seconds' - never a bare number, never 'inf'."""
    try:
        s = float(seconds)
    except Exception:
        return "a moment"
    if not math.isfinite(s) or s < 0:
        return "a moment"
    if s < 1.0:
        return "less than a second"
    if s < 60.0:
        secs = int(round(s))
        return f"{secs} second{'s' if secs != 1 else ''}"
    minutes = int(s // 60)
    secs = int(round(s - minutes * 60))
    if minutes < 60:
        if secs == 0:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        return f"{minutes} minute{'s' if minutes != 1 else ''} " \
               f"{secs} second{'s' if secs != 1 else ''}"
    hours = minutes // 60
    minutes = minutes % 60
    if minutes == 0:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{hours} hour{'s' if hours != 1 else ''} " \
           f"{minutes} minute{'s' if minutes != 1 else ''}"


def _ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th… - the message is read by people."""
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def too_big_message(size: int, limit: int) -> str:
    """The refusal a person can act on, naming the limit and the way out."""
    return (f"That file is {_human_bytes(size)}, and the limit is "
            f"{_human_bytes(limit)}. Save it as Parquet or gzip the CSV and it "
            f"will be far smaller — then try again.")


def timeout_message(name: str, seconds: float) -> str:
    return (f"We stopped reading '{name}' after {_human_duration(seconds)} because "
            f"it was taking too long. Try saving it as Parquet, or splitting it "
            f"into smaller files, then upload again.")


CANCELLED_MESSAGE = ("You cancelled this upload. Nothing was saved — you can "
                     "start it again whenever you like.")
MEMORY_MESSAGE = ("This computer is using a lot of memory right now, so your file "
                  "is waiting for the one being read to finish. Nothing is lost — "
                  "we will start yours the moment there is room.")
EMPTY_MESSAGE = ("There are no rows in that file, so there is nothing to read. "
                 "Check the export and try again.")
NO_COLUMNS_MESSAGE = ("We could not find any columns in that file. It may be empty, "
                      "or it may not be a table we understand.")


# ---------------------------------------------------------------------------
# Process memory (working set) - used to keep the machine out of swap
# ---------------------------------------------------------------------------
class _MemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_WINDOWS_MEMORY_API = None


def _windows_memory_api():
    """(kernel32, psapi) with correct prototypes - cached after the first call."""
    global _WINDOWS_MEMORY_API
    if _WINDOWS_MEMORY_API is None:
        try:
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            _WINDOWS_MEMORY_API = (kernel32, psapi)
        except Exception:
            _WINDOWS_MEMORY_API = (None, None)
    return _WINDOWS_MEMORY_API


def process_rss_bytes() -> Optional[int]:
    """How much memory this process is actually holding, or None if unknown."""
    try:
        if os.name == "nt":
            kernel32, psapi = _windows_memory_api()
            if psapi is None:
                return None
            counters = _MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters),
                                            counters.cb)
            return int(counters.WorkingSetSize) if ok else None
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except Exception:
        return None


# ---------------------------------------------------------------------------
# A tiny fan-out: buffered events for late listeners, live queues for the rest
# ---------------------------------------------------------------------------
class _Feed:
    def __init__(self, cap: int = 4000) -> None:
        self._lock = threading.Lock()
        self._buffer: list[dict] = []
        self._listeners: list[queue.Queue] = []
        self._cap = max(1, int(cap))

    def publish(self, event: dict) -> None:
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) > self._cap:
                del self._buffer[: len(self._buffer) - self._cap]
            listeners = list(self._listeners)
        for q in listeners:
            try:
                q.put_nowait(event)
            except Exception:
                pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            backlog = list(self._buffer)
            self._listeners.append(q)
        for ev in backlog:
            try:
                q.put_nowait(ev)
            except Exception:
                break
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._listeners:
                self._listeners.remove(q)

    def clear(self) -> None:
        with self._lock:
            self._buffer = []


# ---------------------------------------------------------------------------
# One job
# ---------------------------------------------------------------------------
class Job:
    """One file being read (or one big paste being profiled), and how it is going."""

    def __init__(self, kind: str, name: str, bytes_total: int = 0) -> None:
        self.id = ("up_" if kind == "upload" else "pst_") + uuid.uuid4().hex[:6]
        self.kind = kind
        self.name = str(name or "your file")
        self.bytes_total = int(bytes_total or 0)
        self.temp_path: Optional[Path] = None
        self.rows_in: Optional[list[dict]] = None      # paste payload, before parsing

        self.feed = _Feed()
        self.cancel_event = threading.Event()
        self.finished_event = threading.Event()
        self.timed_out = False
        self.future: Optional[Future] = None
        self.profile = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self._lock = threading.RLock()
        self._terminal_sent = False

        self.status = QUEUED
        self.phase = QUEUED
        self.rows_read = 0
        self.rows_total: Optional[int] = None
        self.bytes_read = 0
        self.pct: Optional[float] = None
        self.rate_rows_s = 0.0
        self.eta_s: Optional[float] = None
        self.message = f"Waiting to start on '{self.name}'…"
        self.dataset_id: Optional[str] = None
        self.error: Optional[str] = None
        self.warnings: list[str] = []
        self.queue_position: Optional[int] = None

    # -- state --------------------------------------------------------------
    def update(self, **kw: Any) -> None:
        """Set fields on the job, refusing anything the contract forbids.

        Two guards live here on purpose. A `status` outside the documented set,
        or a `rows_read`/`bytes_read` that moves backwards, is a bug in this
        module - it has to fail loudly in a test rather than reach the user as a
        progress bar that jumps back to zero.
        """
        with self._lock:
            status = kw.get("status")
            if status is not None and status not in VALID_STATUSES:
                raise ProgressRegression(
                    f"unknown job status {status!r}; the contract allows "
                    f"{sorted(VALID_STATUSES)}")
            for key in ("rows_read", "bytes_read"):
                if key not in kw or kw[key] is None:
                    continue
                try:
                    new_value = int(kw[key])
                    old_value = int(getattr(self, key) or 0)
                except (TypeError, ValueError):
                    raise ProgressRegression(
                        f"{key} must be a whole number, got {kw[key]!r}")
                if new_value < old_value:
                    raise ProgressRegression(
                        f"{key} went backwards for job {self.id}: "
                        f"{old_value} -> {new_value}")
            for key, value in kw.items():
                setattr(self, key, value)

    def snapshot(self) -> dict[str, Any]:
        """Exactly the payload `GET /api/datasets/jobs/{job_id}` promises."""
        with self._lock:
            return {
                "job_id": self.id,
                "status": self.status,
                "phase": self.phase,
                "rows_read": int(self.rows_read or 0),
                "rows_total": None if self.rows_total is None else int(self.rows_total),
                "bytes_read": int(self.bytes_read or 0),
                "bytes_total": int(self.bytes_total or 0) or None,
                "pct": self.pct,
                "rate_rows_s": self.rate_rows_s,
                "eta_s": self.eta_s,
                "message": self.message,
                "dataset_id": self.dataset_id,
                "error": self.error,
                "warnings": list(self.warnings),
                "queue_position": self.queue_position,
            }

    def is_terminal(self) -> bool:
        with self._lock:
            return self.status in TERMINAL

    # -- events -------------------------------------------------------------
    def progress_event(self) -> dict[str, Any]:
        return {"event": "progress", "data": self.snapshot()}

    def publish_progress(self) -> None:
        self.feed.publish(self.progress_event())

    def publish_terminal(self, name: str, data: dict) -> bool:
        """Publish the one closing event. Returns False if one already went out."""
        with self._lock:
            if self._terminal_sent:
                return False
            self._terminal_sent = True
        self.feed.publish({"event": name, "data": data})
        return True


# ---------------------------------------------------------------------------
# The manager: admission control, workers, watchdog, book-keeping
# ---------------------------------------------------------------------------
class _Manager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._waiting: list[str] = []           # large jobs still in the queue
        self._pools: dict[str, tuple[int, ThreadPoolExecutor]] = {}
        #: the working set the last child reader reported. The memory gate has to
        #: consider it: that memory belongs to this machine just as much as ours.
        self.child_rss = 0

    # -- pools --------------------------------------------------------------
    def _pool_for(self, key: str, size: int) -> ThreadPoolExecutor:
        """A pool sized to the *current* concurrency limit for that kind of work.

        Large reads get a pool exactly `MAX_CONCURRENT_UPLOADS` wide, so the
        pool's own queue is the waiting line. Small files use a separate, wider
        pool, so they can never be stuck behind a big read that is holding a
        worker.

        The limit lives at module level so `/api/limits` and the tests can change
        it; the pool is rebuilt if it has moved since we last looked.
        """
        size = max(1, int(size))
        with self._lock:
            current = self._pools.get(key)
            if current is not None and current[0] == size:
                return current[1]
            previous = current[1] if current is not None else None
            pool = ThreadPoolExecutor(max_workers=size,
                                      thread_name_prefix=f"simarena-{key}")
            self._pools[key] = (size, pool)
            if previous is not None:
                try:
                    previous.shutdown(wait=False)
                except Exception:
                    pass
            return pool

    # -- registry -----------------------------------------------------------
    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._trim()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(str(job_id or ""))

    def _trim(self) -> None:
        """Keep memory bounded: forget the oldest finished jobs first."""
        while len(self._order) > MAX_RETAINED_JOBS:
            victim_id = None
            for candidate in self._order:
                job = self._jobs.get(candidate)
                if job is not None and job.is_terminal():
                    victim_id = candidate
                    break
            if victim_id is None:
                break
            self._order.remove(victim_id)
            job = self._jobs.pop(victim_id, None)
            if job is not None:
                job.feed.clear()

    # -- the queue ----------------------------------------------------------
    def enqueue_large(self, job: Job) -> None:
        with self._lock:
            self._waiting.append(job.id)
            self._refresh_positions_locked()
        job.future = self._pool_for("upload", MAX_CONCURRENT_UPLOADS).submit(
            _run_upload, job)

    def leave_queue(self, job: Job) -> None:
        with self._lock:
            if job.id in self._waiting:
                self._waiting.remove(job.id)
            self._refresh_positions_locked()

    def _refresh_positions_locked(self) -> None:
        """Tell everyone still waiting exactly where they are in the line."""
        pending: list[Job] = []
        for position, job_id in enumerate(list(self._waiting), start=1):
            job = self._jobs.get(job_id)
            if job is None or job.cancel_event.is_set() or job.is_terminal():
                continue
            job.update(queue_position=position, status=QUEUED, phase=QUEUED,
                       message=self._queue_message(position, job.name))
            pending.append(job)
        for job in pending:
            job.publish_progress()

    @staticmethod
    def _queue_message(position: int, name: str) -> str:
        if position <= 1:
            tail = "you are next in line"
        else:
            tail = f"you are {_ordinal(position)} in line"
        return f"Another file is being read right now — {tail}."


_MANAGER = _Manager()


# ---------------------------------------------------------------------------
# Public lookups
# ---------------------------------------------------------------------------
def get(job_id: str) -> Optional[Job]:
    return _MANAGER.get(job_id)


def active_count() -> int:
    with _MANAGER._lock:
        return sum(1 for j in _MANAGER._jobs.values()
                   if not j.is_terminal())


def wait_for_job(job: Job, timeout_s: Optional[float] = None) -> bool:
    """Block until a job finishes. Returns True if it reached a terminal state.

    This is the whole of the `?sync=true` path. It deliberately *waits* on a job
    that already runs on a worker thread - or in its own process, for a big file
    - so the event loop is never the thing doing the reading. The route calls it
    through `asyncio.to_thread`, which is what lets a script keep getting a
    profile back synchronously without being able to freeze the server.
    """
    if timeout_s is None:
        timeout_s = max(60.0, float(PARSE_TIMEOUT_S) + 120.0)
    return bool(job.finished_event.wait(float(timeout_s)))


# ---------------------------------------------------------------------------
# Starting work
# ---------------------------------------------------------------------------
def start_upload(name: str, path: Path, bytes_total: int) -> Job:
    """Begin reading a spooled upload. Returns immediately."""
    job = Job("upload", name, bytes_total)
    job.temp_path = Path(path)
    _MANAGER.add(job)
    if int(bytes_total or 0) >= int(LARGE_FILE_BYTES):
        job.update(status=QUEUED, phase=QUEUED,
                   message=_MANAGER._queue_message(1, job.name))
        _MANAGER.enqueue_large(job)
    else:
        job.update(status=QUEUED, phase=QUEUED,
                   message=f"Starting on '{job.name}'…")
        job.future = _MANAGER._pool_for("small", 4).submit(_run_upload, job)
    return job


def start_paste(name: str, rows: list[dict]) -> Job:
    """Begin profiling a big paste. Returns immediately."""
    job = Job("paste", name, 0)
    job.rows_in = rows
    job.rows_total = len(rows)
    job.update(status=QUEUED, phase=QUEUED,
               message=f"Getting {_human_count(len(rows))} pasted rows ready…")
    _MANAGER.add(job)
    job.future = _MANAGER._pool_for("small", 4).submit(_run_paste, job)
    return job


def reject(name: str, bytes_total: int, message: str) -> Job:
    """A job that never starts: refused on size before a byte is read."""
    job = Job("upload", name, bytes_total)
    _MANAGER.add(job)
    job.update(status=REJECTED, phase="refused", error=message, message=message,
               eta_s=None, pct=None, queue_position=None)
    job.finished_event.set()
    job.ended_at = time.time()
    job.publish_terminal("error", {"message": message, "trace": "",
                                   "status": REJECTED})
    return job


def cancel(job: Job) -> str:
    """Stop a job. Returns the status the job is left in."""
    if job.is_terminal():
        return job.snapshot()["status"]

    job.cancel_event.set()
    _MANAGER.leave_queue(job)
    job.update(status=CANCELLED, phase="cancelled", message=CANCELLED_MESSAGE,
               queue_position=None, eta_s=None, error=None)

    future = job.future
    if future is not None and future.cancel():
        # it never started, so nobody else will close it out - or free the
        # spooled upload it is still holding
        _finalise(job, CANCELLED, CANCELLED_MESSAGE, "")
        _release(job)
    else:
        job.publish_progress()
    return job.snapshot()["status"]


# ---------------------------------------------------------------------------
# The workers
# ---------------------------------------------------------------------------
class _IngestProblem(Exception):
    """A read that ended in a state the parent must turn into a status."""

    def __init__(self, message: str, status: str = FAILED) -> None:
        super().__init__(message)
        self.message = str(message)
        self.status = status


def _run_upload(job: Job) -> None:
    started = time.perf_counter()
    job.update(started_at=started)
    _MANAGER.leave_queue(job)
    with _MANAGER._lock:
        _MANAGER.child_rss = 0
    _start_watchdog(job, started)
    try:
        if job.cancel_event.is_set():
            raise ingest.IngestCancelled()
        _wait_for_memory(job)
        if job.cancel_event.is_set():
            raise ingest.IngestCancelled()

        job.update(status=READING, phase=READING, queue_position=None,
                   message=f"Opening '{job.name}'…")
        job.publish_progress()

        use_child = (int(CHILD_INGEST_BYTES) > 0
                     and int(job.bytes_total or 0) >= int(CHILD_INGEST_BYTES))
        if use_child:
            # The child wrote the parquet cache and the profile. We deliberately
            # do NOT rebuild the rows here: turning ~2 million rows into Python
            # dicts holds the GIL for seconds and starves the event loop, which
            # is exactly the outage this whole path exists to prevent. The
            # dataset is described by its profile and served off the cache.
            columns, profile, staged = _ingest_in_child(job, started)
            dataset_id = store.put_dataset(profile, [], persist=False, lazy=True)
            if not store.adopt_parquet(dataset_id, staged):
                store.drop_dataset(dataset_id)
                _finalise(job, FAILED,
                          f"We read '{job.name}' but could not keep the rows "
                          f"anywhere. Nothing was saved - please try again.", "")
                return
            rows = None
        else:
            rows, columns = ingest.read_table(
                job.temp_path, job.name,
                narrator=None,
                progress=lambda payload: _on_progress(job, payload),
                cancel=job.cancel_event.is_set,
                limits=ingest_limits(),
            )
            if not columns:
                _finalise(job, FAILED, NO_COLUMNS_MESSAGE, "")
                return
            if not rows:
                _finalise(job, FAILED, EMPTY_MESSAGE, "")
                return

            job.update(status=PROFILING, phase=PROFILING_PHASE_TEXT, pct=None,
                       eta_s=None, rate_rows_s=0.0,
                       message=(f"Read {_human_count(len(rows))} rows. Working out "
                                f"what each column holds…"))
            job.publish_progress()

            profile = ingest.profile(rows, columns, job.name)
            dataset_id = store.put_dataset(profile, rows)

        job.update(warnings=list(profile.warnings or []))
        job.profile = profile

        kept = len(rows) if rows is not None else int(profile.n_rows or 0)
        # QA NEW-05: name the file's real size as well as the size we used. The
        # old message said only "we used a fair sample of N rows", so the two
        # numbers the user needs - how big their file was, and how much of it
        # this run used - were never on the screen together.
        in_file = int(getattr(profile, "n_rows_in_file", 0) or 0)
        sampled = bool(getattr(profile, "sampled", False))
        if rows is not None:
            sampled = bool(getattr(rows, "sampled", False)) or sampled
        if sampled:
            # Prefer the reader's own sentence: it names the record key when the
            # sample was drawn in whole business records (QA NEW-10 / TC-02).
            note = str(getattr(profile, "sample_note", "") or "")
            if not note:
                note = (f" Your file holds {_human_count(in_file or kept)} rows; this run "
                        f"uses an evenly spread sample of {_human_count(kept)} of them.")
        else:
            note = ""
        ready = (f"'{job.name}' is ready — {_human_count(kept)} rows and "
                 f"{len(columns)} columns.{note}")
        # `rows_read`/`rows_total` are deliberately not set here: the reading
        # phase already counted every row we looked at, and a sampled file would
        # otherwise appear to go *backwards* (5M seen -> 2M kept) at the end.
        job.update(status=READY, phase="done", dataset_id=dataset_id, pct=100.0,
                   eta_s=0.0,
                   rate_rows_s=round(kept / max(1e-6, time.perf_counter() - started), 1),
                   bytes_read=job.bytes_total, message=ready)
        job.publish_progress()
        job.publish_terminal("done", {"dataset_id": dataset_id,
                                      "profile": profile.model_dump(),
                                      "status": READY})

    except ingest.IngestRejected as exc:
        message = str(getattr(exc, "message", "") or exc)
        if job.timed_out:
            _finalise(job, FAILED, message, "")
        else:
            _finalise(job, REJECTED, message, "")
    except ingest.IngestCancelled:
        if job.timed_out:
            _finalise(job, FAILED, timeout_message(job.name, PARSE_TIMEOUT_S), "")
        else:
            _finalise(job, CANCELLED, CANCELLED_MESSAGE, "")
    except _IngestProblem as exc:
        _finalise(job, exc.status, exc.message, "")
    except Exception as exc:                       # never a silent hang, never a 500
        message = (f"Something went wrong while reading '{job.name}': "
                   f"{type(exc).__name__}. Try saving it as Parquet or CSV and "
                   f"uploading again.")
        _finalise(job, FAILED, message, traceback.format_exc())
    finally:
        _release(job)


def _run_paste(job: Job) -> None:
    started = time.perf_counter()
    job.update(started_at=started)
    _start_watchdog(job, started)
    try:
        rows_in = job.rows_in or []
        total = max(1, len(rows_in))
        rows: list[dict] = []
        columns: list[str] = []
        seen: set[str] = set()
        for i, item in enumerate(rows_in):
            if not isinstance(item, dict):
                continue
            for key in item.keys():
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
            rows.append(dict(item))
            if i and i % 20_000 == 0:
                if job.cancel_event.is_set():
                    raise ingest.IngestCancelled()
                pct = round(98.0 * i / total, 2)
                job.update(status=READING, phase="reading", rows_read=len(rows),
                           rows_total=total, pct=pct, queue_position=None,
                           message=(f"Getting your pasted rows ready — "
                                    f"{_human_count(i)} of {_human_count(total)}…"))
                job.publish_progress()

        columns = ingest.sanitise_headers(columns)
        if not columns:
            _finalise(job, FAILED, NO_COLUMNS_MESSAGE, "")
            return
        rows = [{c: _jsonable(r.get(c, None)) for c in columns} for r in rows]

        job.update(status=PROFILING, phase=PROFILING_PHASE_TEXT, pct=None,
                   eta_s=None, rate_rows_s=0.0,
                   rows_read=len(rows), rows_total=len(rows),
                   message=(f"Working out what each of the {len(columns)} columns "
                            f"holds…"))
        job.publish_progress()

        profile = ingest.profile(rows, columns, job.name)
        job.update(warnings=list(profile.warnings or []))
        dataset_id = store.put_dataset(profile, rows)
        job.profile = profile
        ready = (f"'{job.name}' is ready — {_human_count(len(rows))} rows and "
                 f"{len(columns)} columns.")
        job.update(status=READY, phase="done", dataset_id=dataset_id, pct=100.0,
                   eta_s=0.0, message=ready)
        job.publish_progress()
        job.publish_terminal("done", {"dataset_id": dataset_id,
                                      "profile": profile.model_dump(),
                                      "status": READY})
    except ingest.IngestCancelled:
        _finalise(job, CANCELLED, CANCELLED_MESSAGE, "")
    except Exception as exc:
        message = (f"Something went wrong with the data you pasted: "
                   f"{type(exc).__name__}.")
        _finalise(job, FAILED, message, traceback.format_exc())
    finally:
        job.rows_in = None
        _release(job)


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, float):
        return None if v != v else v
    if isinstance(v, (list, tuple, dict, set)):
        return v
    return str(v)


# ---------------------------------------------------------------------------
# Reading a big file in its own process
#
# Polars releases the GIL while it is actually parsing, so a worker thread is
# enough for that. Turning a million Python dicts into a frame is a different
# story: it is one long call that holds the GIL, and while it runs no other
# thread - including the one answering /api/health - gets to execute a single
# bytecode. Measured on this machine: 1.4 seconds for 1.9 million rows.
#
# So a big file is read in a child process. The event loop cannot be frozen by
# anything a child does, the child can be killed outright on cancel, and its
# memory is a separate, bounded thing.
# ---------------------------------------------------------------------------
def _child_ingest(source_path: str, name: str, limits_kwargs: dict,
                  progress_q: Any, cancel_event: Any, result_q: Any,
                  parquet_path: str) -> None:
    """Runs in the CHILD. Reads, profiles and caches, then reports back."""
    try:
        limits = ingest.IngestLimits(**limits_kwargs)

        def report(payload: dict) -> None:
            message = dict(payload)
            message["rss"] = process_rss_bytes()
            try:
                progress_q.put_nowait(message)
            except Exception:
                pass

        rows, columns = ingest.read_table(
            source_path, name, narrator=None, progress=report,
            cancel=cancel_event.is_set, limits=limits)
        if not columns:
            result_q.put({"ok": False, "kind": "no_columns"})
            return
        if not rows:
            result_q.put({"ok": False, "kind": "empty"})
            return

        # The profiling step has no honest percentage or time-left, and the
        # rows we *read* are not the rows we *kept*: report the number we
        # actually looked at so the counter never jumps backwards at the end.
        n_seen = int(getattr(rows, "n_rows_in_file", 0) or len(rows))
        kept = len(rows)
        if kept < n_seen:
            note = (f"Read {_human_count(n_seen)} rows and kept a "
                    f"{_human_count(kept)}-row sample spread evenly across the "
                    f"file. Working out what each column holds…")
        else:
            note = (f"Read {_human_count(n_seen)} rows. Working out what each "
                    f"column holds…")
        report({"phase": PROFILING, "message": note,
                "rows_read": n_seen, "rows_total": n_seen,
                "pct": None, "rate_rows_s": 0.0, "eta_s": None})
        profile = ingest.profile(rows, columns, name)

        store.write_parquet_chunked(Path(parquet_path), rows)
        result_q.put({
            "ok": True,
            "profile": profile.model_dump(),
            "columns": [str(c) for c in columns],
            "n_rows": len(rows),
            "sampled": bool(getattr(rows, "sampled", False)),
            "sample_note": str(getattr(rows, "sample_note", "") or ""),
            "parquet": str(parquet_path),
        })
    except ingest.IngestRejected as exc:
        result_q.put({"ok": False, "kind": "rejected",
                      "message": str(getattr(exc, "message", "") or exc)})
    except ingest.IngestCancelled:
        result_q.put({"ok": False, "kind": "cancelled"})
    except Exception as exc:
        result_q.put({"ok": False, "kind": "error",
                      "message": f"{type(exc).__name__}",
                      "trace": traceback.format_exc()})


def _ingest_in_child(job: Job, started: float):
    """Drive a child reader: forward its progress, and stop it hard if asked."""
    ctx = multiprocessing.get_context("spawn")
    store.ensure_dirs()
    progress_q = ctx.Queue()
    result_q = ctx.Queue()
    cancel_event = ctx.Event()
    staged = store.UPLOAD_DIR / f"{job.id}.parquet"
    limits = ingest_limits()

    proc = ctx.Process(
        target=_child_ingest,
        args=(str(job.temp_path), job.name,
              {"max_file_bytes": int(limits.max_file_bytes),
               "max_working_rows": int(limits.max_working_rows),
               "parse_timeout_s": float(limits.parse_timeout_s),
               "chunk_rows": int(limits.chunk_rows)},
              progress_q, cancel_event, result_q, str(staged)),
        daemon=True, name=f"ingest-{job.id}")
    proc.start()

    deadline = started + max(1.0, float(PARSE_TIMEOUT_S))
    result: Optional[dict] = None
    try:
        while True:
            while True:                       # drain whatever the child has said
                try:
                    payload = progress_q.get_nowait()
                except Exception:
                    break
                _on_child_progress(job, payload)

            if job.cancel_event.is_set():
                cancel_event.set()
                raise ingest.IngestCancelled()
            if time.perf_counter() > deadline:
                cancel_event.set()
                job.timed_out = True
                raise ingest.IngestRejected(
                    timeout_message(job.name, PARSE_TIMEOUT_S))

            try:
                result = result_q.get(timeout=0.1)
                break
            except queue.Empty:
                pass
            if not proc.is_alive():
                try:                          # it exited; collect its last word
                    result = result_q.get(timeout=10.0)
                except queue.Empty:
                    raise _IngestProblem(
                        f"We lost contact with the reader for '{job.name}'. "
                        f"Nothing was saved - please try again.")
                break
    finally:
        _terminate(proc)
        for q in (progress_q, result_q):
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass

    result = result or {}
    if not result.get("ok"):
        _discard_staged(staged)
        kind = result.get("kind")
        if kind == "cancelled":
            raise ingest.IngestCancelled()
        if kind == "rejected":
            raise ingest.IngestRejected(result.get("message") or "")
        if kind == "no_columns":
            raise _IngestProblem(NO_COLUMNS_MESSAGE)
        if kind == "empty":
            raise _IngestProblem(EMPTY_MESSAGE)
        raise _IngestProblem(
            f"We could not finish reading '{job.name}' "
            f"({result.get('message') or 'the reader stopped'}). Please try again.")

    if not staged.exists():
        raise _IngestProblem(
            f"We read '{job.name}' but could not keep the rows. Please try again.")
    columns = list(result.get("columns") or [])
    profile = _profile_from_payload(result["profile"])
    return columns, profile, staged


def _discard_staged(path: Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except BaseException:
        pass


def _profile_from_payload(payload: dict):
    """Rebuild the DatasetProfile the child sent back."""
    from ..core.contracts import DatasetProfile
    return DatasetProfile.model_validate(payload)


def _on_child_progress(job: Job, payload: dict) -> None:
    rss = payload.get("rss")
    if rss:
        with _MANAGER._lock:
            _MANAGER.child_rss = max(_MANAGER.child_rss, int(rss))
    _on_progress(job, payload)


def _terminate(proc: Any) -> None:
    """Stop a child reader for good."""
    try:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass
            proc.join(timeout=2.0)
    except Exception:
        pass
    try:
        proc.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Progress, memory gate, watchdog, close-out
# ---------------------------------------------------------------------------
def _on_progress(job: Job, payload: dict) -> None:
    """Turn the reader's progress into the contract's payload, in plain English.

    Three of QA's five defects are fixed right here:

    * **profiling says nothing it cannot back up.** Once the rows are in, the
      reader reports `phase == "profiling"`. There is no honest percentage or
      time-left for that step, so the job goes indeterminate (`pct: null`) with
      `eta_s: null` and a plain-English phase - never "98% - almost done" for
      eight seconds.
    * **the byte counter is carried forward.** The child reader's profiling
      report does not carry a byte count, so the old code reset `bytes_read` to
      0 and the UI had to hide the counter. A payload without a byte count now
      keeps the number we already had.
    * **no zero ETA while working.** An ETA of zero (or a negative/NaN one) is
      only ever published for a finished job; while work is going on it becomes
      `null`, which the UI reads as "not known yet".
    """
    if job.cancel_event.is_set() or job.is_terminal():
        return                                   # a cancelled job says nothing more
    try:
        rows_read = int(payload.get("rows_read") or 0)
        rows_total = payload.get("rows_total")
        rows_total = int(rows_total) if rows_total else None
        raw_pct = payload.get("pct")
        rate = float(payload.get("rate_rows_s") or 0.0)
        eta = payload.get("eta_s")
        eta = float(eta) if eta is not None else None
        incoming_phase = str(payload.get("phase") or READING)
        profiling = incoming_phase == PROFILING
        bytes_total = payload.get("bytes_total")
        sampling = bool(rows_total and rows_total > int(MAX_WORKING_ROWS))

        # The reader measures rows a second over a short rolling window, so the
        # last few samples of a fast read can flatter it. Never claim a rate
        # faster than the whole job's own average.
        elapsed = time.perf_counter() - (job.started_at or time.perf_counter())
        if elapsed > 0.2 and rows_read > 0:
            rate = min(rate, rows_read / elapsed)

        # The reader's own sentence is used for the opening "sizing up your big
        # file" notice; once there are rows to count we build the sentence from
        # the numbers we have already vetted.
        message = str(payload.get("message") or "") if rows_read <= 0 else ""

        if profiling:
            # Defect 1: the columns are being worked out. Say so plainly, drop
            # to an indeterminate bar and admit we do not know how long is left.
            phase = PROFILING_PHASE_TEXT
            pct: Optional[float] = None
            eta = None
            rate = 0.0
            message = str(payload.get("message") or "") or PROFILING_MESSAGE
        else:
            if not message:
                message = (_sampling_message(rows_read, eta, rate) if sampling
                           else _reading_message(rows_read, eta, rate))
            # the read is 98% of the job; profiling and storing are the rest
            pct = None if raw_pct is None else round(min(98.0, float(raw_pct) * 0.98), 2)
            # Defect 1 (invariant): a job that is still working must never claim
            # a zero ETA. `null` means "not known yet", which is the truth.
            if eta is not None and (not math.isfinite(eta) or eta <= 0.0):
                eta = None
            phase = incoming_phase

        # Defect 2: never let the byte counter fall back to zero. A payload that
        # carries no byte count keeps the value we already had; `bytes_total`
        # never shrinks either. (A payload that carries a *smaller* count is a
        # real regression, and `Job.update` raises on it.)
        incoming_bytes = int(payload.get("bytes_read") or 0)
        bytes_read = incoming_bytes if incoming_bytes > 0 else int(job.bytes_read or 0)
        incoming_total = int(bytes_total) if bytes_total else 0
        stable_total = max(int(job.bytes_total or 0), incoming_total)

        job.update(
            status=PROFILING if profiling else (SAMPLING if sampling else READING),
            phase=phase,
            rows_read=rows_read,
            rows_total=rows_total,
            bytes_read=bytes_read,
            bytes_total=stable_total,
            pct=pct,
            rate_rows_s=round(rate, 1),
            eta_s=None if eta is None else round(eta, 2),
            message=message,
            queue_position=None,
        )
    except ProgressRegression:
        raise                    # a bug in this module - do not hide it
    except Exception:
        return
    job.publish_progress()


def _reading_message(rows: int, eta: Optional[float], rate: float) -> str:
    word = "row" if rows == 1 else "rows"
    text = f"Reading your file — {_human_count(rows)} {word} so far"
    if eta is not None and eta >= 1.0:
        text += f", about {_human_duration(eta)} left"
    elif rate > 0:
        text += f", {_human_count(rate)} rows a second"
    return text


def _sampling_message(rows: int, eta: Optional[float], rate: float) -> str:
    # QA NEW-10 / TC-02: say that we keep whole records, so a reader is never
    # left thinking we cut their file up line by line.
    text = (f"Reading your file — {_human_count(rows)} rows so far. Your file is "
            f"bigger than we work with, so we are keeping a fair sample of "
            f"{_human_count(MAX_WORKING_ROWS)} rows, taking whole records rather "
            f"than loose lines")
    if eta is not None and eta >= 1.0:
        text += f", about {_human_duration(eta)} left"
    elif rate > 0:
        text += f" ({_human_count(rate)} rows a second)"
    return text


def _memory_room_available() -> bool:
    """Is this process (plus any child reader) under the working-set ceiling?"""
    ceiling = int(MEMORY_CEILING_BYTES or 0)
    if ceiling <= 0:
        return True
    used = process_rss_bytes() or 0
    with _MANAGER._lock:
        used = max(used, int(_MANAGER.child_rss or 0))
    return used < ceiling


def _wait_for_memory(job: Job) -> None:
    """Do not start a big read while the machine is already close to swapping."""
    if int(MEMORY_CEILING_BYTES or 0) <= 0:
        return
    announced = False
    while not job.cancel_event.is_set():
        if _memory_room_available():
            if announced:
                job.update(status=QUEUED, phase=QUEUED,
                           message=f"Starting on '{job.name}'…")
                job.publish_progress()
            return
        if not announced:
            announced = True
            job.update(status=QUEUED, phase=QUEUED, pct=None, eta_s=None,
                       message=MEMORY_MESSAGE)
            job.publish_progress()
        job.cancel_event.wait(2.0)
    raise ingest.IngestCancelled()


def _start_watchdog(job: Job, started_at: float) -> None:
    """Enforce the parse deadline, and make sure a stuck read still ends."""
    def watch() -> None:
        deadline = started_at + max(1.0, float(PARSE_TIMEOUT_S))
        while not job.finished_event.wait(1.0):
            if job.cancel_event.is_set():
                continue
            if time.perf_counter() < deadline:
                continue
            job.timed_out = True
            job.cancel_event.set()               # the reader stops at its next chunk
            message = timeout_message(job.name, PARSE_TIMEOUT_S)
            job.update(status=FAILED, phase="failed", message=message,
                       error=message, eta_s=None, queue_position=None)
            job.publish_progress()
            # the reader should come back within a chunk; if it does not, do not
            # leave the user staring at a spinner for ever.
            if job.finished_event.wait(min(120.0, max(10.0, PARSE_TIMEOUT_S))):
                return
            _finalise(job, FAILED, message,
                      "the reader did not stop when asked")
            return

    threading.Thread(target=watch, name=f"watch-{job.id}", daemon=True).start()


def _finalise(job: Job, status: str, message: str, trace: str) -> None:
    """Move a job to its terminal state and close every listener out."""
    job.update(status=status, phase=status, message=message,
               error=message if status in (FAILED, REJECTED) else None,
               eta_s=None, queue_position=None)
    job.ended_at = time.time()
    job.publish_progress()
    if status == READY:
        job.publish_terminal("done", {"dataset_id": job.dataset_id,
                                      "profile": (job.profile.model_dump()
                                                  if job.profile else None),
                                      "status": READY})
    else:
        # Defect 3: the closing event names the state it ended in, so the client
        # can tell a cancellation from a refusal from a crash without asking.
        job.publish_terminal("error", {"message": message, "trace": trace or "",
                                       "status": status})
    job.finished_event.set()
    if status in (CANCELLED, FAILED):
        store.drop_dataset(job.dataset_id)
        job.profile = None


def _release(job: Job) -> None:
    """Free everything a finished job was holding, and drop its spooled file."""
    job.finished_event.set()
    if job.ended_at is None:
        job.ended_at = time.time()
    job.rows_in = None
    if job.status in (CANCELLED, FAILED, REJECTED):
        job.profile = None
    with _MANAGER._lock:
        _MANAGER.child_rss = 0                 # that memory is gone now
    path = job.temp_path
    job.temp_path = None
    if path is not None:
        try:
            Path(path).unlink(missing_ok=True)
        except BaseException:
            pass
    if job.status in (CANCELLED, FAILED, REJECTED):
        _discard_staged(store.UPLOAD_DIR / f"{job.id}.parquet")


def shutdown() -> None:
    """Stop the workers and clear any spooled uploads still on disk."""
    with _MANAGER._lock:
        pools = [entry[1] for entry in _MANAGER._pools.values()]
        _MANAGER._pools = {}          # a later job builds a fresh pool
        _MANAGER._waiting = []
    for pool in pools:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            try:
                pool.shutdown(wait=False)
            except Exception:
                pass
    try:
        store.clear_uploads()
    except Exception:
        pass
