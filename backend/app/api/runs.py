"""The run pipeline: request -> role-mapped rows -> simulation -> artifacts.

A run happens on a worker thread so the event loop stays free to serve the
live log stream (SSE). The narrator emits plain-English lines while the
simulation is running; a pump thread forwards each one to every listener the
moment it happens.

Nothing in here is allowed to raise out of ``start_run``: a broken run becomes
an `error` event on the stream, never a dead server.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from typing import Any, Optional

from ..core.contracts import LogEvent, RunRequest, RunResult
from ..core.narrate import Narrator
from ..core.registry import RunContext
from ..core import registry
from ..core.rng import make_rng
from ..io import ingest, schema_map
from ..reporting import exports as reporting
from . import store
from .deps import json_safe, plain_number

#: how long we let the SSE generator wait before giving up on a stuck run.
#: The env override exists so the timeout-then-retry journey (QA SP-05) can be
#: driven end to end in seconds rather than fifteen minutes. A stream that ends
#: this way is NOT the end of the run: the run keeps going on its worker thread,
#: and the page falls back to asking for the result directly.
MAX_STREAM_SECONDS = float(os.environ.get("SIMARENA_MAX_STREAM_SECONDS", "") or 900.0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def start_run(req: RunRequest) -> str:
    """Create a run, kick it off on a worker thread, return its id."""
    run_id = store.create_run(req.use_case_id)
    worker = threading.Thread(target=_execute, args=(run_id, req),
                              name=f"run-{run_id}", daemon=True)
    worker.start()
    return run_id


def status_of(run_id: str) -> str:
    return store.get_status(run_id)


def result_of(run_id: str) -> Optional[RunResult]:
    return store.get_result(run_id)


def log_of(run_id: str) -> list[LogEvent]:
    """Everything that was said during a run (replay for a late listener)."""
    result = store.get_result(run_id)
    if result is not None and result.log:
        return list(result.log)
    out: list[LogEvent] = []
    for ev in store.events(run_id):
        if ev.get("event") == "log":
            try:
                out.append(LogEvent.model_validate(ev.get("data") or {}))
            except Exception:
                continue
    return out


# ---------------------------------------------------------------------------
# Narrator -> SSE pump
# ---------------------------------------------------------------------------
def _pump(run_id: str, q: queue.Queue, stop: threading.Event) -> None:
    """Forward narrator events to every SSE listener as they happen."""

    def drain() -> None:
        while True:
            try:
                ev = q.get_nowait()
            except queue.Empty:
                return
            store.publish(run_id, {"event": "log", "data": json_safe(ev.model_dump())})

    while True:
        try:
            ev = q.get(timeout=0.15)
        except queue.Empty:
            if stop.is_set():
                drain()
                return
            continue
        store.publish(run_id, {"event": "log", "data": json_safe(ev.model_dump())})


# ---------------------------------------------------------------------------
# Row resolution
# ---------------------------------------------------------------------------
def _columns_of(rows: list[dict]) -> list[str]:
    """Union of the keys, in first-seen order, cleaned up."""
    seen: list[str] = []
    for r in rows or []:
        for k in (r or {}).keys():
            if k not in seen:
                seen.append(k)
    return ingest.sanitise_headers(seen)


def _resolve_rows(req: RunRequest, narrator: Narrator):
    """(rows, profile). Request rows win; otherwise the stored dataset."""
    if req.rows:
        narrator.say("Opening your file…", phase="prepare", progress=0.03)
        rows = [dict(r or {}) for r in req.rows]
        columns = _columns_of(rows)
        rows = [{c: r.get(c, None) for c in columns} for r in rows]
        profile = ingest.profile(rows, columns, req.dataset_id or "Pasted data")
        narrator.say(f"Found {len(rows):,} rows and {len(columns)} columns.",
                     phase="prepare", progress=0.10)
        return rows, profile

    if req.dataset_id:
        narrator.say("Opening your file…", phase="prepare", progress=0.03)
        found = store.get_dataset(req.dataset_id)
        if found is None:
            narrator.warn("We could not find that data anymore, so this run uses "
                          "your assumptions only.")
            return [], None
        profile, rows = found
        rows = list(rows or [])
        held = int(getattr(profile, "n_rows_in_file", 0) or 0)
        if bool(getattr(profile, "sampled", False)) and held:
            narrator.say(f"Using {len(rows):,} rows of the {held:,} in "
                         f"'{profile.source_name}'.", phase="prepare", progress=0.10,
                         detail=f"source: {profile.source_name}")
        else:
            narrator.say(f"Found {len(rows):,} rows and {profile.n_cols} columns.",
                         phase="prepare", progress=0.10,
                         detail=f"source: {profile.source_name}")
        return rows, profile
    return [], None


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------
def _build_mapping(req: RunRequest, roles, profile, narrator: Narrator) -> dict[str, Any]:
    """The mapping that will actually be used - resolved once, never re-derived.

    Correctness contract rule 8: *the role mapping the user saw must be the
    mapping actually used.* `schema_map.resolve_mapping` is the single source of
    truth for that, and the dict it returns is handed to `apply_mapping`
    untouched. (The old code re-ran auto-map over only the roles the user had
    left blank, which let a column already spoken for - `account_id` - be handed
    to a second role, so the screen said "tier: not found" while the run quietly
    used `account_id` as the tier.)
    """
    columns = [c.name for c in (profile.columns if profile else [])]
    profiles = list(profile.columns) if profile else []
    n_rows = getattr(profile, "n_rows", None) if profile else None
    mapping, _confidence, notes, _was_auto = schema_map.resolve_mapping(
        req.role_mapping, columns, profiles, roles, n_rows=n_rows)
    for line in notes:
        narrator.say(line, phase="prepare", progress=0.16)
    for line in schema_map.describe_mapping(mapping, roles):
        narrator.say(line, phase="prepare", progress=0.16, detail="mapping used")
    return mapping


def _merge_params(meta, req: RunRequest) -> dict[str, Any]:
    """defaults <- request.parameters <- manual_inputs (later wins)."""
    params: dict[str, Any] = {p.key: p.default for p in (meta.parameters or [])}
    params.update({k: v for k, v in (req.parameters or {}).items() if v is not None})
    params.update({k: v for k, v in (req.manual_inputs or {}).items() if v is not None})

    kinds = {p.key: (p.kind or "number") for p in (meta.parameters or [])}
    for key, kind in kinds.items():
        if key not in params or params[key] is None:
            continue
        value = params[key]
        try:
            if kind == "number":
                coerced = plain_number(value)
                params[key] = value if coerced is None else coerced
            elif kind == "integer":
                coerced = plain_number(value)
                params[key] = value if coerced is None else int(coerced)
            elif kind == "bool":
                params[key] = value if isinstance(value, bool) else \
                    str(value).strip().lower() in ("true", "1", "yes", "y", "on")
        except Exception:
            pass
    return params


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def _execute(run_id: str, req: RunRequest) -> None:
    t0 = time.perf_counter()
    narrator = Narrator()
    listener = narrator.subscribe()
    stop = threading.Event()
    pump = threading.Thread(target=_pump, args=(run_id, listener, stop),
                            name=f"pump-{run_id}", daemon=True)
    pump.start()

    try:
        store.set_status(run_id, store.STATUS_RUNNING)
        store.publish(run_id, {"event": "status", "data": {"status": "running"}})

        use_case = registry.get(req.use_case_id)
        if use_case is None:
            raise LookupError(f"we do not have a simulation called '{req.use_case_id}'")
        meta = use_case.meta

        rows, profile = _resolve_rows(req, narrator)
        if not rows:
            narrator.say("No data was attached, so this run uses the numbers you "
                         "typed in.", phase="prepare", progress=0.12)

        # QA NEW-05 / rule 8: if the app is running on less than the whole file,
        # that has to reach the user through the run log AND the result - not
        # only through an upload panel that is gone by the time they read the
        # answer.
        reduction_note = ""
        if profile is not None and bool(getattr(profile, "sampled", False)):
            reduction_note = str(getattr(profile, "sample_note", "") or "")
            if not reduction_note:
                reduction_note = ingest._sample_note(
                    int(getattr(profile, "n_rows_in_file", 0) or profile.n_rows),
                    int(profile.n_rows or 0))
            narrator.say(reduction_note, phase="prepare", level="warn", progress=0.11)

        roles = list(meta.input_roles or [])
        mapping = _build_mapping(req, roles, profile, narrator)
        mapped, issues = schema_map.apply_mapping(rows, mapping, roles)
        # rule 8: write down the mapping that was actually used, the moment it
        # is used. `result` does not exist yet, so keep the lines and attach
        # them below - they must reach result.assumptions either way.
        mapping_lines = schema_map.describe_mapping(mapping, roles)

        params = _merge_params(meta, req)

        if not rows:
            narrator.say("Checking your assumptions…", phase="prepare", progress=0.20)
        else:
            narrator.say(f"Read {len(mapped):,} records into the model.",
                         phase="prepare", progress=0.20)
        narrator.say("Running the simulation…", phase="simulate", progress=0.25)

        ctx = RunContext(
            narrator=narrator,
            rng_factory=lambda *tags: make_rng(req.seed, *tags),
            rows=mapped,
            params=params,
            run_id=run_id,
            req=req,
            profile=profile,
        )

        result = use_case.run(req, ctx)
        if not isinstance(result, RunResult):
            raise TypeError("the simulation did not return a result we understand")
        result.run_id = run_id
        result.use_case_id = meta.id
        result.use_case_title = result.use_case_title or meta.title
        result.seed = req.seed
        result.status = "succeeded"
        result.runtime_ms = int((time.perf_counter() - t0) * 1000)

        # rule 8: what the user saw is what the run used.
        for line in mapping_lines:
            if line not in result.assumptions:
                result.assumptions.append(line)

        if reduction_note:
            if reduction_note not in result.warnings:
                result.warnings.append(reduction_note)
            used = int(getattr(profile, "n_rows", 0) or len(rows or []))
            held = int(getattr(profile, "n_rows_in_file", 0) or used)
            short = (f"Data used: {used:,} of the {held:,} rows in your file "
                     f"({(100.0 * used / held) if held else 100.0:.1f}%), drawn evenly "
                     f"across it.")
            if short not in result.assumptions:
                result.assumptions.append(short)

        for line in issues or []:
            if line not in result.warnings:
                result.warnings.append(line)

        narrator.say("Building your report…", phase="report", progress=0.92)
        formats = list(req.output_formats or ["html"])
        try:
            built = reporting.build_artifacts(result, formats, str(store.run_dir(run_id)))
        except Exception as exc:                      # a bad export must not lose a run
            built = {}
            result.warnings.append(
                f"We could not build your downloadable files ({type(exc).__name__}). "
                f"The numbers above are still correct.")
        result.artifacts = dict(built)
        if built:
            names = ", ".join(sorted(built.keys()))
            narrator.say(f"Your report is ready to download ({names}).",
                         phase="report", progress=0.97)
        result.log = narrator.events()

        store.put_result(run_id, result)
        store.set_status(run_id, store.STATUS_SUCCEEDED)
        store.publish(run_id, {"event": "status", "data": {"status": "succeeded"}})
        store.publish(run_id, {"event": "result", "data": json_safe(result.model_dump())})

    except Exception as exc:
        reason = str(exc) or type(exc).__name__
        # QA NEW-03: say what actually happened. The old text left the user to
        # read "we lost the connection" from the browser, so they retried a run
        # that fails identically. This is the model's own check, not the network.
        message = (f"We could not finish this run: {reason} "
                   f"(this is a check on the model's own numbers, not a connection "
                   f"problem - the line above in the live log says where it stopped).")
        tb = traceback.format_exc()
        try:
            narrator.fail(message)
        except Exception:
            pass
        # let the pump flush that last line before we close the stream
        stop.set()
        try:
            pump.join(timeout=2.0)
        except Exception:
            pass
        store.put_error(run_id, message, tb)
        store.set_status(run_id, store.STATUS_FAILED)
        store.publish(run_id, {"event": "status", "data": {"status": "failed"}})
        store.publish(run_id, {"event": "error",
                               "data": {"message": message, "trace": tb}})
        try:
            print(f"[run {run_id}] {message}\n{tb}", flush=True)
        except Exception:
            pass
    finally:
        stop.set()
        try:
            pump.join(timeout=2.0)
        except Exception:
            pass
