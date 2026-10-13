"""HTTP surface of Simulation Arena.

Every endpoint below is exactly the contract in BRIEFING.md - the front-end is
coded against it, so paths, verbs and payload shapes must not drift.

Two house rules:
    * nothing user-facing is ever technical: plain English, always;
    * an exception never becomes a 500 - it becomes a plain-English message.

Reading a file is a *job*. `POST /api/datasets/upload` answers 202 immediately
and the reading happens on a worker thread (`app/api/jobs.py`), so a 1.4 GB
upload can never stop the server answering anything else. Progress, a row
counter and an ETA are available from `GET /api/datasets/jobs/{job_id}` and
streamed from `.../stream`. `?sync=true` keeps the blocking *contract* for
scripts and tests - it answers with the DatasetProfile - but it runs the same
job and waits for it off the event loop, so it cannot hang the server either.
"""
from __future__ import annotations

import asyncio
import json
import queue
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..core.contracts import DatasetProfile, LogEvent, RunRequest, RunResult
from ..core import registry
from ..io import ingest, samples
from ..io import schema_map
from ..reporting import exports as reporting
from . import jobs, runs, store
from .deps import as_profile, as_result, json_safe, rows_preview

router = APIRouter()

VERSION = "1.0.0"

MEDIA_TYPES = {
    "html": "text/html; charset=utf-8",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "application/zip",
    "pdf": "application/pdf",
}


def ok(payload: Any) -> JSONResponse:
    return JSONResponse(content=json_safe(payload))


def problem(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


def accepted(payload: Any) -> JSONResponse:
    return JSONResponse(status_code=202, content=json_safe(payload))


# ---------------------------------------------------------------------------
# Health / catalogue
# ---------------------------------------------------------------------------
@router.get("/health")
def health():
    registry.discover()
    return ok({
        "status": "ok",
        "version": VERSION,
        "use_cases": len(registry.all_use_cases()),
        "load_errors": registry.load_errors(),
    })


@router.get("/usecases")
def list_usecases():
    registry.discover()
    metas = registry.catalog()
    return ok([m.model_dump() for m in metas])


@router.get("/usecases/{use_case_id}")
def get_usecase(use_case_id: str):
    registry.discover()
    uc = registry.get(use_case_id)
    if uc is None:
        raise problem(404, f"We do not have a simulation called '{use_case_id}'.")
    return ok(uc.meta.model_dump())


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), sync: bool = False):
    """Take a file and start reading it.

    Returns 202 {"job_id","status"} at once; the reading happens on a worker
    thread. `?sync=true` keeps the old blocking *contract* for scripts and
    tests - it still answers with the DatasetProfile - but it no longer blocks
    the server: it starts the very same background job and waits for it on a
    worker thread, so one big file can never stop the server answering.
    """
    name = (file.filename or "uploaded file").strip() or "uploaded file"

    if sync:
        return await _upload_blocking(file, name)

    try:
        path = await _spool_upload(file)
    except Exception as exc:
        raise problem(400, f"We could not take the file you sent "
                           f"({type(exc).__name__}).")

    try:
        size = int(path.stat().st_size)
    except Exception:
        size = 0
    if size <= 0:
        _discard(path)
        raise problem(400, f"'{name}' is empty - there is nothing in it to read.")

    # Refuse an oversized file BEFORE reading a single byte of it, and drop the
    # spooled copy - there is no reason to keep a file we have just refused.
    limit = int(jobs.MAX_FILE_BYTES or 0)
    if limit and size > limit:
        _discard(path)
        job = jobs.reject(name, size, jobs.too_big_message(size, limit))
        return accepted(job.snapshot())

    job = jobs.start_upload(name, path, size)
    return accepted({"job_id": job.id, "status": jobs.QUEUED})


async def _upload_blocking(file: UploadFile, name: str) -> JSONResponse:
    """The `?sync=true` path: the same job, waited on, off the event loop.

    This used to read the file on the event loop and hand back the profile. One
    big file therefore stopped the server answering *anything* else for the
    length of the read - QA's C2 outage (25-51 minutes of total unavailability).
    It now spools the upload in bounded chunks, refuses anything over the same
    size limit, and starts exactly the same background job as the default path:
    the same parse deadline, the same memory ceiling and, for a big file, the
    same separate reader process. It then waits for that job on a worker thread,
    so the event loop is free the whole time and the server can never hang.
    """
    try:
        path = await _spool_upload(file)
    except Exception as exc:
        raise problem(400, f"We could not take the file you sent "
                           f"({type(exc).__name__}).")

    try:
        size = int(path.stat().st_size)
    except Exception:
        size = 0
    if size <= 0:
        _discard(path)
        raise problem(400, f"'{name}' is empty - there is nothing in it to read.")

    limit = int(jobs.MAX_FILE_BYTES or 0)
    if limit and size > limit:
        _discard(path)
        raise problem(400, jobs.too_big_message(size, limit))

    job = jobs.start_upload(name, path, size)
    finished = await asyncio.to_thread(jobs.wait_for_job, job)
    if not finished:
        raise problem(504, jobs.timeout_message(name, jobs.PARSE_TIMEOUT_S))

    snapshot = job.snapshot()
    if snapshot.get("status") == jobs.READY and job.profile is not None:
        return ok(job.profile.model_dump())
    raise problem(400, snapshot.get("error") or snapshot.get("message")
                  or f"We could not read '{name}' as a table.")


async def _spool_upload(file: UploadFile) -> Path:
    """Copy the incoming upload to our own file on disk, in bounded chunks.

    The bytes have to land somewhere; putting them on disk (rather than in RAM)
    is what keeps a 1.4 GB upload from costing 1.4 GB of memory. Each chunk is
    awaited, so the event loop keeps serving everything else while it happens.
    """
    store.ensure_dirs()
    path = store.UPLOAD_DIR / f"up_{uuid.uuid4().hex[:12]}.tmp"
    written = 0
    try:
        with open(path, "wb") as handle:
            while True:
                chunk = await file.read(4 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
    except Exception:
        _discard(path)
        raise
    if written <= 0:
        _discard(path)
    return path


def _discard(path: Optional[Path]) -> None:
    """Delete a spooled upload. Housekeeping must never become an error."""
    try:
        if path is not None:
            Path(path).unlink(missing_ok=True)
    except BaseException:
        pass


@router.post("/datasets/paste")
async def paste_dataset(body: dict, sync: bool = False):
    """Take a pasted table.

    A small paste is profiled on the spot and answered with the DatasetProfile,
    exactly as before. A big one (over `PASTE_ASYNC_ROWS` rows) becomes a job,
    the same as an upload, so it reports progress instead of freezing the page.
    """
    payload = body if isinstance(body, dict) else {}
    name = str(payload.get("name") or "Pasted data").strip() or "Pasted data"
    raw_rows = payload.get("rows") or []
    if not isinstance(raw_rows, list):
        raise problem(400, "We expected a list of rows, and got something else.")
    if not raw_rows:
        raise problem(400, "There are no rows in what you pasted.")

    if not sync and len(raw_rows) > int(jobs.PASTE_ASYNC_ROWS):
        job = jobs.start_paste(name, raw_rows)
        return accepted({"job_id": job.id, "status": jobs.QUEUED})

    try:
        profile = await asyncio.to_thread(_profile_paste, name, raw_rows)
    except _PasteProblem as exc:
        raise problem(400, exc.message)
    return ok(profile.model_dump())


class _PasteProblem(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = str(message)


def _profile_paste(name: str, raw_rows: list) -> DatasetProfile:
    rows: list[dict] = []
    columns: list[str] = []
    seen: set[str] = set()
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        for key in item.keys():
            if key not in seen:
                seen.add(key)
                columns.append(key)
        rows.append(dict(item))
    if not columns:
        raise _PasteProblem("We could not find any columns in what you pasted.")
    columns = ingest.sanitise_headers(columns)
    rows = [{c: json_safe(r.get(c, None)) for c in columns} for r in rows]
    profile = ingest.profile(rows, columns, name)
    store.put_dataset(profile, rows)
    return profile


# ---------------------------------------------------------------------------
# Reading jobs: progress, live stream, cancel
# ---------------------------------------------------------------------------
@router.get("/limits")
def get_limits():
    """The ceilings currently in force, in numbers and in plain English."""
    return ok(jobs.limits_payload())


@router.get("/datasets/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise problem(404, "We could not find that upload - it may have finished "
                           "and been cleared. Please upload the file again.")
    return ok(job.snapshot())


@router.get("/datasets/jobs/{job_id}/stream")
async def stream_job(job_id: str, request: Request):
    """Server-sent events: a `progress` line every moment, then `done`/`error`."""
    job = jobs.get(job_id)
    if job is None:
        raise problem(404, "We could not find that upload - it may have finished "
                           "and been cleared. Please upload the file again.")
    return StreamingResponse(
        _job_stream(job, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _job_stream(job, request: Request):
    listener = job.feed.subscribe()
    started = time.perf_counter()
    last_sent = time.perf_counter()
    try:
        while True:
            try:
                event = listener.get_nowait()
            except queue.Empty:
                event = None

            if event is None:
                if await _client_gone(request):
                    return
                if time.perf_counter() - started > jobs.MAX_STREAM_SECONDS:
                    yield ":keepalive\n\n"
                    return
                if time.perf_counter() - last_sent >= 15.0:
                    last_sent = time.perf_counter()
                    yield ":keepalive\n\n"
                await asyncio.sleep(0.05)
                continue

            name = str(event.get("event") or "message")
            try:
                data = json.dumps(json_safe(event.get("data")))
            except Exception:
                data = "{}"
            last_sent = time.perf_counter()
            yield f"event: {name}\ndata: {data}\n\n"

            if name in ("done", "error"):
                return
    finally:
        job.feed.unsubscribe(listener)


@router.post("/datasets/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Stop a read. The worker stops at its next chunk and nothing is kept."""
    job = jobs.get(job_id)
    if job is None:
        raise problem(404, "We could not find that upload - it may have finished "
                           "and been cleared.")
    jobs.cancel(job)
    return ok({"status": jobs.CANCELLED})


@router.post("/datasets/sample/{name}")
async def load_sample_dataset(name: str):
    path = samples.sample_path(name)
    if not path.exists():
        raise problem(404, f"We do not have a sample dataset called '{name}'.")
    try:
        rows, columns = await asyncio.to_thread(samples.load_sample, name)
    except Exception as exc:
        raise problem(400, f"We could not open the sample '{name}' "
                           f"({type(exc).__name__}).")
    if not columns:
        raise problem(400, f"The sample '{name}' could not be read.")
    profile = await asyncio.to_thread(ingest.profile, rows, columns, path.name)
    store.put_dataset(profile, rows)
    return ok(profile.model_dump())


@router.get("/datasets")
def list_datasets():
    """What is loaded right now - small enough to refresh the UI with.

    Just id, name, row count and column count: enough for the UI to show what it
    still has after a page reload, without shipping a whole profile per dataset.
    """
    return ok(store.list_loaded())


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    """The stored description of one dataset.

    The profile normally arrives on the `done` event of the reading stream. This
    is the way back to it when that stream was dropped, so a lost connection
    never means uploading the same file twice.
    """
    profile = store.get_profile(dataset_id)
    if profile is None:
        raise problem(404, "That dataset is no longer loaded — please upload it "
                           "again.")
    return ok(profile.model_dump())


@router.get("/datasets/{dataset_id}/preview")
def preview_dataset(dataset_id: str, n: int = 20):
    profile = store.get_profile(dataset_id)
    if profile is None:
        raise problem(404, "We could not find that data - it may have been cleared. "
                           "Please upload it again.")
    columns = [c.name for c in profile.columns]
    n = 20 if n is None else max(1, min(int(n), 2000))
    # Only the first `n` rows are read - a preview of a very large file must not
    # pull the whole table back into the server process.
    return ok({"columns": columns,
               "rows": rows_preview(store.preview_rows(dataset_id, n), columns, n)})


@router.get("/datasets/{dataset_id}/distinct")
def distinct_dataset_values(dataset_id: str, column: str, limit: int = 50):
    """Distinct values in one column.

    QA DF-12 / UX-15: lets a Step 4 dropdown offer the values that are actually
    in the user's file (e.g. a "Business Case" stage) instead of only the
    standard template.
    """
    profile = store.get_profile(dataset_id)
    if profile is None:
        raise problem(404, "We could not find that data - it may have been cleared. "
                           "Please upload it again.")
    known = {c.name for c in profile.columns}
    if column not in known:
        raise problem(404, f"There is no column called '{column}' in that file.")
    return ok({"column": column,
               "values": store.distinct_values(dataset_id, column, limit)})


@router.get("/samples")
def list_samples():
    try:
        items = samples.list_samples()
    except Exception as exc:
        raise problem(500, f"We could not prepare the sample files "
                           f"({type(exc).__name__}).")
    return ok([{
        "name": i.get("name"),
        "title": i.get("title"),
        "use_case_id": i.get("use_case_id"),
        "rows": i.get("rows"),
        "n_cols": i.get("n_cols"),
        "blurb": i.get("blurb", ""),
    } for i in items])


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------
@router.post("/usecases/{use_case_id}/automap")
def automap(use_case_id: str, body: dict):
    registry.discover()
    uc = registry.get(use_case_id)
    if uc is None:
        raise problem(404, f"We do not have a simulation called '{use_case_id}'.")

    payload = body if isinstance(body, dict) else {}
    dataset_id = payload.get("dataset_id")
    profile = store.get_profile(dataset_id) if dataset_id else None
    if profile is None:
        raise problem(404, "We could not find that data - please upload it again "
                           "before we can match its columns.")

    columns = [c.name for c in profile.columns]
    roles = list(uc.meta.input_roles or [])
    try:
        # `n_rows` is what lets the cardinality rule fire: a column with almost
        # one value per row is an identifier, and an identifier is not a group.
        mapping, confidence, notes = schema_map.auto_map(
            columns, profile.columns, roles, n_rows=profile.n_rows)
    except Exception as exc:
        raise problem(500, f"We could not match the columns ({type(exc).__name__}).")

    return ok({
        "mapping": {k: (v if v is not None else None) for k, v in mapping.items()},
        "confidence": confidence,
        "notes": notes,
    })


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
@router.post("/runs")
def create_run(body: dict):
    registry.discover()
    payload = body if isinstance(body, dict) else {}
    try:
        req = RunRequest.model_validate(payload)
    except Exception as exc:
        raise problem(400, "We could not understand that run request. "
                           f"({type(exc).__name__})")

    if registry.get(req.use_case_id) is None:
        raise problem(404, f"We do not have a simulation called '{req.use_case_id}'.")
    if req.dataset_id and not store.has_dataset(req.dataset_id):
        raise problem(400, "We could not find the data for this run - please upload "
                           "it again.")

    try:
        run_id = runs.start_run(req)
    except Exception as exc:
        raise problem(500, f"We could not start this run ({type(exc).__name__}).")
    return ok({"run_id": run_id})


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: str, request: Request):
    """Server-sent events: log lines while it runs, then one `result`."""
    if store.get_status(run_id) == "unknown":
        raise problem(404, "We could not find that run.")

    return StreamingResponse(
        _event_stream(run_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _event_stream(run_id: str, request: Request):
    listener = store.subscribe(run_id)
    started = time.perf_counter()
    last_sent = time.perf_counter()
    stream_limit = float(getattr(runs, "MAX_STREAM_SECONDS", 900.0) or 900.0)
    try:
        while True:
            try:
                event = listener.get_nowait()
            except Exception:
                event = None

            if event is None:
                if await _client_gone(request):
                    return
                if time.perf_counter() - started > stream_limit:
                    yield ":keepalive\n\n"
                    return
                if time.perf_counter() - last_sent >= 15.0:
                    last_sent = time.perf_counter()
                    yield ":keepalive\n\n"
                await asyncio.sleep(0.05)
                continue

            name = str(event.get("event") or "message")
            try:
                data = json.dumps(json_safe(event.get("data")))
            except Exception:
                data = "{}"
            last_sent = time.perf_counter()
            yield f"event: {name}\ndata: {data}\n\n"

            if name in ("result", "error"):
                return
    finally:
        store.unsubscribe(run_id, listener)


async def _client_gone(request: Request) -> bool:
    try:
        return bool(await request.is_disconnected())
    except Exception:
        return False


@router.get("/runs/{run_id}/result")
def run_result(run_id: str):
    status = store.get_status(run_id)
    if status == "unknown":
        raise problem(404, "We could not find that run.")

    result = store.get_result(run_id)
    if result is not None:
        return ok(result.model_dump())

    if status == store.STATUS_FAILED:
        err = store.get_error(run_id) or {}
        message = err.get("message") or "We could not finish this run."
        failed = _failed_result(run_id, message, err.get("trace", ""))
        return JSONResponse(status_code=200, content=json_safe(failed.model_dump()))

    return JSONResponse(status_code=202, content={"status": status or "running"})


def _failed_result(run_id: str, message: str, trace: str = "") -> RunResult:
    """A run that blew up still answers with a RunResult, so the front-end can
    render the same screen - just with a plain-English apology in it."""
    result = RunResult(run_id=run_id, use_case_id=store.use_case_of(run_id),
                       status="failed", headline=message)
    result.warnings = [message]
    result.assumptions = []
    result.log = runs.log_of(run_id) or [
        LogEvent(t=0.0, phase="summarize", level="error", message=message, detail=trace)
    ]
    return result


@router.get("/runs/{run_id}/log")
def run_log(run_id: str):
    if store.get_status(run_id) == "unknown":
        raise problem(404, "We could not find that run.")
    return ok([e.model_dump() for e in runs.log_of(run_id)])


@router.get("/runs/{run_id}/artifact/{fmt}")
def run_artifact(run_id: str, fmt: str):
    key = str(fmt or "").strip().lower()
    result = store.get_result(run_id)
    filename = None
    if result is not None:
        filename = (result.artifacts or {}).get(key)
    filename = filename or reporting.artifact_filename(key)

    path = store.run_dir(run_id) / filename
    if not path.exists():
        fallback = store.run_dir(run_id) / reporting.artifact_filename(key)
        if fallback.exists():
            path = fallback
        else:
            raise problem(404, f"We do not have a {key} file for this run yet.")

    return FileResponse(
        str(path),
        media_type=MEDIA_TYPES.get(key, "application/octet-stream"),
        filename=f"simulation-arena-{key}-{run_id}.{path.suffix.lstrip('.') or key}",
    )
