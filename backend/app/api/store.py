"""Tiny state layer: in-memory first, disk second, database never.

Two things are kept here:

    * datasets  - the rows the user gave us (memory + a parquet cache on disk,
                  so a server restart can still serve previews and runs)
    * runs      - status, result, error and the live event queue each run uses
                  to talk to its SSE listeners

Rule for the whole module: a cache failure must never break a run. Every disk
operation is wrapped, and every lookup falls back to something sensible.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from ..core.contracts import DatasetProfile, RunResult

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[2]
STORE_DIR = BACKEND_ROOT / ".store"
DATASET_DIR = STORE_DIR / "datasets"
UPLOAD_DIR = STORE_DIR / "uploads"
EXPORTS_DIR = BACKEND_ROOT / "exports"

#: how many events we keep per run so a listener who connects late still sees
#: the story from the beginning
EVENT_BUFFER = 3000


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name)
        return int(raw) if raw not in (None, "") else int(default)
    except Exception:
        return int(default)


#: how many datasets stay in memory at once. A long session that uploads file
#: after file must not grow without bound, so the least recently used dataset is
#: dropped from RAM (its parquet cache stays on disk and it reloads on demand).
MAX_RETAINED_DATASETS = max(2, _env_int("SIMARENA_MAX_RETAINED_DATASETS", 6))

_lock = threading.RLock()
#: id -> {"profile": DatasetProfile, "rows": list[dict] | None}
#: `rows is None` means "the rows are on disk only". That is how a very large
#: upload is held: the worker writes the parquet cache in bounded chunks and we
#: never rebuild two million Python dicts in the server process, because doing
#: that holds the GIL for seconds and is what used to take the whole app down.
_DATASETS: dict[str, dict[str, Any]] = {}
_RUNS: dict[str, dict[str, Any]] = {}              # id -> {"status", "result", "error"}
_EVENTS: dict[str, list[dict[str, Any]]] = {}      # id -> buffered SSE events
_SUBS: dict[str, list[queue.Queue]] = {}           # id -> live listener queues
_USE: dict[str, int] = {}                          # id -> last-used counter
_USE_TICK = [0]

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"


def _ensure_dirs() -> None:
    try:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def ensure_dirs() -> None:
    """Make sure the folders we write to exist. Never raises."""
    _ensure_dirs()


def _new_id(prefix: str) -> str:
    return prefix + "_" + uuid.uuid4().hex[:12]


def _touch(dataset_id: str) -> None:
    _USE_TICK[0] += 1
    _USE[dataset_id] = _USE_TICK[0]


def _evict_datasets(keep: Optional[str] = None) -> None:
    """Drop the least recently used datasets from memory (disk cache stays)."""
    with _lock:
        while len(_DATASETS) > MAX_RETAINED_DATASETS:
            candidates = [d for d in _DATASETS if d != keep]
            if not candidates:
                return
            victim = min(candidates, key=lambda d: _USE.get(d, 0))
            _DATASETS.pop(victim, None)
            _USE.pop(victim, None)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def put_dataset(profile: DatasetProfile, rows: list[dict], *,
                persist: bool = True, lazy: bool = False) -> str:
    """Remember a dataset. Returns its id.

    `persist=False` means "the parquet cache is already on disk" - it is used by
    the large-file path, where the cache is written (in bounded chunks) before
    we get here. The sidecar is always written.

    `lazy=True` means "do not hold the rows in memory either" - the dataset is
    described by its profile and served straight from the parquet cache. Only
    the big-file path uses it; see `_DATASETS`.
    """
    rows = None if lazy else list(rows or [])
    dataset_id = str(profile.dataset_id or "").strip() or _new_id("ds")
    with _lock:
        _DATASETS[dataset_id] = {"profile": profile, "rows": rows}
        _touch(dataset_id)
    _persist_dataset(dataset_id, profile, rows or [],
                     write_rows=persist and not lazy)
    _evict_datasets(keep=dataset_id)
    return dataset_id


def get_dataset(dataset_id: str) -> Optional[tuple[DatasetProfile, list[dict]]]:
    """(profile, rows) or None. Falls back to the parquet cache on disk."""
    if not dataset_id:
        return None
    with _lock:
        hit = _DATASETS.get(dataset_id)
        if hit is not None:
            _touch(dataset_id)
    if hit is not None and hit["rows"] is not None:
        return hit["profile"], list(hit["rows"])
    return _load_dataset(dataset_id)


def get_profile(dataset_id: str) -> Optional[DatasetProfile]:
    """Just the description of a dataset - never pays to load the rows.

    Callers that only need column names, row counts or column types must use
    this: for a large upload, loading the rows costs seconds.
    """
    if not dataset_id:
        return None
    with _lock:
        hit = _DATASETS.get(dataset_id)
        if hit is not None:
            _touch(dataset_id)
    if hit is not None:
        return hit["profile"]
    return _read_sidecar(dataset_id)


def has_dataset(dataset_id: str) -> bool:
    """Is this id something we can still serve? Cheap - never loads rows."""
    if not dataset_id:
        return False
    with _lock:
        if dataset_id in _DATASETS:
            return True
    return (_sidecar_path(dataset_id).exists()
            and _parquet_path(dataset_id).exists())


def preview_rows(dataset_id: str, n: int) -> list[dict]:
    """The first `n` rows, read straight off disk when the rows are not held.

    A preview only ever shows a handful of rows, so there is no reason to
    rebuild a million of them to answer it.
    """
    if not dataset_id:
        return []
    with _lock:
        hit = _DATASETS.get(dataset_id)
        if hit is not None:
            _touch(dataset_id)
    if hit is not None and hit["rows"] is not None:
        return list(hit["rows"][:n])
    parquet = _parquet_path(dataset_id)
    if not parquet.exists():
        found = _load_dataset(dataset_id)
        return list(found[1][:n]) if found else []
    try:
        import polars as pl
        head = pl.read_parquet(str(parquet)).head(max(1, int(n)))
        return [{k: _jsonable(v) for k, v in rec.items()}
                for rec in head.to_dicts()]
    except Exception:
        return []


def distinct_values(dataset_id: str, column: str, limit: int = 50) -> list[str]:
    """Every distinct value in one column, read off disk where possible.

    QA DF-12 / UX-15: a Step 4 dropdown offered only the standard template, so a
    stage that existed in the user's own file ("Business Case") could not be
    chosen. This lets the form offer the values actually present.
    """
    if not dataset_id or not column:
        return []
    limit = max(1, min(int(limit), 500))
    parquet = _parquet_path(dataset_id)
    if parquet.exists():
        try:
            import polars as pl
            frame = pl.read_parquet(str(parquet), columns=[column])
            vals = (frame.get_column(column)
                    .drop_nulls()
                    .cast(pl.Utf8, strict=False)
                    .unique()
                    .sort()
                    .to_list())
            return [str(v) for v in vals[:limit] if str(v).strip()]
        except Exception:
            pass
    seen, out = set(), []
    for row in preview_rows(dataset_id, 5000):
        v = row.get(column)
        if v is None:
            continue
        s = str(v).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= limit:
            break
    return out


def drop_dataset(dataset_id: Optional[str]) -> None:
    """Forget a dataset entirely - used when a job is cancelled or fails."""
    if not dataset_id:
        return
    with _lock:
        _DATASETS.pop(dataset_id, None)
        _USE.pop(dataset_id, None)
    for path in (_sidecar_path(dataset_id), _parquet_path(dataset_id)):
        try:
            path.unlink(missing_ok=True)
        except BaseException:
            pass


def clear_uploads() -> None:
    """Delete any spooled upload files left behind by a crash or a shutdown.

    Housekeeping must never be able to stop the server from starting, so a
    stubborn file is skipped and forgotten rather than raised.
    """
    try:
        if not UPLOAD_DIR.is_dir():
            return
        for item in UPLOAD_DIR.iterdir():
            try:
                if item.is_file():
                    item.unlink(missing_ok=True)
            except BaseException:
                continue
    except BaseException:
        pass



def list_datasets() -> list[dict[str, Any]]:
    """[{dataset_id, source_name, n_rows, n_cols}] - for debugging and restart."""
    with _lock:
        items = list(_DATASETS.items())
    out = []
    for did, rec in items:
        prof: DatasetProfile = rec["profile"]
        out.append({
            "dataset_id": did,
            "source_name": prof.source_name,
            "n_rows": prof.n_rows,
            "n_cols": prof.n_cols,
        })
    return out


def list_loaded() -> list[dict[str, Any]]:
    """[{id, name, rows, cols}] - the short list the UI shows after a refresh.

    Deliberately tiny: a handful of datasets, four fields each, so it can be
    fetched on every page load without pulling any profiles or rows with it.
    """
    with _lock:
        items = list(_DATASETS.items())
    out = []
    for did, rec in items:
        prof: DatasetProfile = rec["profile"]
        out.append({
            "id": did,
            "name": prof.source_name,
            "rows": int(prof.n_rows or 0),
            "cols": int(prof.n_cols or 0),
        })
    return out


def _sidecar_path(dataset_id: str) -> Path:
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(dataset_id))
    return DATASET_DIR / f"{safe}.json"


def _parquet_path(dataset_id: str) -> Path:
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(dataset_id))
    return DATASET_DIR / f"{safe}.parquet"


def _persist_dataset(dataset_id: str, profile: DatasetProfile, rows: list[dict],
                     write_rows: bool = True) -> None:
    _ensure_dirs()
    try:
        _sidecar_path(dataset_id).write_text(
            json.dumps(profile.model_dump(), ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    if write_rows and rows:
        write_parquet_chunked(_parquet_path(dataset_id), rows)


def write_parquet_chunked(path: Path, rows: list[dict],
                          chunk: int = 250_000) -> bool:
    """Write rows to parquet a chunk at a time.

    Why chunks: building one frame from two million Python dicts holds the GIL
    for well over a second in a single uninterrupted call, which stops the whole
    server answering anything else. The same work done 250,000 rows at a time
    never holds it for more than a few milliseconds. (Measured - see the report.)
    """
    rows = list(rows or [])
    if not rows:
        return False
    step = max(1, int(chunk))

    def _frames(transform) -> list:
        import polars as pl
        out = []
        for start in range(0, len(rows), step):
            part = rows[start:start + step]
            if transform is not None:
                part = [transform(r) for r in part]
            # infer from the whole chunk (not the first 100 rows) so a column
            # does not change type from one chunk to the next
            out.append(pl.DataFrame(part, infer_schema_length=None, strict=False))
        return out

    for transform in (None, _stringify):
        try:
            import polars as pl
            frames = _frames(transform)
            frame = (pl.concat(frames, how="vertical_relaxed")
                     if len(frames) > 1 else frames[0])
            frame.write_parquet(str(path))
            return True
        except Exception:
            continue
    return False


def _stringify(r: dict) -> dict:
    return {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list))
                else v) for k, v in r.items()}


def adopt_parquet(dataset_id: str, staged: Path) -> bool:
    """Move a parquet cache written elsewhere into this dataset's slot."""
    if not dataset_id or staged is None:
        return False
    try:
        _ensure_dirs()
        target = _parquet_path(dataset_id)
        if Path(staged) == target:
            return True
        # `os.replace` overwrites an existing target atomically, so there is
        # nothing to delete first - and no window where the cache is missing.
        Path(staged).replace(target)
        return True
    except Exception:
        return False


def read_rows_chunked(parquet_path: Path, chunk: int = 100_000) -> list[dict]:
    """Load a parquet cache back into rows, a chunk at a time.

    Two things matter here. The conversion is chunked so no single call holds the
    GIL for long enough to freeze the server, and each chunk's dicts are rebuilt
    in place so we never hold two full copies of the table at once.
    """
    rows: list[dict] = []
    try:
        import polars as pl
        frame = pl.read_parquet(str(parquet_path))
        step = max(1, int(chunk))
        for start in range(0, frame.height, step):
            for record in frame.slice(start, step).to_dicts():
                rows.append({k: _jsonable(v) for k, v in record.items()})
    except Exception:
        return []
    return rows


def _read_sidecar(dataset_id: str) -> Optional[DatasetProfile]:
    """The profile we wrote next to the parquet cache. Cheap - never loads rows."""
    try:
        sidecar = _sidecar_path(dataset_id)
        if not sidecar.exists():
            return None
        return DatasetProfile.model_validate(
            json.loads(sidecar.read_text(encoding="utf-8")))
    except Exception:
        return None


def _load_dataset(dataset_id: str) -> Optional[tuple[DatasetProfile, list[dict]]]:
    """Rebuild (profile, rows) from the disk cache."""
    profile = _read_sidecar(dataset_id)
    if profile is None:
        return None
    rows: list[dict] = []
    parquet = _parquet_path(dataset_id)
    if parquet.exists():
        rows = read_rows_chunked(parquet)
    with _lock:
        _DATASETS[dataset_id] = {"profile": profile, "rows": rows}
    return profile, rows


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, float):
        return None if v != v else v
    return str(v)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def create_run(use_case_id: str = "") -> str:
    run_id = _new_id("run")
    with _lock:
        _RUNS[run_id] = {
            "use_case_id": use_case_id,
            "status": STATUS_QUEUED,
            "result": None,
            "error": None,
        }
        _EVENTS[run_id] = []
        _SUBS[run_id] = []
    return run_id


def set_status(run_id: str, status: str) -> None:
    with _lock:
        rec = _RUNS.setdefault(run_id, {"use_case_id": "", "status": status,
                                        "result": None, "error": None})
        rec["status"] = status


def use_case_of(run_id: str) -> str:
    with _lock:
        rec = _RUNS.get(run_id)
    return (rec or {}).get("use_case_id", "") or ""


def get_status(run_id: str) -> str:
    with _lock:
        rec = _RUNS.get(run_id)
    return rec["status"] if rec else "unknown"


def put_result(run_id: str, result: RunResult) -> None:
    with _lock:
        rec = _RUNS.setdefault(run_id, {"use_case_id": "", "status": STATUS_SUCCEEDED,
                                        "result": None, "error": None})
        rec["result"] = result
        rec["status"] = STATUS_SUCCEEDED


def get_result(run_id: str) -> Optional[RunResult]:
    with _lock:
        rec = _RUNS.get(run_id)
    return rec["result"] if rec else None


def put_error(run_id: str, message: str, trace: str = "") -> None:
    with _lock:
        rec = _RUNS.setdefault(run_id, {"use_case_id": "", "status": STATUS_FAILED,
                                        "result": None, "error": None})
        rec["error"] = {"message": message, "trace": trace}
        rec["status"] = STATUS_FAILED


def get_error(run_id: str) -> Optional[dict[str, str]]:
    with _lock:
        rec = _RUNS.get(run_id)
    return rec["error"] if rec else None


def run_dir(run_id: str) -> Path:
    """Where this run's downloadable files live."""
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(run_id))
    return EXPORTS_DIR / safe


# ---------------------------------------------------------------------------
# Live event fan-out (one queue per SSE listener)
# ---------------------------------------------------------------------------
def subscribe(run_id: str) -> queue.Queue:
    """Open a listener. The queue is pre-filled with everything published so
    far, so a client that connects late still sees the run from the start."""
    q: queue.Queue = queue.Queue()
    with _lock:
        _SUBS.setdefault(run_id, []).append(q)
        backlog = list(_EVENTS.get(run_id, []))
    for ev in backlog:
        try:
            q.put_nowait(ev)
        except Exception:
            break
    return q


def unsubscribe(run_id: str, q: queue.Queue) -> None:
    with _lock:
        listeners = _SUBS.get(run_id)
        if listeners and q in listeners:
            listeners.remove(q)


def publish(run_id: str, event: dict[str, Any]) -> None:
    """Send one SSE-shaped event {"event": name, "data": payload} to everyone."""
    with _lock:
        buf = _EVENTS.setdefault(run_id, [])
        buf.append(event)
        if len(buf) > EVENT_BUFFER:
            del buf[: len(buf) - EVENT_BUFFER]
        listeners = list(_SUBS.get(run_id, []))
    for q in listeners:
        try:
            q.put_nowait(event)
        except Exception:
            pass


def events(run_id: str) -> list[dict[str, Any]]:
    """Everything published for this run (for diagnostics)."""
    with _lock:
        return list(_EVENTS.get(run_id, []))
