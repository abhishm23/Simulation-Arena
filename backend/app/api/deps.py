"""Small shared helpers for the HTTP layer.

Nothing here knows about business logic - it is only about getting Python
objects safely out to the browser (numpy scalars, sets, dates and friends do
not survive ``json.dumps`` on their own).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any

from ..core.contracts import DatasetProfile, RunRequest, RunResult


def _fallback(obj: Any) -> Any:
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _fallback(item())
        except Exception:
            pass
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def json_safe(obj: Any) -> Any:
    """Anything -> something `json.dumps` can write. Never raises."""
    try:
        return json.loads(json.dumps(obj, default=_fallback))
    except Exception:
        try:
            return json.loads(json.dumps(str(obj)))
        except Exception:
            return None


def rows_preview(rows: list[dict], columns: list[str], n: int) -> list[dict]:
    """First `n` rows, shaped to the given column order."""
    n = max(0, min(int(n or 0), 2000))
    out = []
    for r in list(rows or [])[:n]:
        out.append({c: json_safe(r.get(c, None)) for c in columns})
    return out


def as_profile(profile: DatasetProfile) -> dict:
    return json_safe(profile.model_dump())


def as_result(result: RunResult) -> dict:
    return json_safe(result.model_dump())


def as_request(req: RunRequest) -> RunRequest:
    return req


def plain_number(v: Any) -> Any:
    """Coerce a value the browser sent (often a string) into a number."""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
    s = str(v).strip().replace(",", "").replace("%", "").replace("$", "")
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return v
    return None if (math.isnan(f) or math.isinf(f)) else f
