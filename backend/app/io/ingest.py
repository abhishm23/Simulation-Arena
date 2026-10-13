"""Data ingestion: read whatever file the user brings, and describe it in plain English.

The reader is deliberately paranoid. A finance user will upload a file that was
hand-edited, exported from three systems, saved twice and emailed. So:

    * every reader is tried in order (polars -> pandas -> stdlib csv),
    * nothing ever raises out of `read_table` or `profile`,
    * bad cells become None and are reported as warnings, not exceptions.

It is also deliberately fast. Files arrive at 245 MB - 1.4 GB and the old
row-by-row Python walk cost 1.65-9.08 ms per row (988 seconds for 200,000
rows). Nothing here walks rows in Python any more:

    * delimited files are parsed in bounded chunks by polars' batched reader,
    * parquet is read batch-by-batch from the file footer,
    * spreadsheets stream through openpyxl in read-only mode,
    * the profile is one pass of vectorised polars expressions,
    * `list[dict]` is built once, at the very end, for the rows we keep.

Public API
----------
    IngestLimits                            size / row / time limits
    IngestRejected                          plain-English refusal (.message)
    IngestCancelled                         the caller asked us to stop
    probe(source)                           cheap pre-flight, reads no data
    read_table(source, name, narrator=None, progress=None, cancel=None,
               limits=None)                 -> (rows, column_names)
    profile(rows, columns, name, *,
            sampled=False, sample_note="")  -> DatasetProfile
    coerce_value(v, dtype)                  -> cleaned value (or None)
    infer_dtype(series)                     -> numeric|categorical|datetime|boolean|identifier
    sanitise_headers(cols)                  -> safe, unique column names

`rows` is a list of plain dicts whose values are JSON-safe primitives
(str / float / int / bool / None). Values keep their original *text* (for example
"$1,200,000" stays as text here) - turning text into numbers is the job of
`schema_map.apply_mapping`, which knows what each column means.

The list returned by `read_table` is a `_RowList` - a real `list` subclass that
also carries `sampled`, `sample_note`, `n_rows_in_file` and `columns`. It behaves
exactly like the plain list it always was; `profile()` reads those attributes so
the "we ran on a sample" warning reaches the user without any caller changes.
"""
from __future__ import annotations

import csv as _csv
import datetime as _dt
import hashlib
import io
import json
import math
import os
import re
import time
import warnings as _warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import polars as pl

from ..core.contracts import ColumnProfile, DatasetProfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOM = "\ufeff"

#: text that means "nothing here"
NULL_TOKENS = {
    "", "-", "--", "na", "n/a", "#n/a", "#na", "nan", "none", "null", "nil",
    "nat", "<na>", "?", "#value!", "#div/0!",
}

_TRUE_WORDS = {"yes", "y", "true", "t", "won", "win", "1", "1.0", "yep", "yeah"}
_FALSE_WORDS = {"no", "n", "false", "f", "lost", "loss", "0", "0.0", "nope"}

_DATE_FORMATS = (
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
    "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%d %B %Y", "%b %d %Y",
    "%d-%b-%y", "%d/%m/%y", "%m/%d/%y", "%Y%m%d", "%Y-%m", "%b-%Y", "%B %Y",
    "%b %Y", "%B-%Y", "%m/%Y", "%m-%Y",
)

_MANGLED = (
    re.compile(r"^(?P<base>.+?)_duplicated_\d+$"),   # polars
    re.compile(r"^(?P<base>.+?)\.\d+$"),             # pandas
)

#: characters we trim off the ends of every text cell (BOM + ordinary whitespace)
_BOM_CHARS = "\ufeff \t\r\n\v\f"
#: the same null words, as a list, for polars' `is_in`
_NULL_LIST = sorted(NULL_TOKENS)
#: fixed so two sessions sampling the same file keep the same rows
_SAMPLE_SEED = 20260101
#: how often `progress` is allowed to fire
_PROGRESS_INTERVAL_S = 0.25
#: the rolling window the rows-per-second figure is measured over
_RATE_WINDOW_S = 3.0
#: how often a plain-English line goes into the run log while reading
_NARRATE_INTERVAL_S = 10.0
#: rows per polars batch. Small enough that a cancel is noticed promptly and
#: progress fires well inside a quarter of a second.
_READ_BATCH = 32_768
#: QA NEW-14: how many kept chunks we hold before merging them into one frame.
#: A kept chunk is a slice of the block it was read from, so holding one chunk
#: per block pins the whole file; merging every this many forces a fresh copy.
_COMPACT_EVERY = 64
#: QA NEW-14: how much of a delimited file the streaming reader holds at once.
#: Measured on a 2.38 GB / 16.2M-row export: polars' batched CSV reader peaked
#: at 2.54 GB (the whole file resident) however small the batch was, while
#: pyarrow's streaming reader with a small block stays flat. 2 MB keeps the read
#: within a few seconds of the fast reader without letting the file size decide
#: how much memory we use (measured committed memory after the read: 1537 MB at
#: 8 MB, 971 MB at 2 MB, 878 MB at 1 MB).
_ARROW_BLOCK_BYTES = 2 * 1024 * 1024
#: block size for the cheap newline count
_COUNT_BLOCK = 8 * 1024 * 1024
#: only bother telling the user we are sizing the file when it is worth knowing
_SIZING_NOTICE_BYTES = 32 * 1024 * 1024
#: a column is worth stratifying on only if it is small enough to be a category
_STRATA_MAX_CARDINALITY = 200
_STRATA_MAX_SHARE = 0.25
#: how many chunks we look at while hunting for a category column
_STRATA_TRIES = 4

# ---------------------------------------------------------------------------
# Whole-record sampling (QA NEW-10 / TC-02)
# ---------------------------------------------------------------------------
#: The old sampler took every Nth *line*. On a file whose rows are order lines
#: grouped by quote, that splits deals down the middle: half a deal's lines are
#: kept and the other half thrown away, so the deal-level dependency the use
#: case exists to measure is destroyed. Sampling has to move in whole business
#: records, so we look for the column that *groups* the rows and take every row
#: of the records whose turn it is.
#:
#: A record key looks like this and nothing else does:
#:   * it repeats on adjacent rows (the file is grouped by it), and
#:   * it holds roughly as many distinct values as it has runs - i.e. it is an
#:     identifier, not a category such as a tier or a region.
#: How many chunks we look at before giving up on finding one.
_RECORD_TRIES = 3
#: at least this share of adjacent row pairs must carry the same value
_RECORD_MIN_RUN_SHARE = 0.25
#: ... and there must be at least this many distinct runs to judge by
_RECORD_MIN_RUNS = 16
#: ... and this many distinct values per run, or it is a category not a key
_RECORD_UNIQ_PER_RUN = 0.8

_warnings.filterwarnings(
    "ignore", message=r".*read_csv_batched is deprecated.*", category=DeprecationWarning
)


# ---------------------------------------------------------------------------
# Limits and refusals
# ---------------------------------------------------------------------------
@dataclass
class IngestLimits:
    """The ceilings that keep one big upload from taking the product down."""

    #: QA NEW-14: the product owner's requirement is gigabyte-scale data - a
    #: 4-5 GB export with tens of millions of rows. The old 2 GiB ceiling
    #: refused exactly the files the tool exists for. The reader streams, so
    #: the size of the file is not what costs memory: what costs memory is the
    #: working set below. 6 GiB leaves headroom over "4-5 GB" without letting a
    #: mistyped file eat the machine's disk.
    max_file_bytes: int = 6 * 1024**3        # 6 GiB
    #: The working set. A use case works on `list[dict]`, and one row of ~17
    #: columns costs about half a kilobyte of Python objects, so 250k rows is
    #: roughly 120 MB for the rows the run holds - plus the mapped copy. This
    #: number is the *only* row cap in the product: what is read is what the
    #: run uses, so there is exactly one reduction to disclose (QA NEW-04 /
    #: NEW-05). At 2,000,000 the run path held ~1 GB of dicts in the server
    #: process on top of the cache and the export buffers, which is how a
    #: single upload reached 7.5 GB RSS.
    max_working_rows: int = 250_000
    parse_timeout_s: float = 900.0
    chunk_rows: int = 250_000


class IngestRejected(Exception):
    """Raised with a plain-English .message when a file cannot be accepted."""

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = str(message)


class IngestCancelled(Exception):
    """Raised when the caller's cancel() callback returns True."""

    def __init__(self, message: str = "You cancelled this upload."):
        super().__init__(message)
        self.message = str(message)


class _RowList(list):
    """A plain list of rows that also remembers how the file was read.

    It *is* a list - same shape, same behaviour, same JSON - so every existing
    caller keeps working. The extra attributes are what let `profile()` tell the
    user that we ran on a sample without changing the return signature.
    """

    sampled: bool = False
    sample_note: str = ""
    n_rows_in_file: int = 0
    columns: list = []          # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _say(narrator: Any, message: str, **kw) -> None:
    """Emit a plain-language line if the caller gave us a narrator."""
    if narrator is None:
        return
    try:
        if hasattr(narrator, "say"):
            narrator.say(message, **kw)
        elif callable(narrator):
            narrator(message)
    except Exception:
        pass


def _is_null(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        return s == "" or s in NULL_TOKENS
    return False


def _json_safe(v: Any) -> Any:
    """Make a value safe to put in JSON (and safe to show in the UI)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else float(v)
    if isinstance(v, str):
        s = v.replace(BOM, "").strip()
        if s.lower() in NULL_TOKENS:
            return None
        return s
    if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", "replace")
        except Exception:
            return None
    item = getattr(v, "item", None)          # numpy / pyarrow scalars
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return str(v)


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


def _rows_word(n: Any) -> str:
    """'1 row', '2 rows' - the message is read by people, not machines."""
    try:
        return "row" if int(n) == 1 else "rows"
    except Exception:
        return "rows"


def _human_duration(seconds: Any) -> str:
    """'about 12 seconds left' - never a bare number, never 'inf'."""
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
        return (f"{minutes} minute{'s' if minutes != 1 else ''} "
                f"{secs} second{'s' if secs != 1 else ''}")
    hours = minutes // 60
    minutes = minutes % 60
    if minutes == 0:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return (f"{hours} hour{'s' if hours != 1 else ''} "
            f"{minutes} minute{'s' if minutes != 1 else ''}")


def _too_big_message(size: int, limit: int) -> str:
    return (f"That file is {_human_bytes(size)}, and the limit is "
            f"{_human_bytes(limit)}. Save it as Parquet or gzip the CSV and it "
            f"will be far smaller — then try again.")


def _timeout_message(name: str, seconds: float) -> str:
    return (f"We stopped reading '{name}' after {_human_duration(seconds)} because "
            f"it was taking too long. Try saving it as Parquet, or splitting it "
            f"into smaller files, then upload again.")


def _sample_note(n_seen: int, kept: int, record_key: str = "") -> str:
    """One sentence naming the file's size and the size we actually used.

    QA NEW-05: the old wording only reached the upload panel, and the run log
    then reported the *sample* as "Found N rows", so the user had no way to know
    half their file had been left out. This sentence goes into the run log and
    into the result as well, and it names both numbers.

    QA NEW-10 / TC-02: it also names *what* was left out. When the rows are
    grouped into business records we keep whole records, and the note says so
    and names the column that told us how to group them - a reader has to be
    able to tell "a sample of lines" from "a sample of whole deals".
    """
    share = (100.0 * kept / n_seen) if n_seen else 100.0
    unit = (f"whole records (grouped by your '{record_key}' column)"
            if record_key else "rows")
    return (f"Your file holds {_human_count(n_seen)} rows. We are running on an "
            f"evenly spread sample of {_human_count(kept)} of them "
            f"({share:.1f}%), keeping {unit} intact rather than splitting a "
            f"record down the middle, because a working set that big does not fit "
            f"in memory while keeping the tool responsive. Every number below comes "
            f"from those {_human_count(kept)} rows.")


# ---------------------------------------------------------------------------
# Header cleaning
# ---------------------------------------------------------------------------
def sanitise_headers(cols: Sequence[Any]) -> list[str]:
    """Turn whatever the file called its columns into safe, unique names.

    Fixes: BOM, leading/trailing spaces, doubled-up spaces, blank header cells
    (-> `unnamed_3`) and duplicate names (-> `col`, `col__2`, `col__3`).
    """
    out: list[str] = []
    seen: dict[str, int] = {}
    for i, raw in enumerate(cols or []):
        s = str(raw if raw is not None else "").replace(BOM, "")
        s = re.sub(r"\s+", " ", s).strip()
        for pat in _MANGLED:                      # undo a reader's own renaming
            m = pat.match(s)
            if m and m.group("base").strip():
                s = m.group("base").strip()
                break
        if not s or s.lower() in ("unnamed", "unnamed:", "none", "null"):
            s = f"unnamed_{i + 1}"
        key = s.lower()
        if key in seen:
            n = seen[key] + 1
            base = s
            while f"{base}__{n}".lower() in seen:
                n += 1
            seen[key] = n
            s = f"{base}__{n}"
            seen[s.lower()] = 1
        else:
            seen[key] = 1
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------
def _clean_number_text(s: str) -> Optional[float]:
    """'$1,200,000' -> 1200000.0 ; '42%' -> 42.0 ; '(350)' -> -350.0"""
    s = str(s).replace(BOM, "").strip()
    if not s or s.lower() in NULL_TOKENS:
        return None
    s = s.replace("\u00a0", " ").replace("\u2212", "-")
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1].strip()
    # European thousands: 1.200.000,50
    if re.fullmatch(r"[+-]?\d{1,3}(\.\d{3})+,\d+", s):
        s = s.replace(".", "").replace(",", ".")
    # US thousands: 1,200,000.50
    elif re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+\.\d+", s):
        s = s.replace(",", "")
    # trailing minus: 350-
    if s.endswith("-"):
        negative, s = True, s[:-1]
    s = re.sub(r"(?i)^(usd|eur|gbp|inr|rs|rm|\$|£|€|¥)", "", s).strip()
    s = re.sub(r"[,\s\u00a0%$£€¥]", "", s)
    s = re.sub(r"(?i)(usd|eur|gbp|percent|pct)$", "", s)
    if s in ("", "-", "+", ".", ","):
        return None
    # must look like a number (a trailing unit such as 'kg' or '%' is allowed);
    # this is what stops '2024-03-05' being read as the year 2024
    if not re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?[a-zA-Z]?", s):
        return None
    try:
        val = float(s.rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return -val if negative else val


def _parse_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if v in (0, 1):
            return bool(v)
        return None
    s = str(v).strip().lower()
    if s in _TRUE_WORDS:
        return True
    if s in _FALSE_WORDS:
        return False
    return None


def _parse_datetime(v: Any) -> Optional[_dt.datetime]:
    if isinstance(v, _dt.datetime):
        return v
    if isinstance(v, _dt.date):
        return _dt.datetime(v.year, v.month, v.day)
    s = str(v).replace(BOM, "").strip()
    if not s or s.lower() in NULL_TOKENS:
        return None
    try:                                        # ISO first (fast path)
        return _dt.datetime.fromisoformat(s.replace("Z", "+00:00").replace("z", "+00:00"))
    except Exception:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return _dt.datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def coerce_value(v: Any, dtype: str) -> Any:
    """Force one cell into the dtype the use case asked for. Never raises.

    dtype in numeric | categorical | datetime | boolean | identifier.
    Anything unreadable becomes None - the caller decides what to do about it.

    Numeric note: '40' and '0.40' are both accepted as-is. Deciding whether a
    number is a fraction or a percentage is the caller's job, not ours.
    """
    dtype = (dtype or "categorical").lower()
    if _is_null(v):
        return None
    if dtype == "numeric":
        b = _parse_bool(v)
        if b is not None and not isinstance(v, (int, float)):
            return 1.0 if b else 0.0            # "yes"/"won" -> 1
        return _clean_number_text(str(v)) if isinstance(v, str) else _clean_number_text(v)
    if dtype == "boolean":
        return _parse_bool(v)
    if dtype == "datetime":
        d = _parse_datetime(v)
        if d is None:
            return None
        if (d.hour, d.minute, d.second) == (0, 0, 0):
            return d.date().isoformat()
        return d.isoformat(sep=" ")
    if dtype == "identifier":
        s = str(v).replace(BOM, "").strip()
        return s or None
    # categorical
    s = str(v).replace(BOM, "").strip()
    return s or None


def infer_dtype(series: Sequence[Any]) -> str:
    """Guess what a column holds, from its actual contents."""
    vals = [v for v in (series or []) if not _is_null(v)]
    n = len(vals)
    if n == 0:
        return "categorical"
    if all(isinstance(v, bool) for v in vals):
        return "boolean"
    numeric = sum(1 for v in vals if _clean_number_text(v) is not None)
    if numeric / n >= 0.90:
        return "numeric"
    dates = sum(1 for v in vals if _parse_datetime(v) is not None)
    if dates / n >= 0.80:
        return "datetime"
    words = {str(v).strip().lower() for v in vals if isinstance(v, str)}
    if words and words <= (_TRUE_WORDS | _FALSE_WORDS):
        return "boolean"
    uniq = len({str(v) for v in vals})
    if n >= 8 and uniq / n >= 0.98:
        return "identifier"
    return "categorical"


# ---------------------------------------------------------------------------
# Vectorised equivalents of the cleaning above
#
# These are the same rules, expressed as polars expressions so that a million
# rows cost one pass instead of a million Python calls. `profile()` uses them;
# the scalar functions above stay exactly as they were for callers that want
# to coerce one value at a time.
# ---------------------------------------------------------------------------
def _text_expr(e: pl.Expr) -> pl.Expr:
    """Text cell -> trimmed text, or null when it means 'nothing here'."""
    s = e.cast(pl.Utf8, strict=False).str.strip_chars(_BOM_CHARS)
    return (pl.when(s.is_null() | s.str.to_lowercase().is_in(_NULL_LIST))
            .then(None).otherwise(s))


def _num_expr(e: pl.Expr) -> pl.Expr:
    """Vectorised twin of `_clean_number_text`: text -> float (or null)."""
    s = e.cast(pl.Utf8, strict=False).str.strip_chars()
    neg = s.str.starts_with("(") & s.str.ends_with(")")
    s = (pl.when(neg)
         .then(s.str.slice(1, s.str.len_chars() - 2).str.strip_chars())
         .otherwise(s))
    eu = s.str.contains(r"^[+-]?\d{1,3}(\.\d{3})+,\d+$")
    us = s.str.contains(r"^[+-]?\d{1,3}(,\d{3})+\.\d+$")
    s = (pl.when(eu).then(s.str.replace_all(r"\.", "").str.replace(",", "."))
         .when(us).then(s.str.replace_all(",", ""))
         .otherwise(s))
    tail = s.str.ends_with("-")
    s = pl.when(tail).then(s.str.slice(0, s.str.len_chars() - 1)).otherwise(s)
    s = s.str.replace(r"(?i)^(usd|eur|gbp|inr|rs|rm|\$|£|€|¥)", "")
    s = s.str.replace_all(r"[,\s\u00a0%$£€¥]", "")
    s = s.str.replace(r"(?i)(usd|eur|gbp|percent|pct)$", "")
    ok = s.str.contains(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?[a-zA-Z]?$")
    s = pl.when(ok).then(s.str.replace(r"[a-zA-Z]$", "")).otherwise(None)
    val = s.cast(pl.Float64, strict=False)
    val = pl.when(val.is_finite()).then(val).otherwise(None)
    return pl.when(neg | tail).then(-val).otherwise(val)


#: the same date shapes as `_DATE_FORMATS`, spelled for polars' parser. Each one
#: is tried on its own so that a single format polars dislikes cannot take the
#: whole test down with it.
_POLARS_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M:%SZ",
    "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
    "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%d %B %Y", "%b %d %Y",
    "%d-%b-%y", "%d/%m/%y", "%m/%d/%y", "%Y%m%d", "%Y-%m", "%b-%Y", "%B %Y",
    "%b %Y", "%B-%Y", "%m/%Y", "%m-%Y",
)

#: how many values a column's dtype decision is based on. Every sample dataset
#: and every hand-built file is far below this, so their answer is exact; above
#: it, a 20,000-value spread is statistically the same answer at a fraction of
#: the cost.
_INFER_SAMPLE = 20_000


def _sample_series(s: pl.Series, k: int) -> pl.Series:
    """Up to `k` non-null values taken evenly across the whole column."""
    try:
        present = s.drop_nulls()
    except Exception:
        present = s
    n = present.len()
    if n == 0 or n <= k:
        return present
    step = max(1, n // k)
    try:
        return present.gather(list(range(0, n, step))[:k])
    except Exception:
        return present.head(k)


def _spread_sample(s: pl.Series, k: int) -> list:
    """The same thing as a plain list, with 'nothing here' values dropped."""
    try:
        return [v for v in _sample_series(s, k).to_list() if not _is_null(v)]
    except Exception:
        return []


def _date_frac_of(values: pl.Series) -> float:
    """Fraction of values that read as a date in any format we accept."""
    n = values.len()
    if n == 0:
        return 0.0
    try:
        df = values.cast(pl.Utf8, strict=False).str.strip_chars(_BOM_CHARS).to_frame()
    except Exception:
        return 0.0
    col = df.columns[0]
    acc: Optional[pl.Series] = None
    for fmt in _POLARS_DATE_FORMATS:
        try:
            hit = df.select(pl.col(col).str.to_datetime(format=fmt, strict=False)
                            .is_not_null().alias("m")).get_column("m")
        except Exception:
            continue
        acc = hit if acc is None else (acc | hit)
    if acc is None:
        return 0.0
    try:
        return float(acc.sum() or 0) / float(n)
    except Exception:
        return 0.0


def _numeric_frac_of(values: pl.Series) -> float:
    """Fraction of values that `_clean_number_text` would read as a number."""
    n = values.len()
    if n == 0:
        return 0.0
    try:
        df = values.cast(pl.Utf8, strict=False).to_frame()
        got = df.select(_num_expr(pl.col(df.columns[0])).is_not_null().sum()
                        .alias("n")).item() or 0
        return float(got) / float(n)
    except Exception:
        hits = sum(1 for v in values.to_list() if _clean_number_text(v) is not None)
        return float(hits) / float(n)


def _numeric_values(s: pl.Series) -> pl.Series:
    """The numeric value of every cell, matching `_clean_number_text` exactly.

    Plain numbers - which is nearly everything, nearly always - go through
    polars' own float parser in one pass. Only the handful of cells that fails
    on (money with thousands separators, brackets, percentages, trailing units)
    pay for the full cleaning chain, and those are the ones it exists for.
    """
    try:
        text = s.cast(pl.Utf8, strict=False).str.strip_chars(_BOM_CHARS)
        fast = text.cast(pl.Float64, strict=False)
    except Exception:
        return pl.Series("v", [], dtype=pl.Float64)
    ok = fast.is_not_null() & fast.is_finite()
    try:
        if ok.all():
            return fast
    except Exception:
        pass
    odd = ~ok
    try:
        rest = text.filter(odd)
    except Exception:
        return fast
    if rest.len() == 0:
        return fast
    try:
        fixed = rest.to_frame().select(_num_expr(pl.col(rest.name))).get_column(rest.name)
        return fast.scatter(odd.arg_true(), fixed)
    except Exception:
        return fast


def _infer_dtype_vec(df: pl.DataFrame, col: str, n_present: int, uniq: int,
                     sample: pl.Series) -> str:
    """Vectorised twin of `infer_dtype`, same rules, same order.

    The numeric and date tests run over a spread of up to 20,000 values rather
    than every row: the thresholds (90% numeric, 80% dates) do not move at that
    sample size, and a million-row column no longer costs a million regexes.
    """
    s = df.get_column(col)
    if s.dtype == pl.Boolean:
        return "boolean"
    if s.dtype in (pl.Date, pl.Datetime):
        return "datetime"
    if n_present <= 0:
        return "categorical"
    k = sample.len()
    if k == 0:
        return "categorical"

    if _numeric_frac_of(sample) >= 0.90:
        return "numeric"

    # dates are expensive to test for, so only try when a few values look like
    # dates - a column of customer names never does.
    try:
        head = [v for v in sample.head(400).to_list() if not _is_null(v)]
    except Exception:
        head = []
    if head:
        hits = sum(1 for v in head if _parse_datetime(v) is not None)
        if hits and _date_frac_of(sample) >= 0.80:
            return "datetime"

    if 0 < uniq <= 20:
        try:
            words = {str(v).strip().lower() for v in sample.unique().to_list()
                     if not _is_null(v)}
        except Exception:
            words = set()
        if words and words <= (_TRUE_WORDS | _FALSE_WORDS):
            return "boolean"
    if n_present >= 8 and uniq / float(n_present) >= 0.98:
        return "identifier"
    return "categorical"


def _clean_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the JSON-safe rules to a whole chunk at once.

    Text is trimmed and 'nothing here' words become null; timestamps become the
    same ISO text `_json_safe` would have produced; numbers keep their type.
    """
    exprs = []
    for c in df.columns:
        try:
            dtype = df.schema[c]
        except Exception:
            dtype = pl.Utf8
        if dtype == pl.Boolean:
            continue
        if dtype == pl.Utf8:
            exprs.append(_text_expr(pl.col(c)).alias(c))
        elif dtype in (pl.Date, pl.Datetime):
            fmt = "%Y-%m-%d" if dtype == pl.Date else "%Y-%m-%dT%H:%M:%S%.f"
            exprs.append(pl.col(c).dt.to_string(fmt).alias(c))
        elif dtype in (pl.Float32, pl.Float64):
            exprs.append(pl.when(pl.col(c).is_finite()).then(pl.col(c))
                         .otherwise(None).alias(c))
        elif dtype == pl.Null:
            exprs.append(pl.col(c).cast(pl.Utf8).alias(c))
    return df.with_columns(exprs) if exprs else df


# ---------------------------------------------------------------------------
# Source handling: path or raw bytes
# ---------------------------------------------------------------------------
def _as_source(source: Any):
    """polars/pandas want a path or a file object; give them one."""
    if isinstance(source, io.BytesIO):
        source.seek(0)
        return source
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(bytes(source))
    return str(source)


def _peek(source: Any, n: int = 4096) -> bytes:
    try:
        if isinstance(source, io.BytesIO):
            pos = source.tell()
            source.seek(0)
            head = source.read(n)
            source.seek(pos)
            return head
        if isinstance(source, (bytes, bytearray)):
            return bytes(source[:n])
        with open(str(source), "rb") as fh:
            return fh.read(n)
    except Exception:
        return b""


def _text(source: Any) -> str:
    if isinstance(source, io.BytesIO):
        source.seek(0)
        raw = source.read()
        source.seek(0)
        return raw.decode("utf-8-sig", "replace")
    if isinstance(source, (bytes, bytearray)):
        return bytes(source).decode("utf-8-sig", "replace")
    with open(str(source), "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        return fh.read()


def _sniff_separator(source: Any, name: str) -> str:
    ext = Path(str(name or "")).suffix.lower()
    if ext in (".tsv", ".tab"):
        return "\t"
    if ext == ".psv":
        return "|"
    head = _peek(source, 8192).decode("utf-8", "replace")
    line = ""
    for ln in head.splitlines():
        if ln.strip():
            line = ln
            break
    if not line:
        return ","
    counts = {",": line.count(","), ";": line.count(";"), "\t": line.count("\t"),
              "|": line.count("|")}
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] > 0 else ","


_FMT_BY_SUFFIX = {
    ".parquet": "parquet", ".pq": "parquet", ".parq": "parquet",
    ".xlsx": "excel", ".xls": "excel", ".xlsm": "excel", ".xlsb": "excel",
    ".ods": "excel",
    ".json": "json", ".jsonl": "json", ".ndjson": "json",
    ".csv": "csv", ".tsv": "csv", ".txt": "csv", ".tab": "csv", ".psv": "csv",
}


def _fmt_from_name(name: str) -> Optional[str]:
    return _FMT_BY_SUFFIX.get(Path(str(name or "")).suffix.lower())


def _detect_kind(source: Any, name: str) -> str:
    kind = _fmt_from_name(name)
    if kind:
        return kind
    head = _peek(source, 8)
    if head[:4] == b"PAR1":
        return "parquet"
    if head[:2] == b"PK":
        return "excel"
    if head[:1] in (b"{", b"["):
        return "json"
    return "csv"


# ---------------------------------------------------------------------------
# Public: probe
# ---------------------------------------------------------------------------
def probe(source: Union[str, Path, bytes, bytearray, io.BytesIO]) -> dict:
    """Cheap pre-flight: {"bytes_total": int, "fmt": str, "path": str|None}.

    It measures the source and sniffs the format from the file name; it never
    parses, decodes or counts anything. (If - and only if - the file name has no
    recognisable extension it reads the first 8 bytes to tell Parquet, Excel and
    JSON apart.) The point is to refuse an oversized file *before* we start.
    """
    path: Optional[str] = None
    size = 0
    if isinstance(source, io.BytesIO):
        try:
            pos = source.tell()
            source.seek(0, io.SEEK_END)
            size = int(source.tell())
            source.seek(pos)
        except Exception:
            size = 0
    elif isinstance(source, (bytes, bytearray)):
        size = len(source)
    else:
        path = str(source)
        try:
            size = int(os.path.getsize(path))
        except Exception:
            size = 0
    fmt = _fmt_from_name(path or "")
    if fmt is None:
        fmt = _detect_kind(source, path or "")
    return {"bytes_total": int(size), "fmt": str(fmt), "path": path}


def _count_rows(source: Any, name: str, rd: Optional["_Reader"] = None) -> Optional[int]:
    """How many lines are in a delimited file - a newline scan, not a parse.

    C-speed (`bytes.count`) so a 1.4 GB file costs a couple of seconds, and it
    buys an honest percentage-complete and ETA instead of a guess. It counts
    newlines inside quoted fields too, so treat it as a very good estimate; the
    final progress event always reports the true number.

    It is also interruptible: `rd.check()` runs every few blocks, so cancelling
    during the scan does not have to wait for a 2 GB file to be walked.
    """
    try:
        total = 0
        if isinstance(source, (bytes, bytearray)):
            total = bytes(source).count(b"\n")
        else:
            stream = source if isinstance(source, io.BytesIO) else None
            handle = None
            restore = 0
            if stream is not None:
                restore = stream.tell()
                stream.seek(0)
            else:
                handle = open(str(source), "rb")
            try:
                seen = 0
                while True:
                    if rd is not None and seen % 4 == 0:
                        rd.check()
                    block = stream.read(_COUNT_BLOCK) if stream is not None \
                        else handle.read(_COUNT_BLOCK)
                    if not block:
                        break
                    total += block.count(b"\n")
                    seen += 1
            finally:
                if stream is not None:
                    stream.seek(restore)
                elif handle is not None:
                    handle.close()
        if total <= 0:
            return None
        return max(0, total - 1)                 # the header line
    except (IngestRejected, IngestCancelled):
        raise
    except Exception:
        return None


def _materialise(df: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
    """A private copy of `df`, holding no reference to the block it came from.

    QA NEW-14: polars slices rather than copies, so a kept chunk is a *window*
    onto the whole block it was read out of. Keeping one chunk per block
    therefore pinned every block of the file: a 2.38 GB export held ~1 GB
    resident while keeping only 31 MB of actual rows, and the total grew in
    step with the file. Rebuilding the frame from its own values gives it its
    own memory and lets the blocks go. Measured on that file, committed memory
    after the read: 1537 MB before, 971 MB after. The dtype of every column is
    carried over explicitly so a rebuilt column is the same column.
    """
    if df is None or df.height == 0:
        return df
    try:
        schema = df.schema
        return pl.DataFrame(
            {c: pl.Series(c, df.get_column(c).to_list(), dtype=schema[c])
             for c in df.columns})
    except Exception:
        return df


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
class _Sampler:
    """Keeps a bounded, evenly-spread sample of a long stream.

    Rows are taken on a fixed stride measured from the file's row count, so the
    sample is spread from the first row to the last instead of piling up at the
    front - and the cost is proportional to what we keep, not to what we read.

    When the file has a low-cardinality column (a segment, a region, a tier) the
    stride is applied *inside each category*, and the first row of every
    category is always kept. A rare segment is therefore never sampled away,
    which is the whole point of stratifying.

    When the file has a *record key* - a column that repeats on adjacent rows
    and holds an identifier per record (a quote id, an order id, a deal id) -
    the stride is applied to whole records instead of to loose lines, and every
    row of a kept record is kept. QA NEW-10 / TC-02: striding lines split deals
    down the middle, which silently destroys any dependency that lives at the
    deal level. The reduction is disclosed with the record key named, so the
    user can see that whole deals - not loose lines - were left out.
    """

    def __init__(self, cap: int, total_rows: Optional[int], seed: int = _SAMPLE_SEED):
        self.cap = max(1, int(cap))
        self.seed = int(seed)
        self.total_rows = int(total_rows) if total_rows else 0
        self.stride = 1
        if self.total_rows > self.cap:
            self.stride = int(math.ceil(self.total_rows / float(self.cap)))
        self.strata: Optional[str] = None
        self._strata_tries = 0
        self._offsets: dict[Any, int] = {}
        self._seen_strata: set = set()
        self._frames: list[pl.DataFrame] = []
        self._height = 0
        self.seen = 0
        self._last_frame: Optional[pl.DataFrame] = None
        #: the column that groups rows into whole business records, if any
        self.record_key: Optional[str] = None
        self._record_tries = 0
        self._group_ordinal = -1        # ordinal of the last record opened
        self._cur_key: Any = None       # the record that was still open
        self.kept_groups = 0
        self.total_groups = 0

    # -- the record key -----------------------------------------------------
    def set_total(self, total_rows: Optional[int]) -> None:
        """Adopt a row count discovered after construction (parquet metadata).

        Without it the stride stays at 1 and a huge parquet file would be read
        whole - the one path where "bounded memory" could still be broken.
        """
        if not total_rows:
            return
        self.total_rows = int(total_rows)
        if self.total_rows > self.cap and self.stride <= 1:
            self.stride = int(math.ceil(self.total_rows / float(self.cap)))

    def choose_record_key(self, df: pl.DataFrame) -> None:
        """Find the column that groups rows into whole business records.

        Cheap and best-effort: it runs on the first few chunks only, and if
        nothing qualifies the sampler falls back to the line stride it always
        used. A wrong guess here cannot corrupt an answer - it only changes
        which whole records are kept.
        """
        if self.record_key is not None or self._record_tries >= _RECORD_TRIES:
            return
        self._record_tries += 1
        if self.stride <= 1 or df is None or df.height < 64:
            return
        try:
            self._choose_record_key(df)
        except Exception:
            self.record_key = None      # never let this stop the read

    def _choose_record_key(self, df: pl.DataFrame) -> None:
        best: Optional[str] = None
        best_uniq = -1
        for c in df.columns:
            s = df.get_column(c)
            if s.dtype == pl.Boolean:
                continue
            try:
                text = s.cast(pl.Utf8, strict=False)
                prev = text.shift(1)
                same = (text == prev).fill_null(False)
            except Exception:
                continue
            n = df.height
            run_share = float(same.sum()) / max(1, n - 1)
            if run_share < _RECORD_MIN_RUN_SHARE:
                continue
            runs = int((~same).sum())
            if runs < _RECORD_MIN_RUNS:
                continue
            try:
                uniq = int(text.n_unique())
            except Exception:
                continue
            # an identifier has about one distinct value per run; a category
            # such as a tier or a region has three or four however many runs it
            # happens to form.
            if uniq < _RECORD_UNIQ_PER_RUN * runs:
                continue
            if uniq > best_uniq:
                best, best_uniq = c, uniq
        self.record_key = best

    # -- choosing what to stratify on ---------------------------------------
    def choose_strata(self, df: pl.DataFrame) -> None:
        """Pick the category column to sample within, from the first few chunks.

        Cheap on purpose: it runs at most `_STRATA_TRIES` times and only on a
        chunk, never over the whole file.
        """
        if self.strata is not None or self._strata_tries >= _STRATA_TRIES:
            return
        self._strata_tries += 1
        if self.stride <= 1 or df.height == 0:
            return
        try:
            self._choose_strata(df)
        except Exception:
            self.strata = None          # never let this stop the read

    def _choose_strata(self, df: pl.DataFrame) -> None:
        for c in df.columns:
            s = df.get_column(c)
            if s.dtype == pl.Boolean:
                continue
            try:
                uniq = int(s.cast(pl.Utf8, strict=False).n_unique())
            except Exception:
                continue
            if uniq < 2 or uniq > _STRATA_MAX_CARDINALITY:
                continue
            if uniq > _STRATA_MAX_SHARE * df.height:
                continue
            sample = _spread_sample(s, 200)
            if not sample:
                continue
            numeric = sum(1 for v in sample if _clean_number_text(v) is not None)
            if numeric / len(sample) >= 0.90:      # a number is not a category
                continue
            self.strata = c
            return

    # -- streaming ----------------------------------------------------------
    def add(self, df: pl.DataFrame) -> None:
        if df is None or df.height == 0:
            return
        self._last_frame = df.tail(1)
        offset = self.seen
        self.seen += df.height

        if self.stride <= 1:
            self._frames.append(df)
            self._height += df.height
            self._compact()
            return

        if self.record_key is not None:
            kept = self._select_groups(df)
        else:
            kept = self._select(df, offset)
        if kept is not None and kept.height:
            self._frames.append(_materialise(kept))
            self._height += kept.height
        self._compact()
        if self._height > self.cap * 1.5:
            # only reached when the row estimate was badly low: the stride
            # sample came out far bigger than we budgeted for.
            self._shrink()

    def _compact(self) -> None:
        """Merge the kept chunks into one frame every so often.

        QA NEW-14: a kept chunk is a window onto the block it was read from, not
        a private copy - polars slices rather than copies. Holding one chunk per
        block therefore pinned every block of the file: a 2.38 GB export kept
        only 31 MB of real rows but held ~1 GB resident. Merging the chunks
        forces one fresh copy of the rows we actually want, and the blocks they
        came from are released. Bounded by `_COMPACT_EVERY` blocks at a time
        instead of one per block in the whole file.
        """
        if len(self._frames) < _COMPACT_EVERY:
            return
        try:
            merged = pl.concat(self._frames, how="vertical_relaxed").rechunk()
        except Exception:
            return
        self._frames = [merged]

    def _select_groups(self, df: pl.DataFrame) -> pl.DataFrame:
        """Keep every row of the records whose turn it is.

        The stride counts *records*, not rows, so a kept deal arrives whole.
        Records that straddle a chunk boundary are handled by remembering the
        record that was still open and the ordinal it was given.
        """
        key = self.record_key
        if not key or key not in df.columns:
            return df
        try:
            return self._select_groups_inner(df)
        except Exception:
            # A failure here must degrade to "keep the chunk", never to
            # "silently keep nothing" - the caller cannot tell the difference.
            return df

    def _select_groups_inner(self, df: pl.DataFrame) -> pl.DataFrame:
        key = self.record_key
        try:
            text = df.get_column(key).cast(pl.Utf8, strict=False)
            prev = text.shift(1)
            starts = (text != prev).fill_null(True).to_numpy()
            keys = text.to_numpy()
        except Exception:
            return df
        n = df.height
        if n == 0:
            return df
        first_continues = bool(self._cur_key is not None and keys[0] == self._cur_key)
        if first_continues:
            starts[0] = False
        local = np.cumsum(starts)                       # 1..G within this chunk
        g_count = int(local[-1]) if n else 0
        # the ordinal of local record 1 depends on whether it is a continuation
        offset = (self._group_ordinal - 1) if first_continues else self._group_ordinal
        keep = np.zeros(g_count + 1, dtype=bool)
        if g_count:
            ordinals = offset + np.arange(1, g_count + 1, dtype=np.int64)
            keep[1:] = (ordinals % self.stride) == 0
        self._group_ordinal = offset + g_count
        self.total_groups = self._group_ordinal + 1
        self._cur_key = keys[-1]
        self.kept_groups += int(keep[1:].sum())
        if not keep.any():
            return df.head(0)
        return df.filter(pl.Series("__keep_group", keep[local]))

    def _trim_groups(self, df: pl.DataFrame, cap: int) -> pl.DataFrame:
        """Trim a frame back to `cap` rows without ever splitting a record."""
        try:
            return self._trim_groups_inner(df, cap)
        except Exception:
            return df.head(cap) if df.height > cap else df

    def _trim_groups_inner(self, df: pl.DataFrame, cap: int) -> pl.DataFrame:
        key = self.record_key
        if not key or key not in df.columns or df.height <= cap:
            return df.head(cap) if df.height > cap else df
        try:
            text = df.get_column(key).cast(pl.Utf8, strict=False)
            starts = (text != text.shift(1)).fill_null(True).to_numpy()
        except Exception:
            return df.head(cap)
        group = np.cumsum(starts)
        last = np.flatnonzero(np.append(group[1:] != group[:-1], True))
        sizes = np.diff(np.concatenate(([-1], last)))
        cumulative = np.cumsum(sizes)
        keep = cumulative <= cap
        if keep.size:
            keep[0] = True                   # always keep at least one record
        return df.filter(pl.Series("__keep_group", keep[group - 1]))

    def _select(self, df: pl.DataFrame, offset: int) -> Optional[pl.DataFrame]:
        col = self.strata
        if not col or col not in df.columns:
            start = (-offset) % self.stride
            if start >= df.height:
                return df.head(0)
            return df[start::self.stride]

        # per-category position inside this chunk
        idx = pl.int_range(pl.len()).over(col)
        try:
            values = df.get_column(col).unique().to_list()
        except Exception:
            values = []
        mapping = {v: self._offsets.get(v, 0) for v in values}
        try:
            base = pl.col(col).replace_strict(mapping, default=0)
        except Exception:
            base = pl.lit(0)
        global_idx = base + idx
        if self._seen_strata:
            is_new = ~pl.col(col).is_in(list(self._seen_strata)).fill_null(False)
        else:
            is_new = pl.lit(True)
        mask = ((global_idx % self.stride) == 0) | (is_new & (idx == 0))
        kept = df.filter(mask)

        try:
            counts = df.get_column(col).value_counts()
            for v, n in zip(counts.get_column(col).to_list(),
                            counts.get_column("count").to_list()):
                self._offsets[v] = self._offsets.get(v, 0) + int(n)
            self._seen_strata.update(values)
        except Exception:
            pass
        return kept

    def _shrink(self) -> None:
        """Safety net: only reached if the row estimate was far too low."""
        if not self._frames:
            return
        buf = (pl.concat(self._frames, how="vertical_relaxed")
               if len(self._frames) > 1 else self._frames[0])
        if self.record_key is not None:
            # trimming by rows here would cut a record in half, which is the
            # whole defect this path exists to avoid - trim by whole records.
            if buf.height > self.cap:
                buf = self._trim_groups(buf, self.cap)
            self.stride *= 2
            self._frames = [buf]
            self._height = buf.height
            return
        factor = 2
        while buf.height // factor > self.cap:
            factor *= 2
        buf = buf[::factor]
        self.stride *= factor
        self._frames = [buf]
        self._height = buf.height

    # -- finishing ----------------------------------------------------------
    def finish(self) -> Optional[pl.DataFrame]:
        frames = list(self._frames)
        if not frames:
            return None
        dropped = self._height < self.seen
        tail = self._last_frame
        if dropped and tail is not None and tail.height and self.record_key is None:
            # make sure the very last row of the file is represented: a stride
            # sample lands on it only by luck. With whole-record sampling the
            # last row is only meaningful together with the rest of its record,
            # so it is not pinned on its own.
            try:
                same = frames[-1].row(-1) == tail.row(0)
            except Exception:
                same = False
            if not same:
                frames.append(tail)
        df = (pl.concat(frames, how="vertical_relaxed")
              if len(frames) > 1 else frames[0])
        if df.height > self.cap:
            if self.record_key is not None:
                df = self._trim_groups(df, self.cap)
            else:
                # Trim back to the cap. The rows arriving here are already spread
                # across the whole file, so a seeded draw keeps that spread and
                # stays reproducible from one session to the next; the file's final
                # row is pinned back on afterwards.
                keep = max(1, self.cap - 1)
                try:
                    trimmed = df.sample(n=keep, seed=self.seed, shuffle=False)
                except Exception:
                    trimmed = df.head(keep)
                if tail is not None and tail.height:
                    df = pl.concat([trimmed, tail], how="vertical_relaxed")
                else:
                    df = trimmed
        return df


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------
class _Reader:
    """One read in flight: progress, cancellation, timing and the sample."""

    def __init__(self, name: str, narrator: Any, progress_cb: Any, cancel: Any,
                 limits: IngestLimits, t0: float, total_bytes: int,
                 rows_total: Optional[int], phase: str = "reading") -> None:
        self.name = name
        self.narrator = narrator
        self.progress_cb = progress_cb if callable(progress_cb) else None
        self.cancel = cancel if callable(cancel) else None
        self.limits = limits
        self.t0 = t0
        self.total_bytes = int(total_bytes or 0)
        self.rows_total = int(rows_total) if rows_total else None
        self.phase = phase
        self.rows_read = 0
        self.bytes_read = 0
        self.callbacks = 0
        self._window: deque = deque()
        self._last_emit = -1.0
        self._last_say = 0.0

    # -- guards -------------------------------------------------------------
    def check(self) -> None:
        """Called at least once per chunk: honour cancel() and the timeout."""
        if self.cancel is not None:
            try:
                stop = bool(self.cancel())
            except Exception:
                stop = False
            if stop:
                raise IngestCancelled()
        limit = float(self.limits.parse_timeout_s or 0)
        if limit > 0 and (time.perf_counter() - self.t0) > limit:
            raise IngestRejected(_timeout_message(self.name, limit))

    def sizing(self) -> None:
        """One plain-English line while we look through a big file to size it up."""
        self.callbacks += 1
        now = time.perf_counter() - self.t0
        self._last_emit = now
        if self.progress_cb is None:
            return
        try:
            self.progress_cb({
                "phase": "reading",
                "message": (f"Looking through your {_human_bytes(self.total_bytes)} "
                            f"file first, so we can show you how far along we are…"),
                "rows_read": 0,
                "rows_total": None,
                "bytes_read": 0,
                "bytes_total": self.total_bytes or None,
                "pct": None,
                "rate_rows_s": 0.0,
                "eta_s": None,
            })
        except Exception:
            pass

    # -- rate ---------------------------------------------------------------
    def _rate(self) -> float:
        w = self._window
        if len(w) >= 2:
            dt = w[-1][0] - w[0][0]
            dr = w[-1][1] - w[0][1]
            if dt > 0.05 and dr > 0:
                return float(dr) / float(dt)
        elapsed = time.perf_counter() - self.t0
        if elapsed > 0.2 and self.rows_read > 0:
            return float(self.rows_read) / elapsed
        return 0.0

    # -- emission -----------------------------------------------------------
    def tick(self, rows_read: int, bytes_read: Optional[int] = None,
             force: bool = False) -> None:
        """Called once per chunk (and whenever a reader can do better)."""
        self.rows_read = int(rows_read)
        if bytes_read is not None:
            self.bytes_read = int(bytes_read)
        now = time.perf_counter() - self.t0
        self._window.append((now, self.rows_read))
        while len(self._window) > 2 and now - self._window[0][0] > _RATE_WINDOW_S:
            self._window.popleft()
        if not force and (now - self._last_emit) < _PROGRESS_INTERVAL_S:
            return
        self._emit(now)

    def _emit(self, now: float) -> None:
        self.callbacks += 1
        self._last_emit = now
        rows = self.rows_read
        total = self.rows_total
        rate = self._rate()

        pct: Optional[float] = None
        if total:
            pct = max(0.0, min(100.0, 100.0 * rows / float(total)))

        bytes_read = self.bytes_read
        if self.total_bytes:
            if total:
                bytes_read = int(min(self.total_bytes, self.total_bytes * rows / float(total)))
            elif pct is not None:
                bytes_read = int(self.total_bytes * pct / 100.0)
            self.bytes_read = bytes_read

        eta: Optional[float] = None
        if total and rate > 0:
            remaining = total - rows
            candidate = remaining / rate if remaining > 0 else 0.0
            if math.isfinite(candidate) and candidate >= 0:
                eta = candidate

        word = _rows_word(rows)
        if eta is not None and eta >= 1.0:
            message = (f"Reading your file — {_human_count(rows)} {word} so far, "
                       f"about {_human_duration(eta)} left")
        elif rate > 0:
            message = (f"Reading your file — {_human_count(rows)} {word} so far, "
                       f"{_human_count(rate)} rows a second")
        else:
            message = f"Reading your file — {_human_count(rows)} {word} so far"

        if self.progress_cb is not None:
            try:
                self.progress_cb({
                    "phase": self.phase,
                    "message": message,
                    "rows_read": rows,
                    "rows_total": total,
                    "bytes_read": bytes_read,
                    "bytes_total": self.total_bytes or None,
                    "pct": None if pct is None else round(pct, 2),
                    "rate_rows_s": round(rate, 1),
                    "eta_s": None if eta is None else round(eta, 2),
                })
            except Exception:
                pass

        if self.narrator is not None and (now - self._last_say) >= _NARRATE_INTERVAL_S:
            self._last_say = now
            _say(self.narrator, message, phase="read", progress=None)

    def finish(self, columns: int, kept: Optional[int] = None) -> None:
        """One last, honest event: we know the true row count now."""
        self.phase = "profiling"
        self.rows_read = self.rows_read or 0
        self.rows_total = self.rows_read or self.rows_total
        now = time.perf_counter() - self.t0
        self._last_emit = now
        word = _rows_word(self.rows_read)
        if kept is not None and kept < self.rows_read:
            message = (f"Read {_human_count(self.rows_read)} {word} and "
                       f"{_human_count(columns)} columns, and kept a "
                       f"{_human_count(kept)}-row sample spread evenly across the "
                       f"file. Working out what each column holds…")
        else:
            message = (f"Read {_human_count(self.rows_read)} {word} and "
                       f"{_human_count(columns)} columns. Working out what each "
                       f"column holds…")
        if self.progress_cb is not None:
            try:
                self.progress_cb({
                    "phase": "profiling",
                    "message": message,
                    "rows_read": self.rows_read,
                    "rows_total": self.rows_read,
                    "bytes_read": self.total_bytes or self.bytes_read,
                    "bytes_total": self.total_bytes or None,
                    "pct": 100.0,
                    "rate_rows_s": round(self._rate(), 1),
                    "eta_s": 0.0,
                })
                self.callbacks += 1
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Readers - each one streams, reports and hands chunks to the sampler
# ---------------------------------------------------------------------------
def _peek_header(source: Any, sep: str) -> list[str]:
    """The first line of a delimited file, for a file with a header but no rows."""
    head = _peek(source, 65536).decode("utf-8-sig", "replace")
    for line in head.splitlines():
        if not line.strip():
            continue
        try:
            row = next(_csv.reader([line], delimiter=sep))
        except Exception:
            return []
        return [str(c) for c in row]
    return []


def _finish_or_empty(sampler: _Sampler, columns: Optional[list[str]],
                     rd: Optional[_Reader] = None) -> Optional[pl.DataFrame]:
    """The sampled frame - or an empty frame that still carries the columns.

    A file with a header and no rows is a real thing users upload (an export
    that returned nothing). It has no rows to show, but it does have columns,
    and the profile must still be able to say so.
    """
    if rd is not None:
        rd.check()
    df = sampler.finish()
    if rd is not None:
        rd.check()
    if df is None and columns:
        return pl.DataFrame({c: pl.Series(c, [], dtype=pl.Utf8) for c in columns})
    return df


def _arrow_csv_reader(source: Any, sep: str):
    """A CSV reader that holds one block of the file, not the whole file.

    QA NEW-14: this is the fix for "raise the ceiling but keep memory bounded".
    polars' batched CSV reader keeps the entire file resident - measured at
    2.54 GB peak for a 2.38 GB file with a 32,768-row batch - so raising the
    size ceiling alone would simply have moved the failure from "refused" to
    "the machine runs out of memory". pyarrow's streaming reader peaked at
    150 MB on the same file with the same rows read.

    Only a real file on disk is streamed here: that is the case that gets big
    enough to matter, and it keeps this reader away from the in-memory sources
    the paste path hands us. Every column is read as text, exactly as the
    polars path does with `infer_schema_length=0`, so a leading zero on an
    identifier survives and the rows that come out are the same rows.
    """
    import pyarrow as pa
    import pyarrow.csv as pacsv

    def _skip_invalid(_row: Any) -> str:
        # polars spells this `truncate_ragged_lines` + `ignore_errors`; pyarrow
        # asks the caller what to do with a row that does not fit the header.
        return "skip"

    read = pacsv.ReadOptions(block_size=int(_ARROW_BLOCK_BYTES))
    parse = pacsv.ParseOptions(delimiter=sep, invalid_row_handler=_skip_invalid)
    probe = pacsv.open_csv(str(source), read_options=read, parse_options=parse)
    names = [str(n) for n in probe.schema.names]
    convert = pacsv.ConvertOptions(
        column_types={n: pa.string() for n in names}, strings_can_be_null=False)
    reader = pacsv.open_csv(str(source), read_options=read, parse_options=parse,
                            convert_options=convert)
    return reader, names


def _read_delimited_arrow(source: Any, sep: str, rd: _Reader,
                          sampler: _Sampler) -> tuple[Optional[list[str]], bool]:
    """Stream a delimited file with a bounded working set.

    Returns `(columns, failed)`. `failed` is True when the stream stopped
    part-way through the file - the caller uses it to decide whether re-reading
    with another reader is safe, since rows already counted must never be
    counted twice.
    """
    reader, names = _arrow_csv_reader(source, sep)
    columns: Optional[list[str]] = None
    failed = False
    while True:
        rd.check()
        try:
            batch = reader.read_next_batch()
        except StopIteration:
            break
        except Exception:
            failed = True
            break
        if batch is None or batch.num_rows == 0:
            continue
        # `_clean_frame` rewrites every text column, so what the sampler keeps
        # is a fresh allocation rather than a window onto the block we just
        # read - the batch can then be dropped and its memory returned.
        chunk = _clean_frame(pl.from_arrow(batch))
        if columns is None:
            columns = sanitise_headers(list(chunk.columns))
        if chunk.width == len(columns):
            chunk.columns = columns
        sampler.choose_record_key(chunk)
        sampler.choose_strata(chunk)
        sampler.add(chunk)
        rd.tick(sampler.seen, force=True)
        del batch, chunk
    if columns is None:
        columns = sanitise_headers(names)
    return columns, failed


def _read_delimited(source: Any, name: str, rd: _Reader,
                    sampler: _Sampler) -> tuple[Optional[pl.DataFrame], str]:
    sep = _sniff_separator(source, name)
    batch = max(1, min(int(rd.limits.chunk_rows or _READ_BATCH), _READ_BATCH))
    how = "the fast columnar reader"

    # QA NEW-14: stream first when the source is a file on disk, because that is
    # the only shape that gets big enough for the reader's memory to matter.
    if not isinstance(source, (io.BytesIO, bytes, bytearray)):
        try:
            is_file = Path(str(source)).is_file()
        except Exception:
            is_file = False
        if is_file:
            try:
                columns, failed = _read_delimited_arrow(source, sep, rd, sampler)
            except (IngestRejected, IngestCancelled):
                raise
            except Exception:
                columns, failed = None, True
            if columns and not failed:
                return _finish_or_empty(sampler, columns, rd), how
            if sampler.seen:
                # The stream broke part-way and we have already counted rows.
                # Re-reading with another reader would double-count them and
                # make the progress counter jump backwards, so keep what we
                # read - the same thing the polars path does with a file that
                # breaks half-way through.
                return _finish_or_empty(sampler, columns, rd), how

    reader = None
    try:
        reader = pl.read_csv_batched(
            _as_source(source), separator=sep, infer_schema_length=0,
            truncate_ragged_lines=True, has_header=True, ignore_errors=True,
            encoding="utf8-lossy", batch_size=batch, low_memory=True,
        )
    except Exception:
        reader = None

    columns: Optional[list[str]] = None
    if reader is not None:
        try:
            while True:
                rd.check()
                got = reader.next_batches(1)
                if not got:
                    break
                for raw in got:
                    if raw is None or raw.height == 0:
                        continue
                    chunk = _clean_frame(raw)
                    if columns is None:
                        columns = sanitise_headers(list(chunk.columns))
                    if chunk.width == len(columns):
                        chunk.columns = columns
                    sampler.choose_record_key(chunk)
                    sampler.choose_strata(chunk)
                    sampler.add(chunk)
                    rd.tick(sampler.seen, force=True)
        except (IngestRejected, IngestCancelled):
            raise
        except Exception:
            # a broken file half-way through: keep whatever we already read
            if columns is None:
                raise
        if columns is not None:
            return _finish_or_empty(sampler, columns, rd), how

    # the batched reader gave us nothing usable - fall back to a whole-file read
    try:
        df = pl.read_csv(_as_source(source), separator=sep, infer_schema_length=0,
                         truncate_ragged_lines=True, has_header=True,
                         ignore_errors=True, encoding="utf8-lossy")
        df = _clean_frame(df)
        columns = sanitise_headers(list(df.columns))
        df.columns = columns
        if df.height:
            rd.check()
            sampler.choose_record_key(df)
            sampler.choose_strata(df)
            sampler.add(df)
            rd.tick(sampler.seen, force=True)
        return _finish_or_empty(sampler, columns, rd), how
    except (IngestRejected, IngestCancelled):
        raise
    except Exception:
        pass

    # last resort: the plain text reader, which copes with anything
    header = _peek_header(source, sep)
    if not header:
        return None, "the plain text reader"
    text = _text(source)
    reader2 = _csv.reader(io.StringIO(text), delimiter=sep)
    data = [r for r in reader2 if any((c or "").strip() for c in r)]
    if not data:
        columns = sanitise_headers(header)
        return _finish_or_empty(sampler, columns, rd), "the plain text reader"
    header = data[0]
    width = max(len(header), max((len(r) for r in data), default=0))
    header = list(header) + [f"unnamed_{i + 1}" for i in range(len(header), width)]
    columns = sanitise_headers(header)
    records = []
    for r in data[1:]:
        records.append({c: _json_safe(r[i] if i < len(r) else None)
                        for i, c in enumerate(columns)})
    rd.check()
    if records:
        sampler.choose_record_key(df)
        sampler.choose_strata(df)
        sampler.add(df)
        rd.tick(sampler.seen, force=True)
    return _finish_or_empty(sampler, columns, rd), "the plain text reader"


def _read_parquet(source: Any, name: str, rd: _Reader,
                  sampler: _Sampler) -> tuple[Optional[pl.DataFrame], str]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if isinstance(source, io.BytesIO):
        source.seek(0)
        handle: Any = pa.BufferReader(source.read())
    elif isinstance(source, (bytes, bytearray)):
        handle = pa.BufferReader(bytes(source))
    else:
        handle = str(source)

    pf = pq.ParquetFile(handle)
    total = int(pf.metadata.num_rows or 0)
    rd.rows_total = total or None
    # The row count is only known from the footer, after the sampler was built:
    # hand it over so the stride is right and a huge parquet is not read whole.
    sampler.set_total(total or None)
    batch = max(1, min(int(rd.limits.chunk_rows or _READ_BATCH), _READ_BATCH))

    columns: Optional[list[str]] = None
    for record_batch in pf.iter_batches(batch_size=batch):
        rd.check()
        chunk = _clean_frame(pl.from_arrow(record_batch))
        if columns is None:
            columns = sanitise_headers(list(chunk.columns))
            chunk.columns = columns
        sampler.choose_record_key(chunk)
        sampler.choose_strata(chunk)
        sampler.add(chunk)
        rd.tick(sampler.seen, force=True)
    return _finish_or_empty(sampler, columns, rd), "the fast columnar reader"


def _read_excel(source: Any, name: str, rd: _Reader,
                sampler: _Sampler) -> tuple[Optional[pl.DataFrame], str]:
    how = "the spreadsheet reader"
    try:
        import openpyxl
    except Exception:
        openpyxl = None

    if openpyxl is not None:
        book = None
        try:
            book = openpyxl.load_workbook(_as_source(source), read_only=True,
                                          data_only=True)
            sheet = book[book.sheetnames[0]]
            try:
                total = int(sheet.max_row or 0) - 1
                rd.rows_total = total if total > 0 else None
            except Exception:
                pass
            batch = max(1, min(int(rd.limits.chunk_rows or _READ_BATCH), _READ_BATCH))
            columns: Optional[list[str]] = None
            pending: list[list] = []
            for row in sheet.iter_rows(values_only=True):
                values = [_json_safe(v) for v in row]
                if columns is None:
                    if not any(v is not None and str(v).strip() for v in values):
                        continue
                    columns = sanitise_headers(
                        ["" if v is None else str(v) for v in values])
                    continue
                if not any(v is not None for v in values):
                    continue                        # skip blank lines
                if len(values) < len(columns):
                    values = values + [None] * (len(columns) - len(values))
                pending.append(values[:len(columns)])
                if len(pending) >= batch:
                    rd.check()
                    sampler.choose_record_key(chunk)
                    sampler.choose_strata(chunk)
                    sampler.add(_frame_from_lists(pending, columns))
                    pending = []
                    rd.tick(sampler.seen, force=True)
            if pending:
                rd.check()
                sampler.choose_record_key(chunk)
                sampler.choose_strata(chunk)
                sampler.add(_frame_from_lists(pending, columns))
                rd.tick(sampler.seen, force=True)
            if columns is not None:
                return _finish_or_empty(sampler, columns, rd), how
        except (IngestRejected, IngestCancelled):
            raise
        except Exception:
            pass
        finally:
            if book is not None:
                try:
                    book.close()
                except Exception:
                    pass

    # fallback: let polars have a go at the whole sheet
    df = pl.read_excel(_as_source(source))
    columns = sanitise_headers(list(df.columns))
    df.columns = columns
    rd.check()
    sampler.choose_record_key(df)
    sampler.choose_strata(df)
    sampler.add(df)
    rd.tick(sampler.seen, force=True)
    return sampler.finish(), "the backup spreadsheet reader"


def _frame_from_lists(rows: list[list], columns: list[str]) -> pl.DataFrame:
    try:
        return pl.DataFrame(rows, schema=columns, orient="row", strict=False)
    except Exception:
        return pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(columns)},
                            strict=False)


def _records_to_frame(records: list[dict], columns: list[str]) -> pl.DataFrame:
    out = []
    for rec in records:
        out.append({c: _json_safe(rec.get(c)) for c in columns})
    return pl.DataFrame(out, infer_schema_length=None, strict=False)


def _read_json(source: Any, name: str, rd: _Reader,
               sampler: _Sampler) -> tuple[Optional[pl.DataFrame], str]:
    text = _text(source)
    data: Any = None
    try:
        data = json.loads(text)
    except Exception:
        data = None
    if data is None:                              # JSON lines
        recs = []
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
        data = recs
    if isinstance(data, dict):
        for key in ("data", "rows", "records", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            if data and all(isinstance(v, list) for v in data.values()):
                length = max((len(v) for v in data.values()), default=0)
                data = [{k: (v[i] if i < len(v) else None) for k, v in data.items()}
                        for i in range(length)]
            else:
                data = [data]
    if not isinstance(data, list):
        data = [data]

    columns: list[str] = []
    seen = set()
    for d in data:
        if not isinstance(d, dict):
            continue
        for k in d.keys():
            if k not in seen:
                seen.add(k)
                columns.append(k)
    if not columns:
        return None, "the built-in JSON reader"
    columns = sanitise_headers(columns)
    rd.rows_total = len(data) or None

    batch = max(1, min(int(rd.limits.chunk_rows or _READ_BATCH), _READ_BATCH))
    for start in range(0, len(data), batch):
        rd.check()
        if not chunk_records:
            continue
        chunk = _records_to_frame(chunk_records, columns)
        sampler.choose_record_key(chunk)
        sampler.choose_strata(chunk)
        sampler.add(chunk)
        rd.tick(sampler.seen, force=True)
    return _finish_or_empty(sampler, columns, rd), "the built-in JSON reader"


# ---------------------------------------------------------------------------
# Public: read_table
# ---------------------------------------------------------------------------
def read_table(source: Union[str, Path, bytes, bytearray, io.BytesIO],
               name: str = "dataset",
               narrator: Any = None,
               progress: Any = None,
               cancel: Any = None,
               limits: Optional[IngestLimits] = None,
               ) -> tuple[list[dict], list[str]]:
    """Read a table from a filesystem path or from raw bytes.

    Supports .csv .tsv .txt .xlsx .xls .parquet .json (records or columnar).
    Returns (rows, column_names).

    The file is streamed: chunks are parsed, sampled and released one at a time,
    so memory stays bounded no matter how big the file is. `progress(payload)` is
    called at least once per chunk with a plain-English message, a row counter, a
    rolling rows-per-second rate and an ETA; `cancel()` is polled every chunk.

    Raises `IngestRejected` when the file is over `limits.max_file_bytes` or the
    read runs past `limits.parse_timeout_s`, and `IngestCancelled` when the
    caller's `cancel()` returns True. Anything else unreadable gives ([], []) plus
    a plain-language narrator message, exactly as before.
    """
    limits = limits if isinstance(limits, IngestLimits) else IngestLimits()
    t0 = time.perf_counter()

    info = probe(source)
    total_bytes = int(info.get("bytes_total") or 0)
    max_bytes = int(limits.max_file_bytes or 0)
    if max_bytes and total_bytes > max_bytes:
        raise IngestRejected(_too_big_message(total_bytes, max_bytes))

    fmt = str(info.get("fmt") or "csv")
    _say(narrator, "Opening your file…", phase="read", progress=0.02,
         detail=f"source: {name}")

    rows_total = None
    rd = _Reader(name=name, narrator=narrator, progress_cb=progress, cancel=cancel,
                 limits=limits, t0=t0, total_bytes=total_bytes, rows_total=None)
    if fmt == "csv":
        # Counting rows first is what makes the percentage and the ETA honest.
        # For a big file that walk takes a couple of seconds, so say so.
        if total_bytes > _SIZING_NOTICE_BYTES:
            rd.sizing()
        rows_total = _count_rows(source, name, rd)
        rd.rows_total = rows_total
    sampler = _Sampler(limits.max_working_rows, rows_total)

    df: Optional[pl.DataFrame] = None
    how = "the fast columnar reader"
    try:
        if fmt == "parquet":
            df, how = _read_parquet(source, name, rd, sampler)
        elif fmt == "excel":
            df, how = _read_excel(source, name, rd, sampler)
        elif fmt == "json":
            df, how = _read_json(source, name, rd, sampler)
        else:
            df, how = _read_delimited(source, name, rd, sampler)
    except (IngestRejected, IngestCancelled):
        raise
    except Exception as exc:                      # last line of defence
        df, how = None, "no reader"
        _say(narrator,
             f"We could not read '{name}' as a table, so we skipped it.",
             phase="read", level="error",
             detail=f"{type(exc).__name__}: {exc}")

    if df is None or not df.width:
        _say(narrator,
             f"We could not find any columns in '{name}' - the file may be empty "
             f"or not a table we understand.",
             phase="read", level="warn", progress=0.12)
        empty = _RowList()
        empty.columns = []
        return empty, []

    columns = list(df.columns)
    n_seen = int(sampler.seen)
    sampled = n_seen > int(limits.max_working_rows)
    # QA NEW-10 / TC-02: the note has to say *what* was sampled, not just how
    # much - whole records, and which column told us how to group them.
    note = _sample_note(n_seen, df.height, sampler.record_key or "") if sampled else ""

    rows = _RowList()
    # Build the one and only `list[dict]` in slices, so a cancel during the
    # conversion is still noticed promptly and we never hold the frame and a
    # giant intermediate list at the same time.
    slice_rows = max(1, min(int(limits.chunk_rows or _READ_BATCH), 250_000))
    for start in range(0, df.height, slice_rows):
        rd.check()
        rows.extend(df.slice(start, slice_rows).to_dicts())
    rows.sampled = bool(sampled)
    rows.sample_note = note
    rows.n_rows_in_file = n_seen
    rows.columns = columns

    rd.finish(len(columns), df.height)
    elapsed = time.perf_counter() - t0
    if sampled:
        _say(narrator, note, phase="read", level="warn", progress=0.16)
    # QA NEW-05: this line used to print the *sample* size as if it were the
    # file's size ("Found 1,125,577 rows" for a 2,251,151-row file). When the
    # two differ, say so right here.
    if sampled:
        _say(narrator,
             f"Found {_human_count(n_seen)} rows in the file; using "
             f"{_human_count(len(rows))} of them.", phase="read", progress=0.12,
             detail="columns: " + ", ".join(columns[:14])
                    + ("…" if len(columns) > 14 else ""))
    else:
        _say(narrator, f"Found {len(rows):,} rows and {len(columns)} columns.",
             phase="read", progress=0.12,
             detail="columns: " + ", ".join(columns[:14])
                    + ("…" if len(columns) > 14 else ""))
    _say(narrator, f"Read in {elapsed:.2f} seconds using {how}.",
         phase="read", progress=0.16, detail=f"kind={fmt}")
    return rows, columns


# ---------------------------------------------------------------------------
# Public: profile
# ---------------------------------------------------------------------------
def _dataset_id(name: str, columns: Sequence[str], rows: Sequence[dict]) -> str:
    h = hashlib.sha256()
    h.update(str(name).encode("utf-8", "replace"))
    h.update(b"|")
    h.update("|".join(columns).encode("utf-8", "replace"))
    h.update(b"|")
    h.update(str(len(rows)).encode())
    for r in rows[:300]:
        h.update("|".join("" if r.get(c) is None else str(r.get(c))
                          for c in columns).encode("utf-8", "replace"))
    return "ds_" + h.hexdigest()[:6]


def _columns_from_rows(rows: Sequence[dict]) -> list[str]:
    cols: list[str] = []
    seen = set()
    for r in rows or []:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)
    return cols


def _frame_from_rows(rows: Sequence[dict], cols: Sequence[str]) -> pl.DataFrame:
    """One vectorised frame for the whole table - no Python row walk."""
    if not cols:
        return pl.DataFrame()
    if not rows:
        return pl.DataFrame({c: pl.Series(c, [], dtype=pl.Utf8) for c in cols})
    try:
        df = pl.DataFrame(list(rows), infer_schema_length=None, strict=False)
    except Exception:
        df = pl.DataFrame({c: [r.get(c) for r in rows] for c in cols}, strict=False)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None, dtype=pl.Utf8).alias(c) for c in missing])
    extra = [c for c in df.columns if c not in set(cols)]
    if extra:
        df = df.drop(extra)
    return df.select(list(cols))


def profile(rows: list[dict], columns: Optional[list[str]] = None,
            name: str = "dataset", *,
            sampled: bool = False,
            sample_note: str = "") -> DatasetProfile:
    """Describe a table: what each column holds, how much is missing, and
    anything that looks wrong - in plain English.

    One pass of vectorised polars expressions, so the cost is a handful of
    linear scans instead of one Python function call per cell.
    """
    carried_sampled = bool(getattr(rows, "sampled", False))
    carried_note = str(getattr(rows, "sample_note", "") or "")
    carried_file_rows = int(getattr(rows, "n_rows_in_file", 0) or 0)
    rows = list(rows or [])
    cols = list(columns) if columns else _columns_from_rows(rows)
    n = len(rows)
    warnings: list[str] = []

    if not sampled:
        sampled = carried_sampled
    if sampled and not sample_note:
        sample_note = carried_note

    if n == 0:
        warnings.append("This file has no rows in it, so there is nothing to simulate yet.")
    if not cols:
        warnings.append("We could not find any columns in this file.")
    if len(cols) > 200:
        warnings.append(
            f"This file is very wide ({len(cols)} columns). Only the first few hundred "
            f"are practical to map by hand - consider trimming it to the columns you need.")
    if sampled and sample_note:
        warnings.append(sample_note)

    df = _frame_from_rows(rows, cols)
    profiles: list[ColumnProfile] = []

    for c in cols:
        try:
            series = df.get_column(c)
        except Exception:
            profiles.append(ColumnProfile(name=c, dtype="categorical", null_pct=100.0,
                                          unique=0, sample_values=[]))
            continue
        try:
            n_present = int(series.len() - series.null_count())
        except Exception:
            n_present = 0
        null_pct = 0.0 if n == 0 else round(100.0 * (n - n_present) / n, 1)

        try:
            uniq = int(series.cast(pl.Utf8, strict=False).n_unique())
        except Exception:
            try:
                uniq = int(series.n_unique())
            except Exception:
                uniq = 0

        samples: list[Any] = []
        try:
            for v in series.drop_nulls().unique(maintain_order=True).head(5).to_list():
                samples.append(_json_safe(v))
        except Exception:
            samples = []

        infer_sample = _sample_series(series, _INFER_SAMPLE)
        dtype = _infer_dtype_vec(df, c, n_present, uniq, infer_sample)

        numeric_stats = None
        if dtype == "numeric" and n_present:
            try:
                values = _numeric_values(series).drop_nulls()
            except Exception:
                values = None
            if values is not None and values.len():
                try:
                    numeric_stats = {
                        "min": round(float(values.min()), 4),
                        "max": round(float(values.max()), 4),
                        "mean": round(float(values.mean()), 4),
                        "p50": round(float(values.median()), 4),
                    }
                except Exception:
                    numeric_stats = None

        if n_present == 0 and n > 0:
            warnings.append(f"'{c}' is empty in every row, so we will ignore it.")
        elif null_pct > 50.0:
            warnings.append(
                f"'{c}' is missing in {null_pct:.0f}% of rows - any result that leans on "
                f"it will be thin.")
        elif n_present > 1 and uniq == 1:
            warnings.append(
                f"'{c}' has the same value ({samples[0] if samples else ''}) in every row, "
                f"so it cannot explain why outcomes differ.")

        profiles.append(ColumnProfile(
            name=c, dtype=dtype, null_pct=null_pct, unique=uniq,
            sample_values=samples, numeric_stats=numeric_stats,
        ))

    preview = [{c: _json_safe(r.get(c, None)) for c in cols} for r in rows[:8]]
    return DatasetProfile(
        dataset_id=_dataset_id(name, cols, rows),
        source_name=str(name),
        n_rows=n,
        n_cols=len(cols),
        columns=profiles,
        warnings=warnings,
        preview=preview,
        sampled=bool(sampled),
        n_rows_in_file=(carried_file_rows or (n if sampled else 0)),
        sample_note=(sample_note if sampled else ""),
    )
