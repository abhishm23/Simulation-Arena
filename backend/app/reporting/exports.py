"""Single entry point for turning a RunResult into downloadable artifacts.

    build_artifacts(result, ["html", "xlsx", "csv", "pdf"], out_dir)

Never raises. A format that fails is skipped, and a plain-English line is
appended to ``result.warnings`` so the user is told rather than left guessing.
The mapping of format -> filename is also written back to ``result.artifacts``.
"""
from __future__ import annotations

import os
import traceback
from typing import Any, Optional

from ..core.contracts import RunResult

FILENAMES = {
    "html": "report.html",
    "xlsx": "report.xlsx",
    "csv": "results.zip",
    "pdf": "report.pdf",
}

FORMAT_LABEL = {
    "html": "interactive report (HTML)",
    "xlsx": "Excel workbook",
    "csv": "spreadsheet bundle (zip of CSV files)",
    "pdf": "PDF report",
}


def artifact_filename(fmt: str) -> str:
    """'html' -> 'report.html'. Unknown formats still get a safe name."""
    key = str(fmt or "").strip().lower()
    if key in FILENAMES:
        return FILENAMES[key]
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in key)[:24]
    return f"report.{safe}" if safe else "report.out"


def build_artifacts(result: RunResult, formats: list[str], out_dir: str) -> dict[str, str]:
    """Render every requested format into ``out_dir``.

    Returns {format: filename}. Failures are skipped (and recorded on the
    result) instead of raising - a broken export must never lose a run.
    """
    res = result if result is not None else RunResult(run_id="unknown", use_case_id="unknown")
    out: dict[str, str] = {}
    wanted: list[str] = []
    for f in (formats or []):
        key = str(f or "").strip().lower()
        if key and key not in wanted:
            wanted.append(key)

    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as exc:
        _warn(res, f"We could not create the export folder, so no downloadable "
                   f"files were produced ({type(exc).__name__}).")
        return out

    for key in wanted:
        if key not in FILENAMES:
            continue
        filename = artifact_filename(key)
        path = os.path.join(out_dir, filename)
        try:
            if key == "html":
                from . import html_report
                html_report.render(res, path)
            elif key == "xlsx":
                from . import excel_report
                excel_report.render(res, path)
            elif key == "csv":
                from . import csv_bundle
                csv_bundle.render(res, path)
            elif key == "pdf":
                from . import pdf_report
                pdf_report.render(res, path)
            if not os.path.exists(path):
                raise RuntimeError("renderer produced no file")
            out[key] = filename
        except Exception as exc:  # noqa: BLE001 - a run must survive a bad export
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
            _warn(res, f"The {FORMAT_LABEL.get(key, key)} could not be created, so it is "
                       f"missing from your downloads ({type(exc).__name__}). "
                       f"The other formats are unaffected.")
            try:
                traceback.print_exc()
            except Exception:
                pass

    try:
        res.artifacts = dict(res.artifacts or {})
        res.artifacts.update(out)
    except Exception:
        pass
    return out


def _warn(res: RunResult, message: str) -> None:
    try:
        warnings = list(res.warnings or [])
        if message not in warnings:
            warnings.append(message)
            res.warnings = warnings
    except Exception:
        pass
