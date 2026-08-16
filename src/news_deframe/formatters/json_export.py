"""JSON serialisation of DiffReport.

Exports a clean, schema-compliant JSON string from a :class:`DiffReport`
with optional pretty-printing.
"""
from __future__ import annotations

import json
from pathlib import Path

from news_deframe.schemas import DiffReport


def report_to_json(report: DiffReport, indent: int = 2) -> str:
    """Serialise *report* to a JSON string.

    Parameters
    ----------
    report:
        The diff report to serialise.
    indent:
        Indentation level for pretty-printing (0 = compact).

    Returns
    -------
    str
        UTF-8 JSON string.
    """
    return report.model_dump_json(indent=indent)


def save_report(report: DiffReport, output_path: str | Path, indent: int = 2) -> Path:
    """Write the JSON report to *output_path*.

    Parameters
    ----------
    report:
        The diff report to save.
    output_path:
        Destination file path.  Parent directories must exist.
    indent:
        Indentation level.

    Returns
    -------
    Path
        The resolved output path.
    """
    path = Path(output_path)
    path.write_text(report_to_json(report, indent=indent), encoding="utf-8")
    return path
