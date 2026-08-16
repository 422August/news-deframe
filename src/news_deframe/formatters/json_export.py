"""JSON serialisation of DiffReport and EventAnalysis.

Exports a clean, schema-compliant JSON string from a :class:`DiffReport`
or :class:`EventAnalysis` with optional pretty-printing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from pydantic import BaseModel

from news_deframe.schemas import DiffReport
from news_deframe.analysis.schemas import EventAnalysis

ReportType = Union[DiffReport, EventAnalysis, BaseModel]


def report_to_json(report: ReportType, indent: int = 2) -> str:
    """Serialise *report* to a JSON string.

    Parameters
    ----------
    report:
        The diff report or event analysis to serialise.
    indent:
        Indentation level for pretty-printing (0 = compact).

    Returns
    -------
    str
        UTF-8 JSON string.
    """
    return report.model_dump_json(indent=indent)


def save_report(report: ReportType, output_path: str | Path, indent: int = 2) -> Path:
    """Write the JSON report to *output_path*.

    Parameters
    ----------
    report:
        The diff report or event analysis to save.
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


def event_to_json(analysis: EventAnalysis, indent: int = 2) -> str:
    """Serialise *analysis* to a JSON string."""
    return analysis.model_dump_json(indent=indent)


def save_event_analysis(
    analysis: EventAnalysis, output_path: str | Path, indent: int = 2
) -> Path:
    """Write the event analysis JSON to *output_path*."""
    return save_report(analysis, output_path, indent=indent)
