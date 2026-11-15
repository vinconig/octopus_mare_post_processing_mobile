"""Final, hand-made selection of the walks that ship on Zenodo.

The automatic gates (cleaning, matching, package_for_zenodo's window and
MIN_MATCHED_EDGES) run first; this is the human pass on top of them. The panel
(curation_map.py, scripts/curate_release.py) reads the resulting package and
exports a JSON file that scripts/apply_curation.py applies to it:

    {"version": 1,
     "walks": {"OM_1_20251201_afternoon": {"decision": "keep" | "discard",
                                           "trim_start": "2025-12-01 15:31:00",
                                           "trim_end":   "2025-12-01 15:40:12",
                                           "cuts": [["2025-12-01 15:34:00",
                                                     "2025-12-01 15:35:30"]]}}}

A walk is keyed by (track, date, period) because the filename's trailing
duration changes when a walk is trimmed. A walk without an entry is
unreviewed and ships unchanged. `trim_start`/`trim_end` are inclusive, in the
shipped (local, UTC-3) timestamp format, and either may be absent for a
one-sided trim. `cuts` (optional) removes inclusive [start, end] ranges from
the middle of the walk, in the same format. Rows are removed, never moved: the
remaining rows keep their recorded timestamps, so a cut leaves a time jump in
the shipped file. A cut that overlaps the head/tail trim is harmless.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .recording_naming import _TRIP_FILENAME_RE

CURATION_VERSION = 1
DECISIONS = ("keep", "discard")
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"  # same as zenodo_columns.to_local_timestamp
_KEY_RE = re.compile(r"^OM_\d+_\d{8}_(morning|afternoon)$")


def walk_key(filename: str) -> str | None:
    """'OM_1_20251201_afternoon' for 'OM_1_20251201_afternoon_10durmin[.csv]'; None if unparseable."""
    stem = filename[:-4] if filename.endswith(".csv") else filename
    m = _TRIP_FILENAME_RE.match(stem)
    return f"{m['track']}_{m['date']}_{m['period_word']}" if m else None


def _check_timestamp(key: str, field: str, value: object) -> None:
    try:
        parsed = pd.to_datetime(value, format=_TIMESTAMP_FORMAT)
    except (ValueError, TypeError):
        raise ValueError(f"{key}: {field}={value!r} is not 'YYYY-MM-DD HH:MM:SS'") from None
    if parsed.strftime(_TIMESTAMP_FORMAT) != value:
        raise ValueError(f"{key}: {field}={value!r} is not 'YYYY-MM-DD HH:MM:SS'")


def validate_curation(data: dict) -> dict[str, dict]:
    """The `walks` mapping of a curation document, or ValueError naming the first problem."""
    if data.get("version") != CURATION_VERSION:
        raise ValueError(f"unsupported curation version {data.get('version')!r} (expected {CURATION_VERSION})")
    walks = data.get("walks")
    if not isinstance(walks, dict):
        raise ValueError("curation file has no 'walks' object")
    for key, entry in walks.items():
        if not _KEY_RE.match(key):
            raise ValueError(f"{key!r} is not a walk key like 'OM_1_20251201_afternoon'")
        if entry.get("decision") not in DECISIONS:
            raise ValueError(f"{key}: decision must be one of {DECISIONS}, got {entry.get('decision')!r}")
        for field in ("trim_start", "trim_end"):
            if entry.get(field) is not None:
                _check_timestamp(key, field, entry[field])
        if entry.get("trim_start") and entry.get("trim_end") and entry["trim_start"] > entry["trim_end"]:
            raise ValueError(f"{key}: trim_start is after trim_end")
        _check_cuts(key, entry.get("cuts"))
    return walks


def _check_cuts(key: str, cuts: object) -> None:
    if cuts is None:
        return
    if not isinstance(cuts, list):
        raise ValueError(f"{key}: cuts must be a list of [start, end] pairs")
    for cut in cuts:
        if not isinstance(cut, list) or len(cut) != 2:
            raise ValueError(f"{key}: each cut must be a [start, end] pair, got {cut!r}")
        _check_timestamp(key, "cuts start", cut[0])
        _check_timestamp(key, "cuts end", cut[1])
        if cut[0] > cut[1]:
            raise ValueError(f"{key}: cut start {cut[0]} is after its end {cut[1]}")


def load_curation(path: Path | str) -> dict[str, dict]:
    """walk key -> entry from a curation file; {} when the file doesn't exist."""
    path = Path(path)
    if not path.is_file():
        return {}
    return validate_curation(json.loads(path.read_text()))


def is_discarded(entry: dict | None) -> bool:
    return bool(entry) and entry["decision"] == "discard"


def has_trim(entry: dict | None) -> bool:
    return bool(entry) and (entry.get("trim_start") is not None or entry.get("trim_end") is not None)


def has_cuts(entry: dict | None) -> bool:
    return bool(entry) and bool(entry.get("cuts"))


def has_edit(entry: dict | None) -> bool:
    """True if the entry removes any row: a head/tail trim and/or middle cuts."""
    return has_trim(entry) or has_cuts(entry)


def trim_mask(local_timestamps: pd.Series, entry: dict | None) -> pd.Series:
    """True for rows that survive the entry's trim and cuts (all True without either).

    A row survives if it is inside [trim_start, trim_end] and outside every
    inclusive cut range. local_timestamps are the shipped 'YYYY-MM-DD HH:MM:SS'
    strings, which sort chronologically as plain strings, so no parsing is
    needed. A row with no timestamp (empty) does not survive any edit.
    """
    keep = pd.Series(True, index=local_timestamps.index)
    if not has_edit(entry):
        return keep
    if entry.get("trim_start") is not None:
        keep &= local_timestamps >= entry["trim_start"]
    if entry.get("trim_end") is not None:
        keep &= local_timestamps <= entry["trim_end"]
    for start, end in entry.get("cuts") or []:
        keep &= ~((local_timestamps >= start) & (local_timestamps <= end))
    return keep & local_timestamps.notna()
