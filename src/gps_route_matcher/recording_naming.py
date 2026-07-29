"""Shared parsing for the TRACK_YYYYMMDD_HR_DURATIONdurmin__RECrecid.csv
recording filename convention, used by both visualization.py (to browse
matched/edge-analysis recordings) and analysis.py (to tag recordings when
building a cross-recording DataFrame). Kept dependency-light (no plotly/
geopandas) so analysis.py doesn't have to pull in visualization.py's heavy
plotting stack just to reuse this regex.
"""

from __future__ import annotations

import re

import pandas as pd

from .config import MatchConfig

# Filenames follow TRACK_YYYYMMDD_HR_DURATIONdurmin__RECrecid.csv, e.g.
# "OM_1_20251009_16_15durmin__REC000135.csv" → track=OM_1, date=20251009,
# hour=16, duration=15 (min), recid=135.
_RECORDING_FILENAME_RE = re.compile(
    r"^(?P<track>OM_\d+)_(?P<date>\d{8})_(?P<hour>\d{1,2})_(?P<duration>\d+)durmin__REC(?P<recid>\d+)$"
)

# A handful of source filenames carry a garbage date (bad device clock at
# recording time, e.g. year 2015 or 2027) that has nothing to do with GPS
# timestamp validity but still sorts to the very start/end of the calendar's
# date list, hijacking its default view. Reuse MatchConfig's own plausible
# capture window (already used to bound GPS timestamps, see
# cleaning._absolute_range_mask) as the same bound for the filename date.
_RECORDING_DATE_MIN = (
    pd.Timestamp(MatchConfig().valid_timestamp_min).strftime("%Y%m%d")
    if MatchConfig().valid_timestamp_min
    else None
)
_RECORDING_DATE_MAX = (
    pd.Timestamp(MatchConfig().valid_timestamp_max).strftime("%Y%m%d")
    if MatchConfig().valid_timestamp_max
    else None
)


def _period_for_hour(hour: int) -> str | None:
    """Classify a recording's start hour into Manhã (10-14h) / Tarde (15-20h).

    Hours outside both windows return None: they have no slot in the
    calendar's morning/afternoon toggle and are excluded from browsing.
    """
    if 10 <= hour <= 14:
        return "manha"
    if 15 <= hour <= 20:
        return "tarde"
    return None


def _parse_recording_filename(stem: str) -> dict | None:
    """Parse a matched-recording filename stem, or None if it doesn't match
    the naming convention, its hour falls outside the manhã/tarde windows, or
    its date falls outside MatchConfig's plausible capture window (see
    _RECORDING_DATE_MIN/_MAX above)."""
    m = _RECORDING_FILENAME_RE.match(stem)
    if not m:
        return None
    date_str = m["date"]
    if _RECORDING_DATE_MIN is not None and date_str < _RECORDING_DATE_MIN:
        return None
    if _RECORDING_DATE_MAX is not None and date_str > _RECORDING_DATE_MAX:
        return None
    period = _period_for_hour(int(m["hour"]))
    if period is None:
        return None
    return {
        "track": m["track"],
        "date": m["date"],
        "period": period,
        "recid": int(m["recid"]),
    }
