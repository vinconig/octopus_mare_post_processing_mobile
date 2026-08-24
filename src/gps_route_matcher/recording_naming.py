"""Shared parsing for recording filename conventions, used by both
visualization.py (to browse matched/edge-analysis recordings) and
analysis.py (to tag recordings when building a cross-recording DataFrame).
Kept dependency-light (no plotly/geopandas) so analysis.py doesn't have to
pull in visualization.py's heavy plotting stack just to reuse this regex.

Two conventions are supported:
  TRACK_YYYYMMDD_HR_DURATIONdurmin__RECrecid.csv     (data/OM_<n>_a_raw, the
      original per-recording exports; period is derived from the hour)
  TRACK_YYYYMMDD_<morning|evening>_DURATIONdurmin.csv (data/OM_<n>_c_prepared,
      one CSV per trip already; period is named directly, no recid)
"""

from __future__ import annotations

import re

import pandas as pd

# e.g. "OM_1_20251009_16_15durmin__REC000135.csv" → track=OM_1,
# date=20251009, hour=16, duration=15 (min), recid=135.
_RECORDING_FILENAME_RE = re.compile(
    r"^(?P<track>OM_\d+)_(?P<date>\d{8})_(?P<hour>\d{1,2})_(?P<duration>\d+)durmin__REC(?P<recid>\d+)$"
)

# e.g. "OM_1_20151019_evening_5durmin.csv" → track=OM_1, date=20151019,
# period_word=evening, duration=5 (min). No recid: each (track, date, period)
# has at most one file in data/OM_<n>_c_prepared.
_TRIP_FILENAME_RE = re.compile(
    r"^(?P<track>OM_\d+)_(?P<date>\d{8})_(?P<period_word>morning|evening)_(?P<duration>\d+)durmin$"
)
_PERIOD_WORD_TO_PERIOD = {"morning": "manha", "evening": "tarde"}

# A handful of source filenames carry a garbage date (bad device clock at
# recording time, e.g. year 2015 or 2027) that has nothing to do with GPS
# timestamp validity but still sorts to the very start/end of the calendar's
# date list, hijacking its default view.
#
# This bound is deliberately its own constant rather than MatchConfig's
# valid_timestamp_min/max, which it used to borrow. Those bound *GPS
# timestamps* and are now only a loose backstop behind the per-recording
# filename anchor (see cleaning.flag_timestamp_outliers), so widening them
# would silently repopulate the calendar with bad-clock recordings. Two
# separate questions, two separate constants.
_RECORDING_DATE_MIN = "20250901"
_RECORDING_DATE_MAX = "20301231"


def parse_capture_time(stem: str) -> pd.Timestamp | None:
    """The capture time a raw recording's filename declares (UTC), or None if
    the stem isn't a raw per-recording export.

    Used by sessions.clean_recording as the absolute anchor for that file's
    timestamps. It is a better anchor than any fixed calendar window: the
    device writes this name when it closes the file, by which point its clock
    has synced, whereas the rows it rejects were logged before the sync — so
    the two disagree precisely in the cases worth catching. 806 of the 807 raw
    filenames present parse to a plausible time.

    Resolution is one hour (the filename carries no minutes), so callers must
    compare with a tolerance rather than for equality.
    """
    m = _RECORDING_FILENAME_RE.match(stem)
    if not m:
        return None
    try:
        return pd.Timestamp(
            f"{m['date']} {int(m['hour']):02d}:00:00", tz="UTC"
        )
    except ValueError:
        return None


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


def _date_out_of_range(date_str: str) -> bool:
    """True if date_str falls outside MatchConfig's plausible capture window
    (see _RECORDING_DATE_MIN/_MAX above) — a bad device clock at recording
    time, unrelated to GPS timestamp validity."""
    if _RECORDING_DATE_MIN is not None and date_str < _RECORDING_DATE_MIN:
        return True
    if _RECORDING_DATE_MAX is not None and date_str > _RECORDING_DATE_MAX:
        return True
    return False


def _parse_recording_filename(stem: str) -> dict | None:
    """Parse a matched-recording filename stem, or None if it doesn't match
    either naming convention (see module docstring), its hour/period falls
    outside the manhã/tarde windows, or its date falls outside MatchConfig's
    plausible capture window (see _date_out_of_range)."""
    m = _RECORDING_FILENAME_RE.match(stem)
    if m:
        if _date_out_of_range(m["date"]):
            return None
        period = _period_for_hour(int(m["hour"]))
        if period is None:
            return None
        return {
            "track": m["track"],
            "date": m["date"],
            "period": period,
            "recid": int(m["recid"]),
            "hour": int(m["hour"]),
        }

    m = _TRIP_FILENAME_RE.match(stem)
    if m:
        if _date_out_of_range(m["date"]):
            return None
        return {
            "track": m["track"],
            "date": m["date"],
            "period": _PERIOD_WORD_TO_PERIOD[m["period_word"]],
            "recid": 0,
            "hour": None,
        }

    return None
