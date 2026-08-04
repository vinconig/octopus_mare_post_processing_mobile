"""Recording-scope and sensor-scope cleaning: everything that has to happen
before rows from different recordings are pooled.

Two scopes live here, both upstream of the per-trip cleaning in cleaning.py:

  S1, one raw recording (clean_recording)
      A raw CSV is one continuous logging session by one device, with rows in
      device emit order. That is the only place three of the timestamp checks
      have their evidence:
        * the filename declares the session's true capture hour, so it can
          anchor an absolute check that needs no hand-maintained constant;
        * the median/MAD spread is one session's, not several merged walks';
        * forward monotonicity is only a question while the rows are still in
          emit order — after any sort by timestamp it cannot fire.
      Missing timestamps are also filled here, in that same emit order, which
      is what makes positional bracketing meaningful.

  S2, one sensor's history (dedupe_rows / sessionize / label_sessions)
      Duplicate recordings and overlapping restarts are only visible once a
      sensor's files are pooled. And a *walk* is a contiguous run of samples,
      so trips are cut on silence and only then labelled morning/evening —
      labelling row by row instead splits any walk that crosses a window edge.

Everything here writes flags and never drops a row; scripts/make_trip_data.py
decides what to keep and quarantines the rest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import (
    TS_CLEAN_COL,
    TS_INFERRED_COL,
    TS_REASON_COL,
    TS_VALID_COL,
    flag_timestamp_outliers,
    infer_missing_timestamps,
)
from .config import MatchConfig
from .recording_naming import parse_capture_time

SESSION_ID_COL = "session_id"
SESSION_PERIOD_COL = "session_period"
SESSION_REJECT_COL = "session_reject_reason"

# Columns this module adds. Registered in analysis._NON_ENV_COLUMNS so they're
# never mistaken for environmental sensor readings.
SESSION_COLUMNS = (
    TS_CLEAN_COL,
    TS_VALID_COL,
    TS_INFERRED_COL,
    TS_REASON_COL,
    SESSION_ID_COL,
    SESSION_PERIOD_COL,
    SESSION_REJECT_COL,
)


def clean_recording(
    df: pd.DataFrame, stem: str, config: MatchConfig
) -> pd.DataFrame:
    """Validate and back-fill one raw recording's timestamps (S1).

    `stem` is the file's name without extension; when it follows the raw
    recording convention its declared capture hour anchors the absolute check
    (see recording_naming.parse_capture_time).

    Never drops a row and never touches the original Timestamp text, so the
    concatenated file stays a faithful union of the raw rows. Adds:

        ts_clean         : parsed pd.Timestamp (UTC), NaT where rejected and
                           not recoverable.
        ts_valid         : bool – ts_clean is usable (parsed or inferred).
        ts_inferred      : bool – ts_clean was back-filled, not read.
        ts_reject_reason : str | NA – why the raw reading was rejected, kept
                           even for rows later back-filled.
    """
    df = df.copy()

    parsed = pd.to_datetime(
        df[config.timestamp_col], errors="coerce", utc=True, format="mixed"
    )
    reason = flag_timestamp_outliers(parsed, config, anchor=parse_capture_time(stem))

    # infer_missing_timestamps works on the matcher's column names and
    # requires acquisition order, which is exactly what a raw file is in.
    work = pd.DataFrame(
        {
            "timestamp": parsed.mask(reason.notna()),
            "gps_valid": True,  # this pass is about time only
        }
    )
    work["timestamp_valid"] = work["timestamp"].notna()
    work["timestamp_inferred"] = False
    work = infer_missing_timestamps(work, config)

    df[TS_CLEAN_COL] = work["timestamp"]
    df[TS_VALID_COL] = work["timestamp_valid"]
    df[TS_INFERRED_COL] = work["timestamp_inferred"]
    df[TS_REASON_COL] = reason
    return df


def dedupe_rows(df: pd.DataFrame, config: MatchConfig) -> tuple[pd.DataFrame, int]:
    """Drop rows that repeat an earlier row of the same sensor (S2).

    Only visible once a sensor's recordings are pooled: the same session
    exported twice, or a device restart that re-logs an interval it had
    already written. Duplicates would otherwise double-count in
    route_edge_analysis's per-edge means and inflate the Viterbi candidate set.

    Identity is (cleaned timestamp, position, sensor payload) — deliberately
    not source_file, which differs precisely in the re-export case this is
    meant to catch. Returns the deduplicated frame and how many rows went.
    """
    key_cols = [
        c
        for c in df.columns
        if c not in {"source_file", *SESSION_COLUMNS}
    ]
    if not key_cols:
        return df, 0
    keep = ~df.duplicated(subset=[TS_CLEAN_COL, *key_cols], keep="first")
    # Rows with no usable timestamp are all-NaT in the key and would collapse
    # into each other; they are quarantined later anyway, so leave them be.
    keep |= df[TS_CLEAN_COL].isna()

    
    return df.loc[keep].reset_index(drop=True), int((~keep).sum())


def _idle_mask(
    ts: np.ndarray, lon: np.ndarray, lat: np.ndarray, config: MatchConfig
) -> np.ndarray:
    """True for samples that are idle time rather than walk data.

    A run of samples that never leaves config.stationary_radius_m of where it
    started, and lasts longer than config.max_stationary_gap_s, is the device
    sitting somewhere — the sensor left running between the morning and the
    evening walk, most often. The row that ends such a run is the first step
    of the next walk and is kept; the run itself is not walk data.

    Samples with no usable fix carry no movement information, so they neither
    end a run nor start one; they inherit the verdict of the run they fall in.
    A stretch with no fix at all is idle by this rule, which is the right
    answer either way: those rows cannot be map-matched.
    """
    n = len(ts)
    idle = np.zeros(n, dtype=bool)
    if n == 0:
        return idle

    has_fix = np.isfinite(lon) & np.isfinite(lat) & ~((lon == 0.0) & (lat == 0.0))
    # Equirectangular metres — exact enough over a bairro, and this only ever
    # feeds a threshold comparison.
    lat_scale = 111_320.0
    lon_scale = lat_scale * np.cos(np.radians(np.nanmean(lat[has_fix]))) if has_fix.any() else lat_scale

    def _close_run(start: int, end: int) -> None:
        """Mark [start+1, end) idle when the run lasted too long."""
        if end - start < 2:
            return
        if (ts[end - 1] - ts[start]) / np.timedelta64(1, "s") > config.max_stationary_gap_s:
            idle[start + 1 : end] = True

    anchor = int(np.argmax(has_fix)) if has_fix.any() else 0
    _close_run(0, anchor)  # a leading block with no fix at all
    for i in range(anchor + 1, n):
        if not has_fix[i]:
            continue
        moved_m = np.hypot(
            (lon[i] - lon[anchor]) * lon_scale, (lat[i] - lat[anchor]) * lat_scale
        )
        if moved_m > config.stationary_radius_m:
            _close_run(anchor, i)
            anchor = i
    _close_run(anchor, n)
    return idle


def sessionize(df: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    """Cut a sensor's pooled rows into sessions (S2).

    A session is one walk: a contiguous run of samples that is also in motion.
    It ends either when logging stops for longer than
    config.session_split_gap_s, or when the walking does (see _idle_mask) —
    both are needed, because a device left running between two walks goes
    still without ever going quiet.

    Rows with no usable timestamp, and rows that fall in idle time, get
    session_id = NA and are quarantined by the caller.

    Returns the frame sorted by ts_clean with SESSION_ID_COL added.
    """
    df = df.sort_values(TS_CLEAN_COL, kind="stable").reset_index(drop=True)
    df[SESSION_ID_COL] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    timed = df.index[df[TS_VALID_COL].fillna(False).astype(bool) & df[TS_CLEAN_COL].notna()]
    if len(timed) == 0:
        return df

    ts = pd.to_datetime(df.loc[timed, TS_CLEAN_COL].values)
    idle = _idle_mask(
        ts,
        pd.to_numeric(df.loc[timed, config.lon_col], errors="coerce").to_numpy(float),
        pd.to_numeric(df.loc[timed, config.lat_col], errors="coerce").to_numpy(float),
        config,
    )

    dt_s = np.full(len(timed), np.nan)
    dt_s[1:] = (ts[1:] - ts[:-1]) / np.timedelta64(1, "s")
    # A new session starts on a long silence, and on the first sample after an
    # idle stretch. Idle samples themselves belong to no session.
    starts = (dt_s > config.session_split_gap_s) | (~idle & np.r_[False, idle[:-1]])
    session_ids = np.cumsum(starts)
    df.loc[timed[~idle], SESSION_ID_COL] = session_ids[~idle]
    return df


def label_sessions(df: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    """Label each whole session morning / evening (S2).

    The classification is per session, not per row: a session belongs to the
    window it overlaps most, and to neither if it overlaps neither.

    Both parts matter, because walks straddle a window edge in both
    directions. 13:44-14:58 is a morning walk that ran an hour long; testing
    its rows individually against [11, 14) keeps the first 15 minutes and
    discards the other 59. 16:37-17:28 is an evening walk that set off early;
    labelling the session by its *start* hour instead would throw the whole
    thing away. Counting overlap gets both right, while a session recorded
    entirely outside both windows still matches neither and is quarantined by
    the caller.

    A session longer than config.max_session_duration_s isn't a walk at all
    and is rejected outright, whichever windows it happens to span.

    Adds SESSION_PERIOD_COL and, for sessions that get no period,
    SESSION_REJECT_COL explaining which test they failed.
    """
    df = df.copy()
    df[SESSION_PERIOD_COL] = pd.Series(pd.NA, index=df.index, dtype="object")
    df[SESSION_REJECT_COL] = pd.Series(pd.NA, index=df.index, dtype="object")

    timed = df.dropna(subset=[SESSION_ID_COL])
    if timed.empty:
        return df

    hours = timed[TS_CLEAN_COL].dt.hour
    per_session = pd.DataFrame(
        {
            "session": timed[SESSION_ID_COL],
            "morning": hours.between(config.morning_hours[0], config.morning_hours[1] - 1),
            "evening": hours.between(config.evening_hours[0], config.evening_hours[1] - 1),
            "ts": timed[TS_CLEAN_COL],
        }
    )
    overlap = per_session.groupby("session").agg(
        morning=("morning", "sum"),
        evening=("evening", "sum"),
        first=("ts", "min"),
        last=("ts", "max"),
    )

    for session_id, row in overlap.iterrows():
        rows = df[SESSION_ID_COL] == session_id
        duration_s = (row["last"] - row["first"]).total_seconds()
        if duration_s > config.max_session_duration_s:
            df.loc[rows, SESSION_REJECT_COL] = "session_too_long"
        elif row["morning"] == 0 and row["evening"] == 0:
            df.loc[rows, SESSION_REJECT_COL] = "session_off_window"
        else:
            df.loc[rows, SESSION_PERIOD_COL] = (
                "morning" if row["morning"] >= row["evening"] else "evening"
            )
    return df
