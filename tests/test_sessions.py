"""Tests for sessions.py — the per-recording and per-sensor cleaning that has
to happen before rows from different recordings are pooled."""

import pandas as pd

from gps_route_matcher.cleaning import (
    TS_CLEAN_COL,
    TS_DUPLICATE_TIMESTAMP,
    TS_FILENAME_MISMATCH,
    TS_INFERRED_COL,
    TS_REASON_COL,
    TS_UNPARSEABLE,
    TS_VALID_COL,
)
from gps_route_matcher.config import MatchConfig
from gps_route_matcher.sessions import (
    SESSION_ID_COL,
    SESSION_PERIOD_COL,
    SESSION_REJECT_COL,
    clean_recording,
    dedupe_rows,
    label_sessions,
    sessionize,
)

# Matches the raw per-recording convention, so its date/hour anchors the file.
STEM = "OM_2_20260309_12_16durmin__REC000044"


# ~7 m per 5 s sample: walking pace, and comfortably past
# MatchConfig.stationary_radius_m so a fixture reads as a walk rather than as
# idle time. Pass step_m=0 for a device that is sitting still.
_M_PER_DEG_LAT = 111_320.0


def _rows(start, n, step_s=5, lon=-43.24, lat=-22.85, step_m=7.0):
    return [
        {
            "Timestamp": (start + pd.Timedelta(seconds=i * step_s)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Longitude": lon,
            "Latitude": lat + i * step_m / _M_PER_DEG_LAT,
        }
        for i in range(n)
    ]


def _df(rows):
    return pd.DataFrame(rows, columns=["Timestamp", "Longitude", "Latitude"])


# ── S1: clean_recording ──────────────────────────────────────────────────────


def test_filename_anchor_rejects_pre_sync_warmup_block():
    # The real OM_2_20260309_..._REC000044 shape: the device logs a warm-up
    # prefix stamped with its un-synced RTC default (2015) before the clock
    # catches up. The filename is written at export time, after the sync, so
    # it disagrees with exactly those rows.
    warmup = _rows(pd.Timestamp("2015-10-19 12:23:23"), 6)
    real = _rows(pd.Timestamp("2026-03-09 12:00:00"), 10)
    out = clean_recording(_df(warmup + real), STEM, MatchConfig())

    assert list(out[TS_REASON_COL].iloc[:6]) == [TS_FILENAME_MISMATCH] * 6
    assert out[TS_REASON_COL].iloc[6:].isna().all()
    assert not out[TS_VALID_COL].iloc[:6].any()
    assert out[TS_VALID_COL].iloc[6:].all()


def test_monotonicity_fires_in_emit_order():
    # Only meaningful before any sort by timestamp: sorting by the very column
    # being checked would make this unfireable, which is what happened while
    # the check ran on already-time-sorted trip files.
    rows = _rows(pd.Timestamp("2026-03-09 12:00:00"), 8)
    rows[5]["Timestamp"] = "2026-03-09 12:00:02"  # jumps backwards mid-file
    out = clean_recording(_df(rows), STEM, MatchConfig())

    assert out[TS_REASON_COL].iloc[5] == "non_monotonic"
    assert out[TS_REASON_COL].drop(index=5).isna().all()


def test_duplicate_timestamp_run_flags_every_row_not_just_the_repeats():
    # A device clock/GPS freeze repeats one reading across several consecutive
    # rows while other channels keep updating. Nothing distinguishes which of
    # the frozen rows (if any) carries the genuine timestamp, so all of them
    # are flagged — not just the ones after the first.
    rows = _rows(pd.Timestamp("2026-03-09 12:00:00"), 8)
    rows[3]["Timestamp"] = rows[2]["Timestamp"]
    rows[4]["Timestamp"] = rows[2]["Timestamp"]
    out = clean_recording(_df(rows), STEM, MatchConfig())

    assert list(out[TS_REASON_COL].iloc[2:5]) == [TS_DUPLICATE_TIMESTAMP] * 3
    assert out[TS_REASON_COL].drop(index=[2, 3, 4]).isna().all()


def test_duplicate_timestamp_pair_flags_both_rows():
    rows = _rows(pd.Timestamp("2026-03-09 12:00:00"), 8)
    rows[5]["Timestamp"] = rows[4]["Timestamp"]
    out = clean_recording(_df(rows), STEM, MatchConfig())

    assert out[TS_REASON_COL].iloc[4] == TS_DUPLICATE_TIMESTAMP
    assert out[TS_REASON_COL].iloc[5] == TS_DUPLICATE_TIMESTAMP
    assert out[TS_REASON_COL].drop(index=[4, 5]).isna().all()


def test_duplicate_run_across_an_unparseable_row_is_still_flagged():
    # "Consecutive" means consecutive among survivors, not by row index: an
    # unparseable row physically between two equal timestamps must not hide
    # the duplicate from either side of it.
    rows = _rows(pd.Timestamp("2026-03-09 12:00:00"), 8)
    rows[4]["Timestamp"] = "0.00"
    rows[5]["Timestamp"] = rows[3]["Timestamp"]
    out = clean_recording(_df(rows), STEM, MatchConfig())

    assert out[TS_REASON_COL].iloc[3] == TS_DUPLICATE_TIMESTAMP
    assert out[TS_REASON_COL].iloc[4] == TS_UNPARSEABLE
    assert out[TS_REASON_COL].iloc[5] == TS_DUPLICATE_TIMESTAMP
    assert out[TS_REASON_COL].drop(index=[3, 4, 5]).isna().all()


def test_row_already_rejected_by_monotonicity_does_not_join_a_duplicate_run():
    # A genuine backwards jump is rejected by Stage 2 before Stage 3 runs, so
    # it must neither be pulled into a later duplicate run itself nor block
    # detection of one on either side of it.
    rows = _rows(pd.Timestamp("2026-03-09 12:00:00"), 9)
    rows[3]["Timestamp"] = "2026-03-09 12:00:02"  # jumps backwards
    rows[5]["Timestamp"] = rows[4]["Timestamp"]  # duplicate, unrelated to idx 3
    out = clean_recording(_df(rows), STEM, MatchConfig())

    assert out[TS_REASON_COL].iloc[3] == "non_monotonic"
    assert out[TS_REASON_COL].iloc[4] == TS_DUPLICATE_TIMESTAMP
    assert out[TS_REASON_COL].iloc[5] == TS_DUPLICATE_TIMESTAMP
    assert out[TS_REASON_COL].drop(index=[3, 4, 5]).isna().all()


def test_distinct_timestamps_are_unaffected_by_duplicate_check():
    rows = _rows(pd.Timestamp("2026-03-09 12:00:00"), 8)
    out = clean_recording(_df(rows), STEM, MatchConfig())
    assert out[TS_REASON_COL].isna().all()


def test_unparseable_timestamp_is_recorded_as_such():
    rows = _rows(pd.Timestamp("2026-03-09 12:00:00"), 8)
    rows[3]["Timestamp"] = "0.00"  # what the device writes before its RTC sets
    out = clean_recording(_df(rows), STEM, MatchConfig())
    assert out[TS_REASON_COL].iloc[3] == TS_UNPARSEABLE


def test_interior_dropout_is_backfilled_but_leading_one_is_not():
    # An interior gap is bracketed by valid timestamps, so its rows can be
    # placed in time and their sensor readings recovered. A leading block has
    # no earlier anchor; extrapolating it backwards would invent membership in
    # a walk that hadn't started.
    rows = _rows(pd.Timestamp("2026-03-09 12:00:00"), 10)
    rows[0]["Timestamp"] = "0.00"
    rows[1]["Timestamp"] = "0.00"
    rows[5]["Timestamp"] = "0.00"
    out = clean_recording(_df(rows), STEM, MatchConfig())

    assert bool(out[TS_VALID_COL].iloc[5]) is True
    assert bool(out[TS_INFERRED_COL].iloc[5]) is True
    assert out[TS_CLEAN_COL].iloc[5] == pd.Timestamp("2026-03-09 12:00:25", tz="UTC")
    assert not out[TS_VALID_COL].iloc[:2].any()
    assert out[TS_CLEAN_COL].iloc[:2].isna().all()
    # The reason the original reading was rejected survives the back-fill.
    assert out[TS_REASON_COL].iloc[5] == TS_UNPARSEABLE


def test_original_timestamp_column_is_never_rewritten():
    rows = _rows(pd.Timestamp("2015-10-19 12:23:23"), 6)
    out = clean_recording(_df(rows), STEM, MatchConfig())
    assert list(out["Timestamp"]) == [r["Timestamp"] for r in rows]
    assert len(out) == len(rows)


def test_unparseable_filename_falls_back_to_the_absolute_window():
    rows = _rows(pd.Timestamp("2015-10-19 12:23:23"), 6)
    out = clean_recording(_df(rows), "not-a-recording-name", MatchConfig())
    assert list(out[TS_REASON_COL]) == ["absolute_window"] * 6


def test_future_timestamps_are_rejected_without_a_maintained_constant():
    # The real OM_2_..._REC000124 shape: the clock was wrong at export time
    # too, so the filename agrees with the rows and cannot catch them. Only an
    # upper bound does — and it is derived from the current time, so it needs
    # no maintenance and cannot expire.
    future = pd.Timestamp.now() + pd.Timedelta(days=365)
    rows = _rows(future, 6)
    stem = f"OM_2_{future:%Y%m%d}_12_1durmin__REC000124"
    out = clean_recording(_df(rows), stem, MatchConfig())
    assert list(out[TS_REASON_COL]) == ["absolute_window"] * 6


def test_filename_is_ignored_when_the_files_own_centre_contradicts_it():
    # Some exports are named for the day they were downloaded, not recorded.
    # Rejecting every row of such a file would throw away a whole good
    # recording, so the anchor is only trusted when the bulk corroborates it.
    rows = _rows(pd.Timestamp("2025-12-23 12:29:24"), 20)
    out = clean_recording(_df(rows), "OM_3_20251226_12_17durmin__REC000143", MatchConfig())
    assert out[TS_REASON_COL].isna().all()
    assert out[TS_VALID_COL].all()


# ── S2: sessionize / label_sessions / dedupe_rows ────────────────────────────


def _cleaned(rows, stem=STEM, config=None):
    return clean_recording(_df(rows), stem, config or MatchConfig())


def test_sessionize_splits_on_silence_not_on_the_clock():
    cfg = MatchConfig(session_split_gap_s=1800.0)
    morning = _rows(pd.Timestamp("2026-03-09 12:00:00"), 5)
    evening = _rows(pd.Timestamp("2026-03-09 18:00:00"), 5)
    out = sessionize(_cleaned(morning + evening), cfg)
    assert list(out[SESSION_ID_COL]) == [0] * 5 + [1] * 5


def test_a_walk_running_past_a_window_edge_stays_one_morning_session():
    # The OM_3 13:44-14:58 case: a single 74-minute walk. Row-wise hour
    # filtering kept its first 15 minutes and discarded the other 59.
    cfg = MatchConfig(session_split_gap_s=1800.0)
    walk = _rows(pd.Timestamp("2026-03-09 13:44:00"), 900, step_s=5)  # 13:44-15:00
    out = label_sessions(sessionize(_cleaned(walk, stem="OM_3_20260309_13_74durmin__REC000001"), cfg), cfg)

    assert out[SESSION_ID_COL].nunique() == 1
    assert set(out[SESSION_PERIOD_COL]) == {"morning"}
    assert len(out) == 900


def test_a_walk_starting_before_a_window_edge_is_still_that_window():
    # The OM_1 16:37-17:28 case, the other straddle direction: an evening walk
    # that set off early. Labelling the session by its start hour would drop
    # the whole thing; by overlap it is correctly evening.
    cfg = MatchConfig(session_split_gap_s=1800.0)
    walk = _rows(pd.Timestamp("2026-03-09 16:37:00"), 620, step_s=5)  # 16:37-17:28
    out = label_sessions(sessionize(_cleaned(walk, stem="OM_1_20260309_16_51durmin__REC000135"), cfg), cfg)

    assert set(out[SESSION_PERIOD_COL]) == {"evening"}
    assert len(out) == 620


def test_a_session_too_long_to_be_a_walk_is_rejected():
    # The OM_2 33 h case: a sensor in transit moves continuously, so neither
    # silence nor stillness cuts it. Recorded walks top out around 75 min.
    cfg = MatchConfig(session_split_gap_s=1800.0, max_session_duration_s=10800.0)
    logged = _rows(pd.Timestamp("2026-03-09 12:00:00"), 5000, step_s=30)  # ~42 h
    out = label_sessions(
        sessionize(_cleaned(logged, stem="OM_2_20260309_12_2500durmin__REC000001"), cfg), cfg
    )
    assert out[SESSION_PERIOD_COL].isna().all()
    assert set(out[SESSION_REJECT_COL]) == {"session_too_long"}


def test_idle_time_between_two_walks_splits_them_without_a_logging_gap():
    # The real OM_2 2025-12-29 case: a morning walk, four hours of the device
    # sitting on a table still logging, then the evening walk. There is no
    # silence to cut on, so cutting on silence alone returns one 6.4 h
    # "session" that is neither walk.
    cfg = MatchConfig(session_split_gap_s=1800.0, max_stationary_gap_s=1200.0)
    morning = _rows(pd.Timestamp("2026-03-09 12:30:00"), 240)          # 20 min, moving
    idle = _rows(pd.Timestamp("2026-03-09 12:50:00"), 2400, step_m=0)  # 3 h20, still
    evening = _rows(pd.Timestamp("2026-03-09 18:20:00"), 240, lat=-22.80)
    out = label_sessions(sessionize(_cleaned(morning + idle + evening), cfg), cfg)

    walks = out.dropna(subset=[SESSION_ID_COL])
    assert walks[SESSION_ID_COL].nunique() == 2
    assert set(walks[SESSION_PERIOD_COL]) == {"morning", "evening"}
    # The idle block belongs to neither walk and is left for quarantine.
    assert out[SESSION_ID_COL].isna().sum() > 2000


def test_session_overlapping_neither_window_gets_no_label():
    cfg = MatchConfig(session_split_gap_s=1800.0)
    walk = _rows(pd.Timestamp("2026-03-09 15:30:00"), 10)
    out = label_sessions(
        sessionize(_cleaned(walk, stem="OM_3_20260309_15_5durmin__REC000002"), cfg), cfg
    )
    assert out[SESSION_PERIOD_COL].isna().all()


def test_rows_without_a_usable_timestamp_join_no_session():
    cfg = MatchConfig(session_split_gap_s=1800.0)
    rows = _rows(pd.Timestamp("2026-03-09 12:00:00"), 6)
    rows[0]["Timestamp"] = "0.00"  # leading, so never back-filled
    out = sessionize(_cleaned(rows), cfg)
    assert out[SESSION_ID_COL].isna().sum() == 1


def test_dedupe_drops_a_re_exported_recording():
    cfg = MatchConfig()
    once = _cleaned(_rows(pd.Timestamp("2026-03-09 12:00:00"), 5))
    once["source_file"] = "REC000044.csv"
    twice = once.copy()
    twice["source_file"] = "REC000044_copy.csv"  # differs only by source

    out, n_dropped = dedupe_rows(
        pd.concat([once, twice], ignore_index=True), cfg
    )
    assert n_dropped == 5
    assert len(out) == 5


def test_dedupe_keeps_distinct_readings_at_the_same_instant():
    cfg = MatchConfig()
    a = _cleaned(_rows(pd.Timestamp("2026-03-09 12:00:00"), 3, lon=-43.24))
    b = _cleaned(_rows(pd.Timestamp("2026-03-09 12:00:00"), 3, lon=-43.25))
    out, n_dropped = dedupe_rows(pd.concat([a, b], ignore_index=True), cfg)
    assert n_dropped == 0
    assert len(out) == 6
