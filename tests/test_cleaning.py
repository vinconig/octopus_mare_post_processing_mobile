"""Tests for cleaning.py."""

import pandas as pd
import pytest

from gps_route_matcher.cleaning import clean_gps_dataframe, infer_missing_timestamps
from gps_route_matcher.config import MatchConfig


def _make_df(rows):
    """Build a test dataframe using MatchConfig's default column names."""
    df = pd.DataFrame(rows, columns=["timestamp", "lon", "lat"])
    return df.rename(
        columns={"timestamp": "Timestamp", "lon": "Longitude", "lat": "Latitude"}
    )


def test_zero_zero_marked_invalid():
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": 0.0, "lat": 0.0},
    ])
    cfg = MatchConfig(treat_zero_zero_as_missing=True)
    out = clean_gps_dataframe(df, cfg)
    assert out["gps_valid"].iloc[0] is False or out["gps_valid"].iloc[0] == False


def test_zero_zero_valid_when_disabled():
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": 0.0, "lat": 0.0},
    ])
    cfg = MatchConfig(treat_zero_zero_as_missing=False)
    out = clean_gps_dataframe(df, cfg)
    assert bool(out["gps_valid"].iloc[0]) is True


def test_na_timestamp_invalid():
    df = _make_df([
        {"timestamp": "N/A", "lon": 10.0, "lat": 20.0},
        {"timestamp": None, "lon": 10.0, "lat": 20.0},
    ])
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert bool(out["timestamp_valid"].iloc[0]) is False
    assert bool(out["timestamp_valid"].iloc[1]) is False


def test_valid_row_usable_for_hmm():
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": 10.0, "lat": 20.0},
    ])
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert bool(out["usable_for_hmm"].iloc[0]) is True


def test_no_rows_dropped():
    df = _make_df([
        {"timestamp": "N/A", "lon": 0.0, "lat": 0.0},
        {"timestamp": "2026-01-01 00:00:00", "lon": 10.0, "lat": 20.0},
        {"timestamp": "2026-01-01 00:00:05", "lon": 0.0, "lat": 0.0},
    ])
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert len(out) == 3


def test_original_index_preserved():
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": 1.0, "lat": 2.0},
        {"timestamp": "N/A", "lon": 0.0, "lat": 0.0},
    ])
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert list(out["original_index"]) == [0, 1]


def test_raw_columns_preserved():
    df = _make_df([
        {"timestamp": "2026-01-01", "lon": "10.5", "lat": "20.5"},
    ])
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert "timestamp_raw" in out.columns
    assert "lon_raw" in out.columns
    assert "lat_raw" in out.columns


def test_mixed_valid_invalid():
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": 10.0, "lat": 20.0},  # valid
        {"timestamp": "N/A", "lon": 0.0, "lat": 0.0},                    # invalid
        {"timestamp": "2026-01-01 00:00:10", "lon": 0.0, "lat": 0.0},   # bad GPS
        {"timestamp": "2026-01-01 00:00:15", "lon": 11.0, "lat": 21.0}, # valid
    ])
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert list(out["usable_for_hmm"]) == [True, False, False, True]


def _sequential_rows(start, n, step_s=5, lon=10.0, lat=20.0):
    return [
        {
            "timestamp": (start + pd.Timedelta(seconds=i * step_s)).isoformat(),
            "lon": lon,
            "lat": lat,
        }
        for i in range(n)
    ]


def test_isolated_bad_timestamp_flagged_invalid():
    rows = _sequential_rows(pd.Timestamp("2026-03-09 12:00:00", tz="UTC"), 10)
    rows[5]["timestamp"] = pd.Timestamp("1970-01-01", tz="UTC").isoformat()
    df = _make_df(rows)
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert bool(out["timestamp_valid"].iloc[5]) is False
    assert pd.isna(out["timestamp"].iloc[5])
    assert list(out["timestamp_valid"].drop(index=5)) == [True] * 9


def test_contiguous_corrupted_burst_flagged_invalid():
    # Mirrors the real OM_2_20260309_...REC000044.csv case: a run of rows
    # that is internally 5s-spaced and self-consistent, but anchored to the
    # wrong decade. Local dt alone would not catch this.
    good_before = _sequential_rows(pd.Timestamp("2026-03-09 12:00:00", tz="UTC"), 5)
    burst = _sequential_rows(pd.Timestamp("2015-10-19 12:23:23", tz="UTC"), 4)
    good_after = _sequential_rows(pd.Timestamp("2026-03-09 12:00:25", tz="UTC"), 5)
    df = _make_df(good_before + burst + good_after)
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert list(out["timestamp_valid"].iloc[5:9]) == [False] * 4
    assert list(out["timestamp_valid"].iloc[0:5]) == [True] * 5
    assert list(out["timestamp_valid"].iloc[9:14]) == [True] * 5


def test_legitimate_gap_not_flagged():
    cluster_a = _sequential_rows(pd.Timestamp("2026-03-09 12:00:00", tz="UTC"), 10)
    cluster_b = _sequential_rows(
        pd.Timestamp("2026-03-09 12:25:00", tz="UTC"), 10
    )  # ~25 min real gap, well within Stage-1 tolerance
    df = _make_df(cluster_a + cluster_b)
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert out["timestamp_valid"].all()
    assert out["usable_for_hmm"].all()


def test_insufficient_data_skips_sequentiality_check():
    rows = _sequential_rows(pd.Timestamp("2026-03-09 12:00:00", tz="UTC"), 3)
    rows[1]["timestamp"] = pd.Timestamp("1970-01-01", tz="UTC").isoformat()
    df = _make_df(rows)
    cfg = MatchConfig(min_timestamps_for_sequentiality_check=5)
    out = clean_gps_dataframe(df, cfg)
    # Below the minimum sample size, the statistical/monotonicity stages are
    # skipped entirely -- only the absolute capture-window bound still runs
    # and flags the 1970 row.
    assert bool(out["timestamp_valid"].iloc[1]) is False
    assert bool(out["timestamp_valid"].iloc[0]) is True
    assert bool(out["timestamp_valid"].iloc[2]) is True


def test_detect_timestamp_outliers_can_be_disabled():
    rows = _sequential_rows(pd.Timestamp("2026-03-09 12:00:00", tz="UTC"), 10)
    rows[5]["timestamp"] = pd.Timestamp("1970-01-01", tz="UTC").isoformat()
    df = _make_df(rows)
    cfg = MatchConfig(detect_timestamp_outliers=False)
    out = clean_gps_dataframe(df, cfg)
    assert bool(out["timestamp_valid"].iloc[5]) is True


def test_flagged_row_backfilled_by_infer_missing_timestamps():
    rows = _sequential_rows(pd.Timestamp("2026-03-09 12:00:00", tz="UTC"), 10)
    rows[5]["timestamp"] = pd.Timestamp("1970-01-01", tz="UTC").isoformat()
    df = _make_df(rows)
    cfg = MatchConfig(infer_missing_timestamps=True)
    cleaned = clean_gps_dataframe(df, cfg)
    assert bool(cleaned["timestamp_valid"].iloc[5]) is False

    inferred = infer_missing_timestamps(cleaned, cfg)
    assert bool(inferred["timestamp_valid"].iloc[5]) is True
    assert bool(inferred["timestamp_inferred"].iloc[5]) is True
    expected = pd.Timestamp("2026-03-09 12:00:25", tz="UTC")
    assert inferred["timestamp"].iloc[5] == expected
