"""Tests for cleaning.py."""

import pandas as pd
import pytest

from gps_route_matcher.cleaning import clean_gps_dataframe
from gps_route_matcher.config import MatchConfig


def _make_df(rows):
    return pd.DataFrame(rows, columns=["timestamp", "lon", "lat"])


def test_zero_zero_marked_invalid():
    df = _make_df([
        {"timestamp": "2024-01-01 00:00:00", "lon": 0.0, "lat": 0.0},
    ])
    cfg = MatchConfig(treat_zero_zero_as_missing=True)
    out = clean_gps_dataframe(df, cfg)
    assert out["gps_valid"].iloc[0] is False or out["gps_valid"].iloc[0] == False


def test_zero_zero_valid_when_disabled():
    df = _make_df([
        {"timestamp": "2024-01-01 00:00:00", "lon": 0.0, "lat": 0.0},
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
        {"timestamp": "2024-01-01 00:00:00", "lon": 10.0, "lat": 20.0},
    ])
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert bool(out["usable_for_hmm"].iloc[0]) is True


def test_no_rows_dropped():
    df = _make_df([
        {"timestamp": "N/A", "lon": 0.0, "lat": 0.0},
        {"timestamp": "2024-01-01 00:00:00", "lon": 10.0, "lat": 20.0},
        {"timestamp": "2024-01-01 00:00:05", "lon": 0.0, "lat": 0.0},
    ])
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert len(out) == 3


def test_original_index_preserved():
    df = _make_df([
        {"timestamp": "2024-01-01 00:00:00", "lon": 1.0, "lat": 2.0},
        {"timestamp": "N/A", "lon": 0.0, "lat": 0.0},
    ])
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert list(out["original_index"]) == [0, 1]


def test_raw_columns_preserved():
    df = _make_df([
        {"timestamp": "2024-01-01", "lon": "10.5", "lat": "20.5"},
    ])
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert "timestamp_raw" in out.columns
    assert "lon_raw" in out.columns
    assert "lat_raw" in out.columns


def test_mixed_valid_invalid():
    df = _make_df([
        {"timestamp": "2024-01-01 00:00:00", "lon": 10.0, "lat": 20.0},  # valid
        {"timestamp": "N/A", "lon": 0.0, "lat": 0.0},                    # invalid
        {"timestamp": "2024-01-01 00:00:10", "lon": 0.0, "lat": 0.0},   # bad GPS
        {"timestamp": "2024-01-01 00:00:15", "lon": 11.0, "lat": 21.0}, # valid
    ])
    cfg = MatchConfig()
    out = clean_gps_dataframe(df, cfg)
    assert list(out["usable_for_hmm"]) == [True, False, False, True]
