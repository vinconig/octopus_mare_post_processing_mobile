"""Tests for gaps.py."""

import numpy as np
import pandas as pd

from gps_route_matcher.config import MatchConfig
from gps_route_matcher.gaps import assign_gaps_and_segments


def _frame(rows):
    """rows: (offset_seconds, usable). Timestamps count from a fixed base."""
    base = pd.Timestamp("2026-03-09 12:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(seconds=s) for s, _ in rows],
            "usable_for_hmm": [u for _, u in rows],
        }
    )


def test_dt_and_gap_id_only_count_usable_rows():
    cfg = MatchConfig(max_bridge_gap_s=90.0, max_split_gap_s=300.0)
    df = assign_gaps_and_segments(
        _frame([(0, True), (5, False), (10, True), (200, True)]), cfg
    )
    # dt is measured between consecutive *usable* observations, so the second
    # usable row is 10 s after the first, not 5 s after the unusable one.
    assert np.isnan(df["dt_s"].iloc[0])
    assert np.isnan(df["dt_s"].iloc[1])
    assert df["dt_s"].iloc[2] == 10.0
    assert df["dt_s"].iloc[3] == 190.0
    assert list(df["gap_before"]) == [False, False, False, True]
    assert list(df["gap_id"]) == [0, 0, 0, 1]


def test_non_usable_row_after_segment_zero_gets_its_own_segment():
    # The case that used to be mislabelled: initialising segment_id to a
    # literal 0 left ffill with no NaN to propagate, so every non-usable row
    # past the first segment kept segment 0 and both interpolation paths then
    # looked up its anchors in the wrong segment.
    cfg = MatchConfig(max_split_gap_s=300.0)
    df = assign_gaps_and_segments(
        _frame([(0, True), (5, True), (1000, True), (1005, False), (1010, True)]), cfg
    )
    assert list(df["segment_id"]) == [0, 0, 1, 1, 1]


def test_leading_non_usable_rows_fall_back_to_segment_zero():
    cfg = MatchConfig(max_split_gap_s=300.0)
    df = assign_gaps_and_segments(
        _frame([(0, False), (5, True), (1000, True)]), cfg
    )
    assert list(df["segment_id"]) == [0, 0, 1]
    assert list(df["gap_id"]) == [0, 0, 1]


def test_no_usable_rows_leaves_defaults():
    cfg = MatchConfig()
    df = assign_gaps_and_segments(_frame([(0, False), (5, False)]), cfg)
    assert list(df["segment_id"]) == [0, 0]
    assert list(df["gap_id"]) == [0, 0]
    assert df["dt_s"].isna().all()


def test_gap_and_segment_thresholds_are_independent():
    # A 200 s gap exceeds max_bridge_gap_s but not max_split_gap_s: it opens a
    # new gap_id while staying inside the same Viterbi segment.
    cfg = MatchConfig(max_bridge_gap_s=90.0, max_split_gap_s=300.0)
    df = assign_gaps_and_segments(_frame([(0, True), (200, True)]), cfg)
    assert list(df["gap_id"]) == [0, 1]
    assert list(df["segment_id"]) == [0, 0]
