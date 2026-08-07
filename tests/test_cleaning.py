"""Tests for cleaning.py."""

import pandas as pd
import pytest
from shapely.geometry import Polygon

from gps_route_matcher.cleaning import (
    categorize_oob,
    clean_gps_dataframe,
    finalize_usable_for_hmm,
    infer_missing_timestamps,
    inspect_gps_oob,
    inspect_gps_ootb,
    prepare_for_matching,
)
from gps_route_matcher.config import MatchConfig
from gps_route_matcher.route import build_route_1d


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


def test_bracketed_fill_divides_span_not_nominal_interval():
    """A gap whose span isn't a multiple of the interval still closes on it.

    Rows 3-5 are missing and the surviving anchors are 17s apart, not the
    nominal 5s x 4. Stepping at the interval would put row 5 past row 6.
    """
    rows = _sequential_rows(pd.Timestamp("2026-03-09 12:00:00", tz="UTC"), 8)
    rows[6]["timestamp"] = (
        pd.Timestamp("2026-03-09 12:00:10", tz="UTC") + pd.Timedelta(seconds=17)
    ).isoformat()
    rows[7]["timestamp"] = (
        pd.Timestamp("2026-03-09 12:00:10", tz="UTC") + pd.Timedelta(seconds=22)
    ).isoformat()
    for i in (3, 4, 5):
        rows[i]["timestamp"] = None
    df = _make_df(rows)
    cfg = MatchConfig(expected_sample_interval_s=5.0, infer_missing_timestamps=True)

    out = infer_missing_timestamps(clean_gps_dataframe(df, cfg), cfg)

    base = pd.Timestamp("2026-03-09 12:00:10", tz="UTC")  # row 2, last valid
    step = pd.Timedelta(seconds=17 / 4)                   # span / rows spanned
    assert list(out["timestamp"].iloc[3:6]) == [base + step * k for k in (1, 2, 3)]
    assert out["timestamp"].is_monotonic_increasing


# ── inspect_gps_oob / categorize_oob / inspect_gps_ootb ──────────────────────

_ROUTE = build_route_1d([(-43.0, -22.0), (-43.0, -21.999)])  # short N-S line


def test_inspect_gps_oob_flags_distance_without_touching_gps_valid():
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": -43.0, "lat": -22.0},      # on route
        {"timestamp": "2026-01-01 00:00:05", "lon": -42.999, "lat": -22.0},    # ~111m off
    ])
    cfg = MatchConfig(oob_buffer_m=20.0)
    cleaned = clean_gps_dataframe(df, cfg)
    out = inspect_gps_oob(cleaned, cfg, _ROUTE)

    assert bool(out["out_of_bounds"].iloc[0]) is False
    assert bool(out["out_of_bounds"].iloc[1]) is True
    # gps_valid / usable_for_hmm are untouched by this step
    assert out["gps_valid"].tolist() == cleaned["gps_valid"].tolist()
    assert out["usable_for_hmm"].tolist() == cleaned["usable_for_hmm"].tolist()


def test_categorize_oob_flags_inside_and_outside_polygon():
    polygon = Polygon([(-43.01, -22.01), (-43.01, -21.99), (-42.99, -21.99), (-42.99, -22.01)])
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": -43.0, "lat": -22.0},   # inside
        {"timestamp": "2026-01-01 00:00:05", "lon": -40.0, "lat": -22.0},   # outside
    ])
    cfg = MatchConfig()
    cleaned = clean_gps_dataframe(df, cfg)
    out = categorize_oob(cleaned, polygon)

    assert bool(out["gps_inside_polygon"].iloc[0]) is True
    assert bool(out["gps_inside_polygon"].iloc[1]) is False


def test_inspect_gps_ootb_marks_outside_core_window_and_recomputes_usable_for_hmm():
    # Row 0: before arriving at the route (out_of_bounds). Rows 1-2: on the
    # route (the core walk). Row 3: after leaving the route (out_of_bounds).
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": -42.9, "lat": -22.0},   # far off route
        {"timestamp": "2026-01-01 00:05:00", "lon": -43.0, "lat": -22.0},   # on route
        {"timestamp": "2026-01-01 00:10:00", "lon": -43.0, "lat": -21.9995},  # on route
        {"timestamp": "2026-01-01 00:15:00", "lon": -42.9, "lat": -22.0},   # far off route
    ])
    cfg = MatchConfig(oob_buffer_m=20.0)
    cleaned = clean_gps_dataframe(df, cfg)
    out = inspect_gps_ootb(cleaned, cfg, _ROUTE)  # out_of_bounds not precomputed

    assert "out_of_bounds" in out.columns
    assert list(out["out_of_time_bounds"]) == [True, False, False, True]
    assert list(out["usable_for_hmm"]) == [False, True, True, False]


def test_inspect_gps_ootb_anchors_t_min_on_fix_closest_to_route_start():
    # Row 0: an early, spurious in-buffer fix picked up while walking toward
    # the route start, but it crosses the route partway along (a route_s_m
    # larger than the real start's, though still smaller than the real
    # walk's furthest point) rather than at the start. The naive "earliest
    # anchor timestamp" rule would wrongly anchor t_min here. Row 1: still
    # approaching, off route. Rows 2-3: the real on-route walk, starting at
    # the route's own start point (route_s_m ~ 0) and progressing toward its
    # far end. Row 4: leaving the route.
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": -43.0, "lat": -21.9995},  # mid-route, spurious
        {"timestamp": "2026-01-01 00:05:00", "lon": -42.9, "lat": -22.0},     # off route
        {"timestamp": "2026-01-01 00:10:00", "lon": -43.0, "lat": -22.0},     # on route, at start
        {"timestamp": "2026-01-01 00:15:00", "lon": -43.0, "lat": -21.9992},  # on route, near end
        {"timestamp": "2026-01-01 00:20:00", "lon": -42.9, "lat": -22.0},     # off route
    ])
    cfg = MatchConfig(oob_buffer_m=20.0)
    cleaned = clean_gps_dataframe(df, cfg)
    out = inspect_gps_ootb(cleaned, cfg, _ROUTE)

    # t_min is anchored on row 2 (route_s_m ~ 0), not row 0 (earliest anchor
    # timestamp overall) — so rows 0-1 fall outside the derived window.
    assert list(out["out_of_time_bounds"]) == [True, True, False, False, True]
    assert list(out["usable_for_hmm"]) == [False, False, True, True, False]


def test_inspect_gps_ootb_anchors_t_max_on_fix_closest_to_route_end():
    # Mirror of the t_min case: after properly leaving the route, the walk
    # can re-cross it (e.g. doubling back near the start) well after the
    # real walk ended. Row 0: off route, before start. Row 1: on route, at
    # the start (route_s_m ~ 0). Row 2: on route, near the route's *end*
    # (route_s_m largest) — this is the true end of the walk. Row 3: off
    # route, properly leaving. Row 4: a late, spurious in-buffer fix back
    # near the start (route_s_m ~ 0) — NOT part of the real walk. The naive
    # "latest anchor timestamp" rule would wrongly anchor t_max here.
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": -42.9, "lat": -22.0},     # off route
        {"timestamp": "2026-01-01 00:05:00", "lon": -43.0, "lat": -22.0},     # on route, at start
        {"timestamp": "2026-01-01 00:10:00", "lon": -43.0, "lat": -21.9991},  # on route, near end
        {"timestamp": "2026-01-01 00:15:00", "lon": -42.9, "lat": -22.0},     # off route
        {"timestamp": "2026-01-01 00:20:00", "lon": -43.0, "lat": -22.0},     # near start again, spurious
    ])
    cfg = MatchConfig(oob_buffer_m=20.0)
    cleaned = clean_gps_dataframe(df, cfg)
    out = inspect_gps_ootb(cleaned, cfg, _ROUTE)

    # t_max is anchored on row 2 (route_s_m largest), not row 4 (latest
    # anchor timestamp overall) — so rows 3-4 fall outside the derived
    # window, even though row 4 is itself spatially in-buffer.
    assert list(out["out_of_time_bounds"]) == [True, False, False, True, True]
    assert list(out["usable_for_hmm"]) == [False, True, True, False, False]


def test_inspect_gps_ootb_falls_back_when_s_anchored_window_would_invert():
    # Two independent spurious crossings at opposite route ends, with
    # inverted timing relative to each other: row 0 is near the route's end
    # (largest route_s_m) but earliest in time; row 2 is at the route's
    # start (smallest route_s_m) but later in time. Anchoring t_min/t_max
    # purely on route_s_m would give t_min=00:10 (row 2) > t_max=00:00 (row
    # 0) — an inverted, empty window. This must fall back to the simple
    # earliest/latest anchor timestamp window ([00:00, 00:10]) instead of
    # marking every row out_of_time_bounds.
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": -43.0, "lat": -21.9991},  # near route end
        {"timestamp": "2026-01-01 00:05:00", "lon": -42.9, "lat": -22.0},     # off route
        {"timestamp": "2026-01-01 00:10:00", "lon": -43.0, "lat": -22.0},     # on route, at start
        {"timestamp": "2026-01-01 00:15:00", "lon": -42.9, "lat": -22.0},     # off route
    ])
    cfg = MatchConfig(oob_buffer_m=20.0)
    cleaned = clean_gps_dataframe(df, cfg)
    out = inspect_gps_ootb(cleaned, cfg, _ROUTE)

    assert list(out["out_of_time_bounds"]) == [False, False, False, True]
    assert list(out["usable_for_hmm"]) == [True, False, True, False]


def test_inspect_gps_ootb_no_anchor_marks_everything_out_of_time_bounds():
    # Every row is out_of_bounds, so no confirmed on-route fix exists to
    # anchor a temporal window.
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": -42.9, "lat": -22.0},
        {"timestamp": "2026-01-01 00:05:00", "lon": -42.9, "lat": -22.0},
    ])
    cfg = MatchConfig(oob_buffer_m=20.0)
    cleaned = clean_gps_dataframe(df, cfg)
    out = inspect_gps_ootb(cleaned, cfg, _ROUTE)

    assert out["out_of_time_bounds"].all()
    assert not out["usable_for_hmm"].any()


# ── finalize_usable_for_hmm ──────────────────────────────────────────────────


def _one_good_row():
    df = _make_df([{"timestamp": "2026-01-01 00:00:00", "lon": -43.0, "lat": -22.0}])
    return clean_gps_dataframe(df, MatchConfig())


def test_finalize_usable_for_hmm_folds_in_only_the_flags_present():
    base = _one_good_row()
    assert bool(finalize_usable_for_hmm(base)["usable_for_hmm"].iloc[0]) is True

    for col, bad_value in (
        ("out_of_bounds", True),
        ("out_of_time_bounds", True),
        ("gps_inside_polygon", False),
    ):
        df = base.copy()
        df[col] = bad_value
        out = finalize_usable_for_hmm(df)
        assert bool(out["usable_for_hmm"].iloc[0]) is False, col


def test_finalize_usable_for_hmm_is_idempotent():
    df = _one_good_row()
    df["out_of_bounds"] = False
    once = finalize_usable_for_hmm(df)
    twice = finalize_usable_for_hmm(once)
    assert list(once["usable_for_hmm"]) == list(twice["usable_for_hmm"])


def test_out_of_bounds_is_folded_in_without_a_polygon():
    # Used to be silently skipped: out_of_bounds was computed but only ever
    # reached usable_for_hmm through inspect_gps_ootb, which needed a polygon,
    # so a caller with a route and no polygon still matched off-route points.
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": -43.0, "lat": -22.0},
        {"timestamp": "2026-01-01 00:05:00", "lon": -42.9, "lat": -22.0},
    ])
    cfg = MatchConfig(oob_buffer_m=20.0)
    out = prepare_for_matching(df, cfg, route=_ROUTE)

    assert list(out["out_of_bounds"]) == [False, True]
    assert list(out["usable_for_hmm"]) == [True, False]


# ── prepare_for_matching ─────────────────────────────────────────────────────


def test_prepare_for_matching_without_a_route_skips_route_checks():
    # scripts/infer_route.py's case, and the one that used to raise
    # AttributeError partway through inspect_gps_oob when no GPX existed yet.
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": -43.0, "lat": -22.0},
        {"timestamp": "2026-01-01 00:00:05", "lon": -43.0, "lat": -22.0},
    ])
    out = prepare_for_matching(df, MatchConfig(), route=None)

    assert "out_of_bounds" not in out.columns
    assert "out_of_time_bounds" not in out.columns
    assert out["usable_for_hmm"].all()


def test_prepare_for_matching_assigns_gaps_and_segments_last():
    # The two on-route fixes sit at different arc-lengths along _ROUTE so the
    # derived walk window actually spans them.
    df = _make_df([
        {"timestamp": "2026-01-01 00:00:00", "lon": -43.0, "lat": -22.0},
        {"timestamp": "2026-01-01 00:00:05", "lon": -42.9, "lat": -22.0},  # off-route
        {"timestamp": "2026-01-01 00:00:10", "lon": -43.0, "lat": -21.9995},
    ])
    cfg = MatchConfig(oob_buffer_m=20.0, max_bridge_gap_s=90.0)
    out = prepare_for_matching(df, cfg, route=_ROUTE)

    # The off-route row is excluded from usable_for_hmm, so dt_s must be
    # measured across it (10 s), not to it — which only holds if gap
    # assignment ran after usable_for_hmm was final.
    assert list(out["usable_for_hmm"]) == [True, False, True]
    assert out["dt_s"].iloc[2] == 10.0


def test_prepare_for_matching_adopts_precleaned_timestamps():
    # A frame coming out of scripts/make_trip_data.py already carries the
    # per-recording verdicts; the per-trip pass must not second-guess them.
    df = _make_df([
        {"timestamp": "2015-10-19 12:23:23", "lon": -43.0, "lat": -22.0},
        {"timestamp": "2015-10-19 12:23:28", "lon": -43.0, "lat": -22.0},
    ])
    df["ts_clean"] = ["2026-01-01 00:00:00+00:00", "2026-01-01 00:00:05+00:00"]
    df["ts_valid"] = True
    df["ts_inferred"] = [False, True]

    out = prepare_for_matching(df, MatchConfig())
    assert out["timestamp_valid"].all()
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    assert list(out["timestamp_inferred"]) == [False, True]
