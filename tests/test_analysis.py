"""Tests for edge-level aggregation of matched route recordings."""

import numpy as np
import pandas as pd

import gps_route_matcher.analysis as analysis_module
from gps_route_matcher.analysis import (
    aggregate_by_edge,
    assemble_output,
    assign_points_to_edges,
    reindex_to_full_route,
)


def _route_metadata() -> dict:
    """3-edge route: (1,2,0) -> (2,3,0) -> (3,4,0), in that order."""
    return {
        "route_id": "test",
        "edges": [
            {"u": 1, "v": 2, "key": 0, "edge_order": 0},
            {"u": 2, "v": 3, "key": 0, "edge_order": 1},
            {"u": 3, "v": 4, "key": 0, "edge_order": 2},
        ],
        "edge_geometry": {
            "(1,2,0)": {"geometry": "LINESTRING (0 0, 1 0)", "length": 10.0},
            "(2,3,0)": {"geometry": "LINESTRING (1 0, 2 0)", "length": 20.0},
            "(3,4,0)": {"geometry": "LINESTRING (2 0, 3 0)", "length": 30.0},
        },
    }


def _trip_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Timestamp": ["t0", "t1", "t2", "t3", "t4"],
        "Latitude": [0.0] * 5,
        "Longitude": [0.0] * 5,
        "Temperature": [20.0, 22.0, 25.0, 99.0, 30.0],
        "Humidity": [50.0, 52.0, 60.0, 99.0, 70.0],
        "timestamp": [
            "2024-01-01 00:00:00", "2024-01-01 00:00:10",
            "2024-01-01 00:01:00",
            "2024-01-01 00:02:00",
            "2024-01-01 00:03:00",
        ],
        "matched_edge_u": [1.0, 1.0, 2.0, np.nan, 9.0],
        "matched_edge_v": [2.0, 2.0, 3.0, np.nan, 9.0],
        "matched_edge_key": [0.0, 0.0, 0.0, np.nan, 0.0],
        "match_status": ["matched", "interpolated", "matched", "unresolved_large_gap", "matched"],
        "interpolated": [False, True, False, False, False],
    })


def test_assign_points_to_edges_filters_and_attaches_edge_order():
    df = assign_points_to_edges(_trip_df(), _route_metadata())

    # unresolved row (NaN edge) and out-of-route edge (9,9,0) are dropped;
    # matched + interpolated rows on route edges are kept.
    assert len(df) == 3
    assert set(df["edge_order"]) == {0, 1}
    assert df["timestamp"].dtype.kind == "M"  # parsed to datetime


def test_aggregate_by_edge_computes_avg_and_timestamp_mean():
    assigned = assign_points_to_edges(_trip_df(), _route_metadata())
    agg = aggregate_by_edge(assigned)

    edge0 = agg[agg["edge_order"] == 0].iloc[0]
    assert edge0["n_points"] == 2
    assert edge0["temperature_avg"] == 21.0
    assert edge0["humidity_avg"] == 51.0
    assert edge0["start_timestamp"] == pd.Timestamp("2024-01-01 00:00:00")
    assert edge0["end_timestamp"] == pd.Timestamp("2024-01-01 00:00:10")
    assert edge0["timestamp_mean"] == pd.Timestamp("2024-01-01 00:00:05")
    # row1 (edge0's second point) is "interpolated"; row0 is not.
    assert edge0["includes_interpolated_pts"] == True  # noqa: E712

    edge1 = agg[agg["edge_order"] == 1].iloc[0]
    assert edge1["n_points"] == 1
    assert edge1["temperature_avg"] == 25.0
    assert edge1["includes_interpolated_pts"] == False  # noqa: E712


def test_reindex_to_full_route_fills_zero_coverage_edges():
    route_metadata = _route_metadata()
    assigned = assign_points_to_edges(_trip_df(), route_metadata)
    agg = aggregate_by_edge(assigned)
    full = reindex_to_full_route(agg, route_metadata)

    assert len(full) == 3
    assert list(full["edge_order"]) == [0, 1, 2]

    edge2 = full[full["edge_order"] == 2].iloc[0]
    assert edge2["n_points"] == 0
    assert edge2["includes_interpolated_pts"] == False  # noqa: E712
    assert pd.isna(edge2["temperature_avg"])
    assert pd.isna(edge2["start_timestamp"])
    assert edge2["length"] == 30.0
    assert edge2["geometry"] == "LINESTRING (2 0, 3 0)"


def test_assemble_output_column_order_and_route_id():
    route_metadata = _route_metadata()
    assigned = assign_points_to_edges(_trip_df(), route_metadata)
    agg = aggregate_by_edge(assigned)
    full = reindex_to_full_route(agg, route_metadata)
    out = assemble_output(full, route_id="OM_1")

    assert list(out.columns) == [
        "route_id", "edge_order", "u", "v", "key", "length", "geometry",
        "start_timestamp", "end_timestamp", "timestamp_mean",
        "humidity_avg", "temperature_avg", "n_points", "includes_interpolated_pts",
    ]
    assert (out["route_id"] == "OM_1").all()
    assert len(out) == 3


def test_load_route_recordings_concatenates_and_tags_files(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_module, "PROJECT_ROOT", tmp_path)
    edge_dir = tmp_path / "data" / "OM_1_edge_analysis"
    edge_dir.mkdir(parents=True)

    def _write(name: str, temp_avg: list[float]) -> None:
        pd.DataFrame({
            "route_id": ["OM_1"] * 2,
            "edge_order": [0, 1],
            "u": [1, 2], "v": [2, 3], "key": [0, 0],
            "length": [10.0, 20.0],
            "geometry": ["LINESTRING (0 0, 1 0)", "LINESTRING (1 0, 2 0)"],
            "temperature_avg": temp_avg,
            "n_points": [5, 3],
        }).to_csv(edge_dir / name, index=False)

    _write("OM_1_20251009_16_15durmin__REC000135.csv", [20.0, 25.0])
    _write("OM_1_20251009_17_10durmin__REC000136.csv", [30.0, 35.0])
    (edge_dir / "not_a_recording.csv").write_text("junk\n")

    long_df = analysis_module.load_route_recordings("OM_1")

    # 2 valid recordings x 2 edges; the non-conforming filename is skipped.
    assert len(long_df) == 4
    assert set(long_df["recording_id"]) == {
        "OM_1_20251009_16_15durmin__REC000135",
        "OM_1_20251009_17_10durmin__REC000136",
    }
    assert set(long_df["date"]) == {"20251009"}
    assert set(long_df["period"]) == {"tarde"}
