"""Tests for netmatch.py: network gap distance and cross-segment bridging."""

import math

import networkx as nx
import numpy as np
import pandas as pd
import pyproj
import pytest
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from gps_route_matcher.config import MatchConfig
from gps_route_matcher.netmatch import (
    CandidateTraceRow,
    EdgeCandidate,
    GraphIndex,
    _AnchorLink,
    _anchor_path_positions,
    _link_segment_paths,
    _network_gap_distance,
    _walk_forward,
    match_trip_to_network_rows,
)


def _make_graph():
    """Small directed graph: 1 -> 2 -> 3 -> 4, plus a disconnected node 9."""
    g = nx.MultiDiGraph()
    g.add_edge(1, 2, key=0, length=10.0)
    g.add_edge(2, 3, key=0, length=40.0)
    g.add_edge(3, 4, key=0, length=5.0)
    g.add_node(9)
    return g


_EDGE_LENGTHS = [10.0, 40.0, 5.0]  # parallel to (1,2,0), (2,3,0), (3,4,0)


def _cand(edge, offset_m):
    return EdgeCandidate(
        obs_index=0,
        edge_idx=0,
        edge=edge,
        offset_m=offset_m,
        distance_m=0.0,
        emission_cost=0.0,
        x=0.0,
        y=0.0,
    )


def _cand_full(edge, edge_idx, offset_m, obs_index=0):
    return EdgeCandidate(
        obs_index=obs_index,
        edge_idx=edge_idx,
        edge=edge,
        offset_m=offset_m,
        distance_m=0.0,
        emission_cost=0.0,
        x=0.0,
        y=0.0,
    )


class _MiniIndex:
    """Duck-typed stand-in for GraphIndex exposing only the attributes
    _link_segment_paths needs (graph_proj, edge_lengths)."""

    def __init__(self, graph_proj, edge_lengths):
        self.graph_proj = graph_proj
        self.edge_lengths = edge_lengths


def test_same_edge_forward():
    prev = _cand((1, 2, 0), offset_m=2.0)
    curr = _cand((1, 2, 0), offset_m=7.0)
    edge_lengths = [10.0]
    assert _network_gap_distance(prev, curr, _make_graph(), edge_lengths) == 5.0


def test_shared_node_no_path_search():
    """prev ends exactly where curr's edge starts: remaining + curr offset."""
    prev = _cand((1, 2, 0), offset_m=8.0)   # 2 m remaining on edge 1->2
    curr = _cand((2, 3, 0), offset_m=3.0)
    edge_lengths = [10.0]
    dist = _network_gap_distance(prev, curr, _make_graph(), edge_lengths)
    assert dist == 2.0 + 3.0


def test_multi_hop_network_path():
    """prev's edge and curr's edge are separated by an intermediate edge."""
    prev = _cand((1, 2, 0), offset_m=10.0)  # 0 m remaining on edge 1->2
    curr = _cand((3, 4, 0), offset_m=1.0)
    edge_lengths = [10.0]
    # remaining (0) + shortest path 2->3 (40) + curr offset (1)
    dist = _network_gap_distance(prev, curr, _make_graph(), edge_lengths)
    assert dist == 0.0 + 40.0 + 1.0


def test_unreachable_returns_inf():
    prev = _cand((1, 2, 0), offset_m=10.0)
    curr = _cand((9, 9, 0), offset_m=0.0)  # disconnected node, self-loop key
    edge_lengths = [10.0]
    dist = _network_gap_distance(prev, curr, _make_graph(), edge_lengths)
    assert math.isinf(dist)


# ── _link_segment_paths ────────────────────────────────────────────────────


def _mini_index():
    return _MiniIndex(_make_graph(), _EDGE_LENGTHS)


def test_link_first_anchor_has_no_gap():
    seg = [[_cand_full((1, 2, 0), 0, 2.0), _cand_full((1, 2, 0), 0, 7.0)]]
    links = _link_segment_paths(seg, _mini_index(), MatchConfig())
    assert links[0].gap_m is None
    assert links[0].bridged is True
    assert links[0].bridge_edges == []


def test_link_within_segment_step():
    seg = [[_cand_full((1, 2, 0), 0, 2.0), _cand_full((1, 2, 0), 0, 7.0)]]
    links = _link_segment_paths(seg, _mini_index(), MatchConfig())
    assert links[1].crossing_gap is False
    assert links[1].bridged is True
    assert links[1].gap_m == 5.0


def test_link_bridgeable_boundary():
    seg0 = [_cand_full((1, 2, 0), 0, 5.0), _cand_full((1, 2, 0), 0, 8.0)]  # 2 pts
    seg1 = [_cand_full((2, 3, 0), 1, 3.0), _cand_full((2, 3, 0), 1, 6.0)]  # 2 pts
    links = _link_segment_paths([seg0, seg1], _mini_index(), MatchConfig())
    boundary = links[2]  # first point of segment 1
    assert boundary.crossing_gap is True
    assert boundary.gap_m == 2.0 + 3.0  # remaining on seg0's edge + curr offset
    assert boundary.bridged is True
    assert boundary.bridge_edges == [(2, 3, 0)]


def test_link_boundary_too_far_left_open():
    seg0 = [_cand_full((1, 2, 0), 0, 5.0), _cand_full((1, 2, 0), 0, 10.0)]  # 2 pts
    seg1 = [_cand_full((3, 4, 0), 2, 1.0), _cand_full((3, 4, 0), 2, 3.0)]  # 2 pts
    links = _link_segment_paths([seg0, seg1], _mini_index(), MatchConfig())
    boundary = links[2]
    # remaining(0) + shortest 2->3 (40) + offset(1) = 41 > default max_gap_bridge_m (20)
    assert boundary.gap_m == 41.0
    assert boundary.bridged is False
    assert boundary.bridge_edges == []


def test_link_boundary_too_short_left_open():
    seg0 = [_cand_full((1, 2, 0), 0, 8.0)]  # only 1 pt, below min_segment_points_to_bridge_gap
    seg1 = [_cand_full((2, 3, 0), 1, 3.0), _cand_full((2, 3, 0), 1, 6.0)]
    links = _link_segment_paths([seg0, seg1], _mini_index(), MatchConfig())
    boundary = links[1]
    assert boundary.gap_m == 2.0 + 3.0  # short distance-wise, but segment too small
    assert boundary.bridged is False


# ── _anchor_path_positions ──────────────────────────────────────────────────


def _link(bridged, gap_m):
    return _AnchorLink(cand=None, seg_idx=0, crossing_gap=False, gap_m=gap_m, bridged=bridged, bridge_edges=[])


def test_anchor_path_positions_accumulates_within_group():
    links = [_link(True, None), _link(True, 5.0), _link(True, 3.0)]
    group_id, path_m = _anchor_path_positions(links)
    assert list(group_id) == [0, 0, 0]
    assert list(path_m) == [0.0, 5.0, 8.0]


def test_anchor_path_positions_resets_on_left_open_boundary():
    links = [_link(True, None), _link(True, 5.0), _link(False, 41.0), _link(True, 4.0)]
    group_id, path_m = _anchor_path_positions(links)
    assert list(group_id) == [0, 0, 1, 1]
    assert path_m[2] == 0.0
    assert path_m[3] == 4.0


# ── _walk_forward ────────────────────────────────────────────────────────


def _walk_index():
    edge_keys = [(1, 2, 0), (2, 3, 0), (3, 4, 0)]
    edge_lines = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 0.0), (10.0, 40.0)]),
        LineString([(10.0, 40.0), (10.0, 45.0)]),
    ]
    return GraphIndex(
        graph=None,
        graph_proj=None,
        crs_metric="EPSG:32723",
        transformer_fwd=None,
        edge_keys=edge_keys,
        edge_lines=edge_lines,
        edge_lengths=np.array([10.0, 40.0, 5.0]),
        kdtree=None,
        sample_edge_idx=None,
        edge_index_by_key={k: i for i, k in enumerate(edge_keys)},
    )


def test_walk_forward_stays_on_start_edge():
    index = _walk_index()
    start = _cand_full((1, 2, 0), 0, 2.0)
    x, y, edge = _walk_forward(start, [], 3.0, index)
    assert (x, y) == pytest.approx((5.0, 0.0))
    assert edge == (1, 2, 0)


def test_walk_forward_spills_onto_one_bridge_edge():
    index = _walk_index()
    start = _cand_full((1, 2, 0), 0, 8.0)  # 2 m remaining on edge 1
    x, y, edge = _walk_forward(start, [(2, 3, 0)], 7.0, index)  # 2 + 5 more
    assert (x, y) == pytest.approx((10.0, 5.0))
    assert edge == (2, 3, 0)


def test_walk_forward_spills_across_two_bridge_edges():
    index = _walk_index()
    start = _cand_full((1, 2, 0), 0, 10.0)  # 0 m remaining
    # 40 m along edge 2, then 3 m along edge 3
    x, y, edge = _walk_forward(start, [(2, 3, 0), (3, 4, 0)], 43.0, index)
    assert (x, y) == pytest.approx((10.0, 43.0))
    assert edge == (3, 4, 0)


def test_walk_forward_clips_beyond_path_end():
    index = _walk_index()
    start = _cand_full((1, 2, 0), 0, 10.0)
    x, y, edge = _walk_forward(start, [(2, 3, 0), (3, 4, 0)], 1000.0, index)
    assert (x, y) == pytest.approx((10.0, 45.0))
    assert edge == (3, 4, 0)


def test_walk_forward_nonfinite_distance_clamps_to_start():
    index = _walk_index()
    start = _cand_full((1, 2, 0), 0, 4.0)
    for bad in (float("inf"), float("-inf"), float("nan")):
        x, y, edge = _walk_forward(start, [(2, 3, 0)], bad, index)
        assert (x, y) == pytest.approx((4.0, 0.0))
        assert edge == (1, 2, 0)


# ── match_trip_to_network_rows (end-to-end on a hand-built L-shaped graph) ──


def _make_l_shaped_index():
    """Two-edge L-shaped road: node 1 -(A-B, 100m)-> node 2 -(B-C, 80m)-> node 3.

    Built by hand (bypassing build_graph_index/osmnx) with a real UTM 23S
    (Rio de Janeiro) transform, so tests exercise genuine WGS84<->metric
    round-tripping without needing network access or an osmnx-shaped graph.
    """
    crs_metric = "EPSG:32723"
    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs_metric, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs_metric, "EPSG:4326", always_xy=True)

    ox_e, oy_n = 660000.0, 7470000.0  # arbitrary UTM 23S origin near Rio de Janeiro
    coords = {1: (ox_e, oy_n), 2: (ox_e + 100.0, oy_n), 3: (ox_e + 100.0, oy_n + 80.0)}

    g = nx.MultiDiGraph()
    for n, (x, y) in coords.items():
        g.add_node(n, x=x, y=y)
    g.add_edge(1, 2, key=0, length=100.0)
    g.add_edge(2, 3, key=0, length=80.0)

    edge_keys = [(1, 2, 0), (2, 3, 0)]
    edge_lines = [
        LineString([coords[1], coords[2]]),
        LineString([coords[2], coords[3]]),
    ]
    edge_lengths = np.array([100.0, 80.0])

    sample_pts, sample_edge_idx = [], []
    for ei, line in enumerate(edge_lines):
        for i in range(21):
            pt = line.interpolate(line.length * i / 20)
            sample_pts.append((pt.x, pt.y))
            sample_edge_idx.append(ei)

    index = GraphIndex(
        graph=g,
        graph_proj=g,
        crs_metric=crs_metric,
        transformer_fwd=fwd,
        edge_keys=edge_keys,
        edge_lines=edge_lines,
        edge_lengths=edge_lengths,
        kdtree=cKDTree(np.asarray(sample_pts)),
        sample_edge_idx=np.asarray(sample_edge_idx, dtype=int),
        edge_index_by_key={k: i for i, k in enumerate(edge_keys)},
    )
    return index, inv


def _row(inv, x, y, ts, lon=None, lat=None):
    if lon is None or lat is None:
        lon, lat = inv.transform(x, y)
    return {"Timestamp": ts, "Longitude": lon, "Latitude": lat}


def _base_config(**overrides):
    return MatchConfig(detect_timestamp_outliers=False, check_oob=False, **overrides)


def test_rows_interpolates_across_a_road_corner_not_a_straight_chord():
    index, inv = _make_l_shaped_index()
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    rows = [
        _row(inv, 660010.0, 7470000.0, base),                       # 0: A-B offset 10
        _row(inv, 660090.0, 7470000.0, base + pd.Timedelta(seconds=30)),   # 1: A-B offset 90
        {"Timestamp": base + pd.Timedelta(seconds=60), "Longitude": 0.0, "Latitude": 0.0},  # 2: missing GPS
        _row(inv, 660100.0, 7470020.0, base + pd.Timedelta(seconds=90)),   # 3: B-C offset 20
        _row(inv, 660100.0, 7470060.0, base + pd.Timedelta(seconds=120)),  # 4: B-C offset 60
    ]
    trip_df = pd.DataFrame(rows)
    config = _base_config()

    result = match_trip_to_network_rows(trip_df, "t0", index, config)

    assert list(result["match_status"][[0, 1, 3, 4]]) == ["matched"] * 4
    row2 = result.iloc[2]
    assert row2["match_status"] == "interpolated"
    assert row2["interpolated"]
    assert bool(row2["group_id"] == result.iloc[1]["group_id"])  # same group as its segment

    # The interpolated point must sit almost exactly on the road geometry...
    fwd = index.transformer_fwd
    ix, iy = fwd.transform(row2["matched_lon"], row2["matched_lat"])
    road = LineString([(660000.0, 7470000.0), (660100.0, 7470000.0), (660100.0, 7470080.0)])
    assert road.distance(Point(ix, iy)) < 0.5

    # ...but measurably off the straight chord between the two bracketing anchors.
    chord = LineString([(660090.0, 7470000.0), (660100.0, 7470020.0)])
    assert chord.distance(Point(ix, iy)) > 3.0


def test_rows_bridges_group_id_across_a_short_cross_segment_gap():
    index, inv = _make_l_shaped_index()
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    rows = [
        _row(inv, 660010.0, 7470000.0, base),
        _row(inv, 660020.0, 7470000.0, base + pd.Timedelta(seconds=10)),
        {"Timestamp": base + pd.Timedelta(seconds=210), "Longitude": 0.0, "Latitude": 0.0},  # missing, mid-gap
        _row(inv, 660025.0, 7470000.0, base + pd.Timedelta(seconds=410)),  # new segment (>300s gap), 5m away
        _row(inv, 660040.0, 7470000.0, base + pd.Timedelta(seconds=420)),
    ]
    trip_df = pd.DataFrame(rows)
    config = _base_config()  # default max_split_gap_s=300, max_gap_bridge_m=20

    result = match_trip_to_network_rows(trip_df, "t1", index, config)

    assert result.iloc[0]["segment_id"] != result.iloc[3]["segment_id"]  # two distinct HMM segments
    assert result.iloc[0]["group_id"] == result.iloc[3]["group_id"]  # but bridged into one group
    gap_row = result.iloc[2]
    assert gap_row["match_status"] == "interpolated"
    assert gap_row["group_id"] == result.iloc[0]["group_id"]
    assert result.iloc[1]["path_m"] < gap_row["path_m"] < result.iloc[3]["path_m"]


def test_rows_leaves_gap_unresolved_when_too_sparse_to_bridge():
    index, inv = _make_l_shaped_index()
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    rows = [
        _row(inv, 660010.0, 7470000.0, base),  # only 1 point in segment 0
        # 350s from the only segment-0 anchor: too far to nearest-snap
        # (default max_interpolation_gap_s=300), and segment 0 is too short
        # to bridge onward even though the network gap itself is small.
        {"Timestamp": base + pd.Timedelta(seconds=350), "Longitude": 0.0, "Latitude": 0.0},
        _row(inv, 660020.0, 7470000.0, base + pd.Timedelta(seconds=700)),
        _row(inv, 660030.0, 7470000.0, base + pd.Timedelta(seconds=710)),
    ]
    trip_df = pd.DataFrame(rows)
    config = _base_config()

    result = match_trip_to_network_rows(trip_df, "t2", index, config)

    assert result.iloc[0]["group_id"] != result.iloc[2]["group_id"]
    assert result.iloc[1]["match_status"] == "unresolved_large_gap"


def test_rows_all_gps_missing_no_exceptions():
    index, inv = _make_l_shaped_index()
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    rows = [
        {"Timestamp": base + pd.Timedelta(seconds=i * 30), "Longitude": 0.0, "Latitude": 0.0}
        for i in range(4)
    ]
    trip_df = pd.DataFrame(rows)
    config = _base_config()

    result = match_trip_to_network_rows(trip_df, "t3", index, config)

    assert (result["match_status"] == "unresolved_large_gap").all()
    assert result["matched_lon"].isna().all()


def test_rows_preserve_input_row_order():
    index, inv = _make_l_shaped_index()
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    # Deliberately out-of-time-order rows
    rows = [
        _row(inv, 660050.0, 7470000.0, base + pd.Timedelta(seconds=60)),
        _row(inv, 660010.0, 7470000.0, base),
        _row(inv, 660090.0, 7470000.0, base + pd.Timedelta(seconds=90)),
    ]
    trip_df = pd.DataFrame(rows)
    config = _base_config()

    result = match_trip_to_network_rows(trip_df, "t4", index, config)

    assert list(result.index) == list(range(len(rows)))
    pd.testing.assert_series_equal(
        result["Timestamp"].reset_index(drop=True), trip_df["Timestamp"].reset_index(drop=True)
    )


def test_rows_trace_sink_records_candidates_and_winning_path():
    index, inv = _make_l_shaped_index()
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    rows = [
        _row(inv, 660010.0, 7470000.0, base),
        _row(inv, 660050.0, 7470000.0, base + pd.Timedelta(seconds=30)),
        _row(inv, 660090.0, 7470000.0, base + pd.Timedelta(seconds=60)),
    ]
    trip_df = pd.DataFrame(rows)
    config = _base_config()

    trace_sink: list[CandidateTraceRow] = []
    result = match_trip_to_network_rows(trip_df, "t5", index, config, trace_sink=trace_sink)

    assert trace_sink, "trace_sink should be populated when passed in"
    assert all(isinstance(r, CandidateTraceRow) for r in trace_sink)

    by_obs: dict[int, list[CandidateTraceRow]] = {}
    for r in trace_sink:
        by_obs.setdefault(r.obs_index, []).append(r)

    # First observation of the (only) segment has no predecessor.
    first_obs = min(by_obs)
    for r in by_obs[first_obs]:
        assert r.transition_cost is None
        assert r.pred_obs_index is None
        assert r.cumulative_cost == pytest.approx(r.emission_cost)

    # Later observations' candidates all have a resolved predecessor + transition cost.
    for oi, group in by_obs.items():
        if oi == first_obs:
            continue
        for r in group:
            assert r.transition_cost is not None
            assert r.pred_obs_index is not None
            assert r.lon is not None and r.lat is not None

    # Exactly one candidate per observation is on the winning path, and its edge
    # matches the edge actually chosen by match_trip_to_network_rows.
    matched = result[result["match_status"] == "matched"].set_index("original_index")
    for oi, group in by_obs.items():
        winners = [r for r in group if r.is_on_winning_path]
        assert len(winners) == 1
        winner = winners[0]
        row = matched.loc[oi]
        assert (row["matched_edge_u"], row["matched_edge_v"], row["matched_edge_key"]) == winner.edge
