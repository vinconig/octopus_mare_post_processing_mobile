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
    NetworkDistanceCache,
    _AnchorLink,
    _anchor_path_positions,
    _is_reversal,
    _link_segment_paths,
    _network_gap_distance,
    _viterbi_segment,
    _walk_forward,
    generate_edge_candidates,
    match_trip_to_network_rows,
    street_key,
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


# ── reversal detection / penalty ────────────────────────────────────────────


def test_route_distance_same_edge_backward_returns_inf():
    """NetworkDistanceCache.route_distance (used inside Viterbi's own candidate
    selection, unlike _network_gap_distance) must reject a backward offset move
    on the same directed edge outright, without falling through to a wasted
    Dijkstra search back to the edge's own start node."""
    cache = NetworkDistanceCache(_make_graph())
    prev = _cand((2, 3, 0), offset_m=30.0)
    curr = _cand((2, 3, 0), offset_m=10.0)
    dist = cache.route_distance(prev, curr, np.asarray(_EDGE_LENGTHS), cutoff=1000.0)
    assert math.isinf(dist)


def test_is_reversal_same_edge_forward_is_false():
    prev = _cand_full((1, 2, 0), edge_idx=0, offset_m=2.0)
    curr = _cand_full((1, 2, 0), edge_idx=0, offset_m=7.0)
    assert _is_reversal(prev, curr) is False


def test_is_reversal_same_edge_backward_is_true():
    prev = _cand_full((1, 2, 0), edge_idx=0, offset_m=7.0)
    curr = _cand_full((1, 2, 0), edge_idx=0, offset_m=2.0)
    assert _is_reversal(prev, curr) is True


def test_is_reversal_paired_reverse_edge_is_true():
    prev = _cand_full((1, 2, 0), edge_idx=0, offset_m=8.0)
    curr = _cand_full((2, 1, 0), edge_idx=1, offset_m=2.0)
    assert _is_reversal(prev, curr) is True


def test_is_reversal_unrelated_edge_is_false():
    """Continuing through an intersection onto a different street is not a
    reversal, even though it's also a different edge_idx."""
    prev = _cand_full((1, 2, 0), edge_idx=0, offset_m=8.0)
    curr = _cand_full((2, 3, 0), edge_idx=2, offset_m=2.0)
    assert _is_reversal(prev, curr) is False


def _reversal_graph():
    """A two-way street 1<->2 (both directed edges present) plus a continuation
    2->3, used to test that Viterbi prefers continuing forward over flipping to
    the reverse-direction edge of the street just traversed."""
    g = nx.MultiDiGraph()
    g.add_edge(1, 2, key=0, length=10.0)
    g.add_edge(2, 1, key=0, length=10.0)
    g.add_edge(2, 3, key=0, length=10.0)
    return g


_REV_EDGE_LENGTHS = np.asarray([10.0, 10.0, 10.0])  # (1,2,0), (2,1,0), (2,3,0)


class _RevIndex:
    """Duck-typed stand-in exposing only what _viterbi_segment needs (edge_lengths)."""

    def __init__(self, edge_lengths):
        self.edge_lengths = edge_lengths


def _viterbi_reversal_fixture(network_reversal_penalty):
    """Two observations: the first is matched at the end of street 1->2; the
    second has two candidates of near-identical cost — continuing onto 2->3,
    or flipping to the reverse-direction edge 2->1 (walking back the way you
    came). Emission cost is stacked in the reversal candidate's favor (0.0 vs
    1.0) so the only thing that can save the forward candidate is the penalty.
    """
    config = MatchConfig(gps_sigma_m=1.0, max_speed_mps=5.0, high_speed_penalty=1000.0,
                          network_reversal_penalty=network_reversal_penalty)
    dist_cache = NetworkDistanceCache(_reversal_graph())
    index = _RevIndex(_REV_EDGE_LENGTHS)

    cand0 = _cand_full((1, 2, 0), edge_idx=0, offset_m=8.0, obs_index=0)
    cand_forward = EdgeCandidate(
        obs_index=1, edge_idx=2, edge=(2, 3, 0), offset_m=2.0,
        distance_m=0.0, emission_cost=1.0, x=0.0, y=0.0,
    )
    cand_reversal = EdgeCandidate(
        obs_index=1, edge_idx=1, edge=(2, 1, 0), offset_m=2.0,
        distance_m=0.0, emission_cost=0.0, x=0.0, y=0.0,
    )

    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    ts1 = ts0 + pd.Timedelta(seconds=5)
    obs = [(0, ts0, 0.0, 0.0), (1, ts1, 1.0, 0.0)]
    cands = {0: [cand0], 1: [cand_forward, cand_reversal]}

    path = _viterbi_segment(obs, cands, dist_cache, index, config)
    return path[-1]


def test_reversal_penalty_prefers_continuing_forward():
    winner = _viterbi_reversal_fixture(network_reversal_penalty=10000.0)
    assert winner.edge == (2, 3, 0)


def test_without_reversal_penalty_would_pick_the_reversal():
    """Confirms the fixture is actually load-bearing: with the penalty
    disabled, the (deliberately cheaper-emission) reversal candidate wins,
    proving test_reversal_penalty_prefers_continuing_forward is testing the
    penalty itself and not some other tie-break."""
    winner = _viterbi_reversal_fixture(network_reversal_penalty=0.0)
    assert winner.edge == (2, 1, 0)


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
    return MatchConfig(detect_timestamp_outliers=False, **overrides)


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
    # Bridging (group_id merge) only needs max_gap_bridge_m/min_segment_points_
    # to_bridge_gap and is independent of max_interpolation_gap_s. But since
    # the row bracketing this gap spans the full 400s split, it needs a
    # max_interpolation_gap_s that actually covers that total span — the
    # default (300s) would leave it unresolved (see the test right below this
    # one for that default-config case).
    config = _base_config(max_interpolation_gap_s=500.0)

    result = match_trip_to_network_rows(trip_df, "t1", index, config)

    assert result.iloc[0]["segment_id"] != result.iloc[3]["segment_id"]  # two distinct HMM segments
    assert result.iloc[0]["group_id"] == result.iloc[3]["group_id"]  # but bridged into one group
    gap_row = result.iloc[2]
    assert gap_row["match_status"] == "interpolated"
    assert gap_row["group_id"] == result.iloc[0]["group_id"]
    assert result.iloc[1]["path_m"] < gap_row["path_m"] < result.iloc[3]["path_m"]


def test_rows_bridged_gap_row_unresolved_when_total_span_exceeds_cap_at_default_config():
    """Same shape as the test above, but at default max_interpolation_gap_s
    (300s). Bridging still succeeds (group_id merges), but the row inside the
    gap sees dt_before=200s and dt_after=200s — each individually under 300s,
    but summing to 400s of genuinely unobserved time. The old per-side check
    would have accepted this; the sum-based check correctly does not."""
    index, inv = _make_l_shaped_index()
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    rows = [
        _row(inv, 660010.0, 7470000.0, base),
        _row(inv, 660020.0, 7470000.0, base + pd.Timedelta(seconds=10)),
        {"Timestamp": base + pd.Timedelta(seconds=210), "Longitude": 0.0, "Latitude": 0.0},
        _row(inv, 660025.0, 7470000.0, base + pd.Timedelta(seconds=410)),
        _row(inv, 660040.0, 7470000.0, base + pd.Timedelta(seconds=420)),
    ]
    trip_df = pd.DataFrame(rows)
    config = _base_config()  # default max_split_gap_s=300, max_interpolation_gap_s=300

    result = match_trip_to_network_rows(trip_df, "t1b", index, config)

    assert result.iloc[0]["group_id"] == result.iloc[3]["group_id"]  # bridging still succeeds
    gap_row = result.iloc[2]
    assert gap_row["match_status"] == "unresolved_large_gap"
    assert pd.isna(gap_row["matched_lon"])
    assert not gap_row["interpolated"]


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


def test_rows_unbracketable_row_near_a_single_anchor_is_unresolved_not_snapped():
    """A row close in time to the only anchor in its group, but with no
    second anchor to bracket against, must not be pinned to that one
    anchor's position — even though it's well within max_interpolation_gap_s."""
    index, inv = _make_l_shaped_index()
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    rows = [
        _row(inv, 660010.0, 7470000.0, base),  # only anchor in this group
        # 10s later, un-bracketable (no anchor after it reachable in-group).
        {"Timestamp": base + pd.Timedelta(seconds=10), "Longitude": 0.0, "Latitude": 0.0},
    ]
    trip_df = pd.DataFrame(rows)
    config = _base_config()

    result = match_trip_to_network_rows(trip_df, "t2b", index, config)

    row = result.iloc[1]
    assert row["match_status"] == "unresolved_large_gap"
    assert pd.isna(row["matched_lon"])
    assert pd.isna(row["matched_lat"])
    assert not row["interpolated"]


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

    # The per-term breakdown adds back up to the totals the DP used.
    for r in trace_sink:
        assert r.off_route_cost <= r.emission_cost
        if r.transition_cost is None:
            assert r.detour_cost is None and r.reversal_cost is None
        else:
            terms = r.detour_cost + r.speed_cost + r.reversal_cost + r.blocked_cost
            assert terms == pytest.approx(r.transition_cost)

    # Exactly one candidate per observation is on the winning path, and its edge
    # matches the edge actually chosen by match_trip_to_network_rows.
    matched = result[result["match_status"] == "matched"].set_index("original_index")
    for oi, group in by_obs.items():
        winners = [r for r in group if r.is_on_winning_path]
        assert len(winners) == 1
        winner = winners[0]
        row = matched.loc[oi]
        assert (row["matched_edge_u"], row["matched_edge_v"], row["matched_edge_key"]) == winner.edge


# ── candidate generation: never on a node, densified along route edges ──────


class _IdentityTransformer:
    """Stand-in for GraphIndex.transformer_fwd in tests, so candidate fixtures
    can use plain metres as 'lon/lat' instead of going through a real CRS."""

    @staticmethod
    def transform(lon, lat):
        return lon, lat


def _geometry_index(edges):
    """Real GraphIndex over hand-built metric geometries.

    edges: {(u, v, key): LineString}. Mirrors build_graph_index's sampling but
    without osmnx/CRS machinery, so candidate generation can be tested against
    exact coordinates.
    """
    edge_keys, edge_lines, lengths, pts, sample_edge_idx = [], [], [], [], []
    for i, (key, line) in enumerate(edges.items()):
        edge_keys.append(key)
        edge_lines.append(line)
        lengths.append(line.length)
        n = max(2, int(math.ceil(line.length / 5.0)) + 1)
        for j in range(n):
            p = line.interpolate(line.length * j / (n - 1))
            pts.append((p.x, p.y))
            sample_edge_idx.append(i)
    return GraphIndex(
        graph=None,
        graph_proj=None,
        crs_metric="test",
        transformer_fwd=_IdentityTransformer(),
        edge_keys=edge_keys,
        edge_lines=edge_lines,
        edge_lengths=np.asarray(lengths),
        kdtree=cKDTree(np.asarray(pts)),
        sample_edge_idx=np.asarray(sample_edge_idx, dtype=int),
        edge_index_by_key={k: i for i, k in enumerate(edge_keys)},
    )


def _t_junction_index():
    """Stem running west→east into node (0,0); crossbar running south→north
    through it. Both 100 m long, so their shared node sits mid-figure."""
    return _geometry_index({
        (1, 2, 0): LineString([(-100.0, 0.0), (0.0, 0.0)]),    # ends at the node
        (3, 4, 0): LineString([(0.0, -50.0), (0.0, 50.0)]),    # passes through it
    })


def _candidate_config(**overrides):
    return MatchConfig(node_inset_m=5.0, candidate_radius_m=50.0, candidate_k=30, **overrides)


def test_candidate_never_lands_on_a_node():
    """A point past the east end of the stem must not be matched to the stem's
    end node: the offset stays node_inset_m clear of it."""
    index = _t_junction_index()
    config = _candidate_config()
    cands = generate_edge_candidates(index, 10.0, 3.0, 0, config)

    stem = [c for c in cands if c.edge == (1, 2, 0)]
    assert stem, "the stem should still be a candidate edge"
    length = float(index.edge_lengths[index.edge_index_by_key[(1, 2, 0)]])
    for c in stem:
        assert config.node_inset_m <= c.offset_m <= length - config.node_inset_m


def test_candidates_at_a_shared_node_have_distinct_positions():
    """The degeneracy this change exists to remove: two edges meeting at a node
    must not produce candidates at the same coordinate with the same cost."""
    index = _t_junction_index()
    cands = generate_edge_candidates(index, 6.0, -6.0, 0, _candidate_config())

    positions = {(round(c.x, 6), round(c.y, 6)) for c in cands}
    assert len(positions) == len(cands)
    assert (0.0, 0.0) not in positions  # the node itself is never a candidate


def test_short_edge_offset_stays_inside():
    """On an edge shorter than 2 * node_inset_m the inset is capped by
    max_node_inset_fraction, so the interval never collapses or inverts."""
    index = _geometry_index({(1, 2, 0): LineString([(0.0, 0.0), (4.0, 0.0)])})
    config = _candidate_config()
    cands = generate_edge_candidates(index, 100.0, 0.0, 0, config)

    assert len(cands) == 1
    inset = config.max_node_inset_fraction * 4.0
    assert inset <= cands[0].offset_m <= 4.0 - inset


def test_route_edges_are_densified_and_others_are_not():
    index = _t_junction_index()
    config = _candidate_config(route_candidates_per_edge=5, candidate_spacing_m=10.0)
    cands = generate_edge_candidates(index, -50.0, 2.0, 0, config, route_edges={(1, 2, 0)})

    stem_offsets = sorted(c.offset_m for c in cands if c.edge == (1, 2, 0))
    assert len(stem_offsets) == 5
    assert stem_offsets == pytest.approx([30.0, 40.0, 50.0, 60.0, 70.0])
    assert len([c for c in cands if c.edge == (3, 4, 0)]) == 1


def test_route_edges_none_keeps_one_candidate_per_edge():
    index = _t_junction_index()
    cands = generate_edge_candidates(index, -50.0, 2.0, 0, _candidate_config())
    assert len(cands) == len({c.edge for c in cands})


def test_densified_offsets_clamp_to_the_inset_interval():
    """Densification near an edge end must not step outside [inset, len-inset]."""
    index = _t_junction_index()
    config = _candidate_config(route_candidates_per_edge=5, candidate_spacing_m=10.0)
    cands = generate_edge_candidates(index, -2.0, 2.0, 0, config, route_edges={(1, 2, 0)})

    length = float(index.edge_lengths[index.edge_index_by_key[(1, 2, 0)]])
    stem = [c for c in cands if c.edge == (1, 2, 0)]
    assert stem
    for c in stem:
        assert config.node_inset_m <= c.offset_m <= length - config.node_inset_m
    # clamping collapses duplicates rather than emitting the same offset twice
    assert len({c.offset_m for c in stem}) == len(stem)


# ── off-route penalty ───────────────────────────────────────────────────────


def test_off_route_penalty_applies_only_to_edges_outside_the_route():
    index = _t_junction_index()
    base = generate_edge_candidates(index, -50.0, 2.0, 0, _candidate_config(),
                                    route_edges={(1, 2, 0)})
    pen = generate_edge_candidates(index, -50.0, 2.0, 0, _candidate_config(off_route_penalty=8.0),
                                   route_edges={(1, 2, 0)})
    by_edge = lambda cs: {c.edge: c.emission_cost for c in cs}
    assert by_edge(pen)[(1, 2, 0)] == pytest.approx(by_edge(base)[(1, 2, 0)])
    assert by_edge(pen)[(3, 4, 0)] == pytest.approx(by_edge(base)[(3, 4, 0)] + 8.0)


def test_off_route_penalty_is_inert_without_route_edges():
    """Route inference has no route; a configured penalty must not touch it."""
    index = _t_junction_index()
    plain = generate_edge_candidates(index, -50.0, 2.0, 0, _candidate_config())
    pen = generate_edge_candidates(index, -50.0, 2.0, 0, _candidate_config(off_route_penalty=8.0))
    assert [c.emission_cost for c in pen] == pytest.approx([c.emission_cost for c in plain])


def test_final_matching_config_shares_sigma_and_adds_penalty():
    final, default = MatchConfig.for_final_matching(), MatchConfig()
    assert final.gps_sigma_m == default.gps_sigma_m == 0.25
    assert final.off_route_penalty > 0 and default.off_route_penalty == 0.0


# ── list-Viterbi path_sink: top-2 distinct edge sequences per candidate ─────


def _diamond_graph():
    """1 -a-> 2, then two routes to 5 (b,d via 3 and c,e via 4), plus the
    reverse of a. Edge indices follow _DIAMOND_EDGES."""
    g = nx.MultiDiGraph()
    for u, v in _DIAMOND_EDGES:
        g.add_edge(u, v, key=0, length=10.0)
    return g


_DIAMOND_EDGES = [(1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (2, 1)]


_REACHABLE_EDGES = [[0], [0, 1, 2], [1, 2], [1, 2, 3, 4], [3, 4]]   # per obs, walking 1 -> 5


def _kbest_fixture(seed, n_obs=5, reachable=False):
    """Random observations over the diamond graph: 2-4 candidates each, at
    random offsets with random emission costs.

    reachable=True draws each observation's edges from _REACHABLE_EDGES so a
    finite path always exists. With fully random edges some states can be
    unreachable from every predecessor; there the main DP clamps cost at
    _LARGE_COST (its strict-< update never fires) with an arbitrary
    back-pointer, while path_sink entries keep the true >=1e12 cost, so the
    two only agree on the winner when a finite path exists.
    """
    rng = np.random.default_rng(seed)
    config = MatchConfig(gps_sigma_m=1.0, max_speed_mps=1.0, high_speed_penalty=3.0,
                         network_reversal_penalty=50.0)
    dist_cache = NetworkDistanceCache(_diamond_graph())
    index = _RevIndex(np.full(len(_DIAMOND_EDGES), 10.0))
    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    obs, cands = [], {}
    for oi in range(n_obs):
        obs.append((oi, ts0 + pd.Timedelta(seconds=5 * oi), float(rng.uniform(0, 30)), 0.0))
        cands[oi] = [
            EdgeCandidate(
                obs_index=oi, edge_idx=ei, edge=(*_DIAMOND_EDGES[ei], 0),
                offset_m=float(rng.uniform(1, 9)), distance_m=0.0,
                emission_cost=float(rng.uniform(0, 5)), x=0.0, y=0.0,
                off_route_cost=0.0,
            )
            for ei in rng.choice(
                _REACHABLE_EDGES[oi] if reachable else len(_DIAMOND_EDGES),
                size=int(rng.integers(2, 5)),
            )
        ]
    return obs, cands, index, config, dist_cache


def _brute_force_top2_distinct(obs, cands, index, config, dist_cache):
    """Enumerate every candidate sequence, cost it with the matcher's own
    cost functions, collapse to edge sequences, and return the two cheapest
    distinct-sequence costs."""
    import itertools
    from gps_route_matcher.netmatch import network_transition_cost

    best: dict[tuple, float] = {}
    for combo in itertools.product(*[cands[oi] for oi, *_ in obs]):
        cost = combo[0].emission_cost
        for t in range(1, len(combo)):
            _, ts, x, y = obs[t]
            _, p_ts, p_x, p_y = obs[t - 1]
            dt_s = (ts - p_ts).total_seconds()
            cutoff = max(300.0, 2.0 * config.max_speed_mps * max(dt_s, 1.0) + 200.0)
            nd = dist_cache.route_distance(combo[t - 1], combo[t], index.edge_lengths, cutoff)
            cost += network_transition_cost(nd, math.hypot(x - p_x, y - p_y), dt_s, config)
            if _is_reversal(combo[t - 1], combo[t]):
                cost += config.network_reversal_penalty
            cost += combo[t].emission_cost
        streets = [street_key(c.edge) for c in combo]
        seq = tuple(e for i, e in enumerate(streets) if i == 0 or e != streets[i - 1])
        best[seq] = min(cost, best.get(seq, math.inf))
    return sorted(best.values())[:2]


@pytest.mark.parametrize("seed", range(12))
def test_path_sink_top2_matches_brute_force(seed):
    obs, cands, index, config, dist_cache = _kbest_fixture(seed)
    path_sink: list = []
    _viterbi_segment(obs, cands, dist_cache, index, config, path_sink=path_sink)

    last = obs[-1][0]
    ends = sorted(e.cumulative_cost for e in path_sink if e.obs_index == last)
    # Distinct sequences across end candidates: group by sig, keep the cheapest.
    by_sig: dict[int, float] = {}
    for e in path_sink:
        if e.obs_index == last:
            by_sig[e.sig] = min(e.cumulative_cost, by_sig.get(e.sig, math.inf))
    got = sorted(by_sig.values())[:2]
    expected = _brute_force_top2_distinct(obs, cands, index, config, dist_cache)
    assert got == pytest.approx(expected, rel=1e-9)
    assert ends  # sanity: entries exist


@pytest.mark.parametrize("seed", range(6))
def test_path_sink_rank0_chain_is_the_viterbi_path_and_terms_add_up(seed):
    obs, cands, index, config, dist_cache = _kbest_fixture(seed, reachable=True)
    plain = _viterbi_segment(obs, cands, dist_cache, index, config)
    path_sink: list = []
    with_sink = _viterbi_segment(obs, cands, dist_cache, index, config, path_sink=path_sink)
    assert [(c.obs_index, c.edge, c.offset_m) for c in with_sink] == \
           [(c.obs_index, c.edge, c.offset_m) for c in plain]

    winners = [e for e in path_sink if e.is_on_winning_path]
    if winners[-1].cumulative_cost >= 1e11:
        pytest.skip("no reachable path for this random draw (see _kbest_fixture)")
    assert [(e.obs_index, e.edge) for e in sorted(winners, key=lambda e: e.obs_index)] == \
           [(c.obs_index, c.edge) for c in plain]
    assert all(e.rank == 0 for e in winners)

    for e in path_sink:
        terms = (e.cum_distance + e.cum_off_route + e.cum_detour + e.cum_speed
                 + e.cum_reversal + e.cum_blocked)
        assert terms == pytest.approx(e.cumulative_cost, rel=1e-9)
        # parents precede children, and a rank-1 entry never beats rank 0 on
        # the same candidate or repeats its edge sequence
        assert e.parent_id is None or e.parent_id < e.id
    by_state: dict[tuple[int, int], list] = {}
    for e in path_sink:
        by_state.setdefault((e.obs_index, e.candidate_idx), []).append(e)
    for entries in by_state.values():
        assert len({e.sig for e in entries}) == len(entries) <= 2
        assert [e.rank for e in entries] == list(range(len(entries)))
        assert entries == sorted(entries, key=lambda e: e.cumulative_cost)


def test_path_sink_through_match_trip_to_network_rows_leaves_output_unchanged():
    index, inv = _make_l_shaped_index()
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    rows = [
        _row(inv, 660010.0, 7470000.0, base),
        _row(inv, 660050.0, 7470000.0, base + pd.Timedelta(seconds=30)),
        _row(inv, 660090.0, 7470000.0, base + pd.Timedelta(seconds=60)),
    ]
    trip_df = pd.DataFrame(rows)
    config = _base_config()
    plain = match_trip_to_network_rows(trip_df, "t6", index, config)
    path_sink: list = []
    with_sink = match_trip_to_network_rows(trip_df, "t6", index, config, path_sink=path_sink)
    pd.testing.assert_frame_equal(plain, with_sink)
    assert path_sink and {e.obs_index for e in path_sink} == {0, 1, 2}
