"""Tests for the road-network route-inference pipeline (no OSM download needed)."""

import networkx as nx
import numpy as np
import pytest

from gps_route_matcher.config import MatchConfig
from gps_route_matcher.io import load_gpx_route, write_gpx_route
from gps_route_matcher.netmatch import network_transition_cost
from gps_route_matcher.route_inference import (
    RouteInferenceConfig,
    _radius_cluster,
    _support_weight,
    build_edge_counts,
    edge_sequence_to_points,
    highest_support_path,
)
from gps_route_matcher.netmatch import TripMatchResult


def _grid_graph() -> nx.MultiDiGraph:
    """3-node line graph a→b→c plus a longer detour a→d→c, all with counts."""
    g = nx.MultiDiGraph()
    coords = {"a": (0.0, 0.0), "b": (0.001, 0.0), "c": (0.002, 0.0), "d": (0.001, 0.001)}
    for n, (x, y) in coords.items():
        g.add_node(n, x=x, y=y)
    g.add_edge("a", "b", key=0, length=100.0)
    g.add_edge("b", "c", key=0, length=100.0)
    g.add_edge("a", "d", key=0, length=100.0)
    g.add_edge("d", "c", key=0, length=100.0)
    return g


# ── transition cost ───────────────────────────────────────────────────────────

def test_transition_prefers_direct_route():
    cfg = MatchConfig()
    direct = network_transition_cost(100.0, 100.0, dt_s=30.0, config=cfg)
    detour = network_transition_cost(400.0, 100.0, dt_s=30.0, config=cfg)
    assert direct < detour


def test_transition_penalises_impossible_speed():
    cfg = MatchConfig(max_speed_mps=6.0)
    ok = network_transition_cost(100.0, 100.0, dt_s=30.0, config=cfg)      # 3.3 m/s
    fast = network_transition_cost(1000.0, 1000.0, dt_s=10.0, config=cfg)  # 100 m/s
    assert fast > ok + 1000


def test_transition_large_dt_allows_large_distance():
    cfg = MatchConfig(max_speed_mps=6.0)
    slow = network_transition_cost(300.0, 300.0, dt_s=60.0, config=cfg)   # 5 m/s
    assert slow < 1.0


def test_transition_unreachable_is_large():
    cfg = MatchConfig()
    assert network_transition_cost(float("inf"), 10.0, dt_s=5.0, config=cfg) >= 1e12


# ── edge counts + highest-support path ───────────────────────────────────────

def test_counts_and_path_prefer_supported_route():
    g = _grid_graph()
    # 5 trips took a→b→c, none took the detour
    trips = [
        TripMatchResult(f"t{i}", [("a", "b", 0), ("b", "c", 0)], 2, 2, 1, 0)
        for i in range(5)
    ]
    counted = build_edge_counts(g, trips)
    assert counted["a"]["b"][0]["count"] == 5
    assert counted["a"]["d"][0]["count"] == 0

    cfg = RouteInferenceConfig(gamma=2.0)
    path = highest_support_path(counted, "a", "c", cfg)
    assert path == [("a", "b", 0), ("b", "c", 0)]


def test_path_via_node_makes_loop():
    g = _grid_graph()
    g.add_edge("c", "a", key=0, length=250.0)  # return edge to close the loop
    counted = build_edge_counts(g, [])
    cfg = RouteInferenceConfig()
    path = highest_support_path(counted, "a", "a", cfg, via_node="c")
    assert len(path) >= 3
    assert path[0][0] == "a" and path[-1][1] == "a"


def test_no_path_raises_clear_error():
    g = _grid_graph()
    g.add_node("z", x=1.0, y=1.0)  # disconnected node
    counted = build_edge_counts(g, [])
    with pytest.raises(ValueError, match="component"):
        highest_support_path(counted, "a", "z", RouteInferenceConfig())


def test_support_weight_positive_and_finite():
    # networkx passes {key: data} for multigraphs — the callable must unwrap it.
    w = _support_weight(2.0)
    assert w(None, None, {0: {"length": 100.0, "count": 0}}) == 100.0
    assert w(None, None, {0: {"length": 100.0, "count": 9}}) == 1.0
    assert w(None, None, {0: {"length": 100.0, "count": 10000}}) > 0.0
    # min over parallel edges
    assert w(None, None, {0: {"length": 100.0, "count": 0},
                          1: {"length": 100.0, "count": 9}}) == 1.0


def test_path_prefers_frequency_over_hop_count():
    """Regression: a never-traversed shortcut (fewer hops, shorter) must lose
    to a well-traveled longer corridor. The original weight function read
    attributes off the wrong dict level and degenerated to hop counting."""
    g = nx.MultiDiGraph()
    for n, (x, y) in {"a": (0, 0), "b": (0.001, 0.0005), "c": (0.002, 0)}.items():
        g.add_node(n, x=x, y=y)
    g.add_edge("a", "c", key=0, length=150.0)                # 1-hop shortcut
    g.add_edge("a", "b", key=0, length=100.0)
    g.add_edge("b", "c", key=0, length=100.0)
    trips = [
        TripMatchResult(f"t{i}", [("a", "b", 0), ("b", "c", 0)], 2, 2, 1, 0)
        for i in range(10)
    ]
    counted = build_edge_counts(g, trips)
    path = highest_support_path(counted, "a", "c", RouteInferenceConfig(gamma=2.0))
    assert path == [("a", "b", 0), ("b", "c", 0)], (
        f"took the untraversed shortcut: {path}"
    )


def test_path_picks_higher_count_parallel_edge():
    g = nx.MultiDiGraph()
    for n, (x, y) in {"a": (0, 0), "b": (0.001, 0)}.items():
        g.add_node(n, x=x, y=y)
    g.add_edge("a", "b", key=0, length=100.0)
    g.add_edge("a", "b", key=1, length=100.0)
    trips = [TripMatchResult(f"t{i}", [("a", "b", 1)], 1, 1, 1, 0) for i in range(5)]
    counted = build_edge_counts(g, trips)
    path = highest_support_path(counted, "a", "b", RouteInferenceConfig(gamma=2.0))
    assert path == [("a", "b", 1)]


# ── endpoint clustering ───────────────────────────────────────────────────────

def test_radius_cluster_groups_nearby_points():
    pts = np.array([[0, 0], [5, 0], [10, 0], [500, 500], [505, 500]])
    labels = _radius_cluster(pts, eps_m=20.0)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]


# ── GPX round trip ────────────────────────────────────────────────────────────

def test_write_gpx_route_roundtrip(tmp_path):
    points = [(-43.236, -22.871), (-43.237, -22.872), (-43.238, -22.873)]
    path = str(tmp_path / "route.gpx")
    write_gpx_route(points, path, name="test route")
    loaded = load_gpx_route(path)
    assert len(loaded) == 3
    for (lon1, lat1), (lon2, lat2) in zip(points, loaded):
        assert abs(lon1 - lon2) < 1e-9
        assert abs(lat1 - lat2) < 1e-9


# ── edge sequence → points ────────────────────────────────────────────────────

def test_edge_sequence_to_points_dedupes_boundaries():
    g = _grid_graph()
    pts = edge_sequence_to_points(g, [("a", "b", 0), ("b", "c", 0)])
    # a, b, c — the duplicate 'b' at the edge boundary must be removed
    assert pts == [(0.0, 0.0), (0.001, 0.0), (0.002, 0.0)]
