"""HMM map matching of GPS trips onto a road network graph.

States are (edge, offset) positions on any edge of a directed road graph.
Observation cleaning and gap handling are reused from cleaning.py and gaps.py.

Emission model:
    emission_cost = perpendicular_distance² / (2 · gps_sigma²)
                    + off_route_penalty   (only for edges outside route_edges)

Transition model (Newson–Krumm style):
    The plausibility of moving between two candidate positions is judged by
    how much the network (routed) distance exceeds the straight-line distance
    between the two GPS fixes, plus a speed penalty using the real dt between
    observations. Large dt therefore allows proportionally more distance
    without penalty, which is what makes gap handling work.

Gap handling:
    Within a segment, consecutive observations are connected by the Viterbi
    transition (real dt, no artificial 1-second steps). Across segments
    (dt > max_split_gap_s) trips are NOT dropped: the matched edge before and
    after the gap are bridged with a network shortest path, so partial
    coverage still contributes evidence for every edge actually traversed —
    but only when both segments have at least
    config.min_segment_points_to_bridge_gap matched points AND the network
    distance being bridged is at most config.max_gap_bridge_m. Otherwise the
    gap is left open (no inferred edges inserted) rather than guessing a long
    stretch of unobserved road from too little evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyproj
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from .cleaning import prepare_for_matching
from .config import MatchConfig
from .route import Route1D

_LARGE_COST = 1e12

EdgeKey = tuple[int, int, int]  # (u, v, key) in the MultiDiGraph


# ── Graph indexing ────────────────────────────────────────────────────────────


@dataclass
class GraphIndex:
    """Spatial index over a projected road graph for candidate generation.

    Every edge geometry is densified into sample points; a single cKDTree over
    all sample points maps a query point to nearby edges in O(log n). Exact
    offsets/distances are then computed by projecting onto the edge LineString.
    """

    graph: nx.MultiDiGraph              # original WGS-84 graph
    graph_proj: nx.MultiDiGraph         # metric (auto-UTM) projection
    crs_metric: str
    transformer_fwd: pyproj.Transformer  # WGS84 → metric
    edge_keys: list[EdgeKey]
    edge_lines: list[LineString]         # metric LineStrings, parallel to edge_keys
    edge_lengths: np.ndarray             # metres, parallel to edge_keys
    kdtree: cKDTree
    sample_edge_idx: np.ndarray          # sample point i belongs to edge_keys[...]
    edge_index_by_key: dict[EdgeKey, int] = field(default_factory=dict)  # reverse of edge_keys

    @property
    def n_edges(self) -> int:
        return len(self.edge_keys)


def build_graph_index(
    graph: nx.MultiDiGraph, sample_spacing_m: float = 15.0
) -> GraphIndex:
    """Project the graph to a metric CRS and build the edge spatial index."""
    import osmnx as ox

    graph_proj = ox.project_graph(graph)
    crs_metric = str(graph_proj.graph["crs"])
    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs_metric, always_xy=True)

    edge_keys: list[EdgeKey] = []
    edge_lines: list[LineString] = []
    lengths: list[float] = []
    sample_pts: list[tuple[float, float]] = []
    sample_edge_idx: list[int] = []

    for u, v, k, data in graph_proj.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            # Straight edge between node coordinates
            geom = LineString([
                (graph_proj.nodes[u]["x"], graph_proj.nodes[u]["y"]),
                (graph_proj.nodes[v]["x"], graph_proj.nodes[v]["y"]),
            ])
        edge_idx = len(edge_keys)
        edge_keys.append((u, v, k))
        edge_lines.append(geom)
        lengths.append(float(data.get("length", geom.length)))

        n_samples = max(2, int(math.ceil(geom.length / sample_spacing_m)) + 1)
        for i in range(n_samples):
            pt = geom.interpolate(geom.length * i / (n_samples - 1))
            sample_pts.append((pt.x, pt.y))
            sample_edge_idx.append(edge_idx)

    return GraphIndex(
        graph=graph,
        graph_proj=graph_proj,
        crs_metric=crs_metric,
        transformer_fwd=fwd,
        edge_keys=edge_keys,
        edge_lines=edge_lines,
        edge_lengths=np.asarray(lengths),
        kdtree=cKDTree(np.asarray(sample_pts)),
        sample_edge_idx=np.asarray(sample_edge_idx, dtype=int),
        edge_index_by_key={k: i for i, k in enumerate(edge_keys)},
    )


# ── Candidates ────────────────────────────────────────────────────────────────


@dataclass
class EdgeCandidate:
    """A candidate match position on one directed edge of the road graph."""

    obs_index: int
    edge_idx: int          # index into GraphIndex.edge_keys
    edge: EdgeKey          # (u, v, key)
    offset_m: float        # distance from edge start (node u) along the edge
    distance_m: float      # perpendicular distance GPS point → edge
    emission_cost: float
    x: float               # projected metric coords of the matched position
    y: float


def _candidate_offsets(
    line: LineString,
    pt: Point,
    n_positions: int,
    config: MatchConfig,
) -> list[float]:
    """Candidate offsets along one edge, never inside the node inset zone.

    The nearest projection is clamped into [inset, length - inset] so that a
    candidate never lands on a node (see MatchConfig.node_inset_m for why).
    When n_positions > 1, further offsets are added alternating downstream and
    upstream of it in candidate_spacing_m steps, each clamped to the same
    interval and de-duplicated — the densification described in
    MatchConfig.route_candidates_per_edge.
    """
    length = line.length
    inset = min(config.node_inset_m, config.max_node_inset_fraction * length)
    lo = inset
    hi = max(inset, length - inset)

    base = min(max(float(line.project(pt)), lo), hi)
    offsets = [base]
    for j in range(1, n_positions):
        step = math.ceil(j / 2) * config.candidate_spacing_m * (1 if j % 2 else -1)
        offset = min(max(base + step, lo), hi)
        if all(abs(offset - existing) > 1e-6 for existing in offsets):
            offsets.append(offset)
    return offsets


def generate_edge_candidates(
    index: GraphIndex,
    lon: float,
    lat: float,
    obs_index: int,
    config: MatchConfig,
    route_edges: set[EdgeKey] | None = None,
) -> list[EdgeCandidate]:
    """Return up to candidate_k candidate positions for one GPS observation.

    Nearby edges are found via the sample-point KD-tree; the point is then
    projected exactly onto each edge geometry, with the offset kept clear of
    both end nodes (_candidate_offsets). Always returns at least the nearest
    candidate, even when it lies beyond candidate_radius_m.

    route_edges, when given, is the set of edges making up the inferred route
    for this trip; those edges each contribute config.route_candidates_per_edge
    positions instead of one, so a continuous match on the correct edge can win
    over a jump to a side street near a junction. Every other edge contributes
    its single nearest position, and the added positions are strictly farther
    from the GPS fix than their own edge's nearest projection — so this widens
    the state space without making any route edge cheaper to match.
    """
    x, y = index.transformer_fwd.transform(lon, lat)

    # Query enough sample points to see several distinct edges.
    k_query = min(40, len(index.sample_edge_idx))
    dists, idxs = index.kdtree.query((x, y), k=k_query)
    dists = np.atleast_1d(dists)
    idxs = np.atleast_1d(idxs)

    # Also do a radius query so wide candidate radii aren't truncated by k.
    within = index.kdtree.query_ball_point((x, y), r=config.candidate_radius_m)
    edge_ids = set(index.sample_edge_idx[i] for i in within)
    edge_ids.update(index.sample_edge_idx[i] for i in idxs)

    pt = Point(x, y)
    # (distance, edge_idx, offset, x, y)
    scored: list[tuple[float, int, float, float, float]] = []
    for ei in edge_ids:
        ei = int(ei)
        line = index.edge_lines[ei]
        n_positions = (
            config.route_candidates_per_edge
            if route_edges and index.edge_keys[ei] in route_edges
            else 1
        )
        for offset in _candidate_offsets(line, pt, n_positions, config):
            pos = line.interpolate(offset)
            scored.append((float(pt.distance(pos)), ei, offset, float(pos.x), float(pos.y)))

    scored.sort(key=lambda t: t[0])
    filtered = [s for s in scored if s[0] <= config.candidate_radius_m]
    if not filtered and scored:
        filtered = [scored[0]]  # keep nearest even if outside the radius
    filtered = filtered[: config.candidate_k]

    sigma2 = config.gps_sigma_m ** 2

    def off_route_cost(edge: EdgeKey) -> float:
        return config.off_route_penalty if route_edges and edge not in route_edges else 0.0

    return [
        EdgeCandidate(
            obs_index=obs_index,
            edge_idx=ei,
            edge=index.edge_keys[ei],
            offset_m=offset,
            distance_m=d,
            emission_cost=d ** 2 / (2.0 * sigma2) + off_route_cost(index.edge_keys[ei]),
            x=px,
            y=py,
        )
        for d, ei, offset, px, py in filtered
    ]


@dataclass
class CandidateTraceRow:
    """One (observation, candidate) row of Viterbi debug trace.

    Populated only when a trace_sink list is passed into
    match_trip_to_network_rows / _match_usable_segments / _viterbi_segment;
    costs nothing for callers that don't pass one. transition_cost and the
    pred_* fields are None for the first candidate of a segment (no
    predecessor exists yet). cumulative_cost is the DP cost to reach this
    candidate (the same quantity as _viterbi_segment's `costs` array).
    is_on_winning_path is set True, after backtracking, for exactly the
    candidates Viterbi actually selected. lon/lat are filled in bulk by
    match_trip_to_network_rows (from x/y) once the DP work is done.
    """

    obs_index: int
    segment_id: int
    candidate_idx: int
    edge: EdgeKey
    offset_m: float
    distance_m: float
    emission_cost: float
    x: float
    y: float
    pred_obs_index: int | None
    pred_candidate_idx: int | None
    transition_cost: float | None
    cumulative_cost: float
    is_on_winning_path: bool = False
    lon: float | None = None
    lat: float | None = None


# ── Network distances ─────────────────────────────────────────────────────────


class NetworkDistanceCache:
    """Bounded-radius Dijkstra distances with per-source caching.

    Consecutive GPS observations reuse the same source nodes constantly, so a
    plain dict cache keyed by (source_node, cutoff) has a very high hit rate
    and keeps the total number of Dijkstra runs manageable across 800+ trips.
    """

    def __init__(self, graph_proj: nx.MultiDiGraph):
        self._graph = graph_proj
        self._cache: dict[tuple[int, float], dict[int, float]] = {}

    def distances_from(self, source: int, cutoff: float) -> dict[int, float]:
        key = (source, cutoff)
        hit = self._cache.get(key)
        if hit is None:
            hit = nx.single_source_dijkstra_path_length(
                self._graph, source, cutoff=cutoff, weight="length"
            )
            self._cache[key] = hit
            if len(self._cache) > 200_000:   # crude memory guard
                self._cache.clear()
        return hit

    def route_distance(
        self,
        prev: EdgeCandidate,
        curr: EdgeCandidate,
        edge_lengths: np.ndarray,
        cutoff: float,
    ) -> float:
        """Directed network distance from prev's position to curr's position.

        Same edge, forward move: distance along the edge. Otherwise: remaining
        length of prev's edge + node-to-node shortest path + curr's offset.
        Returns inf when unreachable within cutoff.
        """
        if prev.edge_idx == curr.edge_idx:
            if curr.offset_m >= prev.offset_m:
                return curr.offset_m - prev.offset_m
            # Backward move on the same directed edge — never a real forward
            # route (a Dijkstra search back to this edge's own start node
            # would almost always fail anyway); reversal is instead penalised
            # explicitly by _is_reversal in _viterbi_segment.
            return math.inf

        exit_node = prev.edge[1]     # v of (u, v, key): direction respected
        entry_node = curr.edge[0]    # u of the target edge
        remaining = float(edge_lengths[prev.edge_idx]) - prev.offset_m

        if exit_node == entry_node:
            return remaining + curr.offset_m

        dists = self.distances_from(exit_node, cutoff)
        node_dist = dists.get(entry_node)
        if node_dist is None:
            return math.inf
        return remaining + node_dist + curr.offset_m


# ── Transition cost ───────────────────────────────────────────────────────────


def network_transition_cost(
    net_dist_m: float,
    straight_dist_m: float,
    dt_s: float,
    config: MatchConfig,
) -> float:
    """Cost of moving net_dist_m through the network in dt_s seconds.

    Two additive terms:

    1. Detour term: |network distance − straight-line distance| between the
       GPS fixes (Newson–Krumm). A candidate pair whose routed distance far
       exceeds the direct displacement is an implausible match.
    2. Speed term: only speeds above max_speed_mps are penalised,
       quadratically, so a large dt legitimately allows a large distance
       (this is the whole point of using real dt in gaps).
    """
    if math.isinf(net_dist_m):
        return _LARGE_COST
    if dt_s <= 0:
        return 0.0 if net_dist_m < 1e-3 else _LARGE_COST

    detour = abs(net_dist_m - straight_dist_m)
    # Scale like the emission term so the two are comparable (β ≈ σ).
    cost = detour / config.gps_sigma_m

    speed = net_dist_m / dt_s
    if speed > config.max_speed_mps:
        cost += config.high_speed_penalty * (speed - config.max_speed_mps) ** 2
    return cost


def _is_reversal(prev: EdgeCandidate, curr: EdgeCandidate) -> bool:
    """True if curr reverses direction on the same physical street prev was on:
    either a backward offset move on the same directed edge, or a switch to
    that edge's paired reverse-direction edge (u, v, key) -> (v, u, key).
    """
    if prev.edge_idx == curr.edge_idx:
        return curr.offset_m < prev.offset_m
    return curr.edge[0] == prev.edge[1] and curr.edge[1] == prev.edge[0]


# ── Per-trip matching ─────────────────────────────────────────────────────────


@dataclass
class TripMatchResult:
    """Matched output for one trip."""

    trip_id: str
    edge_sequence: list[EdgeKey]          # ordered traversed edges (deduped)
    n_obs_total: int
    n_obs_matched: int
    n_segments: int
    n_bridged_gaps: int
    matched_length_m: float = 0.0         # total length of traversed edges
    n_gaps_left_open: int = 0             # segment boundaries NOT bridged (too short/far)


def _viterbi_segment(
    obs: list[tuple[int, pd.Timestamp, float, float]],   # (obs_idx, ts, x, y)
    cands: dict[int, list[EdgeCandidate]],
    dist_cache: NetworkDistanceCache,
    index: GraphIndex,
    config: MatchConfig,
    segment_id: int = -1,
    trace_sink: list[CandidateTraceRow] | None = None,
) -> list[EdgeCandidate]:
    """Minimum-cost candidate path for one segment (Viterbi/DP).

    If trace_sink is given, one CandidateTraceRow per (step, candidate) is
    appended to it as the DP runs, recording each candidate's emission cost,
    the transition cost from whichever predecessor the DP selected as best
    for it, and the resulting cumulative cost — the same values the DP
    itself computes, just stashed instead of discarded.
    """
    steps = [(oi, ts, x, y) for oi, ts, x, y in obs if cands.get(oi)]
    if not steps:
        return []

    first = cands[steps[0][0]]
    costs = np.array([c.emission_cost for c in first])
    back: list[list[int]] = [[-1] * len(first)]
    all_cands: list[list[EdgeCandidate]] = [first]
    trace_rows: list[list[CandidateTraceRow]] | None = None
    if trace_sink is not None:
        trace_rows = [[
            CandidateTraceRow(
                obs_index=cc.obs_index, segment_id=segment_id, candidate_idx=kc,
                edge=cc.edge, offset_m=cc.offset_m, distance_m=cc.distance_m,
                emission_cost=cc.emission_cost, x=cc.x, y=cc.y,
                pred_obs_index=None, pred_candidate_idx=None,
                transition_cost=None, cumulative_cost=float(costs[kc]),
            )
            for kc, cc in enumerate(first)
        ]]

    for t in range(1, len(steps)):
        oi, ts, x, y = steps[t]
        p_oi, p_ts, p_x, p_y = steps[t - 1]
        curr = cands[oi]
        prev = all_cands[-1]

        dt_s = (ts - p_ts).total_seconds()
        straight = math.hypot(x - p_x, y - p_y)
        # Reachable radius: generous bound so valid transitions aren't cut off.
        cutoff = max(300.0, 2.0 * config.max_speed_mps * max(dt_s, 1.0) + 200.0)

        new_costs = np.full(len(curr), _LARGE_COST)
        bp = np.zeros(len(curr), dtype=int)
        best_trans = np.full(len(curr), np.nan)
        for kc, cc in enumerate(curr):
            for jc, pc in enumerate(prev):
                nd = dist_cache.route_distance(pc, cc, index.edge_lengths, cutoff)
                trans_cost = network_transition_cost(nd, straight, dt_s, config)
                if _is_reversal(pc, cc):
                    trans_cost += config.network_reversal_penalty
                c = costs[jc] + trans_cost + cc.emission_cost
                if c < new_costs[kc]:
                    new_costs[kc] = c
                    bp[kc] = jc
                    best_trans[kc] = trans_cost
        costs = new_costs
        back.append(bp.tolist())
        all_cands.append(curr)

        if trace_rows is not None:
            trace_rows.append([
                CandidateTraceRow(
                    obs_index=cc.obs_index, segment_id=segment_id, candidate_idx=kc,
                    edge=cc.edge, offset_m=cc.offset_m, distance_m=cc.distance_m,
                    emission_cost=cc.emission_cost, x=cc.x, y=cc.y,
                    pred_obs_index=prev[bp[kc]].obs_index,
                    pred_candidate_idx=int(bp[kc]),
                    transition_cost=float(best_trans[kc]),
                    cumulative_cost=float(costs[kc]),
                )
                for kc, cc in enumerate(curr)
            ])

    k = int(np.argmin(costs))
    path = [k]
    for t in range(len(all_cands) - 1, 0, -1):
        k = back[t][k]
        path.append(k)
    path.reverse()

    if trace_rows is not None:
        for t, k in enumerate(path):
            trace_rows[t][k].is_on_winning_path = True
        for rows in trace_rows:
            trace_sink.extend(rows)

    return [all_cands[t][k] for t, k in enumerate(path)]


def _network_gap_distance(
    prev: EdgeCandidate,
    curr: EdgeCandidate,
    graph_proj: nx.MultiDiGraph,
    edge_lengths: np.ndarray,
) -> float:
    """Network distance from prev's matched position to curr's, in metres.

    Same geometry as NetworkDistanceCache.route_distance but uncached and
    uncutoff: this only runs once per segment boundary (not inside Viterbi's
    inner loop), and needs a definite answer — including "unreachable" — to
    decide whether a cross-segment gap is short enough to bridge.
    """
    if prev.edge_idx == curr.edge_idx and curr.offset_m >= prev.offset_m:
        return curr.offset_m - prev.offset_m

    exit_node = prev.edge[1]
    entry_node = curr.edge[0]
    remaining = float(edge_lengths[prev.edge_idx]) - prev.offset_m

    if exit_node == entry_node:
        return remaining + curr.offset_m

    try:
        node_dist = nx.shortest_path_length(graph_proj, exit_node, entry_node, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return math.inf
    return remaining + node_dist + curr.offset_m


def _edges_between(
    prev: EdgeCandidate,
    curr: EdgeCandidate,
    graph_proj: nx.MultiDiGraph,
) -> list[EdgeKey]:
    """Expand the traversal prev → curr into the ordered list of edges used.

    Includes curr's edge as the last element; excludes prev's edge (the caller
    already has it). Falls back to [curr.edge] when no path exists.
    """
    if prev.edge_idx == curr.edge_idx:
        return []
    exit_node, entry_node = prev.edge[1], curr.edge[0]
    if exit_node == entry_node:
        return [curr.edge]
    try:
        node_path = nx.shortest_path(graph_proj, exit_node, entry_node, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [curr.edge]

    edges: list[EdgeKey] = []
    for a, b in zip(node_path[:-1], node_path[1:]):
        # Pick the shortest parallel edge (multi-edge key) between a and b.
        data = graph_proj.get_edge_data(a, b)
        best_key = min(data, key=lambda kk: data[kk].get("length", 0.0))
        edges.append((a, b, best_key))
    edges.append(curr.edge)
    return edges


def _match_usable_segments(
    df: pd.DataFrame,
    index: GraphIndex,
    config: MatchConfig,
    dist_cache: NetworkDistanceCache,
    trace_sink: list[CandidateTraceRow] | None = None,
    route_edges: set[EdgeKey] | None = None,
) -> list[tuple[int, list[EdgeCandidate]]]:
    """Generate edge candidates for every usable row and run per-segment Viterbi.

    `df` must already have gone through cleaning.prepare_for_matching
    (usable_for_hmm, segment_id, original_index, lon,
    lat, timestamp all present). Shared by match_trip_to_network and
    match_trip_to_network_rows so the two never diverge.

    route_edges is passed straight through to generate_edge_candidates (see
    there); None means one candidate position per nearby edge.

    Returns (segment_id, winning EdgeCandidate list) pairs, in
    usable["segment_id"].unique() order, skipping segments that produced no
    Viterbi path (e.g. no candidates found for any of their observations).
    """
    usable = df[df["usable_for_hmm"]]
    if usable.empty:
        return []

    # Candidates for every usable observation
    cands: dict[int, list[EdgeCandidate]] = {}
    for _, row in usable.iterrows():
        oi = int(row["original_index"])
        cl = generate_edge_candidates(
            index, float(row["lon"]), float(row["lat"]), oi, config, route_edges=route_edges
        )
        if cl:
            cands[oi] = cl

    # Viterbi per segment
    segment_paths: list[tuple[int, list[EdgeCandidate]]] = []
    for seg_id in usable["segment_id"].unique():
        seg = usable[usable["segment_id"] == seg_id]
        obs = []
        for _, row in seg.iterrows():
            oi = int(row["original_index"])
            if oi in cands:
                x, y = index.transformer_fwd.transform(float(row["lon"]), float(row["lat"]))
                obs.append((oi, row["timestamp"], x, y))
        path = _viterbi_segment(
            obs, cands, dist_cache, index, config,
            segment_id=int(seg_id), trace_sink=trace_sink,
        )
        if path:
            segment_paths.append((int(seg_id), path))

    return segment_paths


@dataclass
class _AnchorLink:
    """How one matched anchor connects to the anchor immediately before it,
    in trip-wide time order (flattened across all segments)."""

    cand: EdgeCandidate
    seg_idx: int                 # index into segment_paths
    crossing_gap: bool           # True iff this is the first point of a segment > 0
    gap_m: float | None          # network distance from the previous anchor; None for the first anchor overall
    bridged: bool                # False only at a left-open segment boundary
    bridge_edges: list[EdgeKey]  # edges from the previous anchor to this one; [] for the first anchor


def _link_segment_paths(
    segment_paths: list[list[EdgeCandidate]],
    index: GraphIndex,
    config: MatchConfig,
) -> list[_AnchorLink]:
    """Flatten segment_paths into one time-ordered list of _AnchorLink,
    applying the min_segment_points_to_bridge_gap / max_gap_bridge_m
    bridge-eligibility rule at every segment boundary.
    """
    links: list[_AnchorLink] = []
    prev_cand: EdgeCandidate | None = None

    for seg_idx, seg_path in enumerate(segment_paths):
        for i, cand in enumerate(seg_path):
            crossing_gap = i == 0 and seg_idx > 0

            if prev_cand is None:
                links.append(_AnchorLink(cand, seg_idx, False, None, True, []))
            elif crossing_gap:
                gap_m = _network_gap_distance(
                    prev_cand, cand, index.graph_proj, index.edge_lengths
                )
                prev_seg_len = len(segment_paths[seg_idx - 1])
                can_bridge = (
                    prev_seg_len >= config.min_segment_points_to_bridge_gap
                    and len(seg_path) >= config.min_segment_points_to_bridge_gap
                    and gap_m <= config.max_gap_bridge_m
                )
                if can_bridge:
                    bridge_edges = _edges_between(prev_cand, cand, index.graph_proj)
                    links.append(_AnchorLink(cand, seg_idx, True, gap_m, True, bridge_edges))
                else:
                    links.append(_AnchorLink(cand, seg_idx, True, gap_m, False, []))
            else:
                gap_m = _network_gap_distance(
                    prev_cand, cand, index.graph_proj, index.edge_lengths
                )
                bridge_edges = _edges_between(prev_cand, cand, index.graph_proj)
                links.append(_AnchorLink(cand, seg_idx, False, gap_m, True, bridge_edges))
            prev_cand = cand

    return links


def match_trip_to_network(
    trip_df: pd.DataFrame,
    trip_id: str,
    index: GraphIndex,
    config: MatchConfig,
    dist_cache: NetworkDistanceCache | None = None,
    route_edges: set[EdgeKey] | None = None,
    polygon=None,
) -> TripMatchResult:
    """Match one trip's GPS points to the road network.

    Reuses cleaning.prepare_for_matching, runs Viterbi per segment, expands the
    chosen candidates into an ordered edge sequence, and bridges across-segment
    gaps with a network shortest path instead of dropping the trip.

    polygon is the bairro boundary; fixes outside it are excluded from matching
    (the road graph was built from that same boundary, so a fix beyond it has
    no edges it could honestly belong to).

    route_edges is passed through to generate_edge_candidates (see there).
    """
    if dist_cache is None:
        dist_cache = NetworkDistanceCache(index.graph_proj)

    # No route argument: the inferred route is this stage's own output, so the
    # route-buffer and walk-window checks aren't available yet. That absence is
    # the only way this differs from match_trip_to_network_rows' cleaning.
    df = prepare_for_matching(trip_df, config, polygon=polygon)

    n_total = len(df)
    seg_pairs = _match_usable_segments(df, index, config, dist_cache, route_edges=route_edges)
    if not seg_pairs:
        return TripMatchResult(trip_id, [], n_total, 0, 0, 0)
    segment_paths = [path for _, path in seg_pairs]

    # Expand candidate sequences into edge sequences. Within a segment,
    # consecutive candidates were already validated by Viterbi's transition
    # cost, so they're always stitched together. Across a segment boundary
    # (a real time gap), a network-path bridge is only inserted when both
    # segments have enough support and the bridge itself is short — otherwise
    # the gap is left open (a break in edge_seq) rather than inventing a long
    # stretch of unobserved road.
    links = _link_segment_paths(segment_paths, index, config)

    edge_seq: list[EdgeKey] = []
    n_bridged = 0
    n_gaps_left_open = 0
    for link in links:
        if link.gap_m is None:
            edge_seq.append(link.cand.edge)
        elif link.bridged:
            edge_seq.extend(link.bridge_edges)
            if link.crossing_gap:
                n_bridged += 1
        else:
            n_gaps_left_open += 1
            edge_seq.append(link.cand.edge)  # fresh start, no bridge inserted
    n_matched = len(links)

    # Deduplicate consecutive repeats (staying on the same edge across obs)
    deduped: list[EdgeKey] = []
    for e in edge_seq:
        if not deduped or deduped[-1] != e:
            deduped.append(e)

    g = index.graph_proj
    matched_length = sum(
        float(g[u][v][k].get("length", 0.0))
        for u, v, k in deduped
        if g.has_edge(u, v, k)
    )

    return TripMatchResult(
        trip_id=trip_id,
        edge_sequence=deduped,
        n_obs_total=n_total,
        n_obs_matched=n_matched,
        n_segments=len(segment_paths),
        n_bridged_gaps=n_bridged,
        matched_length_m=matched_length,
        n_gaps_left_open=n_gaps_left_open,
    )


def match_all_trips(
    tracks_df: pd.DataFrame,
    index: GraphIndex,
    config: MatchConfig,
    trip_id_col: str = "track_id",
    verbose: bool = True,
    polygon=None,
) -> list[TripMatchResult]:
    """Match every trip in the concatenated tracks dataframe.

    Trips are processed independently (grouped by trip_id_col) but share one
    NetworkDistanceCache, since they overwhelmingly traverse the same streets.

    polygon is forwarded to match_trip_to_network (see there).
    """
    dist_cache = NetworkDistanceCache(index.graph_proj)
    results: list[TripMatchResult] = []
    trip_ids = tracks_df[trip_id_col].unique()

    for i, tid in enumerate(trip_ids):
        trip = tracks_df[tracks_df[trip_id_col] == tid]
        res = match_trip_to_network(
            trip, str(tid), index, config, dist_cache, polygon=polygon
        )
        results.append(res)
        if verbose and (i + 1) % 25 == 0:
            print(f"  matched {i + 1}/{len(trip_ids)} trips")

    return results


# ── Per-row output ──────────────────────────────────────────────────────────


def _walk_forward(
    start: EdgeCandidate,
    bridge_edges: list[EdgeKey],
    distance_m: float,
    index: GraphIndex,
) -> tuple[float, float, EdgeKey]:
    """Metric (x, y) position + the EdgeKey it lands on, reached by walking
    distance_m forward from start's matched position along start's own edge,
    then along bridge_edges in order (the same path _edges_between produced
    for this anchor pair — bridge_edges may be [] when start and the
    destination anchor share an edge).

    Clips to the end of the last edge if distance_m exceeds the path's total
    length (defensive only — callers pass a target already bounded by the
    pair's own gap_m).

    A non-finite distance_m (nan/inf) is clamped to 0, i.e. the walk stays at
    start's own position. This is the fallback for the rare case where two
    Viterbi-chosen candidates are connected only via a cutoff-bounded search
    (network_transition_cost caps an unreachable pair's cost at _LARGE_COST
    rather than truly infinite, so Viterbi can still pick such a pair) but
    the uncapped shortest path used for gap_m/path_m genuinely can't reach
    curr from prev (disconnected component or one-way mismatch) — poisoning
    path_m with inf/nan for that group.
    """
    if not math.isfinite(distance_m):
        distance_m = 0.0
    start_len = float(index.edge_lengths[start.edge_idx])
    remaining_on_start = start_len - start.offset_m

    if distance_m <= remaining_on_start or not bridge_edges:
        offset = min(max(start.offset_m + distance_m, 0.0), start_len)
        pt = index.edge_lines[start.edge_idx].interpolate(offset)
        return float(pt.x), float(pt.y), start.edge

    remaining = distance_m - remaining_on_start
    for i, edge_key in enumerate(bridge_edges):
        ei = index.edge_index_by_key[edge_key]
        elen = float(index.edge_lengths[ei])
        is_last = i == len(bridge_edges) - 1
        if remaining <= elen or is_last:
            offset = min(max(remaining, 0.0), elen)
            pt = index.edge_lines[ei].interpolate(offset)
            return float(pt.x), float(pt.y), edge_key
        remaining -= elen

    raise AssertionError("unreachable: bridge_edges is non-empty in this branch")


def _anchor_path_positions(links: list[_AnchorLink]) -> tuple[np.ndarray, np.ndarray]:
    """Return (group_id, path_m) int/float arrays, one entry per link (same order).

    group_id increments by 1 at every link with bridged=False (a left-open
    segment boundary) and is otherwise inherited from the previous anchor.
    path_m accumulates link.gap_m and resets to 0.0 at each group boundary.
    """
    n = len(links)
    group_id = np.zeros(n, dtype=int)
    path_m = np.zeros(n, dtype=float)

    for i in range(1, n):
        if links[i].bridged:
            group_id[i] = group_id[i - 1]
            path_m[i] = path_m[i - 1] + (links[i].gap_m or 0.0)
        else:
            group_id[i] = group_id[i - 1] + 1
            path_m[i] = 0.0

    return group_id, path_m


def _add_empty_network_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add NaN/unresolved match columns when there are no usable observations."""
    df = df.copy()
    n = len(df)
    df["matched_lon"] = np.nan
    df["matched_lat"] = np.nan
    df["distance_to_route_m"] = np.nan
    df["group_id"] = pd.Series([pd.NA] * n, index=df.index, dtype="Int64")
    df["path_m"] = np.nan
    df["matched_edge_u"] = pd.Series([pd.NA] * n, index=df.index, dtype="Int64")
    df["matched_edge_v"] = pd.Series([pd.NA] * n, index=df.index, dtype="Int64")
    df["matched_edge_key"] = pd.Series([pd.NA] * n, index=df.index, dtype="Int64")
    df["interpolated"] = False
    df["match_status"] = np.where(
        df["timestamp_valid"], "unresolved_large_gap", "unresolved_missing_time"
    )
    return df


def match_trip_to_network_rows(
    trip_df: pd.DataFrame,
    trip_id: str,
    index: GraphIndex,
    config: MatchConfig,
    dist_cache: NetworkDistanceCache | None = None,
    transformer_inv: pyproj.Transformer | None = None,
    route: Route1D | None = None,
    polygon=None,
    trace_sink: list[CandidateTraceRow] | None = None,
    route_edges: set[EdgeKey] | None = None,
) -> pd.DataFrame:
    """Match one trip's GPS points to the road network and return a per-row
    enriched dataframe.

    Reuses _match_usable_segments / _link_segment_paths, so the set of
    traversed edges and bridge/leave-open decisions never diverges from
    match_trip_to_network. Unlike match_trip_to_network, this always runs
    infer_missing_timestamps and fills matched_lon/matched_lat for every row,
    interpolating across gaps that couldn't be matched directly by walking
    forward along the actual matched road path between the two bracketing
    anchors (time-weighted, arc-length interpolation along the matched path).

    `route` is this trip's inferred route (routes/Inferred_Route_OM_x.gpx):
    points beyond config.oob_buffer_m from that line get out_of_bounds = True
    and the walk's own time window is derived from it (cleaning.inspect_gps_oob
    / inspect_gps_ootb) — gps_valid itself is left untouched. `polygon` (the
    Bairro.kml boundary) adds gps_inside_polygon. Passing either as None just
    skips the checks that need it; see cleaning.prepare_for_matching, which
    owns the ordering.

    Returns all columns produced by cleaning.prepare_for_matching, plus:
        matched_lon, matched_lat, distance_to_route_m, match_status,
        interpolated, group_id, path_m,
        matched_edge_u, matched_edge_v, matched_edge_key
    match_status is one of "matched" | "interpolated" |
    "unresolved_missing_time" | "unresolved_large_gap" |
    "unresolved_out_of_time_bounds" — the last is only possible when a route
    was given, since deriving the walk's time window needs one; such rows are
    never interpolated regardless of gap size. Row order matches the input
    dataframe.

    If `trace_sink` is given (an empty list), it is filled in place with one
    CandidateTraceRow per (observation, candidate) considered by the Viterbi
    DP across all segments — every candidate's emission cost, the transition
    cost from whichever predecessor the DP selected as best for it, the
    resulting cumulative DP cost, and whether it ended up on the winning
    path. Used by visualization.generate_hmm_debug_map; has no effect on the
    returned dataframe or on behavior when left as None.

    If `route_edges` is given (the {(u, v, key)} set of this trip's inferred
    route, as written by route_inference.save_route_metadata), those edges get
    config.route_candidates_per_edge candidate positions each instead of one —
    see generate_edge_candidates.
    """
    if dist_cache is None:
        dist_cache = NetworkDistanceCache(index.graph_proj)
    if transformer_inv is None:
        transformer_inv = pyproj.Transformer.from_crs(
            index.crs_metric, "EPSG:4326", always_xy=True
        )

    df = prepare_for_matching(trip_df, config, route=route, polygon=polygon)

    seg_pairs = _match_usable_segments(
        df, index, config, dist_cache, trace_sink=trace_sink, route_edges=route_edges
    )
    if trace_sink:
        xs = np.array([r.x for r in trace_sink])
        ys = np.array([r.y for r in trace_sink])
        lons, lats = transformer_inv.transform(xs, ys)
        for r, lon, lat in zip(trace_sink, lons, lats):
            r.lon = float(lon)
            r.lat = float(lat)
    if not seg_pairs:
        df = _add_empty_network_match_columns(df)
        return df.sort_values("original_index", kind="stable").reset_index(drop=True)

    seg_ids = [sid for sid, _ in seg_pairs]
    segment_paths = [path for _, path in seg_pairs]
    links = _link_segment_paths(segment_paths, index, config)
    group_arr, path_m_arr = _anchor_path_positions(links)

    link_by_oi: dict[int, _AnchorLink] = {l.cand.obs_index: l for l in links}
    group_by_oi = {l.cand.obs_index: int(g) for l, g in zip(links, group_arr)}
    path_m_by_oi = {l.cand.obs_index: float(p) for l, p in zip(links, path_m_arr)}

    # segment_id -> group_id, and group_id -> chronologically-ordered obs_index list
    seg_id_to_group: dict[int, int] = {}
    groups: dict[int, list[int]] = {}
    for link, g in zip(links, group_arr):
        seg_id = seg_ids[link.seg_idx]
        seg_id_to_group.setdefault(seg_id, int(g))
        groups.setdefault(int(g), []).append(link.cand.obs_index)

    # ── Initialize output columns ──────────────────────────────────────────
    n = len(df)
    df = df.copy()
    df["matched_lon"] = np.nan
    df["matched_lat"] = np.nan
    df["distance_to_route_m"] = np.nan
    df["match_status"] = pd.NA
    df["interpolated"] = False
    df["group_id"] = pd.Series([pd.NA] * n, index=df.index, dtype="Int64")
    df["path_m"] = np.nan
    df["matched_edge_u"] = pd.Series([pd.NA] * n, index=df.index, dtype="Int64")
    df["matched_edge_v"] = pd.Series([pd.NA] * n, index=df.index, dtype="Int64")
    df["matched_edge_key"] = pd.Series([pd.NA] * n, index=df.index, dtype="Int64")

    # ── Fill anchor (directly matched) rows ────────────────────────────────
    anchor_mask = df["original_index"].isin(list(link_by_oi.keys()))
    for idx in df.index[anchor_mask]:
        oi = int(df.at[idx, "original_index"])
        cand = link_by_oi[oi].cand
        lon, lat = transformer_inv.transform(cand.x, cand.y)
        df.at[idx, "matched_lon"] = lon
        df.at[idx, "matched_lat"] = lat
        df.at[idx, "distance_to_route_m"] = cand.distance_m
        df.at[idx, "match_status"] = "matched"
        df.at[idx, "group_id"] = group_by_oi[oi]
        df.at[idx, "path_m"] = path_m_by_oi[oi]
        df.at[idx, "matched_edge_u"] = cand.edge[0]
        df.at[idx, "matched_edge_v"] = cand.edge[1]
        df.at[idx, "matched_edge_key"] = cand.edge[2]

    # ── Interpolate remaining rows, per group, in network path-distance
    #    space (path_m), time-weighted between bracketing anchors ──────────
    df_ts = df.set_index("original_index")["timestamp"]
    group_anchor_ns: dict[int, np.ndarray] = {}
    group_anchor_path: dict[int, np.ndarray] = {}
    group_anchor_oi: dict[int, np.ndarray] = {}
    for g, oi_list in groups.items():
        ns = np.array(
            [int(pd.Timestamp(df_ts.loc[oi]).value) for oi in oi_list], dtype=np.int64
        )
        pm = np.array([path_m_by_oi[oi] for oi in oi_list], dtype=float)
        order = np.argsort(ns)
        group_anchor_ns[g] = ns[order]
        group_anchor_path[g] = pm[order]
        group_anchor_oi[g] = np.array(oi_list, dtype=int)[order]

    has_ootb_col = "out_of_time_bounds" in df.columns
    for idx in df.index[~anchor_mask]:
        if not df.at[idx, "timestamp_valid"]:
            df.at[idx, "match_status"] = "unresolved_missing_time"
            continue
        if has_ootb_col and bool(df.at[idx, "out_of_time_bounds"]):
            df.at[idx, "match_status"] = "unresolved_out_of_time_bounds"
            continue

        seg_id = int(df.at[idx, "segment_id"])
        g = seg_id_to_group.get(seg_id)
        if g is None:
            df.at[idx, "match_status"] = "unresolved_large_gap"
            continue

        anchor_ns = group_anchor_ns[g]
        anchor_path = group_anchor_path[g]
        anchor_oi_arr = group_anchor_oi[g]

        row_ns = int(pd.Timestamp(df.at[idx, "timestamp"]).value)
        before_mask = anchor_ns < row_ns
        after_mask = anchor_ns > row_ns

        if not before_mask.any() or not after_mask.any():
            # Can't bracket within this group — no extrapolation past a group
            # boundary, and no snap to the nearest anchor either: that would
            # be a guess at this row's position, not a measurement of it.
            df.at[idx, "match_status"] = "unresolved_large_gap"
            continue

        prev_ns = int(anchor_ns[before_mask].max())
        next_ns = int(anchor_ns[after_mask].min())
        dt_before = (row_ns - prev_ns) / 1e9
        dt_after = (next_ns - row_ns) / 1e9
        # total_dt > 0 always: prev_ns and next_ns come from strict < / >
        # masks against row_ns, so next_ns > prev_ns by construction.
        total_dt = (next_ns - prev_ns) / 1e9
        if total_dt > config.max_interpolation_gap_s:
            # The real elapsed time between the two anchors bracketing this
            # row — not just this row's distance to the nearer one — is what
            # bounds confidence in the interpolated position. A per-side
            # check alone would accept e.g. dt_before=dt_after=250s (500s
            # total) inside a long bridged cross-segment gap, where nothing
            # was actually observed for most of that span.
            df.at[idx, "match_status"] = "unresolved_large_gap"
            continue

        prev_pos = int(np.where(anchor_ns == prev_ns)[0][0])
        next_pos = int(np.where(anchor_ns == next_ns)[0][0])
        prev_oi = int(anchor_oi_arr[prev_pos])
        next_oi = int(anchor_oi_arr[next_pos])
        prev_path = float(anchor_path[prev_pos])
        next_path = float(anchor_path[next_pos])

        target_extra_m = (dt_before / total_dt) * (next_path - prev_path)

        prev_cand = link_by_oi[prev_oi].cand
        next_link = link_by_oi[next_oi]  # bridge_edges: prev anchor -> this anchor
        x, y, landed_edge = _walk_forward(prev_cand, next_link.bridge_edges, target_extra_m, index)
        lon, lat = transformer_inv.transform(x, y)

        df.at[idx, "matched_lon"] = lon
        df.at[idx, "matched_lat"] = lat
        df.at[idx, "group_id"] = g
        df.at[idx, "path_m"] = prev_path + target_extra_m
        df.at[idx, "matched_edge_u"] = landed_edge[0]
        df.at[idx, "matched_edge_v"] = landed_edge[1]
        df.at[idx, "matched_edge_key"] = landed_edge[2]
        df.at[idx, "interpolated"] = True
        df.at[idx, "match_status"] = "interpolated"

    # Safety net: any row somehow still unset (shouldn't happen given the
    # branches above are exhaustive)
    still_missing = df["match_status"].isna()
    no_ts = still_missing & ~df["timestamp_valid"]
    df.loc[no_ts, "match_status"] = "unresolved_missing_time"
    df.loc[still_missing & ~no_ts, "match_status"] = "unresolved_large_gap"

    return df.sort_values("original_index", kind="stable").reset_index(drop=True)


def match_trip_csv_to_network(
    csv_path: str | Path,
    trip_id: str,
    index: GraphIndex,
    config: MatchConfig,
    output_csv_path: str | Path | None = None,
    dist_cache: NetworkDistanceCache | None = None,
    transformer_inv: pyproj.Transformer | None = None,
    route: Route1D | None = None,
    polygon=None,
    route_edges: set[EdgeKey] | None = None,
) -> pd.DataFrame:
    """Load one trip CSV, match it to the road network, and optionally save
    output. Thin I/O wrapper around match_trip_to_network_rows.
    """
    trip_df = pd.read_csv(csv_path)
    result = match_trip_to_network_rows(
        trip_df, trip_id, index, config, dist_cache, transformer_inv,
        route=route, polygon=polygon, route_edges=route_edges,
    )
    if output_csv_path is not None:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
    return result
