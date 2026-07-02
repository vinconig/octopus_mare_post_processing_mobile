"""HMM map matching of GPS trips onto a road network graph.

This is the road-network counterpart of the single-route matcher in
candidates.py / hmm.py. The observation cleaning and gap handling are reused
from cleaning.py and gaps.py; what changes is the state space:

  single-route matcher : states are positions s_m on ONE polyline.
  network matcher      : states are (edge, offset) positions on ANY edge of
                         a directed road graph.

Emission model (unchanged in spirit from candidates.py):
    emission_cost = perpendicular_distance² / (2 · gps_sigma²)

Transition model (Newson–Krumm style, adapted from hmm.py):
    The plausibility of moving between two candidate positions is judged by
    how much the network (routed) distance exceeds the straight-line distance
    between the two GPS fixes, plus a speed penalty using the real dt between
    observations — exactly like hmm.py, but with network shortest-path
    distance in place of the 1-D Δs along a single route. Large dt therefore
    allows proportionally more distance without penalty, which is what makes
    gap handling work.

Gap handling:
    Within a segment, consecutive observations are connected by the Viterbi
    transition (real dt, no artificial 1-second steps). Across segments
    (dt > max_split_gap_s) trips are NOT dropped: the matched edge before and
    after the gap are bridged with a network shortest path, so partial
    coverage still contributes evidence for every edge actually traversed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
import pyproj
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from .cleaning import clean_gps_dataframe
from .config import MatchConfig
from .gaps import assign_match_segments, detect_gaps

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


def generate_edge_candidates(
    index: GraphIndex,
    lon: float,
    lat: float,
    obs_index: int,
    config: MatchConfig,
) -> list[EdgeCandidate]:
    """Return up to candidate_k candidate edges for one GPS observation.

    Nearby edges are found via the sample-point KD-tree; the point is then
    projected exactly onto each edge geometry. Always returns at least the
    nearest edge, even when it lies beyond candidate_radius_m.
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

    from shapely.geometry import Point

    pt = Point(x, y)
    scored: list[tuple[float, int, float]] = []  # (distance, edge_idx, offset)
    for ei in edge_ids:
        line = index.edge_lines[ei]
        offset = float(line.project(pt))
        nearest = line.interpolate(offset)
        d = float(pt.distance(nearest))
        scored.append((d, int(ei), offset))

    scored.sort(key=lambda t: t[0])
    filtered = [s for s in scored if s[0] <= config.candidate_radius_m]
    if not filtered and scored:
        filtered = [scored[0]]  # keep nearest even if outside the radius
    filtered = filtered[: config.candidate_k]

    sigma2 = config.gps_sigma_m ** 2
    out: list[EdgeCandidate] = []
    for d, ei, offset in filtered:
        line = index.edge_lines[ei]
        pos = line.interpolate(offset)
        out.append(EdgeCandidate(
            obs_index=obs_index,
            edge_idx=ei,
            edge=index.edge_keys[ei],
            offset_m=offset,
            distance_m=d,
            emission_cost=d ** 2 / (2.0 * sigma2),
            x=pos.x,
            y=pos.y,
        ))
    return out


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
        if prev.edge_idx == curr.edge_idx and curr.offset_m >= prev.offset_m:
            return curr.offset_m - prev.offset_m

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

    Two additive terms, mirroring hmm.py's transition model:

    1. Detour term: |network distance − straight-line distance| between the
       GPS fixes (Newson–Krumm). A candidate pair whose routed distance far
       exceeds the direct displacement is an implausible match.
    2. Speed term: identical to hmm.py — only speeds above max_speed_mps are
       penalised, quadratically, so a large dt legitimately allows a large
       distance (this is the whole point of using real dt in gaps).
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


def _viterbi_segment(
    obs: list[tuple[int, pd.Timestamp, float, float]],   # (obs_idx, ts, x, y)
    cands: dict[int, list[EdgeCandidate]],
    dist_cache: NetworkDistanceCache,
    index: GraphIndex,
    config: MatchConfig,
) -> list[EdgeCandidate]:
    """Minimum-cost candidate path for one segment (same algorithm as hmm.py)."""
    steps = [(oi, ts, x, y) for oi, ts, x, y in obs if cands.get(oi)]
    if not steps:
        return []

    first = cands[steps[0][0]]
    costs = np.array([c.emission_cost for c in first])
    back: list[list[int]] = [[-1] * len(first)]
    all_cands: list[list[EdgeCandidate]] = [first]

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
        for kc, cc in enumerate(curr):
            for jc, pc in enumerate(prev):
                nd = dist_cache.route_distance(pc, cc, index.edge_lengths, cutoff)
                c = costs[jc] + network_transition_cost(nd, straight, dt_s, config) \
                    + cc.emission_cost
                if c < new_costs[kc]:
                    new_costs[kc] = c
                    bp[kc] = jc
        costs = new_costs
        back.append(bp.tolist())
        all_cands.append(curr)

    k = int(np.argmin(costs))
    path = [k]
    for t in range(len(all_cands) - 1, 0, -1):
        k = back[t][k]
        path.append(k)
    path.reverse()
    return [all_cands[t][k] for t, k in enumerate(path)]


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


def match_trip_to_network(
    trip_df: pd.DataFrame,
    trip_id: str,
    index: GraphIndex,
    config: MatchConfig,
    dist_cache: NetworkDistanceCache | None = None,
) -> TripMatchResult:
    """Match one trip's GPS points to the road network.

    Reuses clean_gps_dataframe / detect_gaps / assign_match_segments, runs
    Viterbi per segment, expands the chosen candidates into an ordered edge
    sequence, and bridges across-segment gaps with a network shortest path
    instead of dropping the trip.
    """
    if dist_cache is None:
        dist_cache = NetworkDistanceCache(index.graph_proj)

    df = clean_gps_dataframe(trip_df, config)
    df = df.sort_values("timestamp", kind="stable")  # trips must be time-ordered
    df = detect_gaps(df, config)
    df = assign_match_segments(df, config)

    usable = df[df["usable_for_hmm"]]
    n_total = len(df)
    if usable.empty:
        return TripMatchResult(trip_id, [], n_total, 0, 0, 0)

    # Candidates for every usable observation
    cands: dict[int, list[EdgeCandidate]] = {}
    for _, row in usable.iterrows():
        oi = int(row["original_index"])
        cl = generate_edge_candidates(index, float(row["lon"]), float(row["lat"]), oi, config)
        if cl:
            cands[oi] = cl

    # Viterbi per segment
    segment_paths: list[list[EdgeCandidate]] = []
    for seg_id in usable["segment_id"].unique():
        seg = usable[usable["segment_id"] == seg_id]
        obs = []
        for _, row in seg.iterrows():
            oi = int(row["original_index"])
            if oi in cands:
                x, y = index.transformer_fwd.transform(float(row["lon"]), float(row["lat"]))
                obs.append((oi, row["timestamp"], x, y))
        path = _viterbi_segment(obs, cands, dist_cache, index, config)
        if path:
            segment_paths.append(path)

    if not segment_paths:
        return TripMatchResult(trip_id, [], n_total, 0, 0, 0)

    # Expand candidate sequences into edge sequences; bridge between segments
    # via shortest path (gap bridging – partial trips still contribute).
    edge_seq: list[EdgeKey] = []
    n_matched = 0
    n_bridged = 0
    prev_cand: EdgeCandidate | None = None

    for seg_path in segment_paths:
        for cand in seg_path:
            n_matched += 1
            if prev_cand is None:
                edge_seq.append(cand.edge)
            else:
                edge_seq.extend(_edges_between(prev_cand, cand, index.graph_proj))
            prev_cand = cand
        # Segment boundary: the next iteration's first candidate will be
        # connected by _edges_between too — that IS the gap bridge.
        if seg_path is not segment_paths[-1]:
            n_bridged += 1

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
    )


def match_all_trips(
    tracks_df: pd.DataFrame,
    index: GraphIndex,
    config: MatchConfig,
    trip_id_col: str = "track_id",
    verbose: bool = True,
) -> list[TripMatchResult]:
    """Match every trip in the concatenated tracks dataframe.

    Trips are processed independently (grouped by trip_id_col) but share one
    NetworkDistanceCache, since they overwhelmingly traverse the same streets.
    """
    dist_cache = NetworkDistanceCache(index.graph_proj)
    results: list[TripMatchResult] = []
    trip_ids = tracks_df[trip_id_col].unique()

    for i, tid in enumerate(trip_ids):
        trip = tracks_df[tracks_df[trip_id_col] == tid]
        res = match_trip_to_network(trip, str(tid), index, config, dist_cache)
        results.append(res)
        if verbose and (i + 1) % 25 == 0:
            print(f"  matched {i + 1}/{len(trip_ids)} trips")

    return results
