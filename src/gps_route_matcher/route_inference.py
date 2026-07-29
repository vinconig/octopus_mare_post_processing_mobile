"""Infer a ground-truth route from many matched trips.

Steps:
  3. Frequency-weighted graph — count how often each directed edge was traversed.
  4. Canonical start/end — cluster trip endpoints, snap to graph nodes.
  5. Highest-support path — Dijkstra with cost = length / (count + 1)^gamma.

KNOWN LIMITATION: edges near common recording start points are systematically
undercounted relative to their true popularity — recordings that begin
mid-route never had the chance to record the first edges. We accept this for
now; it mainly affects counts near the canonical start, and the (count + 1)
smoothing in the cost keeps such edges usable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyproj
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from .netmatch import EdgeKey, GraphIndex, TripMatchResult


@dataclass
class RouteInferenceConfig:
    """Parameters for ground-truth route inference (Steps 3–5)."""

    gamma: float = 2.0                  # frequency dominance in the edge cost
    endpoint_cluster_eps_m: float = 75.0   # radius for endpoint clustering
    endpoint_min_cluster_size: int = 3
    # Optional explicit overrides — when set, clustering is skipped.
    start_latlon: tuple[float, float] | None = None   # (lat, lon)
    end_latlon: tuple[float, float] | None = None
    start_node: int | None = None
    end_node: int | None = None
    # Optional via point for loop routes (start == end). When None and the
    # route is a loop, the densest cluster of trip midpoints is used.
    via_latlon: tuple[float, float] | None = None
    via_node: int | None = None
    # Trips whose matched length is below this are excluded from endpoint /
    # midpoint clustering (they are usually stationary recordings at a base).
    # They still contribute to edge traversal counts.
    min_trip_length_m: float = 500.0


# ── Step 3: frequency-weighted graph ─────────────────────────────────────────


def build_edge_counts(
    graph: nx.MultiDiGraph,
    trip_results: list[TripMatchResult],
) -> nx.MultiDiGraph:
    """Copy the graph and set a 'count' attribute on every directed edge.

    Each traversal of a directed edge (u, v, key) in any trip's matched edge
    sequence increments its count. Full and partial trips contribute equally —
    a partial trip is simply evidence for fewer edges.
    """
    counted = graph.copy()
    nx.set_edge_attributes(counted, 0, "count")

    skipped = 0
    for res in trip_results:
        for u, v, k in res.edge_sequence:
            if counted.has_edge(u, v, k):
                counted[u][v][k]["count"] += 1
            else:
                skipped += 1
    if skipped:
        print(f"  warning: {skipped} traversals referenced edges missing from the graph")
    return counted


def edge_counts_to_gdf(counted_graph: nx.MultiDiGraph):
    """Return the edges of the counted graph as a GeoDataFrame (for inspection
    / heatmap plotting independent of the path-finding step)."""
    import osmnx as ox

    _, edges = ox.graph_to_gdfs(counted_graph)
    return edges[["count", "length", "geometry"]].reset_index()


def save_edge_counts(counted_graph: nx.MultiDiGraph, path: str | Path) -> Path:
    """Persist per-edge traversal counts as GeoJSON for debugging/plotting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = edge_counts_to_gdf(counted_graph)
    gdf.to_file(path, driver="GeoJSON")
    return path


# ── Step 4: canonical start / end ────────────────────────────────────────────


def _radius_cluster(points_xy: np.ndarray, eps_m: float) -> np.ndarray:
    """Simple DBSCAN-like clustering: connected components of the eps-ball
    graph (union-find over cKDTree pairs). Avoids a scikit-learn dependency.

    Returns an integer label per point (no noise label; singletons are their
    own cluster).
    """
    n = len(points_xy)
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree = cKDTree(points_xy)
    for i, j in tree.query_pairs(r=eps_m):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def _densest_cluster_centroid(
    points_lonlat: list[tuple[float, float]],
    eps_m: float,
    min_size: int,
    crs_metric: str,
    what: str,
) -> tuple[float, float]:
    """Centroid (lon, lat) of the densest endpoint cluster, with size logging."""
    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs_metric, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs_metric, "EPSG:4326", always_xy=True)

    arr = np.asarray(points_lonlat)
    xs, ys = fwd.transform(arr[:, 0], arr[:, 1])
    xy = np.column_stack([xs, ys])

    labels = _radius_cluster(xy, eps_m)
    sizes = pd.Series(labels).value_counts()

    # Log cluster sizes so a broken "common route" assumption is visible
    # (e.g. trips split 50/50 between two different starts).
    top = sizes.head(5)
    print(f"  {what} clusters (top 5 of {sizes.size}): "
          + ", ".join(f"{int(c)} pts" for c in top.values)
          + f"  (total {len(arr)} trips)")
    if sizes.iloc[0] < min_size:
        print(f"  warning: densest {what} cluster has only {int(sizes.iloc[0])} "
              f"points (< {min_size}) — canonical {what} may be unreliable")
    if sizes.size > 1 and sizes.iloc[1] >= 0.5 * sizes.iloc[0]:
        print(f"  warning: second-densest {what} cluster is ≥ half the densest — "
              f"the 'common route' assumption may not hold")

    best = sizes.index[0]
    cx, cy = xy[labels == best].mean(axis=0)
    lon, lat = inv.transform(cx, cy)
    return float(lon), float(lat)


def determine_endpoints(
    trip_results: list[TripMatchResult],
    tracks_df: pd.DataFrame,
    index: GraphIndex,
    config: RouteInferenceConfig,
    trip_id_col: str = "track_id",
) -> tuple[int, int]:
    """Return (start_node, end_node) on the road graph.

    Uses explicit overrides from the config when given; otherwise clusters the
    first/last valid GPS fix of every trip and snaps the densest cluster's
    centroid to the nearest graph node.
    """
    import osmnx as ox

    if config.start_node is not None and config.end_node is not None:
        return config.start_node, config.end_node

    def snap(lon: float, lat: float) -> int:
        # Snap on the projected graph: nearest_nodes then uses a scipy KD-tree
        # (the unprojected variant would require scikit-learn).
        fwd = pyproj.Transformer.from_crs("EPSG:4326", index.crs_metric, always_xy=True)
        x, y = fwd.transform(lon, lat)
        return int(ox.distance.nearest_nodes(index.graph_proj, X=x, Y=y))

    start_lonlat: tuple[float, float] | None = None
    end_lonlat: tuple[float, float] | None = None
    if config.start_latlon is not None:
        start_lonlat = (config.start_latlon[1], config.start_latlon[0])
    if config.end_latlon is not None:
        end_lonlat = (config.end_latlon[1], config.end_latlon[0])

    if start_lonlat is None or end_lonlat is None:
        starts: list[tuple[float, float]] = []
        ends: list[tuple[float, float]] = []
        # Stationary/near-empty recordings would swamp the clusters with the
        # recording base's location — only real traversals define the anchors.
        long_trips = [
            r for r in trip_results
            if r.edge_sequence and r.matched_length_m >= config.min_trip_length_m
        ]
        print(f"  endpoint clustering uses {len(long_trips)}/{len(trip_results)} trips "
              f"(≥ {config.min_trip_length_m:.0f} m matched)")
        matched_ids = {r.trip_id for r in long_trips}
        for tid, trip in tracks_df.groupby(trip_id_col, sort=False):
            if str(tid) not in matched_ids:
                continue
            valid = trip[(trip["Latitude"] != 0) | (trip["Longitude"] != 0)]
            if valid.empty:
                continue
            starts.append((float(valid["Longitude"].iloc[0]), float(valid["Latitude"].iloc[0])))
            ends.append((float(valid["Longitude"].iloc[-1]), float(valid["Latitude"].iloc[-1])))

        if start_lonlat is None:
            start_lonlat = _densest_cluster_centroid(
                starts, config.endpoint_cluster_eps_m,
                config.endpoint_min_cluster_size, index.crs_metric, "start",
            )
        if end_lonlat is None:
            end_lonlat = _densest_cluster_centroid(
                ends, config.endpoint_cluster_eps_m,
                config.endpoint_min_cluster_size, index.crs_metric, "end",
            )

    start_node = config.start_node if config.start_node is not None else snap(*start_lonlat)
    end_node = config.end_node if config.end_node is not None else snap(*end_lonlat)
    return start_node, end_node


def determine_via_node(
    trip_results: list[TripMatchResult],
    index: GraphIndex,
    config: RouteInferenceConfig,
) -> int:
    """Data-driven turnaround point for loop routes (start node == end node).

    For each matched trip, take the node at half the cumulative traversed
    length of its edge sequence, cluster those midpoints, and snap the densest
    cluster's centroid to the nearest graph node. Overridable via config.
    """
    import osmnx as ox

    if config.via_node is not None:
        return config.via_node

    if config.via_latlon is not None:
        lon, lat = config.via_latlon[1], config.via_latlon[0]
    else:
        graph = index.graph
        midpoints: list[tuple[float, float]] = []
        for res in trip_results:
            if not res.edge_sequence or res.matched_length_m < config.min_trip_length_m:
                continue
            lengths = [
                float(graph[u][v][k].get("length", 0.0))
                for u, v, k in res.edge_sequence
                if graph.has_edge(u, v, k)
            ]
            total = sum(lengths)
            if total <= 0:
                continue
            acc = 0.0
            mid_node = res.edge_sequence[0][0]
            for (_, v, _), seg_len in zip(res.edge_sequence, lengths):
                acc += seg_len
                if acc >= total / 2.0:
                    mid_node = v
                    break
            midpoints.append(
                (float(graph.nodes[mid_node]["x"]), float(graph.nodes[mid_node]["y"]))
            )
        if not midpoints:
            raise ValueError("Cannot determine a via node: no matched trips")
        lon, lat = _densest_cluster_centroid(
            midpoints, config.endpoint_cluster_eps_m,
            config.endpoint_min_cluster_size, index.crs_metric, "midpoint",
        )

    fwd = pyproj.Transformer.from_crs("EPSG:4326", index.crs_metric, always_xy=True)
    x, y = fwd.transform(lon, lat)
    return int(ox.distance.nearest_nodes(index.graph_proj, X=x, Y=y))


# ── Step 5: highest-support path ─────────────────────────────────────────────


def _edge_cost(data: dict, gamma: float) -> float:
    """Support-weighted cost of ONE edge: length / (count + 1)^gamma.

    All costs stay positive and finite (no negative-cycle problems, unlike a
    naive -log(count) formulation); higher gamma makes traversal frequency
    dominate over raw distance.
    """
    return float(data.get("length", 1.0)) / (float(data.get("count", 0)) + 1.0) ** gamma


def _support_weight(gamma: float):
    """networkx weight callable for a MultiDiGraph.

    IMPORTANT: for multigraphs networkx passes the whole {key: data} dict of
    parallel edges to a callable weight — NOT one edge's attribute dict. A
    function that does data.get("length")/data.get("count") on that outer dict
    silently gets the defaults for every edge (cost 1.0 everywhere), turning
    Dijkstra into plain hop-count minimisation. The min over parallel edges
    must be taken here, explicitly.
    """
    def weight(u, v, keyed_data) -> float:
        return min(_edge_cost(d, gamma) for d in keyed_data.values())
    return weight


def highest_support_path(
    counted_graph: nx.MultiDiGraph,
    start_node: int,
    end_node: int,
    config: RouteInferenceConfig,
    via_node: int | None = None,
) -> list[EdgeKey]:
    """Dijkstra from start to end on the frequency-weighted graph.

    Returns the ordered edge sequence [(u, v, key), ...]; at each hop the
    parallel edge with the lowest support-weighted cost is chosen.

    via_node forces the path through an intermediate node (two Dijkstra legs).
    Required for loop routes (start == end), where the direct shortest path
    would be empty.
    """
    if via_node is not None:
        leg1 = highest_support_path(counted_graph, start_node, via_node, config)
        leg2 = highest_support_path(counted_graph, via_node, end_node, config)
        return leg1 + leg2

    for node, name in ((start_node, "start"), (end_node, "end")):
        if node not in counted_graph:
            raise ValueError(f"{name} node {node} is not in the road graph")

    try:
        node_path = nx.shortest_path(
            counted_graph, start_node, end_node, weight=_support_weight(config.gamma)
        )
    except nx.NetworkXNoPath:
        # Diagnose disconnected components instead of crashing obscurely.
        comps = list(nx.weakly_connected_components(counted_graph))
        comp_of = {}
        for ci, comp in enumerate(comps):
            for n in comp:
                comp_of[n] = ci
        raise ValueError(
            f"No path between start node {start_node} (component "
            f"{comp_of.get(start_node)}) and end node {end_node} (component "
            f"{comp_of.get(end_node)}). Graph has {len(comps)} weakly connected "
            f"components with sizes {sorted((len(c) for c in comps), reverse=True)[:5]}. "
            f"Check polygon clipping / endpoint snapping."
        ) from None

    edges: list[EdgeKey] = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        data = counted_graph.get_edge_data(u, v)
        best_key = min(data, key=lambda kk: _edge_cost(data[kk], config.gamma))
        edges.append((u, v, best_key))
    return edges


def path_count_diagnostics(
    counted_graph: nx.MultiDiGraph, path_edges: list[EdgeKey]
) -> dict[str, float]:
    """Min/median/max traversal count along the path.

    A path leaning heavily on count=0 edges signals that the start/end anchors
    or gamma need adjusting.
    """
    counts = np.array([
        counted_graph[u][v][k].get("count", 0) for u, v, k in path_edges
    ])
    diag = {
        "n_edges": int(len(counts)),
        "count_min": float(counts.min()) if len(counts) else float("nan"),
        "count_median": float(np.median(counts)) if len(counts) else float("nan"),
        "count_max": float(counts.max()) if len(counts) else float("nan"),
        "n_zero_count_edges": int((counts == 0).sum()),
    }
    print(
        f"  path: {diag['n_edges']} edges | count min/median/max = "
        f"{diag['count_min']:.0f}/{diag['count_median']:.0f}/{diag['count_max']:.0f} "
        f"| zero-count edges: {diag['n_zero_count_edges']}"
    )
    if diag["n_zero_count_edges"] > 0.2 * max(diag["n_edges"], 1):
        print("  warning: >20% of path edges were never traversed — "
              "check endpoint anchors or lower gamma")
    return diag


# ── Step 6 helper: edge sequence → point list ────────────────────────────────


def edge_sequence_to_points(
    graph: nx.MultiDiGraph, path_edges: list[EdgeKey]
) -> list[tuple[float, float]]:
    """Concatenate edge geometries into one ordered (lon, lat) point list.

    Uses each edge's shapely geometry when OSMnx kept one, otherwise a straight
    line between node coordinates. Consecutive duplicate points at edge
    boundaries (last point of edge N == first point of edge N+1) are removed.
    """
    points: list[tuple[float, float]] = []
    for u, v, k in path_edges:
        data = graph[u][v][k]
        geom = data.get("geometry")
        if geom is not None:
            coords = [(float(x), float(y)) for x, y in geom.coords]
        else:
            coords = [
                (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])),
                (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])),
            ]
        for pt in coords:
            if not points or points[-1] != pt:
                points.append(pt)
    return points


def _edge_geometry_wkt(graph: nx.MultiDiGraph, u: int, v: int, k: int) -> str:
    """WKT for one edge's geometry, falling back to a straight node-to-node
    line when OSMnx didn't keep one (same fallback as edge_sequence_to_points)."""
    data = graph[u][v][k]
    geom = data.get("geometry")
    if geom is None:
        geom = LineString([
            (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])),
            (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])),
        ])
    return geom.wkt


def save_route_metadata(
    graph: nx.MultiDiGraph, path_edges: list[EdgeKey], route_id: str, path: str | Path
) -> Path:
    """Persist the inferred route's ordered edge sequence + per-edge geometry
    as JSON, so downstream analysis (e.g. edge-level environmental
    aggregation) has a durable source of truth for "which edges, in what
    order, make up this route" without needing to reconstruct it from the
    graph or re-run route inference.

    edges: one entry per position in path_edges (route order preserved,
    including any repeated edge on via-node backtracks).
    edge_geometry: one entry per distinct (u, v, key), keyed as "(u,v,key)".
    """
    edges = [
        {"u": int(u), "v": int(v), "key": int(k), "edge_order": i}
        for i, (u, v, k) in enumerate(path_edges)
    ]

    edge_geometry: dict[str, dict] = {}
    for u, v, k in path_edges:
        key = f"({int(u)},{int(v)},{int(k)})"
        if key in edge_geometry:
            continue
        edge_geometry[key] = {
            "geometry": _edge_geometry_wkt(graph, u, v, k),
            "length": float(graph[u][v][k].get("length", 0.0)),
        }

    data = {"route_id": route_id, "edges": edges, "edge_geometry": edge_geometry}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    return path
