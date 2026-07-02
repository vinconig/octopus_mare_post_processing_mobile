"""Infer the ground-truth route for the bairro from all recorded trips.

Pipeline: load bairro polygon → fetch/cache road graph → load all trips →
HMM-match each trip to the road network → count edge traversals → cluster
trip endpoints → highest-support path → export GPX + frequency visualization.

The resulting GPX is the track that the existing single-CSV matching pipeline
(gps_route_matcher.pipeline) can then match future recordings against.
"""

import argparse
import json
import os
import webbrowser

from gps_route_matcher.config import MatchConfig
from gps_route_matcher.io import load_kml_polygon, write_gpx_route
from gps_route_matcher.netmatch import build_graph_index, match_all_trips
from gps_route_matcher.paths import PROJECT_ROOT
from gps_route_matcher.roadnet import get_road_graph
from gps_route_matcher.route_inference import (
    RouteInferenceConfig,
    build_edge_counts,
    determine_endpoints,
    determine_via_node,
    edge_sequence_to_points,
    highest_support_path,
    path_count_diagnostics,
    save_edge_counts,
)
from gps_route_matcher.visualization import _BAIRRO_KML_PATH, _load_all_tracks, generate_tracks_matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-trips", type=int, default=None,
                        help="Only match the first N trips (for quick tests)")
    parser.add_argument("--folders", nargs="+", default=None,
                        help="Restrict to these data folders (e.g. OM_2). Each device "
                             "tends to have its own route, so per-folder runs usually "
                             "give a much cleaner result than the all-device default.")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Frequency dominance in edge cost (default 2.0)")
    parser.add_argument("--start", type=str, default=None, metavar="LAT,LON",
                        help='Manual route start, e.g. --start "-22.8544,-43.2416". '
                             "Skips start clustering; snapped to the nearest graph node.")
    parser.add_argument("--end", type=str, default=None, metavar="LAT,LON",
                        help="Manual route end (same format as --start)")
    parser.add_argument("--via", type=str, default=None, metavar="LAT,LON",
                        help="Force the path through this point (two Dijkstra legs); "
                             "also what loop routes use automatically")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open the visualization when done")
    args = parser.parse_args()

    def parse_latlon(s: str | None) -> tuple[float, float] | None:
        if s is None:
            return None
        lat, lon = (float(p) for p in s.split(","))
        return (lat, lon)

    out_dir = PROJECT_ROOT / "output"
    inter_dir = out_dir / "intermediate"
    inter_dir.mkdir(parents=True, exist_ok=True)
    # Per-folder runs write to distinct files so they don't overwrite each other.
    tag = "_".join(args.folders) if args.folders else "all"

    match_cfg = MatchConfig()          # column names already default to the CSV schema
    infer_cfg = RouteInferenceConfig(
        gamma=args.gamma,
        start_latlon=parse_latlon(args.start),
        end_latlon=parse_latlon(args.end),
        via_latlon=parse_latlon(args.via),
    )

    print("1/6 Loading bairro polygon and road graph …")
    polygon = load_kml_polygon(str(_BAIRRO_KML_PATH)).union_all()
    graph = get_road_graph(polygon, network_type="all")
    print(f"    graph: {len(graph.nodes)} nodes, {len(graph.edges)} directed edges")

    print("2/6 Loading trips …")
    tracks = _load_all_tracks(polygon)
    if args.folders:
        tracks = tracks[tracks["source_folder"].isin(args.folders)]
    trip_ids = tracks["track_id"].unique()
    if args.max_trips:
        trip_ids = trip_ids[: args.max_trips]
        tracks = tracks[tracks["track_id"].isin(trip_ids)]
    print(f"    {len(trip_ids)} trips, {len(tracks)} points")

    print("3/6 Matching trips to the road network …")
    index = build_graph_index(graph)
    results = match_all_trips(tracks, index, match_cfg)
    matched = [r for r in results if r.edge_sequence]
    print(f"    matched {len(matched)}/{len(results)} trips "
          f"({sum(r.n_bridged_gaps for r in results)} gaps bridged)")

    # Persist per-trip edge sequences for diagnostics (trip identity kept).
    seq_path = inter_dir / f"trip_edge_sequences_{tag}.json"
    with open(seq_path, "w") as fh:
        json.dump(
            {r.trip_id: r.edge_sequence for r in matched}, fh, default=list
        )
    print(f"    per-trip edge sequences → {seq_path}")

    print("4/6 Building frequency-weighted graph …")
    counted = build_edge_counts(graph, matched)
    counts_path = save_edge_counts(counted, inter_dir / f"edge_counts_{tag}.geojson")
    print(f"    edge counts → {counts_path}")

    # Network-wide traversal summary for sanity checking the evidence base.
    import numpy as np
    counts = np.array([d["count"] for _, _, d in counted.edges(data=True)])
    traversed = counts[counts > 0]
    print(f"    {len(traversed)}/{len(counts)} edges traversed | "
          f"count p50={np.median(traversed):.0f} p90={np.percentile(traversed, 90):.0f} "
          f"max={traversed.max()}" if len(traversed) else "    no edges traversed")

    print("5/6 Determining endpoints and highest-support path …")
    start_node, end_node = determine_endpoints(matched, tracks, index, infer_cfg)
    print(f"    start node: {start_node}, end node: {end_node}")

    # Loop routes (recordings that start and end at the same base) would give
    # an empty direct path — route via a data-driven turnaround point instead.
    via_node = None
    if start_node == end_node or infer_cfg.via_node or infer_cfg.via_latlon:
        via_node = determine_via_node(matched, index, infer_cfg)
        print(f"    loop route detected — via node: {via_node}")

    path_edges = highest_support_path(counted, start_node, end_node, infer_cfg, via_node)
    path_count_diagnostics(counted, path_edges)

    print("6/6 Exporting GPX and visualization …")
    route_points = edge_sequence_to_points(graph, path_edges)
    gpx_path = PROJECT_ROOT / "routes" / f"Inferred_Route_{tag}.gpx"
    write_gpx_route(route_points, str(gpx_path), name=f"Inferred route ({tag})")
    print(f"    GPX → {gpx_path}")

    viz_path = generate_tracks_matched(
        counted, str(out_dir / f"tracks_matched_{tag}.html"), route_points
    )
    print(f"    visualization → {viz_path}")

    if not args.no_browser:
        webbrowser.open(f"file://{viz_path}")


if __name__ == "__main__":
    main()
