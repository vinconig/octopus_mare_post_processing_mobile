"""Infer the ground-truth route for the bairro from all recorded trips.

Pipeline: load bairro polygon → fetch/cache road graph → load all trips →
HMM-match each trip to the road network → count edge traversals → cluster
trip endpoints → highest-support path → export GPX + frequency visualization.

The resulting GPX is the route that scripts/map_matching.py's network matcher
then matches future recordings against.
"""

import argparse
import json
import os
import webbrowser

from gps_route_matcher.config import MatchConfig, RouteInferenceConfig
from gps_route_matcher.io import load_kml_polygon, write_gpx_route
from gps_route_matcher.netmatch import build_graph_index, match_all_trips
from gps_route_matcher.paths import BAIRRO_KML_PATH, PROJECT_ROOT, RAW_SUFFIX, TRACKS, WALKS_SUFFIX, warn_if_stale_sibling
from gps_route_matcher.roadnet import get_road_graph
from gps_route_matcher.route_inference import (
    build_edge_counts,
    determine_endpoints,
    determine_via_node,
    edge_sequence_to_points,
    highest_support_path,
    path_count_diagnostics,
    save_edge_counts,
    save_route_metadata,
)
from gps_route_matcher.visualization import _load_all_tracks, generate_tracks_matched


def _parse_latlon(s: str | None) -> tuple[float, float] | None:
    if s is None:
        return None
    lat, lon = (float(p) for p in s.split(","))
    return (lat, lon)


def infer_route(
    folders: list[str] | None = None,
    stage: str = "walks",
    max_trips: int | None = None,
    gamma: float = 2.0,
    start: str | tuple[float, float] | None = None,
    end: str | tuple[float, float] | None = None,
    via: str | tuple[float, float] | None = None,
    no_browser: bool = False,
    force_refresh: bool = False,
    overwrite: bool = True,
) -> None:
    """Infer the ground-truth route for the bairro from recorded trips.

    Parameters
    ----------
    folders:
        Restrict to these tracks, by bare track name (e.g. ["OM_2"]). Each
        device tends to have its own route, so per-track runs usually give a
        much cleaner result than the all-device default. Default: all of
        OM_1..OM_4.
    stage:
        Which data/ stage to source trips from for this track: "walks" (the
        cleaned data/OM_n_c_walks/ every other stage reads — the default, and
        what scripts/main.py uses) or "raw" (data/OM_n_a_raw/, for a quick
        look before running the cleaning stages). Non-default stages get
        their own tag suffix so they never collide with the canonical route.
    max_trips:
        Only match the first N trips (for quick tests).
    gamma:
        Frequency dominance in edge cost.
    start, end:
        Manual route endpoints, either "LAT,LON" strings or (lat, lon)
        tuples. Skips start/end clustering; snapped to the nearest graph
        node.
    via:
        Force the path through this point (two Dijkstra legs); also what
        loop routes use automatically.
    no_browser:
        Don't open the visualization when done.
    force_refresh:
        Bypass both the roadnetwork GraphML cache and osmnx's own Overpass
        response cache, forcing a fresh OSM download (use after editing OSM
        data).
    overwrite:
        If False and this tag's GPX already exists, skip the run entirely.
    """
    def latlon(v: str | tuple[float, float] | None) -> tuple[float, float] | None:
        return _parse_latlon(v) if isinstance(v, str) else v

    if stage not in ("walks", "raw"):
        raise ValueError(f"stage must be 'walks' or 'raw', got {stage!r}")

    target_tracks = folders if folders else TRACKS
    input_suffix = RAW_SUFFIX if stage == "raw" else WALKS_SUFFIX
    input_folders = [f"{t}{input_suffix}" for t in target_tracks]

    out_dir = PROJECT_ROOT / "output" / "interactivemaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    inter_dir = out_dir / "intermediate"
    inter_dir.mkdir(parents=True, exist_ok=True)
    # Per-track runs write to distinct files so they don't overwrite each
    # other; a non-default stage gets its own tag suffix so an experimental
    # raw-data run never collides with the canonical walks-based route.
    tag = "_".join(target_tracks) if folders else "all"
    if stage != "walks":
        tag = f"{tag}_{stage}"

    gpx_path = PROJECT_ROOT / "routes" / f"Inferred_Route_{tag}.gpx"
    if not overwrite and gpx_path.exists():
        print(f"skip {tag}: {gpx_path} already exists")
        return

    for track in target_tracks:
        warn_if_stale_sibling(track, f"{track}{input_suffix}")

    match_cfg = MatchConfig()          # column names already default to the CSV schema
    infer_cfg = RouteInferenceConfig(
        gamma=gamma,
        start_latlon=latlon(start),
        end_latlon=latlon(end),
        via_latlon=latlon(via),
    )

    print("1/6 Loading bairro polygon and road graph …")
    polygon = load_kml_polygon(str(BAIRRO_KML_PATH)).union_all()
    graph = get_road_graph(polygon, network_type="walk", force_refresh=force_refresh) #walk: bidirectional for one-way streets (pedestrians can walk both ways), still includes service roads/paths
    print(f"    graph: {len(graph.nodes)} nodes, {len(graph.edges)} directed edges")

    print("2/6 Loading trips and preprocessing data …")
    # Trips arrive already carrying their cleaning flags from
    # scripts/prepare_trips.py; the matcher adds only what needs a route,
    # which is precisely what this stage is about to produce.
    tracks = _load_all_tracks(folders=input_folders)
    trip_ids = tracks["track_id"].unique()
    if max_trips:
        trip_ids = trip_ids[:max_trips]
        tracks = tracks[tracks["track_id"].isin(trip_ids)]
    print(f"    {len(trip_ids)} trips, {len(tracks)} points")

    print("3/6 Matching trips to the road network …")
    index = build_graph_index(graph)
    results = match_all_trips(tracks, index, match_cfg, polygon=polygon)
    matched = [r for r in results if r.edge_sequence]
    print(f"    matched {len(matched)}/{len(results)} trips "
          f"({sum(r.n_bridged_gaps for r in results)} gaps bridged, "
          f"{sum(r.n_gaps_left_open for r in results)} left open)")

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
    write_gpx_route(route_points, str(gpx_path), name=f"Inferred route ({tag})")
    print(f"    GPX → {gpx_path}")

    metadata_path = PROJECT_ROOT / "routes" / "route_metadata" / f"{tag}_inferred_route.json"
    save_route_metadata(graph, path_edges, route_id=tag, path=metadata_path)
    print(f"    route metadata → {metadata_path}")

    viz_path = generate_tracks_matched(
        counted, str(out_dir / f"{tag}_edge_traversal_frequency.html"), route_points
    )
    print(f"    visualization → {viz_path}")

    if not no_browser:
        webbrowser.open(f"file://{viz_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-trips", type=int, default=None,
                        help="Only match the first N trips (for quick tests)")
    parser.add_argument("--folders", nargs="+", default=None,
                        help="Restrict to these tracks, by bare track name (e.g. OM_2). "
                             "Each device tends to have its own route, so per-track runs "
                             "usually give a much cleaner result than the all-device default.")
    parser.add_argument("--stage", choices=["walks", "raw"], default="walks",
                        help="Which data/ stage to source trips from (default: walks, "
                             "the cleaned data every other stage reads)")
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
    parser.add_argument("--force-refresh", action="store_true",
                        help="Bypass both the roadnetwork GraphML cache and osmnx's "
                             "own Overpass response cache, forcing a fresh OSM download "
                             "(use after editing OSM data)")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                        help="Skip this run if its GPX output already exists")
    args = parser.parse_args()

    infer_route(
        folders=args.folders,
        stage=args.stage,
        max_trips=args.max_trips,
        gamma=args.gamma,
        start=args.start,
        end=args.end,
        via=args.via,
        no_browser=args.no_browser,
        force_refresh=args.force_refresh,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
