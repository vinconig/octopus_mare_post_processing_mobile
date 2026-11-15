"""Map-match every recorded GPS trip in data/OM_1_c_prepared..OM_4_c_prepared
onto the road network.

Per file: read raw CSV -> match_trip_to_network_rows (network HMM matching +
timestamp inference + network-path-aware interpolation across gaps) -> write
the enriched CSV to a sibling data/OM_n_d_matched/ directory under the same
filename. The road graph, GraphIndex, NetworkDistanceCache and inverse
transformer are built once and reused across all files (same pattern as
scripts/infer_route.py), and only the top-level CSVs directly inside each
data/OM_n_c_prepared folder are processed — nested date/time subfolders hold
stray non-standard leftovers, not raw CSVs, and are skipped entirely.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import pyproj

from gps_route_matcher.analysis import load_route_metadata
from gps_route_matcher.cleaning import TS_CLEAN_COL, TS_INFERRED_COL, TS_VALID_COL
from gps_route_matcher.config import MatchConfig
from gps_route_matcher.edge_bonus import load_bonus_edges
from gps_route_matcher.io import load_gpx_route, load_kml_polygon
from gps_route_matcher.netmatch import (
    NetworkDistanceCache,
    build_graph_index,
    match_trip_to_network_rows,
)
from gps_route_matcher.paths import (
    BAIRRO_KML_PATH,
    MATCHED_SUFFIX,
    PREPARED_SUFFIX,
    PROJECT_ROOT,
    TRACKS,
    warn_if_stale_sibling,
)
from gps_route_matcher.roadnet import get_road_graph
from gps_route_matcher.route import build_route_1d

# Columns that are necessary earlier in the pipeline (cross-file dedup /
# session-splitting for the ts_* trio; the segment_id derivation for dt_s)
# but superseded or unread by the time a trip reaches the matched CSV — see
# .claude/matching_methodology.md. Dropped only here, at the final write:
# map_matching.py re-reads c_prepared from a separate process and re-derives
# timestamp/timestamp_valid/timestamp_inferred from ts_clean/ts_valid/
# ts_inferred if present, so those three must still be on disk in c_prepared.
# (session_id isn't listed here — it's dropped earlier, at the b_walks write
# in make_trip_data.py, and never re-enters the pipeline after that.)
_FINAL_CSV_DROP_COLUMNS = [TS_CLEAN_COL, TS_VALID_COL, TS_INFERRED_COL, "dt_s"]


def _iter_csv_files(folder_path: Path) -> list[Path]:
    """Top-level CSV files only (mirrors visualization._load_all_tracks);
    nested date/time subfolders are never visited."""
    return sorted(
        p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"
    )


def map_match(
    folders: list[str] | None = None,
    max_files: int | None = None,
    force_refresh: bool = False,
    overwrite: bool = True,
) -> None:
    """Match every CSV in the given (or all) data/<track>_c_prepared folders onto
    the road network, writing enriched output to sibling data/<track>_d_matched/
    folders.

    folders defaults to TRACKS (OM_1..OM_4), by bare track name.

    If overwrite is False, CSVs whose matched output already exists in
    data/<track>_d_matched/ are skipped (counted as skipped, not processed),
    so a second run only fills in files that are missing or new.
    """
    print("1/4 Loading bairro polygon and road graph …")
    polygon = load_kml_polygon(str(BAIRRO_KML_PATH)).union_all()
    graph = get_road_graph(polygon, network_type="walk", force_refresh=force_refresh)
    print(f"    graph: {len(graph.nodes)} nodes, {len(graph.edges)} directed edges")

    print("2/4 Building graph index …")
    index = build_graph_index(graph)
    dist_cache = NetworkDistanceCache(index.graph_proj)
    transformer_inv = pyproj.Transformer.from_crs(index.crs_metric, "EPSG:4326", always_xy=True)

    target_tracks = folders if folders else TRACKS
    print(f"3/4 Matching CSV files in {target_tracks} …")

    status_counts: Counter = Counter()
    n_files_processed = 0
    n_files_skipped = 0
    n_bridged_total = 0
    n_left_open_total = 0

    for track in target_tracks:
        in_dir = PROJECT_ROOT / "data" / f"{track}{PREPARED_SUFFIX}"
        if not in_dir.exists():
            print(f"    skip {track}: {in_dir} not found")
            continue
        warn_if_stale_sibling(track, in_dir.name)

        # Per-track final settings (config.FINAL_MATCH_OVERRIDES).
        config = MatchConfig.for_final_matching(track)
        print(f"    {track}: sigma={config.gps_sigma_m:g}, "
              f"off_route_penalty={config.off_route_penalty:g}, "
              f"route_edge_bonus={config.route_edge_bonus:g}")

        out_dir = PROJECT_ROOT / "data" / f"{track}{MATCHED_SUFFIX}"
        out_dir.mkdir(parents=True, exist_ok=True)

        gpx_path = PROJECT_ROOT / "routes" / f"Inferred_Route_{track}.gpx"
        route = build_route_1d(load_gpx_route(str(gpx_path))) if gpx_path.is_file() else None

        # Edges of this track's inferred route get several candidate positions
        # each instead of one, so a continuous match on the correct edge can beat
        # a jump to a side street near a junction (see MatchConfig.
        # route_candidates_per_edge). Missing metadata just means no densification.
        meta_path = PROJECT_ROOT / "routes" / "route_metadata" / f"{track}_inferred_route.json"
        route_edges = (
            {(e["u"], e["v"], e["key"]) for e in load_route_metadata(meta_path)["edges"]}
            if meta_path.is_file()
            else None
        )
        if route_edges is None:
            print(f"    {track}: no route metadata, matching without route densification")

        # Hand-curated route edges that get config.route_edge_bonus (see
        # edge_bonus.py); empty for a track with none, which changes nothing.
        bonus_edges = load_bonus_edges(track)
        if bonus_edges:
            print(f"    {track}: {len(bonus_edges)} curated bonus edge(s)")

        csv_paths = _iter_csv_files(in_dir)
        if max_files is not None:
            csv_paths = csv_paths[:max_files]

        for csv_path in csv_paths:
            out_path = out_dir / csv_path.name
            if not overwrite and out_path.exists():
                n_files_skipped += 1
                continue

            trip_df = pd.read_csv(csv_path)
            if trip_df.empty:
                n_files_skipped += 1
                continue

            result = match_trip_to_network_rows(
                trip_df, csv_path.stem, index, config, dist_cache, transformer_inv,
                route=route, polygon=polygon, route_edges=route_edges,
                bonus_edges=bonus_edges,
            )
            result.drop(columns=_FINAL_CSV_DROP_COLUMNS).to_csv(out_path, index=False)

            status_counts.update(result["match_status"].value_counts(dropna=False).to_dict())
            anchors = result[result["match_status"] == "matched"]
            n_segments = anchors["segment_id"].nunique()
            n_groups = anchors["group_id"].nunique()
            n_bridged_total += max(0, n_segments - n_groups)
            n_left_open_total += max(0, n_groups - 1) if n_groups else 0

            n_files_processed += 1
            if n_files_processed % 25 == 0:
                print(f"    processed {n_files_processed} files …")

    print("4/4 Summary")
    print(f"    files processed: {n_files_processed}, skipped (empty): {n_files_skipped}")
    print(f"    match_status counts: {dict(status_counts)}")
    print(f"    bridged gaps: {n_bridged_total}, left-open gaps: {n_left_open_total}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="+", default=None,
                        help="Restrict to these tracks (e.g. OM_2). Default: all of "
                             "OM_1..OM_4.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Only match the first N CSV files per track (for quick tests)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Bypass both the roadnetwork GraphML cache and osmnx's own "
                             "Overpass response cache, forcing a fresh OSM download")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                        help="Skip CSVs whose matched output already exists, instead of "
                             "overwriting them")
    args = parser.parse_args()

    map_match(folders=args.folders, max_files=args.max_files, force_refresh=args.force_refresh,
              overwrite=args.overwrite)


if __name__ == "__main__":
    main()
