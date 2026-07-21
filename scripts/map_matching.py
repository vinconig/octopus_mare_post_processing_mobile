"""Map-match every recorded GPS trip in data/OM_1..OM_4 onto the road network.

Per file: read raw CSV -> match_trip_to_network_rows (network HMM matching +
timestamp inference + network-path-aware interpolation across gaps) -> write
the enriched CSV to a sibling data/<folder>_matched/ directory under the same
filename. The road graph, GraphIndex, NetworkDistanceCache and inverse
transformer are built once and reused across all files (same pattern as
scripts/infer_route.py), and only the top-level CSVs directly inside each
data/OM_n folder are processed — nested date/time subfolders hold stray
non-standard leftovers, not raw CSVs, and are skipped entirely.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import pyproj

from gps_route_matcher.config import MatchConfig
from gps_route_matcher.io import load_kml_polygon
from gps_route_matcher.netmatch import (
    NetworkDistanceCache,
    build_graph_index,
    match_trip_to_network_rows,
)
from gps_route_matcher.paths import PROJECT_ROOT
from gps_route_matcher.roadnet import get_road_graph
from gps_route_matcher.visualization import _BAIRRO_KML_PATH, _TRACK_FOLDERS


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
    """Match every CSV in the given (or all) data/OM_n folders onto the road
    network, writing enriched output to sibling data/OM_n_matched/ folders.

    If overwrite is False, CSVs whose matched output already exists in
    data/<folder>_matched/ are skipped (counted as skipped, not processed),
    so a second run only fills in files that are missing or new.
    """
    print("1/4 Loading bairro polygon and road graph …")
    polygon = load_kml_polygon(str(_BAIRRO_KML_PATH)).union_all()
    graph = get_road_graph(polygon, network_type="all", force_refresh=force_refresh)
    print(f"    graph: {len(graph.nodes)} nodes, {len(graph.edges)} directed edges")

    print("2/4 Building graph index …")
    index = build_graph_index(graph)
    dist_cache = NetworkDistanceCache(index.graph_proj)
    transformer_inv = pyproj.Transformer.from_crs(index.crs_metric, "EPSG:4326", always_xy=True)
    config = MatchConfig(infer_missing_timestamps=True)

    target_folders = folders if folders else _TRACK_FOLDERS
    print(f"3/4 Matching CSV files in {target_folders} …")

    status_counts: Counter = Counter()
    n_files_processed = 0
    n_files_skipped = 0
    n_bridged_total = 0
    n_left_open_total = 0

    for folder in target_folders: 
        in_dir = PROJECT_ROOT / "data" / folder
        if not in_dir.exists():
            print(f"    skip {folder}: not found")
            continue
        out_dir = PROJECT_ROOT / "data" / f"{folder}_matched"
        out_dir.mkdir(parents=True, exist_ok=True)

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
            )
            result.to_csv(out_path, index=False)

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
                        help="Restrict to these data folders (e.g. OM_2). Default: all of "
                             "OM_1..OM_4.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Only match the first N CSV files per folder (for quick tests)")
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
