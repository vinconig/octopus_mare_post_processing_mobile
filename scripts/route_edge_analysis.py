"""Aggregate every matched CSV in data/OM_1_d_matched..OM_4_d_matched onto its
route's inferred edge sequence.

Per file: read matched CSV -> gps_route_matcher.analysis.run_edge_analysis
(restrict to points on the inferred route, aggregate per edge, reindex to
the full route so every edge appears) -> write one row-per-edge CSV to a
sibling data/OM_n_e_edge_analysis/ directory under the same filename.

Requires routes/route_metadata/<track>_inferred_route.json to exist for
each track processed — that file is produced by scripts/infer_route.py
(see gps_route_matcher.route_inference.save_route_metadata). Run
infer_route.py first if a track is skipped as "route metadata not found".
"""

import argparse
from collections import Counter
from pathlib import Path

from gps_route_matcher.analysis import load_route_metadata, run_edge_analysis, save_edge_analysis
from gps_route_matcher.paths import (
    EDGE_ANALYSIS_SUFFIX,
    MATCHED_SUFFIX,
    PROJECT_ROOT,
    TRACKS,
    warn_if_stale_sibling,
)
from map_matching import _iter_csv_files


def route_edge_analysis(
    folders: list[str] | None = None,
    max_files: int | None = None,
    overwrite: bool = True,
) -> None:
    """Aggregate every matched CSV in the given (or all) data/<track>_d_matched
    folders onto its route's inferred edge sequence, writing one row per
    route edge to sibling data/<track>_e_edge_analysis/ folders.

    folders defaults to TRACKS (OM_1..OM_4), by bare track name.

    If overwrite is False, CSVs whose edge-analysis output already exists are
    skipped, so a second run only fills in files that are missing or new.
    """
    target_tracks = folders if folders else TRACKS
    print(f"1/2 Processing matched CSVs in {target_tracks} …")

    n_files_processed = 0
    n_files_skipped = 0
    n_edge_rows_total = 0
    n_zero_coverage_edges = 0
    status_counts: Counter = Counter()

    for track in target_tracks:
        in_dir = PROJECT_ROOT / "data" / f"{track}{MATCHED_SUFFIX}"
        if not in_dir.exists():
            print(f"    skip {track}: matched data not found ({in_dir})")
            continue
        warn_if_stale_sibling(track, in_dir.name)

        metadata_path = PROJECT_ROOT / "routes" / "route_metadata" / f"{track}_inferred_route.json"
        if not metadata_path.is_file():
            print(f"    skip {track}: route metadata not found (run infer_route.py first)")
            continue
        route_metadata = load_route_metadata(metadata_path)

        out_dir = PROJECT_ROOT / "data" / f"{track}{EDGE_ANALYSIS_SUFFIX}"
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_paths = _iter_csv_files(in_dir)
        if max_files is not None:
            csv_paths = csv_paths[:max_files]

        for csv_path in csv_paths:
            out_path = out_dir / csv_path.name
            if not overwrite and out_path.exists():
                n_files_skipped += 1
                continue

            result = run_edge_analysis(csv_path, route_metadata)
            save_edge_analysis(result, out_path)

            n_files_processed += 1
            n_edge_rows_total += len(result)
            n_zero_coverage_edges += int((result["n_points"] == 0).sum())
            status_counts[track] += 1
            if n_files_processed % 25 == 0:
                print(f"    processed {n_files_processed} files …")

    print("2/2 Summary")
    print(f"    files processed: {n_files_processed}, skipped (already existed): {n_files_skipped}")
    print(f"    edge rows written: {n_edge_rows_total}, zero-coverage edge rows: {n_zero_coverage_edges}")
    print(f"    files per track: {dict(status_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="+", default=None,
                        help="Restrict to these tracks (e.g. OM_2). Default: all of "
                             "OM_1..OM_4.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Only process the first N CSV files per track (for quick tests)")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                        help="Skip CSVs whose edge-analysis output already exists, instead of "
                             "overwriting them")
    args = parser.parse_args()

    route_edge_analysis(folders=args.folders, max_files=args.max_files, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
