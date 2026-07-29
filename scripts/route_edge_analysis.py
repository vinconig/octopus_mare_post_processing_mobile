"""Aggregate every matched CSV in data/OM_1..OM_4_matched onto its route's
inferred edge sequence.

Per file: read matched CSV -> gps_route_matcher.analysis.run_edge_analysis
(restrict to points on the inferred route, aggregate per edge, reindex to
the full route so every edge appears) -> write one row-per-edge CSV to a
sibling data/<folder>_edge_analysis/ directory under the same filename.

Requires routes/route_metadata/<folder>_inferred_route.json to exist for
each folder processed — that file is produced by scripts/infer_route.py
(see gps_route_matcher.route_inference.save_route_metadata). Run
infer_route.py first if a folder is skipped as "route metadata not found".
"""

import argparse
from collections import Counter
from pathlib import Path

from gps_route_matcher.analysis import load_route_metadata, run_edge_analysis, save_edge_analysis
from gps_route_matcher.paths import PROJECT_ROOT
from gps_route_matcher.visualization import _TRACK_FOLDERS
from map_matching import _iter_csv_files


def route_edge_analysis(
    folders: list[str] | None = None,
    max_files: int | None = None,
    overwrite: bool = True,
) -> None:
    """Aggregate every matched CSV in the given (or all) data/OM_n_matched
    folders onto its route's inferred edge sequence, writing one row per
    route edge to sibling data/OM_n_edge_analysis/ folders.

    If overwrite is False, CSVs whose edge-analysis output already exists are
    skipped, so a second run only fills in files that are missing or new.
    """
    target_folders = folders if folders else _TRACK_FOLDERS
    print(f"1/2 Processing matched CSVs in {target_folders} …")

    n_files_processed = 0
    n_files_skipped = 0
    n_edge_rows_total = 0
    n_zero_coverage_edges = 0
    status_counts: Counter = Counter()

    for folder in target_folders:
        in_dir = PROJECT_ROOT / "data" / f"{folder}_matched"
        if not in_dir.exists():
            print(f"    skip {folder}: matched data not found")
            continue

        metadata_path = PROJECT_ROOT / "routes" / "route_metadata" / f"{folder}_inferred_route.json"
        if not metadata_path.is_file():
            print(f"    skip {folder}: route metadata not found (run infer_route.py first)")
            continue
        route_metadata = load_route_metadata(metadata_path)

        out_dir = PROJECT_ROOT / "data" / f"{folder}_edge_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_paths = _iter_csv_files(in_dir)
        if max_files is not None:
            csv_paths = csv_paths[:max_files]

        for csv_path in csv_paths:
            out_path = out_dir / csv_path.name
            if not overwrite and out_path.exists():
                n_files_skipped += 1
                continue

            result = run_edge_analysis(csv_path, route_metadata, route_id=folder)
            save_edge_analysis(result, out_path)

            n_files_processed += 1
            n_edge_rows_total += len(result)
            n_zero_coverage_edges += int((result["n_points"] == 0).sum())
            status_counts[folder] += 1
            if n_files_processed % 25 == 0:
                print(f"    processed {n_files_processed} files …")

    print("2/2 Summary")
    print(f"    files processed: {n_files_processed}, skipped (already existed): {n_files_skipped}")
    print(f"    edge rows written: {n_edge_rows_total}, zero-coverage edge rows: {n_zero_coverage_edges}")
    print(f"    files per folder: {dict(status_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="+", default=None,
                        help="Restrict to these data folders (e.g. OM_2). Default: all of "
                             "OM_1..OM_4.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Only process the first N CSV files per folder (for quick tests)")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                        help="Skip CSVs whose edge-analysis output already exists, instead of "
                             "overwriting them")
    args = parser.parse_args()

    route_edge_analysis(folders=args.folders, max_files=args.max_files, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
