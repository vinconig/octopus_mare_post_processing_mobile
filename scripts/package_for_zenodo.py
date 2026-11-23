"""Assemble a Zenodo-ready deposit from the two most analysis-ready pipeline
outputs: data/OM_n_d_matched/ (point-level matched CSVs) and
data/OM_n_e_edge_analysis/ (per-edge aggregated CSVs).

Per track: read every matched CSV whose filename date falls within
[_EXPORT_DATE_MIN, _EXPORT_DATE_MAX] (inclusive; trips outside this window
are left out of the package entirely), drop every row with
out_of_time_bounds == True (outside the walk's derived active time window —
never usable data for an external consumer; the flag is not shipped as a
column because no shipped row can carry it), apply the column whitelist in
gps_route_matcher.zenodo_columns (a short, lowercase set of columns, with
`timestamp` converted to local time UTC-3 and rounded to whole seconds, and
`temperature_raw`/`humidity_raw`/`pm2.5_raw` being the temperature, humidity
and PM2.5 readings as recorded), and write the result to
output/zenodo_package/data/mobile/. A walk whose shipped rows touch fewer than
MIN_MATCHED_EDGES distinct road-network edges is excluded from the package
entirely (too little network coverage to be a usable trip; see
_n_distinct_matched_edges). The <duration>durmin filename component is
rewritten at export time from the shipped rows' own timestamp span
(floored to whole minutes) rather than carried forward from the walk's
original, uncleaned span — see _real_duration_min — so it reflects what's
actually in the exported file, not the pre-cleaning recording length. The
paired data/mobile_agg_by_edge/ CSV for the same walk (if present) is exported under
that identical, recomputed filename, so the two stay joinable by name; its
own duration is never independently recomputed. It gets the same
whitelist treatment (temperature, humidity and pm2.5 stats only, each
`*_raw_avg`/`*_raw_max`, local-time timestamps). Also copies the companion
files needed to make the export reproducible/self-contained, under metadata/:
each track's routes/route_metadata/<track>_inferred_route.json (the
edge_order/u,v,key anchor) -> metadata/route_metadata/, and
roadnetwork/mare_openstreetmap_road_graph_2026-11-04.graphml (the OSM
snapshot those ids refer to) -> metadata/roadnetwork/.

The package is never curated: it is the complete, deterministic output of the
automatic gates. The human keep/discard/trim/cut pass is a separate step that
reads this package and writes a second one (scripts/apply_curation.py ->
output/zenodo_curated/).

Does not write a README, data dictionary, provenance file or LICENSE — the
license is set in Zenodo's own metadata.

Requires routes/route_metadata/<track>_inferred_route.json to exist for each
track processed (run infer_route.py first if a track is skipped).
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from gps_route_matcher.io import _iter_csv_files
from gps_route_matcher.paths import (
    EDGE_ANALYSIS_SUFFIX,
    MATCHED_SUFFIX,
    PROJECT_ROOT,
    TRACKS,
    ZENODO_PACKAGE_DIR,
)
from gps_route_matcher.recording_naming import _parse_recording_filename, _TRIP_FILENAME_RE
from gps_route_matcher.roadnet import ROAD_GRAPH_PATH
from gps_route_matcher.zenodo_columns import (
    EDGE_ANALYSIS_COLUMNS,
    EDGE_SUBDIR,
    MATCHED_COLUMNS,
    MIN_MATCHED_EDGES,
    MOBILE_SUBDIR,
    ROUTE_METADATA_SUBDIR,
    apply_column_config,
    n_distinct_matched_edges,
    real_duration_min,
)

# This Zenodo deposit is scoped to one field campaign window: only trip CSVs
# whose filename date falls in [_EXPORT_DATE_MIN, _EXPORT_DATE_MAX] (inclusive)
# are exported; every other trip is left out of the package entirely.
_EXPORT_DATE_MIN = "20251201"
_EXPORT_DATE_MAX = "20260430"

# Layout of the deposit, relative to the package root: data/ holds the
# mobile (walked) CSV families plus a folder for the fixed (stationary) sensor
# data, metadata/ everything needed to interpret them.
MATCHED_SUBDIR = MOBILE_SUBDIR  # shared with apply_curation / the panel: see zenodo_columns
EDGE_ANALYSIS_SUBDIR = EDGE_SUBDIR
FIXED_SUBDIR = "data/fixed"  # stationary-sensor data; not exported by this script yet, folder is created empty
ROADNETWORK_SUBDIR = "metadata/roadnetwork"

# MIN_MATCHED_EDGES (the minimum network coverage a walk needs to ship) and the
# two helpers below live in zenodo_columns so apply_curation.py applies the very
# same gate to a walk whose rows a curation edit removed.
_n_distinct_matched_edges = n_distinct_matched_edges
_real_duration_min = real_duration_min


def _in_export_window(csv_path: Path) -> bool:
    """True if csv_path's filename date (see recording_naming.py) falls
    within [_EXPORT_DATE_MIN, _EXPORT_DATE_MAX]. Files whose name doesn't
    parse at all are excluded rather than assumed in-range."""
    parsed = _parse_recording_filename(csv_path.stem)
    if parsed is None:
        return False
    return _EXPORT_DATE_MIN <= parsed["date"] <= _EXPORT_DATE_MAX


def package_for_zenodo(
    folders: list[str] | None = None,
    overwrite: bool = True,
    out_dir: Path | None = None,
) -> None:
    """Assemble a Zenodo-ready deposit for the given (or all) tracks under
    out_dir (default output/zenodo_package/).

    folders defaults to TRACKS (OM_1..OM_4), by bare track name.

    If overwrite is False, CSVs whose export output already exists are
    skipped, so a second run only fills in files that are missing or new.
    If overwrite is True (the default), each target track's pre-existing CSVs
    in matched_out/edge_analysis_out are cleared before this run writes into
    them — otherwise a file that no longer qualifies (e.g. after narrowing
    _EXPORT_DATE_MIN/_EXPORT_DATE_MAX) would linger from a previous run
    instead of disappearing from the package. Scoped to `target_tracks` so a
    partial run (folders=[...]) never deletes other tracks' already-packaged
    data; a track dropped from `folders` therefore keeps its stale files
    until a full run (folders=None) cleans them up.
    """
    target_tracks = folders if folders else TRACKS
    out_dir = out_dir or ZENODO_PACKAGE_DIR

    matched_out = out_dir / MATCHED_SUBDIR
    edge_analysis_out = out_dir / EDGE_ANALYSIS_SUBDIR
    route_metadata_out = out_dir / ROUTE_METADATA_SUBDIR
    roadnetwork_out = out_dir / ROADNETWORK_SUBDIR
    fixed_out = out_dir / FIXED_SUBDIR
    for d in (matched_out, edge_analysis_out, fixed_out, route_metadata_out, roadnetwork_out):
        d.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for d in (matched_out, edge_analysis_out):
            for track in target_tracks:
                for stale_csv in d.glob(f"{track}_*.csv"):
                    stale_csv.unlink()

    print(f"1/3 Exporting matched + edge_analysis CSVs for {target_tracks} …")
    n_matched_files = 0
    n_edge_analysis_files = 0
    n_files_skipped = 0
    n_files_out_of_window = 0
    n_files_too_few_edges = 0
    n_matched_rows_excluded = 0

    for track in target_tracks:
        matched_in = PROJECT_ROOT / "data" / f"{track}{MATCHED_SUFFIX}"
        edge_analysis_in = PROJECT_ROOT / "data" / f"{track}{EDGE_ANALYSIS_SUFFIX}"
        metadata_path = PROJECT_ROOT / "routes" / "route_metadata" / f"{track}_inferred_route.json"

        if not metadata_path.is_file():
            print(f"    skip {track}: route metadata not found (run infer_route.py first)")
            continue
        shutil.copy2(metadata_path, route_metadata_out / metadata_path.name)

        if not matched_in.exists():
            print(f"    skip {track} matched: not found ({matched_in})")
            continue

        # data/mobile drives the export: data/mobile_agg_by_edge is derived from it,
        # so every walk's inclusion, exclusion, and
        # exported filename are decided from its matched data alone, then
        # applied identically to the paired edge_analysis file.
        for csv_path in _iter_csv_files(matched_in):
            if not _in_export_window(csv_path):
                n_files_out_of_window += 1
                continue

            m = _TRIP_FILENAME_RE.match(csv_path.stem)

            raw_df = pd.read_csv(csv_path)
            excluded_mask = raw_df["out_of_time_bounds"].astype(bool)
            n_matched_rows_excluded += int(excluded_mask.sum())
            usable_raw = raw_df[~excluded_mask]
            df = apply_column_config(usable_raw, MATCHED_COLUMNS)

            if _n_distinct_matched_edges(df) < MIN_MATCHED_EDGES:
                n_files_too_few_edges += 1
                continue

            new_name = f"{m['track']}_{m['date']}_{m['period_word']}_{_real_duration_min(df)}durmin.csv"
            out_path = matched_out / new_name
            if not overwrite and out_path.exists():
                n_files_skipped += 1
                continue
            df.to_csv(out_path, index=False)
            n_matched_files += 1

            ea_path = edge_analysis_in / csv_path.name
            if not ea_path.is_file():
                print(f"    skip {track} edge_analysis for {csv_path.name}: not found ({ea_path})")
                continue
            ea_out_path = edge_analysis_out / new_name
            if not overwrite and ea_out_path.exists():
                n_files_skipped += 1
                continue
            ea_df = apply_column_config(pd.read_csv(ea_path), EDGE_ANALYSIS_COLUMNS)
            ea_df.to_csv(ea_out_path, index=False)
            n_edge_analysis_files += 1

    print("2/3 Copying companion files (OSM graph snapshot) …")
    shutil.copy2(ROAD_GRAPH_PATH, roadnetwork_out / ROAD_GRAPH_PATH.name)

    print("3/3 Summary")
    print(f"    matched files: {n_matched_files}, edge_analysis files: {n_edge_analysis_files}, "
          f"skipped (already existed): {n_files_skipped}, "
          f"skipped (outside {_EXPORT_DATE_MIN}-{_EXPORT_DATE_MAX} window): {n_files_out_of_window}")
    print(f"    walks excluded (< {MIN_MATCHED_EDGES} distinct matched edges): {n_files_too_few_edges}")
    print(f"    matched rows excluded (out_of_time_bounds): {n_matched_rows_excluded}")
    print(f"    wrote to {out_dir}")
    print("    NOTE: no README is generated.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="+", default=None,
                        help="Restrict to these tracks (e.g. OM_2). Default: all of "
                             "OM_1..OM_4.")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                        help="Skip CSVs whose export output already exists, instead of "
                             "overwriting them")
    args = parser.parse_args()

    package_for_zenodo(folders=args.folders, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
