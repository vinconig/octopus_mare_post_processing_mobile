"""Assemble a Zenodo-ready deposit from the two most analysis-ready pipeline
outputs: data/OM_n_d_matched/ (point-level matched CSVs) and
data/OM_n_e_edge_analysis/ (per-edge aggregated CSVs).

Per track: read every matched/edge-analysis CSV whose filename date falls
within [_EXPORT_DATE_MIN, _EXPORT_DATE_MAX] (inclusive; trips outside this
window are left out of the package entirely), drop matched rows with
match_status == "unresolved_out_of_time_bounds" (never placed on the road
network — not usable data for an external consumer), apply the column
policy in gps_route_matcher.zenodo_columns (drop pipeline-internal columns,
keep documented data + QA flags), and write the result to
output/zenodo_package/matched/ and output/zenodo_package/edge_analysis/
under the same filenames. Also copies the companion files needed to make the
export reproducible/self-contained: each track's
routes/route_metadata/<track>_inferred_route.json (the edge_order/u,v,key
anchor), roadnetwork/bairro_walk.graphml (the OSM snapshot those ids refer
to), and calibration/calibration_equations.csv +
calibration/cross_validation_metrics.csv (needed to interpret PM2.5_cal).
Also generates interactive_map.html, the same Points / Street Segment
browser as visualization.generate_interactive_map_all_data_incl_edges, but
embedding only the data actually shipped in this package (see
visualization.generate_zenodo_interactive_map) so it's viewable standalone
from the deposit without the rest of this repo, and with its calendar
navigation confined to [_EXPORT_DATE_MIN, _EXPORT_DATE_MAX].

Finally writes a generated data_dictionary.csv, a PROVENANCE.json recording
the code commit and OSM graph snapshot version, and a CC-BY-4.0 LICENSE.

Does not write output/zenodo_package/README.md — the narrative dataset
description is human-authored (see ZENODO_README.md at the repo root) and is
not synthesized from column names.

Requires routes/route_metadata/<track>_inferred_route.json to exist for each
track processed (run infer_route.py first if a track is skipped).
"""

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from gps_route_matcher.paths import (
    CALIBRATION_EQUATIONS_PATH,
    EDGE_ANALYSIS_SUFFIX,
    MATCHED_SUFFIX,
    PROJECT_ROOT,
    TRACKS,
)
from gps_route_matcher.recording_naming import _parse_recording_filename
from gps_route_matcher.roadnet import ROADNETWORK_DIR, load_road_graph
from gps_route_matcher.visualization import generate_zenodo_interactive_map
from gps_route_matcher.zenodo_columns import (
    EDGE_ANALYSIS_COLUMNS,
    MATCHED_COLUMNS,
    apply_column_config,
    build_data_dictionary_rows,
)
from map_matching import _iter_csv_files

# Rows the matcher never placed on the road network at all (as opposed to
# "matched"/"interpolated", which do carry a usable position) — excluded
# from the export rather than shipped as unusable rows.
_EXCLUDED_MATCH_STATUS = "unresolved_out_of_time_bounds"

# This Zenodo deposit is scoped to one field campaign window: only trip CSVs
# whose filename date falls in [_EXPORT_DATE_MIN, _EXPORT_DATE_MAX] (inclusive)
# are exported; every other trip is left out of the package entirely.
_EXPORT_DATE_MIN = "20251201"
_EXPORT_DATE_MAX = "20260430"


def _in_export_window(csv_path: Path) -> bool:
    """True if csv_path's filename date (see recording_naming.py) falls
    within [_EXPORT_DATE_MIN, _EXPORT_DATE_MAX]. Files whose name doesn't
    parse at all are excluded rather than assumed in-range."""
    parsed = _parse_recording_filename(csv_path.stem)
    if parsed is None:
        return False
    return _EXPORT_DATE_MIN <= parsed["date"] <= _EXPORT_DATE_MAX

_CC_BY_4_0_TEXT = """Creative Commons Attribution 4.0 International (CC BY 4.0)

This dataset is licensed under the Creative Commons Attribution 4.0
International License. To view a copy of this license, visit
https://creativecommons.org/licenses/by/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.

You are free to:
  Share   — copy and redistribute the material in any medium or format
  Adapt   — remix, transform, and build upon the material
  for any purpose, even commercially.

Under the following terms:
  Attribution — You must give appropriate credit, provide a link to the
  license, and indicate if changes were made. You may do so in any
  reasonable manner, but not in any way that suggests the licensor endorses
  you or your use.

No additional restrictions — You may not apply legal terms or technological
measures that legally restrict others from doing anything the license
permits.

Full legal text: https://creativecommons.org/licenses/by/4.0/legalcode
"""


def _write_provenance(out_dir: Path, n_matched_files: int, n_edge_analysis_files: int) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    try:
        pipeline_version = version("gps-route-matcher")
    except PackageNotFoundError:
        pipeline_version = "unknown"

    graph = load_road_graph(ROADNETWORK_DIR / "bairro_walk.graphml")
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "pipeline_version": pipeline_version,
        "osm_graph_snapshot": {
            "source_file": "roadnetwork/bairro_walk.graphml",
            "created_date": graph.graph.get("created_date"),
            "created_with": graph.graph.get("created_with"),
        },
        "n_matched_files": n_matched_files,
        "n_edge_analysis_files": n_edge_analysis_files,
    }
    (out_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))


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
    out_dir = out_dir or (PROJECT_ROOT / "output" / "zenodo_package")

    matched_out = out_dir / "matched"
    edge_analysis_out = out_dir / "edge_analysis"
    route_metadata_out = out_dir / "route_metadata"
    roadnetwork_out = out_dir / "roadnetwork"
    calibration_out = out_dir / "calibration"
    for d in (matched_out, edge_analysis_out, route_metadata_out, roadnetwork_out, calibration_out):
        d.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for d in (matched_out, edge_analysis_out):
            for track in target_tracks:
                for stale_csv in d.glob(f"{track}_*.csv"):
                    stale_csv.unlink()

    print(f"1/5 Exporting matched + edge_analysis CSVs for {target_tracks} …")
    n_matched_files = 0
    n_edge_analysis_files = 0
    n_files_skipped = 0
    n_files_out_of_window = 0
    n_matched_rows_excluded = 0
    matched_shipped_cols: dict[str, None] = {}
    edge_analysis_shipped_cols: dict[str, None] = {}

    for track in target_tracks:
        matched_in = PROJECT_ROOT / "data" / f"{track}{MATCHED_SUFFIX}"
        edge_analysis_in = PROJECT_ROOT / "data" / f"{track}{EDGE_ANALYSIS_SUFFIX}"
        metadata_path = PROJECT_ROOT / "routes" / "route_metadata" / f"{track}_inferred_route.json"

        if not metadata_path.is_file():
            print(f"    skip {track}: route metadata not found (run infer_route.py first)")
            continue
        shutil.copy2(metadata_path, route_metadata_out / metadata_path.name)

        if matched_in.exists():
            for csv_path in _iter_csv_files(matched_in):
                if not _in_export_window(csv_path):
                    n_files_out_of_window += 1
                    continue
                out_path = matched_out / csv_path.name
                if not overwrite and out_path.exists():
                    n_files_skipped += 1
                    continue
                raw_df = pd.read_csv(csv_path)
                excluded_mask = raw_df["match_status"] == _EXCLUDED_MATCH_STATUS
                n_matched_rows_excluded += int(excluded_mask.sum())
                df = apply_column_config(raw_df[~excluded_mask], MATCHED_COLUMNS)
                df.to_csv(out_path, index=False)
                matched_shipped_cols.update(dict.fromkeys(df.columns))
                n_matched_files += 1
        else:
            print(f"    skip {track} matched: not found ({matched_in})")

        if edge_analysis_in.exists():
            for csv_path in _iter_csv_files(edge_analysis_in):
                if not _in_export_window(csv_path):
                    n_files_out_of_window += 1
                    continue
                out_path = edge_analysis_out / csv_path.name
                if not overwrite and out_path.exists():
                    n_files_skipped += 1
                    continue
                df = apply_column_config(pd.read_csv(csv_path), EDGE_ANALYSIS_COLUMNS)
                df.to_csv(out_path, index=False)
                edge_analysis_shipped_cols.update(dict.fromkeys(df.columns))
                n_edge_analysis_files += 1
        else:
            print(f"    skip {track} edge_analysis: not found ({edge_analysis_in})")

    print("2/5 Copying companion files (OSM graph snapshot, calibration) …")
    shutil.copy2(ROADNETWORK_DIR / "bairro_walk.graphml", roadnetwork_out / "bairro_walk.graphml")
    shutil.copy2(CALIBRATION_EQUATIONS_PATH, calibration_out / CALIBRATION_EQUATIONS_PATH.name)
    cross_val_path = CALIBRATION_EQUATIONS_PATH.parent / "cross_validation_metrics.csv"
    shutil.copy2(cross_val_path, calibration_out / cross_val_path.name)

    print("3/5 Generating interactive_map.html from the exported data …")
    if any(_iter_csv_files(matched_out)) and any(_iter_csv_files(edge_analysis_out)):
        map_path = generate_zenodo_interactive_map(
            matched_dir=matched_out,
            edge_analysis_dir=edge_analysis_out,
            output_path=str(out_dir / "interactive_map.html"),
            calendar_min=(int(_EXPORT_DATE_MIN[:4]), int(_EXPORT_DATE_MIN[4:6])),
            calendar_max=(int(_EXPORT_DATE_MAX[:4]), int(_EXPORT_DATE_MAX[4:6])),
        )
        print(f"    wrote {map_path}")
    else:
        print("    skip: no exported matched/edge_analysis CSVs to embed")

    print("4/5 Writing data dictionary, provenance, and license …")
    dictionary_rows = build_data_dictionary_rows(
        "matched", pd.DataFrame(columns=list(matched_shipped_cols)), MATCHED_COLUMNS
    ) + build_data_dictionary_rows(
        "edge_analysis", pd.DataFrame(columns=list(edge_analysis_shipped_cols)), EDGE_ANALYSIS_COLUMNS
    )
    pd.DataFrame(dictionary_rows).to_csv(out_dir / "data_dictionary.csv", index=False)

    _write_provenance(out_dir, n_matched_files, n_edge_analysis_files)
    (out_dir / "LICENSE").write_text(_CC_BY_4_0_TEXT)

    print("5/5 Summary")
    print(f"    matched files: {n_matched_files}, edge_analysis files: {n_edge_analysis_files}, "
          f"skipped (already existed): {n_files_skipped}, "
          f"skipped (outside {_EXPORT_DATE_MIN}-{_EXPORT_DATE_MAX} window): {n_files_out_of_window}")
    print(f"    matched rows excluded (match_status == {_EXCLUDED_MATCH_STATUS!r}): "
          f"{n_matched_rows_excluded}")
    print(f"    wrote to {out_dir}")
    print("    NOTE: README.md is not generated — finish ZENODO_README.md at the repo root "
          "(fill in its [PLACEHOLDER: ...] sections) and copy it into the package before uploading.")


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
