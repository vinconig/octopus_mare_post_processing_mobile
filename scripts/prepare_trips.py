"""Apply the route-independent GPS cleaning to every trip, once.

data/OM_n_trips_split/ -> data/OM_n_trips/

The output keeps the plain data/OM_n_trips/ name because it is what every
later stage reads; scripts/make_trip_data.py's raw split is the qualified
intermediate. Naming it the other way round would push "_cleaned" through
every downstream path — including published artifacts like
routes/Inferred_Route_OM_n_trips.gpx — to record a detail none of them care
about.

This is the last cleaning that can be computed without an inferred route, and
both downstream stages need exactly the same answer from it — scripts/
infer_route.py and scripts/map_matching.py each read every trip of every
folder. Materialising it here means it is computed once, is inspectable as a
CSV, and cannot drift between the two consumers.

What is NOT done here, and why:

  * out_of_bounds / out_of_time_bounds need routes/Inferred_Route_<folder>.gpx,
    which is infer_route.py's own output. They stay inside the matcher
    (cleaning.prepare_for_matching with a route), which is the one remaining
    difference between the two matching paths.

  * usable_for_hmm, gap_id and segment_id are deliberately absent. Those route
    checks shrink the usable set, so anything derived from it here would be
    stale by the time map matching reads the file. They are computed in the
    matcher, immediately after usable_for_hmm is finalised.

So this stage writes flags only. No rows are dropped.
"""

import argparse
from pathlib import Path

import pandas as pd

from gps_route_matcher.cleaning import categorize_oob, clean_gps_dataframe
from gps_route_matcher.config import MatchConfig
from gps_route_matcher.io import load_kml_polygon
from gps_route_matcher.paths import PROJECT_ROOT
from gps_route_matcher.visualization import _BAIRRO_KML_PATH, _TRACK_FOLDERS

SPLIT_SUFFIX = "_trips_split"   # what make_trip_data.py writes
TRIPS_SUFFIX = "_trips"         # what every later stage reads

# Written by clean_gps_dataframe / categorize_oob and consumed downstream.
_FLAG_COLUMNS = ["timestamp_valid", "gps_valid", "gps_inside_polygon"]


def _iter_csv_files(folder_path: Path) -> list[Path]:
    """Top-level CSV files only; nested date/time subfolders hold stray
    non-standard leftovers and are never visited."""
    return sorted(
        p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"
    )


def prepare_trips(
    folders: list[str] | None = None,
    overwrite: bool = True,
    config: MatchConfig | None = None,
) -> None:
    """Clean every trip CSV in the given (or all) data/OM_n_trips folders,
    writing to sibling data/OM_n_trips_cleaned/ folders."""
    config = config or MatchConfig()

    print("1/3 Loading bairro polygon …")
    polygon = load_kml_polygon(str(_BAIRRO_KML_PATH)).union_all()

    target_folders = folders or [f"{t}{SPLIT_SUFFIX}" for t in _TRACK_FOLDERS]
    print(f"2/3 Cleaning trips in {target_folders} …")

    n_files = n_rows = n_skipped = 0
    n_gps_invalid = n_outside_polygon = n_no_timestamp = 0

    for folder in target_folders:
        in_dir = PROJECT_ROOT / "data" / folder
        if not in_dir.exists():
            print(f"    skip {folder}: not found")
            continue
        out_dir = PROJECT_ROOT / "data" / f"{folder.removesuffix(SPLIT_SUFFIX)}{TRIPS_SUFFIX}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for csv_path in _iter_csv_files(in_dir):
            out_path = out_dir / csv_path.name
            if not overwrite and out_path.exists():
                n_skipped += 1
                continue

            trip_df = pd.read_csv(csv_path)
            if trip_df.empty:
                n_skipped += 1
                continue

            cleaned = categorize_oob(clean_gps_dataframe(trip_df, config), polygon)
            cleaned.to_csv(out_path, index=False)

            n_files += 1
            n_rows += len(cleaned)
            n_no_timestamp += int((~cleaned["timestamp_valid"]).sum())
            n_gps_invalid += int((~cleaned["gps_valid"]).sum())
            n_outside_polygon += int(
                (cleaned["gps_valid"] & ~cleaned["gps_inside_polygon"]).sum()
            )

    print("3/3 Summary")
    print(f"    trips cleaned: {n_files}, skipped: {n_skipped}, rows: {n_rows:,}")
    print(f"    rows without a usable timestamp   : {n_no_timestamp:,}")
    print(f"    rows without a valid GPS fix      : {n_gps_invalid:,}")
    print(f"    fixes outside the bairro polygon  : {n_outside_polygon:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="+", default=None,
                        help="Restrict to these input folders (e.g. OM_2_trips_split). "
                             "Default: all of OM_1_trips_split..OM_4_trips_split.")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                        help="Skip trips whose cleaned output already exists")
    args = parser.parse_args()
    prepare_trips(folders=args.folders, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
