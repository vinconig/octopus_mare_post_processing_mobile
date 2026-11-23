"""Apply the hand-made keep/discard/trim/cut decisions to the Zenodo package,
writing a second, curated package: the one that gets uploaded.

    package_for_zenodo.py  ->  output/zenodo_package/   complete, never curated
    (human decisions)      ->  curation/release_curation.json
    apply_curation.py      ->  output/zenodo_curated/   package + decisions

Reads output/zenodo_package/ (left untouched) and curation/release_curation.json,
and writes output/zenodo_curated/:

* a walk marked "discard" is left out;
* a walk with a trim and/or middle cuts loses those rows (surviving rows keep their
  recorded timestamps, so a cut leaves a time jump), is renamed for its new
  duration, and has its data/mobile_agg_by_edge file rebuilt from the surviving
  rows. If it then touches fewer than MIN_MATCHED_EDGES road-network edges it is
  left out, like any walk below that gate;
* every other walk — kept or never reviewed — is copied unchanged;
* metadata/ and the rest of the package are copied as they are.

Run it after every package_for_zenodo.py run and every change to the curation file.
Format and trim semantics: gps_route_matcher.curation.
"""

import argparse
from pathlib import Path

from gps_route_matcher.curation_apply import apply_curation as _apply_curation
from gps_route_matcher.paths import (
    CURATION_PATH,
    ZENODO_CURATED_DIR,
    ZENODO_PACKAGE_DIR,
)


def apply_curation(
    folders: list[str] | None = None,
    overwrite: bool = True,
    package_dir: Path = ZENODO_PACKAGE_DIR,
    out_dir: Path = ZENODO_CURATED_DIR,
    curation_path: Path | None = CURATION_PATH,
) -> None:
    """folders takes bare track names ("OM_1"), like every stage script."""
    s = _apply_curation(folders=folders, overwrite=overwrite, package_dir=package_dir,
                        out_dir=out_dir, curation_path=curation_path)
    print(f"    walks: {s.n_unchanged} unchanged, {s.n_edited} trimmed/cut ({s.n_rows_removed} rows removed), "
          f"{s.n_discarded} discarded, {s.n_too_few_edges} left out for too few edges, "
          f"{s.n_skipped_existing} skipped (already existed)")
    if s.n_not_trip_files:
        print(f"    skipped {s.n_not_trip_files} non-trip file(s) in the package (e.g. ' 2.csv' copies)")
    if s.unmatched_keys:
        print(f"    WARNING: {len(s.unmatched_keys)} curation entries match no walk in the package "
              f"(renamed/removed?): {', '.join(s.unmatched_keys[:5])}{' …' if len(s.unmatched_keys) > 5 else ''}")
    print(f"    wrote to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folders", nargs="+", default=None,
                        help="Restrict to these tracks (e.g. OM_2). Default: all of OM_1..OM_4.")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                        help="Leave walks whose output file already exists as they are, instead of rewriting them")
    parser.add_argument("--package-dir", type=Path, default=ZENODO_PACKAGE_DIR,
                        help="Package to curate (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=ZENODO_CURATED_DIR,
                        help="Where to write the curated package (default: %(default)s)")
    parser.add_argument("--curation", type=Path, default=CURATION_PATH,
                        help="Curation file (default: %(default)s)")
    parser.add_argument("--no-curation", action="store_true",
                        help="Ignore the curation file: the output is a plain copy of the package")
    args = parser.parse_args()

    apply_curation(folders=args.folders, overwrite=args.overwrite, package_dir=args.package_dir,
                   out_dir=args.out_dir, curation_path=None if args.no_curation else args.curation)


if __name__ == "__main__":
    main()
