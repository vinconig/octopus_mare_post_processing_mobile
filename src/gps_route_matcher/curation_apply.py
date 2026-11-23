"""Rebuild a walk's per-edge file from its *shipped* point file.

scripts/apply_curation.py works on the Zenodo package, not on data/. When a
curation edit (trim or cut) removes rows from a walk, that walk's
data/mobile_agg_by_edge file must be recomputed from the surviving rows. The
real aggregation (analysis.edge_analysis_from_df) expects the pipeline's
matched-CSV schema, while a packaged data/mobile file has shipped names
(`pm2.5_raw`, `latitude`, ...) and local-time timestamps. This module maps the
shipped schema back — inverting zenodo_columns.MATCHED_COLUMNS rather than
repeating its names — so the aggregation and the shipped column whitelist stay
the single implementation, and a rebuilt file of an *unedited* walk equals the
one package_for_zenodo.py wrote (see tests/test_curation_apply.py).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .analysis import edge_analysis_from_df, load_route_metadata
from .curation import has_edit, is_discarded, load_curation, trim_mask, walk_key
from .paths import CURATION_PATH, TRACKS, ZENODO_CURATED_DIR, ZENODO_PACKAGE_DIR
from .recording_naming import _TRIP_FILENAME_RE
from .zenodo_columns import (
    EDGE_ANALYSIS_COLUMNS,
    EDGE_SUBDIR,
    LOCAL_TZ,
    MATCHED_COLUMNS,
    MIN_MATCHED_EDGES,
    MOBILE_SUBDIR,
    ROUTE_METADATA_SUBDIR,
    apply_column_config,
    n_distinct_matched_edges,
    real_duration_min,
)

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
# Columns the shipped point files carry as nullable integers; reading them back
# as Int64 keeps ids exact and writes them without a trailing ".0".
NULLABLE_INT_COLUMNS = ("matched_edge_u", "matched_edge_v", "matched_edge_key")


def read_shipped_walk(path) -> pd.DataFrame:
    """A packaged data/mobile CSV, read so that writing it back changes nothing:
    exact float round-trip and integer edge ids."""
    return pd.read_csv(path, float_precision="round_trip", dtype={c: "Int64" for c in NULLABLE_INT_COLUMNS})


def shipped_to_pipeline_schema(shipped: pd.DataFrame) -> pd.DataFrame:
    """Undo MATCHED_COLUMNS: shipped names back to source names, local-time
    timestamps back to UTC (the schema analysis.edge_analysis_from_df expects).

    Only the columns the shipped per-edge file draws on are needed, but every
    shipped column is mapped back so nothing is silently left behind.
    `interpolated` is not shipped; it is derived from match_status, which is how
    the matcher sets it.
    """
    source_of = {entry.get("rename", source): source for source, entry in MATCHED_COLUMNS.items()}
    out = shipped.rename(columns=source_of).copy()
    local = pd.to_datetime(out["timestamp"], format=_TIMESTAMP_FORMAT, errors="coerce")
    utc = local - LOCAL_TZ.utcoffset(None)
    out["timestamp"] = utc.dt.strftime(_TIMESTAMP_FORMAT)
    out["interpolated"] = out["match_status"] == "interpolated"
    return out


def edge_analysis_from_shipped(shipped: pd.DataFrame, route_metadata: dict) -> pd.DataFrame:
    """The shipped mobile_agg_by_edge frame for a (possibly row-subset) shipped walk."""
    return apply_column_config(
        edge_analysis_from_df(shipped_to_pipeline_schema(shipped), route_metadata),
        EDGE_ANALYSIS_COLUMNS,
    )


@dataclass
class CurationSummary:
    n_unchanged: int = 0
    n_edited: int = 0
    n_rows_removed: int = 0
    n_discarded: int = 0
    n_too_few_edges: int = 0
    n_skipped_existing: int = 0
    n_not_trip_files: int = 0
    unmatched_keys: list[str] = field(default_factory=list)  # curation entries that matched no walk


def _copy_all_but_walks(package_dir: Path, out_dir: Path) -> None:
    """Copy everything in the package (metadata, fixed data, ...) except
    the two walk folders, which apply_curation writes itself."""
    walk_dirs = {(package_dir / MOBILE_SUBDIR).resolve(), (package_dir / EDGE_SUBDIR).resolve()}

    def ignore(directory: str, names: list[str]) -> list[str]:
        return [n for n in names if (Path(directory) / n).resolve() in walk_dirs or n == ".DS_Store"]

    shutil.copytree(package_dir, out_dir, ignore=ignore, dirs_exist_ok=True)


def apply_curation(
    folders: list[str] | None = None,
    overwrite: bool = True,
    package_dir: Path | str = ZENODO_PACKAGE_DIR,
    out_dir: Path | str = ZENODO_CURATED_DIR,
    curation_path: Path | str | None = CURATION_PATH,
) -> CurationSummary:
    """Apply the hand-made curation decisions to a Zenodo package, writing a second package.

    Reads `package_dir` (package_for_zenodo.py's output, which is never modified)
    and writes the curated deposit to `out_dir`:

    * everything but the two walk folders is copied as is (metadata, ...);
    * a walk marked "discard" is left out;
    * a walk with a trim and/or cuts loses those rows, keeping every other row's
      recorded timestamp; if it then touches fewer than MIN_MATCHED_EDGES edges it
      is left out too, otherwise it is renamed for its new duration and its
      mobile_agg_by_edge file is rebuilt from the surviving rows;
    * every other walk (kept or unreviewed) is copied byte for byte.

    folders: bare track names to process (default all). overwrite=False leaves
    walks whose output file already exists as they are; True first clears the
    processed tracks' walk files from `out_dir`, so a walk discarded since the last
    run does not linger. A missing or None `curation_path` curates nothing, which
    makes `out_dir` a plain copy of the package.
    """
    package_dir, out_dir = Path(package_dir), Path(out_dir)
    in_mobile, in_edge = package_dir / MOBILE_SUBDIR, package_dir / EDGE_SUBDIR
    if not in_mobile.is_dir():
        raise SystemExit(f"no package at {package_dir} (run scripts/package_for_zenodo.py first)")
    if out_dir.resolve() == package_dir.resolve():
        raise SystemExit("the curated package must be a different folder from the package it is made from")

    tracks = list(folders) if folders else list(TRACKS)
    curation = load_curation(curation_path) if curation_path else {}
    used: set[str] = set()
    summary = CurationSummary()

    _copy_all_but_walks(package_dir, out_dir)
    out_mobile, out_edge = out_dir / MOBILE_SUBDIR, out_dir / EDGE_SUBDIR
    out_mobile.mkdir(parents=True, exist_ok=True)
    out_edge.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for d in (out_mobile, out_edge):
            for track in tracks:
                for stale in d.glob(f"{track}_*.csv"):
                    stale.unlink()

    for track in tracks:
        for csv_path in sorted(in_mobile.glob(f"{track}_*.csv")):
            m = _TRIP_FILENAME_RE.match(csv_path.stem)
            if not m:  # e.g. a macOS "… 2.csv" copy
                summary.n_not_trip_files += 1
                continue
            key = walk_key(csv_path.name)
            entry = curation.get(key)
            if entry:
                used.add(key)
            if is_discarded(entry):
                summary.n_discarded += 1
                continue

            edge_path = in_edge / csv_path.name
            if not has_edit(entry):
                if not overwrite and (out_mobile / csv_path.name).exists():
                    summary.n_skipped_existing += 1
                    continue
                shutil.copy2(csv_path, out_mobile / csv_path.name)
                if edge_path.is_file():
                    shutil.copy2(edge_path, out_edge / csv_path.name)
                else:
                    print(f"    {csv_path.name}: no mobile_agg_by_edge file in the package, none copied")
                summary.n_unchanged += 1
                continue

            walk = read_shipped_walk(csv_path)
            keep = trim_mask(walk["timestamp"], entry)
            walk_after = walk[keep]
            if n_distinct_matched_edges(walk_after) < MIN_MATCHED_EDGES:
                print(f"    {key}: left out — after trim/cuts it touches fewer than {MIN_MATCHED_EDGES} road-network edges")
                summary.n_too_few_edges += 1
                continue
            new_name = f"{m['track']}_{m['date']}_{m['period_word']}_{real_duration_min(walk_after)}durmin.csv"
            if not overwrite and (out_mobile / new_name).exists():
                summary.n_skipped_existing += 1
                continue
            meta_path = package_dir / ROUTE_METADATA_SUBDIR / f"{track}_inferred_route.json"
            if not meta_path.is_file():
                raise SystemExit(f"package has no route metadata for {track} ({meta_path}); cannot rebuild {key}")
            walk_after.to_csv(out_mobile / new_name, index=False)
            edge_analysis_from_shipped(walk_after, load_route_metadata(meta_path)).to_csv(out_edge / new_name, index=False)
            summary.n_edited += 1
            summary.n_rows_removed += int((~keep).sum())

    # A partial run (folders=...) legitimately misses the other tracks' entries.
    summary.unmatched_keys = sorted(
        k for k in curation if k not in used and "_".join(k.split("_")[:2]) in tracks
    )
    return summary
