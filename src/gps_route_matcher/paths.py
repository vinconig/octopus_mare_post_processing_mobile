"""Shared filesystem paths, anchored to the installed package location.

Also the single source of truth for the pipeline's data/ folder-naming
convention: each track (OM_1..OM_4) gets one folder per processing stage,
named `{track}_{letter}_{stage}` where the letter encodes processing order
(a=raw, b=split, c=walks, d=matched, e=edge_analysis). Every stage script
imports its input/output suffix from here rather than deriving or
hardcoding its own copy.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

TRACKS: list[str] = ["OM_1", "OM_2", "OM_3", "OM_4"] #still needs to be updated i guess this should read MOBILESENSORS or something as per naming convention(needs to be defined)

RAW_SUFFIX = "_a_raw"
SPLIT_SUFFIX = "_b_split"
WALKS_SUFFIX = "_c_walks"
MATCHED_SUFFIX = "_d_matched"
EDGE_ANALYSIS_SUFFIX = "_e_edge_analysis"
_ALL_SUFFIXES = [RAW_SUFFIX, SPLIT_SUFFIX, WALKS_SUFFIX, MATCHED_SUFFIX, EDGE_ANALYSIS_SUFFIX]

RAW_FOLDERS: list[str] = [f"{t}{RAW_SUFFIX}" for t in TRACKS]
SPLIT_FOLDERS: list[str] = [f"{t}{SPLIT_SUFFIX}" for t in TRACKS]
WALK_FOLDERS: list[str] = [f"{t}{WALKS_SUFFIX}" for t in TRACKS]
MATCHED_FOLDERS: list[str] = [f"{t}{MATCHED_SUFFIX}" for t in TRACKS]
EDGE_ANALYSIS_FOLDERS: list[str] = [f"{t}{EDGE_ANALYSIS_SUFFIX}" for t in TRACKS]

BAIRRO_KML_PATH: Path = PROJECT_ROOT / "borders" / "Bairro.kml"

# Suffixes retired by the a/b/c/d/e rename (including the pre-"_raw" bare
# convention); used only by warn_if_stale_sibling below, kept separate from
# _ALL_SUFFIXES so the two can't silently merge back together.
_OBSOLETE_SUFFIXES = ["_raw", "_trips_split", "_trips_matched", "_trips_edge_analysis", "_trips", ""]


def track_from_folder(folder: str) -> str:
    """Strip a known stage suffix, returning the bare track name (e.g. 'OM_1').
    Returns `folder` unchanged if it doesn't end in a known suffix."""
    for suffix in _ALL_SUFFIXES:
        if folder.endswith(suffix):
            return folder[: -len(suffix)]
    return folder


def ensure_dir(path: Path) -> Path:
    """Create path (and parents) if missing; return it for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def warn_if_stale_sibling(track: str, canonical_folder: str) -> None:
    """Print (never delete) a warning if data/<track><old-suffix> still exists
    next to the canonical folder this stage is about to read/write, so a
    future naming-convention change can't silently orphan data again. See
    scripts/migrate_folder_names.py for actually cleaning these up."""
    data_dir = PROJECT_ROOT / "data"
    for suffix in _OBSOLETE_SUFFIXES:
        stale = data_dir / f"{track}{suffix}"
        if stale.name != canonical_folder and stale.is_dir():
            print(
                f"  ! {stale} looks like it's from an old folder-naming convention "
                f"(this stage now uses data/{canonical_folder}). Not touching it — "
                f"see scripts/migrate_folder_names.py if it needs migrating/removing."
            )
