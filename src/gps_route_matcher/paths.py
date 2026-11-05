"""Shared filesystem paths, anchored to the installed package location.

Also the single source of truth for the pipeline's data/ folder-naming
convention: each track (OM_1..OM_4) gets one folder per processing stage,
named `{track}_{letter}_{stage}` where the letter encodes processing order
(a=raw, b=walks, c=prepared, d=matched, e=edge_analysis). Every stage script
imports its input/output suffix from here rather than deriving or
hardcoding its own copy.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

TRACKS: list[str] = ["OM_1", "OM_2", "OM_3", "OM_4"]

RAW_SUFFIX = "_a_raw"
WALKS_SUFFIX = "_b_walks"
PREPARED_SUFFIX = "_c_prepared"
MATCHED_SUFFIX = "_d_matched"
EDGE_ANALYSIS_SUFFIX = "_e_edge_analysis"
_ALL_SUFFIXES = [RAW_SUFFIX, WALKS_SUFFIX, PREPARED_SUFFIX, MATCHED_SUFFIX, EDGE_ANALYSIS_SUFFIX]

RAW_FOLDERS: list[str] = [f"{t}{RAW_SUFFIX}" for t in TRACKS]
WALKS_FOLDERS: list[str] = [f"{t}{WALKS_SUFFIX}" for t in TRACKS]
PREPARED_FOLDERS: list[str] = [f"{t}{PREPARED_SUFFIX}" for t in TRACKS]
MATCHED_FOLDERS: list[str] = [f"{t}{MATCHED_SUFFIX}" for t in TRACKS]
EDGE_ANALYSIS_FOLDERS: list[str] = [f"{t}{EDGE_ANALYSIS_SUFFIX}" for t in TRACKS]

# Per-config matched rows cached by scripts/sweep_dropped_points.py (regenerable).
SWEEP_DIR: Path = PROJECT_ROOT / "output" / "sweep_dropped_points"

BAIRRO_KML_PATH: Path = PROJECT_ROOT / "borders" / "Bairro.kml"
MARE_WHOLE_KML_PATH: Path = PROJECT_ROOT / "borders" / "mare_whole.kml"
MARE_AREAS_KML_PATH: Path = PROJECT_ROOT / "borders" / "mare_areas.kml"
CALIBRATION_EQUATIONS_PATH: Path = PROJECT_ROOT / "calibration" / "calibration_equations.csv"

# Suffixes retired by the a/b/c/d/e rename (including the pre-"_raw" bare
# convention and the "_b_split"/"_c_walks" names those two stages carried
# before being renamed to "_b_walks"/"_c_prepared"); used only by
# warn_if_stale_sibling below, kept separate from _ALL_SUFFIXES so the two
# can't silently merge back together.
_OBSOLETE_SUFFIXES = ["_raw", "_trips_split", "_trips_matched", "_trips_edge_analysis", "_trips", "_b_split", "_c_walks", ""]


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
