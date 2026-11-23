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

BAIRRO_KML_PATH: Path = PROJECT_ROOT / "borders" / "Bairro.kml"

# The Zenodo deposit in two stages: package_for_zenodo.py writes the complete,
# uncurated package; apply_curation.py reads it plus CURATION_PATH and writes the
# curated one, which is what gets uploaded.
ZENODO_PACKAGE_DIR: Path = PROJECT_ROOT / "output" / "zenodo_package"
ZENODO_CURATED_DIR: Path = PROJECT_ROOT / "output" / "zenodo_curated"

# Hand-made keep/discard/trim/cut decisions for the Zenodo release, applied by
# apply_curation.py. Committed on purpose: unlike output/, these decisions can't
# be regenerated.
CURATION_PATH: Path = PROJECT_ROOT / "curation" / "release_curation.json"

# Hand-picked route edges per track that get MatchConfig.route_edge_bonus in the
# final matching (netmatch's bonus_edges reads it via edge_bonus.load_bonus_edges).
# Committed for the same reason as CURATION_PATH: it is human judgement and can't
# be regenerated.
EDGE_BONUS_PATH: Path = PROJECT_ROOT / "curation" / "edge_bonus.json"


def track_from_folder(folder: str) -> str:
    """Strip a known stage suffix, returning the bare track name (e.g. 'OM_1').
    Returns `folder` unchanged if it doesn't end in a known suffix."""
    for suffix in _ALL_SUFFIXES:
        if folder.endswith(suffix):
            return folder[: -len(suffix)]
    return folder
