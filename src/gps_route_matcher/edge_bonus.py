"""Hand-curated route edges that get a slight matching edge.

On some stretches the GPS fix sits between the inferred route and a neighbouring
street, and the matcher then assigns it to the neighbour (where it is dropped
downstream as off-route). For those stretches `MatchConfig.route_edge_bonus` is
subtracted from the emission cost of candidates on the listed route edges - the
local counterpart of the global `off_route_penalty`.

The edges live in curation/edge_bonus.json (paths.EDGE_BONUS_PATH):

    {"version": 1,
     "tracks": {"OM_1": [{"u": 11214125334, "v": 11214125333, "key": 0,
                          "edge_order": 41, "note": "parallel street 8 m east"}]}}

Entries are identified by the stable (u, v, key); `edge_order` and `note` are for
humans only, since re-running route inference can renumber edge_order.
"""

from __future__ import annotations

import json
from pathlib import Path

from .paths import EDGE_BONUS_PATH

EDGE_BONUS_VERSION = 1

EdgeKey = tuple[int, int, int]


def _read(path: Path) -> dict:
    """The parsed document, or an empty one when the file does not exist."""
    if not path.is_file():
        return {"version": EDGE_BONUS_VERSION, "tracks": {}}
    with open(path) as fh:
        data = json.load(fh)
    if data.get("version") != EDGE_BONUS_VERSION:
        raise ValueError(
            f"{path}: unsupported edge-bonus version {data.get('version')!r} "
            f"(expected {EDGE_BONUS_VERSION})"
        )
    if not isinstance(data.get("tracks"), dict):
        raise ValueError(f"{path}: no 'tracks' object")
    return data


def load_bonus_edges(track: str, path: str | Path = EDGE_BONUS_PATH) -> set[EdgeKey]:
    """The {(u, v, key)} bonus edges for `track`; empty if the file or track is absent."""
    entries = _read(Path(path))["tracks"].get(track, [])
    return {(int(e["u"]), int(e["v"]), int(e["key"])) for e in entries}


def add_bonus_edges(
    track: str,
    edges: list[dict],
    path: str | Path = EDGE_BONUS_PATH,
) -> int:
    """Add edge dicts ({"u","v","key"} plus optional "edge_order"/"note") to `track`.

    Edges already listed are skipped. Returns how many were actually added.
    """
    path = Path(path)
    data = _read(path)
    entries = data["tracks"].setdefault(track, [])
    have = {(e["u"], e["v"], e["key"]) for e in entries}
    added = 0
    for e in edges:
        k = (int(e["u"]), int(e["v"]), int(e["key"]))
        if k in have:
            continue
        have.add(k)
        entries.append({**e, "u": k[0], "v": k[1], "key": k[2]})
        added += 1
    _write(path, data)
    return added


def remove_bonus_edges(
    track: str,
    edges: list[EdgeKey],
    path: str | Path = EDGE_BONUS_PATH,
) -> int:
    """Remove the given (u, v, key) edges from `track`. Returns how many were removed."""
    path = Path(path)
    data = _read(path)
    drop = set(edges)
    entries = data["tracks"].get(track, [])
    kept = [e for e in entries if (e["u"], e["v"], e["key"]) not in drop]
    removed = len(entries) - len(kept)
    if kept:
        data["tracks"][track] = kept
    else:
        data["tracks"].pop(track, None)
    _write(path, data)
    return removed


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
