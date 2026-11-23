"""Road-network loading (OSMnx GraphML snapshot).

The graph is a directed networkx MultiDiGraph clipped to the bairro polygon.
Directionality is preserved throughout: one-way streets and direction of
travel matter for the edge-traversal counts used in route inference.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import osmnx as ox

from .paths import PROJECT_ROOT

ROADNETWORK_DIR: Path = PROJECT_ROOT / "roadnetwork"

# The one road graph every stage uses (walk network, pulled from OSM on 2026-11-04).
# It is tracked in the repo and shipped with the deposit, because the edge ids in
# the data refer to exactly this snapshot.
ROAD_GRAPH_PATH: Path = ROADNETWORK_DIR / "mare_openstreetmap_road_graph_2026-11-04.graphml"


def load_road_graph(path: str | Path = ROAD_GRAPH_PATH) -> nx.MultiDiGraph:
    """Load the saved GraphML road graph; raises if the snapshot is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Road graph snapshot not found: {path}")
    return ox.load_graphml(path)
