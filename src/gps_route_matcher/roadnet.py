"""Road-network acquisition and caching (OSMnx).

The graph is a directed networkx MultiDiGraph clipped to the bairro polygon.
Directionality is preserved throughout: one-way streets and direction of
travel matter for the edge-traversal counts used in route inference.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import osmnx as ox
from shapely.geometry import Polygon

from .paths import PROJECT_ROOT

ROADNETWORK_DIR: Path = PROJECT_ROOT / "roadnetwork"


def download_road_graph(
    polygon: Polygon, network_type: str = "all", force_refresh: bool = False
) -> nx.MultiDiGraph:
    """Fetch a directed road graph clipped to the polygon from the OSM API.

    Parameters
    ----------
    polygon:
        Boundary polygon in WGS-84 (e.g. the bairro polygon from the KML).
    network_type:
        OSMnx network type. "all" includes footpaths/alleys, matching the
        walking/cycling-speed recordings in this project.
    force_refresh:
        Bypass osmnx's own on-disk Overpass response cache (separate from
        this module's GraphML cache) so recent OSM edits are picked up.

    Returns
    -------
    Directed MultiDiGraph in WGS-84 with edge 'length' attributes in metres.
    """
    prev_use_cache = ox.settings.use_cache
    if force_refresh:
        ox.settings.use_cache = False
    try:
        return ox.graph_from_polygon(polygon, network_type=network_type, simplify=True)
    finally:
        ox.settings.use_cache = prev_use_cache


def save_road_graph(graph: nx.MultiDiGraph, path: str | Path) -> Path:
    """Save a road graph as GraphML so re-runs don't hit the OSM API."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, path)
    return path


def load_road_graph(path: str | Path) -> nx.MultiDiGraph:
    """Load a previously saved GraphML road graph."""
    return ox.load_graphml(Path(path))


def get_road_graph(
    polygon: Polygon,
    network_type: str = "all",
    cache_path: str | Path | None = None,
    force_refresh: bool = False,
) -> nx.MultiDiGraph:
    """Return the road graph for the polygon, using the on-disk cache if present.

    The default cache location is repopath/roadnetwork/bairro_<type>.graphml.

    force_refresh:
        Ignore both this GraphML cache and osmnx's own Overpass response
        cache, forcing a fresh download from the OSM API. Use this after
        editing OSM data (e.g. tagging a footway) that should now show up.
    """
    if cache_path is None:
        cache_path = ROADNETWORK_DIR / f"bairro_{network_type}.graphml"
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_refresh:
        return load_road_graph(cache_path)

    graph = download_road_graph(polygon, network_type=network_type, force_refresh=force_refresh)
    save_road_graph(graph, cache_path)
    return graph
