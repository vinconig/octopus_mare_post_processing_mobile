"""I/O helpers: load GPX route files and KML polygon boundaries."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import geopandas as gpd
import gpxpy
import gpxpy.gpx
from shapely.geometry import Polygon

_KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def load_gpx_route(path: str) -> list[tuple[float, float]]:
    """Parse a GPX file and return the track as (lon, lat) tuples.

    Concatenates all track segments in order. Returns an empty list if no
    track points are found.

    Parameters
    ----------
    path:
        Filesystem path to a GPX file with one or more <trk>/<trkseg>.

    Returns
    -------
    list of (lon, lat) tuples in WGS-84 degrees.
    """
    with open(path, "r", encoding="utf-8") as fh:
        gpx = gpxpy.parse(fh)

    points: list[tuple[float, float]] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append((pt.longitude, pt.latitude))

    if not points:
        # Fall back to waypoints / routes if no track data
        for wpt in gpx.waypoints:
            points.append((wpt.longitude, wpt.latitude))
        for route in gpx.routes:
            for pt in route.points:
                points.append((pt.longitude, pt.latitude))

    return points


def write_gpx_route(
    points_lonlat: list[tuple[float, float]],
    path: str,
    name: str = "Inferred route",
) -> str:
    """Write an ordered list of (lon, lat) points as a single-segment GPX track.

    The written file is readable by load_gpx_route, so the inferred
    ground-truth route can feed directly into the existing single-track
    matching pipeline.
    """
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name=name)
    segment = gpxpy.gpx.GPXTrackSegment()
    for lon, lat in points_lonlat:
        segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon))
    track.segments.append(segment)
    gpx.tracks.append(track)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(gpx.to_xml())
    return path


def load_kml_polygon(path: str) -> gpd.GeoDataFrame:
    """Parse a KML file and return its Polygon placemarks as a GeoDataFrame.

    Reads the outer boundary of every <Polygon> in the file (inner
    boundaries/holes are ignored) and drops the altitude value from each
    coordinate, keeping only (lon, lat).

    Parameters
    ----------
    path:
        Filesystem path to a KML file with one or more <Placemark>/<Polygon>.

    Returns
    -------
    GeoDataFrame with a "name" column and Polygon geometries in EPSG:4326.
    """
    root = ET.parse(path).getroot()

    names: list[str | None] = []
    polygons: list[Polygon] = []
    for placemark in root.findall(".//kml:Placemark", _KML_NS):
        ring = placemark.find(".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", _KML_NS)
        if ring is None or not ring.text:
            continue

        coords = [tuple(map(float, pt.split(",")[:2])) for pt in ring.text.split()]
        polygons.append(Polygon(coords))

        name_el = placemark.find("kml:name", _KML_NS)
        names.append(name_el.text if name_el is not None else None)

    return gpd.GeoDataFrame({"name": names}, geometry=polygons, crs="EPSG:4326")
