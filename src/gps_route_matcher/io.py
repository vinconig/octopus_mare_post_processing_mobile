"""I/O helpers: load GPX route files and KML polygon boundaries, and read
the per-trip CSVs under data/."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import gpxpy
import gpxpy.gpx
import pandas as pd
from shapely.geometry import Polygon

from .paths import PROJECT_ROOT, track_from_folder

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


def _iter_csv_files(folder_path: Path) -> list[Path]:
    """Top-level CSV files only; nested date/time subfolders hold stray
    non-standard leftovers and are never visited."""
    return sorted(
        p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"
    )


def _load_all_tracks(folders: list[str]) -> pd.DataFrame:
    """Read every CSV directly inside the given data folders (subdirectories
    are ignored) and tag each with its source track and file (track_id).

    folders are literal directory names under data/ — pass OM_n_a_raw for the
    raw per-recording exports or OM_n_c_prepared for the cleaned per-trip splits.

    A pure loader: no rows are dropped and no values are altered. GPS
    validity and bairro-polygon membership are decided once, in
    cleaning.clean_gps_dataframe / categorize_oob, as non-destructive flags.
    """
    frames: list[pd.DataFrame] = []
    for folder in folders:
        folder_path = PROJECT_ROOT / "data" / folder
        if not folder_path.is_dir():
            raise FileNotFoundError(
                f"No such data folder: {folder_path}. Folder names are literal "
                f"directory names under data/ (e.g. OM_1_a_raw or OM_1_c_prepared)."
            )
        for csv_path in _iter_csv_files(folder_path):
            track_df = pd.read_csv(csv_path)
            if track_df.empty:
                continue

            track_df["source_folder"] = track_from_folder(folder)
            track_df["track_id"] = csv_path.stem
            frames.append(track_df)

    if not frames:
        return pd.DataFrame(columns=["Latitude", "Longitude", "source_folder", "track_id"])
    return pd.concat(frames, ignore_index=True)
