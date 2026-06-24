"""I/O helpers: load GPX route files."""

from __future__ import annotations

import gpxpy
import gpxpy.gpx


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
