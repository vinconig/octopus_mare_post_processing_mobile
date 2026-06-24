"""Route representation and 1-D projection utilities.

The route is projected to a metric CRS (auto-detected UTM) once at construction
time. All distance computations are then done in metres on the projected
coordinates, avoiding repeated CRS transformations in the hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pyproj
from shapely.geometry import LineString, Point


def auto_utm_crs(lon: float, lat: float) -> str:
    """Return the EPSG code for the UTM zone that contains (lon, lat)."""
    zone = math.floor((lon + 180.0) / 6.0) + 1
    if lat >= 0:
        return f"EPSG:326{zone:02d}"
    else:
        return f"EPSG:327{zone:02d}"


@dataclass
class Route1D:
    """A predefined route represented as a 1-D curve parameterised by arc-length.

    Attributes
    ----------
    points_lonlat : list of (lon, lat) in WGS-84 degrees.
    points_xy     : (N, 2) float array of projected metric coordinates.
    crs_metric    : EPSG string used for the metric projection.
    crs_wgs84     : Always "EPSG:4326".
    line          : Shapely LineString in metric CRS.
    length_m      : Total arc-length of the route [m].
    cum_dist      : (N,) cumulative distance at each node [m]; cum_dist[0]=0.
    seg_lengths   : (N-1,) length of each segment [m].
    transformer_fwd : pyproj Transformer  WGS84 → metric.
    transformer_inv : pyproj Transformer  metric → WGS84.
    """

    points_lonlat: list[tuple[float, float]]
    points_xy: np.ndarray
    crs_metric: str
    crs_wgs84: str
    line: LineString
    length_m: float
    cum_dist: np.ndarray
    seg_lengths: np.ndarray
    transformer_fwd: pyproj.Transformer
    transformer_inv: pyproj.Transformer


def build_route_1d(route_points_lonlat: list[tuple[float, float]]) -> Route1D:
    """Build a Route1D from an ordered list of (lon, lat) waypoints.

    The route is projected to the UTM zone of its first point. All subsequent
    route/projection operations use metric coordinates.
    """
    if len(route_points_lonlat) < 2:
        raise ValueError("Route must have at least 2 points.")

    lons = [p[0] for p in route_points_lonlat]
    lats = [p[1] for p in route_points_lonlat]

    # Pick UTM zone from centroid for stability
    crs_metric = auto_utm_crs(float(np.mean(lons)), float(np.mean(lats)))
    crs_wgs84 = "EPSG:4326"

    fwd = pyproj.Transformer.from_crs(crs_wgs84, crs_metric, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs_metric, crs_wgs84, always_xy=True)

    xs, ys = fwd.transform(lons, lats)
    pts_xy = np.column_stack([xs, ys])

    diffs = np.diff(pts_xy, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])

    cum_dist = np.zeros(len(pts_xy))
    cum_dist[1:] = np.cumsum(seg_lengths)

    line = LineString(pts_xy.tolist())

    return Route1D(
        points_lonlat=route_points_lonlat,
        points_xy=pts_xy,
        crs_metric=crs_metric,
        crs_wgs84=crs_wgs84,
        line=line,
        length_m=float(cum_dist[-1]),
        cum_dist=cum_dist,
        seg_lengths=seg_lengths,
        transformer_fwd=fwd,
        transformer_inv=inv,
    )


def route_s_to_xy(route: Route1D, s_m: float) -> tuple[float, float]:
    """Return the metric (x, y) coordinate at arc-length s_m along the route."""
    s_m = float(np.clip(s_m, 0.0, route.length_m))
    # Binary search for the segment containing s_m
    idx = int(np.searchsorted(route.cum_dist, s_m, side="right")) - 1
    idx = int(np.clip(idx, 0, len(route.seg_lengths) - 1))

    t = 0.0
    if route.seg_lengths[idx] > 0:
        t = (s_m - route.cum_dist[idx]) / route.seg_lengths[idx]
    t = float(np.clip(t, 0.0, 1.0))

    A = route.points_xy[idx]
    B = route.points_xy[idx + 1]
    x = A[0] + t * (B[0] - A[0])
    y = A[1] + t * (B[1] - A[1])
    return x, y


def route_s_to_lonlat(route: Route1D, s_m: float) -> tuple[float, float]:
    """Return (lon, lat) WGS-84 at arc-length s_m along the route."""
    x, y = route_s_to_xy(route, s_m)
    lon, lat = route.transformer_inv.transform(x, y)
    return float(lon), float(lat)


def project_lonlat_to_route(
    route: Route1D, lon: float, lat: float
) -> tuple[float, float]:
    """Project a WGS-84 point onto the route.

    Returns
    -------
    (s_m, distance_m)
        s_m        : Arc-length position on the route [m].
        distance_m : Perpendicular distance from point to route [m].
    """
    x, y = route.transformer_fwd.transform(lon, lat)
    pt = Point(x, y)
    s_m = float(route.line.project(pt))
    nearest = route.line.interpolate(s_m)
    distance_m = float(pt.distance(nearest))
    return s_m, distance_m
