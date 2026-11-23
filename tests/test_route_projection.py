"""Tests for route.py: building a route, projecting points, recovering positions."""

import math

import numpy as np
import pytest

from gps_route_matcher.route import (
    Route1D,
    auto_utm_crs,
    build_route_1d,
    project_lonlat_to_route,
    route_s_to_xy,
)


# Simple straight route along a meridian (constant longitude, increasing latitude)
# Points spaced ~111 m apart in latitude
_LON = -43.237
_LATS = [-22.871 + i * 0.001 for i in range(6)]  # 6 points, ~110 m apart
_STRAIGHT_ROUTE = [(_LON, lat) for lat in _LATS]


@pytest.fixture
def straight_route() -> Route1D:
    return build_route_1d(_STRAIGHT_ROUTE)


def test_auto_utm_crs_southern():
    crs = auto_utm_crs(-43.237, -22.87)
    assert crs.startswith("EPSG:327")


def test_auto_utm_crs_northern():
    crs = auto_utm_crs(0.0, 51.5)  # London
    assert crs.startswith("EPSG:326")


def test_route_length_positive(straight_route):
    assert straight_route.length_m > 0


def test_cum_dist_monotone(straight_route):
    assert np.all(np.diff(straight_route.cum_dist) > 0)


def test_project_along_route_monotone(straight_route):
    """Points sampled along the route should yield increasing s_m."""
    sample_lats = [-22.871 + i * 0.0008 for i in range(1, 5)]
    s_values = []
    for lat in sample_lats:
        s, _ = project_lonlat_to_route(straight_route, _LON, lat)
        s_values.append(s)
    assert s_values == sorted(s_values), "s_m not monotonically increasing"


def test_project_returns_small_distance_on_route(straight_route):
    """A point exactly on the route should project with near-zero distance."""
    lon, lat = _STRAIGHT_ROUTE[2]
    _, dist_m = project_lonlat_to_route(straight_route, lon, lat)
    assert dist_m < 2.0, f"Expected near-zero distance, got {dist_m:.2f} m"


def test_route_s_to_xy_clamps_at_ends(straight_route):
    x0, y0 = route_s_to_xy(straight_route, -100.0)
    x_start, y_start = straight_route.points_xy[0]
    assert abs(x0 - x_start) < 1e-6
    assert abs(y0 - y_start) < 1e-6

    x_end, y_end = route_s_to_xy(straight_route, straight_route.length_m + 100.0)
    x_last, y_last = straight_route.points_xy[-1]
    assert abs(x_end - x_last) < 1e-6
    assert abs(y_end - y_last) < 1e-6


def test_project_off_route_has_nonzero_distance(straight_route):
    """A point well off to the side of the route should have a positive distance."""
    # Offset by ~0.002 degrees longitude (~200 m)
    lon_off = _LON + 0.002
    lat_mid = _LATS[2]
    _, dist_m = project_lonlat_to_route(straight_route, lon_off, lat_mid)
    assert dist_m > 50.0
