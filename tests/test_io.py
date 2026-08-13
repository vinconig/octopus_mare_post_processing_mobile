"""Tests for GPX route and KML polygon I/O helpers."""

from gps_route_matcher.io import load_gpx_route, load_kml_polygon, write_gpx_route


def test_write_gpx_route_then_load_gpx_route_roundtrips(tmp_path):
    points = [(-43.25, -22.84), (-43.24, -22.83), (-43.23, -22.82)]
    path = write_gpx_route(points, str(tmp_path / "route.gpx"), name="Test route")

    loaded = load_gpx_route(path)

    assert len(loaded) == 3
    for (lon, lat), (loaded_lon, loaded_lat) in zip(points, loaded):
        assert loaded_lon == lon
        assert loaded_lat == lat


def test_load_gpx_route_returns_empty_list_for_no_track_data(tmp_path):
    gpx_path = tmp_path / "empty.gpx"
    gpx_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1"></gpx>\n'
    )

    assert load_gpx_route(str(gpx_path)) == []


def test_load_gpx_route_falls_back_to_waypoints(tmp_path):
    gpx_path = tmp_path / "waypoints.gpx"
    gpx_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">\n'
        '  <wpt lat="-22.84" lon="-43.25"></wpt>\n'
        '  <wpt lat="-22.83" lon="-43.24"></wpt>\n'
        "</gpx>\n"
    )

    points = load_gpx_route(str(gpx_path))

    assert points == [(-43.25, -22.84), (-43.24, -22.83)]


def test_load_kml_polygon_parses_placemark_name_and_coordinates(tmp_path):
    kml_path = tmp_path / "bairro.kml"
    kml_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        "  <Document>\n"
        "    <Placemark>\n"
        "      <name>Test Bairro</name>\n"
        "      <Polygon>\n"
        "        <outerBoundaryIs>\n"
        "          <LinearRing>\n"
        "            <coordinates>\n"
        "              -43.25,-22.84,0 -43.24,-22.84,0 -43.24,-22.83,0 -43.25,-22.84,0\n"
        "            </coordinates>\n"
        "          </LinearRing>\n"
        "        </outerBoundaryIs>\n"
        "      </Polygon>\n"
        "    </Placemark>\n"
        "  </Document>\n"
        "</kml>\n"
    )

    gdf = load_kml_polygon(str(kml_path))

    assert len(gdf) == 1
    assert gdf.iloc[0]["name"] == "Test Bairro"
    assert gdf.crs.to_string() == "EPSG:4326"
    # altitude (3rd coordinate) is dropped -- exterior ring is 2D (x, y) pairs.
    assert list(gdf.iloc[0].geometry.exterior.coords)[0] == (-43.25, -22.84)


def test_load_kml_polygon_skips_placemarks_without_outer_boundary(tmp_path):
    kml_path = tmp_path / "mixed.kml"
    kml_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        "  <Document>\n"
        "    <Placemark>\n"
        "      <name>No polygon here</name>\n"
        "    </Placemark>\n"
        "    <Placemark>\n"
        "      <name>Valid</name>\n"
        "      <Polygon>\n"
        "        <outerBoundaryIs>\n"
        "          <LinearRing>\n"
        "            <coordinates>0,0,0 1,0,0 1,1,0 0,0,0</coordinates>\n"
        "          </LinearRing>\n"
        "        </outerBoundaryIs>\n"
        "      </Polygon>\n"
        "    </Placemark>\n"
        "  </Document>\n"
        "</kml>\n"
    )

    gdf = load_kml_polygon(str(kml_path))

    assert len(gdf) == 1
    assert gdf.iloc[0]["name"] == "Valid"
