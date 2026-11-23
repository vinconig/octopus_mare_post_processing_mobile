"""Run the whole pipeline: raw recordings -> output/zenodo_curated/.

Expects the raw per-recording CSVs in data/OM_n_a_raw/ (see README.md). Every
stage reads the previous stage's folder and writes the next one; the folder
names and track list live in src/gps_route_matcher/paths.py, and all matching
parameters in src/gps_route_matcher/config.py.
"""

import apply_curation
import infer_route
import make_trip_data
import map_matching
import package_for_zenodo
import prepare_trips
import route_edge_analysis
from gps_route_matcher.paths import TRACKS


if __name__ == "__main__":
    # Bare track names: every stage derives its own input/output data/ folder from
    # them via its fixed suffix.
    folders = TRACKS

    # Validates each raw recording's timestamps first (per raw file, while its emit
    # order/filename/own spread are still available), then concatenates per sensor and
    # cuts the result into one CSV per walk: data/OM_n_a_raw/ -> data/OM_n_b_walks/.
    make_trip_data.make_trip_data(folders=folders)

    # Route-independent GPS cleaning, once: data/OM_n_b_walks/ -> data/OM_n_c_prepared/, so
    # infer_route and map_matching below both read the same flags. Everything that needs the
    # inferred route stays in map_matching — the route is inferred below.
    prepare_trips.prepare_trips(folders=folders)

    # Manual route endpoints per track for inferring the route that was taken
    # ("LAT,LON" or (lat, lon)); leave a track out (or use None, None) to keep automatic
    # start/end clustering.
    manual_endpoints = {
        "OM_1": ("-22.839904,-43.252176", "-22.841699, -43.246335"),
        "OM_2": ("-22.855784, -43.239960", "-22.851474,-43.245772"),
        "OM_3": ("-22.858845,-43.239657", "-22.867317,-43.241040"),
        "OM_4": ("-22.871660, -43.236408", "-22.865276,-43.235786"),
    }

    # Infer each route from the actual data based on traversal frequency
    # -> routes/Inferred_Route_OM_n.gpx and routes/route_metadata/OM_n_inferred_route.json.
    for track in folders:
        start, end = manual_endpoints.get(track, (None, None))
        infer_route.infer_route(folders=[track], start=start, end=end)

    # Match every trip to the road network -> data/OM_n_d_matched/.
    map_matching.map_match(folders=folders)

    # Aggregate the matched points per route edge -> data/OM_n_e_edge_analysis/.
    route_edge_analysis.route_edge_analysis(folders=folders)

    # Assemble the Zenodo deposit (slimmed point-level and per-edge CSVs plus route metadata
    # and the OSM graph snapshot) -> output/zenodo_package/. The export date window
    # (_EXPORT_DATE_MIN/_EXPORT_DATE_MAX) is set in package_for_zenodo.py.
    package_for_zenodo.package_for_zenodo(folders=folders)

    # Apply the hand-made keep/discard/trim/cut decisions in curation/release_curation.json
    # -> output/zenodo_curated/, the folder that gets uploaded.
    apply_curation.apply_curation(folders=folders)
