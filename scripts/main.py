from gps_route_matcher.visualization import generate_interactive_map_all_data_incl_edges, generate_hmm_debug_map
import argparse
import os
import webbrowser
from gps_route_matcher.paths import PROJECT_ROOT, TRACKS
import download_gdrive_data
import infer_route
import map_matching
import route_edge_analysis
import generate_map_info
import generate_route_plots
import generate_route_plots_publication
import make_trip_data
import prepare_trips


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true",
                         help="Don't open the generated maps (hmm debug, per-track "
                              "inferred-route, and final all-data map) in a browser")
    parser.add_argument("--force-refresh", action="store_true",
                         help="Redownload the OSM road graph instead of reusing the cached "
                              "one, once (for the first track) so it's cached for the rest")
    args = parser.parse_args()
    no_browser = args.no_browser

    # #initial data download from google drive (only needs to be done once, commented out after first run)
    # #download_gdrive_data.download_om_folders(PROJECT_ROOT)

    #folder names of all routes/sensors, by bare track name — every stage below
    #derives its own input/output data/ folder from this via its fixed suffix
    #(see src/gps_route_matcher/paths.py)
    folders = TRACKS

    #make trip data from raw CSVs: validates each recording's timestamps first (per raw file,
    #while its emit order/filename/own spread are still available), then concatenates per sensor
    #and cuts the result into one CSV per walk
    make_trip_data.make_trip_data(folders=folders)

    #applies the route-independent GPS cleaning once, data/OM_n_b_split/ -> data/OM_n_c_walks/,
    #so infer_route and map_matching below both read the same flags instead of each deriving them.
    #Everything that needs the inferred route stays in map_matching — the route is inferred below.
    prepare_trips.prepare_trips(folders=folders)

    # Manual route endpoints per track for inferring the route that was taken ("LAT,LON" or (lat, lon)); leave a track out (or use None, None) to keep automatic start/end clustering.
    manual_endpoints = {
        "OM_1": ("-22.839904,-43.252176", "-22.840813,-43.245885"),
        "OM_2": ("-22.854508,-43.241940", "-22.851474,-43.245772"),
        "OM_3": ("-22.858845,-43.239657", "-22.867317,-43.241040"),
        "OM_4": ("-22.871660, -43.236408", "-22.865276,-43.235786"),
    }

    #infer routes from actual data based on traversal frequency
    #All tuneable parameters for the map-matching pipeline are in src/gps_route_matcher/config.py
    for i, track in enumerate(folders):
       start, end = manual_endpoints.get(track, (None, None))
       infer_route.infer_route(folders=[track], start=start, end=end, force_refresh=(args.force_refresh and i == 0), no_browser=no_browser) #only refreshes on the first track (if requested); later tracks reuse the now-cached road graph

    #maps all points per recording to the road network and creates new CSVs with the matched data in data/OM_n_d_matched/ folders
    map_matching.map_match(folders=folders)

     #behind the scenes of map matching, generate a debug map showing the HMM candidate traces and the winning path for a single CSV
    output_path = generate_hmm_debug_map(os.path.join(PROJECT_ROOT, "data/OM_3_a_raw/OM_3_20260306_18_46durmin__REC000050.csv"), os.path.join(PROJECT_ROOT, "output", "interactivemaps", "hmm_debug.html"))
    print(f"Map saved to: {output_path}")
    if not no_browser:
        webbrowser.open(f"file://{output_path}")

    #edge analysis
    route_edge_analysis.route_edge_analysis(folders=folders)

    #generate per-route plots with generic filters (all time periods, all dates, all coverage levels)
    generate_route_plots.generate_route_plots(folders=folders, time_of_day="morning", min_coverage_pct=50, start_date="20251115", end_date="20260430")

    #same recordings and same filters, as publication figures (SVG): one bootstrapped mean profile
    #per route with a 95% CI band, on a fixed y scale so the four routes are comparable
    generate_route_plots_publication.generate_route_plots_publication(folders=folders, time_of_day="morning", min_coverage_pct=50, start_date="20251115", end_date="20260430")

    #generates data on where the map should pan and zoom to
    generate_map_info.generate_map_info()

    #main data panel
    #replaces both generate_interactive_map_all_edge_data and generate_interactive_map_all_data, generates a single map with all data and edge analysis data
    output_path = generate_interactive_map_all_data_incl_edges(
    os.path.join(PROJECT_ROOT, "output", "interactivemaps", "map_interactive_all_data_incl_edge_data.html"), folders=folders)
    if not no_browser:
        webbrowser.open(f"file://{output_path}")
