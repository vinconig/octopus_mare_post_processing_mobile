from gps_route_matcher.visualization import generate_interactive_map, generate_interactive_map_all_edge_data, generate_interactive_map_all_data_incl_edges
from gps_route_matcher.visualization import generate_interactive_map_all_data, generate_hmm_debug_map
from gps_route_matcher.visualization import generate_initial_map
from gps_route_matcher.visualization import generate_tracks
from gps_route_matcher.pipeline import *
import os
import webbrowser
from gps_route_matcher import MatchConfig, match_gps_csv_to_route
from gps_route_matcher.paths import PROJECT_ROOT
import download_gdrive_data
import infer_route
import map_matching
import route_edge_analysis
import generate_map_info
import make_trip_data


if __name__ == "__main__":
    
    #initial data download from google drive (only needs to be done once, commented out after first run)
    download_gdrive_data.download_om_folders(PROJECT_ROOT)

    #make trip data from raw CSVs, concatenates all CSVs per sensor and splits them into morning/evening trips
    make_trip_data.make_trip_data(sensors=["OM_1_raw", "OM_2_raw", "OM_3_raw", "OM_4_raw"])

    #folder names of all routes/sensors, where one csv correlates to one trip, generated from make_trip_data.py
    folders = ["OM_1_trips", "OM_2_trips", "OM_3_trips", "OM_4_trips"]

    # Manual route endpoints per folder for inferring the route that was taken ("LAT,LON" or (lat, lon)); leave a folder out (or use None, None) to keep automatic start/end clustering.
    manual_endpoints = {
        "OM_1_trips": ("-22.839904,-43.252176", "-22.840813,-43.245885"),
        "OM_2_trips": ("-22.854508,-43.241940", "-22.851474,-43.245772"),
        "OM_3_trips": ("-22.858845,-43.239657", "-22.867317,-43.241040"),
        "OM_4_trips": ("-22.871660, -43.236408", "-22.865276,-43.235786"),
    }

    #infer routes from actual data based on traversal frequency
    #All tuneable parameters for the map-matching pipeline are in src/gps_route_matcher/config.py
    for i, folder in enumerate(folders): 
       start, end = manual_endpoints.get(folder, (None, None))
       infer_route.infer_route(folders=[folder], start=start, end=end, force_refresh=False) #would not redownload map (unless its not existing in cache/storage location path)
       #infer_route.infer_route(folders=[folder], start=start, end=end,force_refresh=(i == 0)) #refreshes the map once (for the first folder) to ensure the road graph is cached, then reuses it for the rest
    
    #maps all points per recording to the road network and creates new CSVs with the matched data in data/OM_n_trips_matched/ folders
    map_matching.map_match(folders=folders, force_refresh=False, overwrite=True)

     #behind the scenes of map matching, generate a debug map showing the HMM candidate traces and the winning path for a single CSV
    output_path = generate_hmm_debug_map(os.path.join(PROJECT_ROOT, "data/OM_3_raw/OM_3_20260306_18_46durmin__REC000050.csv"), os.path.join(PROJECT_ROOT, "output/hmm_debug.html"))
    print(f"Map saved to: {output_path}")
    webbrowser.open(f"file://{output_path}")

    #edge analysis
    route_edge_analysis.route_edge_analysis(folders=folders, overwrite=True)

    #generates data on where the map should pan and zoom to
    generate_map_info.generate_map_info() 

    #main data panel
    #replaces both generate_interactive_map_all_edge_data and generate_interactive_map_all_data, generates a single map with all data and edge analysis data
    output_path = generate_interactive_map_all_data_incl_edges(
    os.path.join(PROJECT_ROOT, "output", "map_interactive_all_data_incl_edge_data.html"), folders=folders)
    webbrowser.open(f"file://{output_path}")