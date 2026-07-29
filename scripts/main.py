from gps_route_matcher.visualization import generate_interactive_map, generate_interactive_map_all_edge_data
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
import route_summary_analysis
import generate_map_info


if __name__ == "__main__":
    
    #initial data download from google drive (only needs to be done once, commented out after first run)
    # download_gdrive_data.download_om_folders(PROJECT_ROOT)

    #folder names of all routes/sensors
    folders = ["OM_1", "OM_2", "OM_3", "OM_4"]

    # Manual route endpoints per folder ("LAT,LON" or (lat, lon)); leave a folder out (or use None, None) to keep automatic start/end clustering.
    manual_endpoints = {
        "OM_1": ("-22.839904,-43.252176", "-22.840813,-43.245885"),
        "OM_2": ("-22.854508,-43.241940", "-22.851474,-43.245772"),
        "OM_3": ("-22.858845,-43.239657", "-22.867317,-43.241040"),
        "OM_4": ("-22.871660, -43.236408", "-22.865276,-43.235786"),
    }



    #infer routes from actual data based on traversal frequency
    #All tuneable parameters for the map-matching pipeline are in src/gps_route_matcher/config.py
    for i, folder in enumerate(folders): 
       start, end = manual_endpoints.get(folder, (None, None))
       infer_route.infer_route(folders=[folder], start=start, end=end, force_refresh=False)
        # infer_route.infer_route(folders=[folder], force_refresh=(i == 0)) #refreshes the map once (for the first folder) to ensure the road graph is cached, then reuses it for the rest
    
    #maps all points per recording to the road network and creates new CSVs with the matched data in data/OM_n_matched/ folders
    map_matching.map_match(folders=folders, force_refresh=False, overwrite=True)

    #edge analysis
    route_edge_analysis.route_edge_analysis(folders=folders, overwrite=True)
    route_summary_analysis.route_summary_analysis(folders=folders, overwrite=True)
    output_path = generate_interactive_map_all_edge_data(
    os.path.join(PROJECT_ROOT, "output", "map_interactive_all_edge_data.html"))
    webbrowser.open(f"file://{output_path}")

    

    #main data panal
    generate_map_info.generate_map_info()
    output_path = generate_interactive_map_all_data(os.path.join(PROJECT_ROOT, "output", "map_interactive_all_data.html"))
    print(f"Map saved to: {output_path}")
    webbrowser.open(f"file://{output_path}")

    #behind the scenes of map matching, generate a debug map showing the HMM candidate traces and the winning path for a single CSV
    output_path = generate_hmm_debug_map(os.path.join(PROJECT_ROOT, "data/OM_3/OM_3_20260306_18_46durmin__REC000050.csv"), os.path.join(PROJECT_ROOT, "output/hmm_debug.html"))
    print(f"Map saved to: {output_path}")
    webbrowser.open(f"file://{output_path}")



