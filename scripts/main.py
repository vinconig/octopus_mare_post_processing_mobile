from gps_route_matcher.visualization import generate_interactive_map
from gps_route_matcher.visualization import generate_interactive_map_all_data
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


if __name__ == "__main__":
    #All tuneable parameters for the map-matching pipeline are in src/gps_route_matcher/config.py
    #download_gdrive_data.download_om_folders(PROJECT_ROOT)
    folders = ["OM_1", "OM_2", "OM_3", "OM_4"]
    # Manual route endpoints per folder ("LAT,LON" or (lat, lon)); leave a folder
    # out (or use None, None) to keep automatic start/end clustering.
    manual_endpoints = {
        "OM_1": ("-22.839904,-43.252176", "-22.840813,-43.245885"),
        "OM_2": ("-22.854508,-43.241940", "-22.851474,-43.245772"),
        "OM_3": ("-22.858845,-43.239657", "-22.867317,-43.241040"),
        "OM_4": ("-22.871660, -43.236408", "-22.865276,-43.235786"),
    }
    for i, folder in enumerate(folders): #refreshes the map once (for the first folder) to ensure the road graph is cached, then reuses it for the rest
       start, end = manual_endpoints.get(folder, (None, None))
       infer_route.infer_route(folders=[folder], start=start, end=end, force_refresh=False)
        # infer_route.infer_route(folders=[folder], force_refresh=(i == 0))
    

    map_matching.map_match(folders=folders, force_refresh=False, overwrite=True)
    
    output_path = generate_interactive_map_all_data(os.path.join(PROJECT_ROOT, "output", "map_interactive_all_data.html"))
    print(f"Map saved to: {output_path}")
    webbrowser.open(f"file://{output_path}")



