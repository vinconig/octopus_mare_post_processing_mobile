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
    for i, folder in enumerate(folders): #refreshes the map once (for the first folder) to ensure the road graph is cached, then reuses it for the rest
        infer_route.infer_route(folders=[folder], force_refresh=(i == 0))
    

    map_matching.map_match(folders=folders, force_refresh=False, overwrite=True)
    
    output_path = generate_interactive_map_all_data(os.path.join(PROJECT_ROOT, "output", "map_interactive_all_data.html"))
    print(f"Map saved to: {output_path}")
    webbrowser.open(f"file://{output_path}")



