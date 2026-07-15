from gps_route_matcher.visualization import generate_interactive_map
from gps_route_matcher.visualization import generate_initial_map
from gps_route_matcher.visualization import generate_tracks
from gps_route_matcher.pipeline import *
import os
import webbrowser
from gps_route_matcher import MatchConfig, match_gps_csv_to_route
from gps_route_matcher.paths import PROJECT_ROOT
import download_gdrive_data
import infer_route


if __name__ == "__main__":
    
    download_gdrive_data.download_om_folders()
    infer_route.main()
    



