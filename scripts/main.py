from gps_route_matcher.visualization import generate_interactive_map
from gps_route_matcher.visualization import generate_initial_map
from gps_route_matcher.pipeline import *
import os
import webbrowser
from gps_route_matcher import MatchConfig, match_gps_csv_to_route
from gps_route_matcher.paths import PROJECT_ROOT

if __name__ == "__main__":
    gps_csv_path=os.path.join(PROJECT_ROOT, "data", "OM2", "OM2test.CSV")
    route_path=os.path.join(PROJECT_ROOT, "routes", "OM2_Route.gpx")
    output_csv_path=os.path.join(PROJECT_ROOT, "output", "intermediate", "cleaned_df.csv")


    cfg = MatchConfig(
        timestamp_col="Timestamp",
        lon_col="Longitude",
        lat_col="Latitude",
    )


    df = match_gps_csv_to_route(gps_csv_path=gps_csv_path, route_path=route_path, output_csv_path=output_csv_path, config = cfg)

    output_path = generate_interactive_map(df, output_path=os.path.join(PROJECT_ROOT, "output", "pm25_interactive.html"))
    #output_path = generate_initial_map()
    print(f"Map saved to: {output_path}")
    webbrowser.open(f"file://{output_path}")
