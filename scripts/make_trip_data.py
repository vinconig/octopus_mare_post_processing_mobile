"""Concatenate raw per-recording CSVs into one file per sensor, then split
that file into one CSV per trip (morning/evening).

Per folder (OM_1_raw..OM_4_raw):

1. Concatenate — read every *.csv sitting directly in data/OM_n_raw/ (its
   subfolders, e.g. data/OM_4_raw/25-11/, are NOT scanned — only the raw
   per-recording files at the top level), sorted by filename, and stack
   them as-is into data/OM_n_concatenated.csv. No cleaning is applied here:
   values are read as text and passed through unchanged, so this file is a
   literal concatenation of the raw rows (each tagged with the source
   filename it came from).

2. Split into trips — same grouping method as NotebooksCassiano/Code
   1_Filter moving routes OM.ipynb: parse Timestamp, tag each row
   morning/evening by the same hour windows, build session_id = date +
   period, and groupby(session_id). Unlike the notebook, there is no
   buffer-shapefile step here — a trip is just "every row recorded on this
   date within this time window", written out as one CSV per group into
   data/OM_n_trips/. Rows with an unparseable Timestamp, or whose hour
   falls outside both windows, belong to no trip and are not written
   anywhere (only counted).

   Each trip file is named OM_n_YYYYMMDD_<morning|evening>_<D>durmin.csv,
   where <D> is the trip's own duration in whole minutes (last timestamp
   minus first timestamp) — not the duration embedded in any single source
   recording's filename, since one trip can merge several recordings.
"""

import argparse

import pandas as pd

from gps_route_matcher.paths import PROJECT_ROOT

SENSORS: list[str] = ["OM_1_raw", "OM_2_raw", "OM_3_raw", "OM_4_raw"]

TS_COL = "Timestamp"
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

MORNING = (11, 14)  # [start, end) hour
EVENING = (17, 20)  # [start, end) hour


def read_csv_robust(path):
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8", dtype=str)
    except Exception as e:
        print(f"    [unreadable] {path.name}: {e}")
        return None


def concatenate_sensor(sensor: str) -> pd.DataFrame | None:
    """Stack every top-level CSV in data/<sensor>/ (e.g. data/OM_1_raw/),
    sorted by filename, into one DataFrame (raw text, no cleaning), and
    write it to data/<track>_concatenated.csv, where <track> is <sensor>
    with any trailing "_raw" stripped."""
    sensor_dir = PROJECT_ROOT / "data" / sensor
    track = sensor.removesuffix("_raw")
    files = sorted(sensor_dir.glob("*.csv"), key=lambda p: p.name)
    print(f"  {sensor}: {len(files)} csv files found directly in {sensor_dir}")

    frames, n_unreadable = [], 0
    for f in files:
        raw = read_csv_robust(f)
        if raw is None:
            n_unreadable += 1
            continue
        raw["source_file"] = f.name
        frames.append(raw)

    if not frames:
        print(f"    skip {sensor}: no readable csv files")
        return None

    concatenated = pd.concat(frames, ignore_index=True, sort=False)

    out_path = PROJECT_ROOT / "data" / f"{track}_concatenated.csv"
    concatenated.to_csv(out_path, index=False)
    print(f"    rows concatenated : {len(concatenated):,}  ({n_unreadable} unreadable files skipped)")
    print(f"    -> {out_path}")
    return concatenated


def split_into_trips(sensor: str, concatenated: pd.DataFrame) -> None:
    """Group concatenated rows into morning/evening trips by date, and write
    one CSV per trip into data/<track>_trips/, where <track> is <sensor>
    with any trailing "_raw" stripped."""
    track = sensor.removesuffix("_raw")
    ts = pd.to_datetime(concatenated[TS_COL], format=TS_FORMAT, errors="coerce")
    has_ts = ts.notna()
    n_bad_ts = int((~has_ts).sum())

    h = ts.dt.hour
    is_morning = (h >= MORNING[0]) & (h < MORNING[1])
    is_evening = (h >= EVENING[0]) & (h < EVENING[1])
    period = pd.Series(pd.NA, index=concatenated.index, dtype="object")
    period.loc[has_ts & is_morning] = "morning"
    period.loc[has_ts & is_evening] = "evening"

    df = concatenated.copy()
    df["_ts"] = ts
    df["_period"] = period
    n_off_window = int((has_ts & period.isna()).sum())

    in_trip = df["_period"].notna()
    df["_session_id"] = pd.NA
    df.loc[in_trip, "_session_id"] = (
        df.loc[in_trip, "_ts"].dt.strftime("%Y%m%d") + "_" + df.loc[in_trip, "_period"]
    )

    out_dir = PROJECT_ROOT / "data" / f"{track}_trips"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_trips = 0
    for session_id, g in df.loc[in_trip].groupby("_session_id", sort=True):
        g = g.sort_values("_ts")
        date_str, trip_period = session_id.split("_", 1)
        duration_min = int((g["_ts"].iat[-1] - g["_ts"].iat[0]).total_seconds() // 60)

        trip_name = f"{track}_{date_str}_{trip_period}_{duration_min}durmin.csv"
        g.drop(columns=["_ts", "_period", "_session_id"]).to_csv(out_dir / trip_name, index=False)
        n_trips += 1

    print(f"    rows w/ unparseable timestamp : {n_bad_ts:,}")
    print(f"    rows outside morning/evening  : {n_off_window:,}")
    print(f"    trips written                 : {n_trips}  -> {out_dir}")


def make_trip_data(sensors: list[str] | None = None) -> None:
    target_sensors = sensors if sensors else SENSORS

    print(f"1/2 Concatenating raw recordings for {target_sensors} …")
    concatenated_by_sensor: dict[str, pd.DataFrame] = {}
    for sensor in target_sensors:
        concatenated = concatenate_sensor(sensor)
        if concatenated is not None:
            concatenated_by_sensor[sensor] = concatenated

    print("2/2 Splitting into morning/evening trips …")
    for sensor, concatenated in concatenated_by_sensor.items():
        print(f"  {sensor}:")
        split_into_trips(sensor, concatenated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensors", nargs="+", default=None,
                        help="Restrict to these sensors (e.g. OM_2_raw). Default: all of "
                             "OM_1_raw..OM_4_raw.")
    args = parser.parse_args()
    make_trip_data(sensors=args.sensors)


if __name__ == "__main__":
    main()
