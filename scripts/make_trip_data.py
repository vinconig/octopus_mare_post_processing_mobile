"""Assemble raw per-recording CSVs into one CSV per walk.

Per folder (OM_1_raw..OM_4_raw):

1. Clean + concatenate — read every *.csv sitting directly in data/OM_n_raw/
   (its subfolders, e.g. data/OM_4_raw/25-11/, are NOT scanned), and run
   sessions.clean_recording on each *before* stacking them into
   data/OM_n_concatenated.csv.

   Timestamp validation has to happen here, per file, and not later: a raw CSV
   is one continuous logging session, so it is the only scope where the
   recording's own filename can anchor an absolute check, where the median/MAD
   spread belongs to a single session, and where forward monotonicity is still
   a meaningful question (rows are in device emit order). It is also the last
   moment before the timestamps are used to group the data — validating after
   grouping means validating a field that has already decided which file each
   row landed in.

   Original column values, including Timestamp, are passed through unchanged;
   the verdicts arrive as separate ts_* columns, so the concatenated file
   stays a faithful union of the raw rows.

2. Split into trips — dedupe, cut the sensor's pooled rows into one session
   per walk, label each whole session morning/evening, and write one CSV per
   session into data/OM_n_trips_split/. (scripts/prepare_trips.py then adds
   the GPS cleaning flags and writes data/OM_n_trips/, which is what every
   later stage reads.)

   The label is per session rather than per row on purpose: a walk from 13:44
   to 14:58 is one morning walk, and testing each row against a [11, 14) hour
   window instead would keep its first 15 minutes and drop the other 59.

   Each trip file is named OM_n_YYYYMMDD_<morning|evening>_<D>durmin.csv,
   where <D> is the session's own duration in whole minutes — not the duration
   embedded in any single source recording's filename, since one walk can span
   several recordings.

Rows that don't make it into a trip (unusable timestamp, or a session whose
start hour matches neither window) are written to data/OM_n_quarantine.csv
with the reason, rather than reduced to a printed count.
"""

import argparse

import pandas as pd

from gps_route_matcher.config import MatchConfig
from gps_route_matcher.paths import PROJECT_ROOT
from gps_route_matcher.sessions import (
    SESSION_ID_COL,
    SESSION_PERIOD_COL,
    SESSION_REJECT_COL,
    clean_recording,
    dedupe_rows,
    label_sessions,
    sessionize,
)
from gps_route_matcher.cleaning import TS_CLEAN_COL, TS_REASON_COL, TS_VALID_COL

SENSORS: list[str] = ["OM_1_raw", "OM_2_raw", "OM_3_raw", "OM_4_raw"]


def read_csv_robust(path):
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8", dtype=str)
    except Exception as e:
        print(f"    [unreadable] {path.name}: {e}")
        return None


def concatenate_sensor(sensor: str, config: MatchConfig) -> pd.DataFrame | None:
    """Clean every top-level CSV in data/<sensor>/ per recording, then stack
    them, sorted by filename, into data/<track>_concatenated.csv, where
    <track> is <sensor> with any trailing "_raw" stripped."""
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
        cleaned = clean_recording(raw, f.stem, config)
        cleaned["source_file"] = f.name
        frames.append(cleaned)

    if not frames:
        print(f"    skip {sensor}: no readable csv files")
        return None

    concatenated = pd.concat(frames, ignore_index=True, sort=False)

    out_path = PROJECT_ROOT / "data" / f"{track}_concatenated.csv"
    concatenated.to_csv(out_path, index=False)
    rejected = concatenated[TS_REASON_COL].notna()
    n_rejected = int(rejected.sum())
    n_recovered = int((rejected & concatenated[TS_VALID_COL]).sum())
    print(f"    rows concatenated : {len(concatenated):,}  ({n_unreadable} unreadable files skipped)")
    print(f"    timestamps rejected : {n_rejected:,}  (of which back-filled: {n_recovered:,})")
    print(f"    -> {out_path}")
    return concatenated


def split_into_trips(sensor: str, concatenated: pd.DataFrame, config: MatchConfig) -> None:
    """Cut a sensor's cleaned rows into sessions, label them, and write one
    CSV per session into data/<track>_trips_split/, where <track> is <sensor>
    with any trailing "_raw" stripped.

    The plain data/<track>_trips/ name belongs to scripts/prepare_trips.py's
    output, which is what every later stage reads."""
    track = sensor.removesuffix("_raw")

    df, n_duplicates = dedupe_rows(concatenated, config)
    df = sessionize(df, config)
    df = label_sessions(df, config)

    in_trip = df[SESSION_PERIOD_COL].notna()

    out_dir = PROJECT_ROOT / "data" / f"{track}_trips_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_trips = 0
    for _, g in df.loc[in_trip].groupby(SESSION_ID_COL, sort=True):
        g = g.sort_values(TS_CLEAN_COL)
        ts = pd.to_datetime(g[TS_CLEAN_COL])
        date_str = ts.iat[0].strftime("%Y%m%d")
        period = g[SESSION_PERIOD_COL].iat[0]
        duration_min = int((ts.iat[-1] - ts.iat[0]).total_seconds() // 60)

        trip_name = f"{track}_{date_str}_{period}_{duration_min}durmin.csv"
        g.drop(columns=[SESSION_PERIOD_COL, SESSION_REJECT_COL]).to_csv(
            out_dir / trip_name, index=False
        )
        n_trips += 1

    _write_quarantine(track, df.loc[~in_trip])

    n_no_ts = int((~df[TS_VALID_COL].fillna(False)).sum())
    rejects = df.loc[~in_trip, SESSION_REJECT_COL].value_counts()
    n_idle = int((~in_trip).sum()) - n_no_ts - int(rejects.sum())
    print(f"    duplicate rows dropped        : {n_duplicates:,}")
    print(f"    rows w/ unusable timestamp    : {n_no_ts:,}")
    print(f"    rows of idle (non-walk) time  : {n_idle:,}")
    print(f"    rows in off-window sessions   : {int(rejects.get('session_off_window', 0)):,}")
    print(f"    rows in over-long sessions    : {int(rejects.get('session_too_long', 0)):,}")
    print(f"    trips written                 : {n_trips}  -> {out_dir}")


def _write_quarantine(track: str, rejected: pd.DataFrame) -> None:
    """Persist rows that reached no trip, with why, so the loss is auditable
    instead of being a number in stdout that nobody can act on."""
    out_path = PROJECT_ROOT / "data" / f"{track}_quarantine.csv"
    if rejected.empty:
        out_path.unlink(missing_ok=True)
        return
    rejected = rejected.copy()
    # A row with a usable timestamp is here because of its session's verdict;
    # otherwise the timestamp verdict is the reason, and it is already
    # recorded. Prefer whichever applies.
    rejected["quarantine_reason"] = (
        rejected[SESSION_REJECT_COL]
        .fillna(rejected[TS_REASON_COL])
        .fillna(pd.Series("idle_time", index=rejected.index).where(
            rejected[TS_VALID_COL].fillna(False), "unusable_timestamp"
        ))
    )
    rejected.to_csv(out_path, index=False)
    print(f"    quarantined                   : {len(rejected):,}  -> {out_path}")


def make_trip_data(sensors: list[str] | None = None, config: MatchConfig | None = None) -> None:
    target_sensors = sensors if sensors else SENSORS
    config = config or MatchConfig()

    print(f"1/2 Cleaning and concatenating raw recordings for {target_sensors} …")
    concatenated_by_sensor: dict[str, pd.DataFrame] = {}
    for sensor in target_sensors:
        concatenated = concatenate_sensor(sensor, config)
        if concatenated is not None:
            concatenated_by_sensor[sensor] = concatenated

    print("2/2 Splitting into morning/evening trips …")
    for sensor, concatenated in concatenated_by_sensor.items():
        print(f"  {sensor}:")
        split_into_trips(sensor, concatenated, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensors", nargs="+", default=None,
                        help="Restrict to these sensors (e.g. OM_2_raw). Default: all of "
                             "OM_1_raw..OM_4_raw.")
    args = parser.parse_args()
    make_trip_data(sensors=args.sensors)


if __name__ == "__main__":
    main()
