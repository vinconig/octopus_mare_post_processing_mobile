"""Column policy for the Zenodo publication export of the two most
analysis-ready pipeline outputs: point-level `d_matched` CSVs and per-edge
`e_edge_analysis` CSVs.

`MATCHED_COLUMNS` and `EDGE_ANALYSIS_COLUMNS` are each the single source of
truth for what ships in the export: an ordered whitelist, keyed by the
*source* column name. A source column absent from the config is not shipped
(a new pipeline column must be added here deliberately to reach the
deposit), and the output's column order is the config's order, not the
source file's. An entry may carry:

- "rename" — shipped name, if different from the source column.
- "transform" — callable applied to the column before shipping (e.g.
  `to_local_timestamp`).

What each shipped column means is documented by hand in
output/zenodo_package/ZENODO_README.md.
"""

from datetime import timedelta, timezone
from typing import Any

import pandas as pd

from gps_route_matcher.analysis import _CORE_OUTPUT_COLUMNS

# Local time during data collection (Rio de Janeiro). Brazil has no DST, so a
# fixed offset is exact for the whole export window.
LOCAL_TZ = timezone(timedelta(hours=-3), name="UTC-3")
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def to_local_timestamp(series: pd.Series) -> pd.Series:
    """UTC timestamps (any ISO8601 precision) -> naive local-time strings,
    rounded to the nearest whole second, formatted "YYYY-MM-DD HH:MM:SS".
    Unparseable/missing values become empty."""
    ts = pd.to_datetime(series, utc=True, format="ISO8601", errors="coerce")
    ts = ts.dt.round("s").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return ts.dt.strftime(_TIMESTAMP_FORMAT)


def to_nullable_int(series: pd.Series) -> pd.Series:
    """Float column (floats only because unresolved rows are NaN) -> nullable
    integer, so ids write as `11214125333`, not `11214125333.0`, with
    unresolved rows left empty."""
    return series.astype("Int64")


MATCHED_COLUMNS: dict[str, dict[str, Any]] = {
    "timestamp": {"transform": to_local_timestamp},
    "matched_lon": {"rename": "longitude"},
    "matched_lat": {"rename": "latitude"},
    "Temperature": {"rename": "temperature"},
    "Humidity": {"rename": "humidity"},
    "PM2.5_cal": {"rename": "pm2.5"},
    "match_status": {},
    "matched_edge_u": {"transform": to_nullable_int},
    "matched_edge_v": {"transform": to_nullable_int},
    "matched_edge_key": {"transform": to_nullable_int},
    "Longitude": {"rename": "longitude_raw"},
    "Latitude": {"rename": "latitude_raw"},
}

# Fixed columns of an edge_analysis row (route/edge identity, geometry, and
# per-recording timing/coverage) — listed to match analysis._CORE_OUTPUT_COLUMNS
# so this can't silently drift from the actual pipeline output.
_EDGE_CORE_COLUMNS: dict[str, dict[str, Any]] = {
    "edge_order": {},
    "u": {},
    "v": {},
    "key": {},
    "length": {},
    "geometry": {},
    "start_timestamp": {"transform": to_local_timestamp},
    "end_timestamp": {"transform": to_local_timestamp},
    "timestamp_mean": {"transform": to_local_timestamp},
    "n_points": {},
    "includes_interpolated_pts": {},
}
assert set(_EDGE_CORE_COLUMNS) == _CORE_OUTPUT_COLUMNS, (
    "_EDGE_CORE_COLUMNS has drifted from analysis._CORE_OUTPUT_COLUMNS"
)


def _edge_sensor_stats(source: str, shipped: str) -> dict[str, dict[str, Any]]:
    """The *_avg/*_max entries for one sensor: `source` is the column stem in
    the edge_analysis CSV, `shipped` the stem it ships under."""
    return {f"{source}{suffix}": {"rename": f"{shipped}{suffix}"} for suffix in ("_avg", "_max")}


# Mirrors the matched/ sensor selection: temperature, humidity, and
# calibrated PM2.5 only (shipped as pm2.5, like matched/'s pm2.5 column).
EDGE_ANALYSIS_COLUMNS: dict[str, dict[str, Any]] = {
    **_EDGE_CORE_COLUMNS,
    **_edge_sensor_stats("temperature", "temperature"),
    **_edge_sensor_stats("humidity", "humidity"),
    **_edge_sensor_stats("pm2.5_cal", "pm2.5"),
}


def apply_column_config(df: pd.DataFrame, column_config: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Select, transform, rename, and order df's columns per column_config.

    Output columns are exactly those in column_config (in config order);
    source columns not in the config are dropped. A configured column
    missing from df raises KeyError rather than silently shipping a
    narrower file.
    """
    out = pd.DataFrame(index=df.index)
    for source, entry in column_config.items():
        values = df[source]
        if "transform" in entry:
            values = entry["transform"](values)
        out[entry.get("rename", source)] = values
    return out
