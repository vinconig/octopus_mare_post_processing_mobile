"""Column policy for the Zenodo publication export of the two most
analysis-ready pipeline outputs: point-level `d_matched` CSVs and per-edge
`e_edge_analysis` CSVs.

`MATCHED_COLUMNS` and `EDGE_ANALYSIS_COLUMNS` are each the single source of
truth for what ships in the export, in the same "one dict, no scattered
logic" spirit as `analysis._NON_ENV_COLUMNS`. A column's `action` is one of:

- "keep" — shipped unchanged.
- "keep_caveat" — shipped, but flagged with a caveat in the data dictionary
  (e.g. a QA flag whose meaning depends on pipeline internals).
- "drop" — pipeline-internal bookkeeping with no meaning outside this
  codebase; not shipped.

A column present in the source CSV but absent from the config defaults to
"keep" (never silently drop a column nobody made a decision about) and logs
a warning, so a newly added pipeline column can't vanish from the export by
accident.
"""

import logging
from typing import Any

import pandas as pd

from gps_route_matcher.analysis import _CORE_OUTPUT_COLUMNS

logger = logging.getLogger(__name__)

MATCHED_COLUMNS: dict[str, dict[str, Any]] = {
    "Timestamp": {"action": "keep", "description": "Raw device timestamp, as originally logged (unparsed)"},
    "Latitude": {"action": "keep", "description": "Raw GPS latitude, as originally logged"},
    "Longitude": {"action": "keep", "description": "Raw GPS longitude, as originally logged"},
    "Temperature": {
        "action": "keep_caveat",
        "description": "Raw temperature reading (HS300 sensor, Arduino Nano 33 BLE Sense Rev2)",
        "caveat": "Not comparable across devices or time — the sensor hardware was replaced mid-project on "
        "each device. Use relative intra-recording values only, not absolute values.",
    },
    "Humidity": {
        "action": "keep_caveat",
        "description": "Raw humidity reading (HS300 sensor, Arduino Nano 33 BLE Sense Rev2)",
        "caveat": "Not comparable across devices or time — the sensor hardware was replaced mid-project on "
        "each device. Use relative intra-recording values only, not absolute values.",
    },
    "PM1.0": {"action": "keep", "description": "Raw PM1.0 reading, Sensirion SPS30 (uncalibrated)"},
    "PM2.5": {"action": "keep", "description": "Raw PM2.5 reading, Sensirion SPS30 (uncalibrated)"},
    "PM2.5_cal": {
        "action": "keep",
        "description": "Calibrated PM2.5 (per-track linear equation; see calibration/calibration_equations.csv). "
        "Recommended primary PM2.5 metric — only PM2.5 is calibrated.",
    },
    "PM4.0": {"action": "keep", "description": "Raw PM4.0 reading, Sensirion SPS30 (uncalibrated)"},
    "PM10.0": {"action": "keep", "description": "Raw PM10.0 reading, Sensirion SPS30 (uncalibrated)"},
    "timestamp": {
        "action": "keep",
        "description": "Parsed, validated UTC timestamp (may be interpolated — see timestamp_inferred)",
        "rename": "timestamp_utc",
    },
    "matched_lon": {"action": "keep", "description": "Longitude of the point's position snapped/interpolated onto the road network"},
    "matched_lat": {"action": "keep", "description": "Latitude of the point's position snapped/interpolated onto the road network"},
    "distance_to_route_m": {
        "action": "keep",
        "description": "Residual distance (m) between the raw GPS fix and its matched network position "
        "(median ~5 m, p90 ~16 m across the dataset)",
    },
    "match_status": {
        "action": "keep",
        "description": "One of matched | interpolated | unresolved_missing_time | unresolved_large_gap | "
        "unresolved_out_of_time_bounds — the primary usability field for this row",
    },
    "interpolated": {
        "action": "keep_caveat",
        "description": "True if the row's position was interpolated across a gap rather than directly matched",
        "caveat": "Redundant with match_status == 'interpolated'; kept as a convenience filter column.",
    },
    "matched_edge_u": {
        "action": "keep",
        "description": "OSM node id, start of the matched directed road-network edge. Requires the OSM graph "
        "snapshot shipped in roadnetwork/bairro_walk.graphml (see PROVENANCE.json) to resolve — OSM node ids "
        "drift over time.",
    },
    "matched_edge_v": {"action": "keep", "description": "OSM node id, end of the matched directed road-network edge"},
    "matched_edge_key": {"action": "keep", "description": "OSMnx parallel-edge index (0 unless multiple edges connect the same node pair)"},
    "timestamp_inferred": {
        "action": "keep_caveat",
        "description": "True if `timestamp_utc` was back-filled/interpolated rather than read directly from the device",
        "caveat": "Data-quality/provenance flag, not sensor data.",
    },
    "timestamp_valid": {
        "action": "keep_caveat",
        "description": "True if the row's timestamp parsed successfully during upstream cleaning",
        "caveat": "Pipeline QA gating flag, largely redundant with match_status by this stage.",
    },
    "gps_valid": {
        "action": "keep_caveat",
        "description": "True if the raw lon/lat were numeric, in-range, and not the (0,0) missing-value sentinel",
        "caveat": "Pipeline QA gating flag; explains why matched_lon/matched_lat may be missing.",
    },
    "out_of_time_bounds": {
        "action": "keep_caveat",
        "description": "True if the row's timestamp falls outside the walk's derived active time window",
        "caveat": "Pipeline QA gating flag used to trim idle time before/after a walk.",
    },
    "gps_inside_polygon": {
        "action": "keep_caveat",
        "description": "True if the raw GPS fix falls within the surveyed neighbourhood boundary polygon",
        "caveat": "Pipeline QA gating flag. Combined with gps_valid and out_of_bounds, identifies a fix that "
        "was valid and within the surveyed area but still landed off the matched route (fatal GPS drift).",
    },
    "out_of_bounds": {
        "action": "keep_caveat",
        "description": "True if the raw GPS fix lies outside the buffered corridor around the matched route",
        "caveat": "Pipeline QA gating flag. Combined with gps_valid and gps_inside_polygon, identifies a fix "
        "that was valid and within the surveyed area but still landed off the matched route (fatal GPS drift).",
    },
    "lon": {"action": "drop"},  # near-duplicate of Latitude/Longitude (parsed float vs. raw)
    "lat": {"action": "drop"},
    "ts_reject_reason": {"action": "drop"},  # upstream per-recording timestamp QA code, mostly NA by this stage
    "original_index": {"action": "drop"},  # row position in the original raw file; internal join bookkeeping
    "usable_for_hmm": {"action": "drop"},  # internal HMM gating flag, superseded by match_status
    "distance_to_inferred_route_m": {"action": "drop"},  # route-buffer QA distance, distinct from distance_to_route_m
    "route_s_m": {"action": "drop"},  # internal arc-length helper along the route's reference polyline
    "segment_id": {"action": "drop"},  # internal Viterbi segmentation counter
    "group_id": {"action": "drop"},  # internal path-linking bookkeeping for interpolation
    "path_m": {"action": "drop"},  # internal accumulated network-path distance, used only by interpolation math
    "source_file": {"action": "drop"},  # provenance to a raw per-recording file that isn't itself published
}

# Fixed columns of an edge_analysis row (route/edge identity, geometry, and
# per-recording timing/coverage) — reuses analysis._CORE_OUTPUT_COLUMNS so
# this can't silently drift from the actual pipeline output.
EDGE_ANALYSIS_COLUMNS: dict[str, dict[str, Any]] = {
    "route_id": {"action": "keep", "description": "Track name (OM_1..OM_4)"},
    "edge_order": {"action": "keep", "description": "0-based position of this edge in the route's canonical ordered edge sequence"},
    "u": {"action": "keep", "description": "OSM node id, start of this directed road-network edge"},
    "v": {"action": "keep", "description": "OSM node id, end of this directed road-network edge"},
    "key": {"action": "keep", "description": "OSMnx parallel-edge index (0 unless multiple edges connect the same node pair)"},
    "length": {"action": "keep", "description": "Edge length in meters"},
    "geometry": {
        "action": "keep",
        "description": "WKT LINESTRING of the edge's road-segment geometry (lon/lat)",
        "caveat": "u, v, key, length, and geometry are static per route edge and duplicated in "
        "route_metadata/<track>_inferred_route.json, which is the canonical source if the two ever disagree "
        "(e.g. after a route re-inference regenerates one but not the other). Kept here so each file is "
        "directly usable without a join.",
    },
    "start_timestamp": {"action": "keep", "description": "Timestamp of the earliest point assigned to this edge in this recording"},
    "end_timestamp": {"action": "keep", "description": "Timestamp of the latest point assigned to this edge in this recording"},
    "timestamp_mean": {"action": "keep", "description": "Midpoint of start_timestamp and end_timestamp"},
    "n_points": {
        "action": "keep",
        "description": "Count of raw points aggregated into this edge-row (sample size backing the *_avg/*_max stats; 0 means no coverage this recording)",
    },
    "includes_interpolated_pts": {
        "action": "keep_caveat",
        "description": "True if any point aggregated into this edge was positioned via interpolation rather than a direct match",
        "caveat": "Per-edge provenance/QA flag summarizing the point-level interpolated column.",
    },
}
assert set(EDGE_ANALYSIS_COLUMNS) == _CORE_OUTPUT_COLUMNS, (
    "EDGE_ANALYSIS_COLUMNS' fixed columns have drifted from analysis._CORE_OUTPUT_COLUMNS"
)

_SENSOR_STAT_SUFFIXES = ("_avg", "_max")


def _sensor_stat_description(column: str) -> str | None:
    for suffix in _SENSOR_STAT_SUFFIXES:
        if column.endswith(suffix):
            sensor = column[: -len(suffix)]
            stat = "mean" if suffix == "_avg" else "maximum"
            return f"Per-edge {stat} of {sensor} over points assigned to this edge in this recording"
    return None


def apply_column_config(df: pd.DataFrame, column_config: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Filter df's columns per column_config and apply any renames.

    A column present in df but not in column_config, and not matching the
    dynamic sensor-stat suffix pattern (*_avg/*_max, for edge_analysis),
    defaults to "keep" with a logged warning rather than being silently
    dropped.
    """
    keep_cols = []
    renames = {}
    for col in df.columns:
        entry = column_config.get(col)
        if entry is None:
            if _sensor_stat_description(col) is None:
                logger.warning("column %r not in column config, defaulting to keep", col)
            keep_cols.append(col)
            continue
        if entry["action"] == "drop":
            continue
        keep_cols.append(col)
        if "rename" in entry:
            renames[col] = entry["rename"]

    out = df[keep_cols]
    if renames:
        out = out.rename(columns=renames)
    return out


def build_data_dictionary_rows(dataset: str, df: pd.DataFrame, column_config: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """One row per column actually shipped (post apply_column_config, i.e.
    df's columns already reflect any renames) for the given dataset,
    describing what it is and any caveat. Generated from the same config
    apply_column_config uses, so it can't drift from what's shipped."""
    rename_lookup = {entry["rename"]: original for original, entry in column_config.items() if "rename" in entry}

    rows = []
    for col in df.columns:
        entry = column_config.get(rename_lookup.get(col, col))
        if entry is not None:
            description = entry["description"]
            caveat = entry.get("caveat", "")
        else:
            description = _sensor_stat_description(col) or ""
            caveat = ""
        rows.append(
            {
                "dataset": dataset,
                "column": col,
                "description": description,
                "caveat": caveat,
            }
        )
    return rows
