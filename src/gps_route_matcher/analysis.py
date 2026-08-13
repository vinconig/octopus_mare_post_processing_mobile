"""Edge-level aggregation of matched route recordings.

Turns a point-level matched CSV (data/OM_n_matched/*.csv) into one row per
road-network edge of the route's inferred edge sequence (routes/route_metadata/
*.json), in route order, with per-edge environmental summary statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import EDGE_ANALYSIS_SUFFIX, PROJECT_ROOT
from .recording_naming import _parse_recording_filename

# Columns produced by the map-matching pipeline (cleaning/gaps/netmatch) that
# are not environmental sensor readings. Anything else in a matched CSV is
# treated as an environmental variable to aggregate, so a newly added raw
# sensor column is picked up automatically.
_NON_ENV_COLUMNS = {
    "Timestamp", "Latitude", "Longitude",
    "source_file",
    # sessions.py (per-recording timestamp verdicts, trip assembly)
    "ts_clean", "ts_valid", "ts_inferred", "ts_reject_reason",
    "session_id", "session_period", "session_reject_reason",
    "original_index", "timestamp_raw", "lon_raw", "lat_raw", "timestamp",
    "timestamp_valid", "lon", "lat", "gps_valid", "usable_for_hmm", "timestamp_inferred",
    "distance_to_inferred_route_m", "route_s_m", "out_of_bounds", "gps_inside_polygon", "out_of_time_bounds",
    "dt_s", "gap_before", "gap_id", "segment_id",
    "matched_lon", "matched_lat", "distance_to_route_m", "match_status", "interpolated",
    "group_id", "path_m", "matched_edge_u", "matched_edge_v", "matched_edge_key",
    "edge_order",
}

# match_status values that carry real sensor readings assigned to a matched
# edge (both "matched" and "interpolated" rows do; every other status has
# NaN matched_edge_u/v/key).
_ASSIGNED_STATUSES = {"matched", "interpolated"}

# Per-environmental-column statistics to compute: output suffix -> pandas agg
# name. Add "median"/"std"/"min"/"max" (or a percentile callable) here to
# extend the output without touching the aggregation logic itself.
_ENV_STATS = {"avg": "mean"}

# Output columns that are never environmental statistics.
_CORE_OUTPUT_COLUMNS = {
    "route_id", "edge_order", "u", "v", "key", "length", "geometry",
    "start_timestamp", "end_timestamp", "timestamp_mean", "n_points",
    "includes_interpolated_pts",
}


def load_matched_route(csv_path: str | Path) -> pd.DataFrame:
    """Load a point-level matched CSV as-is (no dtype coercion)."""
    return pd.read_csv(csv_path)


def load_route_metadata(json_path: str | Path) -> dict:
    """Load a route's ordered edge list + per-edge geometry, as written by
    gps_route_matcher.route_inference.save_route_metadata."""
    with open(json_path) as fh:
        return json.load(fh)


def _env_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_ENV_COLUMNS]


def assign_points_to_edges(trip_df: pd.DataFrame, route_metadata: dict) -> pd.DataFrame:
    """Restrict a matched trip's rows to points assigned to an edge of the
    inferred route, and attach that edge's position (edge_order) in the route.

    Keeps both "matched" and "interpolated" rows (both carry real sensor
    readings; only their position was interpolated across a gap). Rows with
    any other match_status, matched to an edge outside the route's edge set,
    or flagged out_of_time_bounds, are dropped — the route metadata is the
    source of truth for valid edges, not the matcher's full-graph output.
    """
    df = trip_df[trip_df["match_status"].isin(_ASSIGNED_STATUSES)].copy()
    if "out_of_time_bounds" in df.columns:
        df = df[~df["out_of_time_bounds"].fillna(False)]
    df = df.dropna(subset=["matched_edge_u", "matched_edge_v", "matched_edge_key"])
    for col in ("matched_edge_u", "matched_edge_v", "matched_edge_key"):
        df[col] = df[col].astype(int)

    # If the route backtracks over the same physical edge (rare, via-node
    # loop legs), the point data alone can't tell which visit a measurement
    # belongs to — last edge_order for that (u, v, key) wins.
    #you could do this via timestamps, but since none of the routes actually has this problem (yet) this can be handled better if it actually comes up, otherwise i would exclude routes that have this backtracking issue
    edge_order_by_key = {
        (e["u"], e["v"], e["key"]): e["edge_order"] for e in route_metadata["edges"]
    }
    df["edge_order"] = [
        edge_order_by_key.get((int(u), int(v), int(k)))
        for u, v, k in zip(df["matched_edge_u"], df["matched_edge_v"], df["matched_edge_key"])
    ]
    df = df.dropna(subset=["edge_order"])
    df["edge_order"] = df["edge_order"].astype(int)
    # format="ISO8601", not inference: interpolated timestamps carry fractional
    # seconds while read ones don't, so a single guessed format fits only one
    # of the two shapes present in the same column.
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df


def aggregate_by_edge(assigned_df: pd.DataFrame) -> pd.DataFrame:
    """Group assigned points by edge_order and compute per-edge summary
    stats. Only edges with >=1 assigned point appear here — pass the result
    through reindex_to_full_route to fill in the rest of the route."""
    env_cols = _env_columns(assigned_df)
    grouped = assigned_df.groupby("edge_order", sort=True)

    out = pd.DataFrame(index=grouped.size().index)
    out["n_points"] = grouped.size()
    out["includes_interpolated_pts"] = grouped["interpolated"].any()
    out["start_timestamp"] = grouped["timestamp"].min()
    out["end_timestamp"] = grouped["timestamp"].max()
    out["timestamp_mean"] = (
        out["start_timestamp"] + (out["end_timestamp"] - out["start_timestamp"]) / 2
    )

    for col in env_cols:
        for suffix, agg in _ENV_STATS.items():
            out[f"{col.lower()}_{suffix}"] = grouped[col].agg(agg)

    return out.reset_index()


def reindex_to_full_route(agg_df: pd.DataFrame, route_metadata: dict) -> pd.DataFrame:
    """Left-join the per-edge aggregation onto the FULL ordered route edge
    list, so every edge of the inferred route appears exactly once, in route
    order, even if no measurement was ever assigned to it (n_points = 0,
    includes_interpolated_pts = False, environmental stats NaN, timestamps
    NaT)."""
    full = (
        pd.DataFrame(route_metadata["edges"])
        .sort_values("edge_order")
        .reset_index(drop=True)
    )
    merged = full.merge(agg_df, on="edge_order", how="left")
    merged["n_points"] = merged["n_points"].fillna(0).astype(int)
    merged["includes_interpolated_pts"] = merged["includes_interpolated_pts"].fillna(False).astype(bool)

    lengths = []
    geometries = []
    for u, v, k in zip(merged["u"], merged["v"], merged["key"]):
        edge = route_metadata["edge_geometry"].get(f"({int(u)},{int(v)},{int(k)})", {})
        lengths.append(edge.get("length"))
        geometries.append(edge.get("geometry"))
    merged["length"] = lengths
    merged["geometry"] = geometries

    return merged


def assemble_output(reindexed_df: pd.DataFrame, route_id: str) -> pd.DataFrame:
    """Select/order the final output columns: route + edge identification,
    geometry, timestamps, environmental stats (alphabetical, so output column
    order is stable even though raw matched-CSV column order varies
    file-to-file), then n_points/includes_interpolated_pts."""
    stat_cols = sorted(c for c in reindexed_df.columns if c not in _CORE_OUTPUT_COLUMNS)
    ordered_cols = [
        "edge_order", "u", "v", "key", "length", "geometry",
        "start_timestamp", "end_timestamp", "timestamp_mean",
        *stat_cols, "n_points", "includes_interpolated_pts",
    ]
    out = reindexed_df[ordered_cols].copy()
    out.insert(0, "route_id", route_id)
    return out


def save_edge_analysis(df: pd.DataFrame, out_path: str | Path) -> None:
    df.to_csv(out_path, index=False)


def run_edge_analysis(
    matched_csv_path: str | Path, route_metadata: dict, route_id: str
) -> pd.DataFrame:
    """Full per-file pipeline: load -> restrict/assign to route edges ->
    aggregate -> reindex to the full route -> assemble the output dataframe."""
    trip_df = load_matched_route(matched_csv_path)
    assigned = assign_points_to_edges(trip_df, route_metadata)
    aggregated = aggregate_by_edge(assigned)
    reindexed = reindex_to_full_route(aggregated, route_metadata)
    return assemble_output(reindexed, route_id)


def load_route_recordings(track: str) -> pd.DataFrame:
    """Concatenate every edge-analysis CSV for one route
    (data/{track}_e_edge_analysis/*.csv, as written by run_edge_analysis) into
    a single long DataFrame -- one row per (recording, edge) -- tagged with
    recording_id/date/period/recid parsed from the filename. Files that don't
    match the TRACK_YYYYMMDD_HR_DURATIONdurmin__RECrecid.csv naming
    convention are skipped, same as the interactive-map recording browser."""
    in_dir = PROJECT_ROOT / "data" / f"{track}{EDGE_ANALYSIS_SUFFIX}"
    frames = []
    for csv_path in sorted(in_dir.glob("*.csv")):
        parsed = _parse_recording_filename(csv_path.stem)
        if parsed is None:
            continue
        df = pd.read_csv(csv_path)
        df["recording_id"] = csv_path.stem
        df["date"] = parsed["date"]
        df["period"] = parsed["period"]
        df["recid"] = parsed["recid"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_across_recordings(long_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a long (recording, edge) DataFrame (see load_route_recordings)
    to one row per edge_order: every `_avg` column becomes the plain,
    unweighted mean of that edge's per-recording averages. Recordings where
    the edge had zero assigned points are already NaN there, so
    groupby().mean() skips them on its own -- a 1-point recording counts the
    same as a 50-point one otherwise. n_recordings counts how many
    recordings had >=1 point on the edge; n_points_total sums n_points
    across every recording (informational only, not used as a weight)."""
    avg_cols = sorted(c for c in long_df.columns if c.endswith("_avg"))
    grouped = long_df.groupby("edge_order", sort=True)

    out = grouped[["u", "v", "key", "length", "geometry"]].first()
    for col in avg_cols:
        out[col] = grouped[col].mean()
    out["n_recordings"] = (long_df["n_points"] > 0).groupby(long_df["edge_order"]).sum().astype(int)
    out["n_points_total"] = grouped["n_points"].sum()

    ordered_cols = [
        "edge_order", "u", "v", "key", "length", "geometry",
        *avg_cols, "n_recordings", "n_points_total",
    ]
    return out.reset_index()[ordered_cols].sort_values("edge_order").reset_index(drop=True)
