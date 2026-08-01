"""Input cleaning: parse timestamps, validate GPS, flag usable rows."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import MatchConfig
from .route import Route1D, project_lonlat_to_route


def _absolute_range_mask(ts: pd.Series, config: MatchConfig) -> pd.Series:
    """True where ts falls outside the configured plausible capture window."""
    mask = pd.Series(False, index=ts.index)
    if config.valid_timestamp_min is not None:
        mask |= ts < pd.Timestamp(config.valid_timestamp_min, tz="UTC")
    if config.valid_timestamp_max is not None:
        mask |= ts > pd.Timestamp(config.valid_timestamp_max, tz="UTC")
    return mask


def _detect_timestamp_outliers(df: pd.DataFrame, config: MatchConfig) -> pd.Series:
    """Boolean Series (index-aligned): True where the parsed timestamp is
    inconsistent with the rest of the file's own recording, regardless of
    whether it parsed successfully.

    Runs three stages, each operating only on rows not already flagged:
      0. Absolute plausible-capture-window bound (config.valid_timestamp_*).
      1. Robust median/MAD range check, catching in-window anomalies (e.g. a
         stale clock reading hours before the recording actually started)
         without false-flagging the edges of a long-but-legitimate recording,
         since the tolerance scales with the file's own spread.
      2. Forward monotonicity pass in row order, catching subtler
         out-of-order rows that survive the statistical checks.
    """
    flagged = pd.Series(False, index=df.index)
    if not config.detect_timestamp_outliers:
        return flagged

    parsed = df["timestamp"].notna()

    # Stage 0: absolute capture-window bound.
    flagged.loc[parsed] = _absolute_range_mask(df.loc[parsed, "timestamp"], config)

    candidate_idx = df.index[parsed & ~flagged]
    if len(candidate_idx) < config.min_timestamps_for_sequentiality_check:
        return flagged

    # Stage 1: robust median/MAD range check among Stage-0 survivors.
    ts = df.loc[candidate_idx, "timestamp"]
    epoch_s = ts.astype("int64") / 1e9
    median_s = np.median(epoch_s)
    mad_s = np.median(np.abs(epoch_s - median_s))
    tolerance_s = 1.4826 * mad_s * config.timestamp_outlier_mad_k
    stage1_outlier = (epoch_s < median_s - tolerance_s) | (epoch_s > median_s + tolerance_s)
    flagged.loc[ts.index[stage1_outlier.values]] = True

    # Stage 2: forward monotonicity pass over remaining survivors, in row order.
    running_max = None
    for idx in df.index[parsed & ~flagged]:
        t = df.at[idx, "timestamp"]
        if running_max is not None and t < running_max:
            flagged.at[idx] = True
        else:
            running_max = t

    return flagged


def clean_gps_dataframe(df: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    """Standardise and validate a raw GPS DataFrame.

    Does NOT drop any rows. Adds the following columns:

    original_index      : row position in the original df (0-based int).
    timestamp_raw       : raw value from config.timestamp_col.
    lon_raw             : raw value from config.lon_col.
    lat_raw             : raw value from config.lat_col.
    timestamp           : parsed pd.Timestamp (UTC) or NaT.
    lon                 : float or NaN.
    lat                 : float or NaN.
    timestamp_valid     : bool – timestamp was successfully parsed.
    gps_valid           : bool – lon/lat are numeric and non-zero (if configured).
    usable_for_hmm      : bool = timestamp_valid & gps_valid.
    timestamp_inferred  : bool – False here; set by infer_missing_timestamps().
    """
    df = df.copy()

    # Preserve original row order
    df["original_index"] = np.arange(len(df), dtype=int)

    # --- Raw copies --------------------------------------------------------
    df["timestamp_raw"] = df[config.timestamp_col]
    df["lon_raw"] = df[config.lon_col]
    df["lat_raw"] = df[config.lat_col]

    # --- Parse timestamp ---------------------------------------------------
    df["timestamp"] = pd.to_datetime(
        df[config.timestamp_col], errors="coerce", utc=True, format="mixed"
    )

    outlier_mask = _detect_timestamp_outliers(df, config)
    df.loc[outlier_mask, "timestamp"] = pd.NaT

    df["timestamp_valid"] = df["timestamp"].notna()

    # --- Parse lon/lat -----------------------------------------------------
    df["lon"] = pd.to_numeric(df[config.lon_col], errors="coerce")
    df["lat"] = pd.to_numeric(df[config.lat_col], errors="coerce")

    # --- GPS validity -------------------------------------------------------
    # Bounds also reject inf/-inf (sensor glitches occasionally emit lat/lon
    # far outside the physically possible range, e.g. lat=161.8 — a value
    # that survives .notna() but blows up the UTM projection downstream).
    lon_ok = df["lon"].notna() & df["lon"].between(-180.0, 180.0)
    lat_ok = df["lat"].notna() & df["lat"].between(-90.0, 90.0)

    if config.treat_zero_zero_as_missing:
        zero_zero = (df["lon"] == 0.0) & (df["lat"] == 0.0)
        gps_valid = lon_ok & lat_ok & ~zero_zero
    else:
        gps_valid = lon_ok & lat_ok

    

    df["gps_valid"] = gps_valid
    df["usable_for_hmm"] = df["timestamp_valid"] & df["gps_valid"]
    df["timestamp_inferred"] = False

    return df


def inspect_gps_oob(
    df: pd.DataFrame, config: MatchConfig, route: Route1D
) -> pd.DataFrame:
    """Flag GPS points that fall outside the route buffer.

    Does NOT drop any rows and does NOT touch gps_valid / usable_for_hmm –
    route-buffer distance is a separate concern from basic GPS validity, so
    it's surfaced as its own column and left for downstream logic (see
    inspect_gps_ootb) to combine as needed. For every row where gps_valid is
    already True, projects (lon, lat) onto `route`; out_of_bounds is True
    when the perpendicular distance to the route exceeds config.oob_buffer_m
    [m] (default 20m). Rows where gps_valid is already False get
    out_of_bounds = False (not evaluable) and NaN distance.

    Intended to run right after clean_gps_dataframe, passing the Route1D
    built from the track's respective Inferred_Route_OM_*.gpx file.

    Adds columns:
    distance_to_inferred_route_m : float – perpendicular distance from the
                                    point to `route` [m]; NaN where
                                    gps_valid was already False.
    out_of_bounds                : bool – distance_to_inferred_route_m
                                    exceeds config.oob_buffer_m.
    route_s_m                    : float – arc-length position of the
                                    point's projection onto `route` [m from
                                    route start]; NaN where gps_valid was
                                    already False. Used downstream (see
                                    inspect_gps_ootb) to anchor the walk's
                                    time window on the point closest to the
                                    route start rather than the earliest
                                    in-bounds timestamp.
    """
    df = df.copy()

    df["distance_to_inferred_route_m"] = np.nan
    df["out_of_bounds"] = False
    df["route_s_m"] = np.nan

    valid_idx = df.index[df["gps_valid"]]
    for idx in valid_idx:
        lon = df.at[idx, "lon"]
        lat = df.at[idx, "lat"]
        s_m, distance_m = project_lonlat_to_route(route, lon, lat)
        df.at[idx, "distance_to_inferred_route_m"] = distance_m
        df.at[idx, "out_of_bounds"] = distance_m > config.oob_buffer_m
        df.at[idx, "route_s_m"] = s_m

    return df


def categorize_oob(df: pd.DataFrame, polygon) -> pd.DataFrame:
    """Flag whether each row's raw GPS point lies within `polygon`.

    Pure spatial check, independent of gps_valid/out_of_bounds. Rows with
    unparseable lon/lat get gps_inside_polygon = False.

    Parameters
    ----------
    polygon :
        A shapely geometry (e.g. the Bairro.kml boundary, loaded once via
        io.load_kml_polygon(...).union_all() and reused across trips).

    Adds column:
    gps_inside_polygon : bool – (lon, lat) falls within `polygon`.
    """
    df = df.copy()

    lon_ok = df["lon"].notna() & df["lat"].notna()
    points = gpd.points_from_xy(df["lon"].fillna(0.0), df["lat"].fillna(0.0))
    within = gpd.GeoSeries(points, crs="EPSG:4326").within(polygon).to_numpy()

    df["gps_inside_polygon"] = lon_ok.to_numpy() & within
    return df


def inspect_gps_ootb(
    df: pd.DataFrame, config: MatchConfig, route: Route1D
) -> pd.DataFrame:
    """Flag GPS points that fall outside the temporal window of the walk,
    and derive the final usable_for_hmm.

    Runs inspect_gps_oob first if out_of_bounds isn't present yet. The
    temporal window is [t_min, t_max] among rows that are gps_valid &
    timestamp_valid & ~out_of_bounds (i.e. confirmed on-route fixes) –
    participants often had the sensor recording before reaching / after
    leaving the mapped route, so the window is derived from the data itself
    rather than assumed to be the full recording.

    t_min/t_max are NOT simply the earliest/latest confirmed on-route
    timestamps: on the way to the route start (or away from its end), the
    walk can briefly cross the route elsewhere (e.g. a nearby street),
    producing an in-buffer fix that isn't actually part of walking the
    route. Instead, t_min/t_max are the timestamps of the confirmed
    on-route fixes whose route_s_m (arc-length projection onto `route`, see
    inspect_gps_oob) is smallest/largest – i.e. the fixes closest to the
    route's own start and end points.

    Because t_min and t_max are chosen independently by spatial extremity,
    two separate spurious crossings at opposite ends of the route (e.g. one
    near the end very early, one near the start very late) could in
    principle invert the window (t_min > t_max). If that happens, this
    falls back to the simple earliest/latest confirmed-on-route timestamp
    window instead, which is always non-inverted by construction.

    If no row ever qualifies as a confirmed on-route fix, every row is
    marked out_of_time_bounds = True (no window can be established).

    Does NOT drop any rows. Adds columns:
    out_of_time_bounds : bool – True when the row's timestamp falls outside
                          the derived window (or the timestamp itself is
                          invalid). False for rows inside the window.
    usable_for_hmm      : bool – recomputed as
                          gps_valid & timestamp_valid & ~out_of_bounds &
                          ~out_of_time_bounds. This is the final,
                          authoritative value (overwrites the simpler
                          version set by clean_gps_dataframe /
                          infer_missing_timestamps).
    """
    if "out_of_bounds" not in df.columns:
        df = inspect_gps_oob(df, config, route)

    df = df.copy()

    anchor_mask = df["gps_valid"] & df["timestamp_valid"] & ~df["out_of_bounds"]
    if anchor_mask.any():
        anchors = df.loc[anchor_mask]
        t_min = anchors.loc[anchors["route_s_m"].idxmin(), "timestamp"]
        t_max = anchors.loc[anchors["route_s_m"].idxmax(), "timestamp"]
        if t_min > t_max:
            # Two independent spurious crossings at opposite route ends
            # inverted the window; fall back to the simple (always
            # non-inverted) earliest/latest anchor timestamp window.
            t_min = anchors["timestamp"].min()
            t_max = anchors["timestamp"].max()
        in_window = df["timestamp_valid"] & df["timestamp"].between(t_min, t_max)
        df["out_of_time_bounds"] = ~in_window
    else:
        df["out_of_time_bounds"] = True

    df["usable_for_hmm"] = (
        df["gps_valid"]
        & df["timestamp_valid"]
        & ~df["out_of_bounds"]
        & ~df["out_of_time_bounds"]
    )
    return df


def infer_missing_timestamps(df: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    """Fill in missing timestamps using a fixed or estimated sample interval.

    Only runs if config.infer_missing_timestamps is True. Inferred timestamps
    are created by linearly interpolating (or extrapolating) between the nearest
    valid timestamps using the interval dt. Rows with inferred timestamps get
    timestamp_inferred=True and timestamp_valid=True so they can be used by the
    interpolation step (though NOT for HMM – usable_for_hmm requires gps_valid).

    The sample interval is taken from config.expected_sample_interval_s when
    provided; otherwise it is estimated as the median dt between consecutive
    valid timestamps.
    """
    if not config.infer_missing_timestamps:
        return df

    df = df.copy()

    valid_mask = df["timestamp_valid"]
    if valid_mask.sum() < 2:
        return df  # not enough anchor points

    # Estimate interval if not provided
    interval_s = config.expected_sample_interval_s
    if interval_s is None:
        valid_ts = df.loc[valid_mask, "timestamp"].sort_values()
        diffs = valid_ts.diff().dropna().dt.total_seconds()
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.empty:
            return df
        interval_s = float(positive_diffs.median())

    invalid_idx = df.index[~valid_mask]
    if invalid_idx.empty:
        return df

    valid_positions = df.index[valid_mask].tolist()
    valid_times = df.loc[valid_mask, "timestamp"].tolist()

    for i in invalid_idx:
        # Find bracketing valid timestamps by position
        before = [(pos, t) for pos, t in zip(valid_positions, valid_times) if pos < i]
        after = [(pos, t) for pos, t in zip(valid_positions, valid_times) if pos > i]

        if before and after:
            prev_pos, prev_ts = before[-1]
            next_pos, next_ts = after[0]
            steps_from_prev = i - prev_pos
            inferred = prev_ts + pd.Timedelta(seconds=steps_from_prev * interval_s)
        elif before:
            prev_pos, prev_ts = before[-1]
            inferred = prev_ts + pd.Timedelta(seconds=(i - prev_pos) * interval_s)
        else:
            next_pos, next_ts = after[0]
            inferred = next_ts - pd.Timedelta(seconds=(next_pos - i) * interval_s)

        df.at[i, "timestamp"] = inferred
        df.at[i, "timestamp_valid"] = True
        df.at[i, "timestamp_inferred"] = True

    # Re-evaluate usable_for_hmm (inferred timestamps don't make GPS valid)
    df["usable_for_hmm"] = df["timestamp_valid"] & df["gps_valid"]
    return df
