"""Input cleaning: parse timestamps, validate GPS, flag usable rows."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import MatchConfig
from .gaps import assign_gaps_and_segments
from .route import Route1D, project_lonlat_to_route


# Why a timestamp was rejected. Recorded rather than discarded so a row that
# is later back-filled by infer_missing_timestamps still says what was wrong
# with the original reading, and so scripts/make_trip_data.py can quarantine
# rejected rows with an explanation instead of a bare count.
TS_UNPARSEABLE = "unparseable"
TS_FILENAME_MISMATCH = "filename_mismatch"
TS_ABSOLUTE_WINDOW = "absolute_window"
TS_MAD_OUTLIER = "mad_outlier"
TS_NON_MONOTONIC = "non_monotonic"
TS_DUPLICATE_TIMESTAMP = "duplicate_timestamp"

# Column carrying the reason, and the pre-cleaned timestamp columns that
# sessions.clean_recording writes. When these are present on an input frame,
# clean_gps_dataframe consumes them instead of re-deriving them — the
# per-recording pass had strictly more evidence (device emit order, one
# session's own spread, the recording filename) than a per-trip pass can have.
TS_REASON_COL = "ts_reject_reason"
TS_CLEAN_COL = "ts_clean"
TS_VALID_COL = "ts_valid"
TS_INFERRED_COL = "ts_inferred"


def _absolute_range_mask(ts: pd.Series, config: MatchConfig) -> pd.Series:
    """True where ts falls outside the plausible capture window.

    The upper bound is the current time rather than a constant: a recording
    cannot have been captured in the future, so this needs no maintenance and
    cannot silently expire the way a hardcoded max date does — that is exactly
    how a stale valid_timestamp_max came to reject every timestamp of every
    recording captured after it. config.max_future_skew_h leaves room for a
    device clock running slightly fast.
    """
    mask = pd.Series(False, index=ts.index)
    if config.valid_timestamp_min is not None:
        mask |= ts < pd.Timestamp(config.valid_timestamp_min, tz="UTC")
    upper = (
        pd.Timestamp(config.valid_timestamp_max, tz="UTC")
        if config.valid_timestamp_max is not None
        else pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=config.max_future_skew_h)
    )
    return mask | (ts > upper)


def flag_timestamp_outliers(
    ts: pd.Series, config: MatchConfig, anchor: pd.Timestamp | None = None
) -> pd.Series:
    """Why each timestamp is untrustworthy, or pd.NA where it looks fine.

    Index-aligned object Series of TS_* reason codes. Operates on an already
    parsed timestamp Series so both the per-recording pass (sessions.py) and
    the per-trip fallback (clean_gps_dataframe) share one implementation.

    Runs five stages, each seeing only the previous stage's survivors, so the
    reason recorded is the first — most specific — thing found wrong:

      0a. Distance from `anchor`, the capture time declared by the recording's
          own filename — but only when the file's own median timestamp
          corroborates it. When the two agree, the filename is the sharpest
          absolute check available, because it is written once the device
          closes the file and its clock has synced, while the rows it catches
          were logged before that. When they disagree wholesale the filename
          is the unreliable one: some exports are named for the day they were
          downloaded rather than the day they were recorded, and trusting the
          name there would reject an entire good recording. Corroboration
          distinguishes the two — a warm-up block is a minority of rows
          disagreeing with a filename the bulk agrees with. Only available per
          raw recording, hence the optional argument.
      0b. Absolute plausible-capture-window bound (config.valid_timestamp_*,
          plus "not in the future"), a backstop for recordings with no usable
          anchor.
      1.  Robust median/MAD range check, catching in-window anomalies (e.g. a
          stale clock reading hours before the recording actually started)
          without false-flagging the edges of a long-but-legitimate recording,
          since the tolerance scales with the recording's own spread.
      2.  Forward monotonicity pass in row order, catching subtler out-of-order
          rows that survive the statistical checks. This one is only meaningful
          while the rows are still in device emit order — run it on a frame
          that has been sorted by timestamp and it can never fire.
      3.  Duplicate-timestamp runs among Stage-2 survivors, in row order (same
          emit-order requirement as Stage 2). A device clock/GPS freeze
          repeats one reading across many consecutive rows while other
          channels keep updating; every row in such a run is flagged, not
          just the rows after the first, since nothing distinguishes which
          single row (if any) in the run carries the genuine timestamp.
          "Consecutive" means consecutive among this stage's survivors, not
          consecutive by row index — an unparseable row between two equal
          timestamps does not break the run.
    """
    reason = pd.Series(pd.NA, index=ts.index, dtype="object")
    reason[ts.isna()] = TS_UNPARSEABLE
    if not config.detect_timestamp_outliers:
        return reason

    parsed = ts.notna()
    tolerance = pd.Timedelta(hours=config.recording_date_tolerance_h)

    # Stage 0a: distance from the filename's declared capture time, if the
    # file's own centre backs that time up (see docstring).
    if anchor is not None and parsed.any():
        corroborated = abs(ts[parsed].median() - anchor) <= tolerance
        if corroborated:
            off_anchor = (ts[parsed] - anchor).abs() > tolerance
            reason.loc[off_anchor.index[off_anchor.to_numpy()]] = TS_FILENAME_MISMATCH

    # Stage 0b: absolute capture-window bound.
    survivors = parsed & reason.isna()
    out_of_window = _absolute_range_mask(ts[survivors], config)
    reason.loc[out_of_window.index[out_of_window.to_numpy()]] = TS_ABSOLUTE_WINDOW

    candidate_idx = ts.index[parsed & reason.isna()]
    if len(candidate_idx) < config.min_timestamps_for_sequentiality_check:
        return reason

    # Stage 1: robust median/MAD range check among Stage-0 survivors.
    kept = ts.loc[candidate_idx]
    epoch_s = kept.astype("int64") / 1e9
    median_s = np.median(epoch_s)
    mad_s = np.median(np.abs(epoch_s - median_s))
    tolerance_s = 1.4826 * mad_s * config.timestamp_outlier_mad_k
    mad_outlier = (epoch_s < median_s - tolerance_s) | (epoch_s > median_s + tolerance_s)
    reason.loc[kept.index[mad_outlier.to_numpy()]] = TS_MAD_OUTLIER

    # Stage 2: forward monotonicity pass over remaining survivors, in row order.
    running_max = None
    for idx in ts.index[parsed & reason.isna()]:
        t = ts.at[idx]
        if running_max is not None and t < running_max:
            reason.at[idx] = TS_NON_MONOTONIC
        else:
            running_max = t

    # Stage 3: duplicate-timestamp runs among Stage-2 survivors, in row order.
    # "Consecutive" is among these survivors, not by row index, so a run of
    # unparseable/already-rejected rows between two equal timestamps doesn't
    # break it (see docstring).
    survivor_idx = ts.index[parsed & reason.isna()]
    if len(survivor_idx) >= 2:
        values = ts.loc[survivor_idx].to_numpy()
        is_new_run = np.empty(len(values), dtype=bool)
        is_new_run[0] = True
        is_new_run[1:] = values[1:] != values[:-1]
        run_id = pd.Series(np.cumsum(is_new_run))
        run_len = run_id.map(run_id.value_counts())
        dup_mask = (run_len >= 2).to_numpy()
        reason.loc[survivor_idx[dup_mask]] = TS_DUPLICATE_TIMESTAMP

    return reason


def finalize_usable_for_hmm(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute usable_for_hmm from every validity flag currently on `df`.

    The single definition of "this row can drive the HMM". Deliberately
    derived from whichever optional flags are present rather than from a fixed
    formula, because the flags become available at different pipeline stages:
    out_of_bounds / out_of_time_bounds need the inferred route, which only
    exists after scripts/infer_route.py has run, and gps_inside_polygon needs
    the bairro boundary. Recomputing from scratch each time makes this
    idempotent, so it is safe (and intended) to call again as each new flag
    lands — see cleaning.prepare_for_matching.

    Callers must run gap/segment assignment *after* this, never before: both
    are derived from the usable set, so a segment_id computed against a larger
    usable set is stale.
    """
    df = df.copy()
    usable = df["gps_valid"] & df["timestamp_valid"]
    if "out_of_bounds" in df.columns:
        usable &= ~df["out_of_bounds"].fillna(False).astype(bool)
    if "out_of_time_bounds" in df.columns:
        usable &= ~df["out_of_time_bounds"].fillna(True).astype(bool)
    if "gps_inside_polygon" in df.columns:
        usable &= df["gps_inside_polygon"].fillna(False).astype(bool)
    df["usable_for_hmm"] = usable
    return df


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
    usable_for_hmm      : bool – see finalize_usable_for_hmm().
    timestamp_inferred  : bool – whether the timestamp was back-filled.
    ts_reject_reason    : str | NA – why the raw timestamp was rejected.

    If the frame already carries the ts_clean / ts_valid / ts_inferred columns
    that sessions.clean_recording writes (i.e. it came out of
    scripts/make_trip_data.py), those are used as-is and no timestamp checking
    is repeated here: that pass ran per raw recording, where it could see the
    device's emit order, a single session's own spread, and the recording
    filename's declared capture time. A per-trip repeat would have strictly
    less evidence and would re-derive the same answer. Frames without them
    (standalone library use, tests) get the checks applied here instead.
    """
    df = df.copy()

    # Preserve original row order
    df["original_index"] = np.arange(len(df), dtype=int)

    # --- Raw copies --------------------------------------------------------
    df["timestamp_raw"] = df[config.timestamp_col]
    df["lon_raw"] = df[config.lon_col]
    df["lat_raw"] = df[config.lat_col]

    # --- Timestamp: adopt the per-recording pass, or run it here -----------
    if TS_CLEAN_COL in df.columns:
        # Same reason as analysis.assign_points_to_edges: this column is read
        # back from CSV and mixes interpolated timestamps (fractional seconds)
        # with read ones (whole seconds). Inferring one format would silently
        # coerce every row of the other shape to NaT.
        df["timestamp"] = pd.to_datetime(
            df[TS_CLEAN_COL], errors="coerce", utc=True, format="ISO8601"
        )
        df["timestamp_inferred"] = df.get(
            TS_INFERRED_COL, pd.Series(False, index=df.index)
        ).fillna(False).astype(bool)
        if TS_REASON_COL not in df.columns:
            df[TS_REASON_COL] = pd.NA
    else:
        df["timestamp"] = pd.to_datetime(
            df[config.timestamp_col], errors="coerce", utc=True, format="mixed"
        )
        df[TS_REASON_COL] = flag_timestamp_outliers(df["timestamp"], config)
        df.loc[df[TS_REASON_COL].notna(), "timestamp"] = pd.NaT
        df["timestamp_inferred"] = False

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

    return finalize_usable_for_hmm(df)


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
    usable_for_hmm      : bool – refreshed via finalize_usable_for_hmm now
                          that out_of_time_bounds exists.
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

    return finalize_usable_for_hmm(df)


def infer_missing_timestamps(df: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    """Fill in missing timestamps by interpolating between the valid ones.

    Only runs if config.infer_missing_timestamps is True. A missing row that is
    bracketed by a valid timestamp on each side gets the two anchors' span
    divided evenly over the rows between them, so the fill always closes
    exactly on the next anchor. Dead-reckoning at a nominal sample interval
    would not: the anchors' real span is rarely an exact multiple of it, so the
    last filled row could overshoot next_ts and break the row order the
    matcher's elapsed-time cost model depends on. Where the span *is* an exact
    multiple, the two agree — that case needs no separate handling.

    Rows with inferred timestamps get timestamp_inferred=True and
    timestamp_valid=True so they can be used by the interpolation step (though
    not automatically for HMM – that still requires gps_valid to be true, which
    is checked later by finalize_usable_for_hmm).

    Rows with no valid timestamp on one side have no span to divide, so they
    fall back to stepping at a sample interval — taken from
    config.expected_sample_interval_s when provided, otherwise estimated as the
    median dt between consecutive valid timestamps. They are only filled when
    config.extrapolate_missing_timestamps is set; see that field for why the
    default is off.

    PRECONDITION: rows must be in acquisition order. Bracketing is positional
    (a missing row's timestamp comes from its position between the two nearest
    valid rows), so this cannot run after a sort that moves rows relative to
    their sample sequence — and conversely, sorting by timestamp before this
    runs would strand every NaT row at the end of the frame. In the assembled
    pipeline that ordering comes for free: this runs per raw recording, in
    device emit order.
    """
    if not config.infer_missing_timestamps:
        return df

    df = df.copy()

    valid_mask = df["timestamp_valid"]
    if valid_mask.sum() < 2:
        return df  # not enough anchor points

    # Estimate the interval if not provided. Only the extrapolation branches
    # below need it, so failing to estimate one leaves interval_s None rather
    # than aborting the bracketed fills, which don't use it.
    interval_s = config.expected_sample_interval_s
    if interval_s is None:
        valid_ts = df.loc[valid_mask, "timestamp"].sort_values()
        diffs = valid_ts.diff().dropna().dt.total_seconds()
        positive_diffs = diffs[diffs > 0]
        if not positive_diffs.empty:
            interval_s = float(positive_diffs.median())

    invalid_idx = df.index[~valid_mask]
    if invalid_idx.empty:
        return df

    valid_positions = df.index[valid_mask].tolist()
    valid_times = df.loc[valid_mask, "timestamp"].tolist()

    # Adding a Timedelta yields a nanosecond-resolution Timestamp, which pandas
    # refuses to write into a coarser column (the raw CSVs parse as
    # datetime64[us]) unless the conversion is lossless. Round each inferred
    # value to the column's own unit instead.
    ts_unit = getattr(df["timestamp"].dtype, "unit", "ns")

    for i in invalid_idx:
        # Find bracketing valid timestamps by position
        before = [(pos, t) for pos, t in zip(valid_positions, valid_times) if pos < i]
        after = [(pos, t) for pos, t in zip(valid_positions, valid_times) if pos > i]

        if before and after:
            # Divide the bracketed span evenly over the rows it spans rather
            # than dead-reckoning at interval_s, which would only land on
            # next_ts when the span happens to be an exact multiple of it.
            prev_pos, prev_ts = before[-1]
            next_pos, next_ts = after[0]
            span_s = (next_ts - prev_ts).total_seconds()
            step_s = span_s / (next_pos - prev_pos)
            inferred = prev_ts + pd.Timedelta(seconds=(i - prev_pos) * step_s)
        elif not config.extrapolate_missing_timestamps or interval_s is None:
            continue  # unbracketed: leave it missing rather than invent a time
        elif before:
            prev_pos, prev_ts = before[-1]
            inferred = prev_ts + pd.Timedelta(seconds=(i - prev_pos) * interval_s)
        else:
            next_pos, next_ts = after[0]
            inferred = next_ts - pd.Timedelta(seconds=(next_pos - i) * interval_s)

        df.at[i, "timestamp"] = inferred.as_unit(ts_unit)
        df.at[i, "timestamp_valid"] = True
        df.at[i, "timestamp_inferred"] = True

    # Re-evaluate usable_for_hmm (inferred timestamps don't make GPS valid)
    return finalize_usable_for_hmm(df)


def prepare_for_matching(
    df: pd.DataFrame,
    config: MatchConfig,
    route: Route1D | None = None,
    polygon=None,
) -> pd.DataFrame:
    """Take a raw or partly-cleaned trip frame to the state both matchers need.

    The single cleaning sequence in the codebase. netmatch.match_trip_to_network
    (route inference), netmatch.match_trip_to_network_rows (map matching) and
    pipeline.match_gps_to_route all call this, so the only thing that can
    differ between them is what evidence they have available — expressed as
    arguments, not as three hand-maintained call orders that drift apart:

      route    the trip's inferred route (routes/Inferred_Route_*.gpx). Absent
               during route inference, because the route is that stage's own
               output; passing None simply skips the checks that need it,
               rather than crashing partway through them.
      polygon  the bairro boundary. Points outside it can't be on the route
               network that was built from it.

    Order, and why:

      1. clean_gps_dataframe   parse + GPS validity. Adopts the per-recording
                               timestamp verdicts when they're on the frame.
      2. infer_missing_ts      positional bracketing, so it must precede the
                               chronological sort in step 6 (see its docstring).
      3. spatial checks        polygon membership and route-buffer distance.
                               Both are purely geometric — neither needs a
                               timestamp — so they belong together rather than
                               split apart by unrelated steps.
      4. temporal check        the walk's own time window, which needs step 3's
                               out_of_bounds to pick its anchors.
      5. finalize_usable       one formula, once, now that every flag exists.
      6. sort by timestamp     matching is a time-ordered problem from here on.
      7. gaps + segments       derived from the usable set, so strictly after 5.
    """
    df = clean_gps_dataframe(df, config)
    df = infer_missing_timestamps(df, config)

    if polygon is not None:
        df = categorize_oob(df, polygon)
    if route is not None:
        df = inspect_gps_oob(df, config, route)
        df = inspect_gps_ootb(df, config, route)

    df = finalize_usable_for_hmm(df)
    df = df.sort_values("timestamp", kind="stable")
    return assign_gaps_and_segments(df, config)
