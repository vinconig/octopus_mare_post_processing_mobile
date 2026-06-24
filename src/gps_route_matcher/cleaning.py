"""Input cleaning: parse timestamps, validate GPS, flag usable rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MatchConfig


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
    df["timestamp"] = pd.to_datetime(df[config.timestamp_col], errors="coerce", utc=True)
    df["timestamp_valid"] = df["timestamp"].notna()

    # --- Parse lon/lat -----------------------------------------------------
    df["lon"] = pd.to_numeric(df[config.lon_col], errors="coerce")
    df["lat"] = pd.to_numeric(df[config.lat_col], errors="coerce")

    # --- GPS validity -------------------------------------------------------
    lon_ok = df["lon"].notna()
    lat_ok = df["lat"].notna()

    if config.treat_zero_zero_as_missing:
        zero_zero = (df["lon"] == 0.0) & (df["lat"] == 0.0)
        gps_valid = lon_ok & lat_ok & ~zero_zero
    else:
        gps_valid = lon_ok & lat_ok

    df["gps_valid"] = gps_valid
    df["usable_for_hmm"] = df["timestamp_valid"] & df["gps_valid"]
    df["timestamp_inferred"] = False

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
