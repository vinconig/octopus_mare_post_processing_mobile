"""Gap detection and HMM segment assignment.

A "gap" is a large time jump between consecutive usable GPS observations.
Segments are contiguous runs of usable observations separated by gaps
large enough to require independent HMM runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MatchConfig


def detect_gaps(df: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    """Annotate the dataframe with gap information.

    Adds columns:
        dt_s        : float – seconds since the previous usable observation
                      (NaN for the first usable row and all non-usable rows).
        gap_before  : bool – True when dt_s > max_bridge_gap_s.
        gap_id      : int  – counter that increments after each gap (0-based).
    """
    df = df.copy()
    df["dt_s"] = np.nan
    df["gap_before"] = False
    df["gap_id"] = 0

    usable = df[df["usable_for_hmm"]].copy()
    if usable.empty:
        return df

    # Compute dt between consecutive usable observations (real time elapsed)
    usable_ts = usable["timestamp"].values
    dts = np.full(len(usable), np.nan)
    dts[1:] = (
        pd.to_datetime(usable_ts[1:]) - pd.to_datetime(usable_ts[:-1])
    ) / np.timedelta64(1, "s")

    df.loc[usable.index, "dt_s"] = dts

    # Mark gaps
    gap_mask = dts > config.max_bridge_gap_s
    df.loc[usable.index[gap_mask], "gap_before"] = True

    # Assign gap_id: increments each time gap_before is True
    gap_cumsum = df.loc[usable.index, "gap_before"].cumsum().values
    df.loc[usable.index, "gap_id"] = gap_cumsum

    # Forward-fill gap_id to non-usable rows so every row belongs to a gap group
    df["gap_id"] = df["gap_id"].replace(0, np.nan)
    df["gap_id"] = df["gap_id"].ffill().fillna(0).astype(int)

    return df


def assign_match_segments(df: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    """Assign each row to an HMM segment.

    Segments are split whenever the time gap between consecutive usable
    observations exceeds max_split_gap_s. Each segment will get its own
    independent Viterbi run.

    Adds column:
        segment_id : int (0-based). Non-usable rows are assigned the segment
                     of the nearest preceding usable row (or 0 if none).
    """
    df = df.copy()
    df["segment_id"] = 0

    usable = df[df["usable_for_hmm"]]
    if usable.empty:
        return df

    usable_idx = usable.index.tolist()
    usable_ts = usable["timestamp"].values

    segment_ids = np.zeros(len(usable_idx), dtype=int)
    current_seg = 0

    for k in range(1, len(usable_idx)):
        dt_s = (
            pd.Timestamp(usable_ts[k]) - pd.Timestamp(usable_ts[k - 1])
        ).total_seconds()
        if dt_s > config.max_split_gap_s:
            current_seg += 1
        segment_ids[k] = current_seg

    df.loc[usable_idx, "segment_id"] = segment_ids

    # Forward-fill to non-usable rows
    df["segment_id"] = df["segment_id"].ffill().fillna(0).astype(int)

    return df
