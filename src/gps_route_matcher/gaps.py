"""Gap detection and HMM segment assignment.

A "gap" is a large time jump between consecutive usable GPS observations.
Segments are contiguous runs of usable observations separated by gaps
large enough to require independent HMM runs.

Both are read off the same elapsed-time series, differing only in threshold
(max_bridge_gap_s vs max_split_gap_s), so assign_gaps_and_segments computes it
once and derives both. Callers must run this *after* usable_for_hmm is final
(cleaning.finalize_usable_for_hmm): every column here is derived from the
usable set, so a segment_id computed against a larger one is stale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MatchConfig


def _backfill_to_non_usable(values: pd.Series) -> pd.Series:
    """Carry a usable-row-only integer label forward onto the non-usable rows
    between labels, so every row belongs to some group.

    The blanks have to become NaN before the ffill: filling the column with a
    literal 0 first and then calling .ffill() leaves nothing for ffill to
    propagate, which silently pins every non-usable row past the first group
    to group 0 — and both interpolation paths then look up that row's anchors
    in the wrong group.
    """
    return values.replace(0, np.nan).ffill().fillna(0).astype(int)


def assign_gaps_and_segments(df: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    """Annotate the dataframe with gap and segment information.

    Adds columns:
        dt_s        : float – seconds since the previous usable observation
                      (NaN for the first usable row and all non-usable rows).
        gap_before  : bool – True when dt_s > max_bridge_gap_s.
        gap_id      : int  – counter that increments after each gap (0-based).
        segment_id  : int  – counter that increments after each dt_s >
                      max_split_gap_s; each segment gets its own Viterbi run.

    Non-usable rows take the gap_id / segment_id of the nearest preceding
    usable row (or 0 if there is none).
    """
    df = df.copy()
    df["dt_s"] = np.nan
    df["gap_before"] = False
    df["gap_id"] = 0
    df["segment_id"] = 0

    usable_idx = df.index[df["usable_for_hmm"]]
    if len(usable_idx) == 0:
        return df

    # Real elapsed time between consecutive usable observations — the one
    # series both thresholds are read off.
    usable_ts = pd.to_datetime(df.loc[usable_idx, "timestamp"].values)
    dts = np.full(len(usable_idx), np.nan)
    dts[1:] = (usable_ts[1:] - usable_ts[:-1]) / np.timedelta64(1, "s")
    df.loc[usable_idx, "dt_s"] = dts

    gap_before = dts > config.max_bridge_gap_s
    df.loc[usable_idx[gap_before], "gap_before"] = True
    df.loc[usable_idx, "gap_id"] = np.cumsum(gap_before)
    df.loc[usable_idx, "segment_id"] = np.cumsum(dts > config.max_split_gap_s)

    df["gap_id"] = _backfill_to_non_usable(df["gap_id"])
    df["segment_id"] = _backfill_to_non_usable(df["segment_id"])

    return df


def detect_gaps(df: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    """Deprecated: use assign_gaps_and_segments, which also assigns segment_id
    from the same elapsed-time series. Kept so existing callers and tests that
    only want dt_s / gap_before / gap_id keep working."""
    return assign_gaps_and_segments(df, config)


def assign_match_segments(df: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    """Deprecated: use assign_gaps_and_segments. Kept as an alias; it is now
    idempotent with detect_gaps rather than a second pass over the same data."""
    return assign_gaps_and_segments(df, config)
