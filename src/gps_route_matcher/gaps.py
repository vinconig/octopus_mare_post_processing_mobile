"""Gap detection and HMM segment assignment.

Segments are contiguous runs of usable GPS observations separated by a time
jump large enough to require an independent HMM run. assign_gaps_and_segments
derives the elapsed-time series once (dt_s) and thresholds segment_id from
it. Callers must run this *after* usable_for_hmm is final
(cleaning.finalize_usable_for_hmm): every column here is derived from the
usable set, so a segment_id computed against a larger one is stale.

Note: cross-segment bridging (whether an inferred network path is drawn
across a segment boundary) is a separate, spatial concept handled entirely
in netmatch.py (config.max_gap_bridge_m / min_segment_points_to_bridge_gap)
— it has no time cap of its own and does not read anything from this module
beyond segment_id itself.
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
        segment_id  : int  – counter that increments after each dt_s >
                      max_split_gap_s; each segment gets its own Viterbi run.

    Non-usable rows take the segment_id of the nearest preceding usable row
    (or 0 if there is none).
    """
    df = df.copy()
    df["dt_s"] = np.nan
    df["segment_id"] = 0

    usable_idx = df.index[df["usable_for_hmm"]]
    if len(usable_idx) == 0:
        return df

    # Real elapsed time between consecutive usable observations.
    usable_ts = pd.to_datetime(df.loc[usable_idx, "timestamp"].values)
    dts = np.full(len(usable_idx), np.nan)
    dts[1:] = (usable_ts[1:] - usable_ts[:-1]) / np.timedelta64(1, "s")
    df.loc[usable_idx, "dt_s"] = dts

    df.loc[usable_idx, "segment_id"] = np.cumsum(dts > config.max_split_gap_s)

    df["segment_id"] = _backfill_to_non_usable(df["segment_id"])

    return df
