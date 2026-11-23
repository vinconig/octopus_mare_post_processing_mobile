"""Tests for the Zenodo export column policy (whitelist/rename/transform/order) and its README documentation."""

from pathlib import Path

import pandas as pd
import pytest

from gps_route_matcher.paths import ZENODO_README_PATH
from gps_route_matcher.zenodo_columns import (
    EDGE_ANALYSIS_COLUMNS,
    MATCHED_COLUMNS,
    apply_column_config,
    to_local_timestamp,
)

_CONFIG = {
    "keep_me": {},
    "rename_me": {"rename": "renamed"},
    "caveat_me": {},
    "unit_me": {},
    "shout_me": {"transform": lambda s: s * 10},
}


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unlisted": [9, 10],
            "unit_me": [0.1, 0.2],
            "shout_me": [1, 2],
            "rename_me": [5, 6],
            "keep_me": [1, 2],
            "caveat_me": [7, 8],
        }
    )


def test_apply_column_config_whitelists_orders_renames_and_transforms():
    out = apply_column_config(_df(), _CONFIG)

    assert list(out.columns) == ["keep_me", "renamed", "caveat_me", "unit_me", "shout_me"]
    assert out["renamed"].tolist() == [5, 6]
    assert out["shout_me"].tolist() == [10, 20]


def test_apply_column_config_missing_configured_column_raises():
    with pytest.raises(KeyError):
        apply_column_config(_df().drop(columns=["keep_me"]), _CONFIG)


def test_to_local_timestamp_shifts_minus_3h_rounds_to_second_and_drops_offset():
    series = pd.Series(
        [
            "2025-12-19 12:30:21+00:00",
            "2025-12-19 12:30:43.500000+00:00",  # edge_analysis timestamp_mean can carry fractional seconds
            "2025-12-19 02:00:00.400000+00:00",  # crosses midnight local
            None,
        ]
    )

    out = to_local_timestamp(series)

    assert out.iloc[0] == "2025-12-19 09:30:21"
    assert out.iloc[1] == "2025-12-19 09:30:44"  # .5 rounds half to even -> :44
    assert out.iloc[2] == "2025-12-18 23:00:00"
    assert pd.isna(out.iloc[3])


def test_matched_edge_ids_write_as_integers_with_unresolved_rows_empty():
    df = pd.DataFrame({
        "matched_edge_u": [11214125333.0, float("nan")],
        "matched_edge_v": [11214125331.0, float("nan")],
        "matched_edge_key": [0.0, float("nan")],
    })
    config = {k: MATCHED_COLUMNS[k] for k in df.columns}

    csv = apply_column_config(df, config).to_csv(index=False)

    assert csv.splitlines() == [
        "matched_edge_u,matched_edge_v,matched_edge_key",
        "11214125333,11214125331,0",
        ",,",
    ]


def test_matched_columns_are_exactly_the_published_set_in_order():
    shipped = [entry.get("rename", source) for source, entry in MATCHED_COLUMNS.items()]

    assert shipped == [
        "timestamp", "longitude", "latitude", "temperature_raw", "humidity_raw", "pm2.5_raw",
        "match_status", "matched_edge_u", "matched_edge_v", "matched_edge_key",
        "longitude_raw", "latitude_raw",
    ]
    assert "PM2.5" in MATCHED_COLUMNS
    assert "out_of_time_bounds" not in MATCHED_COLUMNS


def test_edge_analysis_ships_only_raw_sensor_stats():
    shipped = {entry.get("rename", source) for source, entry in EDGE_ANALYSIS_COLUMNS.items()}

    assert {"pm2.5_raw_avg", "pm2.5_raw_max", "temperature_raw_avg", "humidity_raw_max"} <= shipped
    assert not {"pm1.0_avg", "pm4.0_max", "pm10.0_avg", "pm2.5_avg"} & shipped
    assert "pm2.5_avg" in EDGE_ANALYSIS_COLUMNS  # the raw source column (renamed on the way out)


def test_zenodo_readme_mentions_every_shipped_column():
    readme = ZENODO_README_PATH.read_text()

    for config in (MATCHED_COLUMNS, EDGE_ANALYSIS_COLUMNS):
        for source, entry in config.items():
            shipped = entry.get("rename", source)
            assert f"`{shipped}`" in readme, f"{shipped} is shipped but not documented in ZENODO_README.md"
