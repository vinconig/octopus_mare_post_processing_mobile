"""Tests for the Zenodo export column policy (drop/keep/rename + data dictionary)."""

import logging

import pandas as pd

from gps_route_matcher.zenodo_columns import apply_column_config, build_data_dictionary_rows

_CONFIG = {
    "keep_me": {"action": "keep", "description": "kept as-is"},
    "drop_me": {"action": "drop"},
    "rename_me": {"action": "keep", "description": "kept and renamed", "rename": "renamed"},
    "caveat_me": {"action": "keep_caveat", "description": "kept with caveat", "caveat": "watch out"},
}


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "keep_me": [1, 2],
            "drop_me": [3, 4],
            "rename_me": [5, 6],
            "caveat_me": [7, 8],
        }
    )


def test_apply_column_config_drops_and_renames():
    out = apply_column_config(_df(), _CONFIG)

    assert list(out.columns) == ["keep_me", "renamed", "caveat_me"]
    assert out["renamed"].tolist() == [5, 6]


def test_apply_column_config_unknown_column_defaults_to_keep(caplog):
    df = _df()
    df["unlisted"] = [9, 10]

    with caplog.at_level(logging.WARNING):
        out = apply_column_config(df, _CONFIG)

    assert "unlisted" in out.columns
    assert any("unlisted" in record.message for record in caplog.records)


def test_apply_column_config_sensor_stat_columns_no_warning(caplog):
    df = _df()
    df["pm2.5_cal_avg"] = [1.0, 2.0]
    df["temperature_max"] = [3.0, 4.0]

    with caplog.at_level(logging.WARNING):
        out = apply_column_config(df, _CONFIG)

    assert "pm2.5_cal_avg" in out.columns and "temperature_max" in out.columns
    assert not caplog.records


def test_build_data_dictionary_rows_excludes_dropped_and_includes_caveat():
    out = apply_column_config(_df(), _CONFIG)
    rows = build_data_dictionary_rows("matched", out, _CONFIG)

    by_column = {row["column"]: row for row in rows}
    assert "drop_me" not in by_column
    assert by_column["keep_me"]["caveat"] == ""
    assert by_column["caveat_me"]["caveat"] == "watch out"
    assert by_column["renamed"]["description"] == "kept and renamed"
    assert all(row["dataset"] == "matched" for row in rows)


def test_build_data_dictionary_rows_describes_unlisted_sensor_stat_column():
    df = pd.DataFrame(columns=["pm2.5_cal_avg", "temperature_max"])
    rows = build_data_dictionary_rows("edge_analysis", df, {})

    by_column = {row["column"]: row for row in rows}
    assert "mean" in by_column["pm2.5_cal_avg"]["description"]
    assert "maximum" in by_column["temperature_max"]["description"]
