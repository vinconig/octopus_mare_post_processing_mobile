"""Tests for curation_apply.py: applying curation decisions to a Zenodo package."""

import json

import numpy as np
import pandas as pd
import pytest

from gps_route_matcher.analysis import edge_analysis_from_df
from gps_route_matcher.curation_apply import (
    apply_curation,
    edge_analysis_from_shipped,
    read_shipped_walk,
    shipped_to_pipeline_schema,
)
from gps_route_matcher.zenodo_columns import (
    EDGE_ANALYSIS_COLUMNS,
    MATCHED_COLUMNS,
    MIN_MATCHED_EDGES,
    apply_column_config,
)

N_EDGES, PTS_PER_EDGE, STEP_S = 8, 3, 5
LOCAL_START = pd.Timestamp("2025-12-01 15:30:00")  # UTC-3; 18:30:00 UTC


def _route_metadata() -> dict:
    return {
        "edges": [{"edge_order": i, "u": 100 + i, "v": 101 + i, "key": 0} for i in range(N_EDGES)],
        "edge_geometry": {f"({100 + i},{101 + i},0)": {"length": 10.0 + i, "geometry": f"LINESTRING ({i} 0, {i + 1} 0)"}
                          for i in range(N_EDGES)},
    }


def _source_walk(offset: float = 0.0) -> pd.DataFrame:
    """A matched trip in the pipeline's schema: PTS_PER_EDGE points on each route edge, one
    interpolated point and one unresolved one; values vary with the point so stats differ per edge."""
    n = N_EDGES * PTS_PER_EDGE
    utc = pd.date_range(LOCAL_START + pd.Timedelta(hours=3), periods=n, freq=f"{STEP_S}s")
    i = np.arange(n)
    status = np.array(["matched"] * n, dtype=object)
    status[4] = "interpolated"
    status[7] = "unresolved_large_gap"
    edge = i // PTS_PER_EDGE
    df = pd.DataFrame({
        "timestamp": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in utc],
        "matched_lon": -43.25 + i * 1e-5, "matched_lat": -22.84 + i * 1e-5,
        "Temperature": 30.0 + (i % 5) + offset, "Humidity": 60.0 - (i % 7), "PM2.5": 5.0 + i * 0.5,
        "match_status": status,
        "matched_edge_u": (100 + edge).astype(float), "matched_edge_v": (101 + edge).astype(float),
        "matched_edge_key": np.zeros(n),
        "Longitude": -43.25 + i * 1e-5, "Latitude": -22.84 + i * 1e-5,
        "interpolated": status == "interpolated", "out_of_time_bounds": False,
    })
    for c in ("matched_edge_u", "matched_edge_v", "matched_edge_key"):
        df.loc[status == "unresolved_large_gap", c] = np.nan
    return df


def _ship(source: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(data/mobile frame, data/mobile_agg_by_edge frame) exactly as package_for_zenodo writes them."""
    return (apply_column_config(source, MATCHED_COLUMNS),
            apply_column_config(edge_analysis_from_df(source, meta), EDGE_ANALYSIS_COLUMNS))


def _local(idx: int) -> str:
    return (LOCAL_START + pd.Timedelta(seconds=STEP_S * idx)).strftime("%Y-%m-%d %H:%M:%S")


# ── the adapter ──────────────────────────────────────────────────────────────

def test_shipped_schema_maps_back_to_the_pipeline_schema():
    source = _source_walk()
    back = shipped_to_pipeline_schema(_ship(source, _route_metadata())[0])
    # local -> UTC again (same instants; the separator between date and time is immaterial)
    assert pd.to_datetime(back["timestamp"]).tolist() == pd.to_datetime(source["timestamp"]).tolist()
    assert back["PM2.5"].tolist() == source["PM2.5"].tolist()
    assert back["interpolated"].tolist() == source["interpolated"].tolist()


def test_only_raw_sensor_columns_ship():
    meta = _route_metadata()
    shipped, edges = _ship(_source_walk(), meta)
    sensor_cols = lambda cols: [c for c in cols if c.startswith(("temperature", "humidity", "pm"))]  # noqa: E731
    assert sensor_cols(shipped.columns) == ["temperature_raw", "humidity_raw", "pm2.5_raw"]
    assert sensor_cols(edges.columns) == [
        f"{s}_raw_{stat}" for s in ("temperature", "humidity", "pm2.5") for stat in ("avg", "max")
    ]
    # the shipped PM2.5 is the recorded reading
    assert shipped["pm2.5_raw"].tolist() == _source_walk()["PM2.5"].tolist()


def test_rebuilt_edge_file_equals_the_shipped_one_for_an_untouched_walk():
    meta = _route_metadata()
    shipped, want = _ship(_source_walk(), meta)
    got = edge_analysis_from_shipped(shipped, meta)
    assert list(got.columns) == list(want.columns)
    for col in want.columns:
        if col == "timestamp_mean":  # midpoint of already-rounded timestamps: may differ by 1 s
            diff = (pd.to_datetime(got[col]) - pd.to_datetime(want[col])).abs().dropna()
            assert (diff <= pd.Timedelta(seconds=1)).all()
        else:
            pd.testing.assert_series_equal(got[col], want[col], check_exact=False, rtol=1e-12)


def test_shipped_walk_round_trips_through_csv_byte_for_byte(tmp_path):
    shipped, _ = _ship(_source_walk(), _route_metadata())
    path = tmp_path / "w.csv"
    shipped.to_csv(path, index=False)
    read_shipped_walk(path).to_csv(tmp_path / "again.csv", index=False)
    assert (tmp_path / "again.csv").read_text() == path.read_text()


# ── applying a curation file to a package ────────────────────────────────────

UNCHANGED, DISCARDED, CUT, TRIMMED_AWAY = (
    "OM_1_20251201_afternoon", "OM_1_20251202_morning", "OM_1_20251203_morning", "OM_1_20251204_afternoon")
_FILES = {k: f"{k}_2durmin.csv" for k in (UNCHANGED, DISCARDED, CUT, TRIMMED_AWAY)}


def _make_package(root, keys=tuple(_FILES)):
    meta = _route_metadata()
    for sub in ("data/mobile", "data/mobile_agg_by_edge", "metadata/route_metadata"):
        (root / sub).mkdir(parents=True)
    (root / "metadata/route_metadata/OM_1_inferred_route.json").write_text(json.dumps(meta))
    (root / "ZENODO_README.md").write_text("readme")
    for k in keys:
        shipped, edges = _ship(_source_walk(), meta)
        shipped.to_csv(root / "data/mobile" / _FILES[k], index=False)
        edges.to_csv(root / "data/mobile_agg_by_edge" / _FILES[k], index=False)
    return root


def _curation(tmp_path, walks):
    path = tmp_path / "curation.json"
    path.write_text(json.dumps({"version": 1, "walks": walks}))
    return path


def _decisions(tmp_path):
    n = N_EDGES * PTS_PER_EDGE
    return _curation(tmp_path, {
        DISCARDED: {"decision": "discard"},
        # cut edges 3 and 4 (points 9..14), keep head and tail
        CUT: {"decision": "keep", "cuts": [[_local(9), _local(14)]]},
        # trim to the first four edges: below MIN_MATCHED_EDGES once the rest is gone
        TRIMMED_AWAY: {"decision": "keep", "trim_end": _local(11)},
        "OM_1_20251231_morning": {"decision": "keep"},  # matches no walk in the package
    }), n


def test_apply_curation_discards_cuts_and_copies_the_rest(tmp_path):
    pkg, out = _make_package(tmp_path / "pkg"), tmp_path / "out"
    path, n = _decisions(tmp_path)
    before = {p.name: p.read_bytes() for p in (pkg / "data/mobile").iterdir()}

    s = apply_curation(package_dir=pkg, out_dir=out, curation_path=path, folders=["OM_1"])

    assert (s.n_unchanged, s.n_edited, s.n_discarded, s.n_too_few_edges) == (1, 1, 1, 1)
    assert s.n_rows_removed == 6 and s.unmatched_keys == ["OM_1_20251231_morning"]
    mobile = sorted(p.name for p in (out / "data/mobile").iterdir())
    edges = sorted(p.name for p in (out / "data/mobile_agg_by_edge").iterdir())
    assert mobile == edges and len(mobile) == 2
    # the untouched walk is a byte-for-byte copy
    assert (out / "data/mobile" / _FILES[UNCHANGED]).read_bytes() == before[_FILES[UNCHANGED]]
    # the package itself is never modified
    assert {p.name: p.read_bytes() for p in (pkg / "data/mobile").iterdir()} == before
    # the rest of the package comes along
    assert (out / "ZENODO_README.md").read_text() == "readme"
    assert (out / "metadata/route_metadata/OM_1_inferred_route.json").is_file()

    cut_name = next(n for n in mobile if n.startswith(CUT))
    cut = pd.read_csv(out / "data/mobile" / cut_name)
    full = pd.read_csv(pkg / "data/mobile" / _FILES[CUT])
    assert len(cut) == len(full) - 6
    # survivors keep their real timestamps: head and tail remain, with a jump where the cut was
    assert cut["timestamp"].tolist() == full["timestamp"].drop(range(9, 15)).tolist()
    assert _local(9) not in cut["timestamp"].tolist() and _local(8) in cut["timestamp"].tolist()
    # the filename duration is the first-to-last span, so it still covers the cut stretch
    assert cut_name == f"{CUT}_{(N_EDGES * PTS_PER_EDGE - 1) * STEP_S // 60}durmin.csv"


def test_edited_walks_per_edge_file_is_rebuilt_without_the_cut_rows(tmp_path):
    pkg, out = _make_package(tmp_path / "pkg"), tmp_path / "out"
    path, _ = _decisions(tmp_path)
    apply_curation(package_dir=pkg, out_dir=out, curation_path=path, folders=["OM_1"])
    name = next(p.name for p in (out / "data/mobile_agg_by_edge").iterdir() if p.name.startswith(CUT))
    edges = pd.read_csv(out / "data/mobile_agg_by_edge" / name).set_index("edge_order")
    # edge 2 has 2 points: the unresolved one (point 7) never reaches an edge
    assert edges["n_points"].tolist() == [3, 3, 2, 0, 0, 3, 3, 3]
    # edges outside the cut keep the numbers they had in the untouched walk
    full = pd.read_csv(pkg / "data/mobile_agg_by_edge" / _FILES[CUT]).set_index("edge_order")
    pd.testing.assert_series_equal(edges.loc[[0, 1, 2, 5, 6, 7], "pm2.5_raw_avg"], full.loc[[0, 1, 2, 5, 6, 7], "pm2.5_raw_avg"])
    assert edges.loc[[3, 4], "pm2.5_raw_avg"].isna().all()


def test_without_curation_the_output_is_a_plain_copy(tmp_path):
    pkg, out = _make_package(tmp_path / "pkg"), tmp_path / "out"
    s = apply_curation(package_dir=pkg, out_dir=out, curation_path=None, folders=["OM_1"])
    assert s.n_unchanged == len(_FILES) and s.n_edited == s.n_discarded == 0
    for sub in ("data/mobile", "data/mobile_agg_by_edge"):
        for p in (pkg / sub).iterdir():
            assert (out / sub / p.name).read_bytes() == p.read_bytes()


def test_rerun_clears_walks_that_were_discarded_since(tmp_path):
    pkg, out = _make_package(tmp_path / "pkg"), tmp_path / "out"
    apply_curation(package_dir=pkg, out_dir=out, curation_path=None, folders=["OM_1"])
    path = _curation(tmp_path, {DISCARDED: {"decision": "discard"}})
    apply_curation(package_dir=pkg, out_dir=out, curation_path=path, folders=["OM_1"])
    assert not (out / "data/mobile" / _FILES[DISCARDED]).exists()
    assert not (out / "data/mobile_agg_by_edge" / _FILES[DISCARDED]).exists()


def test_no_overwrite_keeps_existing_output(tmp_path):
    pkg, out = _make_package(tmp_path / "pkg"), tmp_path / "out"
    apply_curation(package_dir=pkg, out_dir=out, curation_path=None, folders=["OM_1"])
    target = out / "data/mobile" / _FILES[UNCHANGED]
    target.write_text("hand-edited")
    s = apply_curation(package_dir=pkg, out_dir=out, curation_path=None, folders=["OM_1"], overwrite=False)
    assert target.read_text() == "hand-edited" and s.n_skipped_existing == len(_FILES)


def test_refuses_to_overwrite_its_own_input_and_a_missing_package(tmp_path):
    pkg = _make_package(tmp_path / "pkg")
    with pytest.raises(SystemExit, match="different folder"):
        apply_curation(package_dir=pkg, out_dir=pkg, curation_path=None)
    with pytest.raises(SystemExit, match="package_for_zenodo"):
        apply_curation(package_dir=tmp_path / "nope", out_dir=tmp_path / "out", curation_path=None)


def test_min_matched_edges_constant_is_what_the_cut_test_assumes():
    assert 4 < MIN_MATCHED_EDGES <= N_EDGES - 2  # 4 edges fail the gate, 6 pass it
