"""Tests for the hand-curated bonus-edge file (edge_bonus.py)."""

import json

import pytest

from gps_route_matcher.edge_bonus import load_bonus_edges


def _write(path, tracks, version=1):
    path.write_text(json.dumps({"version": version, "tracks": tracks}))


def _edge(u, v, key=0, **extra):
    return {"u": u, "v": v, "key": key, **extra}


def test_missing_file_gives_an_empty_set(tmp_path):
    assert load_bonus_edges("OM_1", tmp_path / "nope.json") == set()


def test_missing_track_gives_an_empty_set(tmp_path):
    path = tmp_path / "edge_bonus.json"
    _write(path, {"OM_1": [_edge(1, 2)]})
    assert load_bonus_edges("OM_2", path) == set()


def test_load_returns_edges_as_tuples_ignoring_notes(tmp_path):
    path = tmp_path / "edge_bonus.json"
    _write(path, {"OM_1": [_edge(1, 2, edge_order=7, note="parallel street"), _edge(2, 3)]})
    assert load_bonus_edges("OM_1", path) == {(1, 2, 0), (2, 3, 0)}


def test_the_reverse_twin_is_a_different_edge(tmp_path):
    """Direction matters: (u, v) and (v, u) are separate directed edges."""
    path = tmp_path / "edge_bonus.json"
    _write(path, {"OM_1": [_edge(1, 2)]})
    assert (2, 1, 0) not in load_bonus_edges("OM_1", path)


def test_tracks_are_independent(tmp_path):
    path = tmp_path / "edge_bonus.json"
    _write(path, {"OM_1": [_edge(1, 2)], "OM_2": [_edge(3, 4)]})
    assert load_bonus_edges("OM_1", path) == {(1, 2, 0)}
    assert load_bonus_edges("OM_2", path) == {(3, 4, 0)}


def test_unsupported_version_is_rejected(tmp_path):
    path = tmp_path / "edge_bonus.json"
    _write(path, {}, version=99)
    with pytest.raises(ValueError, match="version"):
        load_bonus_edges("OM_1", path)
