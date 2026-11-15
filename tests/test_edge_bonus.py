"""Tests for the hand-curated bonus-edge file (edge_bonus.py)."""

import json

import pytest

from gps_route_matcher.edge_bonus import (
    add_bonus_edges,
    load_bonus_edges,
    remove_bonus_edges,
)


def _edge(u, v, key=0, **extra):
    return {"u": u, "v": v, "key": key, **extra}


def test_missing_file_gives_an_empty_set(tmp_path):
    assert load_bonus_edges("OM_1", tmp_path / "nope.json") == set()


def test_missing_track_gives_an_empty_set(tmp_path):
    path = tmp_path / "edge_bonus.json"
    add_bonus_edges("OM_1", [_edge(1, 2)], path)
    assert load_bonus_edges("OM_2", path) == set()


def test_add_then_load_round_trips_as_tuples(tmp_path):
    path = tmp_path / "edge_bonus.json"
    n = add_bonus_edges("OM_1", [_edge(1, 2, edge_order=7, note="parallel street"), _edge(2, 3)], path)
    assert n == 2
    assert load_bonus_edges("OM_1", path) == {(1, 2, 0), (2, 3, 0)}


def test_add_skips_edges_already_listed(tmp_path):
    path = tmp_path / "edge_bonus.json"
    add_bonus_edges("OM_1", [_edge(1, 2)], path)
    assert add_bonus_edges("OM_1", [_edge(1, 2), _edge(5, 6)], path) == 1
    assert load_bonus_edges("OM_1", path) == {(1, 2, 0), (5, 6, 0)}


def test_the_reverse_twin_is_a_different_edge(tmp_path):
    """Direction matters: (u, v) and (v, u) are separate directed edges."""
    path = tmp_path / "edge_bonus.json"
    add_bonus_edges("OM_1", [_edge(1, 2)], path)
    assert (2, 1, 0) not in load_bonus_edges("OM_1", path)


def test_tracks_are_independent(tmp_path):
    path = tmp_path / "edge_bonus.json"
    add_bonus_edges("OM_1", [_edge(1, 2)], path)
    add_bonus_edges("OM_2", [_edge(3, 4)], path)
    assert load_bonus_edges("OM_1", path) == {(1, 2, 0)}
    assert load_bonus_edges("OM_2", path) == {(3, 4, 0)}


def test_remove_deletes_only_the_named_edges(tmp_path):
    path = tmp_path / "edge_bonus.json"
    add_bonus_edges("OM_1", [_edge(1, 2), _edge(2, 3)], path)
    assert remove_bonus_edges("OM_1", [(1, 2, 0), (9, 9, 0)], path) == 1
    assert load_bonus_edges("OM_1", path) == {(2, 3, 0)}


def test_removing_the_last_edge_drops_the_track(tmp_path):
    path = tmp_path / "edge_bonus.json"
    add_bonus_edges("OM_1", [_edge(1, 2)], path)
    remove_bonus_edges("OM_1", [(1, 2, 0)], path)
    assert "OM_1" not in json.loads(path.read_text())["tracks"]


def test_unsupported_version_is_rejected(tmp_path):
    path = tmp_path / "edge_bonus.json"
    path.write_text(json.dumps({"version": 99, "tracks": {}}))
    with pytest.raises(ValueError, match="version"):
        load_bonus_edges("OM_1", path)
