"""Tests for curation.py: the keep/discard/trim file the release panel exports."""

import json

import pandas as pd
import pytest

from gps_route_matcher.curation import (
    has_cuts,
    has_edit,
    has_trim,
    is_discarded,
    load_curation,
    trim_mask,
    validate_curation,
    walk_key,
)

KEY = "OM_1_20251201_afternoon"


def _doc(**entry):
    return {"version": 1, "walks": {KEY: {"decision": "keep", **entry}}}


def test_walk_key_ignores_duration_and_extension():
    assert walk_key("OM_1_20251201_afternoon_10durmin.csv") == KEY
    assert walk_key("OM_1_20251201_afternoon_7durmin") == KEY


def test_walk_key_rejects_non_trip_names():
    assert walk_key("OM_1_20251201_afternoon_10durmin 2.csv") is None
    assert walk_key("notes.csv") is None


def test_valid_document_round_trips():
    entry = {"decision": "keep", "trim_start": "2025-12-01 15:31:00", "trim_end": "2025-12-01 15:40:12"}
    assert validate_curation({"version": 1, "walks": {KEY: entry}}) == {KEY: entry}


@pytest.mark.parametrize("doc, message", [
    ({"version": 2, "walks": {}}, "version"),
    ({"version": 1}, "walks"),
    ({"version": 1, "walks": {"OM_1_20251201_afternoon_10durmin": {"decision": "keep"}}}, "walk key"),
    (_doc(decision="maybe"), "decision"),
    (_doc(trim_start="2025-12-01T15:31:00"), "trim_start"),
    (_doc(trim_end="15:31"), "trim_end"),
    (_doc(trim_start="2025-12-01 15:40:00", trim_end="2025-12-01 15:31:00"), "after"),
    (_doc(cuts="2025-12-01 15:31:00"), "cuts must be a list"),
    (_doc(cuts=[["2025-12-01 15:31:00"]]), "pair"),
    (_doc(cuts=[["2025-12-01 15:31:00", "15:40"]]), "cuts end"),
    (_doc(cuts=[["2025-12-01 15:40:00", "2025-12-01 15:31:00"]]), "after its end"),
])
def test_invalid_documents_name_the_problem(doc, message):
    with pytest.raises(ValueError, match=message):
        validate_curation(doc)


def test_load_missing_file_is_empty(tmp_path):
    assert load_curation(tmp_path / "nope.json") == {}


def test_load_reads_walks(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(_doc()))
    assert load_curation(path) == {KEY: {"decision": "keep"}}


def test_discard_and_trim_flags():
    assert is_discarded({"decision": "discard"})
    assert not is_discarded({"decision": "keep"}) and not is_discarded(None)
    assert has_trim({"decision": "keep", "trim_end": "2025-12-01 15:40:00"})
    assert not has_trim({"decision": "keep"}) and not has_trim(None)


def _ts():
    return pd.Series(["2025-12-01 15:30:00", "2025-12-01 15:30:05", "2025-12-01 15:30:10",
                      "2025-12-01 15:30:15", None], index=[10, 11, 12, 13, 14])


def test_trim_mask_is_inclusive_on_both_ends():
    mask = trim_mask(_ts(), {"decision": "keep", "trim_start": "2025-12-01 15:30:05", "trim_end": "2025-12-01 15:30:10"})
    assert mask.tolist() == [False, True, True, False, False]


def test_trim_mask_one_sided():
    assert trim_mask(_ts(), {"decision": "keep", "trim_start": "2025-12-01 15:30:10"}).tolist() == [False, False, True, True, False]
    assert trim_mask(_ts(), {"decision": "keep", "trim_end": "2025-12-01 15:30:05"}).tolist() == [True, True, False, False, False]


def test_no_trim_keeps_every_row_even_without_timestamp():
    assert trim_mask(_ts(), {"decision": "keep"}).all()
    assert trim_mask(_ts(), None).all()


CUT = ["2025-12-01 15:30:05", "2025-12-01 15:30:10"]


def test_cuts_round_trip_and_flags():
    entry = {"decision": "keep", "cuts": [CUT]}
    assert validate_curation({"version": 1, "walks": {KEY: entry}}) == {KEY: entry}
    assert has_cuts(entry) and has_edit(entry)
    assert not has_trim(entry)  # head/tail trim only
    assert not has_cuts({"decision": "keep", "cuts": []}) and not has_edit({"decision": "keep", "cuts": []})
    assert not has_edit(None)


def test_cuts_remove_the_middle_and_keep_both_ends():
    mask = trim_mask(_ts(), {"decision": "keep", "cuts": [CUT]})
    assert mask.tolist() == [True, False, False, True, False]


def test_several_cuts_and_overlap_with_trim():
    entry = {"decision": "keep", "trim_start": "2025-12-01 15:30:05",
             "cuts": [["2025-12-01 15:30:00", "2025-12-01 15:30:05"], ["2025-12-01 15:30:15", "2025-12-01 15:30:15"]]}
    assert trim_mask(_ts(), entry).tolist() == [False, False, True, False, False]
