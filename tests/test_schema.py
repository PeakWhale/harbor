import math

import numpy as np
import pandas as pd
import pytest

from conftest import mixed_frame
from peakwhale_harbor.schema import (
    InputError,
    check_records,
    infer_features,
    input_json_schema,
    match_category,
    parse_number,
)


def by_name(features):
    return {f["name"]: f for f in features}


def reasons(dropped):
    return {d["name"]: d["reason"] for d in dropped}


def test_infers_numeric_and_categorical_columns():
    features, _ = infer_features(mixed_frame(), "target")
    f = by_name(features)
    assert f["size"]["type"] == "numeric"
    assert f["rooms"]["type"] == "numeric"
    assert f["rooms"]["integer"] is True
    assert f["price_text"]["type"] == "numeric", "numbers stored as text are numbers"
    assert f["color"]["type"] == "categorical"
    assert f["color"]["categories"] == ["blue", "green", "red"]
    assert f["flag"]["type"] == "categorical"
    assert f["flag"]["categories"] == ["False", "True"]
    assert "target" not in f


def test_numeric_stats_are_recorded():
    df = mixed_frame()
    features, _ = infer_features(df, "target")
    size = by_name(features)["size"]
    assert size["min"] == pytest.approx(df["size"].min())
    assert size["max"] == pytest.approx(df["size"].max())
    assert size["median"] == pytest.approx(df["size"].median())
    assert size["missing_share"] == 0


def test_drops_columns_that_cannot_help():
    _, dropped = infer_features(mixed_frame(), "target")
    r = reasons(dropped)
    assert "ID" in r["id"]
    assert "IDs or free text" in r["customer_id"]
    assert "same value" in r["constant"]
    assert "empty" in r["empty"]
    assert "free text" in r["note"]
    assert "dates" in r["day"]


def test_user_overrides():
    features, dropped = infer_features(
        mixed_frame(), "target", categorical=["rooms"], drop=["size"], numeric=["flag"]
    )
    f = by_name(features)
    assert f["rooms"]["type"] == "categorical"
    assert f["rooms"]["categories"] == ["1", "2", "3", "4", "5"]
    assert f["flag"]["type"] == "numeric"
    assert reasons(dropped)["size"] == "you asked to drop it"


def test_unknown_override_column_suggests_a_name():
    with pytest.raises(ValueError, match="Did you mean 'rooms'"):
        infer_features(mixed_frame(), "target", categorical=["room"])


def test_column_cannot_be_both_types():
    with pytest.raises(ValueError, match="both"):
        infer_features(mixed_frame(), "target", categorical=["rooms"], numeric=["rooms"])


def test_missing_values_are_counted():
    df = mixed_frame()
    df.loc[:9, "size"] = np.nan
    df.loc[:19, "color"] = None
    f = by_name(infer_features(df, "target")[0])
    assert f["size"]["missing_share"] == pytest.approx(10 / len(df))
    assert f["color"]["missing_share"] == pytest.approx(20 / len(df))


@pytest.mark.parametrize(
    "value, expected",
    [(3, 3.0), (2.5, 2.5), ("4.25", 4.25), (" 7 ", 7.0), (None, None), ("", None), (float("nan"), None)],
)
def test_parse_number(value, expected):
    assert parse_number(value) == expected


@pytest.mark.parametrize("value", ["abc", True, [1], {"a": 1}, float("inf"), "1,5"])
def test_parse_number_rejects_bad_values(value):
    with pytest.raises(ValueError):
        parse_number(value)


def test_match_category():
    cats = ["North", "south", "2"]
    assert match_category("North", cats) == "North"
    assert match_category("north", cats) == "North", "case does not matter"
    assert match_category(2, cats) == "2"
    assert match_category("2.0", cats) == "2"
    assert match_category(None, cats) is None
    with pytest.raises(ValueError, match="Allowed values"):
        match_category("west", cats)


def test_check_records_reports_every_problem():
    features, _ = infer_features(mixed_frame(), "target")
    names = [f["name"] for f in features]
    good = {"size": 50, "rooms": 2, "color": "red", "flag": True, "price_text": 3}
    assert sorted(good) == sorted(names)

    frame, notes = check_records([good, {**good, "size": None}], features)
    assert list(frame.columns) == names
    assert math.isnan(frame.loc[1, "size"])
    assert notes[0] == []
    assert any("empty" in n for n in notes[1])

    with pytest.raises(InputError) as info:
        check_records([{**good, "size": "big", "color": "purple"}, {"size": 1}, "not a dict"], features)
    problems = info.value.problems
    assert {"row": 0, "field": "size", "error": "expected a number, got 'big'"} in problems
    assert any(p.get("field") == "color" for p in problems)
    assert any(p["row"] == 1 and p["error"] == "missing fields" for p in problems)
    assert any(p["row"] == 2 for p in problems)


def test_out_of_range_values_get_a_note():
    features, _ = infer_features(mixed_frame(), "target")
    good = {"size": 5000, "rooms": 2, "color": "red", "flag": True, "price_text": 3}
    _, notes = check_records([good], features)
    assert any("above the highest value" in n for n in notes[0])


def test_input_json_schema():
    features, _ = infer_features(mixed_frame(), "target")
    schema = input_json_schema(features, "Test")
    assert schema["type"] == "object"
    assert set(schema["required"]) == {f["name"] for f in features}
    assert schema["properties"]["color"]["enum"] == ["blue", "green", "red", None]
    assert schema["properties"]["size"]["type"] == ["number", "null"]


def test_example_record_is_valid(bundle):
    frame, notes = check_records([bundle.example()], bundle.features)
    assert len(frame) == 1
    assert notes == [[]]
    assert isinstance(frame, pd.DataFrame)
