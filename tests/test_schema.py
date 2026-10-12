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


@pytest.mark.parametrize(
    "fmt",
    [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y 0:00",
        "%b %d, %Y",
        "%d-%b-%y",
        "%B %Y",
        "%Y-%m-%d %H:%M",
    ],
)
def test_common_date_formats_are_dropped(fmt):
    df = mixed_frame()
    df["when"] = pd.date_range("2001-01-01", periods=len(df), freq="MS").strftime(fmt)
    assert "dates" in reasons(infer_features(df, "target")[1])["when"]


def test_text_that_is_not_a_date_stays_categorical():
    df = mixed_frame()
    rng = np.random.default_rng(2)
    df["month"] = rng.choice(["Jan", "Feb", "Mar"], len(df))
    df["weekday"] = rng.choice(["Mon", "Tue", "Sat"], len(df))
    df["age_group"] = rng.choice(["18-24", "25-34", "35-44"], len(df))
    df["season"] = rng.choice(["2019-2020", "2020-2021"], len(df))
    f = by_name(infer_features(df, "target")[0])
    for name in ("month", "weekday", "age_group", "season"):
        assert f[name]["type"] == "categorical", name


def test_number_id_columns_are_dropped():
    df = mixed_frame()
    rng = np.random.default_rng(3)
    ids = rng.choice(np.arange(10_000, 99_999), len(df), replace=False)
    for name in ("CustomerID", "PassengerId", "customerId", "order-id", "order_no", "invoice_number"):
        df[name] = ids
    df["row"] = rng.permutation(np.arange(1, len(df) + 1))
    df["counter"] = [str(i) for i in range(len(df))]  # text that holds numbers
    df["Paid"] = rng.uniform(0, 100, len(df))
    df["VALID"] = rng.uniform(0, 100, len(df))
    df["year"] = np.arange(1900, 1900 + len(df))
    df["zip_code"] = rng.choice([94110, 94112, 10001], len(df))
    features, dropped = infer_features(df, "target")
    r = reasons(dropped)
    for name in ("CustomerID", "PassengerId", "customerId", "order-id", "order_no", "invoice_number"):
        assert "ID" in r[name], name
    assert "row number" in r["row"] and "row number" in r["counter"]
    f = by_name(features)
    for name in ("Paid", "VALID", "year", "zip_code"):
        assert f[name]["type"] == "numeric", name


def test_user_overrides():
    features, dropped = infer_features(
        mixed_frame(), "target", categorical=["rooms"], drop=["size"], numeric=["flag"]
    )
    f = by_name(features)
    assert f["rooms"]["type"] == "categorical"
    assert f["rooms"]["categories"] == ["1", "2", "3", "4", "5"]
    assert f["flag"]["type"] == "numeric"
    assert reasons(dropped)["size"] == "you asked to drop it"


def test_user_overrides_work_for_predictions(tmp_path):
    from peakwhale_harbor import train

    df = mixed_frame(200)
    b = train(df, "target", tmp_path / "overrides", categorical=["rooms"], drop=["size"], numeric=["flag"])
    f = by_name(b.features)
    assert f["flag"]["true_false"] is True
    assert b.schema["input"]["properties"]["flag"]["type"] == ["number", "boolean", "null"]
    record = {"rooms": 2, "color": "red", "flag": True, "price_text": 3}
    results = b.predict([record, {**record, "flag": "false"}, {**record, "flag": "TRUE"}, {**record, "flag": 0}])
    for r in results:
        assert r["low"] <= r["prediction"] <= r["high"]
    # The CSV path reads every cell as text ("True"), like the batch upload does.
    table = b.predict_table(pd.DataFrame([{**record, "flag": "True", "rooms": "2"}]).astype(str))
    assert len(table) == 1
    frame, _ = check_records([record, {**record, "flag": "false"}, {**record, "flag": 0}], b.features)
    assert frame["flag"].tolist() == [1.0, 0.0, 0.0]


def test_true_false_is_not_a_number_for_other_columns():
    features, _ = infer_features(mixed_frame(), "target")
    good = {"size": True, "rooms": 2, "color": "red", "flag": True, "price_text": 3}
    with pytest.raises(InputError, match="problem"):
        check_records([good], features)


def test_spreadsheet_number_formats_are_numbers():
    df = mixed_frame()
    rng = np.random.default_rng(1)
    money = rng.uniform(1000, 900000, len(df))
    df["price"] = [f"${v:,.2f}" for v in money]
    df["count"] = [f"{int(v):,}" for v in money]
    df["share"] = [f"{v:.1f}%" for v in rng.uniform(0, 100, len(df))]
    f = by_name(infer_features(df, "target")[0])
    for name in ("price", "count", "share"):
        assert f[name]["type"] == "numeric", name
        assert "note" not in f[name], name
    assert f["price"]["max"] == pytest.approx(round(money.max(), 2))
    record = {**{n: 1 for n in f}, "color": "red", "flag": True}
    record.update({"price": "$309,715.32", "count": "2,092", "share": "12.5%"})
    frame, _ = check_records([record], list(f.values()))
    assert frame.loc[0, ["price", "count", "share"]].tolist() == [309715.32, 2092.0, 12.5]


def test_placeholders_are_empty_in_training_and_predictions():
    df = mixed_frame()
    df["size"] = df["size"].astype(object)
    df.loc[0, "size"] = "-"
    df.loc[1, "size"] = "#DIV/0!"
    df.loc[2, "size"] = "?"
    features, _ = infer_features(df, "target")
    size = by_name(features)["size"]
    assert size["type"] == "numeric"
    assert size["missing_tokens"] == ["#DIV/0!", "-", "?"]
    assert size["note"].startswith("3 values were not numbers")
    good = {"size": 50, "rooms": 2, "color": "red", "flag": True, "price_text": 3}
    frame, notes = check_records([{**good, "size": " - "}, {**good, "size": "#DIV/0!"}], features)
    assert frame["size"].isna().all()
    assert "treated as empty" in notes[0][0]
    with pytest.raises(InputError):
        check_records([{**good, "size": "n.a."}], features)


def test_decimal_commas_get_a_clear_reason():
    df = mixed_frame()
    df["weight"] = [f"{v:.2f}".replace(".", ",") for v in df["size"]]
    _, dropped = infer_features(df, "target")
    assert "decimal comma" in reasons(dropped)["weight"]


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
    [
        (3, 3.0),
        (2.5, 2.5),
        ("4.25", 4.25),
        (" 7 ", 7.0),
        (None, None),
        ("", None),
        (float("nan"), None),
        ("$309,715.32", 309715.32),
        ("2,092", 2092.0),
        ("1,234,567", 1234567.0),
        ("12%", 12.0),
        ("-$5", -5.0),
        ("$-5", -5.0),
        ("€ 1,000", 1000.0),
        ("1,000 €", 1000.0),
    ],
)
def test_parse_number(value, expected):
    assert parse_number(value) == expected


@pytest.mark.parametrize(
    "value", ["abc", True, [1], {"a": 1}, float("inf"), "1,5", 10**400, "1e400", "12,34", "$", "-", "%"]
)
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
