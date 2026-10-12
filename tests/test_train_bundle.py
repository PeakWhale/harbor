import json
import shutil

import numpy as np
import pandas as pd
import pytest
import skops.io as sio

from conftest import EXAMPLE_CSV, TARGET, mixed_frame
from peakwhale_harbor import BundleError, TrainingError, load_bundle, train
from peakwhale_harbor.bundle import MODEL_FILE, VersionWarning, file_sha256, predict_with_range
from peakwhale_harbor.schema import to_model_frame
from peakwhale_harbor.train import conformal_offset


def test_bundle_has_all_files(bundle_dir):
    names = sorted(p.name for p in bundle_dir.iterdir())
    assert names == ["metadata.json", "metrics.json", "model.skops", "schema.json"]
    assert not list(bundle_dir.glob("*.pkl")), "no pickle files"


def test_schema_describes_inputs_and_outputs(bundle):
    schema = bundle.schema
    assert schema["format"] == "harbor-bundle"
    assert schema["task"] == "regression"
    assert schema["target"]["name"] == TARGET
    types = {f["name"]: f["type"] for f in schema["features"]}
    assert types["machine_type"] == "categorical"
    assert types["vibration_mm_s"] == "numeric"
    assert "asset_id" in {d["name"] for d in schema["dropped_columns"]}
    assert schema["input"]["required"] == bundle.feature_names
    assert {"prediction", "low", "high"} <= set(schema["output"]["properties"])


def test_metrics_are_honest_and_sane(bundle):
    m = bundle.metrics
    assert m["test_rows"] == 240 and m["train_rows"] == 720 and m["calibration_rows"] == 240
    assert 0 < m["r2"] <= 1
    assert m["mae"] < m["baseline_mae"], "the model must beat always guessing the median"
    assert m["rmse"] >= m["mae"]
    assert 0.8 <= m["range_coverage"] <= 1.0
    assert m["range_coverage_target"] == 0.9


def test_metadata_records_training(bundle, bundle_dir):
    meta = bundle.metadata
    assert meta["source"]["file"] == EXAMPLE_CSV.name
    assert meta["source"]["sha256"] == file_sha256(EXAMPLE_CSV)
    assert meta["rows"]["used"] == 1200
    assert meta["split"]["seed"] == 42
    assert meta["options"]["target"] == TARGET
    assert {"python", "scikit-learn", "pandas", "numpy", "skops"} <= set(meta["versions"])
    assert meta["model_sha256"] == file_sha256(bundle_dir / MODEL_FILE)


def test_round_trip_gives_same_predictions(bundle, bundle_dir):
    df = pd.read_csv(EXAMPLE_CSV)
    again = load_bundle(bundle_dir)
    frame = to_model_frame(df, bundle.features)
    a = bundle.predict_frame(frame)
    b = again.predict_frame(frame)
    for key in ("prediction", "low", "high"):
        np.testing.assert_allclose(a[key], b[key])
    # The record path (used by the API) gives the same numbers as the table path.
    records = df[bundle.feature_names].head(20).to_dict("records")
    results = again.predict(records)
    np.testing.assert_allclose([r["prediction"] for r in results], a["prediction"][:20])


def test_training_is_repeatable(tmp_path, bundle):
    again = train(EXAMPLE_CSV, TARGET, tmp_path / "again")
    assert again.metrics["mae"] == pytest.approx(bundle.metrics["mae"])
    assert again.metrics["range_coverage"] == pytest.approx(bundle.metrics["range_coverage"])


def test_train_from_dataframe(tmp_path):
    b = train(mixed_frame(300), "target", tmp_path / "mixed", title="Mixed")
    assert b.title == "Mixed"
    assert b.metadata["source"]["file"] is None
    assert b.metrics["r2"] > 0.5
    assert set(b.feature_names) == {"size", "rooms", "color", "flag", "price_text"}


def test_rows_with_empty_target_are_skipped(tmp_path):
    df = mixed_frame(300)
    df.loc[:9, "target"] = np.nan
    b = train(df, "target", tmp_path / "skip")
    assert b.metadata["rows"]["skipped_missing_target"] == 10
    assert b.metadata["rows"]["used"] == 290


def test_non_negative_looks_at_every_row(tmp_path):
    from sklearn.model_selection import train_test_split

    df = mixed_frame(300)
    df["target"] = df["target"].abs()
    # Put the only negative target in the test rows, which the models never see.
    _, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    df.loc[test_idx[0], "target"] = -5.0
    b = train(df, "target", tmp_path / "neg")
    assert b.target["non_negative"] is False
    assert b.target["min"] == -5.0


def test_infinite_values_are_treated_as_empty(tmp_path):
    df = mixed_frame(300)
    df.loc[0, "target"] = np.inf
    df.loc[1, "target"] = -np.inf
    df.loc[2, "size"] = np.inf
    df["price_text"] = df["price_text"].astype(object)
    df.loc[3, "price_text"] = "-inf"
    b = train(df, "target", tmp_path / "inf")
    assert b.metadata["rows"]["skipped_missing_target"] == 2
    f = {x["name"]: x for x in b.features}
    assert np.isfinite(f["size"]["max"]) and "1 values were not numbers" in f["size"]["note"]
    assert np.isfinite(f["price_text"]["min"])
    for name in ("schema.json", "metrics.json", "metadata.json"):
        json.loads((tmp_path / "inf" / name).read_text(), parse_constant=pytest.fail)


def test_infinite_target_in_a_csv_file(tmp_path):
    src = tmp_path / "data.csv"
    df = mixed_frame(300)
    df["target"] = df["target"].astype(object)
    df.loc[0, "target"] = "inf"
    df.to_csv(src, index=False)
    b = train(src, "target", tmp_path / "inf")
    assert b.metadata["rows"]["skipped_missing_target"] == 1


def test_metric_that_cannot_be_computed_is_null(tmp_path):
    # One test row: R² is not defined. It must not be written as NaN (not valid JSON).
    with pytest.warns(UserWarning, match="R"):
        b = train(mixed_frame(60), "target", tmp_path / "one", test_size=0.01)
    assert b.metrics["test_rows"] == 1
    assert b.metrics["r2"] is None
    json.loads((tmp_path / "one" / "metrics.json").read_text(), parse_constant=pytest.fail)
    from peakwhale_harbor.server import create_app

    assert create_app(b).test_client().get("/").status_code == 200


def test_unknown_target_suggests_a_column(tmp_path):
    with pytest.raises(TrainingError, match="Did you mean 'repair_hours'"):
        train(EXAMPLE_CSV, "repair_hour", tmp_path / "x")


def test_formatted_target_is_read(tmp_path):
    df = mixed_frame(300)
    df["target"] = [f"${v:,.2f}" for v in df["target"] * 1000]
    lines = []
    b = train(df, "target", tmp_path / "money", log=lines.append)
    assert b.metadata["rows"]["used"] == 300
    assert b.target["max"] > 1000


def test_decimal_comma_target_gets_a_hint(tmp_path):
    df = mixed_frame(300)
    df["target"] = [f"{v:.2f}".replace(".", ",") for v in df["target"]]
    with pytest.raises(TrainingError, match="decimal comma"):
        train(df, "target", tmp_path / "x")


def test_training_log_shows_column_notes(tmp_path):
    df = mixed_frame(300)
    df["size"] = df["size"].astype(object)
    df.loc[0, "size"] = "?"
    lines = []
    train(df, "target", tmp_path / "notes", log=lines.append)
    assert any("input  size" in line and "1 values were not numbers" in line for line in lines)


def test_text_target_is_refused(tmp_path):
    with pytest.raises(TrainingError, match="must contain numbers"):
        train(EXAMPLE_CSV, "machine_type", tmp_path / "x")


def test_too_few_rows(tmp_path):
    with pytest.raises(TrainingError, match="at least"):
        train(mixed_frame(20), "target", tmp_path / "x")


def test_no_usable_columns(tmp_path):
    df = pd.DataFrame({"id": range(100), "constant": 1, "y": np.arange(100) * 2.0})
    with pytest.raises(TrainingError, match=r"no usable input columns.*id \(looks like an ID"):
        train(df, "y", tmp_path / "x")


def test_reads_windows_excel_csv_files(tmp_path):
    """Excel on Windows: Windows-1252 text, and a semicolon in many European languages."""
    df = mixed_frame(300)
    df["color"] = df["color"].map({"red": "rouge", "green": "vert", "blue": "bleu clair é"})
    for sep in (",", ";"):
        src = tmp_path / f"données{sep == ';'}.csv"
        src.write_bytes(df.to_csv(index=False, sep=sep).encode("cp1252"))
        lines = []
        b = train(src, "target", tmp_path / f"win{sep == ';'}", log=lines.append)
        assert "bleu clair é" in {x["name"]: x for x in b.features}["color"]["categories"]
        assert any("Windows-1252" in line for line in lines)
        assert any("';' between columns" in line for line in lines) == (sep == ";")


def test_reads_utf8_with_bom_and_tsv(tmp_path):
    df = mixed_frame(300)
    src = tmp_path / "bom.csv"
    src.write_bytes(df.to_csv(index=False).encode("utf-8-sig"))
    lines = []
    b = train(src, "target", tmp_path / "bom", log=lines.append)
    assert "size" in b.feature_names and not any("UTF-8" in line for line in lines)
    tsv = tmp_path / "data.tsv"
    df.to_csv(tsv, index=False, sep="\t")
    assert "size" in train(tsv, "target", tmp_path / "tsv").feature_names


def test_spreadsheet_files_need_csv(tmp_path):
    xlsx = tmp_path / "data.xlsx"
    xlsx.write_bytes(b"not really excel")
    with pytest.raises(TrainingError, match="Save the sheet as CSV"):
        train(xlsx, "y", tmp_path / "x")


def test_bad_coverage(tmp_path):
    with pytest.raises(TrainingError, match="coverage"):
        train(EXAMPLE_CSV, TARGET, tmp_path / "x", coverage=1.5)


def test_will_not_overwrite_other_folders(tmp_path):
    out = tmp_path / "busy"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    with pytest.raises(BundleError, match="not a Harbor bundle"):
        train(mixed_frame(100), "target", out)
    assert (out / "notes.txt").read_text() == "keep me"


@pytest.fixture()
def copy_of_bundle(bundle_dir, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(bundle_dir, target)
    return target


def test_changed_model_file_is_refused(copy_of_bundle):
    model = copy_of_bundle / MODEL_FILE
    model.write_bytes(model.read_bytes() + b"x")
    with pytest.raises(BundleError, match="checksum"):
        load_bundle(copy_of_bundle)


class NotAllowed:
    """A type that is not on Harbor's list of trusted types."""

    def __init__(self):
        self.value = 1


def test_untrusted_types_are_refused(copy_of_bundle):
    model = copy_of_bundle / MODEL_FILE
    sio.dump({"point": NotAllowed(), "low": None, "high": None, "range_offset": 0.0}, model)
    meta_path = copy_of_bundle / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["model_sha256"] = file_sha256(model)  # a careful attacker fixes the checksum too
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(BundleError, match="does not trust"):
        load_bundle(copy_of_bundle)


def test_model_file_that_cannot_be_loaded(copy_of_bundle, monkeypatch):
    import peakwhale_harbor.bundle as bundle_module

    def broken(*args, **kwargs):
        raise AttributeError("Can't get attribute 'Something' on <module 'sklearn...'>")

    monkeypatch.setattr(bundle_module.sio, "load", broken)
    with pytest.raises(BundleError, match=r"trained with scikit-learn .* train the bundle again"):
        load_bundle(copy_of_bundle)


def test_broken_model_file_is_a_bundle_error(copy_of_bundle):
    model = copy_of_bundle / MODEL_FILE
    model.write_bytes(b"not a zip file")
    meta_path = copy_of_bundle / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["model_sha256"] = file_sha256(model)
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(BundleError, match="cannot load"):
        load_bundle(copy_of_bundle)


def test_folder_that_is_not_a_bundle(tmp_path):
    with pytest.raises(BundleError, match="not found"):
        load_bundle(tmp_path / "nothing-here")
    (tmp_path / "schema.json").write_text("{}")
    with pytest.raises(BundleError, match="not a Harbor bundle"):
        load_bundle(tmp_path)


def test_newer_format_is_refused(copy_of_bundle):
    path = copy_of_bundle / "schema.json"
    schema = json.loads(path.read_text())
    schema["format_version"] = 99
    path.write_text(json.dumps(schema))
    with pytest.raises(BundleError, match="newer Harbor"):
        load_bundle(copy_of_bundle)


def test_other_sklearn_version_warns(copy_of_bundle):
    path = copy_of_bundle / "metadata.json"
    meta = json.loads(path.read_text())
    meta["versions"]["scikit-learn"] = "0.0.1"
    path.write_text(json.dumps(meta))
    with pytest.warns(VersionWarning, match="0.0.1"):
        load_bundle(copy_of_bundle)


# -- prediction ranges ---------------------------------------------------------


def test_range_contains_prediction_for_every_row(bundle):
    df = pd.read_csv(EXAMPLE_CSV)
    out = bundle.predict_frame(to_model_frame(df, bundle.features))
    assert np.all(out["low"] <= out["prediction"])
    assert np.all(out["prediction"] <= out["high"])
    assert np.all(out["low"] >= 0), "repair hours are never negative"


def test_range_contains_prediction_for_odd_inputs(bundle, good_record):
    odd = [
        {**good_record, "age_years": 500, "vibration_mm_s": -3},
        {name: None for name in bundle.feature_names},
        {**good_record, "machine_type": "PRESS", "days_since_service": 0},
    ]
    for result in bundle.predict(odd):
        assert result["low"] <= result["prediction"] <= result["high"]
        assert result["coverage"] == 0.9


class Fixed:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, frame):
        return self.values


def test_crossed_quantiles_still_contain_the_prediction():
    models = {
        "point": Fixed([5.0, -2.0, 10.0]),
        "low": Fixed([8.0, -5.0, 1.0]),  # first row: low above the prediction
        "high": Fixed([6.0, -1.0, 3.0]),  # third row: high below the prediction
        "range_offset": 0.5,
    }
    out = predict_with_range(models, pd.DataFrame(index=range(3)), non_negative=False)
    assert np.all(out["low"] <= out["prediction"]) and np.all(out["prediction"] <= out["high"])
    clipped = predict_with_range(models, pd.DataFrame(index=range(3)), non_negative=True)
    assert np.all(clipped["low"] >= 0)
    assert np.all(clipped["low"] <= clipped["prediction"]) and np.all(clipped["prediction"] <= clipped["high"])


def test_conformal_offset():
    y = np.arange(10, dtype=float)
    low = y - 1  # every true value is 1 inside the band
    high = y + 1
    assert conformal_offset(low, high, y, 0.8) == pytest.approx(-1.0)
    high[0] = -3  # one true value is 3 above the band
    assert conformal_offset(low, high, y, 0.95) == pytest.approx(3.0)


def test_conformal_offset_takes_the_right_order_statistic():
    # Scores 1..10. k = ceil(11 * 0.8) = 9, so the offset is the 9th smallest score.
    y = np.zeros(10)
    scores = np.arange(1, 11, dtype=float)
    low, high = -np.ones(10), -scores  # y - high = score, low - y = -1
    assert conformal_offset(low, high, y, 0.8) == pytest.approx(9.0)
    # k = ceil(11 * 0.95) = 11 is more than 10 scores: use the largest score.
    assert conformal_offset(low, high, y, 0.95) == pytest.approx(10.0)
    # 99 scores at 0.55: k = 55, although 100 * 0.55 is 55.00000000000001 in floats.
    y, scores = np.zeros(99), np.arange(1, 100, dtype=float)
    assert conformal_offset(-np.ones(99), -scores, y, 0.55) == pytest.approx(55.0)


def test_predictions_use_one_openmp_thread():
    """Many OpenMP threads make single predictions very slow on a busy computer."""
    from threadpoolctl import threadpool_info

    seen = []

    class Spy(Fixed):
        def predict(self, frame):
            seen.extend(i["num_threads"] for i in threadpool_info() if i["user_api"] == "openmp")
            return self.values

    models = {"point": Spy([1.0]), "low": Spy([0.5]), "high": Spy([2.0]), "range_offset": 0.0}
    predict_with_range(models, pd.DataFrame(index=range(1)), non_negative=False)
    assert seen and set(seen) == {1}
