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


def test_unknown_target_suggests_a_column(tmp_path):
    with pytest.raises(TrainingError, match="Did you mean 'repair_hours'"):
        train(EXAMPLE_CSV, "repair_hour", tmp_path / "x")


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
