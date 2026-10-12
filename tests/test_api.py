import io
import warnings

import pandas as pd
import pytest

from conftest import EXAMPLE_CSV, TARGET
from peakwhale_harbor.server import MAX_JSON_RECORDS, create_app


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["app"] == "PeakWhale Harbor"


def test_schema(client, bundle):
    r = client.get("/schema")
    assert r.status_code == 200
    body = r.get_json()
    assert body["target"]["name"] == TARGET
    assert [f["name"] for f in body["features"]] == bundle.feature_names
    assert body["input_json_schema"]["required"] == bundle.feature_names
    assert body["prediction_range"]["coverage"] == 0.9
    assert body["metrics"]["test_rows"] == 240
    # The example in the schema is a valid request.
    r = client.post("/predict_api", json=body["example_request"])
    assert r.status_code == 200


def test_predict_one(client, good_record):
    r = client.post("/predict_api", json={"data": good_record})
    assert r.status_code == 200
    body = r.get_json()
    assert body["low"] <= body["prediction"] <= body["high"]
    assert body["coverage"] == 0.9
    assert body["notes"] == []
    assert body["target"]["name"] == TARGET


def test_predict_many(client, good_record):
    records = [good_record, {**good_record, "machine_type": "press", "age_years": 18}]
    r = client.post("/predict_api", json={"data": records})
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["predictions"]) == 2
    for p in body["predictions"]:
        assert p["low"] <= p["prediction"] <= p["high"]


def test_unknown_values_are_allowed_with_a_note(client, good_record):
    r = client.post("/predict_api", json={"data": {**good_record, "vibration_mm_s": None}})
    assert r.status_code == 200
    assert any("vibration_mm_s is empty" in n for n in r.get_json()["notes"])


def test_extra_fields_are_ignored(client, good_record):
    r = client.post("/predict_api", json={"data": {**good_record, "asset_id": "EQ-1"}})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "payload, message",
    [
        (None, '"data"'),
        ({}, '"data"'),
        ({"data": "pump"}, "must be an object"),
        ({"data": 5}, "must be an object"),
    ],
)
def test_bad_request_shape(client, payload, message):
    r = client.post("/predict_api", json=payload) if payload is not None else client.post("/predict_api", data="x")
    assert r.status_code == 400
    assert message in r.get_json()["error"]


def test_missing_field(client, good_record):
    record = dict(good_record)
    del record["site"]
    r = client.post("/predict_api", json={"data": record})
    assert r.status_code == 400
    body = r.get_json()
    assert body["problems"][0]["fields"] == ["site"]
    assert "site" in body["required"]


def test_not_a_number(client, good_record):
    r = client.post("/predict_api", json={"data": {**good_record, "age_years": "old"}})
    assert r.status_code == 400
    problem = r.get_json()["problems"][0]
    assert problem["field"] == "age_years"
    assert "expected a number" in problem["error"]


def test_unknown_category(client, good_record):
    r = client.post("/predict_api", json={"data": {**good_record, "machine_type": "boat"}})
    assert r.status_code == 400
    problem = r.get_json()["problems"][0]
    assert problem["field"] == "machine_type"
    assert "'pump'" in problem["error"]


def test_too_many_records(client, good_record):
    r = client.post("/predict_api", json={"data": [good_record] * (MAX_JSON_RECORDS + 1)})
    assert r.status_code == 400
    assert "Too many" in r.get_json()["error"]


def test_home_page_has_generated_form(client, bundle):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    for name in bundle.feature_names:
        assert f'name="{name}"' in html
    assert "<select" in html and 'value="compressor"' in html
    assert "How good is this model?" in html
    assert "not financial or appraisal advice" in html.lower()


def test_form_shows_prediction_and_range(client, good_record):
    r = client.post("/predict", data={k: str(v) for k, v in good_record.items()})
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Likely between" in html
    assert "90% range" in html


def test_form_shows_errors(client, good_record):
    r = client.post("/predict", data={**{k: str(v) for k, v in good_record.items()}, "age_years": "abc"})
    assert r.status_code == 400
    assert "expected a number" in r.get_data(as_text=True)


def _upload(client, df, ui=False):
    buffer = io.BytesIO(df.to_csv(index=False).encode())
    data = {"file": (buffer, "machines.csv")}
    if ui:
        data["ui"] = "1"
    return client.post("/predict_batch", data=data, content_type="multipart/form-data")


def test_batch_csv(client, bundle):
    df = pd.read_csv(EXAMPLE_CSV).head(50)
    r = _upload(client, df)
    assert r.status_code == 200
    assert r.mimetype == "text/csv"
    assert "machines_predictions.csv" in r.headers["Content-Disposition"]
    out = pd.read_csv(io.BytesIO(r.data))
    assert len(out) == 50
    assert list(out.columns[: df.shape[1]]) == list(df.columns), "input columns are kept"
    pred, low, high = (out[f"{TARGET}_{k}"] for k in ("prediction", "low", "high"))
    assert (low <= pred).all() and (pred <= high).all()


def test_batch_reports_csv_line_numbers(client):
    df = pd.read_csv(EXAMPLE_CSV).head(5)
    df["age_years"] = df["age_years"].astype(object)
    df.loc[1, "age_years"] = "very old"
    r = _upload(client, df)
    assert r.status_code == 400
    problem = r.get_json()["problems"][0]
    assert problem == {"line": 3, "field": "age_years", "error": "expected a number, got 'very old'"}


def test_batch_missing_columns(client):
    df = pd.read_csv(EXAMPLE_CSV).head(5).drop(columns=["site"])
    r = _upload(client, df)
    assert r.status_code == 400
    assert r.get_json()["problems"][0]["fields"] == ["site"]


def test_batch_errors_in_the_page(client):
    df = pd.read_csv(EXAMPLE_CSV).head(5).drop(columns=["site"])
    r = _upload(client, df, ui=True)
    assert r.status_code == 400
    assert "missing columns" in r.get_data(as_text=True)


def test_batch_without_file(client):
    r = client.post("/predict_batch", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_template_csv_round_trip(client, bundle):
    r = client.get("/template.csv")
    assert r.status_code == 200
    template = pd.read_csv(io.BytesIO(r.data))
    assert list(template.columns) == bundle.feature_names
    assert _upload(client, template).status_code == 200


def test_upload_size_limit(bundle, monkeypatch):
    monkeypatch.setenv("HARBOR_MAX_UPLOAD_MB", "2")
    app = create_app(bundle)
    assert app.config["MAX_CONTENT_LENGTH"] == 2 * 1024 * 1024
    app.config["MAX_CONTENT_LENGTH"] = 1000  # keep the test upload small
    big = io.BytesIO(b"a,b\n" + b"1,2\n" * 1000)
    r = app.test_client().post(
        "/predict_batch", data={"file": (big, "big.csv")}, content_type="multipart/form-data"
    )
    assert r.status_code == 413
    assert "too large" in r.get_json()["error"]


def test_app_reads_bundle_from_environment(bundle_dir, monkeypatch):
    monkeypatch.setenv("HARBOR_BUNDLE", str(bundle_dir))
    assert create_app().test_client().get("/health").status_code == 200


def test_no_feature_name_warnings(client, good_record):
    """Old Harbor warned "X does not have valid feature names" on every prediction."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.post("/predict_api", json={"data": good_record})
        client.post("/predict", data={k: str(v) for k, v in good_record.items()})
    assert not [w for w in caught if "feature names" in str(w.message)]
