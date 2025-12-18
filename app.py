import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request

APP_NAME = "PeakWhale Harbor"
APP_SUBTITLE = "Housing value prediction demo"
APP_DESCRIPTION = (
    "This demo predicts the Boston housing target MEDV, median home value in $1000s, "
    "using the classic feature set."
)

TARGET_NAME = "MEDV"
TARGET_DESCRIPTION = "Median home value"
TARGET_UNITS = "$1000s"

FEATURE_SPECS = [
    {"name": "CRIM", "label": "CRIM", "description": "Per capita crime rate by town", "placeholder": "0.10", "step": "any"},
    {"name": "ZN", "label": "ZN", "description": "Proportion of residential land zoned for lots over 25,000 sq ft", "placeholder": "0.0", "step": "any"},
    {"name": "INDUS", "label": "INDUS", "description": "Proportion of non retail business acres per town", "placeholder": "8.0", "step": "any"},
    {"name": "CHAS", "label": "CHAS", "description": "Charles River dummy variable (1 if tract bounds river, else 0)", "placeholder": "0", "step": "1"},
    {"name": "NOX", "label": "NOX", "description": "Nitric oxides concentration (parts per 10 million)", "placeholder": "0.50", "step": "any"},
    {"name": "RM", "label": "RM", "description": "Average number of rooms per dwelling", "placeholder": "6.0", "step": "any"},
    {"name": "AGE", "label": "AGE", "description": "Proportion of owner occupied units built prior to 1940", "placeholder": "65.0", "step": "any"},
    {"name": "DIS", "label": "DIS", "description": "Weighted distances to five Boston employment centres", "placeholder": "4.0", "step": "any"},
    {"name": "RAD", "label": "RAD", "description": "Index of accessibility to radial highways", "placeholder": "4", "step": "1"},
    {"name": "TAX", "label": "TAX", "description": "Full value property tax rate per 10,000 dollars", "placeholder": "300.0", "step": "any"},
    {"name": "PTRATIO", "label": "PTRATIO", "description": "Pupil teacher ratio by town", "placeholder": "18.0", "step": "any"},
    {"name": "B", "label": "B", "description": "1000(Bk minus 0.63)^2 where Bk is the proportion of Black residents by town", "placeholder": "390.0", "step": "any"},
    {"name": "LSTAT", "label": "LSTAT", "description": "Percent lower status of the population", "placeholder": "12.0", "step": "any"},
]

FEATURE_ORDER = [f["name"] for f in FEATURE_SPECS]

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "regmodel.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaling.pkl"

app = Flask(__name__)


def load_artifacts() -> Tuple[Any, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact at {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler artifact at {SCALER_PATH}")

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)

    with SCALER_PATH.open("rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_artifacts()


def coerce_feature_dict_to_row(data: Dict[str, Any]) -> np.ndarray:
    missing = [k for k in FEATURE_ORDER if k not in data]
    if missing:
        raise KeyError(f"Missing required features: {missing}")

    values = []
    for k in FEATURE_ORDER:
        v = data[k]
        values.append(float(v))

    return np.array(values, dtype=float).reshape(1, -1)


def predict_value(x: np.ndarray) -> float:
    x_scaled = scaler.transform(x)
    return float(model.predict(x_scaled)[0])


@app.get("/health")
def health():
    return jsonify(status="ok", app=APP_NAME)


@app.get("/schema")
def schema():
    return jsonify(
        app=APP_NAME,
        subtitle=APP_SUBTITLE,
        description=APP_DESCRIPTION,
        target={"name": TARGET_NAME, "description": TARGET_DESCRIPTION, "units": TARGET_UNITS},
        required_features=FEATURE_SPECS,
        request_format={"data": {k: 0 for k in FEATURE_ORDER}},
        response_format={"prediction": 0.0, "target": {"name": TARGET_NAME, "units": TARGET_UNITS}},
    )


@app.get("/")
def home():
    return render_template(
        "home.html",
        app_name=APP_NAME,
        subtitle=APP_SUBTITLE,
        description=APP_DESCRIPTION,
        target_name=TARGET_NAME,
        target_units=TARGET_UNITS,
        feature_specs=FEATURE_SPECS,
    )


@app.post("/predict_api")
def predict_api():
    payload = request.get_json(silent=True) or {}
    data = payload.get("data")

    if not isinstance(data, dict):
        return jsonify(error="Expected JSON body with key data as an object."), 400

    missing = [k for k in FEATURE_ORDER if k not in data]
    if missing:
        return jsonify(error="Missing required features.", missing=missing, required=FEATURE_ORDER), 400

    try:
        x = coerce_feature_dict_to_row(data)
        y = predict_value(x)
        return jsonify(
            prediction=y,
            target={"name": TARGET_NAME, "description": TARGET_DESCRIPTION, "units": TARGET_UNITS},
            received={k: float(data[k]) for k in FEATURE_ORDER},
        )
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/predict")
def predict():
    try:
        form_data: Dict[str, Any] = {}
        for k in FEATURE_ORDER:
            v = request.form.get(k, "")
            if str(v).strip() == "":
                raise ValueError(f"Missing form field: {k}")
            form_data[k] = float(v)

        x = coerce_feature_dict_to_row(form_data)
        y = predict_value(x)

        return render_template(
            "home.html",
            app_name=APP_NAME,
            subtitle=APP_SUBTITLE,
            description=APP_DESCRIPTION,
            target_name=TARGET_NAME,
            target_units=TARGET_UNITS,
            feature_specs=FEATURE_SPECS,
            prediction_text=f"Predicted {TARGET_NAME}: {y:.2f} {TARGET_UNITS}",
        )
    except Exception as e:
        return render_template(
            "home.html",
            app_name=APP_NAME,
            subtitle=APP_SUBTITLE,
            description=APP_DESCRIPTION,
            target_name=TARGET_NAME,
            target_units=TARGET_UNITS,
            feature_specs=FEATURE_SPECS,
            prediction_text=f"Input error: {e}",
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5050"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
