import os
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

APP_NAME = "PeakWhale Harbor"

FEATURE_ORDER = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "regmodel.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaling.pkl"

app = Flask(__name__)

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

with SCALER_PATH.open("rb") as f:
    scaler = pickle.load(f)


@app.get("/health")
def health():
    return jsonify(status="ok", app=APP_NAME)


@app.get("/")
def home():
    return render_template("home.html", app_name=APP_NAME)


@app.post("/predict_api")
def predict_api():
    payload = request.get_json(silent=True) or {}
    data = payload.get("data")

    if not isinstance(data, dict):
        return jsonify(error="Expected JSON body with key 'data' as an object."), 400

    missing = [k for k in FEATURE_ORDER if k not in data]
    if missing:
        return jsonify(error="Missing required features.", missing=missing), 400

    try:
        x = np.array([float(data[k]) for k in FEATURE_ORDER], dtype=float).reshape(1, -1)
        x_scaled = scaler.transform(x)
        y = float(model.predict(x_scaled)[0])
        return jsonify(prediction=y)
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/predict")
def predict():
    try:
        values = [float(request.form[k]) for k in FEATURE_ORDER]
        x = np.array(values, dtype=float).reshape(1, -1)
        x_scaled = scaler.transform(x)
        y = float(model.predict(x_scaled)[0])
        return render_template(
            "home.html",
            app_name=APP_NAME,
            prediction_text=f"Harbor prediction: {y}",
        )
    except Exception as e:
        return render_template(
            "home.html",
            app_name=APP_NAME,
            prediction_text=f"Input error: {e}",
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
