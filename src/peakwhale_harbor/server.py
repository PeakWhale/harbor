"""Flask web app: a form, a JSON API and CSV batch predictions for one bundle.

Run it with ``harbor serve <bundle>`` for local use, or with Gunicorn::

    HARBOR_BUNDLE=models/california-housing gunicorn "peakwhale_harbor.server:create_app()"
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

from . import __version__
from .bundle import Bundle, load_bundle, read_csv_for_prediction
from .formatting import format_number, format_percent
from .schema import InputError, example_value

APP_NAME = "PeakWhale Harbor"
MAX_JSON_RECORDS = 1000
MAX_BATCH_ROWS = 100_000
#: Uploads with more columns are refused before they are read (plus room for the model's own inputs).
MAX_BATCH_COLUMNS = 1000


def _csv_response(df: pd.DataFrame, filename: str) -> Response:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _upload_limit_mb() -> float:
    text = os.environ.get("HARBOR_MAX_UPLOAD_MB", "10")
    try:
        mb = float(text)
    except ValueError:
        mb = math.nan
    if not (math.isfinite(mb) and mb > 0):
        raise RuntimeError(f"HARBOR_MAX_UPLOAD_MB must be a number of megabytes, such as 10 (got {text!r}).")
    return mb


def create_app(bundle: Bundle | str | Path | None = None) -> Flask:
    """Build the app. ``bundle`` is a Bundle or a folder path.

    Without an argument the folder comes from the HARBOR_BUNDLE environment
    variable.
    """
    if bundle is None:
        bundle = os.environ.get("HARBOR_BUNDLE")
        if not bundle:
            raise RuntimeError("Set HARBOR_BUNDLE to a bundle folder, or pass one to create_app().")
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)

    upload_mb = _upload_limit_mb()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(upload_mb * 1024 * 1024)
    app.json.sort_keys = False
    app.extensions["harbor_bundle"] = bundle
    app.jinja_env.filters["num"] = format_number
    app.jinja_env.filters["pct"] = format_percent

    target = bundle.target
    target_info = {"name": target["name"], "description": target.get("description", "")}
    typical = {f["name"]: example_value(f) for f in bundle.features}
    max_columns = max(MAX_BATCH_COLUMNS, len(bundle.features) + 100)

    def range_bar(result: dict[str, Any]) -> dict[str, float]:
        """Positions (0 to 100) for the small range picture in the UI."""
        lo, hi = min(target["min"], result["low"]), max(target["max"], result["high"])
        span = (hi - lo) or 1.0

        def pos(x: float) -> float:
            return round(max(0.0, min(100.0, (x - lo) / span * 100)), 2)

        return {"left": pos(result["low"]), "right": pos(result["high"]), "point": pos(result["prediction"])}

    def page(values: dict[str, Any] | None = None, status: int = 200, **extra: Any):
        html = render_template(
            "home.html",
            app_name=APP_NAME,
            version=__version__,
            bundle=bundle,
            target=target,
            features=bundle.features,
            values=values if values is not None else typical,
            typical=typical,
            metrics=bundle.metrics,
            coverage=bundle.coverage,
            **extra,
        )
        return html, status

    # -- pages -------------------------------------------------------------

    @app.get("/")
    def home():
        return page()

    @app.post("/predict")
    def predict_form():
        values = {f["name"]: request.form.get(f["name"], "") for f in bundle.features}
        try:
            result = bundle.predict_one(values)
        except InputError as exc:
            return page(values, 400, error=exc.message, problems=exc.problems)
        return page(values, result=result, bar=range_bar(result))

    # -- API ---------------------------------------------------------------

    @app.get("/health")
    def health():
        return jsonify(status="ok", app=APP_NAME, version=__version__, model=bundle.title)

    @app.get("/schema")
    def schema():
        return jsonify(
            app=APP_NAME,
            version=__version__,
            **bundle.describe(),
            endpoints={
                "GET /health": "health check",
                "GET /schema": "this document",
                "POST /predict_api": 'JSON body {"data": {...}} for one record or {"data": [{...}, ...]} for many',
                "POST /predict_batch": "multipart form upload with a CSV file in the field 'file'; returns a CSV",
                "GET /template.csv": "an example CSV with the right columns",
            },
        )

    @app.post("/predict_api")
    def predict_api():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "data" not in payload:
            return jsonify(
                error='Send a JSON body with the key "data".', example={"data": bundle.example()}
            ), 400
        data = payload["data"]
        try:
            if isinstance(data, dict):
                return jsonify(**bundle.predict_one(data), target=target_info)
            if isinstance(data, list):
                if len(data) > MAX_JSON_RECORDS:
                    return jsonify(error=f"Too many records. The limit is {MAX_JSON_RECORDS} per request."), 400
                return jsonify(predictions=bundle.predict(data), coverage=bundle.coverage, target=target_info)
        except InputError as exc:
            return jsonify(**exc.to_dict(), required=bundle.feature_names), 400
        return jsonify(error='"data" must be an object (one record) or a list of objects.'), 400

    @app.post("/predict_batch")
    def predict_batch():
        from_ui = request.form.get("ui") == "1"

        def fail(message: str, problems: list[dict[str, Any]] | None = None):
            if from_ui:
                return page(status=400, batch_error=message, batch_problems=problems or [])
            return jsonify(error=message, problems=problems or []), 400

        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return fail("Upload a CSV file in the form field 'file'.")
        try:
            df = read_csv_for_prediction(upload.stream, name=upload.filename, max_columns=max_columns)
        except Exception as exc:  # pandas raises many error types
            return fail(f"Could not read the file as CSV: {exc}")
        if df.empty:
            return fail("The CSV file has no rows.")
        if len(df) > MAX_BATCH_ROWS:
            return fail(f"Too many rows ({len(df)}). The limit is {MAX_BATCH_ROWS}.")
        try:
            out = bundle.predict_table(df)
        except InputError as exc:
            return fail(exc.message, exc.problems)
        # ASCII only: HTTP headers cannot hold other characters (a file named "données.csv" gives "donnes").
        stem = "".join(c for c in Path(upload.filename).stem if c.isascii() and (c.isalnum() or c in "-_")) or "data"
        return _csv_response(out, f"{stem}_predictions.csv")

    @app.get("/template.csv")
    def template_csv():
        return _csv_response(pd.DataFrame([bundle.example()], columns=bundle.feature_names), "template.csv")

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify(error=f"The upload is too large. The limit is {upload_mb:g} MB."), 413

    return app
