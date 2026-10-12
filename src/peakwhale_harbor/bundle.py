"""Model bundles: one folder with everything needed to serve a model.

::

    my-model/
      model.skops     the fitted models, in the skops format (not pickle)
      schema.json     input fields, target and prediction range settings
      metrics.json    results on held-out test rows
      metadata.json   how, when and with which versions the model was trained

Security: skops is safer than pickle. It refuses to load object types it does
not know, and Harbor only allows a short, fixed list of extra types. Still,
treat a model file like a program: only load bundles that you trained yourself
or that come from someone you trust.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import skops.io as sio
from threadpoolctl import ThreadpoolController

from .schema import InputError, check_records, example_record

BUNDLE_FORMAT = "harbor-bundle"
FORMAT_VERSION = 1
MODEL_FILE = "model.skops"
SCHEMA_FILE = "schema.json"
METRICS_FILE = "metrics.json"
METADATA_FILE = "metadata.json"

#: Types inside a Harbor model file that skops does not trust by default.
#: Harbor allows exactly these and refuses any other unknown type. They are
#: parts of scikit-learn's own gradient boosting and column transformer; which
#: of them skops reports depends on the skops and scikit-learn versions.
EXTRA_TRUSTED_TYPES = frozenset(
    {
        "functools.partial",
        "sklearn.utils.validation.check_array",
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
        "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
        "sklearn.compose._column_transformer._RemainderColsList",
        "sklearn._loss.link.IdentityLink",
        "sklearn._loss.link.Interval",
        "sklearn._loss.loss.HalfSquaredError",
        "sklearn._loss.loss.PinballLoss",
        "_loss.CyHalfSquaredError",
        "_loss.CyPinballLoss",
    }
)


class BundleError(Exception):
    """A bundle folder is missing, incomplete or not safe to load."""


class VersionWarning(UserWarning):
    """The bundle was trained with a different scikit-learn version."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    # allow_nan=False: NaN and Infinity are not valid JSON, so fail here instead of writing them.
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BundleError(f"{path.name} is missing in {path.parent}") from None
    except json.JSONDecodeError as exc:
        raise BundleError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise BundleError(f"{path} must contain a JSON object")
    return data


def is_bundle(path: Path) -> bool:
    try:
        return _read_json(Path(path) / SCHEMA_FILE).get("format") == BUNDLE_FORMAT
    except BundleError:
        return False


@dataclass
class Bundle:
    """A loaded model bundle. Use :func:`load_bundle` to create one."""

    path: Path
    schema: dict[str, Any]
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    models: dict[str, Any] = field(repr=False)

    # -- information -------------------------------------------------------

    @property
    def title(self) -> str:
        return self.schema.get("title") or self.path.name

    @property
    def features(self) -> list[dict[str, Any]]:
        return self.schema["features"]

    @property
    def feature_names(self) -> list[str]:
        return [f["name"] for f in self.features]

    @property
    def target(self) -> dict[str, Any]:
        return self.schema["target"]

    @property
    def coverage(self) -> float:
        return float(self.schema["prediction_range"]["coverage"])

    def example(self) -> dict[str, Any]:
        """A typical input record (median or most common value per field)."""
        return example_record(self.features)

    def describe(self) -> dict[str, Any]:
        """Everything a client needs to know to use the model, as JSON."""
        return {
            "title": self.title,
            "description": self.schema.get("description", ""),
            "task": self.schema["task"],
            "target": self.target,
            "features": self.features,
            "dropped_columns": self.schema.get("dropped_columns", []),
            "prediction_range": self.schema["prediction_range"],
            "metrics": self.metrics,
            "trained": {
                "created_at": self.metadata.get("created_at"),
                "source_file": self.metadata.get("source", {}).get("file"),
                "rows_used": self.metadata.get("rows", {}).get("used"),
                "harbor_version": self.metadata.get("harbor_version"),
            },
            "input_json_schema": self.schema["input"],
            "output_json_schema": self.schema["output"],
            "example_request": {"data": self.example()},
        }

    # -- predictions -------------------------------------------------------

    def predict_frame(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict for a table that already has the right columns and types.

        Returns arrays ``prediction``, ``low`` and ``high``. The range always
        contains the prediction: ``low <= prediction <= high``.
        """
        return predict_with_range(self.models, frame[self.feature_names], self.target.get("non_negative", False))

    def predict(
        self, records: Sequence[Mapping[str, Any]], *, row_key: str = "row", first_row: int = 0
    ) -> list[dict[str, Any]]:
        """Check and predict a list of records (dicts). Raises InputError."""
        if len(records) == 0:
            return []
        frame, notes = check_records(records, self.features, row_key=row_key, first_row=first_row)
        out = self.predict_frame(frame)
        return [
            {
                "prediction": float(out["prediction"][i]),
                "low": float(out["low"][i]),
                "high": float(out["high"][i]),
                "coverage": self.coverage,
                "notes": notes[i],
            }
            for i in range(len(frame))
        ]

    def predict_one(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self.predict([record])[0]

    def predict_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict for a table read from a CSV file (CSV line numbers in errors).

        Returns a copy of ``df`` with ``<target>_prediction``, ``<target>_low``,
        ``<target>_high`` and ``notes`` columns added.
        """
        absent = [n for n in self.feature_names if n not in df.columns]
        if absent:
            raise InputError("The CSV file is missing columns.", [{"error": "missing columns", "fields": absent}])
        results = self.predict(df.to_dict("records"), row_key="line", first_row=2)
        name = self.target["name"]
        out = df.copy()
        out[f"{name}_prediction"] = [r["prediction"] for r in results]
        out[f"{name}_low"] = [r["low"] for r in results]
        out[f"{name}_high"] = [r["high"] for r in results]
        out["notes"] = ["; ".join(r["notes"]) for r in results]
        return out


def read_csv_for_prediction(source: Any) -> pd.DataFrame:
    """Read a CSV for prediction. Every cell stays text until it is checked."""
    df = pd.read_csv(source, dtype=str, keep_default_na=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


_threadpools: ThreadpoolController | None = None


def _one_thread():
    """Run model predictions on one OpenMP thread.

    By default scikit-learn uses one thread per CPU even for a single row. On a
    busy computer, or with several server workers, those threads wait for each
    other and a prediction can take seconds instead of milliseconds.
    """
    global _threadpools
    if _threadpools is None:
        _threadpools = ThreadpoolController()
    return _threadpools.limit(limits=1, user_api="openmp")


def predict_with_range(models: Mapping[str, Any], frame: pd.DataFrame, non_negative: bool) -> dict[str, np.ndarray]:
    with _one_thread():
        point = np.asarray(models["point"].predict(frame), dtype=float)
        offset = float(models["range_offset"])
        low = np.asarray(models["low"].predict(frame), dtype=float) - offset
        high = np.asarray(models["high"].predict(frame), dtype=float) + offset
    # Quantile models can cross. Sort the two ends.
    low, high = np.minimum(low, high), np.maximum(low, high)
    if non_negative:
        point, low, high = (np.maximum(a, 0.0) for a in (point, low, high))
    # The range must always contain the single best guess.
    low = np.minimum(low, point)
    high = np.maximum(high, point)
    return {"prediction": point, "low": low, "high": high}


def save_bundle(
    path: Path,
    *,
    models: Mapping[str, Any],
    schema: Mapping[str, Any],
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not is_bundle(path):
        raise BundleError(f"{path} already exists and is not a Harbor bundle. Choose another --out folder.")
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / MODEL_FILE
    sio.dump(dict(models), model_path, compression=8)
    metadata = {**metadata, "model_sha256": file_sha256(model_path)}
    _write_json(path / SCHEMA_FILE, dict(schema))
    _write_json(path / METRICS_FILE, dict(metrics))
    _write_json(path / METADATA_FILE, metadata)
    return path


def load_bundle(path: str | Path) -> Bundle:
    """Load a bundle folder. Only load bundles you trust (see module notes)."""
    path = Path(path)
    if not path.is_dir():
        raise BundleError(f"bundle folder not found: {path}")
    schema = _read_json(path / SCHEMA_FILE)
    if schema.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"{path} is not a Harbor bundle (schema.json has no 'format': '{BUNDLE_FORMAT}')")
    if int(schema.get("format_version", 0)) > FORMAT_VERSION:
        raise BundleError(f"{path} was made by a newer Harbor. Please upgrade Harbor.")
    metrics = _read_json(path / METRICS_FILE)
    metadata = _read_json(path / METADATA_FILE)

    model_path = path / MODEL_FILE
    if not model_path.is_file():
        raise BundleError(f"{MODEL_FILE} is missing in {path}")
    expected = metadata.get("model_sha256")
    if expected and file_sha256(model_path) != expected:
        raise BundleError(f"{model_path} does not match the checksum in {METADATA_FILE}. The file was changed.")

    try:
        untrusted = sio.get_untrusted_types(file=model_path)
    except Exception as exc:  # broken zip file and similar
        raise BundleError(f"cannot read {model_path}: {exc}") from None
    unexpected = sorted(set(untrusted) - EXTRA_TRUSTED_TYPES)
    if unexpected:
        raise BundleError(f"refusing to load {model_path}: it contains types Harbor does not trust: {unexpected}")

    trained_with = metadata.get("versions", {}).get("scikit-learn")
    if trained_with and trained_with != sklearn.__version__:
        warnings.warn(
            f"{path.name} was trained with scikit-learn {trained_with}, but {sklearn.__version__} is installed. "
            "Predictions may differ or fail. Retrain the bundle to be safe.",
            VersionWarning,
            stacklevel=2,
        )
    models = sio.load(model_path, trusted=list(untrusted))
    missing = {"point", "low", "high", "range_offset"} - set(models)
    if missing:
        raise BundleError(f"{model_path} is incomplete, missing: {sorted(missing)}")
    return Bundle(path=path, schema=schema, metrics=metrics, metadata=metadata, models=models)
