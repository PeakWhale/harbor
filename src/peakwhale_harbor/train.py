"""Train a model bundle from a CSV file.

How it works, in short:

1. Read the file and find the target column (the number to predict).
2. Guess the type of every other column (numeric or categorical).
3. Split the rows: 60% train, 20% calibration, 20% test (by default).
4. Train three gradient boosting models on the train rows: one for the best
   guess and two quantile models for the low and high end of the range.
5. Use the calibration rows to widen or narrow the range so that about 90% of
   true values fall inside it (conformalized quantile regression).
6. Measure everything on the test rows, which the models never saw.
7. Save the bundle folder.
"""

from __future__ import annotations

import math
import platform
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import skops
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from . import __version__
from .bundle import BUNDLE_FORMAT, FORMAT_VERSION, Bundle, file_sha256, load_bundle, predict_with_range, save_bundle
from .schema import (
    NUMERIC_SHARE,
    infer_features,
    input_json_schema,
    numeric_series,
    output_json_schema,
    to_model_frame,
)

DEFAULT_COVERAGE = 0.9
DEFAULT_SEED = 42
MIN_ROWS = 50


class TrainingError(Exception):
    """Training cannot start or finish. The message says what to fix."""


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV (or .csv.gz / .tsv) file into a table."""
    path = Path(path)
    if not path.is_file():
        raise TrainingError(f"file not found: {path}")
    name = path.name.lower()
    if name.endswith((".xlsx", ".xls", ".ods", ".numbers")):
        raise TrainingError("Harbor reads CSV files. Save the sheet as CSV first (File > Save As > CSV).")
    sep = "\t" if name.endswith((".tsv", ".tsv.gz")) else ","
    try:
        df = pd.read_csv(path, sep=sep)
    except Exception as exc:  # pandas raises many error types
        raise TrainingError(f"could not read {path.name} as CSV: {exc}") from None
    df.columns = [str(c).strip() for c in df.columns]
    duplicated = sorted({c for c in df.columns if list(df.columns).count(c) > 1})
    if duplicated:
        raise TrainingError(f"column names must be unique, these repeat: {duplicated}")
    return df


def _title_from(name: str) -> str:
    return " ".join(part.capitalize() for part in name.replace("-", " ").replace("_", " ").split())


def _make_pipeline(features: list[dict[str, Any]], *, loss: str, quantile: float | None, seed: int, n_train: int):
    categorical = [f for f in features if f["type"] == "categorical"]
    numeric = [f["name"] for f in features if f["type"] == "numeric"]
    steps = []
    if categorical:
        encoder = OrdinalEncoder(
            categories=[f["categories"] for f in categorical],
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
            dtype=float,
        )
        steps.append(("categorical", encoder, [f["name"] for f in categorical]))
    if numeric:
        steps.append(("numeric", "passthrough", numeric))
    prepare = ColumnTransformer(steps, verbose_feature_names_out=False)

    early_stopping = n_train >= 1000
    params: dict[str, Any] = {
        "loss": loss,
        "learning_rate": 0.1,
        "max_iter": 300 if early_stopping else 100,
        "early_stopping": early_stopping,
        "n_iter_no_change": 20,
        "validation_fraction": 0.1,
        # The encoder puts categorical columns first.
        "categorical_features": list(range(len(categorical))) or None,
        "random_state": seed,
    }
    if quantile is not None:
        params["quantile"] = quantile
    return Pipeline([("prepare", prepare), ("model", HistGradientBoostingRegressor(**params))])


def conformal_offset(low: np.ndarray, high: np.ndarray, y: np.ndarray, coverage: float) -> float:
    """Conformalized quantile regression (Romano, Patterson and Candes, 2019).

    Each calibration row gets a score: how far the true value is outside the
    quantile range (negative when inside). The returned offset is a high
    quantile of the scores. Adding it to both ends gives a range that holds
    the true value for about ``coverage`` of new rows, if new rows look like
    the training data.
    """
    scores = np.sort(np.maximum(low - y, y - high))
    n = len(scores)
    # The k-th smallest score, with k = ceil((n + 1) * coverage). The small
    # tolerance keeps float noise such as 9.000000000000002 from adding one.
    k = int(np.ceil((n + 1) * coverage - 1e-9))
    if k > n:
        return float(scores[-1])
    return float(scores[k - 1])


def regression_metrics(y: np.ndarray, out: Mapping[str, np.ndarray], train_median: float) -> dict[str, float]:
    inside = (y >= out["low"]) & (y <= out["high"])
    width = out["high"] - out["low"]
    return {
        "mae": float(mean_absolute_error(y, out["prediction"])),
        "rmse": float(root_mean_squared_error(y, out["prediction"])),
        "r2": float(r2_score(y, out["prediction"])),
        "baseline_mae": float(mean_absolute_error(y, np.full_like(y, train_median))),
        "range_coverage": float(inside.mean()),
        "range_mean_width": float(width.mean()),
        "range_median_width": float(np.median(width)),
    }


def train(
    data: str | Path | pd.DataFrame,
    target: str,
    out: str | Path,
    *,
    coverage: float = DEFAULT_COVERAGE,
    test_size: float = 0.2,
    calibration_size: float = 0.2,
    seed: int = DEFAULT_SEED,
    drop: Iterable[str] = (),
    categorical: Iterable[str] = (),
    numeric: Iterable[str] = (),
    title: str | None = None,
    description: str = "",
    descriptions: Mapping[str, str] | None = None,
    log: Callable[[str], None] | None = None,
) -> Bundle:
    """Train on a CSV file (or a pandas DataFrame) and save a bundle to ``out``.

    Returns the saved bundle, loaded back from disk.
    """
    say = log or (lambda _msg: None)
    if not 0.5 <= coverage < 1:
        raise TrainingError("coverage must be at least 0.5 and below 1 (for example 0.9 for a 90% range)")
    if not (0 < test_size < 0.5 and 0 < calibration_size < 0.5 and test_size + calibration_size <= 0.6):
        raise TrainingError("test and calibration sizes must each be between 0 and 0.5, and together at most 0.6")

    source: dict[str, Any] = {"file": None, "sha256": None}
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df.columns = [str(c) for c in df.columns]
    else:
        path = Path(data)
        df = read_table(path)
        source = {"file": path.name, "sha256": file_sha256(path)}
        if title is None:
            title = _title_from(path.name.split(".")[0])
    source.update({"rows": int(len(df)), "columns": int(df.shape[1])})
    title = title or _title_from(str(target)) + " model"
    descriptions = dict(descriptions or {})
    drop, categorical, numeric = list(drop), list(categorical), list(numeric)
    options = {
        "target": target,
        "coverage": coverage,
        "seed": seed,
        "test_size": test_size,
        "calibration_size": calibration_size,
        "drop": drop,
        "categorical": categorical,
        "numeric": numeric,
    }

    if target not in df.columns:
        hint = get_close_matches(target, list(df.columns), n=1)
        extra = f" Did you mean {hint[0]!r}?" if hint else ""
        raise TrainingError(f"target column {target!r} is not in the file.{extra} Columns: {list(df.columns)}")

    y_all = numeric_series(df[target])
    present = df[target].notna().sum()
    if present == 0 or y_all.notna().sum() < NUMERIC_SHARE * present:
        raise TrainingError(
            f"the target column {target!r} must contain numbers. Harbor predicts numbers (regression). "
            "Yes/no or category targets are not supported yet."
        )
    keep = y_all.notna()
    dropped_rows = int((~keep).sum())
    df, y_all = df[keep].reset_index(drop=True), y_all[keep].reset_index(drop=True)
    if dropped_rows:
        say(f"Skipped {dropped_rows} rows with an empty or non-numeric {target}.")
    if len(df) < MIN_ROWS:
        raise TrainingError(f"need at least {MIN_ROWS} rows with a {target} value, found {len(df)}")

    try:
        features, dropped_columns = infer_features(
            df, target, drop=drop, categorical=categorical, numeric=numeric, descriptions=descriptions
        )
    except ValueError as exc:
        raise TrainingError(str(exc)) from None
    if not features:
        skipped = "; ".join(f"{d['name']} ({d['reason']})" for d in dropped_columns) or "none"
        raise TrainingError(f"no usable input columns are left. Skipped columns: {skipped}")
    for f in features:
        say(f"  input  {f['name']:<28} {f['type']}")
    for d in dropped_columns:
        say(f"  skip   {d['name']:<28} {d['reason']}")

    X = to_model_frame(df, features)
    y = y_all.to_numpy(dtype=float)
    idx = np.arange(len(df))
    rest, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)
    train_idx, cal_idx = train_test_split(rest, test_size=calibration_size / (1 - test_size), random_state=seed)
    X_train, X_cal, X_test = X.iloc[train_idx], X.iloc[cal_idx], X.iloc[test_idx]
    y_train, y_cal, y_test = y[train_idx], y[cal_idx], y[test_idx]
    say(f"Rows: {len(train_idx)} train, {len(cal_idx)} calibration, {len(test_idx)} test (seed {seed}).")
    if len(cal_idx) < 100:
        say("Note: few calibration rows, so the prediction range is less exact. More data helps.")

    lower_q, upper_q = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    say("Training the best-guess model and the two range models...")
    common = {"seed": seed, "n_train": len(train_idx)}
    models: dict[str, Any] = {
        "point": _make_pipeline(features, loss="squared_error", quantile=None, **common).fit(X_train, y_train),
        "low": _make_pipeline(features, loss="quantile", quantile=lower_q, **common).fit(X_train, y_train),
        "high": _make_pipeline(features, loss="quantile", quantile=upper_q, **common).fit(X_train, y_train),
    }
    models["range_offset"] = conformal_offset(
        models["low"].predict(X_cal), models["high"].predict(X_cal), y_cal, coverage
    )

    # Look at every used row, not only the train rows: one negative test row is enough.
    non_negative = bool(y.min() >= 0)
    test_out = predict_with_range(models, X_test, non_negative)
    scores = regression_metrics(y_test, test_out, float(np.median(y_train)))
    metrics = {
        "measured_on": "test rows that were not used for training or calibration",
        "train_rows": int(len(train_idx)),
        "calibration_rows": int(len(cal_idx)),
        "test_rows": int(len(test_idx)),
        # A score that is not a finite number (for example R² with one test row) is stored as null.
        **{k: round(v, 6) if math.isfinite(v) else None for k, v in scores.items()},
        "range_coverage_target": coverage,
        "baseline": "always predict the median of the training rows",
    }

    target_spec = {
        "name": target,
        "description": descriptions.get(target, ""),
        "min": float(y.min()),
        "max": float(y.max()),
        "median": float(np.median(y)),
        "non_negative": non_negative,
    }
    prediction_range = {
        "coverage": coverage,
        "method": "conformalized quantile regression",
        "lower_quantile": lower_q,
        "upper_quantile": upper_q,
    }
    schema = {
        "format": BUNDLE_FORMAT,
        "format_version": FORMAT_VERSION,
        "title": title,
        "description": description,
        "task": "regression",
        "target": target_spec,
        "features": features,
        "dropped_columns": dropped_columns,
        "prediction_range": prediction_range,
        "input": input_json_schema(features, title),
        "output": output_json_schema(target_spec, coverage),
    }
    point_params = models["point"].named_steps["model"].get_params()
    metadata = {
        "harbor_version": __version__,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "options": options,
        "source": source,
        "rows": {"total": source["rows"], "skipped_missing_target": dropped_rows, "used": int(len(df))},
        "split": {
            "seed": seed,
            "test_size": test_size,
            "calibration_size": calibration_size,
            "train": int(len(train_idx)),
            "calibration": int(len(cal_idx)),
            "test": int(len(test_idx)),
        },
        "model": {
            "type": "HistGradientBoostingRegressor",
            "params": {k: point_params[k] for k in ("learning_rate", "max_iter", "early_stopping", "random_state")},
            "iterations": {k: int(models[k].named_steps["model"].n_iter_) for k in ("point", "low", "high")},
            "range_offset": models["range_offset"],
        },
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "skops": skops.__version__,
        },
    }
    save_bundle(Path(out), models=models, schema=schema, metrics=metrics, metadata=metadata)
    return load_bundle(out)
