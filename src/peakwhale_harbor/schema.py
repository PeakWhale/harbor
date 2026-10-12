"""Column type inference and input checking.

Harbor knows two kinds of input columns:

* ``numeric``: numbers. Empty cells are allowed, the model handles them.
* ``categorical``: a small set of labels, such as "north" or "south".

Columns that cannot help a model (empty, constant, IDs, free text, dates) are
dropped, and Harbor says why.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from difflib import get_close_matches
from typing import Any

import numpy as np
import pandas as pd

#: Gradient boosting supports up to 255 categories per column. Keep a margin.
MAX_CATEGORIES = 250
#: A text column counts as numeric when at least this share of values are numbers.
NUMERIC_SHARE = 0.95
#: Column names that usually hold row IDs, such as "id", "customer_id" or "Unnamed: 0".
ID_NAME = re.compile(r"^(id|uuid|index|unnamed: ?\d+)$|[_\s]id$", re.IGNORECASE)
DATE_TEXT = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}([ T].*)?$|^\d{1,2}[/.]\d{1,2}[/.]\d{2,4}$")
#: Only this many problems are listed in one error message.
MAX_PROBLEMS = 50


class InputError(ValueError):
    """The input does not match the model's schema.

    ``problems`` is a list of small dicts, one per problem, that an API can
    return as JSON.
    """

    def __init__(self, message: str, problems: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.message = message
        self.problems = problems or []

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.message, "problems": self.problems}


# ---------------------------------------------------------------------------
# Single values
# ---------------------------------------------------------------------------


def is_missing(value: Any) -> bool:
    """True for None, NaN, pandas NA and empty or blank strings."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, np.floating) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def parse_number(value: Any) -> float | None:
    """Read one value as a number. Returns None for a missing value.

    Raises ValueError with a short message when the value is not a number.
    """
    if is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("expected a number, got true/false")
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            number = float(value)
        except OverflowError:  # a whole number too large for a float, such as 10**400
            raise ValueError("expected a finite number") from None
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            raise ValueError(f"expected a number, got {value!r}") from None
    else:
        raise ValueError(f"expected a number, got {type(value).__name__}")
    if math.isnan(number):
        return None
    if math.isinf(number):
        raise ValueError("expected a finite number")
    return number


def category_label(value: Any) -> str | None:
    """Turn one cell into the text label used for a categorical column."""
    if is_missing(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def match_category(value: Any, categories: Sequence[str]) -> str | None:
    """Find the known category for ``value``.

    Tries an exact match, then a match that ignores upper/lower case, then a
    match by number ("2.0" matches "2"). Returns None for a missing value and
    raises ValueError for an unknown one.
    """
    label = category_label(value)
    if label is None:
        return None
    if label in categories:
        return label
    folded = [c for c in categories if c.casefold() == label.casefold()]
    if len(folded) == 1:
        return folded[0]
    try:
        number = float(label)
        for c in categories:
            try:
                if float(c) == number:
                    return c
            except ValueError:
                continue
    except ValueError:
        pass
    shown = ", ".join(repr(c) for c in categories[:20])
    more = f" (and {len(categories) - 20} more)" if len(categories) > 20 else ""
    raise ValueError(f"unknown value {label!r}. Allowed values: {shown}{more}")


# ---------------------------------------------------------------------------
# Whole columns (used for training)
# ---------------------------------------------------------------------------


def numeric_series(series: pd.Series) -> pd.Series:
    """Convert a column to float. Values that are not numbers become NaN."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.map(lambda v: None if is_missing(v) else str(v).strip())
    return pd.to_numeric(text, errors="coerce").astype(float)


def category_series(series: pd.Series) -> pd.Series:
    """Convert a column to text labels. Missing values become NaN."""
    labels = [category_label(v) for v in series.tolist()]
    return pd.Series([np.nan if v is None else v for v in labels], index=series.index, dtype=object)


def _non_missing(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series.dropna()
    return series[~series.map(is_missing)]


def _guess_kind(name: str, values: pd.Series) -> tuple[str | None, str]:
    """Guess the type of one column from its non-missing values.

    Returns (kind, reason). ``kind`` is None when the column should be dropped.
    """
    if pd.api.types.is_bool_dtype(values):
        return "categorical", ""
    if pd.api.types.is_datetime64_any_dtype(values):
        return None, "dates are not supported yet (add columns such as year or month instead)"
    if pd.api.types.is_numeric_dtype(values):
        if ID_NAME.search(name) and values.is_unique:
            return None, "looks like an ID column (a different value in every row)"
        return "numeric", ""

    text = values.astype(str).str.strip()
    numbers = pd.to_numeric(text, errors="coerce")
    if numbers.notna().mean() >= NUMERIC_SHARE:
        if ID_NAME.search(name) and numbers.dropna().is_unique:
            return None, "looks like an ID column (a different value in every row)"
        return "numeric", ""
    if text.str.match(DATE_TEXT).mean() >= NUMERIC_SHARE:
        return None, "looks like dates; dates are not supported yet (add columns such as year or month instead)"

    unique = text.nunique()
    if unique > MAX_CATEGORIES:
        return None, f"too many different text values ({unique}); looks like IDs or free text"
    if unique > max(20, 0.5 * len(text)):
        return None, f"almost every row has a different value ({unique} different values); looks like IDs or free text"
    return "categorical", ""


def _check_names(kind: str, names: Iterable[str], columns: Sequence[str]) -> list[str]:
    names = list(names)
    for n in names:
        if n not in columns:
            hint = get_close_matches(n, columns, n=1)
            extra = f" Did you mean {hint[0]!r}?" if hint else ""
            raise ValueError(f"{kind}: column {n!r} is not in the file.{extra}")
    return names


def _clean_number(x: float) -> float | int:
    """Round-trip friendly number for JSON (ints stay ints)."""
    x = float(x)
    return int(x) if x.is_integer() and abs(x) < 2**53 else x


def infer_features(
    df: pd.DataFrame,
    target: str,
    *,
    drop: Iterable[str] = (),
    categorical: Iterable[str] = (),
    numeric: Iterable[str] = (),
    descriptions: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Decide which columns to use and what type each one has.

    Returns ``(features, dropped)``. Each feature is a JSON-ready dict. Each
    dropped column is ``{"name": ..., "reason": ...}``.
    """
    columns = [str(c) for c in df.columns]
    drop = set(_check_names("--drop", drop, columns))
    categorical = set(_check_names("--categorical", categorical, columns))
    numeric = set(_check_names("--numeric", numeric, columns))
    both = categorical & numeric
    if both:
        raise ValueError(f"columns cannot be both numeric and categorical: {sorted(both)}")
    descriptions = dict(descriptions or {})

    features: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    for column in df.columns:
        name, series = str(column), df[column]
        if name == target:
            continue
        if name in drop:
            dropped.append({"name": name, "reason": "you asked to drop it"})
            continue
        values = _non_missing(series)
        if values.empty:
            dropped.append({"name": name, "reason": "every value is empty"})
            continue

        if name in categorical:
            kind, reason = "categorical", ""
        elif name in numeric:
            kind, reason = "numeric", ""
        else:
            kind, reason = _guess_kind(name, values)
        if kind is None:
            dropped.append({"name": name, "reason": reason})
            continue

        spec: dict[str, Any] = {"name": name, "type": kind, "description": descriptions.get(name, "")}
        n_rows = len(series)
        if kind == "numeric":
            col = numeric_series(series)
            ok = col.dropna()
            if ok.empty:
                dropped.append({"name": name, "reason": "no value could be read as a number"})
                continue
            if ok.nunique() == 1:
                dropped.append({"name": name, "reason": "has the same value in every row"})
                continue
            unreadable = int(len(values) - len(ok))
            integer = bool((ok == ok.round()).all())
            median = float(ok.median())
            # A clean typical value: whole numbers stay whole, no float noise.
            median = float(round(median)) if integer else float(f"{median:.10g}")
            spec.update(
                {
                    "min": _clean_number(ok.min()),
                    "max": _clean_number(ok.max()),
                    "median": _clean_number(median),
                    "integer": integer,
                    "missing_share": round(1 - len(ok) / n_rows, 4),
                }
            )
            if unreadable:
                spec["note"] = f"{unreadable} values were not numbers and are treated as empty"
        else:
            col = category_series(series).dropna()
            counts = col.value_counts()
            if len(counts) == 1:
                dropped.append({"name": name, "reason": "has the same value in every row"})
                continue
            if len(counts) > MAX_CATEGORIES:
                reason = f"too many different values ({len(counts)}, the limit is {MAX_CATEGORIES})"
                dropped.append({"name": name, "reason": reason})
                continue
            spec.update(
                {
                    "categories": sorted(counts.index.tolist()),
                    "most_common": str(counts.index[0]),
                    "missing_share": round(1 - len(col) / n_rows, 4),
                }
            )
        features.append(spec)
    return features, dropped


def example_value(spec: Mapping[str, Any]) -> Any:
    """A typical value for a feature: the median or the most common label."""
    if spec["type"] == "numeric":
        return spec["median"]
    return spec["most_common"]


def example_record(features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {f["name"]: example_value(f) for f in features}


def to_model_frame(df: pd.DataFrame, features: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build the table the model expects: the right columns, order and types."""
    data = {}
    for f in features:
        col = df[f["name"]]
        data[f["name"]] = numeric_series(col) if f["type"] == "numeric" else category_series(col)
    return pd.DataFrame(data, index=df.index)


# ---------------------------------------------------------------------------
# Checking requests
# ---------------------------------------------------------------------------


def check_records(
    records: Sequence[Any],
    features: Sequence[Mapping[str, Any]],
    *,
    row_key: str = "row",
    first_row: int = 0,
) -> tuple[pd.DataFrame, list[list[str]]]:
    """Check request records against the schema and build the model table.

    ``records`` is a list of dicts (one per row). Every feature must be present
    as a key; the value may be empty (null) when it is unknown. Extra keys are
    ignored.

    Returns ``(frame, notes)``; ``notes[i]`` lists friendly warnings for row i,
    such as values outside the range seen in training.

    Raises :class:`InputError` listing every problem found.
    """
    problems: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    notes: list[list[str]] = []
    names = [f["name"] for f in features]

    for i, record in enumerate(records):
        where = {row_key: first_row + i}
        if not isinstance(record, Mapping):
            problems.append({**where, "error": "each record must be an object with one key per input field"})
            continue
        absent = [n for n in names if n not in record]
        if absent:
            problems.append({**where, "error": "missing fields", "fields": absent})
            continue

        row: dict[str, Any] = {}
        row_notes: list[str] = []
        for f in features:
            name = f["name"]
            value = record[name]
            try:
                if f["type"] == "numeric":
                    parsed = parse_number(value)
                    if parsed is not None and parsed < f["min"]:
                        row_notes.append(
                            f"{name} is below the lowest value seen in training ({f['min']}); "
                            "the prediction may be less reliable"
                        )
                    elif parsed is not None and parsed > f["max"]:
                        row_notes.append(
                            f"{name} is above the highest value seen in training ({f['max']}); "
                            "the prediction may be less reliable"
                        )
                else:
                    parsed = match_category(value, f["categories"])
            except ValueError as exc:
                problems.append({**where, "field": name, "error": str(exc)})
                continue
            if parsed is None:
                row[name] = np.nan
                row_notes.append(f"{name} is empty; the model fills the gap as best it can")
            else:
                row[name] = parsed
        rows.append(row)
        notes.append(row_notes)

    if problems:
        shown = problems[:MAX_PROBLEMS]
        message = f"{len(problems)} problem(s) found in the input."
        if len(problems) > MAX_PROBLEMS:
            message += f" Showing the first {MAX_PROBLEMS}."
        raise InputError(message, shown)

    frame = pd.DataFrame(rows, columns=names)
    for f in features:
        if f["type"] == "numeric":
            frame[f["name"]] = frame[f["name"]].astype(float)
        else:
            frame[f["name"]] = frame[f["name"]].astype(object)
    return frame, notes


# ---------------------------------------------------------------------------
# JSON Schema (https://json-schema.org) for the API
# ---------------------------------------------------------------------------


def input_json_schema(features: Sequence[Mapping[str, Any]], title: str) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for f in features:
        description = f.get("description") or ""
        if f["type"] == "numeric":
            hint = f"Seen in training: {f['min']} to {f['max']}."
            prop: dict[str, Any] = {"type": ["number", "null"], "examples": [f["median"]]}
        else:
            hint = "One of the listed values."
            prop = {"type": ["string", "null"], "enum": [*f["categories"], None], "examples": [f["most_common"]]}
        prop["description"] = f"{description} {hint}".strip()
        properties[f["name"]] = prop
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"{title} input",
        "description": "One record. Use null for an unknown value. Extra keys are ignored.",
        "type": "object",
        "properties": properties,
        "required": [f["name"] for f in features],
        "additionalProperties": True,
    }


def output_json_schema(target: Mapping[str, Any], coverage: float) -> dict[str, Any]:
    name = target["name"]
    pct = round(coverage * 100)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"{name} prediction",
        "type": "object",
        "properties": {
            "prediction": {"type": "number", "description": f"Best single guess for {name}."},
            "low": {"type": "number", "description": f"Low end of the {pct}% prediction range."},
            "high": {"type": "number", "description": f"High end of the {pct}% prediction range."},
            "coverage": {"type": "number", "description": "Target share of true values inside the range."},
            "notes": {"type": "array", "items": {"type": "string"}, "description": "Warnings about the input."},
        },
        "required": ["prediction", "low", "high", "coverage", "notes"],
    }
