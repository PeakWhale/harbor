import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

TARGET_COL = "MEDV"

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "boston_housing.csv"

MODEL_PATH = ARTIFACTS_DIR / "regmodel.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaling.pkl"


def load_dataset() -> pd.DataFrame:
    """
    Preferred: load from a local CSV so training is repeatable and offline.
    Fallback: fetch from OpenML (requires internet).
    """
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        return df

    try:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(name="boston", version=1, as_frame=True)
        df = bunch.frame.copy()
        return df
    except Exception as e:
        raise RuntimeError(
            "Could not load dataset. Either add data/boston_housing.csv or run with internet for OpenML fetch."
        ) from e


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    missing_cols = [c for c in (FEATURE_ORDER + [TARGET_COL]) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    X = df[FEATURE_ORDER].astype(float)
    y = df[TARGET_COL].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Training complete")
    print(f"RMSE (root mean squared error): {rmse:.4f}")
    print(f"MAE (mean absolute error): {mae:.4f}")
    print(f"R2 (coefficient of determination): {r2:.4f}")

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    with SCALER_PATH.open("wb") as f:
        pickle.dump(scaler, f)

    print(f"Wrote model to: {MODEL_PATH}")
    print(f"Wrote scaler to: {SCALER_PATH}")


if __name__ == "__main__":
    main()
