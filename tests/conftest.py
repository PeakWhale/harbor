from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peakwhale_harbor import load_bundle, train

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CSV = ROOT / "examples" / "equipment_repairs.csv"
TARGET = "repair_hours"


def mixed_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A small table with every kind of column Harbor has to recognise."""
    rng = np.random.default_rng(seed)
    size = rng.uniform(10, 100, n)
    color = rng.choice(["red", "green", "blue"], n)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "size": size,
            "rooms": rng.integers(1, 6, n),
            "color": color,
            "flag": rng.choice([True, False], n),
            "price_text": [f"{v:.2f}" for v in rng.uniform(1, 9, n)],
            "constant": 5,
            "empty": np.nan,
            "note": [f"free text number {i}" for i in range(n)],
            "day": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
            "target": size * 3 + np.where(color == "red", 50, 0) + rng.normal(0, 5, n),
        }
    )


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("bundles") / "equipment"
    train(EXAMPLE_CSV, TARGET, out)
    return out


@pytest.fixture(scope="session")
def bundle(bundle_dir):
    return load_bundle(bundle_dir)


@pytest.fixture()
def client(bundle):
    from peakwhale_harbor.server import create_app

    app = create_app(bundle)
    app.testing = True
    return app.test_client()


@pytest.fixture()
def good_record(bundle) -> dict:
    return {
        "machine_type": "pump",
        "site": "north",
        "shift": "day",
        "age_years": 6.5,
        "weekly_run_hours": 90,
        "days_since_service": 120,
        "vibration_mm_s": 2.8,
        "spare_part_in_stock": "yes",
    }
