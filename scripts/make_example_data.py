"""Create examples/equipment_repairs.csv, a small made-up data set.

Each row is one machine repair at a fictional factory. The target column is
``repair_hours``: how many hours the repair took. Nothing here is real data.

The random seed is fixed, so running this script again gives the same file:

    python scripts/make_example_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
ROWS = 1200
OUT = Path(__file__).resolve().parent.parent / "examples" / "equipment_repairs.csv"

BASE_HOURS = {"pump": 5.0, "compressor": 8.0, "conveyor": 3.5, "press": 11.0}
SHIFT_FACTOR = {"day": 1.0, "night": 1.25, "weekend": 1.4}


def make(rows: int = ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    machine_type = rng.choice(list(BASE_HOURS), size=rows, p=[0.35, 0.25, 0.25, 0.15])
    site = rng.choice(["north", "south", "east"], size=rows, p=[0.4, 0.35, 0.25])
    shift = rng.choice(list(SHIFT_FACTOR), size=rows, p=[0.6, 0.3, 0.1])
    age_years = rng.uniform(0.5, 20.0, size=rows)
    weekly_run_hours = rng.integers(20, 169, size=rows)
    days_since_service = rng.integers(0, 400, size=rows)
    vibration_mm_s = rng.gamma(2.0, 1.5, size=rows) + 0.3
    spare_part_in_stock = rng.choice(["yes", "no"], size=rows, p=[0.75, 0.25])

    base = np.array([BASE_HOURS[m] for m in machine_type])
    shift_factor = np.array([SHIFT_FACTOR[s] for s in shift])
    part_delay = np.where(spare_part_in_stock == "no", rng.uniform(4.0, 30.0, size=rows), 0.0)
    site_factor = np.where(site == "east", 1.15, 1.0)

    hours = (
        base
        * (1.0 + 0.04 * age_years)
        * (1.0 + days_since_service / 500.0)
        * (1.0 + 0.08 * vibration_mm_s)
        * (1.0 + weekly_run_hours / 400.0)
        * shift_factor
        * site_factor
    )
    # Noise grows with the size of the job, so some repairs are much harder to
    # predict than others. This is where prediction ranges help.
    hours = hours * rng.lognormal(mean=0.0, sigma=0.25, size=rows) + part_delay

    df = pd.DataFrame(
        {
            "asset_id": [f"EQ-{i:05d}" for i in range(1, rows + 1)],
            "machine_type": machine_type,
            "site": site,
            "shift": shift,
            "age_years": age_years.round(1),
            "weekly_run_hours": weekly_run_hours,
            "days_since_service": days_since_service,
            "vibration_mm_s": vibration_mm_s.round(2),
            "spare_part_in_stock": spare_part_in_stock,
            "repair_hours": hours.round(1),
        }
    )
    # Some sensor readings are missing, like in real spreadsheets.
    missing = rng.random(rows) < 0.03
    df.loc[missing, "vibration_mm_s"] = np.nan
    return df


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(OUT, index=False)
    print(f"Wrote {ROWS} rows to {OUT}")


if __name__ == "__main__":
    main()
