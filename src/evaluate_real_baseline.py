
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "real_cgm_forecast.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "real_cgm_baseline_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "real_cgm_baseline_predictions.csv"

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

required = [
    "timestamp",
    "glucose_now",
    "target_glucose_60m",
    "segment_id",
]

missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df = df.dropna(subset=required).copy()

# Chronological split: earliest 80% for development,
# latest 20% for the final holdout evaluation.
split_index = int(len(df) * 0.80)

if split_index <= 0 or split_index >= len(df):
    raise ValueError("Not enough samples for a chronological split.")

test = df.iloc[split_index:].copy()

# Persistence baseline: predict future glucose as current glucose.
test["baseline_prediction"] = test["glucose_now"]
test["absolute_error"] = (
    test["target_glucose_60m"] - test["baseline_prediction"]
).abs()

mae = mean_absolute_error(
    test["target_glucose_60m"],
    test["baseline_prediction"],
)

rmse = np.sqrt(
    mean_squared_error(
        test["target_glucose_60m"],
        test["baseline_prediction"],
    )
)

metrics = pd.DataFrame(
    [
        {
            "model": "Persistence baseline",
            "test_samples": len(test),
            "forecast_minutes": 60,
            "MAE_mg_dl": mae,
            "RMSE_mg_dl": rmse,
            "test_start": test["timestamp"].min(),
            "test_end": test["timestamp"].max(),
        }
    ]
)

metrics.to_csv(RESULTS_PATH, index=False)

test[
    [
        "timestamp",
        "segment_id",
        "glucose_now",
        "target_glucose_60m",
        "baseline_prediction",
        "absolute_error",
    ]
].to_csv(PREDICTIONS_PATH, index=False)

print("\n========== REAL CGM BASELINE EVALUATION ==========")
print(metrics.to_string(index=False))
print("\nSaved metrics:", RESULTS_PATH)
print("Saved predictions:", PREDICTIONS_PATH)
print(
    "\nExploratory result from one participant only. "
    "This is not clinical validation."
)
