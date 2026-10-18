
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "outputs" / "real_cgm_model_predictions.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Predictions file not found: {INPUT_PATH}\n"
        "Run python src/train_real_model.py first."
    )

df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

required = [
    "timestamp",
    "segment_id",
    "actual_glucose_60m",
    "baseline_prediction",
    "model_prediction",
]

missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df = df.dropna(subset=required).copy()

if df.empty:
    raise ValueError("No valid predictions found.")

# Show a manageable chronological sample from the test set.
sample_size = min(200, len(df))
plot_df = df.tail(sample_size).copy()

baseline_mae = mean_absolute_error(
    df["actual_glucose_60m"], df["baseline_prediction"]
)
model_mae = mean_absolute_error(
    df["actual_glucose_60m"], df["model_prediction"]
)

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

axes[0].plot(
    plot_df["timestamp"],
    plot_df["actual_glucose_60m"],
    label="Actual glucose at +60 min",
    linewidth=2,
)
axes[0].plot(
    plot_df["timestamp"],
    plot_df["baseline_prediction"],
    label="Persistence baseline",
    alpha=0.85,
)
axes[0].plot(
    plot_df["timestamp"],
    plot_df["model_prediction"],
    label="Random Forest prediction",
    alpha=0.85,
)

axes[0].set_title("Real CGM: 60-Minute Forecast Comparison")
axes[0].set_ylabel("Glucose (mg/dL)")
axes[0].legend()
axes[0].grid(True, alpha=0.25)

baseline_error = (
    plot_df["actual_glucose_60m"] - plot_df["baseline_prediction"]
).abs()
model_error = (
    plot_df["actual_glucose_60m"] - plot_df["model_prediction"]
).abs()

axes[1].plot(
    plot_df["timestamp"],
    baseline_error,
    label="Baseline absolute error",
    alpha=0.8,
)
axes[1].plot(
    plot_df["timestamp"],
    model_error,
    label="Random Forest absolute error",
    alpha=0.8,
)

axes[1].set_title("Absolute Forecast Error")
axes[1].set_ylabel("Absolute error (mg/dL)")
axes[1].set_xlabel("Timestamp")
axes[1].legend()
axes[1].grid(True, alpha=0.25)

fig.suptitle(
    f"Test-set MAE — Baseline: {baseline_mae:.2f} | "
    f"Random Forest: {model_mae:.2f} mg/dL",
    fontsize=14,
)
fig.tight_layout()

output_path = OUTPUT_DIR / "real_cgm_prediction_comparison.png"
fig.savefig(output_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print("Comparison graph saved to:", output_path)
print("Test samples plotted:", sample_size)
print(f"Baseline MAE: {baseline_mae:.3f} mg/dL")
print(f"Random Forest MAE: {model_mae:.3f} mg/dL")
print(
    "Interpretation: compare both models against the actual values; "
    "lower error is better."
)
