
from pathlib import Path

import pandas as pd

# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "data" / "real_cgm_clean.csv"
OUTPUT_PATH = ROOT / "data" / "real_cgm_forecast.csv"

MAX_GAP_MINUTES = 10
HISTORY_TOLERANCE_MINUTES = 2
FUTURE_TOLERANCE_MINUTES = 5
FORECAST_HORIZON_MINUTES = 60

# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])

required_columns = {"timestamp", "glucose_mg_dl"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

df["glucose_mg_dl"] = pd.to_numeric(
    df["glucose_mg_dl"], errors="coerce"
)

df = (
    df.dropna(subset=["timestamp", "glucose_mg_dl"])
    .sort_values("timestamp")
    .drop_duplicates(subset="timestamp")
    .reset_index(drop=True)
)

if df.empty:
    raise ValueError("No valid CGM readings found.")

# ============================================================
# SPLIT DATA AT LARGE TIME GAPS
# ============================================================

gap_minutes = (
    df["timestamp"].diff().dt.total_seconds() / 60
)

df["segment_id"] = (
    gap_minutes > MAX_GAP_MINUTES
).cumsum().astype(int)

# ============================================================
# BUILD FORECASTING SAMPLES
# ============================================================

def nearest_reading(group, target_time, tolerance_minutes):
    """Return the closest reading within the allowed tolerance."""
    differences = (
        group["timestamp"] - target_time
    ).abs().dt.total_seconds() / 60

    if differences.empty:
        return None

    position = differences.to_numpy().argmin()

    if differences.iloc[position] > tolerance_minutes:
        return None

    return group.iloc[position]["glucose_mg_dl"]


forecast_rows = []

for segment_id, segment in df.groupby("segment_id", sort=True):
    segment = segment.reset_index(drop=True)

    for i in range(len(segment)):
        current_time = segment.loc[i, "timestamp"]
        current_glucose = segment.loc[i, "glucose_mg_dl"]

        # Historical features must use readings from the past.
        glucose_15m = nearest_reading(
            segment,
            current_time - pd.Timedelta(minutes=15),
            HISTORY_TOLERANCE_MINUTES,
        )

        glucose_30m = nearest_reading(
            segment,
            current_time - pd.Timedelta(minutes=30),
            HISTORY_TOLERANCE_MINUTES,
        )

        glucose_60m = nearest_reading(
            segment,
            current_time - pd.Timedelta(minutes=60),
            HISTORY_TOLERANCE_MINUTES,
        )

        # Target must be a future reading about 60 minutes ahead.
        future_readings = segment[
            segment["timestamp"] > current_time
        ]

        if future_readings.empty:
            continue

        target_time = current_time + pd.Timedelta(
            minutes=FORECAST_HORIZON_MINUTES
        )

        target_glucose = nearest_reading(
            future_readings,
            target_time,
            FUTURE_TOLERANCE_MINUTES,
        )

        if any(
            value is None
            for value in [
                glucose_15m,
                glucose_30m,
                glucose_60m,
                target_glucose,
            ]
        ):
            continue

        forecast_rows.append(
            {
                "timestamp": current_time,
                "glucose_now": current_glucose,
                "glucose_15m_ago": glucose_15m,
                "glucose_30m_ago": glucose_30m,
                "glucose_60m_ago": glucose_60m,
                "target_glucose_60m": target_glucose,
                "segment_id": int(segment_id),
            }
        )

# ============================================================
# SAVE AND REPORT
# ============================================================

forecast_df = pd.DataFrame(forecast_rows)

if forecast_df.empty:
    raise ValueError(
        "No valid forecasting windows found. "
        "Check sampling intervals and timestamp tolerances."
    )

forecast_df = forecast_df.sort_values(
    "timestamp"
).reset_index(drop=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
forecast_df.to_csv(OUTPUT_PATH, index=False)

print("\n========== REAL CGM FORECAST DATASET ==========")
print("Input:", INPUT_PATH)
print("Output:", OUTPUT_PATH)
print("Valid forecasting samples:", len(forecast_df))
print("Segments represented:", forecast_df["segment_id"].nunique())
print("Forecast horizon: approximately 60 minutes")

print("\nMissing values:")
print(forecast_df.isna().sum().to_string())

print("\nTimestamp range:")
print(forecast_df["timestamp"].min(), "to", forecast_df["timestamp"].max())

print("\nFirst 5 samples:")
print(forecast_df.head().to_string(index=False))

print("\nTarget glucose summary (mg/dL):")
print(forecast_df["target_glucose_60m"].describe().round(2).to_string())

print("\nDataset preparation completed successfully.")
print("Note: This is exploratory data from one participant, not clinical validation.")
