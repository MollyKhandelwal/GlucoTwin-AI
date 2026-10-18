
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "real_cgm_forecast.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "real_cgm_model.joblib"
METRICS_PATH = OUTPUT_DIR / "real_cgm_model_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "real_cgm_model_predictions.csv"

FEATURES = [
    "glucose_now",
    "glucose_15m_ago",
    "glucose_30m_ago",
    "glucose_60m_ago",
]

TARGET = "target_glucose_60m"
HORIZON_STEPS = 12  # 12 readings × approximately 5 minutes = 60 minutes


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run: python src/prepare_real_forecast.py"
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    required = FEATURES + [TARGET, "timestamp", "segment_id"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=required).copy()
    df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows remain after removing missing values.")

    # Build a chronological train/test split separately for each segment.
    train_parts = []
    test_parts = []

    for segment_id, group in df.groupby("segment_id", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)

        if len(group) < 30:
            print(f"Skipping segment {segment_id}: too few rows ({len(group)}).")
            continue

        split = int(len(group) * 0.80)

        # Purge the forecast horizon before the test boundary.
        train_end = split - HORIZON_STEPS

        if train_end <= 0 or split >= len(group):
            print(f"Skipping segment {segment_id}: insufficient split data.")
            continue

        train_parts.append(group.iloc[:train_end].copy())
        test_parts.append(group.iloc[split:].copy())

    if not train_parts or not test_parts:
        raise ValueError("Unable to create valid chronological train/test splits.")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    print("Training Random Forest on real CGM data...")
    print("Training samples:", len(train_df))
    print("Test samples:", len(test_df))
    print("Segments:", sorted(df["segment_id"].unique().tolist()))

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    baseline_predictions = X_test["glucose_now"].to_numpy()

    model_mae = mean_absolute_error(y_test, predictions)
    model_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "Persistence baseline",
                "test_samples": len(test_df),
                "forecast_minutes": 60,
                "MAE_mg_dl": baseline_mae,
                "RMSE_mg_dl": baseline_rmse,
            },
            {
                "model": "Random Forest",
                "test_samples": len(test_df),
                "forecast_minutes": 60,
                "MAE_mg_dl": model_mae,
                "RMSE_mg_dl": model_rmse,
            },
        ]
    )

    metrics.to_csv(METRICS_PATH, index=False)

    results = test_df[
        ["timestamp", "segment_id", "glucose_now", TARGET]
    ].copy()

    results = results.rename(columns={TARGET: "actual_glucose_60m"})
    results["baseline_prediction"] = baseline_predictions
    results["model_prediction"] = predictions
    results["model_absolute_error"] = (
        results["actual_glucose_60m"] - results["model_prediction"]
    ).abs()

    results.to_csv(PREDICTIONS_PATH, index=False)

    # Save model and metadata separately from the synthetic-data model.
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "target": TARGET,
            "forecast_minutes": 60,
            "training_data": "Dexcom CGM, one participant, exploratory",
        },
        MODEL_PATH,
    )

    print("\n========== REAL CGM MODEL RESULTS ==========")
    print(metrics.round(3).to_string(index=False))
    print("\nFeature importance:")
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
        name="importance",
    ).sort_values(ascending=False)
    print(importance.round(4).to_string())

    print("\nSaved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved predictions:", PREDICTIONS_PATH)

    print(
        "\nIMPORTANT: These are exploratory results from a single "
        "participant's CGM data. They do not establish clinical accuracy "
        "or generalization to other patients."
    )


if __name__ == "__main__":
    main()
