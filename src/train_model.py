
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

EHR_FILE = DATA_DIR / "synthetic_ehr.csv"
CGM_FILE = DATA_DIR / "synthetic_cgm.csv"

MODEL_FILE = MODELS_DIR / "glucose_model.joblib"
METRICS_FILE = OUTPUTS_DIR / "model_metrics.csv"
PREDICTIONS_FILE = OUTPUTS_DIR / "test_predictions.csv"

FORECAST_MINUTES = 60
READING_INTERVAL_MINUTES = 5
HORIZON_STEPS = FORECAST_MINUTES // READING_INTERVAL_MINUTES

RANDOM_SEED = 42

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD AND FUSE EHR + CGM
# ============================================================

def load_and_fuse_data():
    """Load synthetic EHR and CGM, then join by patient ID."""

    if not EHR_FILE.exists() or not CGM_FILE.exists():
        raise FileNotFoundError(
            "Required dataset files are missing.\n"
            "Run: python src/generate_data.py"
        )

    ehr = pd.read_csv(EHR_FILE)
    cgm = pd.read_csv(
        CGM_FILE,
        parse_dates=["timestamp"],
    )

    required_ehr = {
        "patient_id", "age", "bmi", "hba1c_percent"
    }
    required_cgm = {
        "patient_id", "timestamp", "glucose_mg_dl"
    }

    if not required_ehr.issubset(ehr.columns):
        raise ValueError("EHR file is missing required columns.")

    if not required_cgm.issubset(cgm.columns):
        raise ValueError("CGM file is missing required columns.")

    if ehr["patient_id"].duplicated().any():
        raise ValueError("EHR must contain one row per patient.")

    df = cgm.merge(
        ehr,
        on="patient_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = int((df["_merge"] != "both").sum())
    df = df.drop(columns="_merge")

    if unmatched:
        raise ValueError(
            f"{unmatched} CGM rows have no matching EHR record."
        )

    if df["timestamp"].isna().any():
        raise ValueError("CGM timestamps contain missing values.")

    if df["glucose_mg_dl"].isna().any():
        raise ValueError("CGM glucose values contain missing values.")

    if df.duplicated(["patient_id", "timestamp"]).any():
        raise ValueError(
            "Duplicate patient/timestamp readings detected."
        )

    df = df.sort_values(
        ["patient_id", "timestamp"]
    ).reset_index(drop=True)

    print("\nEHR + CGM fusion successful.")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Fused rows: {len(df)}")

    return df


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def prepare_features(df):
    """
    Use only current and past readings as features.
    Predict glucose 60 minutes after the current timestamp.
    """

    df = df.copy()

    grouped = df.groupby(
        "patient_id",
        sort=False,
    )["glucose_mg_dl"]

    df["glucose_now"] = df["glucose_mg_dl"]
    df["glucose_15m_ago"] = grouped.shift(3)
    df["glucose_30m_ago"] = grouped.shift(6)
    df["glucose_60m_ago"] = grouped.shift(12)

    df["trend_15m"] = (
        df["glucose_now"] - df["glucose_15m_ago"]
    )

    df["hour"] = df["timestamp"].dt.hour
    df["minute_of_day"] = (
        df["timestamp"].dt.hour * 60
        + df["timestamp"].dt.minute
    )

    # Target: glucose 12 readings (60 minutes) ahead.
    df["target_glucose_60m"] = grouped.shift(-HORIZON_STEPS)

    feature_cols = [
        "age",
        "bmi",
        "hba1c_percent",
        "glucose_now",
        "glucose_15m_ago",
        "glucose_30m_ago",
        "glucose_60m_ago",
        "trend_15m",
        "hour",
        "minute_of_day",
    ]

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=feature_cols + ["target_glucose_60m"]
    ).copy()

    if df.empty:
        raise ValueError("No usable rows remain after feature engineering.")

    return df, feature_cols


# ============================================================
# 4. CHRONOLOGICAL SPLIT WITH HORIZON GAP
# ============================================================

def chronological_split(df):
    """
    Per-patient chronological split:
    70% train, 15% validation, 15% test.

    Remove boundary samples whose target timestamp crosses
    into the next split. The gap is the forecast horizon.
    """

    train_parts = []
    validation_parts = []
    test_parts = []

    df = df.sort_values(
        ["patient_id", "timestamp"]
    ).reset_index(drop=True)

    for patient_id, group in df.groupby(
        "patient_id",
        sort=False,
    ):
        group = group.copy()
        n = len(group)

        train_end = int(n * 0.70)
        validation_end = int(n * 0.85)

        # Keep a horizon-sized gap at both boundaries.
        train_stop = max(0, train_end - HORIZON_STEPS)
        validation_start = train_end
        validation_stop = max(
            validation_start,
            validation_end - HORIZON_STEPS,
        )
        test_start = validation_end

        train_group = group.iloc[:train_stop].copy()

        validation_group = group.iloc[
            validation_start:validation_stop
        ].copy()

        test_group = group.iloc[test_start:].copy()

        if not train_group.empty:
            train_parts.append(train_group)

        if not validation_group.empty:
            validation_parts.append(validation_group)

        if not test_group.empty:
            test_parts.append(test_group)

    if not train_parts or not validation_parts or not test_parts:
        raise ValueError(
            "A data split is empty. Check the number of readings "
            "available per patient."
        )

    train = pd.concat(train_parts, ignore_index=True)
    validation = pd.concat(validation_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    print("\nChronological split:")
    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Test rows: {len(test)}")
    print(f"Forecast horizon gap: {HORIZON_STEPS} readings")

    return train, validation, test


# ============================================================
# 5. METRIC HELPER
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "MAE_mg_dl": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE_mg_dl": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
    }


# ============================================================
# 6. TRAIN, EVALUATE AND SAVE
# ============================================================

def main():
    df = load_and_fuse_data()
    df, feature_cols = prepare_features(df)

    train, validation, test = chronological_split(df)

    X_train = train[feature_cols]
    y_train = train["target_glucose_60m"]

    X_validation = validation[feature_cols]
    y_validation = validation["target_glucose_60m"]

    X_test = test[feature_cols]
    y_test = test["target_glucose_60m"]

    # Baseline: current glucose predicts glucose 60 minutes ahead.
    baseline_predictions = test["glucose_now"].to_numpy()

    baseline_scores = calculate_metrics(
        y_test,
        baseline_predictions,
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    validation_predictions = model.predict(X_validation)
    test_predictions = model.predict(X_test)

    validation_scores = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    test_scores = calculate_metrics(
        y_test,
        test_predictions,
    )

    metrics = pd.DataFrame([
        {
            "model": "Current glucose baseline",
            "evaluation_split": "test",
            "forecast_minutes": FORECAST_MINUTES,
            **baseline_scores,
        },
        {
            "model": "Random Forest",
            "evaluation_split": "validation",
            "forecast_minutes": FORECAST_MINUTES,
            **validation_scores,
        },
        {
            "model": "Random Forest",
            "evaluation_split": "test",
            "forecast_minutes": FORECAST_MINUTES,
            **test_scores,
        },
    ])

    metrics.to_csv(METRICS_FILE, index=False)

    # Save test predictions for dashboard visualization.
    prediction_results = test[
        [
            "patient_id",
            "timestamp",
            "glucose_now",
            "target_glucose_60m",
        ]
    ].copy()

    prediction_results["predicted_glucose_60m"] = test_predictions

    prediction_results.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # Save model and exact feature order.
    model_bundle = {
        "model": model,
        "features": feature_cols,
        "horizon_minutes": FORECAST_MINUTES,
        "reading_interval_minutes": READING_INTERVAL_MINUTES,
        "model_type": "RandomForestRegressor",
        "data_type": "synthetic",
    }

    joblib.dump(model_bundle, MODEL_FILE)

    print("\nEvaluation metrics:")
    print(metrics.round(3).to_string(index=False))

    print("\nSaved files:")
    print(f"Model: {MODEL_FILE}")
    print(f"Metrics: {METRICS_FILE}")
    print(f"Predictions: {PREDICTIONS_FILE}")

    print(
        "\nWARNING: This is a synthetic-data experiment. "
        "These results do not establish real-world or clinical "
        "forecasting accuracy."
    )


if __name__ == "__main__":
    main()
