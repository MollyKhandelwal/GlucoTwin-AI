
from pathlib import Path

import numpy as np
import pandas as pd

# Project root: GlucoTwin-AI/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
NUM_PATIENTS = 50
DAYS_PER_PATIENT = 3
READING_INTERVAL_MINUTES = 5


def generate_patient_records(rng):
    """Generate clearly labelled synthetic EHR records."""
    records = []

    for i in range(1, NUM_PATIENTS + 1):
        age = int(rng.integers(30, 76))
        bmi = round(float(rng.uniform(19, 36)), 1)
        hba1c = round(float(rng.uniform(5.5, 10.0)), 1)

        records.append({
            "patient_id": f"GT{i:04d}",
            "age": age,
            "bmi": bmi,
            "hba1c_percent": hba1c,
            "diabetes_history": True,
            "data_type": "synthetic",
        })

    return pd.DataFrame(records)


def generate_glucose_readings(patients, rng):
    """Generate synthetic CGM-like readings at five-minute intervals."""
    all_readings = []
    start_time = pd.Timestamp("2026-10-01 00:00:00")

    readings_per_day = (24 * 60) // READING_INTERVAL_MINUTES
    total_readings = DAYS_PER_PATIENT * readings_per_day

    for patient in patients.itertuples(index=False):
        baseline = 105 + (patient.hba1c_percent - 5.5) * 13
        phase = rng.uniform(0, 2 * np.pi)

        timestamps = pd.date_range(
            start=start_time,
            periods=total_readings,
            freq=f"{READING_INTERVAL_MINUTES}min",
        )

        for step, timestamp in enumerate(timestamps):
            daily_pattern = 15 * np.sin(
                2 * np.pi * (timestamp.hour / 24) + phase
            )

            # Synthetic meal-related rises, not real physiological modelling.
            meal_pattern = 0
            minute_of_day = timestamp.hour * 60 + timestamp.minute

            for meal_minute in (8 * 60, 13 * 60, 19 * 60):
                distance = abs(minute_of_day - meal_minute)
                distance = min(distance, 1440 - distance)

                if distance < 120:
                    meal_pattern += 35 * np.exp(-distance / 45)

            glucose = (
                baseline
                + daily_pattern
                + meal_pattern
                + rng.normal(0, 5)
            )

            all_readings.append({
                "patient_id": patient.patient_id,
                "timestamp": timestamp,
                "glucose_mg_dl": round(
                    float(np.clip(glucose, 65, 300)), 1
                ),
                "data_type": "synthetic",
            })

    return pd.DataFrame(all_readings)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    patients = generate_patient_records(rng)
    readings = generate_glucose_readings(patients, rng)

    ehr_path = DATA_DIR / "synthetic_ehr.csv"
    cgm_path = DATA_DIR / "synthetic_cgm.csv"

    patients.to_csv(ehr_path, index=False)
    readings.to_csv(cgm_path, index=False)

    print("Synthetic dataset generated successfully!")
    print(f"Patients: {len(patients)}")
    print(f"CGM readings: {len(readings)}")
    print(f"EHR saved to: {ehr_path}")
    print(f"CGM saved to: {cgm_path}")
    print("\nImportant: These are simulated values, not real patient data.")


if __name__ == "__main__":
    main()
