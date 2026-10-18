from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

st.set_page_config(
    page_title="GlucoTwin AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.7rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #102c3a, #176b68);
        color: white;
        margin-bottom: 1rem;
    }
    .hero p {
        color: #d7f2ee;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE HELPERS
# ============================================================

def read_csv_if_exists(path, **kwargs):
    """Read a CSV safely. Return an empty DataFrame if missing/unreadable."""
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


@st.cache_data
def load_output_csv(filename):
    """Load an output CSV without passing unsupported keyword arguments."""
    return read_csv_if_exists(OUTPUTS_DIR / filename)


@st.cache_data
def load_synthetic_data():
    ehr_path = DATA_DIR / "synthetic_ehr.csv"
    cgm_path = DATA_DIR / "synthetic_cgm.csv"

    if not ehr_path.exists() or not cgm_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        ehr = pd.read_csv(ehr_path)
        cgm = pd.read_csv(cgm_path)
    except Exception as exc:
        st.warning(f"Could not load synthetic data: {exc}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if "timestamp" in cgm.columns:
        cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")

    if "patient_id" not in ehr.columns or "patient_id" not in cgm.columns:
        return ehr, cgm, pd.DataFrame()

    try:
        fused = cgm.merge(
            ehr,
            on="patient_id",
            how="left",
            validate="many_to_one",
        )
    except Exception as exc:
        st.warning(f"Could not combine synthetic EHR and CGM data: {exc}")
        fused = pd.DataFrame()

    if not fused.empty and "timestamp" in fused.columns:
        fused = fused.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    return ehr, cgm, fused


@st.cache_resource
def load_synthetic_model():
    model_path = MODELS_DIR / "glucose_model.joblib"
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.warning(f"Could not load the saved synthetic model: {exc}")
        return None


def parse_timestamp_column(df, column="timestamp"):
    """Convert timestamp values safely when the requested column exists."""
    if not df.empty and column in df.columns:
        df = df.copy()
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def show_time_series(
    df,
    timestamp_col,
    value_columns,
    labels,
    title,
    y_label="Glucose (mg/dL)",
):
    """Plot the requested time-series columns if they are available."""
    if df.empty or timestamp_col not in df.columns:
        st.info("There is no time-series data to display.")
        return

    available = [
        (column, label)
        for column, label in zip(value_columns, labels)
        if column in df.columns
    ]
    if not available:
        st.info("The expected glucose columns were not found.")
        return

    plot_df = df.copy()
    plot_df[timestamp_col] = pd.to_datetime(
        plot_df[timestamp_col], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

    if plot_df.empty:
        st.info("No valid timestamps are available.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    for column, label in available:
        values = pd.to_numeric(plot_df[column], errors="coerce")
        ax.plot(
            plot_df[timestamp_col],
            values,
            label=label,
            linewidth=1.4,
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def first_existing_column(df, candidates):
    """Return the first candidate column present in a DataFrame."""
    return next((name for name in candidates if name in df.columns), None)


# ============================================================
# SIDEBAR AND HEADER
# ============================================================

st.sidebar.title("🧬 GlucoTwin AI")
st.sidebar.caption("Patient Digital Twin Prototype")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Patient Overview",
        "Glucose Forecast",
        "Model Evaluation",
        "Real CGM Experiment",
    ],
)

st.sidebar.divider()
st.sidebar.warning(
    "Research prototype only. Not for diagnosis, treatment, or medical decisions."
)

st.markdown(
    """
    <div class="hero">
        <h1>GlucoTwin AI</h1>
        <p>Patient-specific digital twin for glucose monitoring and short-horizon forecasting.</p>
        <p>Data → Patient State → Forecast → Evaluation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Synthetic and real-data experiments are displayed separately. "
    "Neither experiment establishes clinical validity."
)


# ============================================================
# LOAD DATA
# ============================================================

ehr, cgm, fused = load_synthetic_data()

synthetic_predictions = load_output_csv("test_predictions.csv")
synthetic_metrics = load_output_csv("model_metrics.csv")
synthetic_bundle = load_synthetic_model()

real_cgm = read_csv_if_exists(DATA_DIR / "real_cgm_clean.csv")
real_cgm = parse_timestamp_column(real_cgm)

real_forecast = read_csv_if_exists(DATA_DIR / "real_cgm_forecast.csv")
real_forecast = parse_timestamp_column(real_forecast)

real_baseline_metrics = load_output_csv("real_cgm_baseline_metrics.csv")
real_model_metrics = load_output_csv("real_cgm_model_metrics.csv")

# IMPORTANT: parse timestamps after reading. Do not pass parse_dates into
# load_output_csv, which intentionally accepts only a filename.
real_baseline_predictions = load_output_csv("real_cgm_baseline_predictions.csv")
real_baseline_predictions = parse_timestamp_column(real_baseline_predictions)

real_model_predictions = load_output_csv("real_cgm_model_predictions.csv")
real_model_predictions = parse_timestamp_column(real_model_predictions)


# ============================================================
# PAGE 1: PATIENT OVERVIEW
# ============================================================

if page == "Patient Overview":
    st.subheader("Patient Overview")

    if ehr.empty or cgm.empty:
        st.warning(
            "Synthetic data files were not found. Check for "
            "`data/synthetic_ehr.csv` and `data/synthetic_cgm.csv`."
        )
    elif "patient_id" not in ehr.columns or "patient_id" not in cgm.columns:
        st.error("The synthetic data files are missing the patient_id column.")
    else:
        patient_ids = sorted(ehr["patient_id"].dropna().unique().tolist())

        if not patient_ids:
            st.warning("No synthetic patient records were found.")
        else:
            selected_patient = st.sidebar.selectbox(
                "Select synthetic patient",
                patient_ids,
            )

            patient_rows = ehr.loc[ehr["patient_id"] == selected_patient]
            patient_cgm = cgm.loc[
                cgm["patient_id"] == selected_patient
            ].copy()

            if "timestamp" in patient_cgm.columns:
                patient_cgm = patient_cgm.sort_values("timestamp")

            if patient_rows.empty:
                st.warning("No EHR record exists for this patient.")
            else:
                patient = patient_rows.iloc[0]
                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Patient ID", str(selected_patient))

                if "age" in patient.index and pd.notna(patient["age"]):
                    c2.metric("Age", f"{int(patient['age'])} years")
                else:
                    c2.metric("Age", "N/A")

                if "bmi" in patient.index and pd.notna(patient["bmi"]):
                    c3.metric("BMI", f"{float(patient['bmi']):.1f}")
                else:
                    c3.metric("BMI", "N/A")

                if (
                    "hba1c_percent" in patient.index
                    and pd.notna(patient["hba1c_percent"])
                ):
                    c4.metric("HbA1c", f"{float(patient['hba1c_percent']):.1f}%")
                else:
                    c4.metric("HbA1c", "N/A")

                st.subheader("Latest Sensor Reading")

                if (
                    not patient_cgm.empty
                    and "glucose_mg_dl" in patient_cgm.columns
                ):
                    latest = patient_cgm.iloc[-1]
                    a, b, c = st.columns(3)

                    glucose_value = pd.to_numeric(
                        pd.Series([latest["glucose_mg_dl"]]), errors="coerce"
                    ).iloc[0]
                    a.metric(
                        "Glucose",
                        f"{glucose_value:.1f} mg/dL"
                        if pd.notna(glucose_value)
                        else "N/A",
                    )

                    if (
                        "timestamp" in latest.index
                        and pd.notna(latest["timestamp"])
                    ):
                        b.metric(
                            "Timestamp",
                            latest["timestamp"].strftime("%d %b %H:%M"),
                        )
                    else:
                        b.metric("Timestamp", "N/A")

                    c.metric("Total readings", f"{len(patient_cgm):,}")
                    st.subheader("Recent Glucose History")

                    if "timestamp" in patient_cgm.columns:
                        show_time_series(
                            patient_cgm.tail(288),
                            "timestamp",
                            ["glucose_mg_dl"],
                            ["Synthetic CGM"],
                            "Recent synthetic glucose readings",
                        )
                else:
                    st.info("No glucose readings are available for this patient.")

                with st.expander("View synthetic EHR record"):
                    st.dataframe(
                        patient_rows,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.caption("All EHR and CGM records on this page are synthetic.")


# ============================================================
# PAGE 2: SYNTHETIC GLUCOSE FORECAST
# ============================================================

elif page == "Glucose Forecast":
    st.subheader("Synthetic Glucose Forecast")

    if synthetic_predictions.empty:
        st.warning(
            "Synthetic predictions are missing. Run this from the project root:\n\n"
            "`python src/train_model.py`"
        )
    else:
        result = synthetic_predictions.copy()

        actual_col = first_existing_column(
            result,
            [
                "target_glucose_60m",
                "actual_glucose_60m",
                "actual",
                "y_true",
            ],
        )
        prediction_col = first_existing_column(
            result,
            [
                "predicted_glucose_60m",
                "prediction_glucose_60m",
                "prediction",
                "y_pred",
            ],
        )

        if "patient_id" in result.columns:
            patient_ids = sorted(result["patient_id"].dropna().unique().tolist())
            if patient_ids:
                selected_prediction_patient = st.selectbox(
                    "Prediction patient",
                    patient_ids,
                )
                result = result.loc[
                    result["patient_id"] == selected_prediction_patient
                ].copy()

        if actual_col is None or prediction_col is None:
            st.error(
                "The prediction CSV does not contain recognizable actual and "
                "predicted glucose columns. Check `outputs/test_predictions.csv`."
            )
            st.write("Available columns:", synthetic_predictions.columns.tolist())
            st.dataframe(
                synthetic_predictions.head(10),
                use_container_width=True,
                hide_index=True,
            )
        elif result.empty:
            st.warning("No predictions are available for the selected patient.")
        else:
            result[actual_col] = pd.to_numeric(result[actual_col], errors="coerce")
            result[prediction_col] = pd.to_numeric(
                result[prediction_col], errors="coerce"
            )
            result = result.dropna(subset=[actual_col, prediction_col])

            if result.empty:
                st.warning("No valid actual/predicted values are available.")
            else:
                result["absolute_error"] = (
                    result[actual_col] - result[prediction_col]
                ).abs()

                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction samples", f"{len(result):,}")
                c2.metric("MAE", f"{result['absolute_error'].mean():.2f} mg/dL")
                c3.metric("Forecast horizon", "60 minutes")

                if "timestamp" in result.columns:
                    result = parse_timestamp_column(result)
                    result = result.sort_values("timestamp")
                    show_time_series(
                        result,
                        "timestamp",
                        [actual_col, prediction_col],
                        ["Actual target", "Model prediction"],
                        "Synthetic test-set predictions",
                    )
                else:
                    st.warning("No timestamp column was found; showing sample order.")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(result[actual_col].to_numpy(), label="Actual target")
                    ax.plot(
                        result[prediction_col].to_numpy(),
                        label="Model prediction",
                    )
                    ax.set_xlabel("Test sample")
                    ax.set_ylabel("Glucose (mg/dL)")
                    ax.grid(alpha=0.25)
                    ax.legend()
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                st.subheader("Prediction Records")
                st.dataframe(
                    result.tail(20),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "These are saved holdout predictions, not live forecasts "
                    "of a patient's future glucose."
                )


# ============================================================
# PAGE 3: SYNTHETIC MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":
    st.subheader("Synthetic Model Evaluation")

    if synthetic_metrics.empty:
        st.warning(
            "Synthetic metrics are missing. Run:\n\n`python src/train_model.py`"
        )
    else:
        st.dataframe(
            synthetic_metrics.round(3),
            use_container_width=True,
            hide_index=True,
        )

        required = {"model", "evaluation_split", "MAE_mg_dl"}
        if required.issubset(synthetic_metrics.columns):
            test_rows = synthetic_metrics.loc[
                (synthetic_metrics["model"] == "Random Forest")
                & (synthetic_metrics["evaluation_split"] == "test")
            ]
            baseline_rows = synthetic_metrics.loc[
                synthetic_metrics["model"].astype(str).str.contains(
                    "baseline", case=False, na=False
                )
                & (
                    synthetic_metrics.get(
                        "evaluation_split", pd.Series(index=synthetic_metrics.index)
                    ).astype(str).str.lower()
                    == "test"
                )
            ]

            if not test_rows.empty:
                model_mae = float(test_rows.iloc[0]["MAE_mg_dl"])
                baseline_mae = (
                    float(baseline_rows.iloc[0]["MAE_mg_dl"])
                    if not baseline_rows.empty
                    else None
                )

                if baseline_mae is not None:
                    st.metric(
                        "Random Forest test MAE",
                        f"{model_mae:.3f} mg/dL",
                        delta=f"{model_mae - baseline_mae:+.3f} mg/dL vs baseline",
                        delta_color="inverse",
                    )
                    if model_mae < baseline_mae:
                        st.success(
                            "Random Forest has lower MAE than the baseline "
                            "on this synthetic test set."
                        )
                    else:
                        st.warning(
                            "Random Forest does not outperform the baseline "
                            "on this synthetic test set."
                        )
                else:
                    st.metric("Random Forest test MAE", f"{model_mae:.3f} mg/dL")

    st.subheader("Feature Importance")

    if synthetic_bundle is not None:
        try:
            if isinstance(synthetic_bundle, dict):
                model = synthetic_bundle.get("model")
                features = synthetic_bundle.get("features")
            else:
                model = synthetic_bundle
                features = None

            if (
                features is not None
                and model is not None
                and hasattr(model, "feature_importances_")
            ):
                importance_values = model.feature_importances_
                if len(features) == len(importance_values):
                    importance = pd.DataFrame(
                        {
                            "Feature": features,
                            "Importance": importance_values,
                        }
                    ).sort_values("Importance")

                    fig, ax = plt.subplots(figsize=(9, 5))
                    ax.barh(importance["Feature"], importance["Importance"])
                    ax.set_xlabel("Relative importance")
                    ax.set_title("Random Forest feature importance")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info(
                        "The saved model's feature names do not match its "
                        "feature importance values."
                    )
            else:
                st.info(
                    "Feature importance is unavailable in the saved model bundle."
                )
        except Exception as exc:
            st.warning(f"Could not display feature importance: {exc}")

    st.caption(
        "Synthetic-data evaluation only. These results do not establish "
        "clinical accuracy or real-patient generalization."
    )


# ============================================================
# PAGE 4: REAL CGM EXPERIMENT
# ============================================================

elif page == "Real CGM Experiment":
    st.subheader("Real CGM Experiment")
    st.write(
        "Exploratory forecasting experiment using prepared Dexcom CGM data. "
        "This experiment is kept separate from the synthetic EHR experiment."
    )
    st.warning(
        "The current real-data experiment is based on one participant's CGM "
        "readings. It does not establish generalization to other participants "
        "or clinical forecasting accuracy."
    )

    # Dataset overview and observed time series
    if not real_cgm.empty:
        if {"timestamp", "glucose_mg_dl"}.issubset(real_cgm.columns):
            real_cgm = real_cgm.sort_values("timestamp").copy()

            c1, c2, c3 = st.columns(3)
            c1.metric("CGM readings", f"{len(real_cgm):,}")

            mean_glucose = pd.to_numeric(
                real_cgm["glucose_mg_dl"], errors="coerce"
            ).mean()
            c2.metric(
                "Mean glucose",
                f"{mean_glucose:.2f} mg/dL" if pd.notna(mean_glucose) else "N/A",
            )

            start_time = real_cgm["timestamp"].min()
            end_time = real_cgm["timestamp"].max()
            if pd.notna(start_time) and pd.notna(end_time):
                c3.metric(
                    "Time span",
                    f"{start_time:%d %b} – {end_time:%d %b %Y}",
                )

            st.markdown("### Observed CGM time series")
            show_time_series(
                real_cgm,
                "timestamp",
                ["glucose_mg_dl"],
                ["Observed Dexcom CGM"],
                "Real CGM glucose readings",
            )
        else:
            st.warning(
                "The cleaned CGM file needs 'timestamp' and 'glucose_mg_dl' columns."
            )
    else:
        st.info("Cleaned real CGM data was not found at `data/real_cgm_clean.csv`.")

    # Forecasting metrics
    st.divider()
    st.markdown("### Forecasting metrics")

    metrics_parts = []
    if not real_baseline_metrics.empty:
        temp = real_baseline_metrics.copy()
        if "model" not in temp.columns:
            temp["model"] = "Persistence baseline"
        metrics_parts.append(temp)

    if not real_model_metrics.empty:
        temp = real_model_metrics.copy()
        if "model" not in temp.columns:
            temp["model"] = "Random Forest"
        metrics_parts.append(temp)

    if metrics_parts:
        combined_metrics = pd.concat(metrics_parts, ignore_index=True)
        st.dataframe(
            combined_metrics.round(3),
            use_container_width=True,
            hide_index=True,
        )

        if {"model", "MAE_mg_dl"}.issubset(combined_metrics.columns):
            comparison = combined_metrics[["model", "MAE_mg_dl"]].copy()
            comparison["MAE_mg_dl"] = pd.to_numeric(
                comparison["MAE_mg_dl"], errors="coerce"
            )
            comparison = comparison.dropna(subset=["MAE_mg_dl"])

            if not comparison.empty:
                st.markdown("#### Test-set MAE comparison")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(
                    comparison["model"].astype(str),
                    comparison["MAE_mg_dl"],
                )
                ax.set_ylabel("MAE (mg/dL) — lower is better")
                ax.set_title("Real CGM forecasting comparison")
                ax.tick_params(axis="x", rotation=10)
                ax.grid(axis="y", alpha=0.2)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                baseline_rows = comparison.loc[
                    comparison["model"].astype(str).str.contains(
                        "baseline|persistence", case=False, na=False, regex=True
                    )
                ]
                model_rows = comparison.loc[
                    comparison["model"].astype(str).str.contains(
                        "random forest", case=False, na=False
                    )
                ]

                if not baseline_rows.empty and not model_rows.empty:
                    baseline_mae = float(baseline_rows.iloc[0]["MAE_mg_dl"])
                    model_mae = float(model_rows.iloc[0]["MAE_mg_dl"])

                    st.metric(
                        "Random Forest MAE",
                        f"{model_mae:.3f} mg/dL",
                        delta=(
                            f"{model_mae - baseline_mae:+.3f} mg/dL "
                            "vs persistence baseline"
                        ),
                        delta_color="inverse",
                    )

                    if model_mae < baseline_mae:
                        st.success(
                            "Random Forest has lower MAE than the persistence "
                            "baseline in this experiment."
                        )
                    elif model_mae > baseline_mae:
                        st.warning(
                            "Random Forest does not outperform the persistence "
                            "baseline in this experiment. Do not claim that it "
                            "is the better model."
                        )
                    else:
                        st.info(
                            "Random Forest and persistence baseline have equal MAE."
                        )
    else:
        st.info(
            "Real CGM metrics were not found. Run the real-data evaluation "
            "scripts to generate the metrics CSV files."
        )

    # Saved model predictions
    st.divider()
    st.markdown("### Saved real-data predictions")

    if not real_model_predictions.empty:
        st.caption("Random Forest prediction records")
        show_time_series(
            real_model_predictions,
            "timestamp",
            [
                "actual_glucose_60m",
                "target_glucose_60m",
                "predicted_glucose_60m",
                "model_prediction",
            ],
            [
                "Actual future glucose",
                "Actual future glucose",
                "Random Forest prediction",
                "Random Forest prediction",
            ],
            "Real CGM model predictions",
        )
        st.dataframe(
            real_model_predictions.head(100),
            use_container_width=True,
            hide_index=True,
        )
    elif not real_forecast.empty:
        st.caption(
            "Prepared real-data forecast samples. These are features and future "
            "targets, not model predictions."
        )
        st.dataframe(
            real_forecast.head(100),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No real-data prediction or prepared forecast file was found.")

    if not real_baseline_predictions.empty:
        with st.expander("View persistence-baseline predictions"):
            st.dataframe(
                real_baseline_predictions.head(100),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "GlucoTwin AI | Educational proof of concept | "
    "Synthetic and real CGM experiments are separate | "
    "Not for diagnosis or treatment"
)
