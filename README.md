# 🧬 GlucoTwin AI
### Patient-Specific Glucose Forecasting Digital Twin

**GlucoTwin AI** is an educational proof of concept for exploring patient-state representation and short-horizon glucose forecasting using Continuous Glucose Monitoring (CGM) data and synthetic Electronic Health Record (EHR) data.

The project combines data preparation, time-series feature engineering, machine learning, forecasting evaluation, and an interactive Streamlit dashboard.

> **Disclaimer:** This is a research prototype, not a medical device. It is not intended for diagnosis, treatment, or clinical decision-making.

## 🎯 Problem Statement

Glucose levels can change over time, making it useful to investigate whether recent sensor measurements can help estimate future glucose levels.

GlucoTwin AI explores a digital-twin-inspired workflow that transforms historical glucose readings into patient-state features and uses machine learning to forecast glucose approximately 60 minutes ahead.

## 💡 Proposed Solution

The prototype contains two separate experiments:

1. **Synthetic patient experiment:** Uses generated CGM readings and synthetic EHR records to demonstrate patient-level data integration and forecasting.
2. **Real CGM experiment:** Uses prepared Dexcom CGM data from one participant to evaluate a persistence baseline against a Random Forest model.

The experiments are displayed separately. Synthetic EHR records are not presented as real medical records belonging to the CGM participant.

## ✨ Key Features

- Interactive Streamlit dashboard
- Synthetic patient overview with age, BMI, and HbA1c
- Historical glucose time-series visualization
- Short-horizon glucose forecasting
- Persistence baseline comparison
- Random Forest regression model
- Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) evaluation
- Feature-importance analysis
- Saved predictions and evaluation metrics
- Separate presentation of synthetic and real-data experiments

## 🏗️ System Architecture

```text
CGM Time-Series Data       Synthetic EHR Data
          |                        |
          v                        v
  Data Cleaning              Patient Records
          |                        |
          v                        |
  Time-Series Features <------------+
          |
          v
  Forecast Dataset
          |
          v
  Train / Test Evaluation
          |
          v
  Persistence Baseline vs Random Forest
          |
          v
  Forecasts, Metrics and Visualizations
          |
          v
  Streamlit Dashboard
```

The real CGM experiment is evaluated separately from the synthetic patient experiment.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Forecasting model | Random Forest Regressor |
| Visualization | Matplotlib |
| Interactive dashboard | Streamlit |
| Data format | CSV |
| Version control | Git and GitHub |

## 📁 Repository Structure

```text
GlucoTwin-AI/
├── data/
│   ├── synthetic_cgm.csv
│   └── synthetic_ehr.csv
├── models/
├── outputs/
│   ├── model_metrics.csv
│   └── test_predictions.csv
├── src/
│   ├── app.py
│   ├── generate_data.py
│   ├── train_model.py
│   ├── explore_real_cgm.py
│   ├── prepare_real_forecast.py
│   ├── evaluate_real_baseline.py
│   ├── train_real_model.py
│   └── plot_real_predictions.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MollyKhandelwal/GlucoTwin-AI.git
cd GlucoTwin-AI
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
python -m streamlit run src/app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`.

### 5. Generate synthetic data or train the synthetic model

Run these commands if you want to regenerate the synthetic experiment:

```bash
python src/generate_data.py
python src/train_model.py
```

**Note:** These commands may overwrite generated CSV files and model outputs. Run them only when you intend to regenerate the experiment.

## 📊 Real CGM Experiment

The real-data experiment evaluates a 60-minute forecasting horizon using prepared CGM readings from one participant.

The current recorded test results are:

| Model | Test MAE (mg/dL) | Test RMSE (mg/dL) |
|---|---:|---:|
| Persistence baseline | 9.431 | 14.156 |
| Random Forest | 10.266 | 14.152 |

**Interpretation:** In this experiment, the Random Forest model has a higher MAE than the persistence baseline, while its RMSE is marginally lower. It does not outperform the baseline on both metrics.

These results are exploratory and apply only to the evaluated participant and test period. They do not establish generalization to other people or clinical forecasting accuracy.

## 🧪 Evaluation Methodology

The prototype uses:

- **MAE:** Mean absolute difference between predicted and observed glucose values.
- **RMSE:** Square root of the mean squared prediction error.
- **Persistence baseline:** Uses the current glucose reading as the future estimate.
- **Random Forest:** Learns relationships between recent glucose measurements and the target glucose value.

The target horizon is approximately 60 minutes. Results depend on data quality, sampling intervals, missing values, and the evaluation split.

## 🔐 Data, Privacy and Limitations

- Synthetic EHR and CGM records are generated for demonstration.
- The real CGM experiment uses data from one participant.
- The real CGM data and synthetic EHR records are not a validated, patient-matched clinical dataset.
- A single-participant experiment cannot establish population-level performance.
- Synthetic-data performance must not be interpreted as real-world clinical accuracy.
- Data-use terms and attribution requirements must be followed for any external dataset.
- No clinical recommendations should be made from these model outputs.

## 🔮 Future Scope

- Evaluate on multiple participants and independent datasets.
- Improve time-aware validation and forecasting methodology.
- Investigate temporal models such as gradient boosting and recurrent neural networks.
- Add uncertainty estimates and prediction intervals.
- Improve monitoring of missing readings and sensor irregularities.
- Integrate validated clinical context only when suitable, authorized data is available.
- Conduct rigorous external validation before considering clinical applications.

## 👩‍💻 Author

**Molly Khandelwal**

GitHub: [@MollyKhandelwal](https://github.com/MollyKhandelwal)

## 📄 License

No license has been selected for this repository yet. Before sharing or reusing the code, add an appropriate open-source license after confirming that the project and its included data permit it.

---

*GlucoTwin AI — An educational proof of concept for exploring digital-twin-inspired glucose forecasting.*
