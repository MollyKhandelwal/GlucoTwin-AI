# 🧬 GlucoTwin AI
### Patient-Specific Glucose Forecasting Digital Twin

**GlucoTwin AI** is an educational proof of concept that explores patient-state representation and short-horizon glucose forecasting using Continuous Glucose Monitoring (CGM) data and synthetic Electronic Health Record (EHR) data.

The project combines data preparation, time-series feature engineering, machine learning, forecasting evaluation, and an interactive Streamlit dashboard.

> **Disclaimer:** GlucoTwin AI is a research prototype, not a medical device. It is not intended for diagnosis, treatment, or clinical decision-making.

## 🎯 Problem Statement

Glucose monitoring produces time-series data that can help represent a patient's recent state. This project explores whether historical glucose measurements and patient attributes can support short-horizon glucose forecasting.

## 💡 Proposed Solution

GlucoTwin AI provides an interactive prototype that:

- Displays a synthetic patient's profile and recent glucose history.
- Engineers temporal features from CGM readings.
- Forecasts glucose values 60 minutes ahead in an experimental pipeline.
- Compares model predictions against a persistence baseline.
- Presents metrics, prediction plots, and experiment results.
- Keeps synthetic-data and real-CGM experiments separate.

## ✨ Key Features

- **Patient overview:** Displays synthetic EHR attributes and sensor readings.
- **Glucose forecast:** Presents the synthetic-data forecasting workflow.
- **Model evaluation:** Shows evaluation metrics and prediction comparisons.
- **Real CGM experiment:** Explores forecasting on prepared Dexcom CGM readings.
- **Interactive dashboard:** Built using Streamlit.
- **Architecture documentation:** Describes the system design and workflow.

## 🧠 Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, Random Forest |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| Model persistence | Joblib |
| Documentation | Markdown, PDF |
| Version control | Git and GitHub |

## 🏗️ System Workflow

1. Load synthetic EHR and CGM data or prepared real CGM data.
2. Validate, clean, and prepare the time-series records.
3. Engineer features from recent glucose history.
4. Train or evaluate the forecasting model.
5. Compare predictions against a persistence baseline.
6. Display results in the Streamlit dashboard.

**Architecture document:** [`outputs/GlucoTwin_AI_Architecture.pdf`](outputs/GlucoTwin_AI_Architecture.pdf)

## 📊 Experimental Evaluation

### Synthetic-data experiment

The synthetic experiment uses generated EHR and CGM records. Its results are useful for testing the modelling pipeline, but they do not establish real-world forecasting performance.

### Real-CGM experiment

The real-data experiment uses prepared Dexcom CGM readings from one participant. In the current experiment, the Random Forest model achieved a test MAE of approximately **10.266 mg/dL**, compared with approximately **11.070 mg/dL** for the persistence baseline in the displayed evaluation.

These results are preliminary and limited to the available participant and evaluation setup. They do not establish generalization to other people or clinical accuracy.

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

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
python -m streamlit run src/app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`.

## 📁 Repository Structure

```text
GlucoTwin-AI/
├── data/
│   ├── synthetic_cgm.csv
│   └── synthetic_ehr.csv
├── models/
├── outputs/
│   ├── architecture.md
│   ├── GlucoTwin_AI_Architecture.pdf
│   ├── model_metrics.csv
│   └── test_predictions.csv
├── src/
│   ├── app.py
│   ├── generate_data.py
│   ├── train_model.py
│   ├── train_real_model.py
│   ├── prepare_real_forecast.py
│   ├── evaluate_real_baseline.py
│   ├── explore_real_cgm.py
│   ├── plot_real_predictions.py
│   └── generate_architecture_pdf.py
├── .gitignore
└── requirements.txt
```

*Note: Generated output files may vary as experiments are rerun.*

## ⚠️ Limitations

- Synthetic EHR attributes are not matched to real clinical records.
- The real-CGM experiment currently represents only one participant.
- A lower test error in this experiment does not establish clinical utility.
- The model has not been clinically validated.
- The prototype must not be used to make treatment or medication decisions.

## 🔭 Future Work

- Evaluate on more participants and independent datasets.
- Improve temporal validation and forecasting robustness.
- Investigate uncertainty estimation and prediction intervals.
- Explore interpretable patient-state representations.
- Improve model monitoring, data quality checks, and reproducibility.
- Investigate prospective validation under appropriate ethical and clinical oversight.

## 📄 License

A license has not yet been selected. Licensing will be specified after the project owner chooses an appropriate license.

## 👤 Author

**Molly Khandelwal**

GitHub: [@MollyKhandelwal](https://github.com/MollyKhandelwal)

---

*GlucoTwin AI is an educational research prototype. It is not intended for diagnosis, treatment, or clinical decision-making.*
