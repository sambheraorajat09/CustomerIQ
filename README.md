# CustomerIQ — AI-Powered Customer Segmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/pytest-20%20passed-brightgreen.svg)](https://pytest.org)

An enterprise-grade, unsupervised machine learning application for e-commerce customer segmentation based on **Recency, Frequency, and Monetary (RFM)** analytics. Built with **K-Means Clustering**, compared against **Agglomerative Hierarchical Clustering**, and delivered through an interactive **Streamlit** diagnostic dashboard.

---

## 🌟 Key Features

1. **Multi-Format Ingestion**: Seamlessly load retail transaction logs in `.csv` and `.xlsx` / `.xls` formats.
2. **Automated Data Sanitization**: Robust handling of missing customer IDs, duplicate transactions, cancellation orders (`C` prefix), and non-positive quantities or prices.
3. **RFM Feature Engineering**: Computes customer-level Recency (days since last purchase), Frequency (distinct invoice events), Monetary (total expenditure), Average Order Value (AOV), and customer tenure.
4. **Variance Stabilization & Scaling**: Addresses extreme Pareto skewness via `log1p` or `Yeo-Johnson` transformations followed by `StandardScaler` to prevent centroid distortion in Euclidean space.
5. **K-Means Clustering**: Configurable cluster counts ($k=2$ to $8$) with `k-means++` initialization.
6. **Mathematical Optimization**: Interactive Elbow Method (Inertia / WCSS) and Silhouette Score analysis across candidate $k$ values.
7. **Algorithm Benchmarking**: Direct quantitative comparison between K-Means and Agglomerative Hierarchical Clustering across Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, and runtime latency.
8. **Evidence-Based Interpretation**: Dynamic, empirical segment profiling (Champions, Loyal Regulars, New Buyers, At-Risk High Spenders, Hibernating) based strictly on rank percentiles without fabricated demographic stereotypes.
9. **Cautious Marketing Recommendations**: Actionable playbooks with retention tactics, pricing strategies, operational risks, and controlled A/B test hypotheses.
10. **Interactive Streamlit Dashboard**: Includes 3D Plotly RFM scatter plots, 2D PCA cluster projections, normalized centroid radar charts, and customer-level cohort filtering.
11. **One-Click CSV Export**: Filter by segment, search specific Customer IDs, and export labeled cohorts.
12. **Synthetic Demonstration Mode**: Integrated high-fidelity data generator for offline evaluation, clearly badged in the UI.
13. **Safe Model Persistence**: End-to-end `CustomerIQPipeline` serialization and restoration with `joblib`.

---

## 🏗️ Project Architecture & Folder Structure

```
D:\CustomerIQ
│   app.py                           # Main Streamlit web application
│   requirements.txt                 # Project dependencies
│   README.md                        # Project documentation & execution guide
│   .gitignore                       # Git ignore specifications
│
├───.streamlit
│       config.toml                  # Streamlit visual theme and server configuration
│
├───data
│       README.md                    # Dataset source documentation & hygiene rules
│       Online Retail.xlsx           # UCI Online Retail raw Excel dataset
│       Online_Retail.csv            # Pre-converted high-speed CSV dataset
│
├───models
│       .gitkeep                     # Model directory marker
│       customeriq_pipeline.joblib   # Trained pipeline artifact (scaler + model + metadata)
│
├───src
│       __init__.py                  # Package initialization
│       data_loader.py               # Ingestion, schema validation & synthetic generator
│       preprocessing.py             # Data sanitization, cancellations & audit metrics
│       feature_engineering.py       # RFM calculation, skewness transforms & scaling
│       clustering.py                # K-Means and Hierarchical clustering implementations
│       evaluation.py                # Clustering metrics (Silhouette, DB, CH, benchmark)
│       segment_interpretation.py    # Evidence-based profiling & dynamic segment naming
│       recommendations.py           # Strategic playbooks & A/B test hypotheses
│       visualization.py             # Plotly 3D, PCA 2D, radar & distribution charts
│       inference.py                 # Pipeline serialization, loading & out-of-sample prediction
│
├───tests
│       test_data_loader.py          # Unit tests for loading and schema validation
│       test_feature_engineering.py  # Unit tests for data cleaning & RFM transformations
│       test_clustering.py           # Unit tests for K-Means, Hierarchical & metrics
│       test_inference.py            # Unit tests for pipeline persistence & prediction
│
└───reports
        README.md                    # Reports repository overview
        executive_report.md          # Full executive & technical ML evaluation report
        viva_questions_and_answers.md# Comprehensive viva preparation & theory guide
```

---

## 🔬 Methodological Disclosures & Scientific Guardrails

- **No Demographic Assumptions**: The UCI Online Retail dataset contains transaction records only. CustomerIQ does not fabricate customer age, gender, salary, or ethnicity.
- **Correlation vs Causality**: Clustering identifies geometric similarity in historical purchase logs. It does not establish causal mechanisms of loyalty or guarantee future customer retention.
- **Seasonality & Window Truncation**: Metrics are calculated over an approximate 12-month window. Inactive customers may reflect seasonal gift-buying cycles rather than permanent defection.
- **Generalizability**: Developed on wholesale/retail giftware transactions; application to subscription or grocery models requires custom recency windows and spend thresholds.

---

## 💻 Windows 11 Installation & Execution Guide

CustomerIQ is optimized for Windows 11 and an 8 GB RAM laptop. Open **PowerShell** or **Command Prompt** and execute the commands below from `D:\CustomerIQ`:

### Step 1: Navigate to Project Directory
```powershell
cd D:\CustomerIQ
```

### Step 2: Create Python Virtual Environment
```powershell
python -m venv .venv
```

### Step 3: Install Required Dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 4: Execute Automated Unit Tests
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
*(All 20 unit tests will run and pass with zero warnings)*

### Step 5: Launch the Streamlit Dashboard
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
*(The dashboard will start and open automatically in your browser at `http://localhost:8501`)*

---

## 📊 Dataset Information

- **Name**: UCI Online Retail
- **Source**: UCI Machine Learning Repository
- **URL**: [https://archive.ics.uci.edu/dataset/352/online+retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Timeframe**: 01/12/2010 – 09/12/2011
- **Transactions**: 541,909 raw records | 4,338 unique customers after cleaning.

---

## 🧪 Automated Testing Summary

CustomerIQ includes comprehensive automated tests covering all modules:
- `tests/test_data_loader.py`: Validates schema conformance, case-insensitivity, synthetic generator integrity, and missing column detection.
- `tests/test_feature_engineering.py`: Verifies cancellation filtering, negative spend exclusion, RFM aggregation logic, and skewness transformers.
- `tests/test_clustering.py`: Tests K-Means and Hierarchical clustering fit/predict, invalid parameter bounds ($k < 2$), and metric calculations.
- `tests/test_inference.py`: Verifies pipeline serialization, artifact restoration, prediction consistency, and recommendation mapping.

---

## 📜 License
This project is open-source and licensed under the MIT License.
