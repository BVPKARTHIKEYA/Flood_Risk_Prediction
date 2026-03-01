# 🌊 Flood Risk Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EC6C00?style=for-the-badge&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-2E7D32?style=for-the-badge&logo=lightgbm&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0?style=for-the-badge)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Persistence-6A1B9A?style=for-the-badge)
![tqdm](https://img.shields.io/badge/tqdm-Progress%20Bar-4CAF50?style=for-the-badge)

**A robust ML/DL framework for predicting flood probability and occurrence across India using environmental, hydrological, and infrastructural data.**

[Overview](#-overview) • [Dataset](#-dataset) • [Features](#-feature-engineering) • [Models](#-models) • [Results](#-results) • [Setup](#-getting-started) • [Future Work](#-future-work)

</div>

---

## 📌 Overview

Flooding is one of India's most recurring and destructive natural disasters. This project builds an **end-to-end machine learning and deep learning pipeline** to predict whether a flood event will occur at a given location, based on a rich set of environmental inputs.

### Pipeline Stages

| Stage | Description |
|---|---|
| 📊 EDA | Exploratory Data Analysis on 10K and 1.1M sample datasets |
| 🔧 Feature Engineering | 6 interaction and transformation features |
| ⚙️ Preprocessing | Standard Scaling, Label Encoding |
| 📉 Dimensionality Reduction | PCA & SVD analysis |
| 🤖 ML Models | 11 classical classification + 4 regression models |
| 🧠 Deep Learning | MLP, Shallow DNN, Deep DNN, Residual DNN, LSTM, 1-D CNN |
| 🗺️ Clustering | K-Means composite risk scoring |
| 📤 Submission | Test set prediction generation |

---

## 📁 Repository Structure

```
Flood_Risk_Prediction/
│
├── flood_risk_dataset_india.csv   # EDA dataset (10,000 samples)
├── flood.csv                      # Supplementary flood data
├── train.csv                      # Training split (1,108,895 samples)
├── test.csv                       # Unlabelled test split
├── sample_submission.csv          # Submission format template
├── submission.csv                 # Final model predictions
│
├── train.py                       # Training pipeline (EDA → preprocessing → models)
├── model.py                       # Model definitions and evaluation logic
│
├── flood_outputs/                 # Generated plots, metrics, and output artifacts
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## 📊 Dataset

This project uses two datasets — a smaller curated dataset for EDA and a large-scale dataset for model training.

### Dataset Summary

| Dataset | File | Samples | Target | Purpose |
|---|---|---|---|---|
| EDA Dataset | `flood_risk_dataset_india.csv` | 10,000 | `Flood Occurred` (binary) | Exploratory analysis, feature engineering, PCA |
| Training Dataset | `train.csv` | 1,108,895 | `Flood Probability` (0–1) | Full-scale model training and validation |
| Test Dataset | `test.csv` | — | ❌ Unlabelled | Generating submission predictions |

### EDA Dataset Details

- **Flood events:** 4,943 (49.4%) vs **No-flood events:** 5,057 (50.6%)
- **Geographic scope:** India (Lat: 8°–37°N, Lon: 68°–97°E)
- **Class balance:** ✅ Near-perfect — no resampling needed
- **Training scale:** 1.1M samples enable robust deep learning generalization

### Input Features

| Category | Features |
|---|---|
| **Geographic** | Latitude, Longitude |
| **Meteorological** | Rainfall (mm), Temperature (°C), Humidity (%) |
| **Hydrological** | River Discharge (m³/s), Water Level (m) |
| **Terrain** | Elevation (m), Land Cover, Soil Type |
| **Societal** | Population Density, Infrastructure Score |
| **Historical** | Historical Floods count |

---

## 🔧 Feature Engineering

Six engineered features were created to capture interaction effects and non-linear flood dynamics.

| Engineered Feature | Formula | Rationale |
|---|---|---|
| `Rainfall_x_Humidity` | Rainfall × Humidity | Compounding moisture saturation risk |
| `Discharge_x_WaterLevel` | River Discharge × Water Level | Overflow and inundation risk proxy |
| `Elevation_Risk` | Inverse elevation score | Lower elevation → higher flood susceptibility |
| `High_Risk_Score` | Composite of high-risk indicators | Aggregated multi-factor risk index |
| `Soil_Land_Interaction` | Encoded Soil Type × Land Cover | Surface absorption interaction |
| `Log_Elevation` | log(Elevation + 1) | Reduces right skew in elevation distribution |

### Top Predictors (Post-Engineering, by Pearson Correlation)

| Rank | Feature | Direction |
|---|---|---|
| 1 | Humidity (%) | ↑ Positive |
| 2 | Elevation_Risk | ↑ Positive |
| 3 | Historical Floods | ↑ Positive |
| 4 | Rainfall_x_Humidity | ↑ Positive |
| 5 | High_Risk_Score | ↑ Positive |
| 6 | Temperature (°C) | ↓ Negative |
| 7 | Water Level (m) | ↓ Negative |

> Three of the top five predictors are **engineered features**, directly validating the feature engineering strategy.

---

## ⚙️ Preprocessing

| Step | Details |
|---|---|
| **Standard Scaling** | Applied to wide-range numerics: Rainfall, River Discharge, Elevation, Population Density |
| **Label / Ordinal Encoding** | Applied to: Land Cover, Soil Type |
| **Missing Values** | None detected |
| **Outliers** | No removal required |

### PCA / SVD Analysis

| Components | Cumulative Variance |
|---|---|
| 5 components | ~46% |
| 10 components | ~77% |
| Threshold target | 95% |

> The flat scree curve confirms information is **diffusely spread** across many dimensions — aggressive PCA reduction would cause significant information loss.

---

## 🤖 Models

### Classification

| Category | Models |
|---|---|
| **Linear** | Logistic Regression, Naive Bayes |
| **Instance-Based** | KNN (k=5) |
| **Tree-Based** | Decision Tree (CART), Random Forest, Extra Trees |
| **Boosting** | AdaBoost, Gradient Boosting, XGBoost, LightGBM |
| **Kernel** | SVM (RBF) |

### Deep Learning

| Model | Architecture |
|---|---|
| MLP (sklearn) | Multi-layer Perceptron |
| Shallow DNN | 2-layer dense network |
| Deep DNN | 5+ layer dense network |
| Residual DNN | Skip-connection architecture |
| LSTM | Long Short-Term Memory |
| 1-D CNN | 1-Dimensional Convolutional Network |

### Dimensionality Reduction Pipelines

- PCA + Logistic Regression · PCA + Random Forest · PCA + XGBoost
- SVD + Logistic Regression · SVD + Random Forest · SVD + XGBoost

### Regression

- Linear Regression · XGBoost Regressor · LightGBM Regressor · Random Forest Regressor

---

## 📈 Results

> All metrics, plots, and artifacts are saved to `flood_outputs/` after running the pipeline.

### 🏆 Key Performance Summary

| Metric | Best Model | Score |
|---|---|---|
| **Composite Score** | Shallow DNN | 0.5410 |
| **F1-Score** | Shallow DNN | 0.5813 |
| **Recall** | Shallow DNN | 0.6733 |
| **Accuracy** | Logistic Regression | 0.509 |
| **ROC-AUC** | Naive Bayes / LightGBM / MLP | 0.503 |
| **R² (Regression)** | Linear Regression | ~0.84 |

### Model Ranking (Top 6 by Composite Score)

| Rank | Model | Composite Score |
|---|---|---|
| 🥇 1 | Shallow DNN | 0.5410 |
| 🥈 2 | LSTM | 0.5291 |
| 🥉 3 | MLP sklearn | 0.5203 |
| 4 | LightGBM | 0.5201 |
| 5 | Random Forest | ~0.50 |
| — | Naive Bayes (lowest) | 0.3360 |

### Key Findings

- All 23 ROC curves cluster tightly around the **random classifier diagonal (AUC ~0.47–0.51)**, indicating no model achieves meaningful discriminative ability beyond random chance.
- The **best model (Shallow DNN)** reveals an internal contradiction — highest Recall (0.673) and F1 (0.581), yet ROC-AUC of only 0.5006, achieved by over-predicting the Flood class rather than genuine discrimination.
- **Linear Regression achieves R²~0.84** on flood probability, yet this does not translate to useful binary classification due to near-zero feature-target correlations across all 20 features.
- 16 of 23 models cluster within **0.49–0.52 composite score**, confirming no architectural choice provides a decisive advantage — richer data is the path forward.

---

## 📊 Exploratory Data Analysis

### Part 1 — EDA Dataset (10,000 samples)

#### Target Variable Distribution
<img width="1560" height="520" alt="Target Variable Distribution" src="https://github.com/user-attachments/assets/39c25939-8f1b-4db3-b781-2d35c64394f9" />

Near-perfect class split of **50.6% vs 49.4%** — standard metrics are reliable with no need for resampling strategies like SMOTE.

---

#### Numeric Feature Distributions (by Flood Class)
![Numeric Feature Distributions](https://github.com/user-attachments/assets/0dc29a29-58db-412c-bf6b-eae632f316ca)

Flood and no-flood distributions **overlap almost entirely** across all 9 features — no single feature alone is sufficient for prediction. Signal is subtle and multivariate.

---

#### Box Plots — Numeric Features vs Flood Class
![Box Plots](https://github.com/user-attachments/assets/61b41af9-86a8-41e7-b653-07055ee93f01)

Medians, IQRs, and whiskers are **virtually identical** between classes — confirming a genuinely difficult classification challenge requiring models that capture high-dimensional interactions.

---

#### Correlation Heatmap
![Correlation Heatmap](https://github.com/user-attachments/assets/6abc2935-9850-4d51-a66c-1ce8f87452cd)

All pairwise correlations range only **-0.03 to +0.03**, with near-zero multicollinearity. Each feature contributes independently — ideal for ensemble and deep learning models.

---

#### Categorical Features vs Flood Occurrence
![Categorical Features](https://github.com/user-attachments/assets/74ba7947-f835-4fdb-afd8-18670c26bb58)

Flood rate hovers uniformly at **~48–52%** across all Land Cover and Soil Type categories — their value only emerges through interaction with other features.

---

#### Geographic Distribution of Flood Events
![Geographic Distribution](https://github.com/user-attachments/assets/5b01fa0f-8307-4736-9ee6-2b50fd756341)

Events are **uniformly scattered and intermixed** across India with no regional clustering — simple location-based heuristics cannot predict flood occurrence.

---

#### Before vs After Standard Scaling
![Standard Scaling](https://github.com/user-attachments/assets/6c2afa37-4429-4133-8c9b-9458f993dc1d)

Raw features span vastly different scales (Rainfall: 0–300, Elevation: 0–8,500). After scaling, all features are **re-centred to ~-1.75 to +1.75** with zero mean and unit variance.

---

#### Engineered Features — Distributions by Flood Class
![Engineered Features](https://github.com/user-attachments/assets/d6bb3e94-a4a0-401a-810b-82f3a7665380)

`Elevation_Risk`, `Log_Elevation`, and `Soil_Land_Interaction` introduce **non-linear transformations** confirmed post-preprocessing to be among the strongest flood predictors.

---

#### Feature Correlation with Target (Post-Preprocessing)
![Feature Correlation](https://github.com/user-attachments/assets/1ee85fad-6f85-4ad6-9fba-58f71a5f6fc0)

All 20 features fall within a narrow **Pearson range of -0.02 to +0.034** — engineered features occupy 3 of the top 4 positions.

---

#### PCA Analysis
![PCA Analysis](https://github.com/user-attachments/assets/bff7b1d2-c30a-4554-8cbe-c198ba5c33df)

Gradual decline from ~13% variance (PC1) to ~5% (PC10) with no sharp elbow — 10 components reach only ~77% cumulative variance, making full feature retention preferable.

---

#### Clustering Analysis
![Clustering Analysis](https://github.com/user-attachments/assets/50626a16-2d8b-405f-b83b-279ef350fc39)

Silhouette scores remain consistently low (best: 0.095 at k=2) with no distinct elbow — **data does not form naturally well-separated clusters**, confirming supervised learning is essential.

---

#### K-Means Flood Risk Zones (Geographic)
![K-Means Geographic](https://github.com/user-attachments/assets/51e9d251-749a-44ab-8f4a-ff11cdc510da)

Both clusters show near-equal flood rates of **50.65% and 50.44%** — K-Means failed to identify meaningful geographic flood risk zones.

---

#### Test-Set Metrics — All 23 Models
![Test-Set Metrics](https://github.com/user-attachments/assets/adefd542-9461-46cd-b913-7b4997918bb2)

ROC-AUC and PR-AUC hover in the **0.46–0.51 range** across all 23 models — a genuinely hard classification problem where no model significantly outperforms random chance.

---

#### ROC Curves — All Models
![ROC Curves](https://github.com/user-attachments/assets/5b4cd0ea-937e-4ac0-963e-167ffaf110ba)

All 23 curves huddle tightly around the random classifier diagonal (**AUC: 0.474–0.503**). This is a definitive diagnostic signal that the current feature set lacks sufficient separable information.

---

#### Precision-Recall Curves — All Models
![PR Curves](https://github.com/user-attachments/assets/408b1467-29de-4d6c-83ac-b94f951d1903)

All models show a sharp precision spike near zero recall then collapse to the ~0.50 baseline (**PR-AUC: 0.487–0.511**). Meaningful improvement requires fundamentally richer input data.

---

#### Confusion Matrices — Top 6 Models (by F1)
![Confusion Matrices](https://github.com/user-attachments/assets/7e118ab0-1316-4dcd-b72b-61d1d04a731a)

All top models exhibit a consistent **"Flood" prediction bias** — true positives (437–511) substantially outweigh true negatives (253–322), with high false positive counts across every model.

---

#### Deep Learning Training History
![DL Training History](https://github.com/user-attachments/assets/86228081-3bc9-4da3-8bae-4c1c1f51e362)

Residual DNN and 1-D CNN show the most pronounced overfitting — training AUC climbs to 0.65+ while **validation AUC plateaus near 0.52** across all architectures, confirming a data quality constraint.

---

#### Feature Importances — Ensemble Models
![Feature Importances](https://github.com/user-attachments/assets/d62d1908-5364-454a-b27c-a0e39750eee4)

Strikingly different rankings across Random Forest, XGBoost, and LightGBM — no single feature is universally dominant. Engineered features consistently rank in mid-to-upper importance tiers.

---

#### Model × Metric Heatmap
![Model Metric Heatmap](https://github.com/user-attachments/assets/d6e0d826-1c39-402c-ab9a-02e414306c85)

A **uniformly dark blue band (~0.49–0.51)** spans Accuracy, Precision, ROC-AUC, and PR-AUC across virtually all 23 models. The Recall and F1 rows show the greatest variance.

---

#### Best Model Deep Analysis — Shallow DNN
![Best Model Analysis](https://github.com/user-attachments/assets/0d876a69-781a-44ae-842f-bfeafd0c5f18)

Shallow DNN leads with composite score **0.5306**, but ROC-AUC (0.5006) barely above random — it maximises flood detection via class over-prediction rather than genuine discrimination.

---

#### All Models — Composite Score Ranking
![Composite Score Ranking](https://github.com/user-attachments/assets/aefe2863-e6b6-4e8a-a36d-68562557f9ee)

Shallow DNN (0.5410) leads all 23 models by a clear margin. **16 of 23 models cluster within 0.49–0.52** — confirming no architectural choice provides a decisive advantage.

---

### Part 2 — Training Dataset EDA (1,108,895 samples)

#### Flood Probability Distribution
![Flood Probability Distribution](https://github.com/user-attachments/assets/63ea2b79-dc42-439f-8b5b-32cf3a9494c1)

Near-perfect bell curve **tightly centred around 0.50** (range: 0.40–0.60). This is a scientifically honest reflection of the dataset's near-zero feature-target correlations, not a model error.

---

#### Actual vs Predicted
![Actual vs Predicted](https://github.com/user-attachments/assets/ae7e9b86-7751-42ba-b93b-5685f22f9f8f)

Predictions cluster in a **compressed band of 0.30–0.75** regardless of actual values — the model systematically pulls predictions toward the centre due to insufficient discriminative signal.

---

#### Residual Distribution
![Residual Distribution](https://github.com/user-attachments/assets/e53dca68-86f3-4de2-a8b2-d1bdd692ed34)

Sharply peaked, **right-skewed distribution centred near zero** — the model more frequently under-predicts flood probability, with non-random structured errors indicating missed high-probability subsets.

---

#### Prediction Error Plot
![Prediction Error Plot](https://github.com/user-attachments/assets/8d2459da-b64f-4ddb-8dc4-b435827e9f73)

**Fan-shaped (heteroscedastic) pattern** — errors tightest at low predictions (±0.02), widening to ±0.10 near 0.50–0.60 where most samples concentrate.

---

#### Regression Model Performance
![Model Performance Comparison](https://github.com/user-attachments/assets/219c44ac-32b3-47e2-a1dd-547f48b896ae)

| Model | R² | RMSE |
|---|---|---|
| Linear Regression | ~0.84 | ~0.02 |
| XGBoost | ~0.81 | ~0.02 |
| LightGBM | ~0.76 | ~0.03 |
| Random Forest | ~0.65 | ~0.03 |

> ⚠️ High R² does **not** translate to useful binary classification — fitting the probability distribution centred at 0.50 provides no real discriminative power near the decision boundary.

---

## 📤 Submission

`submission.csv` contains flood probability predictions from the best model (Shallow DNN) on the unlabelled `test.csv`, formatted per `sample_submission.csv`.

### File Format

```
id,FloodProbability
0,0.5012
1,0.4987
2,0.5134
...
```

### Generation Pipeline

```
test.csv
  └── Preprocessing (Scaling + Encoding + Feature Engineering)
        └── Shallow DNN → .predict_proba()
              └── submission.csv
```

### Risk Interpretation

| FloodProbability | Risk Level | Suggested Action |
|---|---|---|
| 0.00 – 0.35 | 🟢 Low | No immediate action required |
| 0.35 – 0.50 | 🟡 Below-average | Monitor conditions |
| 0.50 – 0.65 | 🟠 Above-average | Issue precautionary advisories |
| 0.65 – 1.00 | 🔴 High | Activate early warning systems |

> ⚠️ **Important:** Given ROC-AUC ~0.50, these thresholds are **preliminary indicators only** and should not be relied on for life-safety decisions without richer, real-time data sources.

---

## 🔬 LLM Evaluations

<img width="1202" height="655" alt="LLM Evaluations" src="https://github.com/user-attachments/assets/2e0711a7-4807-4664-b896-33e118b79a91" />

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/BVPKARTHIKEYA/Flood_Risk_Prediction.git
cd Flood_Risk_Prediction

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Run full training pipeline (EDA → preprocessing → models)
python train.py

# Run model evaluation and generate submission
python model.py
```

All outputs and visualizations are saved to `flood_outputs/`.

### Dependencies

```
pandas       numpy        scikit-learn
matplotlib   seaborn      xgboost
tensorflow   scipy        lightgbm
```

See `requirements.txt` for exact pinned versions.

---

## 🗺️ Future Work

### Data Enrichment
- [ ] Temporal features — rolling 7-day and 30-day rainfall accumulation windows
- [ ] Satellite-derived indices — NDVI, soil moisture from remote sensing imagery
- [ ] Real-time river gauge readings — hourly/daily discharge measurements
- [ ] Topographic wetness index — derived from Digital Elevation Model (DEM) data
- [ ] Monsoon seasonality flags — month-of-year encoding, IMD rainfall zone labels

### Modelling Improvements
- [ ] Hyperparameter tuning with `GridSearchCV` / `Optuna`
- [ ] SHAP-based feature importance and model explainability
- [ ] K-Medoids clustering as alternative to K-Means
- [ ] PCA/SVD + supervised hybrid pipelines

### Deployment
- [ ] REST API with FastAPI or Flask
- [ ] Streamlit interactive dashboard for real-time flood risk scoring
- [ ] Geospatial visualization with Folium / Plotly choropleth maps

---

## 👤 Author

**BVPKARTHIKEYA**
GitHub: [@BVPKARTHIKEYA](https://github.com/BVPKARTHIKEYA)

---

## 🙏 Acknowledgements

Dataset represents simulated flood event data across India, incorporating real-world geographic coordinates and environmental parameters to model realistic flood-prone conditions across diverse terrain, soil types, and land cover categories.
