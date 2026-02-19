# 🌊 Flood Risk Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ML](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![DL](https://img.shields.io/badge/Deep%20Learning-DL-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A robust ML/DL framework for predicting flood probability and occurrence across India using environmental, hydrological, and infrastructural data.**

[Overview](#-overview) • [Dataset](#-dataset) • [EDA](#-exploratory-data-analysis) • [Features](#-feature-engineering) • [Models](#-models) • [Results](#-results) • [Setup](#-getting-started)

</div>

---

## 📌 Overview

Flooding is one of India's most recurring and destructive natural disasters. This project builds an end-to-end machine learning and deep learning pipeline to predict whether a flood event will occur at a given location, based on a rich set of environmental inputs: rainfall, river discharge, elevation, soil type, land cover, humidity, temperature, population density, historical flood data, and more.

The pipeline covers:
- 📊 Exploratory Data Analysis (EDA)
- 🔧 Feature Engineering & Interaction Terms
- ⚖️ Preprocessing & Standard Scaling
- 📉 Dimensionality Reduction (PCA / SVD)
- 🤖 Regression & Classification Models
- 🧠 Deep Learning
- 🗺️ Clustering & Composite Risk Scoring
- 📤 Submission Generation

---

## 📁 Repository Structure

```
Flood_Risk_Prediction/
│
├── flood_risk_dataset_india.csv   # Primary EDA dataset (10,000 samples)
├── flood.csv                      # Alternate/supplementary flood data
├── train.csv                      # Training split (11,08,895 samples)
├── test.csv                       # Test split (unlabelled, for submission)
├── sample_submission.csv          # Submission format template
├── submission.csv                 # Generated model predictions
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

This project uses two separate datasets — a smaller curated dataset for EDA and a large-scale dataset for model training and evaluation.

### 🔬 EDA Dataset — `flood_risk_dataset_india.csv`

| Property | Detail |
|---|---|
| Total samples | 10,000 |
| Flood events | 4,943 (49.4%) |
| No-flood events | 5,057 (50.6%) |
| Geographic scope | India (Lat: 8°–37°N, Lon: 68°–97°E) |
| Target variable | `Flood Occurred` (binary: 0 = No Flood, 1 = Flood) |
| Class balance | ✅ Near-perfect — no resampling needed |
| Purpose | Exploratory analysis, feature engineering, PCA |

### 🏋️ Training Dataset — `train.csv`

| Property | Detail |
|---|---|
| Total samples | **11,08,895** (~1.1 million rows) |
| Target variable | `Flood Occurred` (binary: 0 = No Flood, 1 = Flood) |
| Purpose | Full-scale model training and validation |
| Scale | Large-scale dataset enabling robust generalization |

> With over **1.1 million samples**, `train.csv` provides the volume needed to train deep learning models and ensemble methods at production scale, significantly reducing overfitting risk compared to the smaller EDA dataset.

### 🧪 Test Dataset — `test.csv`

| Property | Detail |
|---|---|
| Target variable | ❌ Not included (unlabelled) |
| Purpose | Generating predictions for `submission.csv` |
| Format | Same features as `train.csv`, no `Flood Occurred` column |

> Predictions generated on `test.csv` are saved to `submission.csv` following the format defined in `sample_submission.csv`.

---

### Input Features

| Type | Features |
|---|---|
| **Geographic** | Latitude, Longitude |
| **Meteorological** | Rainfall (mm), Temperature (°C), Humidity (%) |
| **Hydrological** | River Discharge (m³/s), Water Level (m) |
| **Terrain** | Elevation (m), Land Cover, Soil Type |
| **Societal** | Population Density, Infrastructure Score |
| **Historical** | Historical Floods count |

---

## 🔍 Exploratory Data Analysis

A comprehensive EDA was conducted on `flood_risk_dataset_india.csv` across all features:

**Class Distribution**
The target variable is nearly perfectly balanced (50.6% No Flood / 49.4% Flood), confirming that standard accuracy metrics are valid without class-weighting adjustments.

**Numeric Feature Distributions**
Histogram plots by flood class reveal heavily overlapping distributions across all 9 numeric features — indicating no single feature is a strong standalone predictor. The classification signal is inherently multivariate.

**Box Plots**
Feature medians and interquartile ranges are near-identical between flood and no-flood classes, further confirming the dataset's complexity and the need for non-linear models.

**Correlation Heatmap**
Pairwise correlations between all features are extremely low (max ~0.03), confirming minimal multicollinearity and that features contribute independently.

**Categorical Features**
Land Cover (Agricultural, Desert, Forest, Urban, Water Body) and Soil Type (Clay, Loam, Peat, Sandy, Silt) show near-uniform flood rates (~49–52%) across all categories — individual categorical features carry minimal discriminative power alone.

**Geographic Distribution**
Flood and no-flood events are uniformly scattered across India's lat/lon grid with no strong spatial clustering, ruling out simple geographic heuristics.

---

## 🔧 Feature Engineering

Six engineered features were created to capture interaction effects and non-linear flood dynamics:

| Engineered Feature | Logic | Rationale |
|---|---|---|
| `Rainfall_x_Humidity` | Rainfall × Humidity | Compounding moisture saturation risk |
| `Discharge_x_WaterLevel` | River Discharge × Water Level | Overflow and inundation risk proxy |
| `Elevation_Risk` | Inverse elevation score | Lower elevation → higher flood susceptibility |
| `High_Risk_Score` | Composite of high-risk indicators | Aggregated multi-factor risk index |
| `Soil_Land_Interaction` | Encoded Soil Type × Land Cover | Surface absorption interaction |
| `Log_Elevation` | log(Elevation + 1) | Reduces right skew in elevation distribution |

**Top predictors by Pearson correlation with target (post-engineering):**

| Rank | Feature | Direction |
|---|---|---|
| 1 | Humidity (%) | ↑ Positive |
| 2 | Elevation_Risk | ↑ Positive |
| 3 | Historical Floods | ↑ Positive |
| 4 | Rainfall_x_Humidity | ↑ Positive |
| 5 | High_Risk_Score | ↑ Positive |
| 6 | Temperature (°C) | ↓ Negative |
| 7 | Water Level (m) | ↓ Negative |

---

## ⚙️ Preprocessing

- **Standard Scaling** applied to wide-range numeric features (Rainfall, River Discharge, Elevation, Population Density) — normalized to zero mean, unit variance.
- **Label / Ordinal Encoding** applied to categorical features: Land Cover and Soil Type.
- No missing values detected. No outlier removal required.

---

## 📉 PCA / SVD Analysis

Principal Component Analysis was performed as a dimensionality exploration step:

- The **first 5 components** explain ~46% of cumulative variance.
- **10 components** reach ~77% cumulative variance — still below the 95% threshold.
- The flat scree curve confirms information is diffusely spread across many dimensions, consistent with near-zero pairwise correlations.
- PCA-reduced feature sets are explored as alternative inputs to downstream classifiers.

---

## 🤖 Models

The pipeline implemented across `train.py` and `model.py`:

### Classification
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting / XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### Regression
- Linear Regression (flood probability score)
- Ridge / Lasso Regression

### Deep Learning
- Fully Connected Neural Network (MLP) with configurable hidden layers, dropout, and batch normalization

### Unsupervised / Dimensionality Reduction
- PCA & SVD
- K-Means Clustering

### Composite Scoring
- Weighted multi-model ensemble for flood risk probability output

---

## 📈 Results

Output metrics, confusion matrices, ROC curves, and feature importance plots are saved to `flood_outputs/` after running the pipeline.

Key metrics tracked: Accuracy · Precision · Recall · F1-Score · ROC-AUC

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

### Run Training Pipeline

```bash
python train.py
```

### Run Model Evaluation

```bash
python model.py
```

All outputs and visualizations will be saved to `flood_outputs/`.

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
tensorflow
scipy
```

See `requirements.txt` for exact pinned versions.

---

## 🗺️ Future Work

- [ ] Hyperparameter tuning with `GridSearchCV` / `Optuna`
- [ ] SHAP-based feature importance and model explainability
- [ ] Temporal modeling with rolling rainfall windows and monsoon seasonality
- [ ] Geospatial visualization with Folium / Plotly choropleth maps
- [ ] REST API deployment with FastAPI or Flask
- [ ] Streamlit interactive dashboard for real-time flood risk scoring
- [ ] Integration of satellite imagery or remote sensing data

---

## 👤 Author

**BVPKARTHIKEYA**  
GitHub: [@BVPKARTHIKEYA](https://github.com/BVPKARTHIKEYA)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

Dataset represents simulated flood event data across India, incorporating real-world geographic coordinates and environmental parameters to model realistic flood-prone conditions across diverse terrain, soil types, and land cover categories.
