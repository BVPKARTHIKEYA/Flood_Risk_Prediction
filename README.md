# 🌊 Flood Risk Prediction
<img width="1202" height="655" alt="image" src="https://github.com/user-attachments/assets/2e0711a7-4807-4664-b896-33e118b79a91" />

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

[Overview](#-overview) • [Dataset](#-dataset) • [EDA](#-exploratory-data-analysis) • [Feature Engineering](#-feature-engineering) • [Preprocessing](#️-preprocessing) • [Models](#-models) • [Results](#-results--model-evaluation) • [Setup](#-getting-started)

</div>

---

## 📌 Overview

Flooding is one of India's most recurring and destructive natural disasters. This project builds an end-to-end machine learning and deep learning pipeline to predict whether a flood event will occur at a given location, based on a rich set of environmental inputs: rainfall, river discharge, elevation, soil type, land cover, humidity, temperature, population density, historical flood data, and more.

**The pipeline covers:**
- 📊 Exploratory Data Analysis (EDA)
- 🔧 Feature Engineering & Interaction Terms
- ⚖️ Preprocessing & Standard Scaling
- 📉 Dimensionality Reduction (PCA / SVD)
- 🤖 Regression & Classification Models
- 🧠 Deep Learning (MLP, Shallow DNN, Deep DNN, Residual DNN, LSTM, 1-D CNN)
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
| Target variable | `Flood Probability` (continuous, 0–1) |
| Purpose | Full-scale model training and validation |
| Scale | Large-scale dataset enabling robust generalization |

> With over **1.1 million samples**, `train.csv` provides the volume needed to train deep learning models and ensemble methods at production scale, significantly reducing overfitting risk compared to the smaller EDA dataset.

### 🧪 Test Dataset — `test.csv`

| Property | Detail |
|---|---|
| Target variable | ❌ Not included (unlabelled) |
| Purpose | Generating predictions for `submission.csv` |
| Format | Same features as `train.csv`, no target column |

> Predictions generated on `test.csv` are saved to `submission.csv` following the format defined in `sample_submission.csv`.

### 🗂️ Input Features

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

### Part 1 — EDA on `flood_risk_dataset_india.csv` (10,000 samples)

---

### 📊 Target Variable Distribution — Flood Occurred
> *Bar chart (Class Count) and Pie chart (Class Proportion) of the target variable.*

<img width="1560" height="520" alt="Target Variable Distribution" src="https://github.com/user-attachments/assets/39c25939-8f1b-4db3-b781-2d35c64394f9" />

- The dataset contains **5,057 No Flood** and **4,943 Flood** samples, totalling 10,000 observations with a near-perfect class split of **50.6% vs 49.4%** — confirming the dataset is well-balanced with no need for oversampling or undersampling techniques.
- The class proportion pie chart visually reinforces the balance, indicating that **standard evaluation metrics such as accuracy are reliable** without requiring class-weighted adjustments or resampling strategies like SMOTE.

---

### 📊 Numeric Feature Distributions (by Flood Class)
> *3×3 grid of overlapping histograms comparing density distributions of 9 numeric features between Flood (red) and No Flood (blue) classes.*

![Numeric Feature Distributions](https://github.com/user-attachments/assets/0dc29a29-58db-412c-bf6b-eae632f316ca)

- Across all 9 features — Latitude, Longitude, Rainfall, Temperature, Humidity, River Discharge, Water Level, Elevation, and Population Density — the **flood and no-flood distributions overlap almost entirely**, with no feature showing a clear separation between classes, indicating that no single numeric feature alone is sufficient to predict flood occurrence.
- The near-identical density curves confirm that the dataset's predictive signal is **subtle and multivariate in nature**, making it well-suited for ensemble methods and deep learning models that can capture complex feature interactions.

---

### 📊 Box Plots — Numeric Features vs Flood Class
> *3×3 grid of side-by-side box plots comparing the spread, median, and quartiles of 9 numeric features between Flood (red) and No Flood (blue) classes.*

![Box Plots](https://github.com/user-attachments/assets/61b41af9-86a8-41e7-b653-07055ee93f01)

- Across all 9 features, the **medians, interquartile ranges, and whisker extents are virtually identical** between the Flood and No Flood classes, statistically confirming that no individual feature produces a meaningful distributional shift between the two classes.
- The uniformity reinforces that the dataset presents a **genuinely difficult classification challenge**, where flood prediction cannot rely on simple univariate thresholds and instead demands models capable of learning subtle, high-dimensional feature interactions.

---

### 📊 Correlation Heatmap
> *Lower-triangular heatmap showing pairwise Pearson correlation coefficients across all 12 features, color-coded from deep blue (negative) to deep red (positive).*

![Correlation Heatmap](https://github.com/user-attachments/assets/6abc2935-9850-4d51-a66c-1ce8f87452cd)

- All pairwise correlations are **extremely weak, ranging only from -0.03 to +0.03**, with the strongest signals being Humidity (%) vs Flood Occurred (+0.03) and Historical Floods vs Water Level (-0.03) — confirming near-zero linear relationships and virtually no multicollinearity.
- The absence of strongly correlated feature pairs indicates that **each feature contributes independently**, making this an ideal setup for ensemble and deep learning models that can exploit weak, distributed signals across all features simultaneously.

---

### 📊 Categorical Features vs Flood Occurrence
> *Side-by-side grouped bar charts showing flood rate (%) for No Flood (blue) and Flood (red) across all categories of Land Cover (left) and Soil Type (right).*

![Categorical Features](https://github.com/user-attachments/assets/74ba7947-f835-4fdb-afd8-18670c26bb58)

- Across all 5 Land Cover types (Agricultural, Desert, Forest, Urban, Water Body) and all 5 Soil Types (Clay, Loam, Peat, Sandy, Silt), the flood rate hovers **uniformly between ~48–52%** with no single category showing a meaningfully higher or lower probability.
- The near-identical bar heights confirm that **Land Cover and Soil Type carry negligible standalone discriminative power**, and their value likely only emerges through interaction with other features — motivating the engineered `Soil_Land_Interaction` feature.

---

### 📊 Geographic Distribution of Flood Events in India
> *Scatter plot mapping 10,000 data points across India's geographic coordinates (Latitude: 8°–37°N, Longitude: 68°–97°E), with No Flood as blue circles and Flood as red triangles.*

![Geographic Distribution](https://github.com/user-attachments/assets/5b01fa0f-8307-4736-9ee6-2b50fd756341)

- Flood and No Flood events are **scattered uniformly and intermixed across the entire Indian subcontinent** with no discernible regional clustering, hotspots, or geographic boundaries separating the two classes.
- The random spatial distribution rules out simple **location-based heuristics**, reinforcing that the model must rely on environmental and hydrological feature combinations rather than geographic proximity.

---

### 📊 Before vs After Standard Scaling
> *2×4 grid of histograms comparing raw (blue, top row) and standard-scaled (green, bottom row) distributions of Rainfall (mm), River Discharge (m³/s), Elevation (m), and Population Density.*

![Standard Scaling](https://github.com/user-attachments/assets/6c2afa37-4429-4133-8c9b-9458f993dc1d)

- The top row shows raw feature distributions spanning **vastly different scales** — Rainfall (0–300), River Discharge (0–5,000), Elevation (0–8,500), and Population Density (0–10,000) — which would bias distance-based and gradient-based models if left unscaled.
- After Standard Scaling (bottom row), all four features are **re-centred to approximately -1.75 to +1.75** with zero mean and unit variance, preserving the original distribution shape and ensuring every feature contributes equally during training.

---

### 📊 Engineered Features — Distributions by Flood Class
> *2×3 grid of histograms showing distributions of 6 engineered features split by Flood (red) and No Flood (blue) classes.*

![Engineered Features](https://github.com/user-attachments/assets/d6bb3e94-a4a0-401a-810b-82f3a7665380)

- `Rainfall_x_Humidity` and `Discharge_x_WaterLevel` display **right-skewed distributions with heavy class overlap**, while `Elevation_Risk` reveals a stark spike near zero and `High_Risk_Score` shows discrete values at 0–3 — each feature captures a **distinctly different aspect of flood risk** that raw features alone could not express.
- The engineered features — particularly `Elevation_Risk`, `Log_Elevation`, and `Soil_Land_Interaction` — introduce **non-linear transformations and interaction signals** confirmed by post-preprocessing correlation analysis to be among the strongest predictors of flood occurrence.

---

### 📊 Feature Correlation with Target (Post-Preprocessing)
> *Horizontal bar chart ranking all 20 features by Pearson correlation with `Flood Occurred`, with positive correlations in green and negative in red.*

![Feature Correlation](https://github.com/user-attachments/assets/1ee85fad-6f85-4ad6-9fba-58f71a5f6fc0)

- The top positively correlated features are **Humidity (~0.034), Elevation_Risk (~0.021), Historical Floods (~0.017), and Rainfall_x_Humidity (~0.012)** — notably, three of the top four are engineered features, directly validating the feature engineering strategy.
- All 20 features remain within a narrow **Pearson range of -0.02 to +0.034**, with a clear directional split: elevation, temperature, and water level negatively correlated; humidity, historical floods, and risk scores positively correlated.

---

### 📊 PCA Analysis
> *Side-by-side charts: Individual Explained Variance per Component (blue bar chart, left) and Cumulative Explained Variance (red line chart, right) across 10 principal components, with a dashed 95% threshold line.*

![PCA Analysis](https://github.com/user-attachments/assets/bff7b1d2-c30a-4554-8cbe-c198ba5c33df)

- The scree plot shows a **gradual decline from ~13% variance for PC1 to ~5% for PC10** with no sharp elbow — information is diffusely spread across all components rather than dominated by a few principal axes.
- With 10 components reaching only **~77% cumulative variance** (well below the 95% threshold), aggressive dimensionality reduction via PCA would result in significant information loss, making retention of the full feature set preferable.

---

### 📊 Clustering Analysis
> *Three-panel figure: K-Means Elbow Curve (left), Silhouette Scores for k=2–6 (centre), and PCA 2-D projection of K-Means clusters at k=2 (right).*

![Clustering Analysis](https://github.com/user-attachments/assets/50626a16-2d8b-405f-b83b-279ef350fc39)

- The elbow curve shows a **steady linear inertia decrease** from ~182,000 at k=2 to ~150,000 at k=6 with no distinct elbow, while silhouette scores remain consistently low (best: 0.095 at k=2) — the data does not form naturally well-separated clusters.
- The PCA 2-D projection reveals **heavily overlapping Cluster 0 (blue) and Cluster 1 (brown)** with no clear geometric boundary, confirming that unsupervised clustering cannot meaningfully partition flood events and that supervised learning is essential.

---

### 📊 K-Means Flood Risk Zones (Geographic)
> *Side-by-side geographic scatter plots of K-Means Cluster 0 (blue, left) and Cluster 1 (brown, right) across India's lat/lon grid, with each cluster's flood rate as subtitle.*

![K-Means Geographic](https://github.com/user-attachments/assets/51e9d251-749a-44ab-8f4a-ff11cdc510da)

- Both clusters are **uniformly scattered across the entire subcontinent** with virtually identical flood rates of **50.65% and 50.44%** — confirming K-Means has failed to identify any meaningful geographic flood risk zones.
- The near-equal flood rates (differing by only 0.21%) provide definitive evidence that **geographic coordinates alone cannot delineate flood risk**, requiring the full suite of environmental and hydrological features via supervised learning.

---

### 📊 Test-Set Metrics — All Models
> *2×3 grid of horizontal bar charts comparing 23 models across Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC on the test set.*

![Test-Set Metrics](https://github.com/user-attachments/assets/adefd542-9461-46cd-b913-7b4997918bb2)

- **Logistic Regression emerges as the top performer** across most metrics (Accuracy ~0.509, Precision ~0.508, Recall ~0.566, F1 ~0.537), while deep learning models show inconsistent results with notably low Recall scores (as low as 0.073 for LSTM).
- ROC-AUC and PR-AUC hover in the **0.46–0.51 range across all 23 models**, confirming the dataset presents a genuinely hard classification problem where no model significantly outperforms random chance.

---

### 📊 ROC Curves — All Models
> *Overlapping ROC curves for all 23 models against the random classifier baseline (black dashed diagonal), with each model's AUC in the legend.*

![ROC Curves](https://github.com/user-attachments/assets/5b4cd0ea-937e-4ac0-963e-167ffaf110ba)

- All 23 ROC curves **huddle tightly around the random classifier diagonal** with AUC scores ranging narrowly from 0.474 (Residual DNN) to 0.503 (Naive Bayes, LightGBM, MLP) — no model achieves meaningful discriminative ability beyond random chance.
- The near-complete overlap with the diagonal is a **definitive diagnostic signal** that the current feature set lacks sufficient separable information, strongly motivating richer data sources such as temporal rainfall sequences or satellite imagery.

---

### 📊 Precision-Recall Curves — All Models
> *Overlapping PR curves for all 23 models against the random baseline (black dashed at ~0.50), with each model's PR-AUC in the legend.*

![PR Curves](https://github.com/user-attachments/assets/408b1467-29de-4d6c-83ac-b94f951d1903)

- All 23 models show a **sharp precision spike near zero recall** followed by immediate collapse to the ~0.50 baseline, with PR-AUC scores from 0.487 (Residual DNN) to 0.511 (Gradient Boosting) — models cannot maintain any precision advantage as recall increases.
- The rapid convergence of all curves onto the baseline is a **final definitive confirmation** that meaningful improvement requires fundamentally richer input data rather than further model tuning.

---

### 📊 Confusion Matrices — Top 6 Models (by F1)
> *2×3 grid of confusion matrices for: Shallow DNN (F1=0.5813), LSTM (0.5726), PCA + Logistic Regression (0.5458), SVD + Logistic Regression (0.5458), Random Forest (0.5435), MLP sklearn (0.5402).*

![Confusion Matrices](https://github.com/user-attachments/assets/7e118ab0-1316-4dcd-b72b-61d1d04a731a)

- All 6 top models reveal a consistent **"Flood" prediction bias** — true positive counts (437–511) substantially outweigh true negatives (253–322), indicating models default to predicting flood events rather than learning genuine class boundaries.
- High false positive counts (440–500) across every model confirm that **none has learned a reliably discriminative decision boundary**, with the best performer Shallow DNN (F1=0.5813) only marginally outperforming random guessing.

---

### 📊 Deep Learning Training History
> *5×2 grid of Loss and AUC training curves for Shallow DNN, Deep DNN, Residual DNN, LSTM, and 1-D CNN — training (blue) vs validation (orange dashed) across 15 epochs.*

![DL Training History](https://github.com/user-attachments/assets/86228081-3bc9-4da3-8bae-4c1c1f51e362)

- **Residual DNN and 1-D CNN display the most pronounced overfitting** — training AUC climbs steeply to 0.65+ while validation AUC plateaus near 0.52, and training loss drops while validation loss diverges upward.
- Across all 5 models, **validation AUC remains flat and noisy in the 0.50–0.53 range** regardless of architecture depth or type — confirming the performance ceiling is a data quality constraint, not a modelling one.

---

### 📊 Feature Importances — Ensemble Models
> *Side-by-side horizontal bar charts ranking all 20 features by importance across Random Forest (left), XGBoost (centre), and LightGBM (right).*

![Feature Importances](https://github.com/user-attachments/assets/d62d1908-5364-454a-b27c-a0e39750eec4)

- The three ensemble models assign **strikingly different importance rankings** — Random Forest prioritises Temperature and Longitude; XGBoost ranks Latitude and `Log_PopDensity` highest; LightGBM favours Latitude and Longitude — no single feature is universally dominant.
- Engineered features `Discharge_x_WaterLevel`, `Rainfall_x_Humidity`, and `Elevation_Risk` appear **consistently in mid-to-upper importance tiers** across all three models, validating that continuous interaction features contribute more extractable signal than discrete composite scores.

---

### 📊 Model × Metric Heatmap (Test Set)
> *Color-coded heatmap comparing all 23 models (columns) across 6 metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC (rows). Darker blue = higher score, yellow-green = outlier lows.*

![Model Metric Heatmap](https://github.com/user-attachments/assets/d6e0d826-1c39-402c-ab9a-02e414306c85)

- A **remarkably uniform band of dark blue (~0.49–0.51)** spans Accuracy, Precision, ROC-AUC, and PR-AUC for virtually all 23 models — the performance plateau is dataset-wide, not model-specific. Notable outliers are Naive Bayes and Deep DNN on Recall and F1 (yellow-green at 0.018 and 0.275).
- The **Recall and F1 rows show the greatest variance**, with Shallow DNN achieving the highest Recall (0.673) and F1 (0.581), suggesting optimising for F1 rather than accuracy should guide final model selection.

---

### 📊 Best Model Deep Analysis — Shallow DNN
> *Five-panel dashboard: Confusion Matrix, ROC Curve (AUC=0.5006), PR Curve (PR-AUC=0.5028), All Metrics bar chart, and Model Ranking by Composite Weighted Score.*

![Best Model Analysis](https://github.com/user-attachments/assets/0d876a69-781a-44ae-842f-bfeafd0c5f18)

- The Shallow DNN reveals a **strong internal contradiction** — highest Recall (0.6733) and F1 (0.5813) among all models, yet ROC-AUC (0.5006) and PR-AUC (0.5028) barely above random, with 488 false positives against only 253 true negatives — it maximises flood detection by over-predicting the Flood class rather than genuinely discriminating.
- The composite score ranking places **Shallow DNN (0.5306) as the clear best model**, followed by LSTM (0.5291), MLP sklearn (0.5203), and LightGBM (0.5201), while Naive Bayes (0.3360), Deep DNN (0.4405), and 1-D CNN (0.4410) occupy the bottom.

---

### 📊 All Models — Composite Score Ranking
> *Vertical bar chart ranking all 23 models by Composite Weighted Score — Shallow DNN in gold (best), others in blue, red dashed mean line at 0.4947.*

![Composite Score Ranking](https://github.com/user-attachments/assets/aefe2863-e6b6-4e8a-a36d-68562557f9ee)

- **Shallow DNN (0.5410) leads all 23 models** by a clear margin, followed by LSTM (0.5291), MLP sklearn (0.5203), and LightGBM (0.5201), while Naive Bayes (0.3360), Deep DNN (0.4405), and 1-D CNN (0.4410) rank lowest.
- The **tight clustering of 16/23 models within 0.49–0.52** around the mean confirms no architectural choice provides a decisive advantage — fundamentally new approaches or richer data are needed to break the performance ceiling.

---

### Part 2 — EDA on `train.csv` (11,08,895 samples)

---

### 📊 Flood Probability Distribution
> *Histogram with KDE overlay of predicted flood probabilities (x-axis: ~0.25–0.75, y-axis: count up to ~82,000).*

![Flood Probability Distribution](https://github.com/user-attachments/assets/63ea2b79-dc42-439f-8b5b-32cf3a9494c1)

- Predicted flood probabilities form a **near-perfect bell curve tightly centred around 0.50**, with the vast majority falling in the 0.40–0.60 range — confirming the model assigns near-uncertain probabilities to almost every sample due to the dataset's inherently low feature-target signal.
- The absence of bimodal separation near 0 or 1 **visually encapsulates the fundamental modelling limitation**: without stronger discriminative features, threshold-based binary classification is unreliable for real-world flood early warning.

---

### 📊 Actual vs Predicted
> *Scatter plot of actual vs model-predicted flood probability values, with a red diagonal representing perfect prediction.*

![Actual vs Predicted](https://github.com/user-attachments/assets/ae7e9b86-7751-42ba-b93b-5685f22f9f8f)

- Predicted values cluster in a **compressed band between 0.30–0.75** regardless of actual values, with scatter following the red diagonal only loosely — the model systematically underestimates high actual probabilities and overestimates low ones, pulling predictions toward the centre.
- The dense concentration around the 0.40–0.60 mid-range confirms the model **lacks confidence to make extreme predictions**, compressing its output range due to insufficient discriminative signal in the current feature set.

---

### 📊 Residual Distribution
> *Histogram with KDE overlay of residuals (Actual − Predicted), x-axis: ~-0.175 to +0.175, y-axis: count peaking at ~65,000.*

![Residual Distribution](https://github.com/user-attachments/assets/e53dca68-86f3-4de2-a8b2-d1bdd692ed34)

- Residuals form a **sharply peaked, right-skewed distribution centred near zero**, with the majority of errors within ±0.05 — while the jagged right tail reveals systematic positive residuals, meaning the model more frequently under-predicts flood probability than over-predicts it.
- The right-skewed tail extending to +0.15 and the leptokurtic shape indicate **non-random, structured prediction errors** — a diagnostic signal that the model is missing specific subsets of high-probability flood events the current feature set leaves unexplained.

---

### 📊 Prediction Error Plot
> *Scatter plot of residuals (y-axis: ~-0.175 to +0.175) vs predicted flood probability (x-axis: 0.30–0.75), with a red zero-error reference line.*

![Prediction Error Plot](https://github.com/user-attachments/assets/8d2459da-b64f-4ddb-8dc4-b435827e9f73)

- The residuals form a **distinctive fan-shaped (heteroscedastic) pattern** — errors tightly compressed near ±0.02 at predicted values of 0.30–0.35, widening to ±0.10 and beyond near 0.50–0.60, indicating growing model uncertainty exactly where the majority of samples are concentrated.
- The roughly symmetric scatter confirms **no gross systematic prediction bias**, but the presence of diagonal streaks and structured clusters reveals non-random residual patterns — the model has learned partial structure but is missing relationships that temporal or geospatial features could capture.

---

### 📊 Model Performance Comparison (Regression)
> *Grouped bar chart comparing RMSE (blue) and R² (orange) across Linear Regression, XGBoost, LightGBM, and Random Forest.*

![Model Performance Comparison](https://github.com/user-attachments/assets/219c44ac-32b3-47e2-a1dd-547f48b896ae)

- **Linear Regression achieves the highest R² (~0.84)**, followed by XGBoost (~0.81), LightGBM (~0.76), and Random Forest (~0.65), while all four maintain remarkably low RMSE (~0.02–0.03) — the regression models predict flood probability values with high numerical closeness to the continuous target.
- The paradox of high R² alongside near-chance classification performance underscores a critical distinction — **fitting the continuous probability distribution centred around 0.50 does not translate to useful binary flood classification**, as small numerical accuracy near the decision boundary provides no real discriminative power.

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

- The **first 5 components** explain ~46% of cumulative variance.
- **10 components** reach ~77% cumulative variance — still below the 95% threshold.
- The flat scree curve confirms information is diffusely spread across many dimensions.
- PCA-reduced feature sets are explored as alternative inputs to downstream classifiers.

---

## 🤖 Models

### Classification
- Logistic Regression · Naive Bayes · KNN (k=5)
- Decision Tree (CART) · SVM (RBF)
- Random Forest · Extra Trees · AdaBoost
- Gradient Boosting · XGBoost · LightGBM

### Deep Learning
- MLP (sklearn) · Shallow DNN · Deep DNN
- Residual DNN · LSTM · 1-D CNN

### Dimensionality Reduction Pipelines
- PCA + Logistic Regression · PCA + Random Forest · PCA + XGBoost
- SVD + Logistic Regression · SVD + Random Forest · SVD + XGBoost

### Regression
- Linear Regression · XGBoost Regressor · LightGBM Regressor · Random Forest Regressor

### Unsupervised
- K-Means Clustering (k=2–6) · PCA / SVD

### Composite Scoring
- Weighted multi-model ensemble for flood risk probability output

---

## 📈 Results & Model Evaluation

> All output metrics, confusion matrices, ROC curves, and feature importance plots are saved to `flood_outputs/` after running the pipeline.

### 🏆 Key Performance Summary

| Metric | Best Model | Score |
|---|---|---|
| Composite Score | Shallow DNN | 0.5410 |
| F1-Score | Shallow DNN | 0.5813 |
| Recall | Shallow DNN | 0.6733 |
| Accuracy | Logistic Regression | 0.509 |
| ROC-AUC | Naive Bayes / LightGBM / MLP | 0.503 |
| R² (Regression) | Linear Regression | ~0.84 |

### Results — Flood Risk in India

![Result 1](https://github.com/user-attachments/assets/6236a972-f389-4bb4-9b35-fe86ef9b4523)
![Result 2](https://github.com/user-attachments/assets/4d391395-7a2d-47c9-9587-1e942665ad1e)
![Result 3](https://github.com/user-attachments/assets/943dea40-dc6c-4500-a358-37b2fc4699d6)
![Result 4](https://github.com/user-attachments/assets/c3268fb6-1b88-4c68-9e0b-2383b3524bf0)
![Result 5](https://github.com/user-attachments/assets/1d9925fb-8351-424f-adb2-26cedb5530a0)
![Result 6](https://github.com/user-attachments/assets/2d30a671-4281-45e6-b074-7d2730440bf9)
![Result 7](https://github.com/user-attachments/assets/7d649731-770d-46f3-8086-a83ba7ff624a)
![Result 8](https://github.com/user-attachments/assets/71753cb0-8bb3-4c24-9f6d-d662ca7dbbcb)
![Result 9](https://github.com/user-attachments/assets/07173af8-f0e9-4657-a4d6-339f27bc6ee8)
![Result 10](https://github.com/user-attachments/assets/cc83b596-f3f9-4239-9fff-f925a9bceda8)
![Result 11](https://github.com/user-attachments/assets/464bb7f5-65e3-40f3-ad72-7f2d961e7adb)
![Result 12](https://github.com/user-attachments/assets/da9ba8b6-6a41-4425-9bbd-b9616d99c6f7)
![Result 13](https://github.com/user-attachments/assets/30dd3d5a-fc0b-4732-aa94-0847b2203a06)
![Result 14](https://github.com/user-attachments/assets/42d3dd67-00f4-4d36-90ac-920d7beec2ec)
![Result 15](https://github.com/user-attachments/assets/69aa5576-b20d-409a-bff5-bbc31e6584bf)
![Result 16](https://github.com/user-attachments/assets/3badb7a7-27c4-4531-857a-1b72fa89bd1e)

### Results — Flood Risk Training Data

![Training Result](https://github.com/user-attachments/assets/e40452a5-522e-4a21-a8a2-8910e4333074)

---

## 📤 Submission — `submission.csv`

### What is `submission.csv`?

`submission.csv` is the **final output artifact** of the entire pipeline — a file containing flood probability predictions generated by the best-performing model (Shallow DNN) on the unlabelled `test.csv` dataset, formatted according to the structure defined in `sample_submission.csv`.

### File Format

```
id,FloodProbability
0,0.5012
1,0.4987
2,0.5134
3,0.4901
...
```

| Column | Type | Description |
|---|---|---|
| `id` | Integer | Unique row identifier matching the `test.csv` index |
| `FloodProbability` | Float (0–1) | Predicted probability of flood occurrence at that location |

### How It Is Generated

The submission file is produced at the end of `model.py` via the following pipeline:

```
test.csv
  └── Same preprocessing as train.csv
        ├── Standard Scaling (numeric features)
        ├── Label Encoding (categorical features)
        └── Feature Engineering (6 engineered features)
              └── Shallow DNN (best composite score model)
                    └── .predict_proba() → FloodProbability column
                          └── submission.csv
```

### Significance of Each Column

**`id`** — A simple sequential integer index that maps each prediction row back to its corresponding row in `test.csv`. This ensures the submission can be correctly evaluated row-by-row by any external scoring system or competition platform.

**`FloodProbability`** — A continuous value between 0 and 1 representing the model's estimated probability that a flood event occurs at that geographic location and environmental condition:

| Value Range | Model Interpretation |
|---|---|
| **~0.50** | Model is uncertain (expected given dataset characteristics) |
| **> 0.55** | Model leans toward predicting a flood event |
| **< 0.45** | Model leans toward predicting no flood event |

### Why the Predictions Cluster Near 0.50

As demonstrated in the **Flood Probability Distribution** plot in the EDA section, the vast majority of predicted values fall in the **0.40–0.60 range** in a near-perfect bell curve. This is a direct consequence of the dataset's near-zero feature-target correlations across all 20 features — the model cannot confidently assign extreme probabilities and instead converges toward the midpoint for most observations.

> This is **not a model implementation error** — it is a scientifically honest reflection of the current data's limitations. The submission file accurately captures what the best available model can infer from the given feature set.

### Interpreting Predictions for Real-World Use

| FloodProbability Range | Risk Level | Suggested Action |
|---|---|---|
| 0.00 – 0.35 | 🟢 Low | No immediate action required |
| 0.35 – 0.50 | 🟡 Below-average | Monitor conditions |
| 0.50 – 0.65 | 🟠 Above-average | Issue precautionary advisories |
| 0.65 – 1.00 | 🔴 High | Activate early warning systems |

> ⚠️ **Important:** Given the model's near-random discriminative ability (ROC-AUC ~0.50), these thresholds should be treated as **preliminary indicators only** and not relied upon for life-safety decisions without integration of richer, real-time data sources.

 
 ### LLM EVALUATIONS
 ![Uploading image.png…]()


### Improving Future Submissions

To produce more confident and actionable predictions in future iterations, the following data enrichments are recommended:
- **Temporal features** — rolling 7-day and 30-day rainfall accumulation windows
- **Satellite-derived indices** — NDVI, soil moisture from remote sensing imagery
- **Real-time river gauge readings** — hourly/daily discharge measurements
- **Topographic wetness index** — derived from Digital Elevation Model (DEM) data
- **Monsoon seasonality flags** — month-of-year encoding, IMD rainfall zone labels


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
- [ ] K-Medoids clustering as an alternative to K-Means
- [ ] Experiment with PCA/SVD + supervised hybrid pipelines

---

## 👤 Author

**BVPKARTHIKEYA**
GitHub: [@BVPKARTHIKEYA](https://github.com/BVPKARTHIKEYA)

---
## 🙏 Acknowledgements

Dataset represents simulated flood event data across India, incorporating real-world geographic coordinates and environmental parameters to model realistic flood-prone conditions across diverse terrain, soil types, and land cover categories.
