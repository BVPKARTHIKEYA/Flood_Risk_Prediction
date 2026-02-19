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
| Target variable | Flood Probability |
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

**Exploratory Data Analysis of FLOOD RISK PREDICTION IN INDIA **
📊 Target Variable Distribution — Flood Occurred
Bar chart (Class Count) and Pie chart (Class Proportion) of the target variable.
<img width="1560" height="520" alt="image" src="https://github.com/user-attachments/assets/39c25939-8f1b-4db3-b781-2d35c64394f9" />
The dataset contains 5,057 No Flood and 4,943 Flood samples, totalling 10,000 observations with a near-perfect class split of 50.6% vs 49.4% — confirming the dataset is well-balanced with no need for oversampling or undersampling techniques.
The class proportion pie chart visually reinforces the balance, indicating that standard evaluation metrics such as accuracy are reliable without requiring class-weighted adjustments or resampling strategies like SMOTE.
📊 Numeric Feature Distributions (by Flood Class)
3×3 grid of overlapping histograms comparing density distributions of 9 numeric features between Flood (red) and No Flood (blue) classes.
![WhatsApp Image 2026-02-20 at 01 19 39](https://github.com/user-attachments/assets/0dc29a29-58db-412c-bf6b-eae632f316ca)
Across all 9 features — Latitude, Longitude, Rainfall, Temperature, Humidity, River Discharge, Water Level, Elevation, and Population Density — the flood and no-flood distributions overlap almost entirely, with no feature showing a clear separation between classes, indicating that no single numeric feature alone is sufficient to predict flood occurrence.
The near-identical density curves across both classes confirm that the dataset's predictive signal is subtle and multivariate in nature, making it well-suited for ensemble methods and deep learning models that can capture complex feature interactions rather than simple threshold-based rules.
📊 Box Plots — Numeric Features vs Flood Class
3×3 grid of side-by-side box plots comparing the spread, median, and quartiles of 9 numeric features between Flood (red) and No Flood (blue) classes.
![WhatsApp Image 2026-02-20 at 01 19 54](https://github.com/user-attachments/assets/61b41af9-86a8-41e7-b653-07055ee93f01)
Across all 9 features — Latitude, Longitude, Rainfall, Temperature, Humidity, River Discharge, Water Level, Elevation, and Population Density — the medians, interquartile ranges, and whisker extents are virtually identical between the Flood and No Flood classes, statistically confirming that no individual feature produces a meaningful distributional shift between the two classes.
The uniformity in box sizes and whisker lengths across both classes reinforces that the dataset presents a genuinely difficult classification challenge, where flood prediction cannot rely on simple univariate thresholds and instead demands models capable of learning subtle, high-dimensional feature interactions.
📊 Correlation Heatmap
Lower-triangular heatmap showing pairwise Pearson correlation coefficients across all 12 features, color-coded from deep blue (negative) to deep red (positive).
---![WhatsApp Image 2026-02-20 at 01 20 10](https://github.com/user-attachments/assets/6abc2935-9850-4d51-a66c-1ce8f87452cd)
All pairwise correlations across the 12 features are extremely weak, ranging only from -0.03 to +0.03, with the strongest signals being Humidity (%) vs Flood Occurred (+0.03) and Historical Floods vs Water Level (-0.03) — confirming near-zero linear relationships between features and virtually no multicollinearity in the dataset.
The absence of any strongly correlated feature pairs indicates that each feature contributes independently to the dataset's information content, making this an ideal setup for ensemble and deep learning models that can exploit weak, distributed signals across all features simultaneously rather than relying on any dominant predictor.
📊 Categorical Features vs Flood Occurrence
Side-by-side grouped bar charts showing flood rate (%) for No Flood (blue) and Flood (red) classes across all categories of Land Cover (left) and Soil Type (right).
![WhatsApp Image 2026-02-20 at 01 20 18](https://github.com/user-attachments/assets/74ba7947-f835-4fdb-afd8-18670c26bb58)
Across all 5 Land Cover types (Agricultural, Desert, Forest, Urban, Water Body) and all 5 Soil Types (Clay, Loam, Peat, Sandy, Silt), the flood rate hovers uniformly between ~48–52% for both classes, with no single category showing a meaningfully higher or lower flood probability than another.
The near-identical bar heights across every category confirm that Land Cover and Soil Type carry negligible standalone discriminative power for flood prediction, and their value to the model likely only emerges through interaction with other features — motivating the engineered Soil_Land_Interaction feature.
📊 Geographic Distribution of Flood Events in India
Scatter plot mapping 10,000 data points across India's geographic coordinates (Latitude: 8°–37°N, Longitude: 68°–97°E), with No Flood events shown as blue circles and Flood events as red triangles.
![WhatsApp Image 2026-02-20 at 01 20 41](https://github.com/user-attachments/assets/5b01fa0f-8307-4736-9ee6-2b50fd756341)
Flood (red) and No Flood (blue) events are scattered uniformly and intermixed across the entire Indian subcontinent with no discernible regional clustering, hotspots, or geographic boundaries separating the two classes — confirming that latitude and longitude alone are not reliable predictors of flood occurrence.
The random, evenly spread spatial distribution across all regions rules out simple location-based heuristics for flood prediction, reinforcing that the model must rely on environmental and hydrological feature combinations rather than geographic proximity to make accurate classifications.
📊 Before vs After Standard Scaling
2×4 grid of histograms comparing the raw (blue, top row) and standard-scaled (green, bottom row) distributions of Rainfall (mm), River Discharge (m³/s), Elevation (m), and Population Density.
![WhatsApp Image 2026-02-20 at 01 20 59](https://github.com/user-attachments/assets/6c2afa37-4429-4133-8c9b-9458f993dc1d)
The top row shows the raw feature distributions spanning vastly different scales — Rainfall (0–300), River Discharge (0–5,000), Elevation (0–8,500), and Population Density (0–10,000) — which would bias distance-based and gradient-based models towards features with larger magnitudes if left unscaled.
After applying Standard Scaling (bottom row), all four features are re-centred to a common range of approximately -1.75 to +1.75 with zero mean and unit variance, while preserving the original shape of each distribution — ensuring every feature contributes equally during model training regardless of its original measurement scale.
📊 Engineered Features — Distributions by Flood Class

2×3 grid of histograms showing the distributions of 6 engineered features — Rainfall_x_Humidity, Discharge_x_WaterLevel, Elevation_Risk, High_Risk_Score, Soil_Land_Interaction, and Log_Elevation — split by Flood (red) and No Flood (blue) classes.

![WhatsApp Image 2026-02-20 at 01 21 10](https://github.com/user-attachments/assets/d6bb3e94-a4a0-401a-810b-82f3a7665380)
Features like Rainfall_x_Humidity and Discharge_x_WaterLevel display right-skewed distributions with heavy overlap between classes, while Elevation_Risk reveals a stark spike near zero indicating most locations have low elevation risk, and High_Risk_Score shows a discrete distribution at values 0, 1, 2, and 3 — demonstrating that each engineered feature captures a distinctly different aspect of flood risk that raw features alone could not express.
Despite the overlapping class distributions, the engineered features — particularly Elevation_Risk, Log_Elevation, and Soil_Land_Interaction — introduce non-linear transformations and interaction signals that post-preprocessing correlation analysis confirmed to be among the strongest predictors of flood occurrence, validating the feature engineering strategy.
📊 Feature Correlation with Target (Post-Preprocessing)

Horizontal bar chart ranking all 20 features by their Pearson correlation with the target variable Flood Occurred, with positive correlations in green and negative correlations in red.
![WhatsApp Image 2026-02-20 at 01 21 23](https://github.com/user-attachments/assets/1ee85fad-6f85-4ad6-9fba-58f71a5f6fc0)
The top positively correlated features are Humidity (% ) (~0.034), Elevation_Risk (~0.021), Historical Floods (~0.017), and Rainfall_x_Humidity (~0.012) — notably, three of the top four are engineered features, directly validating that the feature engineering step successfully amplified predictive signal that was absent in the raw features alone.
All 20 features remain within a narrow Pearson range of -0.02 to +0.034, confirming that while no single feature is a dominant predictor, the clear directional split — elevation, temperature, and water level negatively correlated; humidity, historical floods, and risk scores positively correlated — provides the model with a meaningful, interpretable set of weak signals that ensemble methods can combine for robust flood classification.
📊 Future Work: PCA Analysis
Side-by-side charts showing the Individual Explained Variance per Component (blue bar chart, left) and Cumulative Explained Variance (red line chart, right) across the first 10 principal components, with a dashed 95% threshold line.
![WhatsApp Image 2026-02-20 at 01 21 44](https://github.com/user-attachments/assets/bff7b1d2-c30a-4554-8cbe-c198ba5c33df)
The scree plot (left) shows a gradual decline from ~13% variance for PC1 down to ~5% for PC10, with no sharp elbow point — indicating that information is spread diffusely and roughly equally across all components rather than being dominated by a few principal axes, consistent with the near-zero pairwise correlations observed in EDA.
The cumulative variance curve (right) reaches only ~77% with 10 components, remaining well below the 95% threshold — suggesting that aggressive dimensionality reduction via PCA would result in significant information loss for this dataset, and that retaining the full feature set or using a large number of components is preferable for downstream model performance.
📊 Future Work: Clustering Analysis
Three-panel figure showing the K-Means Elbow Curve (left), Silhouette Scores for k=2 to 6 (centre), and a PCA 2-D projection of K-Means clusters at k=2 (right).
![WhatsApp Image 2026-02-20 at 01 21 55](https://github.com/user-attachments/assets/50626a16-2d8b-405f-b83b-279ef350fc39)
The elbow curve shows a steady, linear decrease in inertia from ~182,000 at k=2 to ~150,000 at k=6 with no distinct elbow point, while silhouette scores remain consistently low across all values of k (best score of only 0.095 at k=2) — indicating that the data does not form naturally well-separated clusters, which aligns with the diffuse, overlapping feature distributions observed throughout EDA.
The PCA 2-D projection at k=2 (right) reveals heavily overlapping Cluster 0 (blue) and Cluster 1 (brown) with no clear geometric boundary separating them, confirming that unsupervised clustering cannot meaningfully partition flood and no-flood events in this dataset and that supervised learning approaches are essential for achieving reliable flood classification.
📊 K-Means Flood Risk Zones (Geographic)
Side-by-side geographic scatter plots mapping K-Means Cluster 0 (blue, left) and Cluster 1 (brown, right) across India's latitude/longitude grid, with each cluster's flood rate displayed as a subtitle.
![WhatsApp Image 2026-02-20 at 01 22 14](https://github.com/user-attachments/assets/51e9d251-749a-44ab-8f4a-ff11cdc510da)
Both Cluster 0 and Cluster 1 are uniformly scattered across the entire Indian subcontinent with no geographic concentration or regional boundaries separating them, and their flood rates are virtually identical at 50.65% and 50.44% respectively — confirming that K-Means clustering has failed to identify any meaningful flood risk zones geographically.
The near-equal flood rates across both clusters (differing by only 0.21%) provide definitive evidence that geographic coordinates alone cannot delineate high-risk from low-risk flood zones, and that meaningful risk stratification requires incorporating the full suite of environmental, hydrological, and engineered features through supervised learning models.
📊 Test-Set Metrics — All Models
2×3 grid of horizontal bar charts comparing 23 models across 6 evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC on the test set.
![WhatsApp Image 2026-02-20 at 01 22 35](https://github.com/user-attachments/assets/adefd542-9461-46cd-b913-7b4997918bb2)
Logistic Regression emerges as the top performer across most metrics — achieving the highest Accuracy (~0.509), Precision (~0.508), Recall (~0.566), and F1-Score (~0.537) — while deep learning models (Shallow DNN, Deep DNN, Residual DNN, LSTM, 1-D CNN) show inconsistent results with notably low Recall scores (as low as 0.073 for LSTM), suggesting that complex architectures struggle to extract signal from this inherently low-correlation dataset.
ROC-AUC and PR-AUC scores hover in the 0.46–0.51 range across all 23 models — including classical ML, ensemble methods (XGBoost, LightGBM, Random Forest), and PCA/SVD-reduced pipelines — confirming that the dataset presents a genuinely hard classification problem where no model significantly outperforms random chance, and that further performance gains will likely require external data sources, temporal features, or more sophisticated domain-driven feature engineering rather than model complexity alone.
📊 ROC Curves — All Models
Overlapping ROC curves for all 23 models plotted against the random classifier baseline (black dashed diagonal), with each model's AUC score shown in the legend.
![WhatsApp Image 2026-02-20 at 01 22 51](https://github.com/user-attachments/assets/5b4cd0ea-937e-4ac0-963e-167ffaf110ba)
All 23 model ROC curves — spanning classical ML, ensemble methods, deep learning, and PCA/SVD pipelines — huddle tightly around the random classifier diagonal with AUC scores ranging narrowly from 0.474 (Residual DNN) to 0.503 (Naive Bayes, LightGBM, MLP), confirming that no model achieves meaningful discriminative ability beyond random chance on this dataset.
The near-complete overlap of all curves with the diagonal is a definitive diagnostic signal that the current feature set, regardless of model complexity, does not contain sufficient separable information to reliably distinguish flood from no-flood events — strongly motivating future work on richer data sources such as temporal rainfall sequences, satellite imagery, or real-time river gauge readings to break through this performance ceiling.
📊 Precision-Recall Curves — All Models
Overlapping Precision-Recall curves for all 23 models plotted against the random classifier baseline (black dashed line at ~0.50), with each model's PR-AUC score shown in the legend.
![WhatsApp Image 2026-02-20 at 01 23 00](https://github.com/user-attachments/assets/408b1467-29de-4d6c-83ac-b94f951d1903)
All 23 models exhibit a sharp precision spike near zero recall followed by an immediate collapse to the ~0.50 baseline, with PR-AUC scores ranging narrowly from 0.487 (Residual DNN) to 0.511 (Gradient Boosting) — indicating that while models can achieve momentarily high precision on a tiny fraction of the most confident predictions, they cannot maintain any precision advantage as recall increases, effectively confirming near-random performance across the full operating range.
The rapid convergence of all curves onto the dashed baseline — regardless of whether the model is a simple Logistic Regression, a powerful XGBoost/LightGBM ensemble, or a deep learning architecture — serves as a final definitive confirmation that the dataset's current feature set is insufficient for reliable flood classification, and that meaningful improvement will require fundamentally richer input data rather than further model tuning or architectural experimentation.
📊 Confusion Matrices — Top 6 Models (by F1)
2×3 grid of confusion matrices for the top 6 models ranked by F1-score: Shallow DNN (0.5813), LSTM (0.5726), PCA + Logistic Regression (0.5458), SVD + Logistic Regression (0.5458), Random Forest (0.5435), and MLP sklearn (0.5402).
![WhatsApp Image 2026-02-20 at 01 23 08](https://github.com/user-attachments/assets/7e118ab0-1316-4dcd-b72b-61d1d04a731a)
Across all 6 top-performing models, the confusion matrices reveal a consistent "Flood" prediction bias — all models predict the Flood class significantly more often than No Flood, with true positive (Flood correctly predicted) counts ranging from 437–511 substantially outweighing true negative (No Flood correctly predicted) counts of 253–322, indicating that models default to predicting the more frequent patterns associated with flood events rather than learning genuine class boundaries.
The high false positive counts (440–500 No Flood samples misclassified as Flood) across every model further confirm that none of the top 6 models has learned a reliably discriminative decision boundary, with the best performer Shallow DNN (F1=0.5813) only marginally outperforming random guessing — reinforcing that the classification challenge stems from fundamental data limitations rather than model choice or architecture.
📊 Deep Learning Training History
5×2 grid of Loss and AUC training curves for all 5 deep learning models — Shallow DNN, Deep DNN, Residual DNN, LSTM, and 1-D CNN — showing training (blue) vs validation (orange dashed) performance across 15 epochs.
![WhatsApp Image 2026-02-20 at 01 23 19](https://github.com/user-attachments/assets/86228081-3bc9-4da3-8bae-4c1c1f51e362)
Residual DNN and 1-D CNN display the most pronounced overfitting — their training AUC climbs steeply to 0.65+ while validation AUC plateaus near 0.52, and training loss continues dropping while validation loss diverges upward — indicating these architectures are memorising training patterns rather than learning generalisable flood signals from the limited-correlation feature set.
Across all 5 models, validation AUC remains flat and noisy in the 0.50–0.53 range throughout training regardless of architecture depth or type (feedforward, recurrent, or convolutional), while training loss consistently decreases — confirming that increasing model complexity and training duration yields no real-world predictive improvement and that the performance ceiling is a data quality constraint, not a modelling one.
📊 Feature Importances — Ensemble Models
Side-by-side horizontal bar charts ranking all 20 features by importance score across three ensemble models: Random Forest (left), XGBoost (centre), and LightGBM (right).
![WhatsApp Image 2026-02-20 at 01 23 34](https://github.com/user-attachments/assets/d62d1908-5364-454a-b27c-a0e39750eec4)
The three ensemble models assign strikingly different importance rankings to the same features — Random Forest prioritises Temperature, Longitude, and Discharge_x_WaterLevel; XGBoost ranks Latitude, Log_PopDensity, and Discharge_x_WaterLevel highest; while LightGBM heavily favours Latitude, Longitude, and Temperature — indicating that no single feature is universally dominant and that each model is capturing different, complementary aspects of the weak signal distributed across the dataset.
Notably, engineered features such as Discharge_x_WaterLevel, Rainfall_x_Humidity, and Elevation_Risk appear consistently in the mid-to-upper importance tiers across all three models, while High_Risk_Score, Infrastructure, and Historical Floods rank lowest — validating that continuous interaction features contribute more extractable signal to tree-based ensembles than discrete composite scores or historical count variables.
📊 Model × Metric Heatmap (Test Set)
Color-coded heatmap matrix comparing all 23 models (columns) across 6 evaluation metrics — Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC (rows) — with darker blue indicating higher scores and yellow-green indicating outlier lows.
![WhatsApp Image 2026-02-20 at 01 23 46](https://github.com/user-attachments/assets/d6e0d826-1c39-402c-ab9a-02e414306c85)
The heatmap reveals a remarkably uniform band of dark blue (~0.49–0.51) across Accuracy, Precision, ROC-AUC, and PR-AUC for virtually all 23 models, with the only notable outliers being Naive Bayes and Deep DNN on Recall and F1 (highlighted in yellow-green at 0.018 and 0.275 respectively) — confirming that the performance plateau is a dataset-wide phenomenon rather than a failure of any specific model family.
The Recall and F1 rows show the greatest variance across models, with Shallow DNN achieving the highest Recall (0.673) and F1 (0.581) while Naive Bayes collapses to near-zero Recall (0.018) — suggesting that the choice of model primarily affects the precision-recall trade-off rather than overall discriminative ability, and that optimising for F1 rather than accuracy should guide final model selection for this flood prediction task.
📊 Best Model Deep Analysis — Shallow DNN
Five-panel dashboard for the best model (Shallow DNN) showing: Confusion Matrix (top left), ROC Curve with AUC=0.5006 (top centre), Precision-Recall Curve with PR-AUC=0.5028 (top right), All Metrics bar chart (bottom left), and Model Ranking by Composite Weighted Score across all 23 models (bottom right).
![WhatsApp Image 2026-02-20 at 01 23 58](https://github.com/user-attachments/assets/0d876a69-781a-44ae-842f-bfeafd0c5f18)
The Shallow DNN's detailed analysis reveals a strong internal contradiction — it achieves the highest Recall (0.6733) and F1 (0.5813) among all models, yet its ROC-AUC (0.5006) and PR-AUC (0.5028) are barely above random, and its confusion matrix shows 488 false positives against only 253 true negatives — confirming that the model maximises flood detection by aggressively over-predicting the Flood class rather than through genuinely learned discrimination.
The composite weighted score ranking (bottom right) places Shallow DNN (0.5306) as the clear best model, followed closely by LSTM (0.5291), MLP sklearn (0.5203), and LightGBM (0.5201), while Naive Bayes (0.3360), Deep DNN (0.4405), and 1-D CNN (0.4410) rank at the bottom — establishing that moderately complex architectures outperform both overly simple and overly deep models on this low-signal dataset, and that the Shallow DNN's balance of recall sensitivity and reasonable precision makes it the most practical choice for deployment despite the overall performance ceiling.
📊 All Models — Composite Score Ranking
Vertical bar chart ranking all 23 models by their Composite Weighted Score, with Shallow DNN highlighted in gold (best), all other models in blue, and a red dashed mean line at 0.4947.
![WhatsApp Image 2026-02-20 at 01 24 05](https://github.com/user-attachments/assets/aefe2863-e6b6-4e8a-a36d-68562557f9ee)
Shallow DNN (0.5410) leads all 23 models by a clear margin, followed by LSTM (0.5291), MLP sklearn (0.5203), and LightGBM (0.5201) — all of which sit above the mean threshold of 0.4947 — while the bottom tier is occupied by Naive Bayes (0.3360), Deep DNN (0.4405), and 1-D CNN (0.4410), revealing that moderately complex neural architectures consistently outperform both overly simple probabilistic models and overly deep sequential/convolutional networks on this low-signal tabular dataset.
The tight clustering of 16 out of 23 models within the narrow 0.49–0.52 composite score band around the mean line confirms that no architectural choice provides a decisive advantage, and the chart subtitle explicitly flags that PCA/SVD + Supervised pipelines, LSTM, 1-D CNN, K-Means, and K-Medoids are designated as future work directions — acknowledging that the current performance ceiling requires fundamentally new approaches rather than incremental model tuning to achieve meaningful flood prediction gains.
**Exploratory Data Analysis of FLOOD RISK PREDICTION TRAINING DATA **
📊 Flood Probability Distribution
Histogram with KDE overlay showing the distribution of predicted flood probabilities across the full dataset, with FloodProbability on the x-axis (ranging ~0.25–0.75) and Count on the y-axis (up to ~82,000 samples).
![WhatsApp Image 2026-02-20 at 01 24 15](https://github.com/user-attachments/assets/63ea2b79-dc42-439f-8b5b-32cf3a9494c1)
The predicted flood probabilities form a near-perfect bell curve tightly centred around 0.50, with the vast majority of predictions falling in the 0.40–0.60 range and peak counts exceeding 80,000 — confirming that the best model (Shallow DNN) assigns near-uncertain probabilities to almost every sample, unable to confidently push predictions toward either 0 (No Flood) or 1 (Flood) due to the dataset's inherently low feature-target signal.
The absence of any bimodal separation or distinct peaks near 0 or 1 — which would indicate confident class-specific predictions — visually encapsulates the fundamental limitation of the entire modelling pipeline: without stronger discriminative features, the model defaults to predicting ~50% flood probability for most observations, rendering threshold-based binary classification unreliable for real-world flood early warning applications
📊 Actual vs Predicted
Scatter plot comparing actual flood probability values (x-axis) against model-predicted values (y-axis), with a red diagonal line representing perfect prediction.
![WhatsApp Image 2026-02-20 at 01 24 23](https://github.com/user-attachments/assets/ae7e9b86-7751-42ba-b93b-5685f22f9f8f)
The predicted values cluster in a compressed band between 0.30–0.75 regardless of the actual values, with the scatter points following the red perfect-prediction diagonal only loosely and showing significant vertical spread — confirming that the model systematically underestimates high actual probabilities and overestimates low ones, pulling all predictions toward the centre rather than spanning the full 0–1 range.
The dense concentration of points around the 0.40–0.60 mid-range with wide vertical dispersion at every actual value reflects the same finding seen in the Flood Probability Distribution — the model lacks the confidence to make extreme predictions, effectively compressing its output range and confirming that the current feature set provides insufficient signal for the model to learn a precise, well-calibrated mapping between input features and true flood probability.
📊 Residual Distribution
Histogram with KDE overlay showing the distribution of residuals (Actual − Predicted flood probability) across the full dataset, with residual values on the x-axis (ranging ~-0.175 to +0.175) and Count on the y-axis (peaking at ~65,000).
![WhatsApp Image 2026-02-20 at 01 24 31](https://github.com/user-attachments/assets/e53dca68-86f3-4de2-a8b2-d1bdd692ed34)
The residuals form a sharply peaked, right-skewed distribution centred very close to zero, with the overwhelming majority of prediction errors falling within the narrow ±0.05 range and a pronounced spike at 0.00 — indicating that while the model's average prediction error is small, the sharp KDE peak and jagged right tail reveal systematic positive residuals, meaning the model more frequently under-predicts flood probability than over-predicts it.
The near-zero centering of the residual distribution confirms the model is globally unbiased in the mean, but the right-skewed tail extending to +0.15 and the leptokurtic (heavy-peaked) shape indicate non-random, structured prediction errors — a diagnostic signal that the model is missing specific subsets of high-probability flood events entirely, further reinforcing the need for richer temporal or geospatial features to capture the systematic patterns the current feature set leaves unexplained.
📊 Prediction Error Plot
Scatter plot of residuals (y-axis, ranging ~-0.175 to +0.175) against predicted flood probability values (x-axis, ranging 0.30–0.75), with a red horizontal zero-error reference line.
![WhatsApp Image 2026-02-20 at 01 24 41](https://github.com/user-attachments/assets/8d2459da-b64f-4ddb-8dc4-b435827e9f73)
The residuals form a distinctive fan-shaped (heteroscedastic) pattern that widens from low to high predicted values — errors are tightly compressed near ±0.02 at predicted values around 0.30–0.35, but spread increasingly to ±0.10 and beyond as predictions approach 0.50–0.60 — indicating that the model's uncertainty grows significantly for mid-range probability predictions, exactly where the majority of samples are concentrated.
The roughly symmetric scatter above and below the zero reference line confirms no gross systematic bias in prediction direction, but the presence of diagonal streaks and structured point clusters rather than pure random scatter reveals non-random residual patterns — a clear sign that the model has learned some partial structure in the data but is missing underlying relationships that additional features such as temporal sequences, soil saturation levels, or catchment area data could help capture.
📊 Model Performance Comparison (Regression)
Grouped bar chart comparing RMSE (blue) and R² (orange) scores across four regression models: Linear Regression, XGBoost, LightGBM, and Random Forest.
![WhatsApp Image 2026-02-20 at 01 24 48](https://github.com/user-attachments/assets/219c44ac-32b3-47e2-a1dd-547f48b896ae)
Linear Regression achieves the highest R² (~0.84) followed by XGBoost (~0.81), LightGBM (~0.76), and Random Forest (~0.65), while all four models maintain remarkably low RMSE scores (~0.02–0.03) — indicating that the regression models can predict flood probability values with high numerical closeness to the actual continuous target, even though the underlying classification task remains near-random.
The paradox of high R² alongside near-chance classification performance underscores a critical distinction — the regression models are highly effective at fitting the continuous probability distribution (which is tightly centred around 0.50 as seen in the Flood Probability Distribution plot), but this precision in predicting values near 0.50 does not translate to useful binary flood/no-flood classification, as small numerical accuracy around the decision boundary provides no real discriminative power for early warning applications.
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
Results of Flood Risk in India

![WhatsApp Image 2026-02-20 at 01 26 54](https://github.com/user-attachments/assets/6236a972-f389-4bb4-9b35-fe86ef9b4523)

![WhatsApp Image 2026-02-20 at 01 27 25](https://github.com/user-attachments/assets/4d391395-7a2d-47c9-9587-1e942665ad1e)

![WhatsApp Image 2026-02-20 at 01 28 15](https://github.com/user-attachments/assets/943dea40-dc6c-4500-a358-37b2fc4699d6)

![WhatsApp Image 2026-02-20 at 01 28 50](https://github.com/user-attachments/assets/c3268fb6-1b88-4c68-9e0b-2383b3524bf0)

![WhatsApp Image 2026-02-20 at 01 29 19](https://github.com/user-attachments/assets/1d9925fb-8351-424f-adb2-26cedb5530a0)

![WhatsApp Image 2026-02-20 at 01 29 47](https://github.com/user-attachments/assets/2d30a671-4281-45e6-b074-7d2730440bf9)

![WhatsApp Image 2026-02-20 at 01 30 15](https://github.com/user-attachments/assets/7d649731-770d-46f3-8086-a83ba7ff624a)

![WhatsApp Image 2026-02-20 at 01 30 39](https://github.com/user-attachments/assets/71753cb0-8bb3-4c24-9f6d-d662ca7dbbcb)

![WhatsApp Image 2026-02-20 at 01 31 32](https://github.com/user-attachments/assets/07173af8-f0e9-4657-a4d6-339f27bc6ee8)

![WhatsApp Image 2026-02-20 at 01 31 56](https://github.com/user-attachments/assets/cc83b596-f3f9-4239-9fff-f925a9bceda8)

![WhatsApp Image 2026-02-20 at 01 32 13](https://github.com/user-attachments/assets/464bb7f5-65e3-40f3-ad72-7f2d961e7adb)

![WhatsApp Image 2026-02-20 at 01 32 48](https://github.com/user-attachments/assets/da9ba8b6-6a41-4425-9bbd-b9616d99c6f7)

![WhatsApp Image 2026-02-20 at 01 33 08](https://github.com/user-attachments/assets/30dd3d5a-fc0b-4732-aa94-0847b2203a06)

![WhatsApp Image 2026-02-20 at 01 33 30](https://github.com/user-attachments/assets/42d3dd67-00f4-4d36-90ac-920d7beec2ec)

![WhatsApp Image 2026-02-20 at 01 33 56](https://github.com/user-attachments/assets/69aa5576-b20d-409a-bff5-bbc31e6584bf)

![WhatsApp Image 2026-02-20 at 01 34 31](https://github.com/user-attachments/assets/3badb7a7-27c4-4531-857a-1b72fa89bd1e)

Results of Flood Risk Training Data
![WhatsApp Image 2026-02-20 at 01 06 31](https://github.com/user-attachments/assets/e40452a5-522e-4a21-a8a2-8910e4333074)

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
