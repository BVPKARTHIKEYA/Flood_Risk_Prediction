"""
=============================================================================
  FLOOD RISK PREDICTION — COMPLETE ML/DL PIPELINE  (FUTURE WORK EDITION)
  Dataset: flood_risk_dataset_india.csv
  Author : Generated via Claude (Anthropic)
=============================================================================

PIPELINE SECTIONS
─────────────────
  1.  Exploratory Data Analysis (EDA)
  2.  Data Preprocessing
  3.  Post-Preprocessing EDA
  4.  Model Selection & Training
        ├── Classic ML  : Logistic Regression, Naive Bayes, KNN, Decision Tree,
        │                 SVM, Random Forest, Extra Trees, AdaBoost,
        │                 Gradient Boosting, XGBoost, LightGBM, MLP
        ├── Deep Learning: Shallow DNN, Deep DNN, Residual DNN
        ├── Future Work 1: PCA + Supervised Models (LR / RF / XGB)
        ├── Future Work 2: SVD + Supervised Models
        ├── Future Work 3: Clustering Analysis (K-Means, K-Medoids)
        ├── Future Work 4: LSTM (time-series style)
        └── Future Work 5: 1-D CNN (spatial feature extraction)
  5.  Metrics Verification & Plots
  6.  Best Model Determination & Deep Analysis

DEPENDENCIES
─────────────
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm \
              tensorflow keras scikit-learn-extra tqdm joblib
=============================================================================
"""

import warnings
from pathlib import Path

import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection   import (train_test_split, StratifiedKFold,
                                        cross_val_score)
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.decomposition     import PCA, TruncatedSVD
from sklearn.cluster           import KMeans
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay, silhouette_score
)
from sklearn.linear_model      import LogisticRegression
from sklearn.naive_bayes       import GaussianNB
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import (RandomForestClassifier,
                                        GradientBoostingClassifier,
                                        AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm               import SVC
from sklearn.neural_network    import MLPClassifier

import xgboost  as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow       import keras
from tensorflow.keras import layers, callbacks, regularizers

from tqdm  import tqdm
import joblib

# Optional: K-Medoids
try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False
    print("  [INFO] sklearn_extra not installed — K-Medoids will be skipped.")
    print("         Install with: pip install scikit-learn-extra")

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "flood_risk_dataset_india.csv"
OUTPUT_DIR = Path("flood_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PLOT_STYLE = "seaborn-v0_8-whitegrid"
plt.rcParams.update({"figure.dpi": 130, "font.family": "DejaVu Sans",
                     "axes.titlesize": 13})

def savefig(name):
    path = OUTPUT_DIR / f"{name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  [saved] {path}")


# =============================================================================
#  SECTION 1 — EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("  SECTION 1 — EXPLORATORY DATA ANALYSIS")
print("="*70)

df = pd.read_csv(DATA_PATH)
print(f"\n▸ Shape   : {df.shape}  ({df.shape[0]:,} rows × {df.shape[1]} columns)")
print(f"▸ Columns : {list(df.columns)}")
print("\n── Data Types & Non-Null Counts ──────────────────────────────────────")
print(df.info())
print("\n── Descriptive Statistics ────────────────────────────────────────────")
print(df.describe().T.to_string())

missing = df.isnull().sum()
print("\n── Missing Values ────────────────────────────────────────────────────")
print(missing[missing > 0] if missing.sum() > 0 else "  No missing values ✓")
print(f"── Duplicates : {df.duplicated().sum()}")

vc = df["Flood Occurred"].value_counts()
print(f"\n── Target Distribution ──  {dict(vc.items())}")
print(f"  Class ratio (1:0) = {vc[1]/vc[0]:.4f}")

cat_cols = ["Land Cover", "Soil Type"]
bin_cols = ["Infrastructure", "Historical Floods"]
num_cols = [c for c in df.columns if c not in cat_cols + bin_cols + ["Flood Occurred"]]

for c in cat_cols:
    print(f"\n  {c}:\n{df[c].value_counts().to_string()}")

print("\n── Correlation with Target ───────────────────────────────────────────")
print(df.select_dtypes(include="number").corr()["Flood Occurred"]
        .sort_values(ascending=False).to_string())

# PLOT 1: Target Distribution
with plt.style.context(PLOT_STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Target Variable — Flood Occurred", fontsize=14, fontweight="bold")
    vc.plot(kind="bar", ax=axes[0], color=["#42A5F5","#EF5350"],
            edgecolor="white", width=0.5)
    axes[0].set_title("Class Count")
    axes[0].set_xticklabels(["No Flood","Flood"], rotation=0)
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height()):,}",
                         (p.get_x()+p.get_width()/2, p.get_height()),
                         ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[1].pie(vc, labels=["No Flood","Flood"], autopct="%1.1f%%",
                colors=["#42A5F5","#EF5350"], startangle=90,
                wedgeprops={"edgecolor":"white","linewidth":2})
    axes[1].set_title("Class Proportion")
    plt.tight_layout(); savefig("01_target_distribution")

# PLOT 2: Numeric Distributions
with plt.style.context(PLOT_STYLE):
    n, cols_n = len(num_cols), 3
    rows_n = (n + cols_n - 1) // cols_n
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(17, rows_n*4))
    axes = axes.flatten()
    fig.suptitle("Numeric Feature Distributions (by Flood Class)",
                 fontsize=14, fontweight="bold")
    for i, col in enumerate(num_cols):
        for label, color in zip([0,1], ["#42A5F5","#EF5350"]):
            axes[i].hist(df[df["Flood Occurred"]==label][col], bins=40,
                         alpha=0.6, color=color,
                         label="No Flood" if label==0 else "Flood", density=True)
        axes[i].set_title(col); axes[i].legend(fontsize=8)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    plt.tight_layout(); savefig("02_numeric_distributions")

# PLOT 3: Box Plots
with plt.style.context(PLOT_STYLE):
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(17, rows_n*4))
    axes = axes.flatten()
    fig.suptitle("Box Plots — Numeric Features vs Flood Class",
                 fontsize=14, fontweight="bold")
    for i, col in enumerate(num_cols):
        bp = axes[i].boxplot(
            [df[df["Flood Occurred"]==0][col], df[df["Flood Occurred"]==1][col]],
            patch_artist=True, notch=True,
            medianprops={"color":"black","linewidth":2})
        for patch, color in zip(bp["boxes"], ["#42A5F5","#EF5350"]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        axes[i].set_title(col)
        axes[i].set_xticklabels(["No Flood","Flood"])
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    plt.tight_layout(); savefig("03_boxplots")

# PLOT 4: Correlation Heatmap
with plt.style.context(PLOT_STYLE):
    fig, ax = plt.subplots(figsize=(12, 9))
    corr_matrix = df.select_dtypes(include="number").corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink":0.8})
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout(); savefig("04_correlation_heatmap")

# PLOT 5: Categorical vs Target
with plt.style.context(PLOT_STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Categorical Features vs Flood Occurrence",
                 fontsize=14, fontweight="bold")
    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(df[col], df["Flood Occurred"], normalize="index") * 100
        ct.plot(kind="bar", ax=axes[i], color=["#42A5F5","#EF5350"], edgecolor="white")
        axes[i].set_title(f"{col} — Flood Rate (%)")
        axes[i].legend(["No Flood","Flood"])
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout(); savefig("05_categorical_vs_target")

# PLOT 6: Geographic Scatter
with plt.style.context(PLOT_STYLE):
    fig, ax = plt.subplots(figsize=(10, 8))
    for label, color, marker in zip([0,1], ["#42A5F5","#EF5350"], ["o","^"]):
        sub = df[df["Flood Occurred"]==label]
        ax.scatter(sub["Longitude"], sub["Latitude"], c=color, marker=marker,
                   s=3, alpha=0.4, label="No Flood" if label==0 else "Flood")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Geographic Distribution of Flood Events in India",
                 fontsize=13, fontweight="bold")
    ax.legend(markerscale=4)
    plt.tight_layout(); savefig("06_geographic_scatter")


# =============================================================================
#  SECTION 2 — DATA PREPROCESSING
# =============================================================================
print("\n" + "="*70)
print("  SECTION 2 — DATA PREPROCESSING")
print("="*70)

X = df.drop(columns=["Flood Occurred"]).copy()
y = df["Flood Occurred"]

le_land = LabelEncoder(); le_soil = LabelEncoder()
X["Land Cover Enc"] = le_land.fit_transform(X["Land Cover"])
X["Soil Type Enc"]  = le_soil.fit_transform(X["Soil Type"])
X = X.drop(columns=cat_cols)

print("\n── Outlier Capping (IQR × 1.5) ───────────────────────────────────────")
clip_cols = ["River Discharge (m³/s)", "Elevation (m)", "Population Density",
             "Rainfall (mm)", "Water Level (m)"]
for col in clip_cols:
    Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out = ((X[col] < lo) | (X[col] > hi)).sum()
    X[col] = X[col].clip(lo, hi)
    print(f"  {col:<35} | capped: {n_out:4d}")

# Feature Engineering
X["Rainfall_x_Humidity"]    = X["Rainfall (mm)"] * X["Humidity (%)"] / 100
X["Discharge_x_WaterLevel"] = X["River Discharge (m³/s)"] * X["Water Level (m)"]
X["Elevation_Risk"]         = 1 / (X["Elevation (m)"] + 1)
X["High_Risk_Score"]        = (
    (X["Rainfall (mm)"]          > X["Rainfall (mm)"].quantile(0.75)).astype(int) +
    (X["River Discharge (m³/s)"] > X["River Discharge (m³/s)"].quantile(0.75)).astype(int) +
    (X["Water Level (m)"]        > X["Water Level (m)"].quantile(0.75)).astype(int)
)
# Vegetation/Soil interaction (expanded clustering feature)
X["Soil_Land_Interaction"]  = X["Soil Type Enc"] * X["Land Cover Enc"]
# Log transforms for skewed features
X["Log_Elevation"]          = np.log1p(X["Elevation (m)"])
X["Log_PopDensity"]         = np.log1p(X["Population Density"])
print(f"\n  Feature Engineering complete. Total features: {X.shape[1]}")

# Split
X_tv, X_test,  y_tv, y_test  = train_test_split(X, y, test_size=0.15,
                                                  stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.1765,
                                                   stratify=y_tv, random_state=42)
print(f"  Train:{X_train.shape[0]:,}  Val:{X_val.shape[0]:,}  Test:{X_test.shape[0]:,}")

# Scaling
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
feature_names = list(X.columns)
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
print(f"  Scaler saved → {OUTPUT_DIR}/scaler.pkl")

# PLOT 7: Before/After Scaling
with plt.style.context(PLOT_STYLE):
    compare_cols = ["Rainfall (mm)", "River Discharge (m³/s)",
                    "Elevation (m)", "Population Density"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    fig.suptitle("Before vs After Standard Scaling", fontsize=14, fontweight="bold")
    for j, col in enumerate(compare_cols):
        col_idx = feature_names.index(col)
        axes[0,j].hist(X_train[col], bins=40, color="#1976D2", alpha=0.8, edgecolor="white")
        axes[0,j].set_title(f"Before: {col}", fontsize=8)
        axes[1,j].hist(X_train_s[:,col_idx], bins=40, color="#43A047",
                       alpha=0.8, edgecolor="white")
        axes[1,j].set_title(f"After: {col}", fontsize=8)
    plt.tight_layout(); savefig("07_scaling_comparison")


# =============================================================================
#  SECTION 3 — POST-PREPROCESSING EDA
# =============================================================================
print("\n" + "="*70)
print("  SECTION 3 — POST-PREPROCESSING EDA")
print("="*70)

# PLOT 8: Engineered Feature Distributions
with plt.style.context(PLOT_STYLE):
    eng_cols = ["Rainfall_x_Humidity","Discharge_x_WaterLevel",
                "Elevation_Risk","High_Risk_Score",
                "Soil_Land_Interaction","Log_Elevation"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()
    fig.suptitle("Engineered Features — Distributions by Flood Class",
                 fontsize=14, fontweight="bold")
    for i, col in enumerate(eng_cols):
        m0 = y_train.values==0; m1 = y_train.values==1
        axes[i].hist(X_train[col].values[m0], bins=30, alpha=0.6,
                     color="#42A5F5", density=True, label="No Flood")
        axes[i].hist(X_train[col].values[m1], bins=30, alpha=0.6,
                     color="#EF5350", density=True, label="Flood")
        axes[i].set_title(col, fontsize=9); axes[i].legend(fontsize=7)
    plt.tight_layout(); savefig("08_engineered_features")

# PLOT 9: Post-Preprocessing Correlation
with plt.style.context(PLOT_STYLE):
    df_tr = pd.DataFrame(X_train_s, columns=feature_names)
    df_tr["Flood Occurred"] = y_train.values
    corr_pp = df_tr.corr()["Flood Occurred"].drop("Flood Occurred").sort_values()
    fig, ax = plt.subplots(figsize=(10, 9))
    bar_c = ["#EF5350" if v < 0 else "#43A047" for v in corr_pp]
    corr_pp.plot(kind="barh", ax=ax, color=bar_c)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Feature Correlation with Target (Post-Preprocessing)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson Correlation")
    plt.tight_layout(); savefig("09_feature_correlation_target")


# =============================================================================
#  SECTION 4 — MODEL SELECTION & TRAINING
# =============================================================================
print("\n" + "="*70)
print("  SECTION 4 — MODEL SELECTION & TRAINING")
print("="*70)

results   = {}
histories = {}

def get_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy"  : accuracy_score (y_true, y_pred),
        "Precision" : precision_score(y_true, y_pred, zero_division=0),
        "Recall"    : recall_score   (y_true, y_pred, zero_division=0),
        "F1"        : f1_score       (y_true, y_pred, zero_division=0),
        "ROC-AUC"   : roc_auc_score  (y_true, y_prob) if y_prob is not None else np.nan,
        "PR-AUC"    : average_precision_score(y_true, y_prob) if y_prob is not None else np.nan,
    }

def evaluate(model, Xtr, ytr, Xva, yva, Xte, yte):
    model.fit(Xtr, ytr)
    pv  = model.predict(Xva);  pt  = model.predict(Xte)
    ppv = model.predict_proba(Xva)[:,1] if hasattr(model,"predict_proba") else None
    ppt = model.predict_proba(Xte)[:,1] if hasattr(model,"predict_proba") else None
    return {"model": model,
            "val":  get_metrics(yva, pv,  ppv),
            "test": get_metrics(yte, pt,  ppt),
            "proba_test": ppt, "preds_test": pt}

# ── 4.1  Classic ML Models ───────────────────────────────────────────────────
ML_MODELS = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Naive Bayes"         : GaussianNB(),
    "KNN (k=5)"           : KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Decision Tree (CART)": DecisionTreeClassifier(max_depth=8, random_state=42),
    "SVM (RBF)"           : SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=200, max_depth=12,
                                                   n_jobs=-1, random_state=42),
    "Extra Trees"         : ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "AdaBoost"            : AdaBoostClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                       random_state=42),
    "XGBoost"             : xgb.XGBClassifier(n_estimators=300, max_depth=6,
                                               learning_rate=0.05, subsample=0.8,
                                               colsample_bytree=0.8,
                                               eval_metric="logloss",
                                               random_state=42, n_jobs=-1),
    "LightGBM"            : lgb.LGBMClassifier(n_estimators=300, max_depth=6,
                                                learning_rate=0.05, subsample=0.8,
                                                colsample_bytree=0.8, random_state=42,
                                                n_jobs=-1, verbose=-1),
    "MLP (sklearn)"       : MLPClassifier(hidden_layer_sizes=(128,64,32),
                                          activation="relu", max_iter=300,
                                          random_state=42, early_stopping=True,
                                          validation_fraction=0.1),
}

print("\n── Training Classic ML Models ────────────────────────────────────────")
for name, model in tqdm(ML_MODELS.items(), desc="Classic ML"):
    print(f"\n  ▸ {name}")
    res = evaluate(model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test)
    results[name] = res
    print(f"    Val  Acc={res['val']['Accuracy']:.4f}  AUC={res['val']['ROC-AUC']:.4f}")
    print(f"    Test Acc={res['test']['Accuracy']:.4f}  AUC={res['test']['ROC-AUC']:.4f}")


# ── 4.2  Deep Learning Models ────────────────────────────────────────────────
print("\n── Training Deep Learning Models ─────────────────────────────────────")
input_dim = X_train_s.shape[1]

def build_shallow_dnn(d):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x); x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m

def build_deep_dnn(d):
    reg = regularizers.l2(1e-4)
    inp = keras.Input(shape=(d,))
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.35)(x)
    x = layers.Dense(64,  activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x); x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m

def build_residual_dnn(d):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(128, activation="relu")(inp); x = layers.BatchNormalization()(x)
    sc = x
    x = layers.Dense(128, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Add()([x, sc]); x = layers.Activation("relu")(x)
    sc = layers.Dense(64)(x)
    x = layers.Dense(64, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Add()([x, sc]); x = layers.Activation("relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(5e-4), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m

# FUTURE WORK: LSTM (treat each feature as a time step)
def build_lstm(d):
    inp = keras.Input(shape=(d, 1))   # (timesteps=features, features=1)
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m

# FUTURE WORK: 1-D CNN (treat features as 1-D spatial signal)
def build_cnn1d(d):
    inp = keras.Input(shape=(d, 1))
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m

DL_CONFIGS = {
    "Shallow DNN"  : (build_shallow_dnn, False),
    "Deep DNN"     : (build_deep_dnn,    False),
    "Residual DNN" : (build_residual_dnn,False),
    "LSTM"         : (build_lstm,        True),   # True = needs 3D input
    "1-D CNN"      : (build_cnn1d,       True),
}

es = callbacks.EarlyStopping(monitor="val_auc", patience=15,
                              restore_best_weights=True, mode="max")
rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                  patience=7, min_lr=1e-6)

for dl_name, (builder, needs_3d) in DL_CONFIGS.items():
    print(f"\n  ▸ {dl_name}")
    m = builder(input_dim)

    Xtr = X_train_s[:,:,np.newaxis] if needs_3d else X_train_s
    Xva = X_val_s  [:,:,np.newaxis] if needs_3d else X_val_s
    Xte = X_test_s [:,:,np.newaxis] if needs_3d else X_test_s

    h = m.fit(Xtr, y_train, validation_data=(Xva, y_val),
              epochs=100, batch_size=256, callbacks=[es, rl], verbose=0)
    histories[dl_name] = h.history

    ppt = m.predict(Xte, verbose=0).ravel()
    ppv = m.predict(Xva, verbose=0).ravel()
    results[dl_name] = {
        "model": m,
        "val":  get_metrics(y_val,  (ppv>=0.5).astype(int), ppv),
        "test": get_metrics(y_test, (ppt>=0.5).astype(int), ppt),
        "proba_test": ppt, "preds_test": (ppt>=0.5).astype(int),
    }
    print(f"    Val  Acc={results[dl_name]['val']['Accuracy']:.4f}  "
          f"AUC={results[dl_name]['val']['ROC-AUC']:.4f}")
    print(f"    Test Acc={results[dl_name]['test']['Accuracy']:.4f}  "
          f"AUC={results[dl_name]['test']['ROC-AUC']:.4f}")


# ── 4.3  FUTURE WORK: PCA + Supervised Models ────────────────────────────────
print("\n── Future Work: PCA + Supervised Models ──────────────────────────────")
N_PCA = 10   # retain top-10 principal components

pca = PCA(n_components=N_PCA, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_val_pca   = pca.transform(X_val_s)
X_test_pca  = pca.transform(X_test_s)

explained = pca.explained_variance_ratio_.cumsum()
print(f"  PCA: {N_PCA} components explain {explained[-1]*100:.1f}% of variance")

PCA_MODELS = {
    "PCA + Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "PCA + Random Forest"       : RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                          random_state=42),
    "PCA + XGBoost"             : xgb.XGBClassifier(n_estimators=200, eval_metric="logloss",
                                                      random_state=42, n_jobs=-1),
}
for name, model in PCA_MODELS.items():
    print(f"\n  ▸ {name}")
    res = evaluate(model, X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test)
    results[name] = res
    print(f"    Test Acc={res['test']['Accuracy']:.4f}  AUC={res['test']['ROC-AUC']:.4f}")

# PCA Variance Plot
with plt.style.context(PLOT_STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Future Work: PCA Analysis", fontsize=14, fontweight="bold")
    axes[0].bar(range(1, N_PCA+1), pca.explained_variance_ratio_*100,
                color="#1976D2", edgecolor="white")
    axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Variance Explained (%)")
    axes[0].set_title("Individual Explained Variance per Component")
    axes[0].set_xticks(range(1, N_PCA+1))

    axes[1].plot(range(1, N_PCA+1), explained*100, "o-",
                 color="#E53935", linewidth=2.5, markersize=7)
    axes[1].axhline(95, color="gray", linestyle="--", linewidth=1, label="95% threshold")
    axes[1].fill_between(range(1, N_PCA+1), explained*100, alpha=0.15, color="#E53935")
    axes[1].set_xlabel("Number of Components"); axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_title("Cumulative Explained Variance"); axes[1].legend()
    axes[1].set_xticks(range(1, N_PCA+1))
    plt.tight_layout(); savefig("10a_pca_variance")


# ── 4.4  FUTURE WORK: SVD + Supervised Models ────────────────────────────────
print("\n── Future Work: SVD (TruncatedSVD) + Supervised Models ───────────────")
N_SVD = 10

svd = TruncatedSVD(n_components=N_SVD, random_state=42)
X_train_svd = svd.fit_transform(X_train_s)
X_val_svd   = svd.transform(X_val_s)
X_test_svd  = svd.transform(X_test_s)

print(f"  SVD: {N_SVD} components explain "
      f"{svd.explained_variance_ratio_.sum()*100:.1f}% of variance")

SVD_MODELS = {
    "SVD + Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVD + Random Forest"       : RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                          random_state=42),
    "SVD + XGBoost"             : xgb.XGBClassifier(n_estimators=200, eval_metric="logloss",
                                                      random_state=42, n_jobs=-1),
}
for name, model in SVD_MODELS.items():
    print(f"\n  ▸ {name}")
    res = evaluate(model, X_train_svd, y_train, X_val_svd, y_val, X_test_svd, y_test)
    results[name] = res
    print(f"    Test Acc={res['test']['Accuracy']:.4f}  AUC={res['test']['ROC-AUC']:.4f}")


# ── 4.5  FUTURE WORK: Clustering Analysis (K-Means + K-Medoids) ─────────────
print("\n── Future Work: Clustering Analysis — K-Means & K-Medoids ────────────")

# Combine all scaled data for clustering
X_all_s = np.vstack([X_train_s, X_val_s, X_test_s])
y_all   = np.concatenate([y_train, y_val, y_test])

# ── K-Means ───────────────────────────────────────────────────────────────────
print("\n  ▸ K-Means Clustering (k=2, 3, 4, 5)")
inertias    = []
silhouettes = []
K_RANGE     = range(2, 7)

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_all_s)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_all_s, labels, sample_size=2000,
                                        random_state=42))
    print(f"    k={k}  Inertia={km.inertia_:,.0f}  "
          f"Silhouette={silhouettes[-1]:.4f}")

# Best k by silhouette
best_k    = K_RANGE[np.argmax(silhouettes)]
km_best   = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = km_best.fit_predict(X_all_s)
print(f"\n  Best K-Means k={best_k}  "
      f"Silhouette={max(silhouettes):.4f}")

# Flood rate per cluster
print("\n  Flood rate per K-Means cluster:")
for c in range(best_k):
    mask = km_labels == c
    rate = y_all[mask].mean()
    print(f"    Cluster {c}: {mask.sum():4d} samples  |  "
          f"Flood rate = {rate:.3f} ({rate*100:.1f}%)")

# ── K-Medoids ─────────────────────────────────────────────────────────────────
if HAS_KMEDOIDS:
    print("\n  ▸ K-Medoids Clustering (k=2)")
    kmed = KMedoids(n_clusters=2, random_state=42)
    kmed_labels = kmed.fit_predict(X_all_s)
    kmed_sil = silhouette_score(X_all_s, kmed_labels, sample_size=2000, random_state=42)
    print(f"    Silhouette = {kmed_sil:.4f}")
    print("\n  Flood rate per K-Medoids cluster:")
    for c in range(2):
        mask = kmed_labels == c
        rate = y_all[mask].mean()
        print(f"    Cluster {c}: {mask.sum():4d} samples  |  "
              f"Flood rate = {rate:.3f} ({rate*100:.1f}%)")
else:
    kmed_labels = None

# Clustering Plot
with plt.style.context(PLOT_STYLE):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Future Work: Clustering Analysis",
                 fontsize=14, fontweight="bold")

    # Elbow curve
    axes[0].plot(list(K_RANGE), inertias, "o-", color="#1565C0",
                 linewidth=2.5, markersize=8)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    axes[0].set_title("K-Means Elbow Curve")
    axes[0].set_xticks(list(K_RANGE))

    # Silhouette scores
    axes[1].bar(list(K_RANGE), silhouettes, color="#43A047", edgecolor="white")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("K-Means Silhouette Scores")
    axes[1].set_xticks(list(K_RANGE))
    for i, (k, s) in enumerate(zip(K_RANGE, silhouettes)):
        axes[1].text(k, s+0.002, f"{s:.3f}", ha="center", fontsize=9)

    # PCA 2-D projection coloured by best-k cluster
    pca2 = PCA(n_components=2, random_state=42)
    X2   = pca2.fit_transform(X_all_s)
    scatter_colors = plt.cm.tab10(np.linspace(0, 0.5, best_k))
    for c in range(best_k):
        mask = km_labels == c
        axes[2].scatter(X2[mask, 0], X2[mask, 1], s=2, alpha=0.4,
                        color=scatter_colors[c], label=f"Cluster {c}")
    axes[2].set_xlabel("PC-1"); axes[2].set_ylabel("PC-2")
    axes[2].set_title(f"K-Means (k={best_k}) — PCA 2-D Projection")
    axes[2].legend(markerscale=5, fontsize=8)

    plt.tight_layout(); savefig("10b_clustering_analysis")

# Flood Risk Zone Map per Cluster
with plt.style.context(PLOT_STYLE):
    fig, axes = plt.subplots(1, best_k, figsize=(6*best_k, 6))
    if best_k == 1: axes = [axes]
    fig.suptitle("K-Means Flood Risk Zones (Geographic)",
                 fontsize=14, fontweight="bold")
    cluster_colors = plt.cm.tab10(np.linspace(0, 0.5, best_k))
    lat_all = np.concatenate([X_train["Latitude"].values,
                               X_val  ["Latitude"].values,
                               X_test ["Latitude"].values])
    lon_all = np.concatenate([X_train["Longitude"].values,
                               X_val  ["Longitude"].values,
                               X_test ["Longitude"].values])
    for c in range(best_k):
        mask = km_labels == c
        flood_rate = y_all[mask].mean()
        axes[c].scatter(lon_all[mask], lat_all[mask],
                        c=[cluster_colors[c]], s=3, alpha=0.5)
        axes[c].set_title(f"Cluster {c}\nFlood Rate={flood_rate:.2%}",
                          fontweight="bold")
        axes[c].set_xlabel("Longitude"); axes[c].set_ylabel("Latitude")
    plt.tight_layout(); savefig("10c_cluster_geo_map")


# =============================================================================
#  SECTION 5 — METRICS VERIFICATION & PLOTS
# =============================================================================
print("\n" + "="*70)
print("  SECTION 5 — METRICS VERIFICATION & PLOTS")
print("="*70)

metric_keys = ["Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"]
df_test_met = pd.DataFrame(
    {m: {n: results[n]["test"][m] for n in results} for m in metric_keys}).T
df_val_met  = pd.DataFrame(
    {m: {n: results[n]["val"][m]  for n in results} for m in metric_keys}).T

print("\n── Test Metrics ──────────────────────────────────────────────────────")
print(df_test_met.to_string(float_format="{:.4f}".format))
df_test_met.to_csv(OUTPUT_DIR / "model_test_metrics.csv")
print(f"\n  [saved] {OUTPUT_DIR}/model_test_metrics.csv")

model_colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

# PLOT 10: Metrics Comparison
with plt.style.context(PLOT_STYLE):
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()
    fig.suptitle("Test-Set Metrics — All Models", fontsize=15, fontweight="bold")
    for i, metric in enumerate(metric_keys):
        vals  = [results[n]["test"][metric] for n in results]
        names = list(results.keys())
        bars  = axes[i].barh(names, vals, color=model_colors, edgecolor="white")
        axes[i].set_xlim(0, 1)
        axes[i].set_title(metric, fontsize=12, fontweight="bold")
        axes[i].axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        for bar, val in zip(bars, vals):
            axes[i].text(val+0.005, bar.get_y()+bar.get_height()/2,
                         f"{val:.3f}", va="center", fontsize=6.5)
        axes[i].invert_yaxis()
    plt.tight_layout(); savefig("11_metrics_comparison")

# PLOT 11: ROC Curves
with plt.style.context(PLOT_STYLE):
    fig, ax = plt.subplots(figsize=(13, 10))
    for (name, res), color in zip(results.items(), model_colors):
        if res["proba_test"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["proba_test"])
            ax.plot(fpr, tpr, color=color, linewidth=1.5,
                    label=f"{name} ({res['test']['ROC-AUC']:.3f})")
    ax.plot([0,1],[0,1],"k--",linewidth=1,label="Random")
    ax.set_xlabel("False Positive Rate",fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    plt.tight_layout(); savefig("12_roc_curves")

# PLOT 12: PR Curves
with plt.style.context(PLOT_STYLE):
    fig, ax = plt.subplots(figsize=(13, 10))
    for (name, res), color in zip(results.items(), model_colors):
        if res["proba_test"] is not None:
            prec, rec, _ = precision_recall_curve(y_test, res["proba_test"])
            ax.plot(rec, prec, color=color, linewidth=1.5,
                    label=f"{name} ({res['test']['PR-AUC']:.3f})")
    ax.axhline(y_test.mean(), color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    plt.tight_layout(); savefig("13_pr_curves")

# PLOT 13: Confusion Matrices (Top 6)
top6 = sorted(results, key=lambda n: results[n]["test"]["F1"], reverse=True)[:6]
with plt.style.context(PLOT_STYLE):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle("Confusion Matrices — Top 6 Models (by F1)",
                 fontsize=14, fontweight="bold")
    for i, name in enumerate(top6):
        cm   = confusion_matrix(y_test, results[name]["preds_test"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Flood","Flood"])
        disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
        axes[i].set_title(f"{name}\nF1={results[name]['test']['F1']:.4f}", fontsize=9)
    plt.tight_layout(); savefig("14_confusion_matrices_top6")

# PLOT 14: DL Training History
with plt.style.context(PLOT_STYLE):
    n_dl = len(histories)
    fig, axes = plt.subplots(n_dl, 2, figsize=(14, n_dl*4))
    fig.suptitle("Deep Learning Training History", fontsize=14, fontweight="bold")
    for i, (name, hist) in enumerate(histories.items()):
        axes[i,0].plot(hist["loss"],     label="Train", linewidth=2)
        axes[i,0].plot(hist["val_loss"], label="Val",   linewidth=2, linestyle="--")
        axes[i,0].set_title(f"{name} — Loss"); axes[i,0].legend()
        axes[i,1].plot(hist["auc"],     label="Train", linewidth=2)
        axes[i,1].plot(hist["val_auc"], label="Val",   linewidth=2, linestyle="--")
        axes[i,1].set_title(f"{name} — AUC"); axes[i,1].legend()
    plt.tight_layout(); savefig("15_dl_training_history")

# PLOT 15: Feature Importances
with plt.style.context(PLOT_STYLE):
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle("Feature Importances — Ensemble Models",
                 fontsize=14, fontweight="bold")
    for i, mname in enumerate(["Random Forest","XGBoost","LightGBM"]):
        fi_df = pd.Series(results[mname]["model"].feature_importances_,
                          index=feature_names).sort_values(ascending=True)
        fi_df.plot(kind="barh", ax=axes[i], color="#1976D2", edgecolor="white")
        axes[i].set_title(mname, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Importance Score")
    plt.tight_layout(); savefig("16_feature_importances")

# PLOT 16: Model × Metric Heatmap
with plt.style.context(PLOT_STYLE):
    hm = pd.DataFrame({n:[results[n]["test"][m] for m in metric_keys]
                       for n in results}, index=metric_keys)
    fig, ax = plt.subplots(figsize=(22, 5))
    sns.heatmap(hm, annot=True, fmt=".3f", cmap="YlGnBu",
                linewidths=0.4, ax=ax, cbar_kws={"shrink":0.8})
    ax.set_title("Model × Metric Heatmap (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig("17_model_metric_heatmap")


# =============================================================================
#  SECTION 6 — BEST MODEL DETERMINATION
# =============================================================================
print("\n" + "="*70)
print("  SECTION 6 — BEST MODEL DETERMINATION")
print("="*70)

WEIGHTS = {"ROC-AUC":0.30,"F1":0.25,"Accuracy":0.20,
           "Precision":0.10,"Recall":0.10,"PR-AUC":0.05}

composite_scores = {
    name: round(sum(results[name]["test"][m]*w for m,w in WEIGHTS.items()), 6)
    for name in results
}
df_scores = pd.Series(composite_scores).sort_values(ascending=False)

print("\n── Composite Weighted Score Ranking ──────────────────────────────────")
print("  Weights: ROC-AUC=30%  F1=25%  Accuracy=20%  "
      "Precision=10%  Recall=10%  PR-AUC=5%\n")
for rank, (name, score) in enumerate(df_scores.items(), 1):
    m = results[name]["test"]
    medal = ["🥇","🥈","🥉"][rank-1] if rank <= 3 else f"  {rank}."
    print(f"  {medal} {name:<35} | Score={score:.4f}  "
          f"Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}  AUC={m['ROC-AUC']:.4f}")

best_name = df_scores.idxmax()
best_res  = results[best_name]
best_m    = best_res["test"]

print("\n" + "="*70)
print(f"  BEST MODEL : {best_name}")
print("="*70)
print(f"  Composite Score : {composite_scores[best_name]:.4f}")
print(f"  Accuracy        : {best_m['Accuracy']:.4f}")
print(f"  Precision       : {best_m['Precision']:.4f}")
print(f"  Recall          : {best_m['Recall']:.4f}")
print(f"  F1 Score        : {best_m['F1']:.4f}")
print(f"  ROC-AUC         : {best_m['ROC-AUC']:.4f}")
print(f"  PR-AUC          : {best_m['PR-AUC']:.4f}")
print(f"\n  Classification Report:\n")
print(classification_report(y_test, best_res["preds_test"],
                             target_names=["No Flood","Flood"]))

# Cross-validation on best model
if hasattr(best_res["model"], "fit") and not isinstance(best_res["model"], keras.Model):
    print("── 5-Fold Cross-Validation (Best Model) ──────────────────────────────")
    X_full = np.vstack([X_train_s, X_val_s, X_test_s])
    y_full = np.concatenate([y_train, y_val, y_test])
    cv = cross_val_score(best_res["model"], X_full, y_full,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="roc_auc", n_jobs=-1)
    print(f"  CV ROC-AUC : {cv.round(4)}")
    print(f"  Mean ± Std : {cv.mean():.4f} ± {cv.std():.4f}")

# PLOT 17: Best Model Deep Analysis
with plt.style.context(PLOT_STYLE):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"Best Model Deep Analysis — {best_name}",
                 fontsize=16, fontweight="bold", y=1.01)
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # Confusion Matrix
    ax_cm = fig.add_subplot(gs[0,0])
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, best_res["preds_test"]),
        display_labels=["No Flood","Flood"]
    ).plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title("Confusion Matrix", fontweight="bold")

    # ROC Curve
    ax_roc = fig.add_subplot(gs[0,1])
    fpr, tpr, _ = roc_curve(y_test, best_res["proba_test"])
    ax_roc.plot(fpr, tpr, color="#1565C0", linewidth=2.5,
                label=f"AUC={best_m['ROC-AUC']:.4f}")
    ax_roc.fill_between(fpr, tpr, alpha=0.15, color="#1565C0")
    ax_roc.plot([0,1],[0,1],"k--",linewidth=1,label="Random")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC Curve", fontweight="bold"); ax_roc.legend()

    # PR Curve
    ax_pr = fig.add_subplot(gs[0,2])
    prec, rec, _ = precision_recall_curve(y_test, best_res["proba_test"])
    ax_pr.plot(rec, prec, color="#2E7D32", linewidth=2.5,
               label=f"PR-AUC={best_m['PR-AUC']:.4f}")
    ax_pr.fill_between(rec, prec, alpha=0.15, color="#2E7D32")
    ax_pr.axhline(y_test.mean(), color="gray", linestyle="--")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve", fontweight="bold"); ax_pr.legend()

    # Metric Bars
    ax_mb = fig.add_subplot(gs[1,0])
    m_vals = [best_m[m] for m in metric_keys]
    m_cols = ["#1565C0","#2E7D32","#F57F17","#6A1B9A","#C62828","#00838F"]
    bars = ax_mb.bar(metric_keys, m_vals, color=m_cols, edgecolor="white", width=0.6)
    ax_mb.set_ylim(0,1); ax_mb.axhline(0.5, color="gray", linestyle="--")
    for bar, val in zip(bars, m_vals):
        ax_mb.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                   f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax_mb.set_title("All Metrics — Best Model", fontweight="bold")
    ax_mb.set_xticklabels(metric_keys, rotation=20)

    # Composite Ranking
    ax_rank = fig.add_subplot(gs[1,1:])
    rank_colors = ["#FFD700" if n==best_name else
                   "#C0C0C0" if i==1 else
                   "#CD7F32" if i==2 else "#90CAF9"
                   for i,n in enumerate(df_scores.index)]
    bars = ax_rank.barh(list(df_scores.index), list(df_scores.values),
                        color=rank_colors, edgecolor="white")
    ax_rank.set_xlabel("Composite Weighted Score")
    ax_rank.set_title("Model Ranking by Composite Score", fontweight="bold")
    ax_rank.axvline(df_scores.mean(), color="red", linestyle="--", linewidth=1)
    for bar, (n, val) in zip(bars, df_scores.items()):
        ax_rank.text(val+0.0002, bar.get_y()+bar.get_height()/2,
                     f"{val:.4f}", va="center", fontsize=7)
    ax_rank.invert_yaxis()
    ax_rank.text(df_scores.max()+0.001,
                 list(df_scores.index).index(best_name),
                 " BEST", va="center", color="#B8860B",
                 fontweight="bold", fontsize=10)
    plt.tight_layout(); savefig("18_best_model_analysis")

# PLOT 18: Composite Score Bar
with plt.style.context(PLOT_STYLE):
    fig, ax = plt.subplots(figsize=(16, 7))
    bar_colors = ["#FFD700" if n==best_name else "#90CAF9" for n in df_scores.index]
    bars = ax.bar(range(len(df_scores)), df_scores.values,
                  color=bar_colors, edgecolor="white", width=0.7)
    ax.set_xticks(range(len(df_scores)))
    ax.set_xticklabels(df_scores.index, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Composite Weighted Score", fontsize=12)
    ax.set_title("All Models — Composite Score Ranking\n"
                 "(Future Work: PCA/SVD + Supervised | LSTM | 1-D CNN | K-Means | K-Medoids)",
                 fontsize=13, fontweight="bold")
    ax.axhline(df_scores.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean={df_scores.mean():.4f}")
    ax.set_ylim(df_scores.min()-0.005, df_scores.max()+0.012)
    for bar, (name, val) in zip(bars, df_scores.items()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold" if name==best_name else "normal")
    ax.legend(handles=[
        mpatches.Patch(color="#FFD700", label=f"Best: {best_name}"),
        mpatches.Patch(color="#90CAF9", label="Other Models"),
        plt.Line2D([],[],color="red",linestyle="--",
                   label=f"Mean={df_scores.mean():.4f}")
    ])
    plt.tight_layout(); savefig("19_composite_score_ranking")


# =============================================================================
#  FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("  PIPELINE COMPLETE — FINAL SUMMARY")
print("="*70)
print(f"\n  BEST MODEL        : {best_name}")
print(f"  Composite Score   : {composite_scores[best_name]:.4f}")
print(f"  Accuracy          : {best_m['Accuracy']:.4f}")
print(f"  F1 Score          : {best_m['F1']:.4f}")
print(f"  ROC-AUC           : {best_m['ROC-AUC']:.4f}")

print(f"\n  TOP 3 MODELS:")
for i, (name, score) in enumerate(df_scores.head(3).items(), 1):
    m = results[name]["test"]
    print(f"  {i}. {name:<35} | Composite={score:.4f}  "
          f"Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}  AUC={m['ROC-AUC']:.4f}")

print(f"\n  FUTURE WORK MODELS INCLUDED:")
print(f"    PCA + Supervised  : PCA+LR, PCA+RF, PCA+XGBoost")
print(f"    SVD + Supervised  : SVD+LR, SVD+RF, SVD+XGBoost")
print(f"    Deep Learning     : LSTM, 1-D CNN")
print(f"    Clustering        : K-Means (k={best_k})" +
      (" , K-Medoids" if HAS_KMEDOIDS else " (K-Medoids: install scikit-learn-extra)"))

print(f"\n  OUTPUT FILES:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"    {f}")
print("\n" + "="*70)