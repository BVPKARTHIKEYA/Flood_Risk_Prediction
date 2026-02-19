import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb


# ==========================================================
# 1️⃣ LOAD TRAIN DATA
# ==========================================================

train_df = pd.read_csv("train.csv")

print("Train Shape:", train_df.shape)

# Separate features and target
X = train_df.drop(columns=["id", "FloodProbability"])
y = train_df["FloodProbability"]

numeric_cols = X.columns.tolist()

# ==========================================================
# 2️⃣ PREPROCESSOR
# ==========================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols)
])

# ==========================================================
# 3️⃣ TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# 4️⃣ MODELS
# ==========================================================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(),
    "LightGBM": lgb.LGBMRegressor()
}

results = {}
trained_pipelines = {}

# ==========================================================
# 5️⃣ TRAINING LOOP
# ==========================================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results[name] = {
        "RMSE": rmse,
        "R2": r2
    }

    trained_pipelines[name] = pipeline

# ==========================================================
# 6️⃣ MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)

print("\nModel Comparison:\n")
print(results_df)

best_model_name = results_df.index[0]
best_pipeline = trained_pipelines[best_model_name]

print("\nBest Model:", best_model_name)

# ==========================================================
# 7️⃣ VISUALIZATIONS
# ==========================================================

preds = best_pipeline.predict(X_test)
residuals = y_test - preds

# 🔹 Target Distribution
plt.figure(figsize=(8,5))
sns.histplot(y, bins=50, kde=True)
plt.title("Flood Probability Distribution")
plt.show()

# 🔹 Actual vs Predicted
plt.figure(figsize=(7,7))
plt.scatter(y_test, preds, alpha=0.3)
plt.plot([0,1],[0,1], color="red")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

# 🔹 Residual Distribution
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=50, kde=True)
plt.title("Residual Distribution")
plt.show()

# 🔹 Prediction Error Plot
plt.figure(figsize=(8,5))
plt.scatter(preds, residuals, alpha=0.3)
plt.axhline(0, color="red")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Prediction Error Plot")
plt.show()

# 🔹 Model Comparison Plot
results_df.plot(kind="bar", figsize=(8,5))
plt.title("Model Performance Comparison")
plt.xticks(rotation=45)
plt.show()

# ==========================================================
# 8️⃣ PREDICT ON TEST DATA
# ==========================================================

test_df = pd.read_csv("test.csv")

test_ids = test_df["id"]
X_test_final = test_df.drop(columns=["id"])

test_preds = best_pipeline.predict(X_test_final)

submission = pd.DataFrame({
    "id": test_ids,
    "FloodProbability": test_preds
})

submission.to_csv("submission.csv", index=False)

print("\nSubmission file created successfully!")
