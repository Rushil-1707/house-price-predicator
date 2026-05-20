# ─────────────────────────────────────────────
# House Price Prediction - Model Training
# Author: Rushil Popat
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# ── 1. Load Dataset ──────────────────────────
# Using California Housing dataset (built into sklearn - no download needed)
print("Loading dataset...")
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target  # Price is in $100,000s

print(f"Dataset shape: {df.shape}")
print(df.head())

# ── 2. Feature Engineering ───────────────────
# Features we will use for prediction
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
X = df[features]
y = df['Price']

print(f"\nFeatures: {features}")
print(f"Target: Price (in $100,000s)")

# ── 3. Train-Test Split ──────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ── 4. Feature Scaling ───────────────────────
# StandardScaler makes all features on same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 5. Train Models ──────────────────────────

# Model 1: Linear Regression (simple baseline)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)
lr_r2    = r2_score(y_test, lr_preds)
lr_rmse  = np.sqrt(mean_squared_error(y_test, lr_preds))

print(f"\n── Linear Regression ──")
print(f"R² Score : {lr_r2:.4f}")
print(f"RMSE     : {lr_rmse:.4f}")

# Model 2: Random Forest (better accuracy)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
rf_r2    = r2_score(y_test, rf_preds)
rf_rmse  = np.sqrt(mean_squared_error(y_test, rf_preds))

print(f"\n── Random Forest ──")
print(f"R² Score : {rf_r2:.4f}")
print(f"RMSE     : {rf_rmse:.4f}")

# ── 6. Save Best Model ───────────────────────
# Random Forest performs better, so we save that
print("\nSaving model and scaler...")
with open("model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ model.pkl saved")
print("✅ scaler.pkl saved")
print("\nModel training complete!")
