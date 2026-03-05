import pandas as pd
import numpy as np
import joblib

DATA_PATH = "data/processed/final_user_day_features.csv"
XGB_MODEL_PATH = "models/xgb_supervised.pkl"
ISO_MODEL_PATH = "models/isolation_forest.pkl"

OUTPUT_PATH = "results/risk_predictions.csv"

ALPHA = 0.85
RISK_THRESHOLD = 80   # alert threshold (0-100 scale)

# Load data
df = pd.read_csv(DATA_PATH)

feature_cols = [c for c in df.columns if c not in ["user", "day", "label"]]

X = df[feature_cols]

# Load models
xgb = joblib.load(XGB_MODEL_PATH)
iso = joblib.load(ISO_MODEL_PATH)

# Supervised probabilities
xgb_probs = xgb.predict_proba(X)[:, 1]

# Isolation Forest anomaly scores
iso_scores = -iso.decision_function(X)
iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

# Hybrid score
hybrid_score = ALPHA * xgb_probs + (1 - ALPHA) * iso_scores

# Convert to 0–100 risk score
risk_score = hybrid_score * 100

df["risk_score"] = risk_score

# Prediction based on threshold
df["alert"] = (df["risk_score"] >= RISK_THRESHOLD).astype(int)

# Sort by highest risk
df = df.sort_values("risk_score", ascending=False)

# Save results
df[["user", "day", "risk_score", "alert"]].to_csv(OUTPUT_PATH, index=False)

print("Risk scoring complete.")
print("Alerts generated:", df["alert"].sum())
print("Results saved to:", OUTPUT_PATH)

# Show top 10 alerts
print("\nTop 10 High-Risk Alerts:")
print(df[["user", "day", "risk_score"]].head(10))