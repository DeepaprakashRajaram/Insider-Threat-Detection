import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
import joblib

DATA_PATH = "data/processed/final_user_day_features.csv"
MODEL_PATH = "models/isolation_forest.pkl"

# Load data
df = pd.read_csv(DATA_PATH)

feature_cols = [c for c in df.columns if c not in ["user", "day", "label"]]

X = df[feature_cols]
y = df["label"]

# Train only on normal data
X_normal = X[y == 0]

print("Training Isolation Forest on normal samples:", len(X_normal))

iso = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

iso.fit(X_normal)

# Compute anomaly scores on full dataset
scores = -iso.decision_function(X)  # higher = more anomalous

# Normalize scores
scores = (scores - scores.min()) / (scores.max() - scores.min())

# Evaluate
print("\nROC AUC (anomaly model):", roc_auc_score(y, scores))

joblib.dump(iso, MODEL_PATH)
print("Isolation Forest saved to:", MODEL_PATH)