import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/final_user_day_features.csv"
XGB_MODEL_PATH = "models/xgb_supervised.pkl"
ISO_MODEL_PATH = "models/isolation_forest.pkl"

ALPHA = 0.85  # weight for supervised model

# Load data
df = pd.read_csv(DATA_PATH)

feature_cols = [c for c in df.columns if c not in ["user", "day", "label"]]

X = df[feature_cols]
y = df["label"]

# User-based split (same strategy as supervised model)
unique_users = df["user"].astype(str).unique().tolist()

train_users, test_users = train_test_split(
    unique_users,
    test_size=0.2,
    random_state=42
)

test_mask = df["user"].isin(test_users)

X_test = X[test_mask]
y_test = y[test_mask]

# Load models
xgb = joblib.load(XGB_MODEL_PATH)
iso = joblib.load(ISO_MODEL_PATH)

# Get XGB probabilities
xgb_probs = xgb.predict_proba(X_test)[:, 1]

# Get anomaly scores (higher = more anomalous)
iso_scores = -iso.decision_function(X_test)

# Normalize anomaly scores
iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

# Hybrid score
hybrid_score = ALPHA * xgb_probs + (1 - ALPHA) * iso_scores

# Evaluate AUC
print("Hybrid ROC AUC:", roc_auc_score(y_test, hybrid_score))

# Optimize threshold using F1
precisions, recalls, thresholds = precision_recall_curve(y_test, hybrid_score)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print("Best hybrid threshold:", best_threshold)
print("Best hybrid F1:", f1_scores[best_idx])
print("Precision at best threshold:", precisions[best_idx])
print("Recall at best threshold:", recalls[best_idx])

y_pred = (hybrid_score > best_threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))