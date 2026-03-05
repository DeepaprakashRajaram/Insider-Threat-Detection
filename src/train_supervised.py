import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from xgboost import XGBClassifier
import os
import joblib

DATA_PATH = "data/processed/final_user_day_features.csv"
MODEL_PATH = "models/xgb_supervised.pkl"

os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Remove non-feature columns
feature_cols = [col for col in df.columns if col not in ["user", "day", "label"]]

X = df[feature_cols]
y = df["label"]

# User-based split
unique_users = df["user"].astype(str).unique().tolist()
train_users, test_users = train_test_split(
    unique_users,
    test_size=0.2,
    random_state=42
)

train_idx = df["user"].isin(train_users)
test_idx = df["user"].isin(test_users)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("Train positives:", y_train.sum())
print("Test positives:", y_test.sum())

# Handle imbalance
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
y_probs = model.predict_proba(X_test)[:, 1]
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print("\nBest threshold (F1 optimized):", best_threshold)
print("Best F1:", f1_scores[best_idx])
print("Precision at best threshold:", precisions[best_idx])
print("Recall at best threshold:", recalls[best_idx])

# Apply optimal threshold
y_pred = (y_probs > best_threshold).astype(int)

# Save model
joblib.dump(model, MODEL_PATH)
print("\nModel saved to:", MODEL_PATH)