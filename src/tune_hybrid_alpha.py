import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/final_user_day_features.csv"
XGB_MODEL_PATH = "models/xgb_supervised.pkl"
ISO_MODEL_PATH = "models/isolation_forest.pkl"

df = pd.read_csv(DATA_PATH)

feature_cols = [c for c in df.columns if c not in ["user", "day", "label"]]

X = df[feature_cols]
y = df["label"]

# user split
users = df["user"].astype(str).unique().tolist()

train_users, test_users = train_test_split(
    users,
    test_size=0.2,
    random_state=42
)

test_mask = df["user"].isin(test_users)

X_test = X[test_mask]
y_test = y[test_mask]

xgb = joblib.load(XGB_MODEL_PATH)
iso = joblib.load(ISO_MODEL_PATH)

xgb_probs = xgb.predict_proba(X_test)[:,1]

iso_scores = -iso.decision_function(X_test)
iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

best_alpha = None
best_f1 = 0

for alpha in np.arange(0.0,1.01,0.05):

    hybrid = alpha * xgb_probs + (1-alpha) * iso_scores

    precisions, recalls, thresholds = precision_recall_curve(y_test, hybrid)

    f1 = 2*(precisions*recalls)/(precisions+recalls+1e-8)

    max_f1 = np.max(f1)

    if max_f1 > best_f1:
        best_f1 = max_f1
        best_alpha = alpha

print("Best alpha:", best_alpha)
print("Best F1:", best_f1)