import pandas as pd
import joblib

DATA_PATH = "data/processed/final_user_day_features.csv"
MODEL_PATH = "models/xgb_supervised.pkl"

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

feature_cols = [c for c in df.columns if c not in ["user","day","label"]]

importances = model.feature_importances_

importance_df = pd.DataFrame({
    "feature":feature_cols,
    "importance":importances
})

importance_df = importance_df.sort_values("importance",ascending=False)

print("\nTop Behavioral Risk Indicators\n")
print(importance_df.head(10))