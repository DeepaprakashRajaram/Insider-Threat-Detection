import os
import pandas as pd
import numpy as np
from tqdm import tqdm

RAW_PATH = "data/raw/r4.2"
LABELS_PATH = "data/processed/user_day_labels.csv"
OUTPUT_PATH = "data/processed/final_user_day_features.csv"

CHUNK_SIZE = 200_000


def process_file(file_path, activity_name):
    results = []

    for chunk in pd.read_csv(
        file_path,
        engine="python",
        on_bad_lines="skip",
        chunksize=CHUNK_SIZE
    ):

        if not {"id", "date", "user"}.issubset(chunk.columns):
            continue

        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk.dropna(subset=["date"])
        chunk["day"] = chunk["date"].dt.date

        if activity_name == "logon":
            chunk["is_logon"] = chunk["activity"] == "Logon"
            chunk["after_hours"] = (
                (chunk["date"].dt.hour < 6) |
                (chunk["date"].dt.hour > 20)
            )

            grouped = chunk.groupby(["user", "day"]).agg(
                logon_count=("is_logon", "sum"),
                after_hours_logon=("after_hours", "sum")
            )

        elif activity_name == "file":
            grouped = chunk.groupby(["user", "day"]).agg(
                file_access_count=("id", "count")
            )

        elif activity_name == "email":
            chunk["after_hours"] = (
                (chunk["date"].dt.hour < 6) |
                (chunk["date"].dt.hour > 20)
            )

            grouped = chunk.groupby(["user", "day"]).agg(
                email_count=("id", "count"),
                email_after_hours=("after_hours", "sum"),
                total_attachment=("attachments", "sum")
            )

        elif activity_name == "http":
            grouped = chunk.groupby(["user", "day"]).agg(
                http_count=("id", "count")
            )

        elif activity_name == "device":
            chunk["usb_connect"] = chunk["activity"] == "Connect"
            grouped = chunk.groupby(["user", "day"]).agg(
                usb_connect_count=("usb_connect", "sum")
            )

        else:
            continue

        results.append(grouped.reset_index())

    if results:
        return pd.concat(results)
    return pd.DataFrame()


def build_features():
    print("Processing logs...")

    features = []

    log_files = {
        "logon": "logon.csv",
        "file": "file.csv",
        "email": "email.csv",
        "http": "http.csv",
        "device": "device.csv"
    }

    for name, filename in log_files.items():
        file_path = os.path.join(RAW_PATH, filename)
        if os.path.exists(file_path):
            print(f"Processing {filename}")
            df = process_file(file_path, name)
            features.append(df)

    # Merge all feature tables
    final_df = features[0]

    for df in features[1:]:
        final_df = final_df.merge(df, on=["user", "day"], how="outer")

    final_df.fillna(0, inplace=True)

    return final_df


def add_deviation_features(df):
    print("Computing deviation features...")

    df = df.sort_values(["user", "day"])

    feature_cols = [
        "logon_count",
        "file_access_count",
        "email_count",
        "http_count",
        "usb_connect_count"
    ]

    for col in feature_cols:
        if col in df.columns:
            df[f"{col}_z"] = (
                df.groupby("user")[col]
                .transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
            )

    return df


def merge_labels(df):
    labels = pd.read_csv(LABELS_PATH)
    labels["day"] = pd.to_datetime(labels["day"]).dt.date

    df = df.merge(labels, on=["user", "day"], how="left")
    df["label"] = df["label"].fillna(0)

    return df


def main():
    df = build_features()
    df = add_deviation_features(df)
    df = merge_labels(df)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved final dataset to:", OUTPUT_PATH)
    print("Total rows:", len(df))
    print("Malicious rows:", df["label"].sum())


if __name__ == "__main__":
    main()