import os
import pandas as pd
from tqdm import tqdm

RAW_PATH = "data/raw/r4.2"
MALICIOUS_IDS_PATH = "data/processed/malicious_event_ids.csv"
OUTPUT_PATH = "data/processed/user_day_labels.csv"

malicious_ids = set(
    pd.read_csv(MALICIOUS_IDS_PATH)["id"].astype(str)
)

print(f"Loaded {len(malicious_ids)} malicious event IDs")

user_day_dict = {}

CHUNK_SIZE = 200_000

for root, dirs, files in os.walk(RAW_PATH):
    for file in files:
        if file.endswith(".csv"):

            file_path = os.path.join(root, file)
            print(f"Processing {file}...")

            try:
                for chunk in pd.read_csv(
                    file_path,
                    engine="python",
                    on_bad_lines="skip",
                    chunksize=CHUNK_SIZE
                ):

                    if not {"id", "date", "user"}.issubset(chunk.columns):
                        continue

                    chunk["id"] = chunk["id"].astype(str)
                    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                    chunk = chunk.dropna(subset=["date"])
                    chunk["day"] = chunk["date"].dt.date

                    # mark malicious
                    chunk["is_malicious"] = chunk["id"].isin(malicious_ids)

                    # group per user-day
                    grouped = (
                        chunk.groupby(["user", "day"])["is_malicious"]
                        .max()
                        .reset_index()
                    )

                    for _, row in grouped.iterrows():
                        key = (row["user"], row["day"])

                        if key not in user_day_dict:
                            user_day_dict[key] = int(row["is_malicious"])
                        else:
                            user_day_dict[key] = max(
                                user_day_dict[key],
                                int(row["is_malicious"])
                            )

            except Exception as e:
                print(f"Error reading {file}: {e}")

labels_df = pd.DataFrame(
    [(user, day, label) for (user, day), label in user_day_dict.items()],
    columns=["user", "day", "label"]
)

labels_df.to_csv(OUTPUT_PATH, index=False)

print("Saved user-day labels to:", OUTPUT_PATH)
print("Total user-days:", len(labels_df))
print("Malicious user-days:", labels_df["label"].sum())