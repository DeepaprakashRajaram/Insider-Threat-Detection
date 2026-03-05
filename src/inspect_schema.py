import os
import pandas as pd

DATA_PATH = r"C:\Projects\Insider-Threat-Detection\data\raw\r4.2"

FILES = [
    "logon.csv",
    "file.csv",
    "device.csv",
    "http.csv",
    "email.csv",
    "psychometric.csv"
]

def inspect_file(filename):
    path = os.path.join(DATA_PATH, filename)

    if not os.path.exists(path):
        print(f"\n❌ {filename} not found.\n")
        return

    print(f"\n===== {filename} =====")
    try:
        df = pd.read_csv(path, nrows=5)
        print("Columns:")
        print(df.columns.tolist())
        print("\nSample Rows:")
        print(df.head())
    except Exception as e:
        print(f"Error reading {filename}: {e}")


def main():
    for file in FILES:
        inspect_file(file)


if __name__ == "__main__":
    main()