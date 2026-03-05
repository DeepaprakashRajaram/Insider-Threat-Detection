import os
import pandas as pd

ANSWERS_PATH = r"C:\Projects\Insider-Threat-Detection\data\raw\answers"
OUTPUT_PATH = r"C:\Projects\Insider-Threat-Detection\data\processed"

SCENARIOS = ["r4.2-1", "r4.2-2", "r4.2-3"]


def extract_malicious_event_ids():
    malicious_ids = set()

    for scenario in SCENARIOS:
        scenario_path = os.path.join(ANSWERS_PATH, scenario)

        if not os.path.exists(scenario_path):
            print(f"⚠️ Scenario folder not found: {scenario}")
            continue

        print(f"Processing scenario: {scenario}")

        for file in os.listdir(scenario_path):
            if file.endswith(".csv"):
                file_path = os.path.join(scenario_path, file)

                try:
                    df = pd.read_csv(file_path, header=None, engine="python", on_bad_lines="skip")
                    # Event ID is in column 1
                    event_ids = df.iloc[:, 1].tolist()
                    malicious_ids.update(event_ids)
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    return malicious_ids


def main():
    malicious_ids = extract_malicious_event_ids()

    print(f"\nTotal malicious events found: {len(malicious_ids)}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    output_file = os.path.join(OUTPUT_PATH, "malicious_event_ids.csv")
    pd.DataFrame({"id": list(malicious_ids)}).to_csv(output_file, index=False)

    print(f"Saved malicious event IDs to: {output_file}")


if __name__ == "__main__":
    main()