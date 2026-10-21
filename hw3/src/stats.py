"""Stage 2: compute simple statistics and save them as a DVC metrics file."""

import csv
import json
import sys


def main(processed_path: str, metrics_path: str) -> None:
    with open(processed_path, newline="") as f:
        temps = [float(r["temperature_c"]) for r in csv.DictReader(f)]

    metrics = {
        "rows": len(temps),
        "mean_temperature_c": round(sum(temps) / len(temps), 3),
        "min_temperature_c": min(temps),
        "max_temperature_c": max(temps),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
