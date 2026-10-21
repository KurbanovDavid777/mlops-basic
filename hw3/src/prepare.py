"""Stage 1: min-max normalize the temperature column."""

import csv
import sys


def normalize(values: list[float]) -> list[float]:
    """Scale values to the [0, 1] range."""
    low, high = min(values), max(values)
    return [(v - low) / (high - low) for v in values]


def main(raw_path: str, out_path: str) -> None:
    with open(raw_path, newline="") as f:
        rows = list(csv.DictReader(f))

    temps = [float(r["temperature_c"]) for r in rows]
    scaled = normalize(temps)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "temperature_c", "temperature_norm"])
        for row, norm in zip(rows, scaled, strict=True):
            writer.writerow([row["date"], row["temperature_c"], f"{norm:.4f}"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
