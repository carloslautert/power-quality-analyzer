"""Regenerate the versioned demonstration CSV files."""

from pathlib import Path

from src.signal_generator import generate_scenario

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output_directory = ROOT / "data"
    output_directory.mkdir(exist_ok=True)
    for scenario in ("normal", "harmonics", "voltage_sag", "voltage_swell"):
        generate_scenario(scenario).to_csv(output_directory / f"{scenario}.csv", index=False)


if __name__ == "__main__":
    main()

