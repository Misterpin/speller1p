"""Fit a P300 profile from a marked calibration EDF and separate known labels."""
from pathlib import Path
import argparse
import json
from eeg_speller.core.config import load_config
from eeg_speller.physiology.edf_demo import calibrate


def main():
    parser = argparse.ArgumentParser(description="Calibrate P300 using known target/non-target flashes")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = calibrate(load_config(args.profile), args.edf,
                      json.loads(args.labels.read_text(encoding="utf-8")), args.output)
    print(json.dumps(model["calibration"]))


if __name__ == "__main__":
    main()
