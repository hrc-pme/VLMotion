#!/usr/bin/env python3
"""Run SAM3 geometry -> localized VLM checks -> conformal count sets."""

import sys

from point.serve.run_counting_pass1 import main as pass1_main


def main():
    if "--counting-method" not in sys.argv:
        sys.argv.extend(["--counting-method", "sam3_localized_conformal"])
    if "--output" not in sys.argv:
        sys.argv.extend(["--output", "calibration_set/localized_conformal_results.jsonl"])
    if "--output-json" not in sys.argv:
        sys.argv.extend(["--output-json", "calibration_set/localized_conformal_results.json"])
    pass1_main()


if __name__ == "__main__":
    main()
