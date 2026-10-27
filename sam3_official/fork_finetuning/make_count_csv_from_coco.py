#!/usr/bin/env python3
"""Create an image-level count CSV from COCO instance annotations."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    counts = Counter(int(annotation["image_id"]) for annotation in coco["annotations"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["images", "gt_count"])
        writer.writeheader()
        for image in sorted(coco["images"], key=lambda item: item["file_name"]):
            writer.writerow({
                "images": image["file_name"],
                "gt_count": counts[int(image["id"])],
            })


if __name__ == "__main__":
    main()
