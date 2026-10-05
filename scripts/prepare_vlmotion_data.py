#!/usr/bin/env python3
"""Build a small, scene-balanced VLMotion training dataset."""

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


def scene_group(image_name: str) -> str:
    source = Path(image_name).stem
    match = re.match(r"(.+?)_frame_\d+", source)
    if match:
        return match.group(1)
    return re.sub(r"__aug\d+$", "", source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-count", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.image_count < 1:
        raise ValueError("--image-count must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")

    records = json.loads(args.source_dataset.read_text(encoding="utf-8"))
    records_by_image = defaultdict(list)
    for record in records:
        image_name = record.get("image")
        if isinstance(image_name, str) and image_name:
            records_by_image[image_name].append(record)

    images_by_group = defaultdict(list)
    for image_name in sorted(records_by_image):
        images_by_group[scene_group(image_name)].append(image_name)
    if not images_by_group:
        raise ValueError("No image records found")
    if args.image_count > len(records_by_image):
        raise ValueError(
            f"Requested {args.image_count} images, but only "
            f"{len(records_by_image)} are available"
        )

    rng = random.Random(args.seed)
    groups = sorted(images_by_group)
    for group in groups:
        rng.shuffle(images_by_group[group])

    # Round-robin selection prevents the 100-image smoke dataset from being
    # dominated by only one physical panel or video.
    selected = []
    offset = 0
    while len(selected) < args.image_count:
        made_progress = False
        for group in groups:
            if offset < len(images_by_group[group]):
                selected.append(images_by_group[group][offset])
                made_progress = True
                if len(selected) == args.image_count:
                    break
        if not made_progress:
            break
        offset += 1

    selected_set = set(selected)
    selected_records = [
        record for record in records if record.get("image") in selected_set
    ]

    shuffled_groups = groups[:]
    rng.shuffle(shuffled_groups)
    validation_group_count = max(
        1, round(len(groups) * args.validation_fraction)
    )
    validation_groups = set(shuffled_groups[:validation_group_count])
    train_records = [
        record
        for record in selected_records
        if scene_group(record["image"]) not in validation_groups
    ]
    validation_records = [
        record
        for record in selected_records
        if scene_group(record["image"]) in validation_groups
    ]

    image_output = args.output / "images"
    split_output = args.output / "split"
    image_output.mkdir(parents=True)
    split_output.mkdir(parents=True)
    for image_name in selected:
        source = args.source_images / image_name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = image_output / image_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def write_json(path: Path, value) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    write_json(args.output / "labeled_dataset.json", selected_records)
    write_json(split_output / "train.json", train_records)
    write_json(split_output / "validation.json", validation_records)
    audit = {
        "seed": args.seed,
        "requested_images": args.image_count,
        "selected_images": len(selected),
        "selected_records": len(selected_records),
        "all_groups": groups,
        "train_groups": sorted(set(groups) - validation_groups),
        "validation_groups": sorted(validation_groups),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "group_overlap": [],
    }
    write_json(split_output / "audit.json", audit)
    print(
        f"Prepared {len(selected)} images/{len(selected_records)} records: "
        f"train={len(train_records)}, validation={len(validation_records)}"
    )


if __name__ == "__main__":
    main()
