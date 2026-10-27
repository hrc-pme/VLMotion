#!/usr/bin/env python3
"""Prepare a balanced fork/scissors bbox-only COCO dataset for SAM3."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


CONCEPTS = {
    "fork": {"category_names": {"fork", "forks"}, "category_id": 1},
    "scissors": {"category_names": {"scissor", "scissors"}, "category_id": 2},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork-source", type=Path, required=True)
    parser.add_argument("--scissors-source", type=Path, required=True)
    parser.add_argument(
        "--extra-fork-train-source",
        type=Path,
        help="Additional fork COCO source whose every positive image goes to train.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-concept", type=int, default=14)
    return parser.parse_args()


def validate_bbox(annotation):
    bbox = annotation.get("bbox")
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
        and bbox[2] > 0
        and bbox[3] > 0
    )


def load_concept(source_root, concept, category_names=None):
    source_dir = source_root / "train"
    annotation_path = source_dir / "_annotations.coco.json"
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    valid_names = category_names or CONCEPTS[concept]["category_names"]
    category_ids = {
        int(category["id"])
        for category in coco.get("categories", [])
        if str(category["name"]).strip().lower() in valid_names
    }
    if not category_ids:
        raise ValueError(f"No {concept} category found in {annotation_path}")

    images = {int(image["id"]): image for image in coco.get("images", [])}
    annotations = defaultdict(list)
    for annotation in coco.get("annotations", []):
        if int(annotation["category_id"]) in category_ids:
            if not validate_bbox(annotation):
                raise ValueError(
                    f"Invalid bbox in annotation {annotation.get('id')} from {annotation_path}"
                )
            annotations[int(annotation["image_id"])].append(annotation)

    positive_ids = sorted(image_id for image_id in images if annotations[image_id])
    if len(positive_ids) < 2:
        raise ValueError(f"Need at least two positive {concept} images")
    return source_dir, images, annotations, positive_ids


def split_ids(ids, train_fraction, rng):
    shuffled = list(ids)
    rng.shuffle(shuffled)
    train_count = round(len(shuffled) * train_fraction)
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    return shuffled[:train_count], shuffled[train_count:]


def interleave_entries(ids_by_concept):
    entries = []
    max_count = max(len(ids) for ids in ids_by_concept.values())
    for index in range(max_count):
        for concept in CONCEPTS:
            if index < len(ids_by_concept[concept]):
                entries.append((concept, ids_by_concept[concept][index]))
    return entries


def write_split(output_root, split, entries, loaded):
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images = []
    output_annotations = []
    next_image_id = 1
    next_annotation_id = 1

    for sample_index, (source_key, source_image_id) in enumerate(entries):
        concept = "fork" if source_key == "fork_plate" else source_key
        source_dir, images, annotations, _ = loaded[source_key]
        source_image = images[source_image_id]
        source_path = (source_dir / source_image["file_name"]).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        extension = source_path.suffix.lower()
        output_name = f"{source_key}_{sample_index:04d}{extension}"
        output_path = output_dir / output_name
        if output_path.is_symlink() or output_path.exists():
            output_path.unlink()
        output_path.symlink_to(source_path)

        image = dict(source_image)
        image["id"] = next_image_id
        image["file_name"] = output_name
        image["source_image_id"] = source_image_id
        image["concept"] = concept
        output_images.append(image)

        for source_annotation in annotations[source_image_id]:
            annotation = dict(source_annotation)
            annotation["id"] = next_annotation_id
            annotation["image_id"] = next_image_id
            annotation["category_id"] = CONCEPTS[concept]["category_id"]
            annotation["noun_phrase"] = concept
            annotation["segmentation"] = []
            annotation["iscrowd"] = 0
            output_annotations.append(annotation)
            next_annotation_id += 1
        next_image_id += 1

    categories = [
        {"id": values["category_id"], "name": concept, "supercategory": "object"}
        for concept, values in CONCEPTS.items()
    ]
    output_coco = {
        "images": output_images,
        "annotations": output_annotations,
        "categories": categories,
    }
    (output_dir / "_annotations.coco.json").write_text(
        json.dumps(output_coco, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    counts = defaultdict(lambda: {"images": 0, "annotations": 0})
    for image in output_images:
        counts[image["concept"]]["images"] += 1
    category_to_concept = {values["category_id"]: concept for concept, values in CONCEPTS.items()}
    for annotation in output_annotations:
        counts[category_to_concept[annotation["category_id"]]]["annotations"] += 1
    print(f"{split}: {dict(counts)}")
    return dict(counts)


def main():
    args = parse_args()
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between zero and one")
    if args.samples_per_concept < 1:
        raise ValueError("--samples-per-concept must be positive")

    loaded = {
        "fork": load_concept(args.fork_source, "fork"),
        "scissors": load_concept(args.scissors_source, "scissors"),
    }
    if args.extra_fork_train_source is not None:
        loaded["fork_plate"] = load_concept(
            args.extra_fork_train_source,
            "fork",
            category_names={"forks-plate"},
        )
    train_ids = {}
    test_ids = {}
    for offset, concept in enumerate(CONCEPTS):
        rng = random.Random(args.seed + offset)
        train_ids[concept], test_ids[concept] = split_ids(
            loaded[concept][3], args.train_fraction, rng
        )

    train_entries = interleave_entries(train_ids)
    if "fork_plate" in loaded:
        plate_ids = loaded["fork_plate"][3]
        train_entries.extend(("fork_plate", image_id) for image_id in plate_ids)
        train_ids["fork_plate"] = plate_ids
        test_ids["fork_plate"] = []
    test_entries = interleave_entries(test_ids)

    train_concept_counts = {
        "fork": len(train_ids["fork"]) + len(train_ids.get("fork_plate", [])),
        "scissors": len(train_ids["scissors"]),
    }
    if (
        args.extra_fork_train_source is not None
        and args.samples_per_concept < max(train_concept_counts.values())
    ):
        raise ValueError(
            "--samples-per-concept must cover every train image each epoch; "
            f"need at least {max(train_concept_counts.values())}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    train_counts = write_split(args.output, "train", train_entries, loaded)
    test_counts = write_split(args.output, "test", test_entries, loaded)
    manifest = {
        "sources": {
            "fork": str(args.fork_source.resolve()),
            "scissors": str(args.scissors_source.resolve()),
            **({"fork_plate": str(args.extra_fork_train_source.resolve())}
               if args.extra_fork_train_source is not None else {}),
        },
        "prompts": ["fork", "scissors"],
        "annotation_type": "bbox_only",
        "sampling": "per-epoch 1:1 sampling is applied by CocoConceptBalancedDataset",
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "train_source_ids": train_ids,
        "test_source_ids": test_ids,
        "train_samples_per_concept_per_epoch": args.samples_per_concept,
        "extra_fork_all_train": args.extra_fork_train_source is not None,
        "train_counts": train_counts,
        "test_counts": test_counts,
    }
    (args.output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
