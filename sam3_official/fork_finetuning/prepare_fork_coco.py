#!/usr/bin/env python3
"""Prepare fork COCO annotations for SAM3 text-conditioned training."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--prompt", default="fork")
    return parser.parse_args()


def main():
    args = parse_args()
    for split in ("train", "valid", "test"):
        split_dir = args.dataset_root / split
        source_path = split_dir / "_annotations.coco.json"
        output_path = split_dir / "_annotations.sam3.coco.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing COCO annotations: {source_path}")
        coco = json.loads(source_path.read_text(encoding="utf-8"))
        image_ids = {int(image["id"]) for image in coco.get("images", [])}
        missing_images = [
            image["file_name"]
            for image in coco.get("images", [])
            if not (split_dir / image["file_name"]).is_file()
        ]
        if missing_images:
            raise ValueError(
                f"{split}: {len(missing_images)} referenced images are missing; "
                f"first={missing_images[0]}"
            )
        coco["categories"] = [{
            "id": args.category_id,
            "name": args.prompt,
            "supercategory": "utensil",
        }]
        annotations = []
        annotation_ids = set()
        for annotation in coco.get("annotations", []):
            if int(annotation["category_id"]) != args.category_id:
                continue
            annotation = dict(annotation)
            annotation_id = int(annotation["id"])
            if annotation_id in annotation_ids:
                raise ValueError(
                    f"{split}: duplicate annotation id {annotation_id}"
                )
            annotation_ids.add(annotation_id)
            if int(annotation["image_id"]) not in image_ids:
                raise ValueError(
                    f"{split}: annotation {annotation_id} references an unknown image"
                )
            bbox = annotation.get("bbox", [])
            if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
                raise ValueError(
                    f"{split}: annotation {annotation_id} has invalid bbox {bbox}"
                )
            annotation["noun_phrase"] = args.prompt
            annotations.append(annotation)
        coco["annotations"] = annotations
        output_path.write_text(
            json.dumps(coco, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"{split}: images={len(coco.get('images', []))} "
            f"annotations={len(annotations)} output={output_path}"
        )


if __name__ == "__main__":
    main()
