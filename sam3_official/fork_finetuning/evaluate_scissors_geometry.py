#!/usr/bin/env python3
"""Evaluate an exported SAM3 model with the worker geometry pipeline."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--calibration-csv", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-object", default="scissors")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    return parser.parse_args()


def resolve_calibration_image(image_dir, csv_name):
    direct = image_dir / csv_name
    if direct.is_file():
        return direct
    stem = Path(csv_name).stem
    matches = [
        path
        for suffix in (".jpg", ".jpeg", ".png")
        if (path := image_dir / f"{stem}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot uniquely resolve {csv_name}: {matches}")
    return matches[0]


def main():
    args = parse_args()

    # model_worker configures a file logger at import time. Keep this standalone
    # evaluator from writing into the ROS package's runtime log directory.
    from point import constants
    constants.LOGDIR = str(args.output_dir / "logs")
    from point.serve import model_worker

    if args.calibration_csv is not None:
        if args.image_dir is None:
            raise ValueError("--image-dir is required with --calibration-csv")
        with args.calibration_csv.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        samples = [
            {
                "path": resolve_calibration_image(args.image_dir, row["images"]),
                "gt_count": int(row["gt_count"]),
            }
            for row in rows
        ]
        source = "calibration_csv"
    else:
        if args.dataset_root is None:
            raise ValueError("--dataset-root or --calibration-csv is required")
        split_dir = args.dataset_root / args.split
        annotations = json.loads(
            (split_dir / "_annotations.sam3.coco.json").read_text(encoding="utf-8")
        )
        gt_counts = {int(image["id"]): 0 for image in annotations["images"]}
        for annotation in annotations["annotations"]:
            gt_counts[int(annotation["image_id"])] += 1
        samples = [
            {
                "path": split_dir / image_info["file_name"],
                "gt_count": int(gt_counts[int(image_info["id"])]),
            }
            for image_info in sorted(
                annotations["images"], key=lambda item: int(item["id"])
            )
        ]
        source = "coco_split"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    detector_args = SimpleNamespace(
        sam3_detect_buttons=True,
        mm_sam3_vision_tower=str(args.model),
        sam3_detect_device="cuda",
        sam3_detect_dtype="bfloat16",
        sam3_detect_text=args.target_object,
        sam3_detect_threshold=args.score_threshold,
        sam3_detect_mask_threshold=0.5,
        sam3_max_candidates=0,
        sam3_detections_out_path="",
    )

    records = []
    for index, sample in enumerate(samples, start=1):
        image_path = sample["path"]
        image = Image.open(image_path).convert("RGB")
        overlay_path = overlay_dir / f"{image_path.stem}_overlay.jpg"
        detector_args.sam3_detections_out_path = str(overlay_path)
        candidates = model_worker.detect_sam3_candidates(
            image,
            detector_args,
            target_object=args.target_object,
            apply_postprocessing=True,
            score_threshold=args.score_threshold,
            force_presence_one=True,
            max_candidates=0,
            postprocessing_score_threshold=None,
        )
        result = model_worker.build_sam3_iom_experiment_output(
            candidates,
            args.target_object,
            confidence_thresholds=(args.score_threshold,),
            iom_thresholds=(0.50,),
            low_candidate_threshold=args.score_threshold,
            apply_formal_filter=True,
        )
        threshold_result = result["threshold_results"][0]
        predicted_count = int(threshold_result["retained_count"])
        gt_count = int(sample["gt_count"])
        record = {
            "image": image_path.name,
            "gt_count": gt_count,
            "predicted_count": predicted_count,
            "correct": predicted_count == gt_count,
            "count_error": predicted_count - gt_count,
            "overlay": str(overlay_path.resolve()),
            "excluded_confirmed_aggregate_count": int(
                threshold_result["excluded_confirmed_aggregate_count"]
            ),
            "excluded_fragment_count": int(
                threshold_result["excluded_fragment_count"]
            ),
            "excluded_pairwise_duplicate_count": int(
                threshold_result["excluded_pairwise_duplicate_count"]
            ),
            "pair_union_candidate_count": int(
                threshold_result["pair_union_candidate_count"]
            ),
            "suspected_merged_instance_count": int(
                threshold_result["suspected_merged_instance_count"]
            ),
            "result": result,
        }
        records.append(record)
        (args.output_dir / f"{image_path.stem}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{index}/{len(samples)}] {image_path.name}: "
            f"GT={gt_count} pred={predicted_count}",
            flush=True,
        )

    errors = [record["count_error"] for record in records]
    summary = {
        "model": str(args.model.resolve()),
        "prompt": args.target_object,
        "source": source,
        "split": args.split,
        "image_count": len(records),
        "exact_count": sum(error == 0 for error in errors),
        "exact_count_accuracy": sum(error == 0 for error in errors) / len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "under_count": sum(error < 0 for error in errors),
        "over_count": sum(error > 0 for error in errors),
        "zero_detection": sum(record["predicted_count"] == 0 for record in records),
        "presence_override": 1.0,
        "score_threshold": args.score_threshold,
        "iom_threshold": 0.50,
        "custom_geometry_applied": True,
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}))


if __name__ == "__main__":
    main()
