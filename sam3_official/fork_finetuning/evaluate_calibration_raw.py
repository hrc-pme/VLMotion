#!/usr/bin/env python3
"""Evaluate the fine-tuned SAM3 directly on the local calibration set."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from validate_hf_export import save_overlay


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--prompt", default="fork")
    parser.add_argument(
        "--iom-threshold",
        type=float,
        help="Suppress lower-scoring small masks contained by a larger mask.",
    )
    parser.add_argument(
        "--iom-min-parent-area-ratio",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--iom-aggregate-first",
        action="store_true",
        help="Remove confirmed aggregate parents before suppressing fragments.",
    )
    parser.add_argument(
        "--iom-tie-break-by-area",
        action="store_true",
        help="Allow a larger mask to parent an equal-scoring contained mask.",
    )
    parser.add_argument(
        "--use-model-presence",
        action="store_true",
        help="Use the model's presence logit instead of forcing presence to one.",
    )
    return parser.parse_args()


def resolve_image(image_dir, csv_name):
    direct = image_dir / csv_name
    if direct.is_file():
        return direct
    stem = Path(csv_name).stem
    matches = [
        image_dir / f"{stem}{extension}"
        for extension in IMAGE_EXTENSIONS
        if (image_dir / f"{stem}{extension}").is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot uniquely resolve {csv_name}: {matches}")
    return matches[0]


def mask_metadata(result, passed_query_indices, image_size):
    masks = result["masks"].detach().cpu().bool()
    scores = result["scores"].detach().float().cpu()
    boxes = result["boxes"].detach().float().cpu()
    width, height = image_size
    records = []
    for result_index, mask in enumerate(masks):
        ys, xs = torch.where(mask)
        if xs.numel() == 0:
            continue
        records.append({
            "result_index": result_index,
            "query_index": int(passed_query_indices[result_index]),
            "query_match_score": float(scores[result_index]),
            "bbox": [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ],
            "model_bbox": [float(value) for value in boxes[result_index].tolist()],
            "center": [
                float(xs.float().mean()) / width,
                float(ys.float().mean()) / height,
            ],
            "mask_area_fraction": float(mask.sum()) / float(width * height),
        })
    records.sort(key=lambda row: (row["center"][0], row["center"][1]))
    for label, record in enumerate(records, start=1):
        record["id"] = label
    return records


def suppress_contained_fragments(
    result,
    passed_query_indices,
    iom_threshold,
    min_parent_area_ratio,
    aggregate_first=False,
    tie_break_by_area=False,
):
    masks = result["masks"].detach().bool()
    scores = result["scores"].detach().float()
    if len(masks) < 2:
        return result, passed_query_indices, [], []

    areas = masks.flatten(1).sum(1)
    containment = {}
    for child in range(len(masks)):
        candidates = []
        for parent in range(len(masks)):
            if parent == child:
                continue
            if areas[parent] < min_parent_area_ratio * areas[child]:
                continue
            if scores[parent] < scores[child]:
                continue
            if scores[parent] == scores[child] and (
                not tie_break_by_area or areas[parent] <= areas[child]
            ):
                continue
            iom = float((masks[parent] & masks[child]).sum() / areas[child])
            if iom >= iom_threshold:
                candidates.append((iom, float(scores[parent]), parent))
        if candidates:
            containment[child] = max(candidates)

    aggregate_parents = set()
    aggregate_groups = []
    if aggregate_first and containment:
        median_area = statistics.median(float(area) for area in areas)
        for parent in range(len(masks)):
            components = [
                child
                for child, (_, _, candidate_parent) in containment.items()
                if candidate_parent == parent
                and float(areas[parent] / areas[child]) > 1.5
            ]
            distinct_components = []
            for child in sorted(components, key=lambda index: int(areas[index]), reverse=True):
                if any(
                    float(
                        (masks[child] & masks[other]).sum()
                        / min(areas[child], areas[other])
                    ) >= 0.5
                    for other in distinct_components
                ):
                    continue
                distinct_components.append(child)

            if len(distinct_components) < 2:
                continue
            component_union = torch.zeros_like(masks[parent])
            for child in distinct_components:
                component_union |= masks[child]
            intersection = (masks[parent] & component_union).sum()
            union = (masks[parent] | component_union).sum()
            union_similarity = float(intersection / union) if union else 0.0
            area_ratio = float(areas[parent]) / median_area if median_area else 0.0
            min_component_iom = min(
                float((masks[parent] & masks[child]).sum() / areas[child])
                for child in distinct_components
            )
            standard = (
                area_ratio >= 2.0
                and union_similarity >= 0.70
            )
            strong_containment = (
                area_ratio >= 3.0
                and len(distinct_components) >= 3
                and min_component_iom >= 0.95
            )
            if standard or strong_containment:
                aggregate_parents.add(parent)
                aggregate_groups.append({
                    "parent_result_index": parent,
                    "parent_query_index": int(passed_query_indices[parent]),
                    "component_result_indices": distinct_components,
                    "component_query_indices": [
                        int(passed_query_indices[index])
                        for index in distinct_components
                    ],
                    "area_ratio": area_ratio,
                    "union_similarity": union_similarity,
                    "min_component_iom": min_component_iom,
                })

    removed = [
        {
            "result_index": parent,
            "query_index": int(passed_query_indices[parent]),
            "removal_reason": "confirmed_aggregate",
        }
        for parent in sorted(aggregate_parents)
    ]
    for child, (iom, _, parent) in containment.items():
        if parent in aggregate_parents or child in aggregate_parents:
            continue
        removed.append({
            "result_index": child,
            "query_index": int(passed_query_indices[child]),
            "removal_reason": "contained_fragment",
            "child_result_index": child,
            "child_query_index": int(passed_query_indices[child]),
            "parent_result_index": parent,
            "parent_query_index": int(passed_query_indices[parent]),
            "iom": iom,
        })

    removed_indices = {item["result_index"] for item in removed}
    keep = [index for index in range(len(masks)) if index not in removed_indices]
    filtered = {
        key: value[keep]
        if torch.is_tensor(value) and value.ndim > 0 and len(value) == len(masks)
        else value
        for key, value in result.items()
    }
    kept_query_indices = [passed_query_indices[index] for index in keep]
    return filtered, kept_query_indices, removed, aggregate_groups


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    with args.calibration_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    device = torch.device("cuda")
    dtype = torch.bfloat16
    processor = Sam3Processor.from_pretrained(args.model, local_files_only=True)
    model = Sam3Model.from_pretrained(args.model, local_files_only=True)
    model.eval().to(device=device, dtype=dtype)

    records = []
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            image_path = resolve_image(args.image_dir, row["images"])
            gt_count = int(row["gt_count"])
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, text=args.prompt, return_tensors="pt")
            inputs = {
                key: value.to(device=device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
            outputs = model(**inputs)
            query_match_scores = outputs.pred_logits[0].float().sigmoid()
            presence_score = float(outputs.presence_logits[0].float().sigmoid())
            passed_query_indices = torch.where(
                query_match_scores > args.score_threshold
            )[0].detach().cpu().tolist()

            saved_presence = outputs.presence_logits
            if not args.use_model_presence:
                outputs.presence_logits = None
            result = processor.post_process_instance_segmentation(
                outputs,
                threshold=args.score_threshold,
                mask_threshold=args.mask_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
            outputs.presence_logits = saved_presence

            unfiltered_count = len(result["masks"])
            iom_removed = []
            iom_aggregate_groups = []
            if args.iom_threshold is not None:
                (
                    result,
                    passed_query_indices,
                    iom_removed,
                    iom_aggregate_groups,
                ) = suppress_contained_fragments(
                    result,
                    passed_query_indices,
                    args.iom_threshold,
                    args.iom_min_parent_area_ratio,
                    args.iom_aggregate_first,
                    args.iom_tie_break_by_area,
                )

            masks = mask_metadata(result, passed_query_indices, image.size)
            predicted_count = len(masks)
            overlay_path = overlay_dir / f"{image_path.stem}_overlay.jpg"
            save_overlay(image, result, overlay_path, gt_count)
            image_record = {
                "image": image_path.name,
                "gt_count": gt_count,
                "predicted_count": predicted_count,
                "count_error": predicted_count - gt_count,
                "correct": predicted_count == gt_count,
                "presence_override": None if args.use_model_presence else 1.0,
                "model_presence_score": presence_score,
                "score_threshold": args.score_threshold,
                "mask_threshold": args.mask_threshold,
                "custom_geometry_applied": args.iom_threshold is not None,
                "unfiltered_count": unfiltered_count,
                "iom_threshold": args.iom_threshold,
                "iom_min_parent_area_ratio": args.iom_min_parent_area_ratio,
                "iom_aggregate_first": args.iom_aggregate_first,
                "iom_tie_break_by_area": args.iom_tie_break_by_area,
                "iom_removed": iom_removed,
                "iom_aggregate_groups": iom_aggregate_groups,
                "overlay": str(overlay_path.resolve()),
                "masks": masks,
            }
            records.append(image_record)
            (args.output_dir / f"{image_path.stem}.json").write_text(
                json.dumps(image_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"[{index}/{len(rows)}] {image_path.name}: "
                f"GT={gt_count} pred={predicted_count}",
                flush=True,
            )

    errors = [record["count_error"] for record in records]
    summary = {
        "model": str(args.model.resolve()),
        "prompt": args.prompt,
        "image_count": len(records),
        "exact_count": sum(error == 0 for error in errors),
        "exact_count_accuracy": sum(error == 0 for error in errors) / len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "under_count": sum(error < 0 for error in errors),
        "over_count": sum(error > 0 for error in errors),
        "zero_detection": sum(record["predicted_count"] == 0 for record in records),
        "presence_override": None if args.use_model_presence else 1.0,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "custom_geometry_applied": args.iom_threshold is not None,
        "iom_threshold": args.iom_threshold,
        "iom_min_parent_area_ratio": args.iom_min_parent_area_ratio,
        "iom_aggregate_first": args.iom_aggregate_first,
        "iom_tie_break_by_area": args.iom_tie_break_by_area,
        "total_iom_removed": sum(len(record["iom_removed"]) for record in records),
        "total_confirmed_aggregates": sum(
            len(record["iom_aggregate_groups"]) for record in records
        ),
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}))


if __name__ == "__main__":
    main()
