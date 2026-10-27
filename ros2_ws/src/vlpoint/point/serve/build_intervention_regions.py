#!/usr/bin/env python3
"""Build region-level intervention records from pass1 counting JSON."""

import argparse
import csv
import json
from pathlib import Path


REGION_FEATURE_FIELDS = [
    "region_mask_count",
    "region_low_confidence_count",
    "region_low_confidence_ratio",
    "region_min_confidence",
    "region_mean_confidence",
    "region_confidence_std",
    "region_confidence_range",
    "region_mean_nn_distance",
    "region_max_nn_distance",
    "region_mean_pairwise_distance",
    "region_mean_distance_to_center",
    "region_width",
    "region_height",
    "region_area",
    "region_aspect_ratio",
    "region_density",
]


CSV_FIELDS = [
    "image",
    "region_id",
    "region_type",
    "region_intervention_label",
    "label_source",
    "image_intervention_label",
    "estimated_count",
    "gt_count",
    "counting_method",
    "center_x",
    "center_y",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    *REGION_FEATURE_FIELDS,
    "mask_ids",
]


def load_records(path):
    path = Path(path)
    text = path.read_text().strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {path}")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_center(center):
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        return None
    if center[0] is None or center[1] is None:
        return None
    try:
        return (float(center[0]), float(center[1]))
    except (TypeError, ValueError):
        return None


def parse_bbox(mask, center_padding):
    bbox = mask.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(value is not None for value in bbox):
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            return [max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)]
        except (TypeError, ValueError):
            pass

    center = parse_center(mask.get("center"))
    if center is None:
        return None
    cx, cy = center
    return [
        max(0.0, cx - center_padding),
        max(0.0, cy - center_padding),
        min(1.0, cx + center_padding),
        min(1.0, cy + center_padding),
    ]


def valid_masks(record, center_padding):
    masks = []
    for idx, mask in enumerate(record.get("sam3_masks", []), start=1):
        confidence = mask.get("confidence")
        center = parse_center(mask.get("center"))
        if confidence is None or center is None:
            continue
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue
        masks.append({
            "label": str(mask.get("label") if mask.get("label") is not None else idx),
            "confidence": confidence,
            "center": center,
            "bbox": parse_bbox(mask, center_padding),
            "status": mask.get("status", "detected"),
        })
    return masks


def euclidean(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def centroid(points):
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def mean_pairwise_distance(points):
    if len(points) < 2:
        return 0.0
    distances = []
    for idx, point in enumerate(points):
        for other in points[idx + 1:]:
            distances.append(euclidean(point, other))
    return sum(distances) / len(distances)


def nearest_neighbor_distances(points):
    if len(points) < 2:
        return []
    distances = []
    for idx, point in enumerate(points):
        distances.append(min(
            euclidean(point, other)
            for other_idx, other in enumerate(points)
            if other_idx != idx
        ))
    return distances


def union_bbox(bboxes, points, center_padding):
    usable = [bbox for bbox in bboxes if bbox is not None]
    if not usable and points:
        usable = [[
            max(0.0, point[0] - center_padding),
            max(0.0, point[1] - center_padding),
            min(1.0, point[0] + center_padding),
            min(1.0, point[1] + center_padding),
        ] for point in points]
    if not usable:
        return None
    return [
        round(min(bbox[0] for bbox in usable), 6),
        round(min(bbox[1] for bbox in usable), 6),
        round(max(bbox[2] for bbox in usable), 6),
        round(max(bbox[3] for bbox in usable), 6),
    ]


def connected_components(masks, radius):
    components = []
    visited = set()
    for idx in range(len(masks)):
        if idx in visited:
            continue
        stack = [idx]
        visited.add(idx)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for other_idx, other in enumerate(masks):
                if other_idx in visited:
                    continue
                if euclidean(masks[current]["center"], other["center"]) <= radius:
                    visited.add(other_idx)
                    stack.append(other_idx)
        components.append(component)
    return components


def confidence_features(confidences, low_threshold):
    if not confidences:
        return {
            "region_min_confidence": 0.0,
            "region_mean_confidence": 0.0,
            "region_confidence_std": 0.0,
            "region_confidence_range": 0.0,
            "region_low_confidence_count": 0,
            "region_low_confidence_ratio": 0.0,
        }
    mean = sum(confidences) / len(confidences)
    variance = sum((value - mean) ** 2 for value in confidences) / len(confidences)
    low_count = sum(value < low_threshold for value in confidences)
    return {
        "region_min_confidence": round(min(confidences), 6),
        "region_mean_confidence": round(mean, 6),
        "region_confidence_std": round(variance ** 0.5, 6),
        "region_confidence_range": round(max(confidences) - min(confidences), 6),
        "region_low_confidence_count": low_count,
        "region_low_confidence_ratio": round(low_count / len(confidences), 6),
    }


def geometry_features(points, bbox):
    if not points or bbox is None:
        return {
            "region_width": 0.0,
            "region_height": 0.0,
            "region_area": 0.0,
            "region_aspect_ratio": 0.0,
            "region_density": 0.0,
            "region_mean_nn_distance": 0.0,
            "region_max_nn_distance": 0.0,
            "region_mean_pairwise_distance": 0.0,
            "region_mean_distance_to_center": 0.0,
        }
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    area = width * height
    nn_distances = nearest_neighbor_distances(points)
    center = centroid(points)
    mean_distance_to_center = 0.0 if center is None else sum(euclidean(point, center) for point in points) / len(points)
    return {
        "region_width": round(width, 6),
        "region_height": round(height, 6),
        "region_area": round(area, 6),
        "region_aspect_ratio": round(width / height, 6) if height > 0 else 0.0,
        "region_density": round(len(points) / area, 6) if area > 0 else 0.0,
        "region_mean_nn_distance": round(sum(nn_distances) / len(nn_distances), 6) if nn_distances else 0.0,
        "region_max_nn_distance": round(max(nn_distances), 6) if nn_distances else 0.0,
        "region_mean_pairwise_distance": round(mean_pairwise_distance(points), 6),
        "region_mean_distance_to_center": round(mean_distance_to_center, 6),
    }


def weak_region_label(image_intervention_label, region_type, low_count):
    if image_intervention_label is None:
        return None, "no_gt_count"
    if not image_intervention_label:
        return False, "image_count_correct"
    if region_type == "no_sam3_detection":
        return True, "image_count_wrong_and_sam3_no_detection"
    if low_count > 0:
        return True, "image_count_wrong_and_region_has_low_confidence_masks"
    return False, "image_count_wrong_but_region_has_no_low_confidence_masks"


def build_no_detection_region(record):
    image_intervention_label = None
    if record.get("gt_count") is not None:
        image_intervention_label = record.get("estimated_count") != record.get("gt_count")
    label, label_source = weak_region_label(image_intervention_label, "no_sam3_detection", 0)
    return {
        "image": record.get("image"),
        "region_id": f"{Path(str(record.get('image', 'image'))).stem}_no_sam3_detection",
        "region_type": "no_sam3_detection",
        "region_intervention_label": label,
        "label_source": label_source,
        "image_intervention_label": image_intervention_label,
        "estimated_count": record.get("estimated_count"),
        "gt_count": record.get("gt_count"),
        "counting_method": record.get("counting_method"),
        "center": None,
        "bbox": None,
        "mask_ids": [],
        "features": {
            "region_mask_count": 0,
            "region_low_confidence_count": 0,
            "region_low_confidence_ratio": 0.0,
            "region_min_confidence": 0.0,
            "region_mean_confidence": 0.0,
            "region_confidence_std": 0.0,
            "region_confidence_range": 0.0,
            "region_mean_nn_distance": 0.0,
            "region_max_nn_distance": 0.0,
            "region_mean_pairwise_distance": 0.0,
            "region_mean_distance_to_center": 0.0,
            "region_width": 0.0,
            "region_height": 0.0,
            "region_area": 0.0,
            "region_aspect_ratio": 0.0,
            "region_density": 0.0,
        },
    }


def build_region(record, region_idx, component_masks, low_threshold, center_padding):
    points = [mask["center"] for mask in component_masks]
    confidences = [mask["confidence"] for mask in component_masks]
    region_center = centroid(points)
    bbox = union_bbox([mask.get("bbox") for mask in component_masks], points, center_padding)
    conf_features = confidence_features(confidences, low_threshold)
    geom_features = geometry_features(points, bbox)
    region_type = "low_confidence_region" if conf_features["region_low_confidence_count"] > 0 else "high_confidence_region"

    image_intervention_label = None
    if record.get("gt_count") is not None:
        image_intervention_label = record.get("estimated_count") != record.get("gt_count")
    label, label_source = weak_region_label(
        image_intervention_label,
        region_type,
        conf_features["region_low_confidence_count"],
    )
    image_stem = Path(str(record.get("image", "image"))).stem
    return {
        "image": record.get("image"),
        "region_id": f"{image_stem}_r{region_idx:02d}",
        "region_type": region_type,
        "region_intervention_label": label,
        "label_source": label_source,
        "image_intervention_label": image_intervention_label,
        "estimated_count": record.get("estimated_count"),
        "gt_count": record.get("gt_count"),
        "counting_method": record.get("counting_method"),
        "center": [round(region_center[0], 6), round(region_center[1], 6)] if region_center else None,
        "bbox": bbox,
        "mask_ids": [mask["label"] for mask in component_masks],
        "features": {
            "region_mask_count": len(component_masks),
            **conf_features,
            **geom_features,
        },
    }


def build_regions_for_record(record, region_radius, low_threshold, center_padding):
    masks = valid_masks(record, center_padding)
    if not masks:
        return [build_no_detection_region(record)]
    components = connected_components(masks, region_radius)
    regions = []
    for idx, component in enumerate(components, start=1):
        component_masks = [masks[mask_idx] for mask_idx in component]
        regions.append(build_region(record, idx, component_masks, low_threshold, center_padding))
    return regions


def flatten_region(region):
    center = region.get("center") or [None, None]
    bbox = region.get("bbox") or [None, None, None, None]
    row = {
        "image": region.get("image"),
        "region_id": region.get("region_id"),
        "region_type": region.get("region_type"),
        "region_intervention_label": region.get("region_intervention_label"),
        "label_source": region.get("label_source"),
        "image_intervention_label": region.get("image_intervention_label"),
        "estimated_count": region.get("estimated_count"),
        "gt_count": region.get("gt_count"),
        "counting_method": region.get("counting_method"),
        "center_x": center[0],
        "center_y": center[1],
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "mask_ids": "|".join(region.get("mask_ids", [])),
    }
    row.update(region.get("features", {}))
    return row


def group_regions_by_image(records, all_regions):
    grouped = []
    offset = 0
    for record in records:
        image_regions = []
        while offset < len(all_regions) and all_regions[offset].get("image") == record.get("image"):
            image_regions.append(all_regions[offset])
            offset += 1
        grouped.append({
            "image": record.get("image"),
            "counting_method": record.get("counting_method"),
            "estimated_count": record.get("estimated_count"),
            "gt_count": record.get("gt_count"),
            "image_intervention_label": None if record.get("gt_count") is None else record.get("estimated_count") != record.get("gt_count"),
            "regions": image_regions,
        })
    return grouped


def write_json(path, data):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="calibration_set/pass1_results.json")
    parser.add_argument("--output-json", default="calibration_set/pass1_regions.json")
    parser.add_argument("--output-jsonl", default="calibration_set/pass1_regions.jsonl")
    parser.add_argument("--output-csv", default="calibration_set/pass1_regions.csv")
    parser.add_argument("--region-radius", type=float, default=0.06)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.4)
    parser.add_argument("--center-padding", type=float, default=0.025)
    args = parser.parse_args()

    records = load_records(args.input)
    all_regions = []
    for record in records:
        all_regions.extend(build_regions_for_record(
            record,
            region_radius=args.region_radius,
            low_threshold=args.low_confidence_threshold,
            center_padding=args.center_padding,
        ))

    flat_rows = [flatten_region(region) for region in all_regions]
    grouped = group_regions_by_image(records, all_regions)

    write_json(args.output_json, grouped)
    write_jsonl(args.output_jsonl, flat_rows)
    write_csv(args.output_csv, flat_rows)

    for row in flat_rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
