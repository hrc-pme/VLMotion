#!/usr/bin/env python3
"""Analyze pass1 counting JSON and extract intervention features."""

import argparse
import csv
import json
from pathlib import Path


FEATURE_FIELDS = [
    "min_confidence",
    "mean_confidence",
    "low_confidence_count",
    "confidence_std",
    "low_confidence_mean_nn_distance",
    "low_confidence_max_nn_distance",
    "low_confidence_bbox_area",
    "low_confidence_mean_distance_to_all_center",
    "low_confidence_cluster_ratio",
    "low_confidence_spatial_clustered",
]


def load_records(path):
    path = Path(path)
    text = path.read_text().strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def valid_masks(record):
    masks = []
    for mask in record.get("sam3_masks", []):
        confidence = mask.get("confidence")
        center = mask.get("center")
        if confidence is None:
            continue
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue

        parsed_center = None
        if (
            isinstance(center, (list, tuple))
            and len(center) == 2
            and center[0] is not None
            and center[1] is not None
        ):
            try:
                parsed_center = (float(center[0]), float(center[1]))
            except (TypeError, ValueError):
                parsed_center = None

        masks.append({
            "confidence": confidence,
            "center": parsed_center,
        })
    return masks


def numeric_confidences(record):
    return [mask["confidence"] for mask in valid_masks(record)]


def euclidean(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


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
        nearest = min(
            euclidean(point, other)
            for other_idx, other in enumerate(points)
            if other_idx != idx
        )
        distances.append(nearest)
    return distances


def bbox_area(points):
    if len(points) < 2:
        return 0.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def centroid(points):
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def spatial_low_confidence_features(masks, low_threshold, cluster_ratio_threshold):
    all_points = [mask["center"] for mask in masks if mask.get("center") is not None]
    low_points = [
        mask["center"]
        for mask in masks
        if mask.get("center") is not None and mask["confidence"] < low_threshold
    ]
    nn_distances = nearest_neighbor_distances(low_points)
    all_center = centroid(all_points)
    if all_center is None or not low_points:
        mean_distance_to_all_center = 0.0
    else:
        mean_distance_to_all_center = (
            sum(euclidean(point, all_center) for point in low_points) / len(low_points)
        )

    low_mean_pairwise = mean_pairwise_distance(low_points)
    all_mean_pairwise = mean_pairwise_distance(all_points)
    if len(low_points) < 2 or all_mean_pairwise == 0.0:
        cluster_ratio = 0.0
    else:
        cluster_ratio = low_mean_pairwise / all_mean_pairwise

    spatial_clustered = len(low_points) >= 2 and cluster_ratio > 0.0 and cluster_ratio <= cluster_ratio_threshold
    return {
        "low_confidence_mean_nn_distance": round(sum(nn_distances) / len(nn_distances), 6) if nn_distances else 0.0,
        "low_confidence_max_nn_distance": round(max(nn_distances), 6) if nn_distances else 0.0,
        "low_confidence_bbox_area": round(bbox_area(low_points), 6),
        "low_confidence_mean_distance_to_all_center": round(mean_distance_to_all_center, 6),
        "low_confidence_cluster_ratio": round(cluster_ratio, 6),
        "low_confidence_spatial_clustered": spatial_clustered,
    }


def confidence_features(confidences, low_threshold):
    if not confidences:
        return {
            "min_confidence": 0.0,
            "mean_confidence": 0.0,
            "low_confidence_count": 0,
            "confidence_std": 0.0,
        }

    mean = sum(confidences) / len(confidences)
    variance = sum((value - mean) ** 2 for value in confidences) / len(confidences)
    return {
        "min_confidence": round(min(confidences), 6),
        "mean_confidence": round(mean, 6),
        "low_confidence_count": sum(value < low_threshold for value in confidences),
        "confidence_std": round(variance ** 0.5, 6),
    }


def analyze_record(record, low_threshold, cluster_ratio_threshold):
    estimated_count = record.get("estimated_count")
    gt_count = record.get("gt_count")
    intervention_label = None if gt_count is None else estimated_count != gt_count
    masks = valid_masks(record)
    confidences = [mask["confidence"] for mask in masks]
    features = confidence_features(confidences, low_threshold)
    spatial_features = spatial_low_confidence_features(masks, low_threshold, cluster_ratio_threshold)
    no_detection = not confidences

    return {
        "image": record.get("image"),
        "counting_method": record.get("counting_method"),
        "estimated_count": estimated_count,
        "gt_count": gt_count,
        "intervention_label": intervention_label,
        **features,
        **spatial_features,
        "valid_mask_count": len(confidences),
        "sam3_no_detection": no_detection,
    }


def write_json(path, rows):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


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
    fieldnames = [
        "image",
        "counting_method",
        "estimated_count",
        "gt_count",
        "intervention_label",
        *FEATURE_FIELDS,
        "valid_mask_count",
        "sam3_no_detection",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="calibration_set/pass1_results.json")
    parser.add_argument("--output-json", default="calibration_set/pass1_features.json")
    parser.add_argument("--output-jsonl", default="calibration_set/pass1_features.jsonl")
    parser.add_argument("--output-csv", default="calibration_set/pass1_features.csv")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.4)
    parser.add_argument("--cluster-ratio-threshold", type=float, default=0.75)
    args = parser.parse_args()

    records = load_records(args.input)
    rows = [
        analyze_record(record, args.low_confidence_threshold, args.cluster_ratio_threshold)
        for record in records
    ]

    write_json(args.output_json, rows)
    write_jsonl(args.output_jsonl, rows)
    write_csv(args.output_csv, rows)

    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
