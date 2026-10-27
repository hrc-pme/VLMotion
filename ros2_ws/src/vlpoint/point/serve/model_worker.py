"""
A model worker executes the model.
"""
import argparse
import asyncio
import contextlib
import json
import math
import re
import time
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial

from point.constants import WORKER_HEART_BEAT_INTERVAL
from point.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from point.model.builder import load_pretrained_model
from point.mm_utils import process_images, load_image_from_base64, tokenizer_image_token
from point.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextIteratorStreamer
from threading import Thread
from PIL import Image, ImageDraw, ImageFont


GB = 1 << 30

SAM3_CALIBRATED_PROFILE = "sam3_fork_scissors_joint_balanced_official_full_ft"
SAM3_CALIBRATED_SCORE_THRESHOLDS = {
    "fork": 0.65625,
    "scissors": 0.70,
}
SAM3_CALIBRATED_FORK_MIN_BORDER_RATIO = 0.80
SAM3_CALIBRATED_FORK_MAX_SCORE_MARGIN = 0.15
SAM3_CALIBRATED_BORDER_TOLERANCE = 0.002
SAM3_CALIBRATED_FORK_DUPLICATE_MASK_IOM = 0.90
SAM3_CALIBRATED_FORK_BORDERLINE_MASK_IOM = 0.80
SAM3_CALIBRATED_FORK_BORDERLINE_BBOX_IOM = 0.90
SAM3_CALIBRATED_FORK_BORDERLINE_MAX_CENTER_DISTANCE = 0.02
SAM3_CALIBRATED_FORK_DUPLICATE_MIN_SCORE = 3
SAM3_CALIBRATED_FORK_INTERVENTION_SCORE = 2
SAM3_CALIBRATED_FORK_SHADOW_THRESHOLD = 0.35
SAM3_CALIBRATED_FORK_SHADOW_GROUP_IOM = 0.90
SAM3_CALIBRATED_FORK_SHADOW_LOW_IOM = 0.70
SAM3_CALIBRATED_FORK_SHADOW_HIGH_IOM = 0.90
SAM3_CALIBRATED_FORK_SHADOW_MIN_AREA_RATIO = 0.55
SAM3_CALIBRATED_FORK_SHADOW_MAX_AREA_RATIO = 1.40
SAM3_CALIBRATED_FORK_SHADOW_MIN_DISTANCE_BALANCE = 0.50
SAM3_CALIBRATED_SCISSORS_SHADOW_THRESHOLD = 0.45
SAM3_CALIBRATED_SCISSORS_SHADOW_LOW_IOM = 0.70
SAM3_CALIBRATED_SCISSORS_SHADOW_HIGH_IOM = 0.90
SAM3_CALIBRATED_SCISSORS_FRAGMENT_MAX_AREA_RATIO = 0.60
worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_coords(text: str):
    coords = []
    # Match (...) style coords
    for match in re.findall(r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', text):
        coords.append((float(match[0]), float(match[1])))
    # Match [...] style arrays
    for match in re.findall(r'\[\s*([\d.\s,]+)\s*\]', text):
        nums = [float(x.strip()) for x in match.split(',') if x.strip()]
        for i in range(0, len(nums), 2):
            if i + 1 < len(nums):
                coords.append((nums[i], nums[i+1]))
    return coords


def parse_candidate_labels(text: str, valid_labels):
    valid = set(valid_labels)
    labels = []
    for match in re.findall(r'\b(\d+|[A-Z]|P\d+)\b', text.upper()):
        if match in valid and match not in labels:
            labels.append(match)
    return labels




def candidate_distance_sq(coord, det):
    return (det["center"][0] - coord[0]) ** 2 + (det["center"][1] - coord[1]) ** 2


def snap_coords_to_candidates_with_limit(coords, detections, max_points):
    if not coords:
        return []

    if not detections:
        return []

    labels = []
    ys = sorted({det["center"][1] for det in detections})
    row_gaps = [
        ys[idx + 1] - ys[idx]
        for idx in range(len(ys) - 1)
        if ys[idx + 1] - ys[idx] > 0.005
    ]
    max_snap_dist = max(0.035, (sum(row_gaps) / len(row_gaps)) * 1.75) if row_gaps else 0.05
    max_snap_dist_sq = max_snap_dist ** 2

    for coord in coords:
        best = min(detections, key=lambda det: candidate_distance_sq(coord, det))
        if candidate_distance_sq(coord, best) > max_snap_dist_sq:
            print(
                "Skipping generated coordinate far from SAM3 candidates: "
                f"coord=({coord[0]:.3f}, {coord[1]:.3f}), nearest={best['label']}"
            )
            continue
        if best["label"] not in labels:
            labels.append(best["label"])
        if max_points > 0 and len(labels) >= max_points:
            break
    return labels








































def resolve_dtype(dtype_name, device):
    dtype_name = str(dtype_name or "auto").lower()
    if dtype_name in ("auto", ""):
        return torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16
    if dtype_name in ("fp32", "float32"):
        return torch.float32
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def draw_sam3_detections(img, detections, out_path):
    if not out_path or not detections:
        return

    vis = make_sam3_overlay_image(img, detections)
    vis.save(out_path)
    print(f"Saved SAM3 detections: {out_path}")


def exclude_sam3_mask_from_overlay(det):
    confirmed_group_aggregate = (
        bool(det.get("suspected_aggregate", False))
        and det.get("aggregate_type") != "pair_union"
    )
    return bool(
        confirmed_group_aggregate
        or det.get("suspected_duplicate", False)
    )


def make_sam3_overlay_image(img, detections):
    vis = img.copy().convert("RGB")
    visible_detections = [
        (idx, det)
        for idx, det in enumerate(detections)
        if not exclude_sam3_mask_from_overlay(det)
    ]
    width, height = vis.size
    center_radius = max(4, int(min(width, height) * 0.012))
    line_width = max(2, int(min(width, height) * 0.004))
    id_font_size = max(14, int(min(width, height) * 0.026))
    legend_font_size = max(13, int(min(width, height) * 0.021))
    center_radius = max(center_radius, id_font_size // 2 + 3)
    try:
        id_font = ImageFont.truetype("DejaVuSans-Bold.ttf", id_font_size)
        legend_font = ImageFont.truetype("DejaVuSans-Bold.ttf", legend_font_size)
    except OSError:
        id_font = ImageFont.load_default()
        legend_font = ImageFont.load_default()
    colors = [
        (255, 64, 64),
        (45, 180, 90),
        (55, 125, 255),
        (245, 175, 35),
        (175, 80, 220),
        (25, 190, 200),
    ]

    for idx, det in visible_detections:
        mask = candidate_mask_tensor(det)
        if mask is None:
            continue
        mask_image = Image.fromarray(mask.to(torch.uint8).numpy() * 255, mode="L")
        if mask_image.size != vis.size:
            mask_image = mask_image.resize(vis.size, resample=Image.Resampling.NEAREST)
        color = colors[idx % len(colors)]
        colored = Image.new("RGB", vis.size, color)
        blended = Image.blend(vis, colored, 0.38)
        vis.paste(blended, mask=mask_image)

    draw = ImageDraw.Draw(vis)

    for idx, det in visible_detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        score = det["score"]
        color = colors[idx % len(colors)]
        cx_norm, cy_norm = det["center"]
        cx = int(round(cx_norm * width))
        cy = int(round(cy_norm * height))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        draw.ellipse(
            [
                cx - center_radius,
                cy - center_radius,
                cx + center_radius,
                cy + center_radius,
            ],
            fill="yellow",
            outline="black",
            width=line_width,
        )
        draw.text(
            (cx, cy),
            str(label),
            fill="black",
            font=id_font,
            anchor="mm",
            stroke_width=1,
            stroke_fill="white",
        )
        tip = det.get("scissors_tip") or det.get("scissors_tip_candidate")
        if tip and len(tip) == 2:
            tip_x, tip_y = [int(round(float(value))) for value in tip]
            tip_radius = max(4, line_width * 2)
            draw.ellipse(
                [
                    tip_x - tip_radius,
                    tip_y - tip_radius,
                    tip_x + tip_radius,
                    tip_y + tip_radius,
                ],
                fill=(
                    "white"
                    if det.get("scissors_tip_position_valid", False)
                    else (255, 80, 80)
                ),
                outline="black",
                width=line_width,
            )
    if visible_detections:
        row_height = legend_font_size + 9
        panel_width = max(150, int(min(width, height) * 0.28))
        panel_height = 12 + row_height * len(visible_detections)
        draw.rounded_rectangle(
            [8, 8, 8 + panel_width, 8 + panel_height],
            radius=4,
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            width=2,
        )
        for row_idx, (idx, det) in enumerate(visible_detections):
            color = colors[idx % len(colors)]
            row_y = 14 + row_idx * row_height
            draw.rectangle(
                [16, row_y + 3, 16 + legend_font_size, row_y + 3 + legend_font_size],
                fill=color,
                outline="white",
            )
            draw.text(
                (24 + legend_font_size, row_y),
                f"ID {det['label']}   conf {float(det['score']):.2f}",
                fill="white",
                font=legend_font,
            )

    return vis


def candidate_mask_tensor(det):
    mask = det.get("mask") if isinstance(det, dict) else None
    if not isinstance(mask, torch.Tensor):
        return None
    mask = mask.detach().to(device="cpu")
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim != 2:
        return None
    return mask.bool()








def mask_overlap_metrics(det_a, det_b):
    """Return pixel overlap metrics used by the IoM experiment."""
    mask_a = candidate_mask_tensor(det_a)
    mask_b = candidate_mask_tensor(det_b)
    if mask_a is None or mask_b is None or mask_a.shape != mask_b.shape:
        return None
    area_a = int(mask_a.sum().item())
    area_b = int(mask_b.sum().item())
    if area_a <= 0 or area_b <= 0:
        return None
    intersection = int(torch.logical_and(mask_a, mask_b).sum().item())
    union = area_a + area_b - intersection
    return {
        "area_a_pixels": area_a,
        "area_b_pixels": area_b,
        "intersection_pixels": intersection,
        "iom": float(intersection) / float(min(area_a, area_b)),
        "iou": float(intersection) / float(union) if union else 0.0,
        "coverage_a": float(intersection) / float(area_a),
        "coverage_b": float(intersection) / float(area_b),
        "area_ratio": float(max(area_a, area_b)) / float(min(area_a, area_b)),
    }








































def _clamp_normalized_box(box):
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _union_normalized_boxes(boxes, pad=0.08):
    valid_boxes = [_clamp_normalized_box(box) for box in boxes]
    valid_boxes = [box for box in valid_boxes if box]
    if not valid_boxes:
        return None
    x1 = max(0.0, min(box[0] for box in valid_boxes) - pad)
    y1 = max(0.0, min(box[1] for box in valid_boxes) - pad)
    x2 = min(1.0, max(box[2] for box in valid_boxes) + pad)
    y2 = min(1.0, max(box[3] for box in valid_boxes) + pad)
    return [x1, y1, x2, y2]


def append_to_last_user_turn(prompt: str, text: str) -> str:
    for marker in (" ASSISTANT:", "\nASSISTANT:", "Assistant:"):
        idx = prompt.rfind(marker)
        if idx != -1:
            return prompt[:idx] + text + prompt[idx:]
    return prompt + text


def is_counting_pass1_request(params):
    return str(params.get("counting_pass", "")).lower() in {"pass1", "counting_pass1"}


def get_counting_method(params):
    method = str(params.get("counting_method", "sam3_calibrated")).strip().lower()
    if method not in {
        "sam3_raw",
        "sam3_calibrated",
        "sam3_prompt_experiment",
    }:
        raise ValueError(f"Unsupported counting_method: {method}")
    return method


def get_sam3_calibrated_score_threshold(target_object):
    target = str(target_object or "").strip().lower()
    try:
        return SAM3_CALIBRATED_SCORE_THRESHOLDS[target]
    except KeyError as exc:
        supported = ", ".join(sorted(SAM3_CALIBRATED_SCORE_THRESHOLDS))
        raise ValueError(
            f"No calibrated SAM3 threshold for {target_object!r}; "
            f"supported targets: {supported}"
        ) from exc


def build_candidate_counts(estimated_count, radius=2, max_candidates=5):
    estimated_count = max(0, int(round(estimated_count)))
    start = max(0, estimated_count - radius)
    end = estimated_count + radius
    candidates = list(range(start, end + 1))
    if len(candidates) > max_candidates:
        if estimated_count - radius < 0:
            candidates = list(range(0, max_candidates))
        else:
            candidates = candidates[:max_candidates]
    if estimated_count not in candidates:
        candidates.append(estimated_count)
    return sorted(set(candidates))[:max_candidates]


def serialize_sam3_masks(sam3_candidates, target_object="target object"):
    if not sam3_candidates:
        return [{
            "label": None,
            "confidence": None,
            "center": None,
            "status": "no_detection",
            "reason": f"SAM3 did not return any {target_object} candidate mask/region.",
        }]

    masks = []
    for idx, det in enumerate(sam3_candidates, start=1):
        cx, cy = det.get("center", (0.0, 0.0))
        mask = {
            "label": str(det.get("label", idx)),
            "confidence": round(float(det.get("score", 0.0)), 4),
            "center": [round(float(cx), 4), round(float(cy), 4)],
            "status": "detected",
            "geometry_source": "sam3_instance_mask" if candidate_mask_tensor(det) is not None else "bbox",
        }
        if det.get("query_index") is not None:
            mask["query_index"] = int(det["query_index"])
        if det.get("query_match_logit") is not None:
            mask["query_match_logit"] = round(float(det["query_match_logit"]), 6)
        if det.get("query_match_score") is not None:
            mask["query_match_score"] = round(float(det["query_match_score"]), 6)
        if det.get("presence_logit") is not None:
            mask["presence_logit"] = round(float(det["presence_logit"]), 6)
        if det.get("presence_score") is not None:
            mask["presence_score"] = round(float(det["presence_score"]), 6)
        if det.get("mask_area") is not None:
            mask["mask_area_fraction"] = round(float(det["mask_area"]), 6)
        if det.get("scissors_tip") is not None:
            tip_x, tip_y = det["scissors_tip"]
            mask["scissors_tip"] = [
                round(float(tip_x), 2),
                round(float(tip_y), 2),
            ]
            mask["scissors_tip_normalized"] = [
                round(float(tip_x) / max(1.0, float(det.get("image_width", 1.0))), 6),
                round(float(tip_y) / max(1.0, float(det.get("image_height", 1.0))), 6),
            ]
        if det.get("scissors_tip_candidate") is not None:
            mask["scissors_tip_candidate"] = [
                round(float(value), 2)
                for value in det["scissors_tip_candidate"]
            ]
        if det.get("scissors_tip_valid") is not None:
            mask["scissors_tip_valid"] = bool(det["scissors_tip_valid"])
        if det.get("scissors_tip_position_valid") is not None:
            mask["scissors_tip_position_valid"] = bool(
                det["scissors_tip_position_valid"]
            )
        if det.get("scissors_complete_geometry") is not None:
            mask["scissors_complete_geometry"] = bool(
                det["scissors_complete_geometry"]
            )
        if det.get("scissors_candidate_type") is not None:
            mask["scissors_candidate_type"] = str(
                det["scissors_candidate_type"]
            )
        for key in (
            "suspected_upper_scissors_fragment",
            "suspected_broad_scissors_aggregate",
        ):
            if det.get(key) is not None:
                mask[key] = bool(det[key])
        if det.get("upper_fragment_of_query_index") is not None:
            mask["upper_fragment_of_query_index"] = int(
                det["upper_fragment_of_query_index"]
            )
        if det.get("scissors_tip_validation_failures") is not None:
            mask["scissors_tip_validation_failures"] = list(
                det["scissors_tip_validation_failures"]
            )
        if det.get("scissors_tip_position_failures") is not None:
            mask["scissors_tip_position_failures"] = list(
                det["scissors_tip_position_failures"]
            )
        if det.get("scissors_geometry_completeness_failures") is not None:
            mask["scissors_geometry_completeness_failures"] = list(
                det["scissors_geometry_completeness_failures"]
            )
        for key in (
            "scissors_blade_row_continuity",
            "scissors_blade_aspect_ratio",
            "scissors_median_tip_width",
            "scissors_tip_width_ratio",
        ):
            if det.get(key) is not None:
                mask[key] = round(float(det[key]), 6)
        if det.get("scissors_tip_component_reaches_blade_top") is not None:
            mask["scissors_tip_component_reaches_blade_top"] = bool(
                det["scissors_tip_component_reaches_blade_top"]
            )
        if det.get("scissors_blade_box") is not None:
            blade_box = det["scissors_blade_box"]
            mask["scissors_blade_box"] = [
                round(float(value), 2) for value in blade_box
            ]
        if det.get("scissors_handle_boundary_y") is not None:
            mask["scissors_handle_boundary_y"] = round(
                float(det["scissors_handle_boundary_y"]), 2
            )
        if det.get("scissors_handle_boundary_method") is not None:
            mask["scissors_handle_boundary_method"] = str(
                det["scissors_handle_boundary_method"]
            )
        if det.get("scissors_median_blade_width") is not None:
            mask["scissors_median_blade_width"] = round(
                float(det["scissors_median_blade_width"]), 4
            )
        for key in (
            "scissors_tip_cluster_id",
            "scissors_tip_cluster_size",
        ):
            if det.get(key) is not None:
                mask[key] = int(det[key])
        if det.get("scissors_tip_cluster_distance_threshold") is not None:
            mask["scissors_tip_cluster_distance_threshold"] = round(
                float(det["scissors_tip_cluster_distance_threshold"]), 4
            )
        if det.get("refined_from_mask_ids") is not None:
            mask["refined_from_mask_ids"] = list(det["refined_from_mask_ids"])
        if det.get("refined_from_query_indices") is not None:
            mask["refined_from_query_indices"] = [
                int(value) for value in det["refined_from_query_indices"]
            ]
        if det.get("scissors_refinement") is not None:
            refinement = dict(det["scissors_refinement"])
            for key in (
                "tip_cluster_center",
                "blade_prompt_box",
                "source_mask_coverages",
                "source_query_match_scores",
                "local_source_mask_areas",
                "local_source_coverages",
            ):
                if refinement.get(key) is not None:
                    refinement[key] = [
                        round(float(value), 6) for value in refinement[key]
                    ]
            for key in (
                "refined_score",
                "min_source_coverage",
                "median_source_coverage",
                "tip_cluster_diameter",
                "cluster_reference_blade_width",
                "normalized_tip_cluster_diameter",
                "refined_union_iou",
                "local_refined_union_iou",
            ):
                if refinement.get(key) is not None:
                    refinement[key] = round(float(refinement[key]), 6)
            mask["scissors_refinement"] = refinement
        if det.get("scissors_subset_merge") is not None:
            mask["scissors_subset_merge"] = det["scissors_subset_merge"]
        if det.get("aggregate_area_ratio") is not None:
            mask["aggregate_area_ratio"] = round(float(det["aggregate_area_ratio"]), 6)
        if det.get("aggregate_contained_mask_ids") is not None:
            mask["aggregate_contained_mask_ids"] = list(det["aggregate_contained_mask_ids"])
            mask["aggregate_contained_mask_count"] = int(
                det.get("aggregate_contained_mask_count", 0)
            )
        if det.get("aggregate_union_similarity") is not None:
            mask["aggregate_union_similarity"] = round(
                float(det["aggregate_union_similarity"]), 6
            )
        if det.get("aggregate_min_component_iom") is not None:
            mask["aggregate_min_component_iom"] = round(
                float(det["aggregate_min_component_iom"]), 6
            )
        if det.get("aggregate_broad_component_mask_ids") is not None:
            mask["aggregate_broad_component_mask_ids"] = list(
                det["aggregate_broad_component_mask_ids"]
            )
            mask["aggregate_broad_component_count"] = int(
                det.get("aggregate_broad_component_count", 0)
            )
            mask["aggregate_broad_min_component_iom"] = round(
                float(det.get("aggregate_broad_min_component_iom", 0.0)),
                6,
            )
        if det.get("suspected_aggregate") is not None:
            mask["suspected_aggregate"] = bool(det["suspected_aggregate"])
            mask["excluded_from_overlay"] = exclude_sam3_mask_from_overlay(det)
        if det.get("aggregate_type") is not None:
            mask["aggregate_type"] = det["aggregate_type"]
            mask["aggregate_component_mask_ids"] = list(
                det.get("aggregate_component_mask_ids", [])
            )
        if det.get("suspected_merged_instance") is not None:
            mask["suspected_merged_instance"] = bool(
                det["suspected_merged_instance"]
            )
            mask["merged_instance_evidence_type"] = det.get(
                "merged_instance_evidence_type"
            )
            mask["merged_instance_area_ratio"] = round(
                float(det.get("merged_instance_area_ratio", 0.0)),
                6,
            )
            mask["merged_instance_multi_track_ratio"] = round(
                float(det.get("merged_instance_multi_track_ratio", 0.0)),
                6,
            )
            mask["merged_instance_axis_aspect_ratio"] = round(
                float(det.get("merged_instance_axis_aspect_ratio", 0.0)),
                6,
            )
            mask["merged_instance_density_ratio"] = round(
                float(det.get("merged_instance_density_ratio", 0.0)),
                6,
            )
            mask["merged_instance_evidence_score"] = int(
                det.get("merged_instance_evidence_score", 0)
            )
            mask["merged_instance_evidence_signals"] = list(
                det.get("merged_instance_evidence_signals", [])
            )
            mask["merged_instance_shape_signals"] = list(
                det.get("merged_instance_shape_signals", [])
            )
        for source_key, output_key in (
            ("pair_union_bbox_iou", "pair_union_bbox_iou"),
            ("pair_union_area_ratio", "pair_union_area_ratio"),
            ("pair_union_pixel_iou", "pair_union_pixel_iou"),
            ("pair_union_component_iou", "pair_union_component_iou"),
        ):
            if det.get(source_key) is not None:
                mask[output_key] = round(float(det[source_key]), 6)
        box = det.get("box")
        if box is not None and len(box) == 4:
            x1, y1, x2, y2 = box
            width = max(1.0, float(det.get("image_width", 1.0)))
            height = max(1.0, float(det.get("image_height", 1.0)))
            mask["bbox"] = [
                round(float(x1) / width, 4),
                round(float(y1) / height, 4),
                round(float(x2) / width, 4),
                round(float(y2) / height, 4),
            ]
        masks.append(mask)
    return masks




def build_sam3_prompt_experiment_output(sam3_candidates, target_object):
    estimated_count = len(sam3_candidates)
    return {
        "estimated_count": estimated_count,
        "candidate_counts": build_candidate_counts(estimated_count),
        "reason": (
            f"Raw SAM3 text-prompt experiment returned {estimated_count} "
            f"candidate region(s) for {target_object!r}; no aggregate, "
            "duplicate, tip, fragment, or refinement post-processing was applied."
        ),
        "counting_method": "sam3_prompt_experiment",
        "sam3_prompt": target_object,
        "postprocessing_applied": False,
        "sam3_masks": serialize_sam3_masks(sam3_candidates, target_object),
    }


def normalized_bbox_for_det(det):
    box = det.get("box")
    if box is None or len(box) != 4:
        cx, cy = det.get("center", (0.5, 0.5))
        return [
            round(max(0.0, cx - 0.04), 4),
            round(max(0.0, cy - 0.08), 4),
            round(min(1.0, cx + 0.04), 4),
            round(min(1.0, cy + 0.08), 4),
        ]
    width = max(1.0, float(det.get("image_width", 1.0)))
    height = max(1.0, float(det.get("image_height", 1.0)))
    x1, y1, x2, y2 = box
    return [
        round(max(0.0, min(1.0, float(x1) / width)), 4),
        round(max(0.0, min(1.0, float(y1) / height)), 4),
        round(max(0.0, min(1.0, float(x2) / width)), 4),
        round(max(0.0, min(1.0, float(y2) / height)), 4),
    ]




def bbox_iom(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_area = (
        max(0.0, min(ax2, bx2) - max(ax1, bx1))
        * max(0.0, min(ay2, by2) - max(ay1, by1))
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller_area = min(area_a, area_b)
    return inter_area / smaller_area if smaller_area > 0 else 0.0


def build_sam3_calibrated_intervention(
    sam3_candidates,
    target_object,
    score_threshold,
):
    """Request review for calibrated risk signals without changing the count."""
    candidates = list(sam3_candidates or [])
    if not candidates:
        return {
            "requires_intervention": True,
            "intervention_reasons": ["sam3_no_detection"],
            "intervention_explanation": (
                "SAM3 returned no candidate masks at the calibrated score "
                "threshold; reposition the objects or camera and run inference again."
            ),
            "uncertain_regions": [{
                "region_id": "full_image",
                "uncertainty_type": "sam3_no_detection",
                "related_mask_ids": [],
            }],
            "fork_border_contact_count": 0,
            "fork_border_contact_ratio": 0.0,
            "minimum_score_margin": None,
        }

    scores = [
        float(det.get("query_match_score", det.get("score", 0.0)))
        for det in candidates
    ]
    minimum_score_margin = min(scores) - float(score_threshold)
    border_candidates = []
    tolerance = SAM3_CALIBRATED_BORDER_TOLERANCE
    for det in candidates:
        x1, y1, x2, y2 = normalized_bbox_for_det(det)
        if (
            x1 <= tolerance
            or y1 <= tolerance
            or x2 >= 1.0 - tolerance
            or y2 >= 1.0 - tolerance
        ):
            border_candidates.append(det)

    border_ratio = len(border_candidates) / len(candidates)
    fork_risk = (
        str(target_object or "").strip().lower() == "fork"
        and border_ratio >= SAM3_CALIBRATED_FORK_MIN_BORDER_RATIO
        and minimum_score_margin <= SAM3_CALIBRATED_FORK_MAX_SCORE_MARGIN
    )
    if not fork_risk:
        return {
            "requires_intervention": False,
            "intervention_reasons": [],
            "intervention_explanation": (
                "No calibrated intervention condition was detected."
            ),
            "uncertain_regions": [],
            "fork_border_contact_count": len(border_candidates),
            "fork_border_contact_ratio": round(border_ratio, 6),
            "minimum_score_margin": round(minimum_score_margin, 6),
        }

    related_ids = [str(det.get("label", "-")) for det in border_candidates]
    boxes = [normalized_bbox_for_det(det) for det in border_candidates]
    region_bbox = _union_normalized_boxes(boxes, pad=0.0)
    uncertain_region = {
        "region_id": "U1",
        "uncertainty_type": "truncated_view_with_borderline_candidate",
        "related_mask_ids": related_ids,
        "normalized_bbox": region_bbox,
        "evidence": {
            "border_contact_count": len(border_candidates),
            "candidate_count": len(candidates),
            "border_contact_ratio": round(border_ratio, 6),
            "border_ratio_threshold": SAM3_CALIBRATED_FORK_MIN_BORDER_RATIO,
            "minimum_score_margin": round(minimum_score_margin, 6),
            "score_margin_threshold": SAM3_CALIBRATED_FORK_MAX_SCORE_MARGIN,
        },
        "recommended_action": "reframe_scene_to_show_complete_forks",
    }
    return {
        "requires_intervention": True,
        "intervention_reasons": [
            "truncated_view_with_borderline_candidate"
        ],
        "intervention_explanation": (
            f"The raw count was preserved, but {len(border_candidates)}/"
            f"{len(candidates)} fork candidates touch an image border "
            f"({border_ratio:.1%}) and the minimum score is only "
            f"{minimum_score_margin:.6f} above the calibrated threshold. "
            "Incomplete framing plus a borderline candidate can cause a missed "
            "or extra fork; reframe the scene and review the count."
        ),
        "uncertain_regions": [uncertain_region],
        "fork_border_contact_count": len(border_candidates),
        "fork_border_contact_ratio": round(border_ratio, 6),
        "minimum_score_margin": round(minimum_score_margin, 6),
    }


def deduplicate_calibrated_fork_candidates(
    sam3_candidates,
    iom_threshold=SAM3_CALIBRATED_FORK_DUPLICATE_MASK_IOM,
):
    """Score three duplicate signals and drop lower-ranked forks at 3."""
    candidates = list(sam3_candidates or [])
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            -float(item[1].get("query_match_score", item[1].get("score", 0.0))),
            -int(
                candidate_mask_tensor(item[1]).sum().item()
                if candidate_mask_tensor(item[1]) is not None
                else 0
            ),
            item[0],
        ),
    )
    kept = []
    removed = []
    ambiguous_pairs = []
    for original_index, candidate in ranked:
        strongest_match = None
        strongest_ambiguous = None
        for _, retained in kept:
            metrics = mask_overlap_metrics(candidate, retained)
            if metrics is None:
                continue
            candidate_bbox = normalized_bbox_for_det(candidate)
            retained_bbox = normalized_bbox_for_det(retained)
            pair_bbox_iom = bbox_iom(candidate_bbox, retained_bbox)
            center_distance = math.dist(
                candidate.get("center", (0.0, 0.0)),
                retained.get("center", (0.0, 0.0)),
            )
            evidence = {
                "mask_iom": bool(
                    metrics["iom"]
                    >= SAM3_CALIBRATED_FORK_BORDERLINE_MASK_IOM
                ),
                "bbox_iom": bool(
                    pair_bbox_iom
                    >= SAM3_CALIBRATED_FORK_BORDERLINE_BBOX_IOM
                ),
                "center_distance": bool(
                    center_distance
                    <= SAM3_CALIBRATED_FORK_BORDERLINE_MAX_CENTER_DISTANCE
                ),
            }
            duplicate_score = sum(int(value) for value in evidence.values())
            match = {
                "metrics": metrics,
                "retained": retained,
                "bbox_iom": pair_bbox_iom,
                "center_distance": center_distance,
                "duplicate_score": duplicate_score,
                "evidence": evidence,
            }
            ranking_key = (
                duplicate_score,
                float(metrics["iom"]),
                float(pair_bbox_iom),
            )
            if duplicate_score >= SAM3_CALIBRATED_FORK_DUPLICATE_MIN_SCORE:
                if (
                    strongest_match is None
                    or ranking_key > strongest_match["ranking_key"]
                ):
                    match["ranking_key"] = ranking_key
                    strongest_match = match
            elif duplicate_score == SAM3_CALIBRATED_FORK_INTERVENTION_SCORE:
                if (
                    strongest_ambiguous is None
                    or ranking_key > strongest_ambiguous["ranking_key"]
                ):
                    match["ranking_key"] = ranking_key
                    strongest_ambiguous = match
        if strongest_match is None:
            kept.append((original_index, candidate))
            if strongest_ambiguous is not None:
                metrics = strongest_ambiguous["metrics"]
                retained = strongest_ambiguous["retained"]
                ambiguous_pairs.append({
                    "candidate_mask_id": str(candidate.get("label", "-")),
                    "candidate_query_index": candidate.get("query_index"),
                    "retained_mask_id": str(retained.get("label", "-")),
                    "retained_query_index": retained.get("query_index"),
                    "duplicate_score": strongest_ambiguous["duplicate_score"],
                    "evidence": strongest_ambiguous["evidence"],
                    "mask_iom": round(float(metrics["iom"]), 6),
                    "mask_iou": round(float(metrics["iou"]), 6),
                    "bbox_iom": round(
                        float(strongest_ambiguous["bbox_iom"]), 6
                    ),
                    "center_distance": round(
                        float(strongest_ambiguous["center_distance"]), 6
                    ),
                    "mask_area_ratio": round(
                        float(metrics["area_ratio"]), 6
                    ),
                    "normalized_bbox": _union_normalized_boxes(
                        [
                            normalized_bbox_for_det(candidate),
                            normalized_bbox_for_det(retained),
                        ],
                        pad=0.0,
                    ),
                })
            continue

        metrics = strongest_match["metrics"]
        retained = strongest_match["retained"]
        removed.append({
            "removed_mask_id": str(candidate.get("label", "-")),
            "removed_query_index": candidate.get("query_index"),
            "retained_mask_id": str(retained.get("label", "-")),
            "retained_query_index": retained.get("query_index"),
            "removed_score": round(float(
                candidate.get("query_match_score", candidate.get("score", 0.0))
            ), 6),
            "retained_score": round(float(
                retained.get("query_match_score", retained.get("score", 0.0))
            ), 6),
            "mask_iom": round(float(metrics["iom"]), 6),
            "mask_iou": round(float(metrics["iou"]), 6),
            "bbox_iom": round(float(strongest_match["bbox_iom"]), 6),
            "center_distance": round(
                float(strongest_match["center_distance"]), 6
            ),
            "duplicate_score": strongest_match["duplicate_score"],
            "evidence": strongest_match["evidence"],
            "match_rule": "fork_duplicate_evidence_score",
            "removed_coverage": round(float(metrics["coverage_a"]), 6),
            "retained_coverage": round(float(metrics["coverage_b"]), 6),
            "mask_area_ratio": round(float(metrics["area_ratio"]), 6),
        })

    kept.sort(key=lambda item: item[0])
    output = [candidate for _, candidate in kept]
    for rank, candidate in enumerate(output, start=1):
        candidate["label"] = str(rank)
    labels_by_query = {
        candidate.get("query_index"): str(candidate.get("label", "-"))
        for candidate in output
    }
    for pair in ambiguous_pairs:
        pair["candidate_mask_id"] = labels_by_query.get(
            pair["candidate_query_index"], pair["candidate_mask_id"]
        )
        pair["retained_mask_id"] = labels_by_query.get(
            pair["retained_query_index"], pair["retained_mask_id"]
        )
    return output, {
        "applied": bool(removed),
        "iom_threshold": float(iom_threshold),
        "borderline_mask_iom_threshold": (
            SAM3_CALIBRATED_FORK_BORDERLINE_MASK_IOM
        ),
        "borderline_bbox_iom_threshold": (
            SAM3_CALIBRATED_FORK_BORDERLINE_BBOX_IOM
        ),
        "borderline_max_center_distance": (
            SAM3_CALIBRATED_FORK_BORDERLINE_MAX_CENTER_DISTANCE
        ),
        "scoring_method": "three_binary_signals",
        "points_per_signal": 1,
        "duplicate_min_score": SAM3_CALIBRATED_FORK_DUPLICATE_MIN_SCORE,
        "intervention_score": SAM3_CALIBRATED_FORK_INTERVENTION_SCORE,
        "removed_count": len(removed),
        "removed_duplicates": removed,
        "requires_intervention": bool(ambiguous_pairs),
        "ambiguous_pairs": ambiguous_pairs,
    }


def split_calibrated_scissors_candidates(candidates, primary_threshold):
    """Separate formal scissors detections from non-counting shadow masks."""
    primary = [
        det for det in candidates
        if float(det.get("query_match_score", 0.0)) > float(primary_threshold)
    ]
    shadow = [
        det for det in candidates
        if (
            SAM3_CALIBRATED_SCISSORS_SHADOW_THRESHOLD
            < float(det.get("query_match_score", 0.0))
            <= float(primary_threshold)
        )
    ]
    primary.sort(key=lambda det: (
        float(det.get("center", (0.0, 0.0))[0]),
        float(det.get("center", (0.0, 0.0))[1]),
        -float(det.get("query_match_score", 0.0)),
    ))
    shadow.sort(key=lambda det: (
        float(det.get("center", (0.0, 0.0))[0]),
        float(det.get("center", (0.0, 0.0))[1]),
        -float(det.get("query_match_score", 0.0)),
    ))
    for index, det in enumerate(primary, start=1):
        det["label"] = str(index)
    for index, det in enumerate(shadow, start=1):
        det["shadow_label"] = f"S{index}"
    return primary, shadow


def split_calibrated_fork_candidates(candidates, primary_threshold):
    """Separate counted forks from low-score masks used only for review."""
    primary = [
        det for det in candidates
        if float(det.get("query_match_score", 0.0)) > float(primary_threshold)
    ]
    shadow = [
        det for det in candidates
        if (
            SAM3_CALIBRATED_FORK_SHADOW_THRESHOLD
            < float(det.get("query_match_score", 0.0))
            <= float(primary_threshold)
        )
    ]
    primary.sort(key=lambda det: (
        float(det.get("center", (0.0, 0.0))[0]),
        float(det.get("center", (0.0, 0.0))[1]),
    ))
    for rank, det in enumerate(primary, start=1):
        det["label"] = str(rank)
    shadow.sort(key=lambda det: (
        float(det.get("center", (0.0, 0.0))[0]),
        float(det.get("center", (0.0, 0.0))[1]),
    ))
    for rank, det in enumerate(shadow, start=1):
        det["shadow_label"] = f"FS{rank}"
    return primary, shadow


def _principal_axis_projection(candidates):
    centers = [
        tuple(float(value) for value in det.get("center", (0.5, 0.5)))
        for det in candidates
    ]
    if len(centers) < 2:
        return (1.0, 0.0), [center[0] for center in centers]
    mean_x = sum(center[0] for center in centers) / len(centers)
    mean_y = sum(center[1] for center in centers) / len(centers)
    covariance_xx = sum(
        (center[0] - mean_x) ** 2 for center in centers
    ) / len(centers)
    covariance_yy = sum(
        (center[1] - mean_y) ** 2 for center in centers
    ) / len(centers)
    covariance_xy = sum(
        (center[0] - mean_x) * (center[1] - mean_y)
        for center in centers
    ) / len(centers)
    angle = 0.5 * math.atan2(
        2.0 * covariance_xy,
        covariance_xx - covariance_yy,
    )
    axis = (math.cos(angle), math.sin(angle))
    projections = [
        (center[0] - mean_x) * axis[0]
        + (center[1] - mean_y) * axis[1]
        for center in centers
    ]
    return axis, projections


def build_scissors_shadow_intervention(primary_candidates, shadow_candidates):
    """Classify localized shadow masks with pixel IoM without changing count."""
    primary = list(primary_candidates or [])
    shadow = list(shadow_candidates or [])
    if len(primary) < 2 or not shadow:
        return {
            "requires_intervention": False,
            "intervention_reasons": [],
            "intervention_explanation": None,
            "uncertain_regions": [],
            "shadow_candidate_count": len(shadow),
            "shadow_candidates": [],
        }

    axis, primary_projections = _principal_axis_projection(primary)
    mean_x = sum(float(det["center"][0]) for det in primary) / len(primary)
    mean_y = sum(float(det["center"][1]) for det in primary) / len(primary)
    ordered = sorted(
        zip(primary_projections, primary),
        key=lambda item: item[0],
    )
    diagnostics = []
    uncertain_regions = []
    reasons = []

    for det in shadow:
        center = tuple(float(value) for value in det.get("center", (0.5, 0.5)))
        projection = (
            (center[0] - mean_x) * axis[0]
            + (center[1] - mean_y) * axis[1]
        )
        bracket = None
        for index in range(len(ordered) - 1):
            if ordered[index][0] <= projection <= ordered[index + 1][0]:
                bracket = (ordered[index][1], ordered[index + 1][1])
                break
        if bracket is None:
            continue

        overlap_rows = []
        shadow_mask = candidate_mask_tensor(det)
        shadow_area = (
            int(shadow_mask.sum().item())
            if shadow_mask is not None
            else 0
        )
        for neighbor in bracket:
            metrics = mask_overlap_metrics(det, neighbor)
            neighbor_mask = candidate_mask_tensor(neighbor)
            neighbor_area = (
                int(neighbor_mask.sum().item())
                if neighbor_mask is not None
                else 0
            )
            overlap_rows.append({
                "neighbor": neighbor,
                "metrics": metrics,
                "shadow_to_neighbor_area_ratio": (
                    float(shadow_area) / float(neighbor_area)
                    if neighbor_area > 0
                    else None
                ),
            })

        valid_overlap_rows = [
            row for row in overlap_rows if row["metrics"] is not None
        ]
        strongest = (
            max(
                valid_overlap_rows,
                key=lambda row: float(row["metrics"]["iom"]),
            )
            if valid_overlap_rows
            else None
        )
        max_iom = (
            float(strongest["metrics"]["iom"])
            if strongest is not None
            else None
        )
        area_ratio = (
            strongest["shadow_to_neighbor_area_ratio"]
            if strongest is not None
            else None
        )
        if max_iom is None:
            classification = "ambiguous_shadow_candidate"
            triggers_intervention = True
        elif max_iom >= SAM3_CALIBRATED_SCISSORS_SHADOW_HIGH_IOM:
            classification = (
                "fragment_of_existing_instance"
                if area_ratio is not None
                and area_ratio
                < SAM3_CALIBRATED_SCISSORS_FRAGMENT_MAX_AREA_RATIO
                else "duplicate_of_existing_instance"
            )
            triggers_intervention = False
        elif max_iom < SAM3_CALIBRATED_SCISSORS_SHADOW_LOW_IOM:
            classification = "possible_missed_scissors"
            triggers_intervention = True
        else:
            classification = "ambiguous_shadow_candidate"
            triggers_intervention = True

        related_ids = [str(neighbor.get("label", "-")) for neighbor in bracket]
        diagnostic = {
            "shadow_id": str(det.get("shadow_label", "S?")),
            "query_index": det.get("query_index"),
            "query_match_score": round(
                float(det.get("query_match_score", 0.0)), 6
            ),
            "center": [round(value, 6) for value in center],
            "bbox": normalized_bbox_for_det(det),
            "between_mask_ids": related_ids,
            "max_iom": None if max_iom is None else round(max_iom, 6),
            "shadow_to_neighbor_area_ratio": (
                None if area_ratio is None else round(float(area_ratio), 6)
            ),
            "classification": classification,
            "triggers_intervention": triggers_intervention,
        }
        diagnostics.append(diagnostic)
        if not triggers_intervention:
            continue
        reason = (
            "possible_scissors_missed_below_primary_threshold"
            if classification == "possible_missed_scissors"
            else "ambiguous_scissors_shadow_candidate"
        )
        if reason not in reasons:
            reasons.append(reason)
        uncertain_regions.append({
            "region_id": f"S{len(uncertain_regions) + 1}",
            "uncertainty_type": classification,
            "related_mask_ids": related_ids,
            "shadow_id": diagnostic["shadow_id"],
            "normalized_bbox": diagnostic["bbox"],
            "evidence": {
                "query_match_score": diagnostic["query_match_score"],
                "shadow_score_threshold": (
                    SAM3_CALIBRATED_SCISSORS_SHADOW_THRESHOLD
                ),
                "primary_score_threshold": (
                    SAM3_CALIBRATED_SCORE_THRESHOLDS["scissors"]
                ),
                "max_iom": diagnostic["max_iom"],
                "shadow_to_neighbor_area_ratio": diagnostic[
                    "shadow_to_neighbor_area_ratio"
                ],
            },
            "recommended_action": "review_local_scissors_cluster",
        })

    if reasons:
        explanation = (
            f"The formal count was preserved, but {len(uncertain_regions)} "
            "localized scissors shadow candidate(s) between adjacent formal "
            "masks were not explained as high-IoM fragments or duplicates."
        )
    else:
        explanation = (
            "Shadow candidates were absent, outside adjacent formal masks, or "
            "explained as high-IoM fragments/duplicates."
        )
    return {
        "requires_intervention": bool(reasons),
        "intervention_reasons": reasons,
        "intervention_explanation": explanation,
        "uncertain_regions": uncertain_regions,
        "shadow_candidate_count": len(diagnostics),
        "shadow_candidates": diagnostics,
    }


def build_fork_shadow_intervention(primary_candidates, shadow_candidates):
    """Request review for independent low-score fork masks without adding count."""
    primary = list(primary_candidates or [])
    shadow = list(shadow_candidates or [])
    empty = {
        "requires_intervention": False,
        "intervention_reasons": [],
        "intervention_explanation": None,
        "uncertain_regions": [],
        "shadow_candidate_count": 0,
        "shadow_candidates": [],
        "shadow_duplicate_groups": [],
    }
    if len(primary) < 2 or not shadow:
        empty["shadow_candidate_count"] = len(shadow)
        return empty

    ranked_shadow = sorted(
        shadow,
        key=lambda det: (
            -float(det.get("query_match_score", det.get("score", 0.0))),
            -int(
                candidate_mask_tensor(det).sum().item()
                if candidate_mask_tensor(det) is not None
                else 0
            ),
        ),
    )
    deduplicated_shadow = []
    shadow_duplicate_groups = []
    for det in ranked_shadow:
        duplicate = None
        for retained in deduplicated_shadow:
            metrics = mask_overlap_metrics(det, retained)
            if (
                metrics is not None
                and metrics["iom"] > SAM3_CALIBRATED_FORK_SHADOW_GROUP_IOM
            ):
                duplicate = (retained, metrics)
                break
        if duplicate is None:
            deduplicated_shadow.append(det)
            continue
        retained, metrics = duplicate
        shadow_duplicate_groups.append({
            "removed_shadow_id": str(det.get("shadow_label", "FS?")),
            "removed_query_index": det.get("query_index"),
            "retained_shadow_id": str(retained.get("shadow_label", "FS?")),
            "retained_query_index": retained.get("query_index"),
            "mask_iom": round(float(metrics["iom"]), 6),
        })

    axis, primary_projections = _principal_axis_projection(primary)
    mean_x = sum(float(det["center"][0]) for det in primary) / len(primary)
    mean_y = sum(float(det["center"][1]) for det in primary) / len(primary)
    ordered = sorted(
        zip(primary_projections, primary),
        key=lambda item: item[0],
    )
    primary_areas = []
    for det in primary:
        mask = candidate_mask_tensor(det)
        if mask is not None:
            primary_areas.append(int(mask.sum().item()))
    median_primary_area = _median(primary_areas)

    diagnostics = []
    uncertain_regions = []
    reasons = []
    for det in deduplicated_shadow:
        center = tuple(float(value) for value in det.get("center", (0.5, 0.5)))
        projection = (
            (center[0] - mean_x) * axis[0]
            + (center[1] - mean_y) * axis[1]
        )
        bracket = None
        bracket_projections = None
        for index in range(len(ordered) - 1):
            if ordered[index][0] <= projection <= ordered[index + 1][0]:
                bracket = (ordered[index][1], ordered[index + 1][1])
                bracket_projections = (ordered[index][0], ordered[index + 1][0])
                break
        if bracket is None:
            continue

        left_distance = projection - bracket_projections[0]
        right_distance = bracket_projections[1] - projection
        distance_balance = (
            min(left_distance, right_distance) / max(left_distance, right_distance)
            if max(left_distance, right_distance) > 0.0
            else 0.0
        )
        shadow_mask = candidate_mask_tensor(det)
        shadow_area = int(shadow_mask.sum().item()) if shadow_mask is not None else 0
        median_area_ratio = (
            float(shadow_area) / float(median_primary_area)
            if median_primary_area > 0.0
            else 0.0
        )

        overlap_rows = []
        for neighbor in bracket:
            metrics = mask_overlap_metrics(det, neighbor)
            if metrics is not None:
                overlap_rows.append((neighbor, metrics))
        strongest = (
            max(overlap_rows, key=lambda row: float(row[1]["iom"]))
            if overlap_rows
            else None
        )
        max_iom = float(strongest[1]["iom"]) if strongest is not None else None
        geometry_valid = (
            SAM3_CALIBRATED_FORK_SHADOW_MIN_AREA_RATIO
            <= median_area_ratio
            <= SAM3_CALIBRATED_FORK_SHADOW_MAX_AREA_RATIO
            and distance_balance
            >= SAM3_CALIBRATED_FORK_SHADOW_MIN_DISTANCE_BALANCE
        )
        if not geometry_valid:
            classification = "fork_shadow_geometry_rejected"
            triggers_intervention = False
        elif max_iom is None:
            classification = "ambiguous_fork_shadow_candidate"
            triggers_intervention = True
        elif max_iom < SAM3_CALIBRATED_FORK_SHADOW_LOW_IOM:
            classification = "possible_missed_fork"
            triggers_intervention = True
        elif max_iom < SAM3_CALIBRATED_FORK_SHADOW_HIGH_IOM:
            classification = "ambiguous_fork_shadow_candidate"
            triggers_intervention = True
        else:
            classification = "duplicate_of_existing_fork"
            triggers_intervention = False

        related_ids = [str(neighbor.get("label", "-")) for neighbor in bracket]
        diagnostic = {
            "shadow_id": str(det.get("shadow_label", "FS?")),
            "query_index": det.get("query_index"),
            "query_match_score": round(
                float(det.get("query_match_score", 0.0)), 6
            ),
            "center": [round(value, 6) for value in center],
            "bbox": normalized_bbox_for_det(det),
            "between_mask_ids": related_ids,
            "max_iom": None if max_iom is None else round(max_iom, 6),
            "area_to_primary_median_ratio": round(median_area_ratio, 6),
            "bracket_distance_balance": round(distance_balance, 6),
            "classification": classification,
            "triggers_intervention": triggers_intervention,
        }
        diagnostics.append(diagnostic)
        if not triggers_intervention:
            continue
        reason = (
            "possible_fork_missed_below_primary_threshold"
            if classification == "possible_missed_fork"
            else "ambiguous_fork_shadow_candidate"
        )
        if reason not in reasons:
            reasons.append(reason)
        uncertain_regions.append({
            "region_id": f"F{len(uncertain_regions) + 1}",
            "uncertainty_type": classification,
            "related_mask_ids": related_ids,
            "shadow_id": diagnostic["shadow_id"],
            "normalized_bbox": diagnostic["bbox"],
            "evidence": {
                "query_match_score": diagnostic["query_match_score"],
                "shadow_score_threshold": SAM3_CALIBRATED_FORK_SHADOW_THRESHOLD,
                "primary_score_threshold": SAM3_CALIBRATED_SCORE_THRESHOLDS["fork"],
                "max_iom": diagnostic["max_iom"],
                "area_to_primary_median_ratio": diagnostic[
                    "area_to_primary_median_ratio"
                ],
                "bracket_distance_balance": diagnostic[
                    "bracket_distance_balance"
                ],
            },
            "recommended_action": "review_local_fork_cluster",
        })

    explanation = (
        f"The formal count was preserved, but {len(uncertain_regions)} "
        "localized fork shadow candidate(s) may represent missed forks."
        if reasons
        else "Fork shadow candidates were absent, geometrically implausible, "
        "or explained as duplicates of formal masks."
    )
    return {
        "requires_intervention": bool(reasons),
        "intervention_reasons": reasons,
        "intervention_explanation": explanation,
        "uncertain_regions": uncertain_regions,
        "shadow_candidate_count": len(diagnostics),
        "shadow_candidates": diagnostics,
        "shadow_duplicate_groups": shadow_duplicate_groups,
    }


def _median(values):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])




















def detect_sam3_candidates(
    img,
    args,
    target_object=None,
    score_threshold=None,
    force_presence_one=False,
    max_candidates=None,
):
    if not args.sam3_detect_buttons:
        return []

    from transformers import Sam3Model, Sam3Processor

    tower_name = (
        getattr(args, "sam3_vision_tower", None)
        or getattr(args, "mm_sam3_vision_tower", None)
        or "/workspace/sam3_official/fork_finetuning/exported_models/sam3_fork_scissors_joint_balanced_official_full_ft"
    )

    device = torch.device(args.sam3_detect_device)
    if device.type == "cuda":
        try:
            torch.cuda.current_device()
        except Exception as exc:
            print(
                "SAM3 detector requested CUDA but CUDA is unavailable; "
                f"falling back to CPU. Reason: {exc}"
            )
            device = torch.device("cpu")
    dtype = resolve_dtype(args.sam3_detect_dtype, device)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    detection_text = str(target_object or args.sam3_detect_text).strip()
    score_threshold = (
        float(args.sam3_detect_threshold)
        if score_threshold is None
        else float(score_threshold)
    )
    print(
        "SAM3 detector: "
        f"text={detection_text!r}, "
        f"threshold={score_threshold}, "
        f"mask_threshold={getattr(args, 'sam3_detect_mask_threshold', 0.5)}, "
        f"device={device}, "
        f"dtype={dtype}"
    )

    processor = Sam3Processor.from_pretrained(tower_name)
    model = Sam3Model.from_pretrained(tower_name)
    model.requires_grad_(False)
    model.eval()
    model.to(device=device, dtype=dtype)

    inputs = processor(images=img, text=detection_text, return_tensors="pt")
    inputs = {
        key: value.to(device=device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

    with torch.no_grad():
        outputs = model(**inputs)

    query_logits = outputs.pred_logits[0]
    query_match_scores = query_logits.sigmoid()
    if force_presence_one or outputs.presence_logits is None:
        presence_logit = None
        presence_score = query_match_scores.new_tensor(1.0)
    else:
        presence_logit = outputs.presence_logits[0].reshape(-1)[0]
        presence_score = presence_logit.sigmoid()
    final_query_scores = query_match_scores * presence_score
    passed_query_indices = torch.where(
        final_query_scores > score_threshold
    )[0].detach().to(device="cpu")

    postprocess_presence_logits = outputs.presence_logits
    if force_presence_one:
        outputs.presence_logits = None
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=float(getattr(args, "sam3_detect_mask_threshold", 0.5)),
        target_sizes=[img.size[::-1]],
    )[0]
    outputs.presence_logits = postprocess_presence_logits

    boxes = results["boxes"].detach().float().cpu()
    scores = results["scores"].detach().float().cpu()
    masks = results["masks"].detach().to(device="cpu").bool()
    order = torch.argsort(scores, descending=True)

    top_candidates = []
    width, height = img.size
    configured_limit = (
        int(args.sam3_max_candidates)
        if max_candidates is None
        else int(max_candidates)
    )
    candidate_limit = (
        len(order)
        if configured_limit <= 0
        else min(len(order), configured_limit)
    )
    for ordered_idx in order[:candidate_limit]:
        idx = int(ordered_idx.item())
        mask = masks[idx]
        mask_y, mask_x = torch.where(mask)
        if mask_x.numel() == 0:
            continue
        x1 = float(mask_x.min().item())
        y1 = float(mask_y.min().item())
        x2 = float(mask_x.max().item() + 1)
        y2 = float(mask_y.max().item() + 1)
        score = float(scores[idx].item())
        query_index = int(passed_query_indices[idx].item())
        top_candidates.append({
            "score": score,
            "query_index": query_index,
            "query_match_logit": float(
                query_logits[query_index].detach().float().cpu().item()
            ),
            "query_match_score": float(
                query_match_scores[query_index].detach().float().cpu().item()
            ),
            "presence_logit": (
                None
                if presence_logit is None
                else float(presence_logit.detach().float().cpu().item())
            ),
            "presence_score": float(
                presence_score.detach().float().cpu().item()
            ),
            "box": (x1, y1, x2, y2),
            "model_box": tuple(float(value) for value in boxes[idx].tolist()),
            "mask": mask,
            "mask_area": float(mask.sum().item()) / float(max(1, width * height)),
            "center": (
                float(mask_x.float().mean().item()) / width,
                float(mask_y.float().mean().item()) / height,
            ),
            "image_width": width,
            "image_height": height,
        })
        
    # Keep display/evidence IDs spatially stable: left to right by mask centroid.
    top_candidates.sort(key=lambda det: (
        float(det["center"][0]),
        float(det["center"][1]),
        -float(det["score"]),
    ))

    detections = []
    for rank, det in enumerate(top_candidates):
        det["label"] = str(rank + 1)
        detections.append(det)

    print(f"SAM3 detected {len(detections)} raw instance mask(s).")
    for det in detections:
        cx, cy = det["center"]
        x1, y1, x2, y2 = det["box"]
        print(
            f"  {det['label']}: score={det['score']:.3f}, "
            f"center=({cx:.3f}, {cy:.3f}), "
            f"mask_box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), "
            f"mask_area={det.get('mask_area', 0.0):.5f}"
        )

    draw_sam3_detections(img, detections, args.sam3_detections_out_path)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return detections


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, device, use_flash_attn=False,
                 mm_use_sam3_conditioning=False,
                 mm_sam3_vision_tower=None,
                 mm_sam3_blend_alpha=0.35,
                 mm_sam3_mask_gamma=1.0,
                 mm_sam3_device="cpu",
                 mm_sam3_dtype="auto",
                 mm_sam3_unload_after_forward=False):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        self.device = device
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        load_start = time.time()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device, use_flash_attn=use_flash_attn)
        load_elapsed = time.time() - load_start
        logger.info(
            "Model load completed in "
            f"{load_elapsed:.2f}s ({load_elapsed / 60.0:.2f} min)"
        )
        self._apply_sam3_runtime_config(
            mm_use_sam3_conditioning=mm_use_sam3_conditioning,
            mm_sam3_vision_tower=mm_sam3_vision_tower,
            mm_sam3_blend_alpha=mm_sam3_blend_alpha,
            mm_sam3_mask_gamma=mm_sam3_mask_gamma,
            mm_sam3_device=mm_sam3_device,
            mm_sam3_dtype=mm_sam3_dtype,
            mm_sam3_unload_after_forward=mm_sam3_unload_after_forward,
        )

    def _apply_sam3_runtime_config(self, mm_use_sam3_conditioning,
                                   mm_sam3_vision_tower,
                                   mm_sam3_blend_alpha,
                                   mm_sam3_mask_gamma,
                                   mm_sam3_device,
                                   mm_sam3_dtype,
                                   mm_sam3_unload_after_forward):
        mm_use_sam3_conditioning = bool(mm_use_sam3_conditioning)
        mm_sam3_blend_alpha = float(mm_sam3_blend_alpha)
        mm_sam3_mask_gamma = float(mm_sam3_mask_gamma)
        mm_sam3_vision_tower = mm_sam3_vision_tower or None

        if mm_use_sam3_conditioning and not mm_sam3_vision_tower:
            raise ValueError(
                "--mm-sam3-vision-tower is required when "
                "--mm-use-sam3-conditioning is true."
            )

        self.model.config.mm_use_sam3_conditioning = mm_use_sam3_conditioning
        self.model.config.mm_sam3_vision_tower = mm_sam3_vision_tower
        self.model.config.mm_sam3_blend_alpha = mm_sam3_blend_alpha
        self.model.config.mm_sam3_mask_gamma = mm_sam3_mask_gamma
        self.model.config.mm_sam3_device = mm_sam3_device
        self.model.config.mm_sam3_dtype = mm_sam3_dtype
        self.model.config.mm_sam3_unload_after_forward = mm_sam3_unload_after_forward

        vision_tower = self.model.get_vision_tower()
        if vision_tower is not None:
            vision_tower.mm_use_sam3_conditioning = mm_use_sam3_conditioning
            vision_tower.mm_sam3_vision_tower = mm_sam3_vision_tower
            vision_tower.mm_sam3_blend_alpha = mm_sam3_blend_alpha
            vision_tower.mm_sam3_mask_gamma = mm_sam3_mask_gamma
            vision_tower.mm_sam3_device = mm_sam3_device
            vision_tower.mm_sam3_dtype = mm_sam3_dtype
            vision_tower.mm_sam3_unload_after_forward = mm_sam3_unload_after_forward

        logger.info(
            "SAM3 conditioning config: "
            f"enabled={mm_use_sam3_conditioning}, "
            f"tower={mm_sam3_vision_tower}, "
            f"alpha={mm_sam3_blend_alpha}, "
            f"gamma={mm_sam3_mask_gamma}, "
            f"device={mm_sam3_device}, "
            f"dtype={mm_sam3_dtype}, "
            f"unload_after_forward={mm_sam3_unload_after_forward}"
        )

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        lm_head = getattr(model, "lm_head", None)
        amp_dtype = None
        if lm_head is not None and hasattr(lm_head, "weight"):
            if lm_head.weight.dtype in (torch.float16, torch.bfloat16):
                amp_dtype = lm_head.weight.dtype

        prompt = params["prompt"]
        ori_prompt = prompt
        counting_pass1 = is_counting_pass1_request(params)
        counting_method = get_counting_method(params) if counting_pass1 else None
        target_object = params.get("target_object", "fork")

        def determine_task_type(text: str) -> str:
            import re
            pattern = r"(find|point|count|how many|locate).*(fork|forks|instrument)"
            if re.search(pattern, text, re.IGNORECASE):
                return "robopoint"
            return "general"
            
        task_type = "robopoint" if counting_pass1 else determine_task_type(prompt)

        if task_type == "general":
            try:
                import os, sys
                sop_llm_path = "/workspace/ros2_ws/src/SOP_LLM"
                db_path = os.path.join(sop_llm_path, "chroma_db_v2")
                
                logger.info(f"RAG DB Path: {db_path}")

                if sop_llm_path not in sys.path:
                    sys.path.append(sop_llm_path)
                    
                from query_db import get_rag_context
                
                user_query = prompt
                for marker in (" ASSISTANT:", "\nASSISTANT:", "Assistant:"):
                    if marker in user_query:
                        user_query = user_query.split(marker)[-2]
                        break
                if "USER:" in user_query:
                    user_query = user_query.split("USER:")[-1]
                user_query = user_query.replace(DEFAULT_IMAGE_TOKEN, "").strip()
                
                context = get_rag_context(db_path, user_query, k=3)
                print(f"--- 找到的 RAG 上下文 ---\n{context}\n-------------------------")
                if context:
                    rag_instruction = (
                        f"\n\n[參考資料 / Context]\n{context}\n\n"
                        "Instruction: You are a helpful AI assistant. Please answer the user's question. "
                        "CRITICAL RULE: First, evaluate if the [Context] above is relevant to the user's question. "
                        "- If relevant: Answer strictly based on the [Context]. Be concise. "
                        "- If completely irrelevant (e.g., the user is just saying hello, or asking 'who are you'): IGNORE the [Context] completely and answer naturally as an AI."
                    )
                    prompt = append_to_last_user_turn(prompt, rag_instruction)
            except Exception as e:
                logger.error(f"Failed to get RAG context: {e}")

        images = params.get("images", None)
        num_image_tokens = 0
        sam3_candidates = []
        sam3_shadow_candidates = []
        fork_iom_deduplication = {
            "applied": False,
            "iom_threshold": SAM3_CALIBRATED_FORK_DUPLICATE_MASK_IOM,
            "borderline_mask_iom_threshold": (
                SAM3_CALIBRATED_FORK_BORDERLINE_MASK_IOM
            ),
            "borderline_bbox_iom_threshold": (
                SAM3_CALIBRATED_FORK_BORDERLINE_BBOX_IOM
            ),
            "borderline_max_center_distance": (
                SAM3_CALIBRATED_FORK_BORDERLINE_MAX_CENTER_DISTANCE
            ),
            "scoring_method": "three_binary_signals",
            "points_per_signal": 1,
            "duplicate_min_score": SAM3_CALIBRATED_FORK_DUPLICATE_MIN_SCORE,
            "intervention_score": SAM3_CALIBRATED_FORK_INTERVENTION_SCORE,
            "removed_count": 0,
            "removed_duplicates": [],
            "requires_intervention": False,
            "ambiguous_pairs": [],
        }
        if images is not None and len(images) > 0:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                pil_images = [load_image_from_base64(image) for image in images]
                image_sizes = [image.size for image in pil_images]
                if pil_images and task_type == "robopoint":
                    raw_experiment = counting_method == "sam3_raw"
                    calibrated_inference = (
                        counting_method == "sam3_calibrated"
                    )
                    calibrated_scissors = bool(
                        calibrated_inference
                        and "scissor" in str(target_object or "").lower()
                    )
                    calibrated_fork = bool(
                        calibrated_inference
                        and str(target_object or "").strip().lower() == "fork"
                    )
                    sam3_candidate_threshold = float(
                        get_sam3_calibrated_score_threshold(target_object)
                        if calibrated_inference
                        else args.sam3_detect_threshold
                    )
                    sam3_detection_threshold = (
                        SAM3_CALIBRATED_SCISSORS_SHADOW_THRESHOLD
                        if calibrated_scissors
                        else (
                            SAM3_CALIBRATED_FORK_SHADOW_THRESHOLD
                            if calibrated_fork
                            else sam3_candidate_threshold
                        )
                    )
                    sam3_candidates = detect_sam3_candidates(
                        pil_images[0],
                        args,
                        target_object=target_object,
                        score_threshold=(
                            sam3_detection_threshold
                            if raw_experiment or calibrated_inference
                            else None
                        ),
                        force_presence_one=(
                            raw_experiment or calibrated_inference
                        ),
                        max_candidates=(
                            0
                        ) if raw_experiment or calibrated_inference else None,
                    )
                    if calibrated_scissors:
                        (
                            sam3_candidates,
                            sam3_shadow_candidates,
                        ) = split_calibrated_scissors_candidates(
                            sam3_candidates,
                            sam3_candidate_threshold,
                        )
                        draw_sam3_detections(
                            pil_images[0],
                            sam3_candidates,
                            args.sam3_detections_out_path,
                        )
                    elif calibrated_fork:
                        (
                            sam3_candidates,
                            sam3_shadow_candidates,
                        ) = split_calibrated_fork_candidates(
                            sam3_candidates,
                            sam3_candidate_threshold,
                        )
                    if calibrated_fork:
                        (
                            sam3_candidates,
                            fork_iom_deduplication,
                        ) = deduplicate_calibrated_fork_candidates(
                            sam3_candidates,
                        )
                        draw_sam3_detections(
                            pil_images[0],
                            sam3_candidates,
                            args.sam3_detections_out_path,
                        )
                if counting_pass1 and counting_method in {
                    "sam3_raw",
                    "sam3_calibrated",
                    "sam3_prompt_experiment",
                }:
                    response = build_sam3_prompt_experiment_output(
                        sam3_candidates,
                        target_object,
                    )
                    if counting_method in {
                        "sam3_raw",
                        "sam3_calibrated",
                    }:
                            calibrated = counting_method == "sam3_calibrated"
                            no_detection = not sam3_candidates
                            calibrated_intervention = (
                                build_sam3_calibrated_intervention(
                                    sam3_candidates,
                                    target_object,
                                    sam3_candidate_threshold,
                                )
                                if calibrated
                                else {
                                    "requires_intervention": no_detection,
                                    "intervention_reasons": (
                                        ["sam3_no_detection"]
                                        if no_detection
                                        else []
                                    ),
                                    "intervention_explanation": None,
                                    "uncertain_regions": (
                                        [{
                                            "region_id": "full_image",
                                            "uncertainty_type": "sam3_no_detection",
                                            "related_mask_ids": [],
                                        }]
                                        if no_detection
                                        else []
                                    ),
                                    "fork_border_contact_count": 0,
                                    "fork_border_contact_ratio": 0.0,
                                    "minimum_score_margin": None,
                                }
                            )
                            scissors_shadow_review = {
                                "requires_intervention": False,
                                "intervention_reasons": [],
                                "intervention_explanation": None,
                                "uncertain_regions": [],
                                "shadow_candidate_count": 0,
                                "shadow_candidates": [],
                            }
                            fork_shadow_review = {
                                "requires_intervention": False,
                                "intervention_reasons": [],
                                "intervention_explanation": None,
                                "uncertain_regions": [],
                                "shadow_candidate_count": 0,
                                "shadow_candidates": [],
                                "shadow_duplicate_groups": [],
                            }
                            if calibrated and calibrated_scissors:
                                scissors_shadow_review = (
                                    build_scissors_shadow_intervention(
                                        sam3_candidates,
                                        sam3_shadow_candidates,
                                    )
                                )
                            if calibrated and calibrated_fork:
                                fork_shadow_review = build_fork_shadow_intervention(
                                    sam3_candidates,
                                    sam3_shadow_candidates,
                                )
                                if fork_iom_deduplication[
                                    "requires_intervention"
                                ]:
                                    calibrated_intervention[
                                        "requires_intervention"
                                    ] = True
                                    reason = "ambiguous_formal_fork_pair"
                                    if reason not in calibrated_intervention[
                                        "intervention_reasons"
                                    ]:
                                        calibrated_intervention[
                                            "intervention_reasons"
                                        ].append(reason)
                                    for pair_index, pair in enumerate(
                                        fork_iom_deduplication[
                                            "ambiguous_pairs"
                                        ],
                                        start=1,
                                    ):
                                        calibrated_intervention[
                                            "uncertain_regions"
                                        ].append({
                                            "region_id": f"FD{pair_index}",
                                            "uncertainty_type": (
                                                "ambiguous_formal_fork_pair"
                                            ),
                                            "related_mask_ids": [
                                                pair["retained_mask_id"],
                                                pair["candidate_mask_id"],
                                            ],
                                            "normalized_bbox": pair[
                                                "normalized_bbox"
                                            ],
                                            "evidence": {
                                                "duplicate_score": pair[
                                                    "duplicate_score"
                                                ],
                                                "signals": pair["evidence"],
                                                "mask_iom": pair["mask_iom"],
                                                "bbox_iom": pair["bbox_iom"],
                                                "center_distance": pair[
                                                    "center_distance"
                                                ],
                                                "mask_area_ratio": pair[
                                                    "mask_area_ratio"
                                                ],
                                            },
                                            "recommended_action": (
                                                "review_possible_duplicate_forks"
                                            ),
                                        })
                                    calibrated_intervention[
                                        "intervention_explanation"
                                    ] = (
                                        "The formal count was preserved, but "
                                        f"{len(fork_iom_deduplication['ambiguous_pairs'])} "
                                        "fork pair(s) matched exactly 2 of 3 "
                                        "duplicate-evidence signals."
                                    )
                            for shadow_review in (
                                scissors_shadow_review,
                                fork_shadow_review,
                            ):
                                calibrated_intervention[
                                    "requires_intervention"
                                ] = bool(
                                    calibrated_intervention[
                                        "requires_intervention"
                                    ]
                                    or shadow_review[
                                        "requires_intervention"
                                    ]
                                )
                                for reason in shadow_review[
                                    "intervention_reasons"
                                ]:
                                    if reason not in calibrated_intervention[
                                        "intervention_reasons"
                                    ]:
                                        calibrated_intervention[
                                            "intervention_reasons"
                                        ].append(reason)
                                calibrated_intervention[
                                    "uncertain_regions"
                                ].extend(
                                    shadow_review[
                                        "uncertain_regions"
                                    ]
                                )
                                if shadow_review[
                                    "requires_intervention"
                                ]:
                                    calibrated_intervention[
                                        "intervention_explanation"
                                    ] = shadow_review[
                                        "intervention_explanation"
                                    ]
                            active_shadow_review = (
                                scissors_shadow_review
                                if calibrated_scissors
                                else fork_shadow_review
                            )
                            response.update({
                                "counting_method": counting_method,
                                "postprocessing_applied": bool(
                                    calibrated
                                    and fork_iom_deduplication["applied"]
                                ),
                                "presence_override": 1.0,
                                "score_threshold": sam3_candidate_threshold,
                                "mask_threshold": float(
                                    getattr(
                                        args,
                                        "sam3_detect_mask_threshold",
                                        0.5,
                                    )
                                ),
                                "custom_geometry_applied": bool(
                                    calibrated
                                    and str(target_object or "").strip().lower()
                                    == "fork"
                                ),
                                "iom_threshold": (
                                    SAM3_CALIBRATED_FORK_DUPLICATE_MASK_IOM
                                    if calibrated
                                    and str(target_object or "").strip().lower()
                                    == "fork"
                                    else None
                                ),
                                "calibration_profile": (
                                    SAM3_CALIBRATED_PROFILE
                                    if calibrated
                                    else None
                                ),
                                "requires_intervention": calibrated_intervention[
                                    "requires_intervention"
                                ],
                                "intervention_reasons": calibrated_intervention[
                                    "intervention_reasons"
                                ],
                                "intervention_explanation": calibrated_intervention[
                                    "intervention_explanation"
                                ],
                                "correction_applied": bool(
                                    calibrated
                                    and fork_iom_deduplication["applied"]
                                ),
                                "correction_reasons": (
                                    ["duplicate_fork_evidence_score"]
                                    if calibrated
                                    and fork_iom_deduplication["applied"]
                                    else []
                                ),
                                "fork_iom_deduplication": (
                                    fork_iom_deduplication
                                    if calibrated
                                    and str(target_object or "").strip().lower()
                                    == "fork"
                                    else None
                                ),
                                "uncertain_regions": calibrated_intervention[
                                    "uncertain_regions"
                                ],
                                "fork_border_contact_count": calibrated_intervention[
                                    "fork_border_contact_count"
                                ],
                                "fork_border_contact_ratio": calibrated_intervention[
                                    "fork_border_contact_ratio"
                                ],
                                "minimum_score_margin": calibrated_intervention[
                                    "minimum_score_margin"
                                ],
                                "shadow_score_threshold": (
                                    SAM3_CALIBRATED_SCISSORS_SHADOW_THRESHOLD
                                    if calibrated and calibrated_scissors
                                    else (
                                        SAM3_CALIBRATED_FORK_SHADOW_THRESHOLD
                                        if calibrated and calibrated_fork
                                        else None
                                    )
                                ),
                                "shadow_candidate_count": active_shadow_review[
                                    "shadow_candidate_count"
                                ],
                                "shadow_candidates": active_shadow_review[
                                    "shadow_candidates"
                                ],
                                "shadow_duplicate_groups": (
                                    fork_shadow_review["shadow_duplicate_groups"]
                                    if calibrated and calibrated_fork
                                    else []
                                ),
                                "shadow_iom_low_threshold": (
                                    SAM3_CALIBRATED_SCISSORS_SHADOW_LOW_IOM
                                    if calibrated and calibrated_scissors
                                    else (
                                        SAM3_CALIBRATED_FORK_SHADOW_LOW_IOM
                                        if calibrated and calibrated_fork
                                        else None
                                    )
                                ),
                                "shadow_iom_high_threshold": (
                                    SAM3_CALIBRATED_SCISSORS_SHADOW_HIGH_IOM
                                    if calibrated and calibrated_scissors
                                    else (
                                        SAM3_CALIBRATED_FORK_SHADOW_HIGH_IOM
                                        if calibrated and calibrated_fork
                                        else None
                                    )
                                ),
                                "count_preserved_for_review": not bool(
                                    calibrated
                                    and fork_iom_deduplication["applied"]
                                ),
                                "overlay_path": str(
                                    args.sam3_detections_out_path
                                ),
                            })
                            if calibrated:
                                if (
                                    str(target_object or "").strip().lower()
                                    == "fork"
                                ):
                                    response["reason"] = (
                                        "Fine-tuned SAM3 calibrated inference retained "
                                        f"{len(sam3_candidates)} mask(s) for "
                                        f"{target_object!r} at "
                                        f"p_match>{sam3_candidate_threshold:.5f} after "
                                        f"removing {fork_iom_deduplication['removed_count']} "
                                        "duplicate mask(s) using the calibrated "
                                        "three-signal fork duplicate score."
                                    )
                                else:
                                    response["reason"] = (
                                        "Fine-tuned SAM3 calibrated inference retained "
                                        f"{len(sam3_candidates)} mask(s) for "
                                        f"{target_object!r} at "
                                        f"p_match>{sam3_candidate_threshold:.5f}; "
                                        "custom geometry and count-changing IoM "
                                        "post-processing were disabled."
                                    )
                    response_text = json.dumps(
                        response,
                        ensure_ascii=False,
                    )
                    yield json.dumps({"text": ori_prompt + response_text, "error_code": 0}).encode() + b"\0"
                    return
                images = process_images(pil_images, image_processor, model.config)

                image_dtype = amp_dtype if amp_dtype is not None else torch.float16
                if type(images) is list:
                    images = [image.to(self.model.device, dtype=image_dtype) for image in images]
                else:
                    images = images.to(self.model.device, dtype=image_dtype)

                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                if sam3_candidates and not counting_pass1:
                    candidate_lines = []
                    for det in sam3_candidates:
                        cx, cy = det["center"]
                        candidate_lines.append(
                            f"{det['label']}: center=({cx:.3f}, {cy:.3f}), "
                            f"score={det['score']:.3f}"
                        )
                    if args.sam3_max_points == 1:
                        format_suffix = "\nAnswer with exactly one candidate label only, e.g. 1."
                    else:
                        format_suffix = (
                            f"\nAnswer with ALL relevant candidate labels up to {args.sam3_max_points}, "
                            "separated by commas, e.g. 1, 2, 3."
                        )
                    prompt = append_to_last_user_turn(prompt, (
                        "\nSAM3 found these candidate objects:\n"
                        + "\n".join(candidate_lines)
                        + "\nYou must choose only from the SAM3 candidate labels above. "
                        "Do not invent a new coordinate."
                        + format_suffix
                    ))
                elif task_type == "robopoint" and not counting_pass1:
                    if args.sam3_max_points == 1:
                        prompt = append_to_last_user_turn(prompt, (
                            "\nChoose the single best point only. "
                            "Your answer must be formatted as exactly one tuple, i.e. (x, y), "
                            "where x and y are normalized coordinates between 0 and 1."
                        ))
                    else:
                        prompt = append_to_last_user_turn(prompt, (
                            f"\nChoose ALL relevant points up to {args.sam3_max_points} points. "
                            "Your answer should be formatted as a list of tuples, "
                            "i.e. [(x1, y1), (x2, y2), ...], where each tuple contains "
                            "the x and y coordinates of a point satisfying the conditions above. "
                            "The coordinates should be between 0 and 1, indicating the normalized "
                            "pixel locations of the points in the image."
                        ))

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
                image_sizes = None
            image_args = {"images": images, "image_sizes": image_sizes, "task_type": task_type}
        else:
            images = None
            image_args = {"task_type": task_type}

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 512)), 1024)
        use_cache = str2bool(str(params.get("use_cache", False)))
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return

        generation_kwargs = dict(
            inputs=input_ids,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=use_cache,
            **image_args
        )
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        generation_error = []

        def run_generate():
            try:
                autocast_ctx = contextlib.nullcontext()
                if torch.device(self.device).type == "cuda" and amp_dtype is not None:
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
                with torch.no_grad(), autocast_ctx:
                    model.generate(**generation_kwargs)
            except RuntimeError as exc:
                if (
                    "expected mat1 and mat2 to have the same dtype" in str(exc)
                    and lm_head is not None
                    and hasattr(lm_head, "to")
                ):
                    print("Detected dtype mismatch in lm_head; retrying with lm_head cast to float32...")
                    try:
                        lm_head.to(dtype=torch.float32)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        with torch.no_grad():
                            model.generate(**generation_kwargs)
                        return
                    except Exception as retry_exc:
                        generation_error.append(retry_exc)
                else:
                    generation_error.append(exc)
                streamer.end()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                generation_error.append(exc)
                streamer.end()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        thread = Thread(target=run_generate)
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if stop_str and generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
        thread.join()
        if generation_error:
            raise generation_error[0]

        completion_text = generated_text[len(ori_prompt):]
        print(f"Model raw output: {completion_text!r}")
        response_text = completion_text

        if sam3_candidates and not counting_pass1:
            label_to_coord = {det["label"]: det["center"] for det in sam3_candidates}
            selected_labels = parse_candidate_labels(completion_text, label_to_coord.keys())
            generated_coords = parse_coords(completion_text)
            if not selected_labels:
                selected_labels = snap_coords_to_candidates_with_limit(
                    generated_coords,
                    sam3_candidates,
                    args.sam3_max_points,
                )
                if selected_labels:
                    print(f"Snapped generated coordinates to nearest SAM3 labels: {selected_labels}")
            if not selected_labels:
                fallback_count = args.sam3_max_points if args.sam3_max_points > 0 else len(sam3_candidates)
                selected_labels = [det["label"] for det in sam3_candidates[:fallback_count]]
                print(f"No usable model selection; falling back to top SAM3 labels: {selected_labels}")
            
            # 移除最大返回點數限制，讓選到的標籤全數保留
            # if args.sam3_max_points > 0:
            #     selected_labels = selected_labels[:args.sam3_max_points]
            
            label_to_det = {det["label"]: det for det in sam3_candidates}
            coords = [label_to_coord[label] for label in selected_labels]
            scores = [label_to_det[label].get("score", 0.0) for label in selected_labels]
            
            if not generated_coords and selected_labels:
                snapped_points = [
                    (round(nx, 4), round(ny, 4))
                    for nx, ny in coords
                ]
                response_text = (
                    ", ".join(
                        f"{label}: ({round(nx, 4)}, {round(ny, 4)}) [confidence: {round(float(score), 4)}]"
                        for label, (nx, ny), score in zip(selected_labels, coords, scores)
                    ) +
                    f"\n__SNAPPED_POINTS__:{json.dumps(snapped_points)}\n" +
                    f"__CONFIDENCE_SCORES__:{json.dumps([round(float(s), 4) for s in scores])}"
                )
            elif generated_coords and selected_labels:
                snapped_points = [
                    (round(nx, 4), round(ny, 4))
                    for nx, ny in coords
                ]
                response_text = (
                    f"{completion_text}\n"
                    f"__SNAPPED_POINTS__:{json.dumps(snapped_points)}\n"
                    f"__CONFIDENCE_SCORES__:{json.dumps([round(float(s), 4) for s in scores])}"
                )
            print(f"Selected SAM3 labels: {selected_labels}")

        yield json.dumps({"text": ori_prompt + response_text, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": f"Worker error: {type(e).__name__}: {e}",
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=22000)
    parser.add_argument("--worker-address", type=str,
        default="http://127.0.0.1:22000")
    parser.add_argument("--controller-address", type=str,
        default="http://127.0.0.1:11000")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--mm-use-sam3-conditioning", type=str2bool, default=True)
    parser.add_argument(
        "--mm-sam3-vision-tower",
        type=str,
        default="/workspace/sam3_official/fork_finetuning/exported_models/sam3_fork_scissors_joint_balanced_official_full_ft",
    )
    parser.add_argument("--mm-sam3-blend-alpha", type=float, default=0.35)
    parser.add_argument("--mm-sam3-mask-gamma", type=float, default=1.0)
    parser.add_argument("--mm-sam3-device", type=str, default="cuda")
    parser.add_argument("--mm-sam3-dtype", type=str, default="bfloat16")
    parser.add_argument("--mm-sam3-unload-after-forward", type=str2bool, default=False)
    parser.add_argument("--sam3-detect-buttons", type=str2bool, default=True)
    parser.add_argument("--sam3-detect-text", type=str, default="fork")
    parser.add_argument("--sam3-detect-threshold", type=float, default=0.20)
    parser.add_argument("--sam3-detect-mask-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-max-candidates", type=int, default=12)
    parser.add_argument("--sam3-max-points", type=int, default=10) # 輸出限制
    parser.add_argument("--sam3-detect-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sam3-detect-dtype", type=str, default="bfloat16")
    parser.add_argument("--sam3-detections-out-path", type=str, default="/workspace/sam3_worker_candidates.jpg")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.device,
                         use_flash_attn=args.use_flash_attn,
                         mm_use_sam3_conditioning=args.mm_use_sam3_conditioning,
                         mm_sam3_vision_tower=args.mm_sam3_vision_tower,
                         mm_sam3_blend_alpha=args.mm_sam3_blend_alpha,
                         mm_sam3_mask_gamma=args.mm_sam3_mask_gamma,
                         mm_sam3_device=args.mm_sam3_device,
                         mm_sam3_dtype=args.mm_sam3_dtype,
                         mm_sam3_unload_after_forward=args.mm_sam3_unload_after_forward)
    if not args.no_register:
        worker.register_to_controller()
        heart_beat_thread = threading.Thread(
            target=heart_beat_worker, args=(worker,), daemon=True)
        heart_beat_thread.start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
