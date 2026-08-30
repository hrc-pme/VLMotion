"""
A model worker executes the model.
"""
import argparse
import asyncio
import contextlib
import json
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
import string

from point.constants import WORKER_HEART_BEAT_INTERVAL
from point.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from point.model.builder import load_pretrained_model
from point.mm_utils import process_images, load_image_from_base64, tokenizer_image_token
from point.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextIteratorStreamer
from threading import Thread
from PIL import ImageDraw, ImageStat


GB = 1 << 30

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
    for match in re.findall(r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', text):
        coords.append((float(match[0]), float(match[1])))
    return coords


def parse_candidate_labels(text: str, valid_labels):
    valid = set(valid_labels)
    labels = []
    for match in re.findall(r'\b([A-Z]|P\d+)\b', text.upper()):
        if match in valid and match not in labels:
            labels.append(match)
    return labels


def snap_coords_to_candidates(coords, detections, max_points):
    labels = []
    for coord in coords:
        best = min(detections, key=lambda det: candidate_distance_sq(coord, det))
        if best["label"] not in labels:
            labels.append(best["label"])
        if max_points > 0 and len(labels) >= max_points:
            break
    return labels


def candidate_distance_sq(coord, det):
    return (det["center"][0] - coord[0]) ** 2 + (det["center"][1] - coord[1]) ** 2


def snap_coords_to_candidates_with_limit(coords, detections, max_points):
    if not coords:
        return []

    detections = [det for det in detections if det.get("crop_role") != "other"]
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


def snap_coords_to_candidates_with_distances(coords, detections, max_points):
    detections = [det for det in detections if det.get("crop_role") != "other"]
    if not coords or not detections:
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
        dist_sq = candidate_distance_sq(coord, best)
        if dist_sq > max_snap_dist_sq:
            continue
        if best["label"] not in [label for label, _ in labels]:
            labels.append((best["label"], dist_sq ** 0.5))
        if max_points > 0 and len(labels) >= max_points:
            break
    return labels


def _cluster_sorted(values, tolerance):
    clusters = []
    for value in sorted(values):
        if not clusters or abs(value - clusters[-1][-1]) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [sum(cluster) / len(cluster) for cluster in clusters]


def classify_candidate_crop(img, det):
    width, height = img.size
    x1, y1, x2, y2 = det["box"]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad = max(box_w, box_h) * 0.45
    crop_box = (
        max(0, int(round(x1 - pad))),
        max(0, int(round(y1 - pad))),
        min(width, int(round(x2 + pad))),
        min(height, int(round(y2 + pad))),
    )
    crop = img.crop(crop_box).convert("RGB").resize((32, 32))
    pixels = list(crop.getdata())

    def blue_ratio_for(pixel_values):
        blue_count = 0
        for r, g, b in pixel_values:
            if b > 75 and b > r * 1.35 and b > g * 1.12:
                blue_count += 1
        return blue_count / max(1, len(pixel_values))

    inner_box = (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(width, int(round(x2))),
        min(height, int(round(y2))),
    )
    inner_crop = img.crop(inner_box).convert("RGB").resize((24, 24))
    inner_blue_ratio = blue_ratio_for(list(inner_crop.getdata()))
    blue_ratio = blue_ratio_for(pixels)
    if inner_blue_ratio > 0.18 or (blue_ratio > 0.18 and inner_blue_ratio > 0.06):
        return "other", max(blue_ratio, inner_blue_ratio)

    # A nearby wheelchair icon can enter the padded crop even when the SAM3
    # box is still centered on a metal button. Keep those candidates alive and
    # let layout/blue-position scoring decide their direction.
    if blue_ratio > 0.10:
        return "unknown", blue_ratio

    gray = crop.convert("L")
    stat = ImageStat.Stat(gray)
    mean = stat.mean[0]
    stddev = stat.stddev[0]
    threshold = max(8.0, stddev * 0.65)
    gray_pixels = list(gray.getdata())
    mask = [
        1 if abs(value - mean) >= threshold else 0
        for value in gray_pixels
    ]

    center_pixels = []
    top_mass = 0
    bottom_mass = 0
    left_mass = 0
    right_mass = 0
    total = 0
    for y in range(6, 26):
        for x in range(6, 26):
            value = mask[y * 32 + x]
            if not value:
                continue
            total += 1
            center_pixels.append((x, y))
            if y < 16:
                top_mass += 1
            else:
                bottom_mass += 1
            if x < 16:
                left_mass += 1
            else:
                right_mass += 1

    if total < 10:
        return "other", 0.0

    vertical_balance = (top_mass - bottom_mass) / total
    horizontal_balance = abs(left_mass - right_mass) / total
    confidence = abs(vertical_balance) * (1.0 - min(0.8, horizontal_balance))
    if confidence < 0.08:
        return "unknown", confidence
    return ("up" if vertical_balance > 0 else "down"), confidence


def find_blue_regions(img):
    width, height = img.size
    scale = min(1.0, 360.0 / max(width, height))
    small = img.resize((max(1, int(width * scale)), max(1, int(height * scale)))).convert("RGB")
    sw, sh = small.size
    pixels = small.load()
    visited = set()
    regions = []

    def is_blue(pixel):
        r, g, b = pixel
        return b > 95 and g > 45 and b > r * 1.55 and b > g * 1.03

    for y in range(sh):
        for x in range(sw):
            if (x, y) in visited or not is_blue(pixels[x, y]):
                continue
            stack = [(x, y)]
            visited.add((x, y))
            min_x = max_x = x
            min_y = max_y = y
            count = 0
            while stack:
                px, py = stack.pop()
                count += 1
                min_x = min(min_x, px)
                max_x = max(max_x, px)
                min_y = min(min_y, py)
                max_y = max(max_y, py)
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if (
                        nx < 0 or nx >= sw or ny < 0 or ny >= sh
                        or (nx, ny) in visited
                        or not is_blue(pixels[nx, ny])
                    ):
                        continue
                    visited.add((nx, ny))
                    stack.append((nx, ny))

            if count < 12:
                continue
            bw = (max_x - min_x + 1) / sw
            bh = (max_y - min_y + 1) / sh
            area = count / float(sw * sh)
            if area < 0.00045 or bw < 0.015 or bh < 0.015:
                continue
            regions.append({
                "center": ((min_x + max_x + 1) / (2.0 * sw), (min_y + max_y + 1) / (2.0 * sh)),
                "box": (min_x / sw, min_y / sh, (max_x + 1) / sw, (max_y + 1) / sh),
                "area": area,
            })

    return regions


def _column_centers(values):
    sorted_values = sorted(values)
    if len(sorted_values) < 4:
        return _cluster_sorted(values, 0.035)

    gaps = [
        (sorted_values[idx + 1] - sorted_values[idx], idx)
        for idx in range(len(sorted_values) - 1)
    ]
    largest_gap, split_idx = max(gaps, key=lambda item: item[0])
    x_span = sorted_values[-1] - sorted_values[0]
    if largest_gap >= max(0.012, x_span * 0.25):
        left = sorted_values[:split_idx + 1]
        right = sorted_values[split_idx + 1:]
        if left and right:
            return [sum(left) / len(left), sum(right) / len(right)]

    return _cluster_sorted(values, max(0.018, x_span * 0.20))


def estimate_layout_skew(detections):
    usable = [
        det for det in detections
        if det.get("crop_role") != "other"
    ]
    xs = sorted({det["center"][0] for det in usable})
    if len(xs) < 2:
        return 0.0
    column_centers = _column_centers(xs)
    if len(column_centers) < 2:
        return 0.0

    left_center = column_centers[0]
    right_center = column_centers[-1]
    split_x = (left_center + right_center) / 2.0
    left = sorted([det for det in usable if det["center"][0] <= split_x], key=lambda det: det["center"][1])
    right = sorted([det for det in usable if det["center"][0] > split_x], key=lambda det: det["center"][1])
    if len(left) < 2 or len(right) < 2:
        return 0.0

    ys = [det["center"][1] for det in left + right]
    y_span = max(ys) - min(ys) if len(ys) > 1 else 0.0
    max_pair_gap = max(0.045, y_span * 0.18)
    slopes = []
    used_right = set()
    for left_det in left:
        lx, ly = left_det["center"]
        right_choices = [
            (abs(right_det["center"][1] - ly), idx, right_det)
            for idx, right_det in enumerate(right)
            if idx not in used_right
        ]
        if not right_choices:
            continue
        y_gap, right_idx, right_det = min(right_choices, key=lambda item: item[0])
        if y_gap > max_pair_gap:
            continue
        rx, ry = right_det["center"]
        dx = rx - lx
        if abs(dx) < 0.01:
            continue
        used_right.add(right_idx)
        slopes.append((ry - ly) / dx)

    if not slopes:
        return 0.0
    slopes.sort()
    median = slopes[len(slopes) // 2]
    if len(slopes) >= 3:
        deviations = sorted(abs(slope - median) for slope in slopes)
        mad = deviations[len(deviations) // 2]
        if mad > 0:
            slopes = [slope for slope in slopes if abs(slope - median) <= max(0.08, mad * 3.0)]
            slopes.sort()
            median = slopes[len(slopes) // 2]

    return max(-0.35, min(0.35, median))


def apply_deskew(detections, blue_regions):
    slope = estimate_layout_skew(detections)
    usable_xs = [
        det["center"][0] for det in detections
        if det.get("crop_role") != "other"
    ]
    origin_x = sum(usable_xs) / len(usable_xs) if usable_xs else 0.5

    for det in detections:
        cx, cy = det["center"]
        det["deskew_center"] = (cx, cy - slope * (cx - origin_x))

    for region in blue_regions:
        bx, by = region["center"]
        region["deskew_center"] = (bx, by - slope * (bx - origin_x))

    return slope


def annotate_button_layout(detections):
    if not detections:
        return detections

    layout_detections = [
        det for det in detections
        if det.get("crop_role") != "other"
    ]
    if not layout_detections:
        return detections

    max_score = max(det.get("score", 0.0) for det in layout_detections)
    layout_score_floor = max_score * 0.70
    xs = [det.get("deskew_center", det["center"])[0] for det in layout_detections]
    ys = [det.get("deskew_center", det["center"])[1] for det in layout_detections]
    x_span = max(xs) - min(xs) if len(xs) > 1 else 0.0
    y_span = max(ys) - min(ys) if len(ys) > 1 else 0.0
    y_tol = max(0.008, y_span * 0.06)
    column_centers = _column_centers(xs)
    row_centers = _cluster_sorted(ys, y_tol)

    for det in detections:
        cx, cy = det.get("deskew_center", det["center"])
        col_idx = min(range(len(column_centers)), key=lambda idx: abs(column_centers[idx] - cx))
        row_idx = min(range(len(row_centers)), key=lambda idx: abs(row_centers[idx] - cy))
        det.pop("vertical_role", None)
        det.pop("vertical_pair", None)
        det["column_index"] = col_idx
        det["column"] = "left" if len(column_centers) == 1 or col_idx < len(column_centers) / 2 else "right"
        det["row_index"] = row_idx
        det["row_from_top"] = row_idx + 1

    rows = []
    for row_idx in range(len(row_centers)):
        row_dets = [
            det for det in layout_detections
            if det.get("row_index") == row_idx
        ]
        best_by_column = {}
        for det in row_dets:
            col_idx = det.get("column_index")
            current = best_by_column.get(col_idx)
            if current is None or det.get("score", 0.0) > current.get("score", 0.0):
                best_by_column[col_idx] = det
        row_buttons = list(best_by_column.values())
        if max_score > 0:
            row_buttons = [
                det for det in row_buttons
                if det.get("score", 0.0) >= layout_score_floor
            ]
        columns = {det.get("column_index") for det in row_buttons}
        rows.append({
            "index": row_idx,
            "center": row_centers[row_idx],
            "detections": row_buttons,
            "columns": columns,
        })

    rows = [row for row in rows if row["columns"]]
    column_indices = sorted({det.get("column_index") for det in layout_detections})
    for col_idx in column_indices:
        column_rows = []
        for row in rows:
            row_dets = [
                det for det in row["detections"]
                if det.get("column_index") == col_idx
            ]
            if not row_dets:
                continue
            column_rows.append({
                "index": row["index"],
                "center": row["center"],
                "detection": max(row_dets, key=lambda det: det.get("score", 0.0)),
            })

        if len(column_rows) < 2:
            continue
        if len(column_rows) >= 3 and len(column_rows) % 2 == 1:
            column_rows = column_rows[1:]
        if len(column_rows) % 2 == 1:
            column_rows = column_rows[:-1]

        for pair_idx, row_idx in enumerate(range(0, len(column_rows) - 1, 2), start=1):
            upper = column_rows[row_idx]["detection"]
            lower = column_rows[row_idx + 1]["detection"]
            upper["vertical_role"] = "up"
            upper["vertical_pair"] = pair_idx
            lower["vertical_role"] = "down"
            lower["vertical_pair"] = pair_idx

    return detections


def infer_direction_request(prompt):
    text = (prompt or "").lower()
    compact_text = re.sub(r"\s+", "", text)
    up_terms = (
        "向上", "往上",
        "上行", "上樓", "上升", "上方", "上面的", "上面",
    )
    down_terms = (
        "向下", "往下",
        "下行", "下樓", "下降", "下方", "下面的", "下面",
    )
    wants_up = (
        compact_text in {"上", "up"}
        or re.search(r"\b(up|upward|go up|going up)\b", text) is not None
        or any(term in text for term in up_terms)
    )
    wants_down = (
        compact_text in {"下", "down"}
        or re.search(r"\b(down|downward|go down|going down)\b", text) is not None
        or any(term in text for term in down_terms)
    )
    if wants_up and not wants_down:
        return "up"
    if wants_down and not wants_up:
        return "down"
    return None


def infer_direction_from_generated_coords(coords, detections):
    if not coords or not detections:
        return None
    if not any(det.get("vertical_role") for det in detections):
        return None

    det_ys = sorted(det["center"][1] for det in detections)
    row_gaps = [
        det_ys[idx + 1] - det_ys[idx]
        for idx in range(len(det_ys) - 1)
        if det_ys[idx + 1] - det_ys[idx] > 0.005
    ]
    margin = max(0.025, (sum(row_gaps) / len(row_gaps)) * 1.25) if row_gaps else 0.04
    mean_y = sum(coord[1] for coord in coords) / len(coords)
    if mean_y > max(det_ys) + margin:
        return "down"
    if mean_y < min(det_ys) - margin:
        return "up"
    return None


def apply_direction_scores(detections, blue_regions):
    for det in detections:
        det["up_score"] = 0.0
        det["down_score"] = 0.0
        if det.get("crop_role") == "other":
            det["up_score"] -= 2.0
            det["down_score"] -= 2.0
            continue

        if det.get("vertical_role") == "up":
            det["up_score"] += 1.0
        elif det.get("vertical_role") == "down":
            det["down_score"] += 1.0

        crop_role = det.get("crop_role")
        crop_confidence = det.get("crop_confidence", 0.0)
        if crop_role == "up":
            det["up_score"] += min(0.45, crop_confidence)
        elif crop_role == "down":
            det["down_score"] += min(0.45, crop_confidence)

    for region in blue_regions:
        bx, by = region.get("deskew_center", region["center"])
        x1, _, x2, _ = region["box"]
        column_centers = sorted({
            det.get("deskew_center", det["center"])[0]
            for det in detections
            if det.get("crop_role") != "other"
        })
        column_gaps = [
            column_centers[idx + 1] - column_centers[idx]
            for idx in range(len(column_centers) - 1)
            if column_centers[idx + 1] - column_centers[idx] > 0.006
        ]
        nearest_gap = min(column_gaps) if column_gaps else 0.06
        x_window = max(0.025, min(0.055, nearest_gap * 0.58, (x2 - x1) * 1.2))
        nearby = [
            det for det in detections
            if det.get("crop_role") != "other"
            and abs(det.get("deskew_center", det["center"])[0] - bx) <= x_window
        ]
        above = sorted(
            [det for det in nearby if det.get("deskew_center", det["center"])[1] < by],
            key=lambda det: by - det.get("deskew_center", det["center"])[1],
        )[:2]
        below = sorted(
            [det for det in nearby if det.get("deskew_center", det["center"])[1] > by],
            key=lambda det: det.get("deskew_center", det["center"])[1] - by,
        )[:2]
        for pair in (above, below):
            if len(pair) < 2:
                continue
            pair.sort(key=lambda det: det.get("deskew_center", det["center"])[1])
            upper, lower = pair
            upper["up_score"] += 0.85
            lower["down_score"] += 0.85
            upper["blue_relation"] = "blue_pair_up"
            lower["blue_relation"] = "blue_pair_down"

    columns = sorted({det.get("column_index") for det in detections if det.get("column_index") is not None})
    for col_idx in columns:
        col_dets = [
            det for det in detections
            if det.get("column_index") == col_idx and det.get("crop_role") != "other"
        ]
        if not col_dets:
            continue
        bottom = max(col_dets, key=lambda det: det.get("deskew_center", det["center"])[1])
        bottom["down_score"] += 1.25
        bottom["bottom_column_bonus"] = True


def direction_score(det, direction):
    if direction == "down":
        up_penalty = 0.15 if det.get("bottom_column_bonus") else 0.35
        return det.get("down_score", 0.0) - max(0.0, det.get("up_score", 0.0) * up_penalty)
    if direction == "up":
        return det.get("up_score", 0.0) - max(0.0, det.get("down_score", 0.0) * 0.35)
    return 0.0


def select_directional_candidates(detections, direction, max_points):
    scored = [
        det for det in detections
        if is_direction_candidate(det, direction)
    ]
    if scored:
        if direction == "down":
            scored.sort(
                key=lambda det: (
                    direction_score(det, direction),
                    det.get("deskew_center", det["center"])[1],
                ),
                reverse=True,
            )
        else:
            scored.sort(
                key=lambda det: (
                    direction_score(det, direction),
                    -det.get("deskew_center", det["center"])[1],
                ),
                reverse=True,
            )
        if max_points > 0:
            scored = scored[:max_points]
        return [det["label"] for det in scored]

    selected = [
        det for det in detections
        if det.get("vertical_role") == direction
        and not rejects_direction(det, direction)
    ]
    if direction == "down":
        pairs = sorted({det.get("vertical_pair") for det in selected if det.get("vertical_pair") is not None})
        selected.sort(key=lambda det: (-det.get("vertical_pair", -999), det.get("column_index", 999)))
    else:
        selected.sort(key=lambda det: (det.get("vertical_pair", 999), det.get("column_index", 999)))
    if max_points > 0:
        selected = selected[:max_points]
    return [det["label"] for det in selected]


def rejects_direction(det, direction):
    if (
        direction == "down"
        and det.get("vertical_role") == "up"
        and det.get("crop_role") == "up"
        and not det.get("bottom_column_bonus")
        and det.get("blue_relation") != "blue_pair_down"
    ):
        return True
    if (
        direction == "up"
        and det.get("vertical_role") == "down"
        and det.get("crop_role") == "down"
        and not det.get("bottom_column_bonus")
        and det.get("blue_relation") != "blue_pair_up"
    ):
        return True
    if (
        direction == "down"
        and det.get("blue_relation") == "blue_pair_up"
        and (det.get("vertical_role") == "up" or det.get("crop_role") == "up")
    ):
        return True
    if (
        direction == "up"
        and det.get("blue_relation") == "blue_pair_down"
        and (det.get("vertical_role") == "down" or det.get("crop_role") == "down")
    ):
        return True
    crop_role = det.get("crop_role")
    crop_confidence = det.get("crop_confidence", 0.0)
    if crop_role == "other":
        return True
    opposite = "up" if direction == "down" else "down"
    if det.get("vertical_role") == direction:
        return False
    return crop_role == opposite and crop_confidence >= 0.25


def fill_directional_labels(detections, selected_labels, direction, max_points):
    if not direction or max_points <= 0 or len(selected_labels) >= max_points:
        return selected_labels

    candidates = []
    for det in detections:
        if det["label"] in selected_labels or not is_direction_candidate(det, direction):
            continue
        candidates.append(det)

    if direction == "down":
        candidates.sort(
            key=lambda det: (
                direction_score(det, direction),
                det.get("deskew_center", det["center"])[1],
            ),
            reverse=True,
        )
    else:
        candidates.sort(
            key=lambda det: (
                direction_score(det, direction),
                -det.get("deskew_center", det["center"])[1],
            ),
            reverse=True,
        )

    for det in candidates:
        selected_labels.append(det["label"])
        if len(selected_labels) >= max_points:
            break
    return selected_labels


def is_direction_candidate(det, direction, min_score=0.75):
    if det.get("vertical_role") == direction:
        min_score = min(min_score, 0.60)
    elif det.get("vertical_role") is None and not det.get("bottom_column_bonus"):
        min_score = max(min_score, 1.05)
    return (
        direction is not None
        and not rejects_direction(det, direction)
        and direction_score(det, direction) >= min_score
    )


def fits_direction_band(det, direction, cutoff):
    cy = det["center"][1]
    if direction == "down":
        return cy >= cutoff
    if direction == "up":
        return cy <= cutoff
    return True


def order_labels_for_direction(labels, label_to_coord, direction):
    if direction == "down":
        return sorted(labels, key=lambda label: label_to_coord[label][1], reverse=True)
    if direction == "up":
        return sorted(labels, key=lambda label: label_to_coord[label][1])
    return labels


def merge_directional_and_snapped_labels(detections, directional_labels, snapped_labels, direction, max_points):
    if not direction:
        return directional_labels

    by_label = {det["label"]: det for det in detections}
    if snapped_labels and isinstance(snapped_labels[0], tuple):
        snapped_distances = dict(snapped_labels)
        snapped_order = [label for label, _ in snapped_labels]
    else:
        snapped_distances = {label: 0.0 for label in snapped_labels}
        snapped_order = list(snapped_labels)
    snapped_set = set(snapped_order)
    candidate_labels = []
    for label in list(directional_labels) + snapped_order:
        det = by_label.get(label)
        if det is None or rejects_direction(det, direction):
            continue
        if label not in candidate_labels:
            candidate_labels.append(label)

    def rank(label):
        det = by_label[label]
        raw_distance = snapped_distances.get(label)
        raw_bonus = max(0.0, 1.55 - raw_distance * 8.0) if raw_distance is not None else 0.0
        layout_bonus = 0.35 if det.get("vertical_role") == direction else 0.0
        bottom_bonus = 0.25 if det.get("bottom_column_bonus") else 0.0
        y_value = det.get("deskew_center", det["center"])[1]
        y_rank = y_value if direction == "down" else -y_value
        return (
            direction_score(det, direction) + raw_bonus + layout_bonus + bottom_bonus,
            raw_bonus,
            y_rank,
        )

    candidate_labels.sort(key=rank, reverse=True)
    if max_points > 0:
        candidate_labels = candidate_labels[:max_points]
    return candidate_labels


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

    vis = img.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)
    width, height = vis.size
    center_radius = max(4, int(min(width, height) * 0.012))
    line_width = max(2, int(min(width, height) * 0.004))

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        score = det["score"]
        cx_norm, cy_norm = det["center"]
        cx = int(round(cx_norm * width))
        cy = int(round(cy_norm * height))

        draw.rectangle([x1, y1, x2, y2], outline="lime", width=line_width)
        draw.line(
            [cx - center_radius, cy, cx + center_radius, cy],
            fill="yellow",
            width=line_width,
        )
        draw.line(
            [cx, cy - center_radius, cx, cy + center_radius],
            fill="yellow",
            width=line_width,
        )
        draw.ellipse(
            [
                cx - center_radius,
                cy - center_radius,
                cx + center_radius,
                cy + center_radius,
            ],
            outline="yellow",
            width=line_width,
        )
        draw.text((x1 + 3, max(0, y1 - 14)), f"{label} {score:.2f}", fill="lime")

    vis.save(out_path)
    print(f"Saved SAM3 detections: {out_path}")


def append_to_last_user_turn(prompt: str, text: str) -> str:
    for marker in (" ASSISTANT:", "\nASSISTANT:", "Assistant:"):
        idx = prompt.rfind(marker)
        if idx != -1:
            return prompt[:idx] + text + prompt[idx:]
    return prompt + text


def detect_sam3_candidates(img, args):
    if not args.sam3_detect_buttons:
        return []

    from transformers import Sam3Model, Sam3Processor

    tower_name = getattr(args, "sam3_vision_tower", None) or getattr(args, "mm_sam3_vision_tower", None) or "facebook/sam3"

    device = torch.device(args.sam3_detect_device)
    dtype = resolve_dtype(args.sam3_detect_dtype, device)
    print(
        "SAM3 button detector: "
        f"text={args.sam3_detect_text!r}, "
        f"threshold={args.sam3_detect_threshold}, "
        f"device={device}, "
        f"dtype={dtype}"
    )

    processor = Sam3Processor.from_pretrained(tower_name)
    model = Sam3Model.from_pretrained(tower_name)
    model.requires_grad_(False)
    model.eval()
    model.to(device=device, dtype=dtype)

    inputs = processor(images=img, text=args.sam3_detect_text, return_tensors="pt")
    inputs = {
        key: value.to(device=device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        threshold=args.sam3_detect_threshold,
        target_sizes=[img.size[::-1]],
    )[0]

    boxes = results["boxes"].detach().float().cpu()
    scores = results["scores"].detach().float().cpu()
    order = torch.argsort(scores, descending=True)

    detections = []
    labels = list(string.ascii_uppercase)
    width, height = img.size
    for rank, idx in enumerate(order[:args.sam3_max_candidates]):
        x1, y1, x2, y2 = boxes[idx].tolist()
        score = float(scores[idx].item())
        detections.append({
            "label": labels[rank] if rank < len(labels) else f"P{rank + 1}",
            "score": score,
            "box": (x1, y1, x2, y2),
            "center": (((x1 + x2) / 2.0) / width, ((y1 + y2) / 2.0) / height),
        })

    for det in detections:
        crop_role, crop_confidence = classify_candidate_crop(img, det)
        det["crop_role"] = crop_role
        det["crop_confidence"] = crop_confidence

    blue_regions = find_blue_regions(img)
    skew_slope = apply_deskew(detections, blue_regions)
    annotate_button_layout(detections)
    apply_direction_scores(detections, blue_regions)

    print(
        f"SAM3 detected {len(detections)} candidate button(s). "
        f"Blue regions: {len(blue_regions)}. Deskew slope: {skew_slope:.3f}."
    )
    for det in detections:
        cx, cy = det["center"]
        x1, y1, x2, y2 = det["box"]
        layout = ""
        if "vertical_role" in det:
            layout = (
                f", layout={det.get('column')} column, "
                f"pair={det.get('vertical_pair')}, role={det.get('vertical_role')}"
            )
        crop_text = (
            f", crop_role={det.get('crop_role')}"
            f"({det.get('crop_confidence', 0.0):.2f})"
        )
        dcx, dcy = det.get("deskew_center", det["center"])
        deskew_text = f", deskew=({dcx:.3f}, {dcy:.3f})"
        score_text = (
            f", dir_score=(up={det.get('up_score', 0.0):.2f},"
            f" down={det.get('down_score', 0.0):.2f})"
        )
        print(
            f"  {det['label']}: score={det['score']:.3f}, "
            f"center=({cx:.3f}, {cy:.3f}), "
            f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            f"{layout}{crop_text}{deskew_text}{score_text}"
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
        images = params.get("images", None)
        num_image_tokens = 0
        sam3_candidates = []
        if images is not None and len(images) > 0:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                pil_images = [load_image_from_base64(image) for image in images]
                image_sizes = [image.size for image in pil_images]
                if pil_images:
                    sam3_candidates = detect_sam3_candidates(pil_images[0], args)
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

                if sam3_candidates:
                    candidate_lines = []
                    for det in sam3_candidates:
                        cx, cy = det["center"]
                        layout_parts = []
                        if det.get("column"):
                            layout_parts.append(f"column={det['column']}")
                        if det.get("vertical_pair") is not None:
                            layout_parts.append(f"pair={det['vertical_pair']}")
                        if det.get("vertical_role"):
                            layout_parts.append(f"role={det['vertical_role']}")
                        if det.get("crop_role"):
                            layout_parts.append(f"crop_role={det['crop_role']}")
                        layout_parts.append(f"up_score={det.get('up_score', 0.0):.2f}")
                        layout_parts.append(f"down_score={det.get('down_score', 0.0):.2f}")
                        layout_text = ", " + ", ".join(layout_parts) if layout_parts else ""
                        candidate_lines.append(
                            f"{det['label']}: center=({cx:.3f}, {cy:.3f}), score={det['score']:.3f}{layout_text}"
                        )
                    if args.sam3_max_points == 1:
                        format_suffix = "\nAnswer with exactly one candidate label only, e.g. A."
                    else:
                        format_suffix = (
                            f"\nAnswer with at most {args.sam3_max_points} candidate labels only, "
                            "e.g. A, C."
                        )
                    prompt = append_to_last_user_turn(prompt, (
                        "\nSAM3 found these candidate buttons:\n"
                        + "\n".join(candidate_lines)
                        + "\nYou must choose only from the SAM3 candidate labels above. "
                        "Use the role field when the request asks for up/down buttons. "
                        "Do not invent a new coordinate."
                        + format_suffix
                    ))
                elif args.sam3_max_points == 1:
                    prompt = append_to_last_user_turn(prompt, (
                        "\nChoose the single best point only. "
                        "Your answer must be formatted as exactly one tuple, i.e. (x, y), "
                        "where x and y are normalized coordinates between 0 and 1."
                    ))
                else:
                    prompt = append_to_last_user_turn(prompt, (
                        f"\nChoose at most {args.sam3_max_points} points. "
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
            image_args = {"images": images, "image_sizes": image_sizes}
        else:
            images = None
            image_args = {}

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
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

        if sam3_candidates:
            label_to_coord = {det["label"]: det["center"] for det in sam3_candidates}
            selected_labels = parse_candidate_labels(completion_text, label_to_coord.keys())
            generated_coords = parse_coords(completion_text)
            direction = infer_direction_request(ori_prompt)
            if direction is None:
                direction = infer_direction_from_generated_coords(generated_coords, sam3_candidates)
                if direction:
                    print(
                        "Inferred direction from generated coordinates outside SAM3 layout: "
                        f"{direction!r}"
                    )
            directional_labels = select_directional_candidates(
                sam3_candidates,
                direction,
                args.sam3_max_points,
            ) if direction else []
            if directional_labels:
                snapped_labels = snap_coords_to_candidates_with_limit(
                    generated_coords,
                    sam3_candidates,
                    args.sam3_max_points,
                )
                snapped_label_distances = snap_coords_to_candidates_with_distances(
                    generated_coords,
                    sam3_candidates,
                    args.sam3_max_points,
                )
                snapped_labels = order_labels_for_direction(
                    snapped_labels,
                    label_to_coord,
                    direction,
                )
                selected_labels = merge_directional_and_snapped_labels(
                    sam3_candidates,
                    directional_labels,
                    snapped_label_distances or snapped_labels,
                    direction,
                    args.sam3_max_points,
                )
                selected_labels = fill_directional_labels(
                    sam3_candidates,
                    selected_labels,
                    direction,
                    args.sam3_max_points,
                )
                print(
                    f"Direction request {direction!r}; selected SAM3 labels by layout: "
                    f"{selected_labels}"
                )
            elif not selected_labels:
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
            if args.sam3_max_points > 0:
                selected_labels = selected_labels[:args.sam3_max_points]
            coords = [label_to_coord[label] for label in selected_labels]
            if not generated_coords and selected_labels:
                response_text = ", ".join(
                    f"({round(nx, 4)}, {round(ny, 4)})"
                    for nx, ny in coords
                )
            elif generated_coords and selected_labels:
                snapped_points = [
                    (round(nx, 4), round(ny, 4))
                    for nx, ny in coords
                ]
                response_text = (
                    f"{completion_text}\n"
                    f"__SNAPPED_POINTS__:{json.dumps(snapped_points)}"
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
                "text": server_error_msg,
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
        default="http://10.0.0.1:22000")
    parser.add_argument("--controller-address", type=str,
        default="http://10.0.0.1:11000")
    parser.add_argument("--model-path", type=str, default="PME033541/vla13")
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
    parser.add_argument("--mm-sam3-vision-tower", type=str, default="facebook/sam3")
    parser.add_argument("--mm-sam3-blend-alpha", type=float, default=0.35)
    parser.add_argument("--mm-sam3-mask-gamma", type=float, default=1.0)
    parser.add_argument("--mm-sam3-device", type=str, default="cuda")
    parser.add_argument("--mm-sam3-dtype", type=str, default="bfloat16")
    parser.add_argument("--mm-sam3-unload-after-forward", type=str2bool, default=False)
    parser.add_argument("--sam3-detect-buttons", type=str2bool, default=True)
    parser.add_argument("--sam3-detect-text", type=str, default="button")
    parser.add_argument("--sam3-detect-threshold", type=float, default=0.25)
    parser.add_argument("--sam3-max-candidates", type=int, default=12)
    parser.add_argument("--sam3-max-points", type=int, default=4)
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
