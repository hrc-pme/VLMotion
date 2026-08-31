"""Visualize SAM3 candidates as enlarged crops for CLIP/LLM reasoning."""

import math
from PIL import Image, ImageDraw, ImageFont


def deduplicate_candidates(detections, overlap_threshold=0.8):
    """Remove nested SAM3 proposals while preserving distinct nearby objects."""
    kept = []
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        duplicate = False
        for previous in kept:
            px1, py1, px2, py2 = previous["box"]
            previous_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
            intersection = (
                max(0.0, min(x2, px2) - max(x1, px1))
                * max(0.0, min(y2, py2) - max(y1, py1))
            )
            smaller_area = min(area, previous_area)
            if smaller_area > 0 and intersection / smaller_area >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def build_candidate_contact_sheet(image, detections, columns=None, tile_size=224):
    # A fixed four-column one-row sheet gets padded to a square before CLIP,
    # shrinking each crop to roughly 84 px. A square-ish layout preserves
    # substantially more of the arrow/icon detail.
    if columns is None:
        columns = max(1, math.ceil(math.sqrt(len(detections))))
    rows = max(1, math.ceil(len(detections) / columns))
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    font = ImageFont.load_default(size=max(18, tile_size // 9))
    width, height = image.size

    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["box"]
        box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
        padding = max(box_width, box_height) * 0.65
        crop = image.crop((
            max(0, x1 - padding), max(0, y1 - padding),
            min(width, x2 + padding), min(height, y2 + padding),
        )).convert("RGB")
        crop.thumbnail((tile_size - 12, tile_size - 12), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_size, tile_size), "white")
        tile.paste(crop, ((tile_size - crop.width) // 2, (tile_size - crop.height) // 2))
        draw = ImageDraw.Draw(tile)
        label = detection["label"]
        text_box = draw.textbbox((0, 0), label, font=font)
        label_width = text_box[2] - text_box[0] + 14
        label_height = text_box[3] - text_box[1] + 10
        draw.rectangle((0, 0, label_width, label_height), fill="black")
        draw.text((7, 4), label, fill="yellow", font=font)
        draw.rectangle((1, 1, tile_size - 2, tile_size - 2), outline="gray", width=2)
        column, row = index % columns, index // columns
        sheet.paste(tile, (column * tile_size, row * tile_size))
    return sheet


def candidate_prompt_lines(detections):
    lines = []
    for detection in detections:
        cx, cy = detection["center"]
        lines.append(
            f"{detection['label']}: original_center=({cx:.4f}, {cy:.4f})"
        )
    return lines
