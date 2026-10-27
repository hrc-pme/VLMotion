#!/usr/bin/env python3
"""Compare base and fine-tuned HF SAM3 outputs using worker thresholds."""

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3Model, Sam3Processor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--fine-tuned-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--prompt", default="fork")
    return parser.parse_args()


OVERLAY_COLORS = (
    (239, 68, 68),
    (16, 185, 129),
    (59, 130, 246),
    (245, 158, 11),
    (168, 85, 247),
    (6, 182, 212),
)


def save_overlay(image, result, output_path, gt_count):
    masks = result["masks"].detach().cpu().bool()
    scores = result["scores"].detach().float().cpu()
    boxes = result["boxes"].detach().float().cpu()
    order = sorted(
        range(len(masks)),
        key=lambda index: float((boxes[index, 0] + boxes[index, 2]) / 2),
    )

    pixels = np.asarray(image).astype(np.float32).copy()
    for display_index, result_index in enumerate(order):
        mask = masks[result_index].numpy()
        color = np.asarray(
            OVERLAY_COLORS[display_index % len(OVERLAY_COLORS)],
            dtype=np.float32,
        )
        pixels[mask] = pixels[mask] * 0.62 + color * 0.38

    overlay = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 18) if Path(font_path).is_file() else None
    header = f"GT {gt_count}  Pred {len(order)}"
    draw.rectangle((6, 6, 190, 35), fill=(0, 0, 0), outline=(255, 255, 255), width=2)
    draw.text((12, 9), header, fill=(255, 255, 255), font=font)

    for display_index, result_index in enumerate(order, start=1):
        color = OVERLAY_COLORS[(display_index - 1) % len(OVERLAY_COLORS)]
        x1, y1, x2, y2 = [int(round(value)) for value in boxes[result_index].tolist()]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"ID {display_index}  {float(scores[result_index]):.2f}"
        text_y = max(38, y1 - 23)
        text_box = draw.textbbox((x1, text_y), label, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((x1, text_y), label, fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path, quality=95)


def evaluate_model(model_path, model_label, processor, samples, args):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = Sam3Model.from_pretrained(
        model_path,
        local_files_only=True,
    ).eval().to(device=device, dtype=dtype)
    records = []
    for index, sample in enumerate(samples, start=1):
        image_path = sample["path"]
        print(f"[{index}/{len(samples)}] {model_path}: {image_path.name}", flush=True)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=args.prompt, return_tensors="pt")
        inputs = {
            key: value.to(device=device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
        with torch.no_grad():
            outputs = model(**inputs)

        presence_score = float(outputs.presence_logits[0].float().sigmoid().item())
        match_scores = outputs.pred_logits[0].float().sigmoid()
        standard_count = int(
            ((match_scores * presence_score) > args.score_threshold).sum().item()
        )

        saved_presence = outputs.presence_logits
        outputs.presence_logits = None
        result = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        outputs.presence_logits = saved_presence
        masks = result["masks"].detach().cpu().bool()
        if args.overlay_dir is not None:
            save_overlay(
                image,
                result,
                args.overlay_dir / model_label / f"{image_path.stem}_overlay.jpg",
                sample["gt_count"],
            )
        records.append({
            "image": image_path.name,
            "image_id": sample["image_id"],
            "gt_count": sample["gt_count"],
            "presence_score": presence_score,
            "standard_candidate_count": standard_count,
            "force_presence_one_count": int(masks.shape[0]),
            "nonempty_mask_count": int(
                (masks.flatten(1).sum(dim=1) > 0).sum().item()
            ) if masks.numel() else 0,
            "top_query_match_scores": [
                float(value) for value in match_scores.topk(5).values.cpu().tolist()
            ],
        })

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records


def summarize(records, count_key):
    errors = [int(row[count_key]) - int(row["gt_count"]) for row in records]
    return {
        "count_key": count_key,
        "images": len(records),
        "exact_count": sum(error == 0 for error in errors),
        "exact_count_accuracy": sum(error == 0 for error in errors) / len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "under_count": sum(error < 0 for error in errors),
        "over_count": sum(error > 0 for error in errors),
        "zero_detection": sum(int(row[count_key]) == 0 for row in records),
    }


def main():
    args = parse_args()
    annotation_path = (
        args.dataset_root / args.split / "_annotations.sam3.coco.json"
    )
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    gt_counts = {int(image["id"]): 0 for image in annotations["images"]}
    for annotation in annotations["annotations"]:
        gt_counts[int(annotation["image_id"])] += 1
    selected_images = list(annotations["images"])
    random.Random(args.seed).shuffle(selected_images)
    selected_images = selected_images[:args.num_images]
    samples = [
        {
            "path": args.dataset_root / args.split / image["file_name"],
            "image_id": int(image["id"]),
            "gt_count": gt_counts[int(image["id"])],
        }
        for image in selected_images
    ]
    processor = Sam3Processor.from_pretrained(
        args.fine_tuned_model,
        local_files_only=True,
    )
    if args.overlay_dir is None:
        args.overlay_dir = args.output.parent / f"{args.output.stem}_overlays"
    base_records = evaluate_model(
        "facebook/sam3", "base", processor, samples, args
    )
    fine_tuned_records = evaluate_model(
        str(args.fine_tuned_model), "fine_tuned", processor, samples, args
    )
    report = {
        "split": args.split,
        "seed": args.seed,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "prompt": args.prompt,
        "overlay_dir": str(args.overlay_dir.resolve()),
        "images": [sample["path"].name for sample in samples],
        "base": base_records,
        "fine_tuned": fine_tuned_records,
        "summary": {
            "base_standard": summarize(base_records, "standard_candidate_count"),
            "fine_tuned_standard": summarize(
                fine_tuned_records, "standard_candidate_count"
            ),
            "base_presence_one": summarize(
                base_records, "force_presence_one_count"
            ),
            "fine_tuned_presence_one": summarize(
                fine_tuned_records, "force_presence_one_count"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
