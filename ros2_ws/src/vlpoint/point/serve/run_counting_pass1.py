#!/usr/bin/env python3
"""Run counting pass 1 over a calibration set and emit MCQA-ready results."""

import argparse
import base64
import csv
import json
from pathlib import Path

import requests


CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def encode_image(image_path):
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def parse_streaming_response(response, prompt):
    response.raise_for_status()
    final_text = ""
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if not chunk:
            continue
        data = json.loads(chunk.decode("utf-8"))
        if data.get("error_code", 0) != 0:
            raise RuntimeError(data.get("text", "worker returned an error"))
        final_text = data.get("text", "")
    if final_text.startswith(prompt):
        final_text = final_text[len(prompt):]
    return json.loads(final_text.strip())


def get_worker_address(controller_address, model):
    response = requests.post(
        controller_address.rstrip("/") + "/get_worker_address",
        json={"model": model},
        timeout=30,
    )
    response.raise_for_status()
    worker_address = response.json().get("address", "")
    if not worker_address:
        raise RuntimeError(f"No worker available for model {model!r}")
    return worker_address


def call_counting_pass1(
    worker_address,
    model,
    image_path,
    target_object,
    timeout,
    counting_method,
    coverage_repeats=1,
):
    prompt = f"USER: <image>\nHow many visible {target_object}s are in this image?\nASSISTANT:"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 256,
        "images": [encode_image(image_path)],
        "stop": "</s>",
        "counting_pass": "pass1",
        "counting_method": counting_method,
        "target_object": target_object,
        "coverage_repeats": coverage_repeats,
    }
    response = requests.post(
        worker_address.rstrip("/") + "/worker_generate_stream",
        json=payload,
        stream=True,
        timeout=timeout,
    )
    return parse_streaming_response(response, prompt)


def ensure_nonempty_sam3_masks(sam3_masks, target_object):
    if sam3_masks:
        return sam3_masks
    return [{
        "label": None,
        "confidence": None,
        "center": None,
        "status": "no_detection",
        "reason": f"SAM3 did not return any {target_object} candidate mask/region.",
    }]


def format_sam3_mask(mask):
    formatted = {
        "label": mask.get("label"),
        "confidence": mask.get("confidence"),
        "center": mask.get("center"),
    }
    if "bbox" in mask:
        formatted["bbox"] = mask["bbox"]
    if "reason" in mask:
        formatted["reason"] = mask["reason"]
    return formatted


def format_stage1(pass1, target_object):
    stage1 = pass1.get("stage1_sam3", {})
    sam3_masks = ensure_nonempty_sam3_masks(
        stage1.get("sam3_masks", pass1.get("sam3_masks", [])),
        target_object,
    )
    formatted = {
        "candidate_count": stage1.get("candidate_count", len(pass1.get("sam3_masks", []))),
        "sam3_masks": [format_sam3_mask(mask) for mask in sam3_masks],
    }
    if "statistics" in stage1:
        formatted["statistics"] = stage1.get("statistics")
    return formatted


def format_hypothesis(hypothesis):
    formatted = {
        "option": hypothesis.get("option"),
        "count": hypothesis.get("count"),
        "accepted_masks": hypothesis.get("accepted_mask_ids", hypothesis.get("valid_masks", [])),
        "rejected_masks": hypothesis.get("rejected_mask_ids", hypothesis.get("rejected_masks", [])),
        "missing_objects": hypothesis.get(
            "additional_unmasked_count",
            hypothesis.get("missing_objects", 0),
        ),
        "reason": hypothesis.get("reason", hypothesis.get("explanation", "")),
    }
    if hypothesis.get("evidence_type"):
        formatted["evidence_type"] = hypothesis.get("evidence_type")
    if hypothesis.get("correction_type"):
        formatted["correction_type"] = hypothesis.get("correction_type")
    if hypothesis.get("strength"):
        formatted["strength"] = hypothesis.get("strength")
    if "localized" in hypothesis:
        formatted["localized"] = hypothesis.get("localized")
    if hypothesis.get("mask_pairs"):
        formatted["mask_pairs"] = hypothesis.get("mask_pairs")
    if hypothesis.get("merged_mask_ids"):
        formatted["merged_mask_ids"] = hypothesis.get("merged_mask_ids")
    if hypothesis.get("possible_missing_regions"):
        formatted["possible_missing_regions"] = hypothesis.get("possible_missing_regions")
    if hypothesis.get("supporting_observation"):
        formatted["supporting_observation"] = hypothesis.get("supporting_observation")
    if hypothesis.get("limitation"):
        formatted["limitation"] = hypothesis.get("limitation")
    return formatted


def format_stage2(pass1):
    stage2 = pass1.get("stage2_hypothesis_generation", {})
    hypotheses = stage2.get("count_hypotheses", stage2.get("hypotheses", []))
    formatted = {
        "hypotheses": [format_hypothesis(hypothesis) for hypothesis in hypotheses],
    }
    for key in (
        "reduce_evidence",
        "reduce_candidates",
        "none_evidence",
        "expand_evidence",
        "expand_candidates",
        "independent_count_evidence",
        "raw_independent_count_vlm_output",
        "raw_reduce_vlm_output",
        "raw_expand_vlm_output",
        "generation_status",
    ):
        if key in stage2:
            formatted[key] = stage2.get(key)
    return formatted


def format_stage3(pass1, gt_count):
    stage3 = pass1.get("stage3_mcqa_scoring", {})
    predicted_count = stage3.get("selected_count", stage3.get("predicted_count", pass1.get("estimated_count")))
    return {
        "mcqa_prompt": stage3.get("mcqa_prompt", ""),
        "option_logprobs": stage3.get("option_logprobs", {}),
        "option_probabilities": stage3.get("option_probabilities", {}),
        "selected_option": stage3.get("selected_option"),
        "predicted_count": predicted_count,
        "correct": None if gt_count is None else predicted_count == gt_count,
    }


def build_mcqa_record(image_name, gt_count, target_object, pass1, counting_method):
    record = {
        "image": image_name,
        "target_object": target_object,
        "question": f"How many visible {target_object}s are in the image?",
        "gt_count": gt_count,
        "stage1_sam3": format_stage1(pass1, target_object),
        "stage2_hypothesis_generation": format_stage2(pass1),
        "stage3_mcqa_scoring": format_stage3(pass1, gt_count),
    }
    return record


def iter_calibration_rows(calibration_csv):
    with Path(calibration_csv).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row.get("images") or row.get("image") or row.get("filename")
            if not image_name:
                continue
            gt_raw = row.get("gt_count")
            gt_count = int(gt_raw) if gt_raw not in (None, "") else None
            yield image_name, gt_count


def load_conformal_calibration(path):
    if not path:
        return None
    calibration = json.loads(Path(path).read_text())
    required = {"qhat", "alpha", "nonconformity_score"}
    missing = required.difference(calibration)
    if missing:
        raise ValueError(f"Conformal calibration is missing fields: {sorted(missing)}")
    if calibration["nonconformity_score"] != "one_minus_count_support":
        raise ValueError("Unsupported conformal nonconformity score.")
    return calibration


def apply_conformal_calibration(record, calibration):
    if calibration is None or "stage3_conformal_set" not in record:
        return record
    if record.get("decision", {}).get("sam3_count", 0) == 0:
        label_space = record["stage3_conformal_set"].get("label_space", list(range(21)))
        record["stage3_conformal_set"].update({
            "prediction_set": label_space,
            "qhat": None,
            "human_intervention": True,
            "status": "hard_rule_no_detection",
        })
        record["decision"].update({
            "human_intervention": True,
            "accepted_automatically": False,
            "accepted_count": None,
            "ground_truth_in_set": (
                None
                if record.get("gt_count") is None
                else record.get("gt_count") in label_space
            ),
            "wrong_singleton": False,
        })
        return record
    support_scores = record["stage3_conformal_set"].get("count_support_scores", {})
    label_space = record["stage3_conformal_set"].get("label_space", range(21))
    anchor_count = int(record["decision"]["sam3_count"])
    qhat = float(calibration["qhat"])
    prediction_set = {
        anchor_count,
        *(
        int(count)
        for count in label_space
        if 1.0 - float(support_scores.get(str(count), 0.0)) <= qhat + 1e-12
        ),
    }
    prediction_set = sorted(prediction_set)
    intervention = len(prediction_set) != 1
    record["stage3_conformal_set"].update({
        "prediction_set": prediction_set,
        "qhat": round(qhat, 6),
        "alpha": float(calibration["alpha"]),
        "human_intervention": intervention,
        "status": "calibrated",
    })
    gt_count = record.get("gt_count")
    accepted_count = prediction_set[0] if len(prediction_set) == 1 else None
    record["decision"].update({
        "human_intervention": intervention,
        "accepted_automatically": not intervention,
        "accepted_count": accepted_count,
        "ground_truth_in_set": None if gt_count is None else gt_count in prediction_set,
        "wrong_singleton": (
            None
            if gt_count is None
            else len(prediction_set) == 1 and prediction_set[0] != gt_count
        ),
    })
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", default="calibration_set")
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--calibration-csv", default=None)
    parser.add_argument("--output", default="calibration_set/pass1_results.jsonl")
    parser.add_argument("--output-json", default="calibration_set/pass1_results.json")
    parser.add_argument("--worker-address", default="http://127.0.0.1:22000")
    parser.add_argument("--controller-address", default=None)
    parser.add_argument("--model", default="llava-v1.5-13b")
    parser.add_argument("--target-object", default="fork")
    parser.add_argument(
        "--counting-method",
        choices=(
            "sam3_calibrated",
            "sam3_raw",
            "sam3_prompt_experiment",
        ),
        default="sam3_calibrated",
        help="Counting pipeline used by the worker; calibrated SAM3 is the default.",
    )
    parser.add_argument(
        "--coverage-repeats",
        type=int,
        default=1,
        help="Deprecated compatibility option.",
    )
    parser.add_argument(
        "--conformal-calibration",
        default=None,
        help="Conformal qhat JSON used to create count prediction sets.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    calibration_dir = Path(args.calibration_dir)
    images_dir = Path(args.images_dir) if args.images_dir else calibration_dir / "images"
    calibration_csv = Path(args.calibration_csv) if args.calibration_csv else calibration_dir / "calibration.csv"
    worker_address = args.worker_address
    if args.controller_address:
        worker_address = get_worker_address(args.controller_address, args.model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    conformal_calibration = load_conformal_calibration(args.conformal_calibration)

    methods = [args.counting_method]

    with output_path.open("w") as out:
        for image_name, gt_count in iter_calibration_rows(calibration_csv):
            image_path = images_dir / image_name
            for method in methods:
                pass1 = call_counting_pass1(
                    worker_address=worker_address,
                    model=args.model,
                    image_path=image_path,
                    target_object=args.target_object,
                    timeout=args.timeout,
                    counting_method=method,
                    coverage_repeats=args.coverage_repeats,
                )
                record = build_mcqa_record(image_name, gt_count, args.target_object, pass1, method)
                record = apply_conformal_calibration(record, conformal_calibration)
                records.append(record)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(json.dumps(record, ensure_ascii=False))

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
