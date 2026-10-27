#!/usr/bin/env python3
"""Calibrate split-conformal count sets without fitting a predictive model."""

import argparse
import json
import math
from pathlib import Path


def load_records(path):
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list):
        raise ValueError("Input must be a JSON array of counting records.")
    return records


def calibration_examples(records):
    examples = []
    for record in records:
        if int(record.get("stage1_sam3", {}).get("candidate_count", 0)) <= 0:
            continue
        gt_count = record.get("gt_count")
        if gt_count is None:
            continue
        stage3 = record.get("stage3_conformal_set", {})
        support_scores = stage3.get("count_support_scores", {})
        if not support_scores:
            continue
        anchor_count = int(record.get("stage1_sam3", {}).get("candidate_count", 0))
        gt_conformity = (
            1.0
            if int(gt_count) == anchor_count
            else float(support_scores.get(str(int(gt_count)), 0.0))
        )
        examples.append({
            "image": record.get("image"),
            "gt_count": int(gt_count),
            "anchor_count": anchor_count,
            "gt_conformity": gt_conformity,
            "nonconformity": 1.0 - gt_conformity,
            "support_scores": support_scores,
            "label_space": stage3.get("label_space", list(range(21))),
        })
    return examples


def finite_sample_qhat(scores, alpha):
    if not scores:
        raise ValueError("No eligible calibration examples were found.")
    rank = min(len(scores), math.ceil((len(scores) + 1) * (1.0 - alpha)))
    return sorted(scores)[rank - 1], rank


def prediction_set(example, qhat):
    values = {
        int(example["anchor_count"]),
        *(
        int(count)
        for count in example["label_space"]
        if 1.0 - float(example["support_scores"].get(str(count), 0.0)) <= qhat + 1e-12
        ),
    }
    return sorted(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--scored-output", default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")

    records = load_records(args.input)
    examples = calibration_examples(records)
    qhat, rank = finite_sample_qhat(
        [example["nonconformity"] for example in examples],
        args.alpha,
    )
    sets = [prediction_set(example, qhat) for example in examples]
    covered = [example["gt_count"] in values for example, values in zip(examples, sets)]
    wrong_singletons = [
        len(values) == 1 and values[0] != example["gt_count"]
        for example, values in zip(examples, sets)
    ]
    intervention = [len(values) != 1 for values in sets]

    calibration = {
        "method": "split_conformal",
        "nonconformity_score": "one_minus_count_support",
        "alpha": args.alpha,
        "qhat": round(float(qhat), 12),
        "calibration_count": len(examples),
        "finite_sample_rank": rank,
        "decision_rule": "accept only when prediction_set has exactly one count",
        "no_detection_rule": "candidate_count == 0 requires human intervention",
        "trained_model": False,
    }
    Path(args.output).write_text(json.dumps(calibration, indent=2) + "\n")

    report = {
        **calibration,
        "empirical_calibration_coverage": sum(covered) / len(covered),
        "intervention_rate": sum(intervention) / len(intervention),
        "wrong_singleton_rate": sum(wrong_singletons) / len(wrong_singletons),
        "mean_prediction_set_size": sum(map(len, sets)) / len(sets),
        "records": [
            {
                "image": example["image"],
                "gt_count": example["gt_count"],
                "anchor_count": example["anchor_count"],
                "gt_conformity": round(example["gt_conformity"], 6),
                "nonconformity": round(example["nonconformity"], 6),
                "prediction_set": values,
                "covered": is_covered,
                "wrong_singleton": wrong,
            }
            for example, values, is_covered, wrong in zip(
                examples, sets, covered, wrong_singletons
            )
        ],
    }
    report_path = Path(args.report) if args.report else Path(args.output).with_name(
        Path(args.output).stem + "_report.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.scored_output:
        for record in records:
            stage3 = record.get("stage3_conformal_set")
            if not isinstance(stage3, dict):
                continue
            if int(record.get("stage1_sam3", {}).get("candidate_count", 0)) <= 0:
                values = stage3.get("label_space", list(range(21)))
                intervention_required = True
                stage3.update({
                    "prediction_set": values,
                    "qhat": None,
                    "human_intervention": True,
                    "status": "hard_rule_no_detection",
                })
            else:
                example = {
                    "anchor_count": int(
                        record.get("stage1_sam3", {}).get("candidate_count", 0)
                    ),
                    "label_space": stage3.get("label_space", list(range(21))),
                    "support_scores": stage3.get("count_support_scores", {}),
                }
                values = prediction_set(example, qhat)
                intervention_required = len(values) != 1
                stage3.update({
                    "prediction_set": values,
                    "qhat": round(float(qhat), 12),
                    "alpha": args.alpha,
                    "human_intervention": intervention_required,
                    "status": "calibrated",
                })
            decision = record.setdefault("decision", {})
            gt_count = record.get("gt_count")
            decision.update({
                "human_intervention": intervention_required,
                "accepted_automatically": not intervention_required,
                "accepted_count": values[0] if len(values) == 1 else None,
                "ground_truth_in_set": None if gt_count is None else gt_count in values,
                "wrong_singleton": (
                    None
                    if gt_count is None
                    else len(values) == 1 and values[0] != gt_count
                ),
            })
        Path(args.scored_output).write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
