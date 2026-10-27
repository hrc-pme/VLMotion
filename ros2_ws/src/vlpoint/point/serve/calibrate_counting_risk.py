#!/usr/bin/env python3
"""Calibrate image-level SAM3 count risk and a human-intervention threshold."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SAM3_FEATURES = [
    "min_confidence",
    "confidence_std",
    "max_nearest_neighbor_distance",
    "bbox_area_cv",
]

VLM_FEATURES = [
    "localized_support_score",
    "localized_uncertainty_score",
]

FEATURE_SETS = {
    "vlm_only": VLM_FEATURES,
    "proposed": SAM3_FEATURES + VLM_FEATURES,
}


def load_records(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array of coverage-risk records.")
    return data


def record_label(record):
    correct = record.get("decision", {}).get("correct")
    if correct is None:
        return None
    return 0 if correct else 1


def feature_matrix(records, feature_names):
    rows = []
    labels = []
    kept_records = []
    for record in records:
        if int(record.get("stage1_sam3", {}).get("candidate_count", 0)) <= 0:
            continue
        label = record_label(record)
        if label is None:
            continue
        features = record.get("stage3_risk_assessment", {}).get("feature_vector", {})
        rows.append([float(features.get(name, 0.0)) for name in feature_names])
        labels.append(label)
        kept_records.append(record)
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), kept_records


def make_estimator(c_value):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=0,
        ),
    )


def leave_one_out_probabilities(x_values, labels, c_value):
    probabilities = np.zeros(len(labels), dtype=float)
    for train_indices, test_indices in LeaveOneOut().split(x_values):
        estimator = make_estimator(c_value)
        estimator.fit(x_values[train_indices], labels[train_indices])
        probabilities[test_indices] = estimator.predict_proba(x_values[test_indices])[:, 1]
    return probabilities


def choose_threshold(probabilities, labels, target_error_recall):
    error_count = int(labels.sum())
    candidates = sorted(set([0.0, 1.0] + probabilities.tolist()), reverse=True)
    feasible = []
    for threshold in candidates:
        interventions = probabilities >= threshold
        caught_errors = int(labels[interventions].sum())
        recall = caught_errors / error_count if error_count else 1.0
        if recall + 1e-12 >= target_error_recall:
            feasible.append({
                "threshold": float(threshold),
                "error_recall": float(recall),
                "caught_errors": caught_errors,
                "total_errors": error_count,
                "intervention_rate": float(interventions.mean()),
                "intervention_count": int(interventions.sum()),
            })
    if not feasible:
        raise ValueError("No threshold satisfies the requested error recall.")
    return min(feasible, key=lambda item: (item["intervention_rate"], -item["threshold"]))


def bootstrap_interval(probabilities, labels, metric_fn, repetitions, seed=0):
    if repetitions <= 0:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if len(set(sampled_labels.tolist())) < 2:
            continue
        values.append(float(metric_fn(sampled_labels, probabilities[indices])))
    if not values:
        return None
    return [
        round(float(np.percentile(values, 2.5)), 6),
        round(float(np.percentile(values, 97.5)), 6),
    ]


def export_deployment_model(estimator, feature_names, threshold, c_value):
    scaler = estimator.named_steps["standardscaler"]
    classifier = estimator.named_steps["logisticregression"]
    return {
        "model_type": "standardized_logistic_regression",
        "feature_names": feature_names,
        "feature_means": [round(float(value), 12) for value in scaler.mean_],
        "feature_scales": [round(float(value), 12) for value in scaler.scale_],
        "coefficients": [round(float(value), 12) for value in classifier.coef_[0]],
        "intercept": round(float(classifier.intercept_[0]), 12),
        "threshold": round(float(threshold), 12),
        "regularization_c": c_value,
        "decision_rule": "human_intervention = risk_score >= threshold",
        "scope": "candidate_count > 0; candidate_count == 0 uses mandatory human review",
        "threshold_status": "provisional_small_calibration_set",
    }


def evaluate_feature_set(records, feature_names, c_value, target_error_recall, bootstrap_repetitions):
    x_values, labels, kept_records = feature_matrix(records, feature_names)
    if len(labels) < 3 or len(set(labels.tolist())) < 2:
        raise ValueError("Calibration requires at least three labeled records from both classes.")
    probabilities = leave_one_out_probabilities(x_values, labels, c_value)
    threshold_result = choose_threshold(probabilities, labels, target_error_recall)
    estimator = make_estimator(c_value)
    estimator.fit(x_values, labels)
    auroc = float(roc_auc_score(labels, probabilities))
    auprc = float(average_precision_score(labels, probabilities))
    return {
        "feature_names": feature_names,
        "record_count": len(labels),
        "error_count": int(labels.sum()),
        "oof_auroc": round(auroc, 6),
        "oof_auroc_95ci": bootstrap_interval(
            probabilities, labels, roc_auc_score, bootstrap_repetitions, seed=17
        ),
        "oof_auprc": round(auprc, 6),
        "oof_auprc_95ci": bootstrap_interval(
            probabilities, labels, average_precision_score, bootstrap_repetitions, seed=23
        ),
        "threshold_selection": {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in threshold_result.items()
        },
        "oof_predictions": [
            {
                "image": record.get("image"),
                "is_sam3_error": bool(label),
                "risk_score": round(float(probability), 6),
            }
            for record, label, probability in zip(kept_records, labels, probabilities)
        ],
        "deployment_model": export_deployment_model(
            estimator,
            feature_names,
            threshold_result["threshold"],
            c_value,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--target-error-recall", type=float, default=0.95)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()

    if not 0.0 < args.target_error_recall <= 1.0:
        raise ValueError("--target-error-recall must be in (0, 1].")
    if args.regularization_c <= 0:
        raise ValueError("--regularization-c must be positive.")

    records = load_records(args.input)
    evaluations = {
        name: evaluate_feature_set(
            records,
            features,
            args.regularization_c,
            args.target_error_recall,
            args.bootstrap_repetitions,
        )
        for name, features in FEATURE_SETS.items()
    }
    proposed_features = FEATURE_SETS["proposed"]
    feature_ablations = {
        f"without_{removed_feature}": evaluate_feature_set(
            records,
            [name for name in proposed_features if name != removed_feature],
            args.regularization_c,
            args.target_error_recall,
            args.bootstrap_repetitions,
        )
        for removed_feature in proposed_features
    }
    no_detection_count = sum(
        int(record.get("stage1_sam3", {}).get("candidate_count", 0)) == 0
        for record in records
    )
    report = {
        "input": str(Path(args.input).resolve()),
        "target_error_recall": args.target_error_recall,
        "regularization_c": args.regularization_c,
        "evaluation_protocol": "leave-one-out out-of-fold predictions",
        "training_scope": "candidate_count > 0 only",
        "no_detection_hard_rule_count": no_detection_count,
        "threshold_status": "provisional; only candidate_count > 0 errors are modeled",
        "models": evaluations,
        "feature_ablations": feature_ablations,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    model = evaluations["proposed"]["deployment_model"]
    model["calibration_input"] = str(Path(args.input).resolve())
    model["target_error_recall"] = args.target_error_recall
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")

    summary = {
        name: {
            "oof_auroc": result["oof_auroc"],
            "oof_auprc": result["oof_auprc"],
            **result["threshold_selection"],
        }
        for name, result in evaluations.items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
