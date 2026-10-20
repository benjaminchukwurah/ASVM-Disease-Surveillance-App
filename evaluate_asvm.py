#!/usr/bin/env python3
"""Reproduce the dissertation's chronological ASVM evaluation protocol.

Examples:
    python evaluate_asvm.py --csv "Lasa Fever Data/SYNTHETIC_lassa_state_month_features.csv"
    python evaluate_asvm.py --postgres --schema synthetic
    python evaluate_asvm.py --csv real_state_month_table.csv --out results

Required columns: state, period, confirmed, lag1, lag2, roll3, trend, hotspot,
sin_m, cos_m, outbreak.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

FEATURES = ["lag1", "lag2", "roll3", "trend", "hotspot", "sin_m", "cos_m"]
REQUIRED_COLUMNS = {
    "state", "period", "confirmed", *FEATURES, "outbreak"
}


def load_data(csv_path=None, schema="synthetic"):
    if csv_path:
        frame = pd.read_csv(csv_path)
    else:
        connection_string = os.environ.get("DATABASE_URL")
        if not connection_string:
            raise ValueError("Set DATABASE_URL before using --postgres.")
        from psycopg2 import connect, sql

        query = sql.SQL("SELECT * FROM {}.{} ORDER BY period, state").format(
            sql.Identifier(schema), sql.Identifier("lassa_state_month")
        )
        with connect(connection_string, connect_timeout=8) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                columns = [description.name for description in cursor.description]
                frame = pd.DataFrame(cursor.fetchall(), columns=columns)

    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Input data contains no rows.")

    period_values = frame["period"].astype(str).str.strip()
    month_only = period_values.str.fullmatch(r"\d{4}-\d{2}")
    period_values.loc[month_only] = period_values.loc[month_only] + "-01"
    frame["period"] = pd.to_datetime(period_values, errors="raise")
    frame["outbreak"] = pd.to_numeric(frame["outbreak"], errors="raise").astype(int)
    if not frame["outbreak"].isin([0, 1]).all():
        raise ValueError("The outbreak column must contain only binary 0/1 labels.")

    frame[FEATURES] = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Feature columns contain missing or non-finite values.")
    return frame.sort_values(["period", "state"], kind="stable").reset_index(drop=True)


def calculate_metrics(labels, predictions, scores):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not len(labels):
        raise ValueError("Cannot evaluate an empty test set.")
    if not (len(labels) == len(predictions) == len(scores)):
        raise ValueError("Labels, predictions, and scores must have equal lengths.")

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float((tp + tn) / len(labels)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": auc,
    }


def run_asvm(frame, warmup_fraction=0.15, batch_size=100, alpha=1e-4, seed=0):
    features = frame[FEATURES].to_numpy(dtype=float)
    labels = frame["outbreak"].to_numpy(dtype=int)
    row_count = len(frame)
    warmup_count = int(round(warmup_fraction * row_count))
    if warmup_count < 2 or warmup_count >= row_count:
        raise ValueError("The warm-up split must contain at least two rows and leave test rows.")
    if set(np.unique(labels[:warmup_count])) != {0, 1}:
        raise ValueError("Warm-up data must contain both outbreak classes.")

    scaler = StandardScaler().fit(features[:warmup_count])
    scaled_features = scaler.transform(features)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=labels[:warmup_count],
    )
    classifier = SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=alpha,
        average=True,
        class_weight={0: class_weights[0], 1: class_weights[1]},
        random_state=seed,
    )
    classifier.partial_fit(scaled_features[:warmup_count], labels[:warmup_count], classes=[0, 1])

    warmup_margins = classifier.decision_function(scaled_features[:warmup_count]).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=seed)
    calibrator.fit(warmup_margins, labels[:warmup_count])

    all_margins = []
    all_probabilities = []
    evaluated_labels = []
    running_accuracy = []
    for start in range(warmup_count, row_count, batch_size):
        end = min(start + batch_size, row_count)
        batch_features = scaled_features[start:end]
        batch_labels = labels[start:end]
        margins = classifier.decision_function(batch_features)
        probabilities = calibrator.predict_proba(margins.reshape(-1, 1))[:, 1]

        all_margins.extend(margins.tolist())
        all_probabilities.extend(probabilities.tolist())
        evaluated_labels.extend(batch_labels.tolist())
        running_predictions = (np.asarray(all_margins) > 0).astype(int)
        running_accuracy.append(float(np.mean(running_predictions == evaluated_labels)))

        classifier.partial_fit(batch_features, batch_labels)

    margins = np.asarray(all_margins, dtype=float)
    probabilities = np.asarray(all_probabilities, dtype=float)
    evaluated_labels = np.asarray(evaluated_labels, dtype=int)
    predictions = (margins > 0).astype(int)
    result = calculate_metrics(evaluated_labels, predictions, probabilities)
    result.update(
        n_eval=int(len(evaluated_labels)),
        n_outbreaks=int(evaluated_labels.sum()),
        warmup=warmup_count,
        running_accuracy=running_accuracy,
    )

    risk = np.where(probabilities >= 0.50, "HIGH", np.where(probabilities >= 0.20, "MEDIUM", "LOW"))
    result["risk_counts"] = {level: int(np.count_nonzero(risk == level)) for level in ("HIGH", "MEDIUM", "LOW")}
    high_alerts = risk == "HIGH"
    outbreak_count = int(evaluated_labels.sum())
    high_count = int(high_alerts.sum())
    high_true_positives = int(np.count_nonzero(high_alerts & (evaluated_labels == 1)))
    result["HIGH_alert_recall"] = high_true_positives / outbreak_count if outbreak_count else float("nan")
    result["HIGH_alert_precision"] = high_true_positives / high_count if high_count else float("nan")
    return result, (evaluated_labels, probabilities)


def run_linearsvc_baseline(frame, train_fraction=0.70, seed=0):
    features = frame[FEATURES].to_numpy(dtype=float)
    labels = frame["outbreak"].to_numpy(dtype=int)
    train_count = int(train_fraction * len(frame))
    if train_count <= 0 or train_count >= len(frame):
        raise ValueError("The baseline split must leave non-empty training and test sets.")
    if len(np.unique(labels[:train_count])) != 2:
        raise ValueError("LinearSVC training rows must contain both outbreak classes.")

    scaler = StandardScaler().fit(features[:train_count])
    classifier = LinearSVC(class_weight="balanced", max_iter=20000, random_state=seed)
    classifier.fit(scaler.transform(features[:train_count]), labels[:train_count])
    margins = classifier.decision_function(scaler.transform(features[train_count:]))
    test_labels = labels[train_count:]
    result = calculate_metrics(test_labels, (margins > 0).astype(int), margins)
    result.update(n_train=train_count, n_test=len(test_labels))
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="CSV state-month table")
    source.add_argument("--postgres", action="store_true", help="Read from PostgreSQL using DATABASE_URL")
    parser.add_argument("--schema", default="synthetic", help="PostgreSQL schema (default: synthetic)")
    parser.add_argument("--out", default="results", help="Directory for JSON and optional plots")
    args = parser.parse_args()

    frame = load_data(csv_path=args.csv, schema=args.schema)
    is_synthetic = bool(
        "data_origin" in frame.columns
        and frame["data_origin"].astype(str).str.upper().eq("SYNTHETIC").any()
    )
    asvm, (labels, probabilities) = run_asvm(frame)
    baseline = run_linearsvc_baseline(frame)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_synthetic:
        print("*" * 78)
        print("SYNTHETIC DATA: pipeline demonstration only, not results on real surveillance data.")
        print("*" * 78)
    print(
        f"\nRows {len(frame)} | warm-up {asvm['warmup']} | evaluated {asvm['n_eval']} "
        f"| outbreaks {asvm['n_outbreaks']} ({asvm['n_outbreaks'] / asvm['n_eval']:.2%})"
    )
    print("\nASVM (prequential, classification = margin > 0)")
    print(f"  Confusion  TN {asvm['TN']}  FP {asvm['FP']}  FN {asvm['FN']}  TP {asvm['TP']}")
    for metric in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc"):
        value = asvm[metric]
        print(f"  {metric:<12}{value:.3f}" if math.isfinite(value) else f"  {metric:<12}n/a")
    print(
        f"  Risk levels  {asvm['risk_counts']} | HIGH-alert recall "
        f"{asvm['HIGH_alert_recall']:.3f}, precision {asvm['HIGH_alert_precision']:.3f}"
    )
    print(
        f"\nLinearSVC baseline (70/30 temporal split, n_test={baseline['n_test']}): "
        f"ROC-AUC {baseline['roc_auc']:.3f}, recall {baseline['recall']:.3f}, "
        f"precision {baseline['precision']:.3f}"
    )

    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as output_file:
        json.dump(json_safe({"asvm": asvm, "linearsvc": baseline, "synthetic": is_synthetic}), output_file, indent=2, allow_nan=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(5.5, 3.4))
        axis.plot(range(1, len(asvm["running_accuracy"]) + 1), asvm["running_accuracy"])
        axis.set_xlabel("Streaming batch")
        axis.set_ylabel("Running accuracy")
        axis.set_ylim(0.5, 1)
        figure.tight_layout()
        figure.savefig(output_dir / "learning_curve.png", dpi=200)
        plt.close(figure)

        if len(np.unique(labels)) == 2:
            false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
            figure, axis = plt.subplots(figsize=(4.4, 4))
            axis.plot(false_positive_rate, true_positive_rate, label=f"ASVM (AUC = {asvm['roc_auc']:.2f})")
            axis.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
            axis.set_xlabel("False positive rate")
            axis.set_ylabel("True positive rate")
            axis.legend(loc="lower right")
            figure.tight_layout()
            figure.savefig(output_dir / "roc_curve.png", dpi=200)
            plt.close(figure)
        print(f"\nSaved {results_path} and optional plots in {output_dir}")
    except ImportError:
        print(f"\nSaved {results_path}; install matplotlib to generate plots.")


if __name__ == "__main__":
    main()
