"""Training and classification module for the QUBO classification project.

Phase 3 (training) and Phase 4 (prediction):

* ``train``   -> fits one of the three available classifiers on the reduced
                 training set and saves the model with joblib;
* ``predict`` -> loads a saved model, classifies the reduced test set, writes
                 the per-record predictions and the quality statistics.

The module is importable and executable from the command line:

    python model.py train \\
        --classifier random_forest \\
        --in-reduced training_reduced.csv \\
        --target target \\
        --out-model model.joblib \\
        --out-metrics training_metrics.json \\
        --seed 42

    python model.py predict \\
        --input-testset test_reduced.csv \\
        --target target \\
        --model model.joblib \\
        --out-predictions predictions.csv \\
        --out-stats classification_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

__all__ = ["train", "predict", "available_classifiers", "main"]

# The credit-risk target is strongly imbalanced (~1.5% of positives), and the
# project is scored on the F1 of class 1. Class weighting counteracts the bias
# towards the majority class; set to None to train on the raw distribution.
_CLASS_WEIGHT = "balanced"


# --------------------------------------------------------------------------- #
# Classifier registry
# --------------------------------------------------------------------------- #
def _make_random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=_CLASS_WEIGHT,
        n_jobs=-1,
        random_state=seed,
    )


def _make_logistic_regression(seed: int) -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight=_CLASS_WEIGHT,
        n_jobs=-1,
        random_state=seed,
    )


def _make_gradient_boosting(seed: int) -> HistGradientBoostingClassifier:
    # Histogram-based boosting: same algorithm family as GradientBoostingClassifier
    # but orders of magnitude faster on the 1.5M-sample evaluation dataset.
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        class_weight=_CLASS_WEIGHT,
        random_state=seed,
    )


_CLASSIFIERS = {
    "random_forest": _make_random_forest,
    "logistic_regression": _make_logistic_regression,
    "gradient_boosting": _make_gradient_boosting,
}

# Reverse lookup used by predict() to report which classifier produced a model.
_CLASS_NAME_TO_KEY = {
    "RandomForestClassifier": "random_forest",
    "LogisticRegression": "logistic_regression",
    "HistGradientBoostingClassifier": "gradient_boosting",
}


def available_classifiers() -> List[str]:
    """Names accepted by the ``classifier`` parameter."""
    return sorted(_CLASSIFIERS)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _build_classifier(name: str, seed: int):
    key = _normalize_name(name)
    if key not in _CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier {name!r}. Available: {', '.join(available_classifiers())}"
        )
    return key, _CLASSIFIERS[key](seed)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ensure_parent_dir(path: str) -> Path:
    """Create the parent directory of ``path`` if needed (no absolute paths)."""
    file_path = Path(path)
    if file_path.parent and str(file_path.parent) not in ("", "."):
        file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def _read_reduced(csv_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Read a reduced dataset and split it into features and target."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} not found in {csv_path}")

    x = df.drop(columns=[target_column])
    if x.shape[1] == 0:
        raise ValueError(f"{csv_path} contains no feature column.")

    y = pd.to_numeric(df[target_column], errors="coerce")
    if y.isna().any():
        raise ValueError(f"The target column of {csv_path} contains invalid values.")
    return x, y.astype(int)


def _positive_scores(model, x: pd.DataFrame) -> np.ndarray:
    """Probability (or normalized score) of the positive class."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = list(model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return proba[:, -1]

    # Fallback for estimators without predict_proba: squash the margin.
    margins = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-margins))


def _align_features(model, x: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """Reorder the test columns to match the ones seen during training."""
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return x

    expected = list(expected)
    missing = [c for c in expected if c not in x.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing {len(missing)} feature(s) used at training time: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return x[expected]


# --------------------------------------------------------------------------- #
# Phase 3: training
# --------------------------------------------------------------------------- #
def train(
    classifier: str,         # classifier to use
    reducedTrain_csv: str,   # training dataset
    target_column: str,      # target column name
    model_path: str,         # saved trained classifier
    metrics_json: str,       # file with training statistics
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit the selected classifier on the reduced training set.

    Returns
    -------
    dict
        The same dictionary written to ``metrics_json``.
    """
    key, estimator = _build_classifier(classifier, seed)

    t0 = time.perf_counter()
    x_train, y_train = _read_reduced(reducedTrain_csv, target_column)
    dataset_input_time = time.perf_counter() - t0

    n_classes = int(y_train.nunique())
    if n_classes < 2:
        raise ValueError(
            "The training target contains a single class: the classifier "
            "cannot be trained. Check the train/test split ratio."
        )

    t1 = time.perf_counter()
    estimator.fit(x_train, y_train)
    training_time = time.perf_counter() - t1

    model_out = _ensure_parent_dir(model_path)
    joblib.dump(estimator, model_out)

    n_samples = int(len(x_train))
    target_1_count = int((y_train == 1).sum())

    metrics: Dict[str, Any] = {
        "classifier": key,
        "seed": int(seed),
        "training_dataset": Path(reducedTrain_csv).name,
        "target_column": target_column,
        "model_path": str(model_path),
        "n_samples": n_samples,
        "n_features": int(x_train.shape[1]),
        "target_1_percentage": round(100.0 * target_1_count / n_samples, 4) if n_samples else 0.0,
        "dataset_input_time": round(dataset_input_time, 4),
        "training_time": round(training_time, 4),
    }

    metrics_out = _ensure_parent_dir(metrics_json)
    with metrics_out.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    return metrics


# --------------------------------------------------------------------------- #
# Phase 4: classification of the test data
# --------------------------------------------------------------------------- #
def predict(
    reduced_Test_csv: str,     # Input test set
    target_column: str,        # Target column name
    model_path: str,           # saved trained classifier to use
    predictions_csv: str,      # Output predictions
    classif_stats_json: str,   # File with classification stats
) -> Dict[str, Any]:
    """Classify a reduced dataset and write predictions plus quality statistics.

    Returns
    -------
    dict
        The same dictionary written to ``classif_stats_json``.
    """
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    x_test, y_test = _read_reduced(reduced_Test_csv, target_column)
    model = joblib.load(model_file)
    x_test = _align_features(model, x_test, reduced_Test_csv)

    predictions = np.asarray(model.predict(x_test)).astype(int)
    scores = _positive_scores(model, x_test)

    # --- per-record output ---------------------------------------------------
    predictions_df = pd.DataFrame(
        {
            "row_n": np.arange(len(x_test), dtype=int),
            "target": y_test.to_numpy(dtype=int),
            "prediction": predictions,
            "score": np.round(scores, 6),
        }
    )
    predictions_out = _ensure_parent_dir(predictions_csv)
    predictions_df.to_csv(predictions_out, index=False)

    # --- quality statistics --------------------------------------------------
    y_true = y_test.to_numpy(dtype=int)
    n_samples = int(len(y_true))
    target_1_count = int((y_true == 1).sum())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    # ROC-AUC is undefined when the test set holds a single class.
    if np.unique(y_true).size < 2:
        roc_auc: float | None = None
        print(
            "[model] Warning: the test target is constant, ROC-AUC is undefined (null).",
            file=sys.stderr,
        )
    else:
        roc_auc = float(roc_auc_score(y_true, scores))

    stats: Dict[str, Any] = {
        "classifier": _CLASS_NAME_TO_KEY.get(type(model).__name__, type(model).__name__),
        "n_samples": n_samples,
        "target_1_count": target_1_count,
        "target_1_percentage": round(100.0 * target_1_count / n_samples, 4) if n_samples else 0.0,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "class_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "class_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": [[int(v) for v in row] for row in matrix],
        },
    }

    stats_out = _ensure_parent_dir(classif_stats_json)
    with stats_out.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    return stats


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model.py",
        description="Phases 3 and 4: training and classification.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Train a classifier.")
    p_train.add_argument(
        "--classifier", required=True, choices=available_classifiers(),
        help="Classifier to train.",
    )
    p_train.add_argument("--in-reduced", required=True, help="Reduced training set (CSV).")
    p_train.add_argument("--target", required=True, help="Name of the target column.")
    p_train.add_argument("--out-model", required=True, help="Output model file (.joblib).")
    p_train.add_argument("--out-metrics", required=True, help="Output training metrics (JSON).")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")

    p_pred = subparsers.add_parser("predict", help="Classify a reduced dataset.")
    p_pred.add_argument("--input-testset", required=True, help="Reduced test set (CSV).")
    p_pred.add_argument("--target", required=True, help="Name of the target column.")
    p_pred.add_argument("--model", required=True, help="Trained model file (.joblib).")
    p_pred.add_argument("--out-predictions", required=True, help="Output predictions (CSV).")
    p_pred.add_argument("--out-stats", required=True, help="Output classification stats (JSON).")

    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.command == "train":
            metrics = train(
                classifier=args.classifier,
                reducedTrain_csv=args.in_reduced,
                target_column=args.target,
                model_path=args.out_model,
                metrics_json=args.out_metrics,
                seed=args.seed,
            )
            print(
                f"[model] {metrics['classifier']} trained on {metrics['n_samples']} samples "
                f"x {metrics['n_features']} features "
                f"({metrics['target_1_percentage']}% positives) in {metrics['training_time']}s."
            )
            print(f"[model] Model   -> {args.out_model}")
            print(f"[model] Metrics -> {args.out_metrics}")
        else:
            stats = predict(
                reduced_Test_csv=args.input_testset,
                target_column=args.target,
                model_path=args.model,
                predictions_csv=args.out_predictions,
                classif_stats_json=args.out_stats,
            )
            roc = stats["roc_auc"]
            print(
                f"[model] {stats['n_samples']} samples classified. "
                f"accuracy = {stats['accuracy']:.4f}, "
                f"F1(class 1) = {stats['class_1']['f1']:.4f}, "
                f"ROC-AUC = {roc:.4f}" if roc is not None else
                f"[model] {stats['n_samples']} samples classified. "
                f"accuracy = {stats['accuracy']:.4f}, "
                f"F1(class 1) = {stats['class_1']['f1']:.4f}, ROC-AUC = n/a"
            )
            print(f"[model] Predictions -> {args.out_predictions}")
            print(f"[model] Statistics  -> {args.out_stats}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[model] Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())