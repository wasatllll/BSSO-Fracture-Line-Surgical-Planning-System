from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def align_probability_columns(model, probabilities: np.ndarray, n_classes: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Predicted probabilities must be a two-dimensional matrix.")
    if probabilities.shape[1] != n_classes:
        raise ValueError(
            f"Expected {n_classes} probability columns, received {probabilities.shape[1]}."
        )

    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                classes = step.classes_
                break
    if classes is None:
        aligned = probabilities.copy()
    else:
        classes = np.asarray(classes).astype(int)
        if np.array_equal(classes, np.arange(n_classes)):
            aligned = probabilities.copy()
        else:
            aligned = np.zeros_like(probabilities, dtype=float)
            for source_column, class_id in enumerate(classes):
                if 0 <= class_id < n_classes:
                    aligned[:, class_id] = probabilities[:, source_column]
    aligned = np.clip(aligned, 0.0, None)
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return aligned / row_sums


def multiclass_micro_auc(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    y_binary = label_binarize(y_true, classes=np.arange(n_classes))
    return float(roc_auc_score(y_binary.ravel(), probabilities.ravel()))


def multiclass_macro_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    y_binary = label_binarize(y_true, classes=np.arange(n_classes))
    return float(np.mean(np.sum((probabilities - y_binary) ** 2, axis=1)))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray, n_classes: int) -> dict[str, float]:
    y_binary = label_binarize(y_true, classes=np.arange(n_classes))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc_micro": multiclass_micro_auc(y_true, probabilities, n_classes),
        "auc_macro_ovr": multiclass_macro_auc(y_true, probabilities),
        "ap_micro": float(average_precision_score(y_binary, probabilities, average="micro")),
        "ap_macro": float(average_precision_score(y_binary, probabilities, average="macro")),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(n_classes))),
        "brier_multiclass": multiclass_brier(y_true, probabilities, n_classes),
    }
    return metrics


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[str],
) -> pd.DataFrame:
    n_classes = len(class_labels)
    y_binary = label_binarize(y_true, classes=np.arange(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    rows: list[dict[str, float | int | str]] = []
    for class_id, class_label in enumerate(class_labels):
        tp = int(cm[class_id, class_id])
        fn = int(cm[class_id, :].sum() - tp)
        fp = int(cm[:, class_id].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        support = tp + fn
        predicted = tp + fp
        class_accuracy = (tp + tn) / cm.sum() if cm.sum() else np.nan
        precision = tp / predicted if predicted else np.nan
        recall = tp / support if support else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
        try:
            auc_value = float(roc_auc_score(y_binary[:, class_id], probabilities[:, class_id]))
        except ValueError:
            auc_value = np.nan
        try:
            ap_value = float(average_precision_score(y_binary[:, class_id], probabilities[:, class_id]))
        except ValueError:
            ap_value = np.nan
        rows.append(
            {
                "class": class_label,
                "support": support,
                "predicted": predicted,
                "class_accuracy": class_accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "auc_ovr": auc_value,
                "average_precision": ap_value,
            }
        )
    return pd.DataFrame(rows)


def _cluster_bootstrap_indices(patient_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    patient_ids = np.asarray(patient_ids)
    unique_patients = pd.unique(patient_ids)
    sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    return np.concatenate([np.flatnonzero(patient_ids == patient_id) for patient_id in sampled])


def patient_cluster_bootstrap_ci(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float, np.ndarray]:
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    for _ in range(n_bootstrap):
        indices = _cluster_bootstrap_indices(patient_ids, rng)
        try:
            values.append(float(metric(y_true[indices], probabilities[indices])))
        except (ValueError, FloatingPointError):
            continue
    if not values:
        return np.nan, np.nan, np.nan, np.array([], dtype=float)
    array = np.asarray(values, dtype=float)
    alpha = 1.0 - confidence_level
    return (
        float(np.mean(array)),
        float(np.quantile(array, alpha / 2)),
        float(np.quantile(array, 1 - alpha / 2)),
        array,
    )


def classwise_brier_scores(y_true: np.ndarray, probabilities: np.ndarray, class_labels: list[str]) -> pd.DataFrame:
    rows = []
    for class_id, label in enumerate(class_labels):
        binary = (y_true == class_id).astype(int)
        rows.append(
            {
                "class": label,
                "brier_score": float(brier_score_loss(binary, probabilities[:, class_id])),
            }
        )
    return pd.DataFrame(rows)


def hosmer_lemeshow_binary(y_true_binary: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> dict[str, float | int]:
    from scipy.stats import chi2

    data = pd.DataFrame({"observed": y_true_binary.astype(int), "probability": probability.astype(float)})
    data["bin"] = pd.qcut(data["probability"], q=n_bins, labels=False, duplicates="drop")
    grouped = data.groupby("bin", observed=True)
    statistic = 0.0
    actual_bins = 0
    for _, group in grouped:
        n = len(group)
        if n == 0:
            continue
        observed = float(group["observed"].sum())
        expected = float(group["probability"].sum())
        variance = expected * (1.0 - expected / n)
        if variance > 0:
            statistic += (observed - expected) ** 2 / variance
        actual_bins += 1
    degrees_freedom = max(actual_bins - 2, 1)
    p_value = float(1.0 - chi2.cdf(statistic, degrees_freedom))
    return {
        "hl_statistic": float(statistic),
        "degrees_freedom": int(degrees_freedom),
        "p_value": p_value,
        "n_bins": int(actual_bins),
    }


# Standard fast DeLong implementation for paired binary ROC curves.
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    midranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty(len(x), dtype=float)
    result[order] = midranks
    return result


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / m + sy / n
    return aucs, np.atleast_2d(covariance)


def paired_delong_binary(y_true_binary: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict[str, float]:
    y_true_binary = np.asarray(y_true_binary, dtype=int)
    if set(np.unique(y_true_binary)) != {0, 1}:
        raise ValueError("DeLong comparison requires both binary outcome classes.")
    order = np.argsort(-y_true_binary)
    label_1_count = int(y_true_binary.sum())
    predictions = np.vstack([score_a, score_b])[:, order]
    aucs, covariance = _fast_delong(predictions, label_1_count)
    contrast = np.array([[1.0, -1.0]])
    variance = float(contrast @ covariance @ contrast.T)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    difference = float(aucs[0] - aucs[1])
    z_statistic = difference / standard_error if standard_error > 0 else 0.0
    p_value = float(2 * norm.sf(abs(z_statistic)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "difference": difference,
        "standard_error": standard_error,
        "z_statistic": float(z_statistic),
        "p_value": p_value,
    }


def paired_delong_micro_multiclass(
    y_true: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    """Apply paired DeLong to the flattened one-vs-rest micro-average representation."""
    binary = label_binarize(y_true, classes=np.arange(n_classes)).ravel()
    return paired_delong_binary(binary, probabilities_a.ravel(), probabilities_b.ravel())
