from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .constants import MODEL_ORDER, PATIENT_ID_COL


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def _save(fig, path: Path, dpi: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_cv_roc_panels(model_results: dict[str, object], y_train: np.ndarray, class_labels: list[str], output_path: str | Path) -> None:
    """Plot fold-wise mean micro-average ROC curves for the training-set CV analysis."""
    output_path = Path(output_path)
    n_classes = len(class_labels)
    class_indices = np.arange(n_classes)
    mean_fpr = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    axes = axes.ravel()
    for axis, model_name in zip(axes, MODEL_ORDER):
        result = model_results[model_name]
        interpolated_tprs = []
        fold_aucs = []
        for fold_id in sorted(np.unique(result.oof_fold_ids)):
            mask = result.oof_fold_ids == fold_id
            y_binary = label_binarize(y_train[mask], classes=class_indices)
            fpr, tpr, _ = roc_curve(
                y_binary.ravel(), result.oof_probabilities[mask].ravel()
            )
            interpolated = np.interp(mean_fpr, fpr, tpr)
            interpolated[0] = 0.0
            interpolated[-1] = 1.0
            interpolated_tprs.append(interpolated)
            fold_aucs.append(auc(fpr, tpr))

        tpr_array = np.asarray(interpolated_tprs)
        mean_tpr = tpr_array.mean(axis=0)
        mean_tpr[-1] = 1.0
        sd_tpr = tpr_array.std(axis=0, ddof=0)
        lower = np.maximum(mean_tpr - sd_tpr, 0)
        upper = np.minimum(mean_tpr + sd_tpr, 1)
        axis.plot(
            mean_fpr,
            mean_tpr,
            linewidth=2.2,
            label=f"Mean micro-AUC = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs, ddof=0):.3f}",
        )
        axis.fill_between(mean_fpr, lower, upper, alpha=0.22, label="±1 fold SD")
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        axis.set_title(f"{model_name}\nCV accuracy = {result.cv_accuracy:.3f}")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.legend(frameon=False, loc="lower right")
        axis.grid(alpha=0.25)
    for axis in axes[len(MODEL_ORDER):]:
        axis.axis("off")
    fig.tight_layout()
    _save(fig, output_path)


def plot_test_confusion_matrices(model_results: dict[str, object], y_test: np.ndarray, class_labels: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    fig, axes = plt.subplots(3, 2, figsize=(13, 17))
    axes = axes.ravel()
    for axis, model_name in zip(axes, MODEL_ORDER):
        result = model_results[model_name]
        matrix = confusion_matrix(y_test, result.test_predictions, labels=np.arange(len(class_labels)))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=axis,
        )
        axis.set_title(f"{model_name}\nTest accuracy = {result.test_metrics['accuracy']:.3f}")
        axis.set_xlabel("Predicted type")
        axis.set_ylabel("Observed type")
    for axis in axes[len(MODEL_ORDER):]:
        axis.axis("off")
    fig.tight_layout()
    _save(fig, output_path)


def plot_test_roc_all_models(model_results: dict[str, object], y_test: np.ndarray, class_labels: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    y_binary = label_binarize(y_test, classes=np.arange(len(class_labels)))
    fig, axis = plt.subplots(figsize=(8.5, 7.5))
    for model_name in MODEL_ORDER:
        result = model_results[model_name]
        fpr, tpr, _ = roc_curve(y_binary.ravel(), result.test_probabilities.ravel())
        axis.plot(fpr, tpr, linewidth=2.1, label=f"{model_name} ({auc(fpr, tpr):.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Internal test-set micro-average ROC curves")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


def plot_test_pr_all_models(model_results: dict[str, object], y_test: np.ndarray, class_labels: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    y_binary = label_binarize(y_test, classes=np.arange(len(class_labels)))
    fig, axis = plt.subplots(figsize=(8.5, 7.5))
    for model_name in MODEL_ORDER:
        result = model_results[model_name]
        precision, recall, _ = precision_recall_curve(y_binary.ravel(), result.test_probabilities.ravel())
        ap = average_precision_score(y_binary, result.test_probabilities, average="micro")
        axis.plot(recall, precision, linewidth=2.1, label=f"{model_name} ({ap:.3f})")
    axis.axhline(1 / len(class_labels), linestyle="--", linewidth=1, label="Class-balance reference")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Internal test-set micro-average precision-recall curves")
    axis.legend(frameon=False, loc="lower left")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


def plot_test_calibration_all_models(model_results: dict[str, object], y_test: np.ndarray, class_labels: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    y_binary = label_binarize(y_test, classes=np.arange(len(class_labels)))
    fig, axis = plt.subplots(figsize=(8.5, 7.5))
    for model_name in MODEL_ORDER:
        probabilities = model_results[model_name].test_probabilities
        fraction, mean_probability = calibration_curve(
            y_binary.ravel(), probabilities.ravel(), n_bins=8, strategy="quantile"
        )
        axis.plot(mean_probability, fraction, marker="o", linewidth=1.8, label=model_name)
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed fraction")
    axis.set_title("Internal test-set micro-average calibration")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


def plot_external_confusion_matrices(external_results: dict[str, dict[str, object]], class_labels: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, len(external_results), figsize=(7 * len(external_results), 5.8))
    axes = np.atleast_1d(axes)
    for axis, (cohort_name, result) in zip(axes, external_results.items()):
        matrix = confusion_matrix(
            result["y"], result["predictions"], labels=np.arange(len(class_labels))
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=axis,
        )
        axis.set_title(
            f"{cohort_name}\nAccuracy = {result['metrics']['accuracy']:.3f}; "
            f"Macro-F1 = {result['metrics']['f1_macro']:.3f}"
        )
        axis.set_xlabel("Predicted type")
        axis.set_ylabel("Observed type")
    fig.tight_layout()
    _save(fig, output_path)


def _patient_cluster_roc_band(
    y: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
    n_classes: int,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    grid_size: int = 201,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patient-clustered percentile band for a micro-average ROC curve."""
    rng = np.random.default_rng(random_state)
    unique_patients = pd.unique(patient_ids)
    fpr_grid = np.linspace(0, 1, grid_size)
    curves = []
    for _ in range(n_bootstrap):
        sampled_patients = rng.choice(
            unique_patients, size=len(unique_patients), replace=True
        )
        indices = np.concatenate(
            [np.flatnonzero(patient_ids == patient) for patient in sampled_patients]
        )
        y_binary = label_binarize(y[indices], classes=np.arange(n_classes))
        if len(np.unique(y_binary.ravel())) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(
                y_binary.ravel(), probabilities[indices].ravel()
            )
        except ValueError:
            continue
        interpolated = np.interp(fpr_grid, fpr, tpr)
        interpolated[0] = 0.0
        interpolated[-1] = 1.0
        curves.append(interpolated)
    if not curves:
        nan = np.full_like(fpr_grid, np.nan)
        return fpr_grid, nan, nan
    array = np.asarray(curves)
    return (
        fpr_grid,
        np.quantile(array, 0.025, axis=0),
        np.quantile(array, 0.975, axis=0),
    )


def plot_external_roc(
    external_results: dict[str, dict[str, object]],
    class_labels: list[str],
    output_path: str | Path,
    n_bootstrap: int = 1000,
) -> None:
    output_path = Path(output_path)
    n_classes = len(class_labels)
    fig, axis = plt.subplots(figsize=(8.5, 7.5))
    for cohort_index, (cohort_name, result) in enumerate(external_results.items()):
        y_binary = label_binarize(result["y"], classes=np.arange(n_classes))
        fpr, tpr, _ = roc_curve(y_binary.ravel(), result["probabilities"].ravel())
        line = axis.plot(
            fpr, tpr, linewidth=2.2, label=f"{cohort_name} ({auc(fpr, tpr):.3f})"
        )[0]
        band_fpr, lower, upper = _patient_cluster_roc_band(
            result["y"],
            result["probabilities"],
            result["data"][PATIENT_ID_COL].to_numpy(),
            n_classes,
            n_bootstrap=n_bootstrap,
            random_state=42 + cohort_index,
        )
        if np.isfinite(lower).any():
            axis.fill_between(
                band_fpr, lower, upper, color=line.get_color(), alpha=0.18
            )
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("External validation micro-average ROC curves")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


def plot_external_pr(external_results: dict[str, dict[str, object]], class_labels: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    fig, axis = plt.subplots(figsize=(8.5, 7.5))
    for cohort_name, result in external_results.items():
        y_binary = label_binarize(result["y"], classes=np.arange(len(class_labels)))
        precision, recall, _ = precision_recall_curve(y_binary.ravel(), result["probabilities"].ravel())
        ap = average_precision_score(y_binary, result["probabilities"], average="micro")
        axis.plot(recall, precision, linewidth=2.2, label=f"{cohort_name} ({ap:.3f})")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("External validation micro-average precision-recall curves")
    axis.legend(frameon=False, loc="lower left")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


def plot_external_calibration(external_results: dict[str, dict[str, object]], class_labels: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    fig, axis = plt.subplots(figsize=(8.5, 7.5))
    for cohort_name, result in external_results.items():
        y_binary = label_binarize(result["y"], classes=np.arange(len(class_labels)))
        fraction, mean_probability = calibration_curve(
            y_binary.ravel(), result["probabilities"].ravel(), n_bins=6, strategy="quantile"
        )
        axis.plot(mean_probability, fraction, marker="o", linewidth=2.0, label=cohort_name)
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed fraction")
    axis.set_title("External validation micro-average calibration")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


def plot_test_radar(model_results: dict[str, object], output_path: str | Path) -> None:
    from math import pi

    output_path = Path(output_path)
    categories = ["Accuracy", "Precision", "Recall", "Macro-F1", "Micro-AUC", "Micro-AP"]
    angles = [index / len(categories) * 2 * pi for index in range(len(categories))]
    angles += angles[:1]

    fig, axis = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    for model_name in MODEL_ORDER:
        metrics = model_results[model_name].test_metrics
        values = [
            metrics["accuracy"],
            metrics["precision_macro"],
            metrics["recall_macro"],
            metrics["f1_macro"],
            metrics["auc_micro"],
            metrics["ap_micro"],
        ]
        values += values[:1]
        axis.plot(angles, values, linewidth=2, label=model_name)
        axis.fill(angles, values, alpha=0.08)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(categories)
    axis.set_ylim(0, 1)
    axis.set_title("Internal test-set performance comparison", pad=25)
    axis.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.35, 1.12))
    fig.tight_layout()
    _save(fig, output_path)


def plot_feature_importance(model_results: dict[str, object], features: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    models = ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for axis, model_name in zip(axes, models):
        estimator = model_results[model_name].estimator
        if model_name == "Logistic Regression":
            classifier = estimator.named_steps["classifier"]
            importance = np.mean(np.abs(classifier.coef_), axis=0)
            label = "Mean absolute coefficient"
        elif model_name == "CatBoost":
            importance = estimator.get_feature_importance()
            label = "Feature importance"
        else:
            importance = estimator.feature_importances_
            label = "Feature importance"
        order = np.argsort(importance)
        axis.barh(np.asarray(features)[order], np.asarray(importance)[order])
        axis.set_xlabel(label)
        axis.set_title(model_name)
        axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save(fig, output_path)
