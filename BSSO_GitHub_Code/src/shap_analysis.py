from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import PATIENT_ID_COL, SAMPLE_COL, TARGET_COL


def _normalize_shap_values(values, n_classes: int) -> np.ndarray:
    """Return SHAP values with shape (n_samples, n_features, n_classes)."""
    if isinstance(values, list):
        array = np.stack([np.asarray(item) for item in values], axis=-1)
    else:
        array = np.asarray(values)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim == 3 and array.shape[-1] == n_classes:
        return array
    if array.ndim == 3 and array.shape[0] == n_classes:
        return np.moveaxis(array, 0, -1)
    raise ValueError(f"Unsupported SHAP value shape: {array.shape}")


def run_shap_analysis(
    model,
    development_df: pd.DataFrame,
    features: list[str],
    class_labels: list[str],
    output_dir: str | Path,
    background_size: int = 50,
    evaluation_size: int | None = None,
    random_state: int = 42,
    local_case_index: int = 92,
    local_sample_id: str | int | None = None,
) -> dict[str, object]:
    """Generate global, dependence, and local SHAP outputs for the locked model."""
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optional_columns = [SAMPLE_COL] if SAMPLE_COL in development_df.columns else []
    data = development_df[
        [PATIENT_ID_COL, TARGET_COL] + optional_columns + features
    ].dropna(subset=[PATIENT_ID_COL, TARGET_COL] + features).reset_index(drop=True)
    X = data[features]

    requested_position = None
    requested_label = None
    if local_sample_id is not None and SAMPLE_COL in data.columns:
        def normalize_sample(value):
            try:
                numeric = float(value)
                return str(int(numeric)) if numeric.is_integer() else str(numeric)
            except Exception:
                return str(value).strip()

        target_sample = normalize_sample(local_sample_id)
        matches = data[SAMPLE_COL].map(normalize_sample).eq(target_sample)
        if matches.any():
            requested_position = int(np.flatnonzero(matches.to_numpy())[0])
            requested_label = f"sample {target_sample}"

    if evaluation_size is not None and evaluation_size < len(X):
        selected_indices = X.sample(
            evaluation_size, random_state=random_state
        ).index.tolist()
        if requested_position is not None and requested_position not in selected_indices:
            selected_indices[-1] = requested_position
        evaluation = X.loc[sorted(set(selected_indices))].copy()
    else:
        evaluation = X.copy()
    background = shap.sample(X, min(background_size, len(X)), random_state=random_state)

    def prediction_function(values):
        frame = pd.DataFrame(values, columns=features)
        return model.predict_proba(frame)

    explainer = shap.KernelExplainer(prediction_function, background)
    values = explainer.shap_values(evaluation)
    shap_array = _normalize_shap_values(values, len(class_labels))

    artifact = {
        "shap_values": shap_array,
        "feature_names": features,
        "class_labels": class_labels,
        "X": evaluation,
        "row_indices": evaluation.index.to_numpy(),
    }
    joblib.dump(artifact, output_dir / "locked_model_shap_values.joblib")

    mean_absolute = np.mean(np.abs(shap_array), axis=0)
    importance_rows = []
    for feature_index, feature in enumerate(features):
        row = {"feature": feature}
        for class_index, class_label in enumerate(class_labels):
            row[f"mean_abs_shap_type_{class_label}"] = float(
                mean_absolute[feature_index, class_index]
            )
        row["mean_abs_shap_overall"] = float(np.mean(mean_absolute[feature_index]))
        importance_rows.append(row)
    importance = pd.DataFrame(importance_rows).sort_values(
        "mean_abs_shap_overall", ascending=False
    )
    importance.to_csv(output_dir / "locked_model_shap_importance.csv", index=False)

    for class_index, class_label in enumerate(class_labels):
        explanation = shap.Explanation(
            values=shap_array[:, :, class_index],
            data=evaluation.to_numpy(),
            feature_names=features,
        )
        plt.figure(figsize=(8.5, 5.8))
        shap.plots.beeswarm(explanation, max_display=len(features), show=False)
        plt.title(f"Locked model SHAP summary: Type {class_label}")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"shap_beeswarm_type_{class_label}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

        for feature_index, feature in enumerate(features):
            fig, axis = plt.subplots(figsize=(6.5, 5.2))
            axis.scatter(
                evaluation.iloc[:, feature_index],
                shap_array[:, feature_index, class_index],
                s=24,
                alpha=0.75,
            )
            axis.axhline(0, linestyle="--", linewidth=1)
            axis.set_xlabel(feature)
            axis.set_ylabel(f"SHAP value for Type {class_label}")
            axis.set_title(f"{feature} dependence: Type {class_label}")
            axis.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(
                output_dir / f"shap_dependence_{feature.replace('.', '_')}_type_{class_label}.png",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close(fig)

    if requested_position is not None and requested_position in evaluation.index:
        local_position = int(np.flatnonzero(evaluation.index.to_numpy() == requested_position)[0])
        local_label = requested_label or f"row {requested_position}"
        local_filename = f"shap_waterfall_sample_{str(local_sample_id).replace('.', '_')}"
    elif 0 <= local_case_index < len(evaluation):
        local_position = int(local_case_index)
        local_label = f"row {local_case_index}"
        local_filename = f"shap_waterfall_case_{local_case_index}"
    else:
        local_position = None

    if local_position is not None:
        prediction = model.predict_proba(evaluation.iloc[[local_position]])[0]
        predicted_class = int(np.argmax(prediction))
        expected_value = np.asarray(explainer.expected_value, dtype=float).ravel()
        if expected_value.size == 1 and len(class_labels) > 1:
            expected_value = np.repeat(expected_value, len(class_labels))
        local_explanation = shap.Explanation(
            values=shap_array[local_position, :, predicted_class],
            base_values=float(expected_value[predicted_class]),
            data=evaluation.iloc[local_position].to_numpy(),
            feature_names=features,
        )
        plt.figure(figsize=(8.5, 5.5))
        shap.plots.waterfall(local_explanation, max_display=len(features), show=False)
        plt.title(
            f"Local SHAP {local_label}: predicted Type "
            f"{class_labels[predicted_class]} ({prediction[predicted_class]:.3f})"
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{local_filename}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    return {"importance": importance, "artifact": artifact}
