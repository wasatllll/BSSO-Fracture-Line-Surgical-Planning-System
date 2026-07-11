from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import (
    FINAL_FEATURES,
    NEAR_OPTIMAL_RATIO,
    PATIENT_ID_COL,
    TARGET_COL,
)
from .data import clean_class_label
from .metrics import align_probability_columns


def build_reverse_planning_manifest(
    training_df: pd.DataFrame,
    class_labels: list[str],
    output_path: str | Path,
    near_optimal_ratio: float = NEAR_OPTIMAL_RATIO,
) -> dict[str, object]:
    """Build all deployment domains from the predefined training set only."""
    required = [TARGET_COL, "LLBCE", "Depth.of.A", "MRT", "PMBT"]
    data = training_df[required].dropna().copy()
    data[TARGET_COL] = data[TARGET_COL].map(clean_class_label)

    class_domains: dict[str, object] = {}
    for class_label in class_labels:
        subset = data[data[TARGET_COL] == clean_class_label(class_label)]
        if len(subset) < 3:
            raise ValueError(f"Insufficient training observations for class {class_label} domains.")
        class_domains[clean_class_label(class_label)] = {
            "n_sides": int(len(subset)),
            "level1": {
                "name": "IQR strict domain",
                "quantiles": [0.25, 0.75],
                "depth_range": [
                    float(subset["Depth.of.A"].quantile(0.25)),
                    float(subset["Depth.of.A"].quantile(0.75)),
                ],
                "llbce_range": [
                    float(subset["LLBCE"].quantile(0.25)),
                    float(subset["LLBCE"].quantile(0.75)),
                ],
            },
            "level2": {
                "name": "10th-90th percentile strict domain",
                "quantiles": [0.10, 0.90],
                "depth_range": [
                    float(subset["Depth.of.A"].quantile(0.10)),
                    float(subset["Depth.of.A"].quantile(0.90)),
                ],
                "llbce_range": [
                    float(subset["LLBCE"].quantile(0.10)),
                    float(subset["LLBCE"].quantile(0.90)),
                ],
            },
        }

    fixed_input_domain = {}
    for feature in ["MRT", "PMBT"]:
        fixed_input_domain[feature] = {
            "minimum": float(data[feature].min()),
            "maximum": float(data[feature].max()),
            "p01": float(data[feature].quantile(0.01)),
            "p99": float(data[feature].quantile(0.99)),
            "median": float(data[feature].median()),
        }

    manifest = {
        "domain_source": "predefined training set only",
        "external_cohorts_used": False,
        "internal_test_set_used": False,
        "features": FINAL_FEATURES,
        "class_labels": [clean_class_label(label) for label in class_labels],
        "near_optimal_ratio": float(near_optimal_ratio),
        "stable_region_definition": (
            "Connected region containing the optimum, with target probability at least 90% "
            "of the maximum and the target remaining the highest-probability class."
        ),
        "fallback_rule": (
            "IQR strict domain first; if unsupported, 10th-90th percentile strict domain; "
            "otherwise no supported recommendation."
        ),
        "class_domains": class_domains,
        "fixed_input_domain": fixed_input_domain,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def save_deployment_bundle(
    model,
    label_encoder,
    features: list[str],
    manifest: dict[str, object],
    output_path: str | Path,
    metadata: dict[str, object] | None = None,
) -> None:
    bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_names": list(features),
        "class_labels": [clean_class_label(x) for x in label_encoder.classes_],
        "reverse_planning_manifest": manifest,
        "metadata": metadata or {},
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def load_deployment_bundle(path: str | Path) -> dict[str, object]:
    bundle = joblib.load(path)
    required = {
        "model",
        "label_encoder",
        "feature_names",
        "class_labels",
        "reverse_planning_manifest",
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Deployment bundle is missing keys: {sorted(missing)}")
    return bundle


def _connected_component(mask: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    if not bool(mask[origin]):
        return np.zeros_like(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    queue = deque([origin])
    visited[origin] = True
    rows, columns = mask.shape
    while queue:
        row, column = queue.popleft()
        for row_delta, column_delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_row = row + row_delta
            next_column = column + column_delta
            if (
                0 <= next_row < rows
                and 0 <= next_column < columns
                and mask[next_row, next_column]
                and not visited[next_row, next_column]
            ):
                visited[next_row, next_column] = True
                queue.append((next_row, next_column))
    return visited


class ReversePlanner:
    def __init__(self, bundle: dict[str, object], grid_size: int = 150):
        self.model = bundle["model"]
        self.label_encoder = bundle["label_encoder"]
        self.feature_names = list(bundle["feature_names"])
        self.class_labels = [clean_class_label(x) for x in bundle["class_labels"]]
        self.manifest = bundle["reverse_planning_manifest"]
        self.grid_size = int(grid_size)
        self.near_optimal_ratio = float(self.manifest["near_optimal_ratio"])

    def validate_fixed_inputs(self, mrt: float, pmbt: float) -> tuple[bool, list[str]]:
        messages = []
        values = {"MRT": float(mrt), "PMBT": float(pmbt)}
        for feature, value in values.items():
            domain = self.manifest["fixed_input_domain"][feature]
            if value < domain["minimum"] or value > domain["maximum"]:
                messages.append(
                    f"{feature}={value:.3f} is outside the observed training range "
                    f"[{domain['minimum']:.3f}, {domain['maximum']:.3f}]."
                )
        return len(messages) == 0, messages

    def _grid(self, depth_range, llbce_range, mrt: float, pmbt: float):
        depth_values = np.linspace(float(depth_range[0]), float(depth_range[1]), self.grid_size)
        llbce_values = np.linspace(float(llbce_range[0]), float(llbce_range[1]), self.grid_size)
        depth_grid, llbce_grid = np.meshgrid(depth_values, llbce_values)
        values = {
            "LLBCE": llbce_grid.ravel(),
            "PMBT": np.full(depth_grid.size, float(pmbt)),
            "MRT": np.full(depth_grid.size, float(mrt)),
            "Depth.of.A": depth_grid.ravel(),
        }
        grid = pd.DataFrame({feature: values[feature] for feature in self.feature_names})
        return grid, depth_grid, llbce_grid

    def _evaluate_level(self, class_label: str, level_name: str, mrt: float, pmbt: float) -> dict[str, object]:
        class_label = clean_class_label(class_label)
        class_index = self.class_labels.index(class_label)
        domain = self.manifest["class_domains"][class_label][level_name]
        grid, depth_grid, llbce_grid = self._grid(
            domain["depth_range"], domain["llbce_range"], mrt, pmbt
        )
        probabilities = align_probability_columns(
            self.model,
            self.model.predict_proba(grid),
            len(self.class_labels),
        )
        probability_grids = [
            probabilities[:, index].reshape(self.grid_size, self.grid_size)
            for index in range(len(self.class_labels))
        ]
        target_probability = probability_grids[class_index]
        top_class = np.argmax(probabilities, axis=1).reshape(self.grid_size, self.grid_size)
        supported = top_class == class_index

        if not np.any(supported):
            best_position = np.unravel_index(np.argmax(target_probability), target_probability.shape)
            return {
                "success": False,
                "reason": "No grid point retained the target as the highest-probability class.",
                "class_label": class_label,
                "class_index": class_index,
                "level": level_name,
                "domain": domain,
                "depth_grid": depth_grid,
                "llbce_grid": llbce_grid,
                "probability_grids": probability_grids,
                "target_probability": target_probability,
                "top_class": top_class,
                "best_position": best_position,
                "stable_mask": np.zeros_like(target_probability, dtype=bool),
                "maximum_probability": float(target_probability[best_position]),
                "probabilities_at_optimum": probabilities[
                    np.ravel_multi_index(best_position, target_probability.shape)
                ],
            }

        search_surface = target_probability.copy()
        search_surface[~supported] = -np.inf
        best_position = np.unravel_index(np.argmax(search_surface), search_surface.shape)
        maximum_probability = float(target_probability[best_position])
        threshold = self.near_optimal_ratio * maximum_probability
        candidate_mask = (target_probability >= threshold) & supported
        stable_mask = _connected_component(candidate_mask, best_position)
        success = bool(np.any(stable_mask))
        return {
            "success": success,
            "reason": "Stable connected region found." if success else "No connected stable region.",
            "class_label": class_label,
            "class_index": class_index,
            "level": level_name,
            "domain": domain,
            "depth_grid": depth_grid,
            "llbce_grid": llbce_grid,
            "probability_grids": probability_grids,
            "target_probability": target_probability,
            "top_class": top_class,
            "best_position": best_position,
            "stable_mask": stable_mask,
            "maximum_probability": maximum_probability,
            "near_optimal_threshold": threshold,
            "probabilities_at_optimum": probabilities[
                np.ravel_multi_index(best_position, target_probability.shape)
            ],
        }

    def plan_class(self, class_label: str, mrt: float, pmbt: float) -> tuple[dict[str, object], list[dict[str, object]]]:
        in_domain, messages = self.validate_fixed_inputs(mrt, pmbt)
        if not in_domain:
            raise ValueError(" ".join(messages))
        attempts = []
        for level_name in ["level1", "level2"]:
            result = self._evaluate_level(class_label, level_name, mrt, pmbt)
            attempts.append(result)
            if result["success"]:
                return result, attempts
        return attempts[-1], attempts

    def plan_all(self, mrt: float, pmbt: float) -> dict[str, object]:
        selected = []
        attempts = []
        summaries = []
        for class_label in self.class_labels:
            result, class_attempts = self.plan_class(class_label, mrt, pmbt)
            selected.append(result)
            attempts.extend(class_attempts)
            summaries.append(self.summarize(result))
        return {
            "selected_results": selected,
            "attempts": attempts,
            "summary": pd.DataFrame(summaries),
        }

    def summarize(self, result: dict[str, object]) -> dict[str, object]:
        depth_grid = result["depth_grid"]
        llbce_grid = result["llbce_grid"]
        stable_mask = result["stable_mask"]
        best_position = result["best_position"]
        row: dict[str, object] = {
            "type": f"Type {result['class_label']}",
            "supported_recommendation": bool(result["success"]),
            "selected_level": result["level"] if result["success"] else "none",
            "maximum_target_probability": float(result["maximum_probability"]),
            "surface_peak_depth_of_a": float(depth_grid[best_position]),
            "surface_peak_llbce": float(llbce_grid[best_position]),
            "optimal_depth_of_a": (
                float(depth_grid[best_position]) if result["success"] else np.nan
            ),
            "optimal_llbce": (
                float(llbce_grid[best_position]) if result["success"] else np.nan
            ),
        }
        for index, class_label in enumerate(self.class_labels):
            row[f"probability_type_{class_label}_at_optimum"] = float(
                result["probabilities_at_optimum"][index]
            )

        if not result["success"]:
            for name in [
                "depth_mean",
                "depth_sd",
                "depth_median",
                "depth_q1",
                "depth_q3",
                "depth_min",
                "depth_max",
                "llbce_mean",
                "llbce_sd",
                "llbce_median",
                "llbce_q1",
                "llbce_q3",
                "llbce_min",
                "llbce_max",
            ]:
                row[name] = np.nan
            return row

        depth = depth_grid[stable_mask]
        llbce = llbce_grid[stable_mask]
        row.update(
            {
                "depth_mean": float(np.mean(depth)),
                "depth_sd": float(np.std(depth, ddof=0)),
                "depth_median": float(np.median(depth)),
                "depth_q1": float(np.quantile(depth, 0.25)),
                "depth_q3": float(np.quantile(depth, 0.75)),
                "depth_min": float(np.min(depth)),
                "depth_max": float(np.max(depth)),
                "llbce_mean": float(np.mean(llbce)),
                "llbce_sd": float(np.std(llbce, ddof=0)),
                "llbce_median": float(np.median(llbce)),
                "llbce_q1": float(np.quantile(llbce, 0.25)),
                "llbce_q3": float(np.quantile(llbce, 0.75)),
                "llbce_min": float(np.min(llbce)),
                "llbce_max": float(np.max(llbce)),
            }
        )
        return row


def plot_contours(results: list[dict[str, object]], output_path: str | Path | None = None):
    fig, axes = plt.subplots(1, len(results), figsize=(5.6 * len(results), 4.8))
    axes = np.atleast_1d(axes)
    for axis, result in zip(axes, results):
        contour = axis.contourf(
            result["depth_grid"],
            result["llbce_grid"],
            result["target_probability"],
            levels=16,
            cmap="RdYlBu_r",
        )
        if result["success"] and np.any(result["stable_mask"]) and not np.all(result["stable_mask"]):
            axis.contour(
                result["depth_grid"],
                result["llbce_grid"],
                result["stable_mask"].astype(int),
                levels=[0.5],
                linewidths=2,
                colors="black",
            )
        best = result["best_position"]
        axis.scatter(
            result["depth_grid"][best],
            result["llbce_grid"][best],
            marker="*",
            s=120,
            edgecolor="black",
            linewidth=0.8,
        )
        axis.set_title(
            f"Type {result['class_label']}\n"
            + (result["domain"]["name"] if result["success"] else "No supported recommendation")
        )
        axis.set_xlabel("Depth of A (mm)")
        axis.set_ylabel("LLBCE (mm)")
        fig.colorbar(contour, ax=axis, label=f"P(Type {result['class_label']})")
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_surfaces(results: list[dict[str, object]], output_path: str | Path | None = None):
    fig = plt.figure(figsize=(6 * len(results), 5.2))
    for index, result in enumerate(results, start=1):
        axis = fig.add_subplot(1, len(results), index, projection="3d")
        surface = axis.plot_surface(
            result["depth_grid"],
            result["llbce_grid"],
            result["target_probability"],
            cmap="RdYlBu_r",
            linewidth=0,
            antialiased=True,
            alpha=0.9,
        )
        best = result["best_position"]
        axis.scatter(
            result["depth_grid"][best],
            result["llbce_grid"][best],
            result["target_probability"][best],
            marker="*",
            s=120,
            edgecolor="black",
        )
        axis.set_title(f"Type {result['class_label']}")
        axis.set_xlabel("Depth of A (mm)")
        axis.set_ylabel("LLBCE (mm)")
        axis.set_zlabel(f"P(Type {result['class_label']})")
        axis.set_zlim(0, 1)
        fig.colorbar(surface, ax=axis, shrink=0.62, pad=0.08)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    return fig
