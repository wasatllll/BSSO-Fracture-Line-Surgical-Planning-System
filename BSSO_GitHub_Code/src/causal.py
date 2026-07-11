from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .constants import (
    BINARY_CANDIDATES,
    CONTINUOUS_CANDIDATES,
    FINAL_FEATURES,
    PATIENT_ID_COL,
    TARGET_COL,
)


def e_value_rr(rr: float) -> float:
    rr = float(rr)
    if rr < 1:
        rr = 1 / rr
    if rr <= 1:
        return 1.0
    return float(rr + np.sqrt(rr * (rr - 1)))


def e_values_from_rr(rr: float, lower: float, upper: float) -> tuple[float, float]:
    point = e_value_rr(rr)
    bound = lower if rr >= 1 else upper
    if (rr >= 1 and bound <= 1) or (rr < 1 and bound >= 1):
        return point, 1.0
    return point, e_value_rr(bound)


def _gee_formula_columns(model_name: str) -> list[str]:
    if model_name == "M1":
        return FINAL_FEATURES
    if model_name == "M4":
        return CONTINUOUS_CANDIDATES + BINARY_CANDIDATES
    raise KeyError(model_name)


def fit_gee_models(
    development_df: pd.DataFrame,
    output_dir: str | Path,
    delta_mm: float = 0.5,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    needed = [PATIENT_ID_COL, TARGET_COL] + CONTINUOUS_CANDIDATES + BINARY_CANDIDATES
    data = development_df[needed].dropna().copy().reset_index(drop=True)
    classes = sorted(data[TARGET_COL].astype(str).unique(), key=str)

    rows = []
    for model_name in ["M1", "M4"]:
        covariates = _gee_formula_columns(model_name)
        for outcome_class in classes:
            outcome = (data[TARGET_COL].astype(str) == outcome_class).astype(int)
            design = sm.add_constant(data[covariates], has_constant="add")
            fit = sm.GEE(
                endog=outcome,
                exog=design,
                groups=data[PATIENT_ID_COL],
                family=sm.families.Poisson(),
                cov_struct=sm.cov_struct.Independence(),
            ).fit()
            for exposure in ["LLBCE", "Depth.of.A"]:
                beta = float(fit.params[exposure])
                standard_error = float(fit.bse[exposure])
                beta_delta = beta * delta_mm
                se_delta = standard_error * delta_mm
                rr = float(np.exp(beta_delta))
                lower = float(np.exp(beta_delta - 1.96 * se_delta))
                upper = float(np.exp(beta_delta + 1.96 * se_delta))
                point_e, ci_e = e_values_from_rr(rr, lower, upper)
                rows.append(
                    {
                        "outcome_class": outcome_class,
                        "exposure": exposure,
                        "model": model_name,
                        "effect_unit_mm": delta_mm,
                        "rr": rr,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "p_value": float(fit.pvalues[exposure]),
                        "e_value": point_e,
                        "e_value_ci": ci_e,
                        "n_sides": int(len(data)),
                        "n_patients": int(data[PATIENT_ID_COL].nunique()),
                        "covariates": ", ".join(covariates),
                    }
                )
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "gee_m1_m4_results.csv", index=False)
    return results


def _gcomp_pipeline(features: list[str], random_state: int) -> Pipeline:
    continuous = [feature for feature in features if feature in CONTINUOUS_CANDIDATES]
    binary = [feature for feature in features if feature in BINARY_CANDIDATES]
    transformers = []
    if continuous:
        transformers.append(("continuous", StandardScaler(), continuous))
    if binary:
        transformers.append(("binary", "passthrough", binary))
    preprocessor = ColumnTransformer(transformers)
    classifier = LogisticRegression(max_iter=5000, random_state=random_state)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _patient_bootstrap_sample(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    patient_ids = pd.unique(data[PATIENT_ID_COL])
    sampled_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
    blocks = []
    for bootstrap_id, patient_id in enumerate(sampled_ids):
        block = data[data[PATIENT_ID_COL] == patient_id].copy()
        block["_bootstrap_patient_id"] = bootstrap_id
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def _gcomp_difference(
    data: pd.DataFrame,
    features: list[str],
    exposure: str,
    delta: float,
    encoder: LabelEncoder,
    random_state: int,
) -> np.ndarray:
    model = _gcomp_pipeline(features, random_state)
    y = encoder.transform(data[TARGET_COL].astype(str))
    model.fit(data[features], y)
    natural = data[features].copy()
    intervened = natural.copy()
    intervened[exposure] = intervened[exposure] + delta
    natural_probability = model.predict_proba(natural)
    intervened_probability = model.predict_proba(intervened)
    return np.mean(intervened_probability - natural_probability, axis=0)


def run_g_computation(
    development_df: pd.DataFrame,
    output_dir: str | Path,
    delta_mm: float = 0.5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    needed = [PATIENT_ID_COL, TARGET_COL] + CONTINUOUS_CANDIDATES + BINARY_CANDIDATES
    data = development_df[needed].dropna().copy().reset_index(drop=True)
    encoder = LabelEncoder()
    encoder.fit(data[TARGET_COL].astype(str))
    rng = np.random.default_rng(random_state)

    rows = []
    for model_name in ["M1", "M4"]:
        features = _gee_formula_columns(model_name)
        for exposure in ["LLBCE", "Depth.of.A"]:
            for delta in [delta_mm, -delta_mm]:
                point = _gcomp_difference(data, features, exposure, delta, encoder, random_state)
                bootstrap_values = []
                for iteration in range(n_bootstrap):
                    sample = _patient_bootstrap_sample(data, rng)
                    try:
                        value = _gcomp_difference(
                            sample,
                            features,
                            exposure,
                            delta,
                            encoder,
                            random_state + iteration + 1,
                        )
                        bootstrap_values.append(value)
                    except Exception:
                        continue
                bootstrap_array = np.asarray(bootstrap_values)
                if len(bootstrap_array) == 0:
                    lower = np.full_like(point, np.nan)
                    upper = np.full_like(point, np.nan)
                else:
                    lower = np.quantile(bootstrap_array, 0.025, axis=0)
                    upper = np.quantile(bootstrap_array, 0.975, axis=0)

                for class_index, class_label in enumerate(encoder.classes_):
                    rows.append(
                        {
                            "model": model_name,
                            "intervention": f"{exposure} {delta:+.1f} mm",
                            "exposure": exposure,
                            "delta_mm": delta,
                            "outcome_class": str(class_label),
                            "mean_probability_difference": float(point[class_index]),
                            "ci_lower": float(lower[class_index]),
                            "ci_upper": float(upper[class_index]),
                            "mean_probability_difference_percent": float(point[class_index] * 100),
                            "ci_lower_percent": float(lower[class_index] * 100),
                            "ci_upper_percent": float(upper[class_index] * 100),
                            "n_bootstrap_successful": int(len(bootstrap_array)),
                            "covariates": ", ".join(features),
                        }
                    )
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "g_computation_m1_m4_results.csv", index=False)
    return results


def plot_gee(gee_results: pd.DataFrame, output_path: str | Path) -> None:
    data = gee_results.copy()
    data = data[data["model"].isin(["M1", "M4"])]
    labels = []
    for class_label in sorted(data["outcome_class"].unique(), key=str):
        for exposure in ["Depth.of.A", "LLBCE"]:
            labels.append((class_label, exposure))
    y_positions = np.arange(len(labels))[::-1]

    fig, axis = plt.subplots(figsize=(9.5, max(5, len(labels) * 0.65)))
    offsets = {"M1": -0.12, "M4": 0.12}
    for model_name in ["M1", "M4"]:
        xs, lows, highs, ys = [], [], [], []
        for y_position, (class_label, exposure) in zip(y_positions, labels):
            row = data[
                (data["model"] == model_name)
                & (data["outcome_class"].astype(str) == str(class_label))
                & (data["exposure"] == exposure)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            xs.append(row["rr"])
            lows.append(row["rr"] - row["ci_lower"])
            highs.append(row["ci_upper"] - row["rr"])
            ys.append(y_position + offsets[model_name])
        axis.errorbar(
            xs,
            ys,
            xerr=[lows, highs],
            fmt="o",
            capsize=4,
            linewidth=1.8,
            label=model_name,
        )
    axis.axvline(1, linestyle="--", linewidth=1)
    axis.set_yticks(y_positions)
    axis.set_yticklabels([f"Type {class_label} | {exposure}" for class_label, exposure in labels])
    axis.set_xlabel("Relative risk per 0.5-mm increase")
    axis.set_title("DAG-informed patient-clustered GEE analysis")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_g_computation(results: pd.DataFrame, output_path: str | Path) -> None:
    data = results.copy()
    interventions = [
        "LLBCE +0.5 mm",
        "LLBCE -0.5 mm",
        "Depth.of.A +0.5 mm",
        "Depth.of.A -0.5 mm",
    ]
    labels = []
    for intervention in interventions:
        if intervention in set(data["intervention"]):
            for class_label in sorted(data["outcome_class"].unique(), key=str):
                labels.append((intervention, class_label))
    y_positions = np.arange(len(labels))[::-1]
    offsets = {"M1": -0.12, "M4": 0.12}

    fig, axis = plt.subplots(figsize=(10, max(6, len(labels) * 0.48)))
    for model_name in ["M1", "M4"]:
        xs, lows, highs, ys = [], [], [], []
        for y_position, (intervention, class_label) in zip(y_positions, labels):
            row = data[
                (data["model"] == model_name)
                & (data["intervention"] == intervention)
                & (data["outcome_class"].astype(str) == str(class_label))
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            point = row["mean_probability_difference_percent"]
            xs.append(point)
            lows.append(point - row["ci_lower_percent"])
            highs.append(row["ci_upper_percent"] - point)
            ys.append(y_position + offsets[model_name])
        axis.errorbar(
            xs,
            ys,
            xerr=[lows, highs],
            fmt="o",
            capsize=4,
            linewidth=1.8,
            label=model_name,
        )
    axis.axvline(0, linestyle="--", linewidth=1)
    axis.set_yticks(y_positions)
    axis.set_yticklabels([f"{intervention} | Type {class_label}" for intervention, class_label in labels])
    axis.set_xlabel("Average probability difference (percentage points)")
    axis.set_title("Parametric g-computation")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_causal_analysis(
    development_df: pd.DataFrame,
    output_dir: str | Path,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dag(output_dir / "supplementary_figure_s14_dag.png")
    gee = fit_gee_models(development_df, output_dir)
    gcomp = run_g_computation(
        development_df,
        output_dir,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    plot_gee(gee, output_dir / "gee_m1_m4_forest.png")
    plot_g_computation(gcomp, output_dir / "g_computation_m1_m4_forest.png")
    with open(output_dir / "causal_analysis_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "analysis_scope": "exploratory",
                "cluster_unit": PATIENT_ID_COL,
                "exposures": ["LLBCE", "Depth.of.A"],
                "effect_unit_mm": 0.5,
                "g_computation_bootstrap_unit": "patient",
                "g_computation_bootstrap_iterations": n_bootstrap,
            },
            handle,
            indent=2,
        )
    return {"gee": gee, "g_computation": gcomp}


def plot_dag(output_path: str | Path) -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, axis = plt.subplots(figsize=(14, 8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def box(x, y, width, height, text, facecolor):
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="black",
            facecolor=facecolor,
        )
        axis.add_patch(patch)
        axis.text(x, y, text, ha="center", va="center", fontsize=10.5)

    def arrow(x1, y1, x2, y2, curvature=0.0):
        axis.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "->",
                "lw": 1.5,
                "color": "black",
                "connectionstyle": f"arc3,rad={curvature}",
            },
        )

    positions = {
        "patient": (0.13, 0.72),
        "anatomy": (0.40, 0.72),
        "modifiable": (0.63, 0.50),
        "outcome": (0.86, 0.50),
        "surgeon": (0.29, 0.23),
        "unmeasured": (0.63, 0.15),
    }
    box(*positions["patient"], 0.18, 0.13, "Patient factors\n(age, sex, jaw deformity,\nthird molar)", "#f4f4f4")
    box(*positions["anatomy"], 0.25, 0.12, "Baseline mandibular anatomy\n(MRT, PMBT, RAPL, ART, RH, LSND)", "#f4f4f4")
    box(*positions["modifiable"], 0.23, 0.12, "Modifiable osteotomy parameters\nLLBCE and Depth of A", "#eaf2ff")
    box(*positions["outcome"], 0.18, 0.10, "Fracture-line type\n(Type 1 / Type 2 / Type 3)", "#fff2e8")
    box(*positions["surgeon"], 0.21, 0.10, "Surgeon / center /\ninstrument / experience", "#f4f4f4")
    box(*positions["unmeasured"], 0.30, 0.10, "Unmeasured intraoperative factors\n(bone quality, splitting force,\nchisel direction, stress release)", "#f4f4f4")

    arrow(0.22, 0.72, 0.275, 0.72)
    arrow(0.20, 0.66, 0.52, 0.53, -0.08)
    arrow(0.19, 0.78, 0.76, 0.55, -0.18)
    arrow(0.49, 0.68, 0.55, 0.55)
    arrow(0.50, 0.77, 0.77, 0.55, -0.08)
    arrow(0.745, 0.50, 0.77, 0.50)
    arrow(0.39, 0.27, 0.53, 0.44)
    arrow(0.39, 0.27, 0.77, 0.45, 0.12)
    arrow(0.63, 0.20, 0.63, 0.44)
    arrow(0.72, 0.19, 0.78, 0.45)

    axis.text(
        0.5,
        0.95,
        "Directed acyclic graph for exploratory causal framework",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.04,
        "Constructed a priori from clinical knowledge and mechanistic assumptions; "
        "used to define measured and potential unmeasured confounding structures.",
        ha="center",
        va="center",
        fontsize=9.5,
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
