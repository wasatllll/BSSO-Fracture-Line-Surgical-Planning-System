from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import clean_class_label


ALIASES = {
    "patient_id": ["patient_id", "patient", "case_id", "subject_id"],
    "sample": ["sample", "sample_id", "side_id"],
    "predicted_type": ["predicted_type", "Predicted TYPE", "planned_type", "target_type"],
    "observed_type": ["observed_type", "TRUE TYPE", "true_type", "fracture_type"],
    "planned_depth": [
        "planned_depth",
        "Guide-planned Depth of A value(mm)",
        "guide_planned_depth_of_a",
    ],
    "observed_depth": ["observed_depth", "Actual Depth of A(mm)", "actual_depth_of_a"],
    "planned_llbce": [
        "planned_llbce",
        "Guide-planned LLBCE value(mm)",
        "guide_planned_llbce",
    ],
    "observed_llbce": ["observed_llbce", "Actual LLBCE(mm)", "actual_llbce"],
}


def _normalize(value: object) -> str:
    return (
        str(value)
        .replace("\u00a0", " ")
        .replace("\t", " ")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace(".", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
    )


def _find(columns, canonical: str) -> str | None:
    normalized = {_normalize(column): str(column) for column in columns}
    for candidate in [canonical] + ALIASES.get(canonical, []):
        key = _normalize(candidate)
        if key in normalized:
            return normalized[key]
    return None


def read_feasibility_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raise ValueError("Feasibility data must be .xlsx, .xls, or .csv.")

    rename = {}
    for canonical in ALIASES:
        actual = _find(raw.columns, canonical)
        if actual is not None:
            rename[actual] = canonical
    data = raw.rename(columns=rename).copy()
    required = [
        "predicted_type",
        "observed_type",
        "planned_depth",
        "observed_depth",
        "planned_llbce",
        "observed_llbce",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Feasibility data are missing columns: {missing}")

    if "patient_id" not in data.columns:
        if "sample" not in data.columns:
            raise ValueError("Feasibility data require patient_id or sample.")
        sample = pd.to_numeric(data["sample"], errors="raise").astype(int)
        data["patient_id"] = ((sample - 1) // 2 + 1).astype(str)
    else:
        data["patient_id"] = data["patient_id"].astype(str)

    data["predicted_type"] = data["predicted_type"].map(clean_class_label)
    data["observed_type"] = data["observed_type"].map(clean_class_label)
    for column in ["planned_depth", "observed_depth", "planned_llbce", "observed_llbce"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=required).reset_index(drop=True)


def analyze_feasibility(data: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = data.copy()
    analysis["pattern_concordant"] = analysis["predicted_type"] == analysis["observed_type"]
    analysis["depth_absolute_deviation_mm"] = (
        analysis["observed_depth"] - analysis["planned_depth"]
    ).abs()
    analysis["llbce_absolute_deviation_mm"] = (
        analysis["observed_llbce"] - analysis["planned_llbce"]
    ).abs()
    analysis.to_csv(output_dir / "clinical_feasibility_side_level_results.csv", index=False)

    def summarize(values: pd.Series, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            f"{prefix}_median": float(values.median()),
            f"{prefix}_q1": float(values.quantile(0.25)),
            f"{prefix}_q3": float(values.quantile(0.75)),
            f"{prefix}_minimum": float(values.min()),
            f"{prefix}_maximum": float(values.max()),
        }

    summary: dict[str, object] = {
        "patients": int(analysis["patient_id"].nunique()),
        "sides": int(len(analysis)),
        "concordant_sides": int(analysis["pattern_concordant"].sum()),
        "pattern_concordance": float(analysis["pattern_concordant"].mean()),
    }
    summary.update(
        summarize(analysis["depth_absolute_deviation_mm"], "depth_absolute_deviation_mm")
    )
    summary.update(
        summarize(analysis["llbce_absolute_deviation_mm"], "llbce_absolute_deviation_mm")
    )
    pd.DataFrame([summary]).to_csv(
        output_dir / "clinical_feasibility_summary.csv", index=False
    )
    return {"summary": summary, "side_level": analysis}
