from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import (
    COLUMN_ALIASES,
    CONTINUOUS_CANDIDATES,
    PATIENT_ID_COL,
    SAMPLE_COL,
    SIDE_COL,
    TARGET_COL,
)


def _normalize_token(value: object) -> str:
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
    )


def _find_column(columns: Iterable[object], canonical_name: str) -> str | None:
    normalized = {_normalize_token(c): str(c) for c in columns}
    candidates = [canonical_name] + COLUMN_ALIASES.get(canonical_name, [])
    for name in candidates:
        key = _normalize_token(name)
        if key in normalized:
            return normalized[key]
    return None


def clean_class_label(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    for prefix in ("TYPE", "Type", "type", "LSS", "lss"):
        text = text.replace(prefix, "")
    text = text.replace(" ", "")
    try:
        numeric = float(text)
        return str(int(numeric)) if numeric.is_integer() else str(numeric)
    except Exception:
        return text


def _binary_map(series: pd.Series, positive: set[str], negative: set[str], name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {x: 1 for x in positive} | {x: 0 for x in negative}
    out = normalized.map(mapping)
    unknown = sorted(normalized[out.isna()].dropna().unique().tolist())
    if unknown:
        raise ValueError(f"Unrecognized values in {name}: {unknown}")
    return out.astype(int)


def canonicalize_dataframe(raw: pd.DataFrame, require_all_candidates: bool = False) -> pd.DataFrame:
    """Return a copy with canonical column names and encoded categorical predictors."""
    df = raw.copy()
    df.columns = [str(c).replace("\u00a0", " ").replace("\t", " ").strip() for c in df.columns]

    canonical_names = [
        TARGET_COL,
        PATIENT_ID_COL,
        SAMPLE_COL,
        SIDE_COL,
        *CONTINUOUS_CANDIDATES,
        "sex",
        "type of jaw deformity",
        "third molar presence",
    ]
    rename_map: dict[str, str] = {}
    for canonical in canonical_names:
        actual = _find_column(df.columns, canonical)
        if actual is not None and actual != canonical:
            rename_map[actual] = canonical
    df = df.rename(columns=rename_map)

    if TARGET_COL not in df.columns:
        raise ValueError("The outcome column could not be identified.")
    df[TARGET_COL] = df[TARGET_COL].map(clean_class_label)

    for col in CONTINUOUS_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "sex" in df.columns:
        df["sex_male"] = _binary_map(
            df["sex"],
            positive={"male", "m", "1", "man", "\u7537"},
            negative={"female", "f", "0", "woman", "\u5973"},
            name="sex",
        )
    if "type of jaw deformity" in df.columns:
        jaw_raw = df["type of jaw deformity"].astype(str).str.strip().str.lower()
        jaw_numeric = pd.to_numeric(jaw_raw, errors="coerce")
        jaw_map = {
            "ii": 2, "class ii": 2, "skeletal class ii": 2, "\u9aa8\u6027ii\u7c7b": 2, "\u9aa8\u60272\u7c7b": 2,
            "iii": 3, "class iii": 3, "skeletal class iii": 3, "\u9aa8\u6027iii\u7c7b": 3, "\u9aa8\u60273\u7c7b": 3,
        }
        jaw = jaw_numeric.fillna(jaw_raw.map(jaw_map))
        if jaw.isna().any():
            unknown = sorted(jaw_raw[jaw.isna()].unique().tolist())
            raise ValueError(f"Unrecognized jaw-deformity values: {unknown}")
        df["jaw_deformity_type3"] = jaw.eq(3).astype(int)
    if "third molar presence" in df.columns:
        df["third_molar_yes"] = _binary_map(
            df["third molar presence"],
            positive={"yes", "y", "present", "1", "\u6709", "\u662f"},
            negative={"no", "n", "absent", "0", "\u65e0", "\u5426"},
            name="third molar presence",
        )

    if PATIENT_ID_COL not in df.columns:
        if SAMPLE_COL not in df.columns:
            raise ValueError(
                "A de-identified patient_id column is required. A sample column may be used as a fallback."
            )
        sample_numeric = pd.to_numeric(df[SAMPLE_COL], errors="raise").astype(int)
        df[PATIENT_ID_COL] = ((sample_numeric - 1) // 2 + 1).astype(str)
    else:
        df[PATIENT_ID_COL] = df[PATIENT_ID_COL].astype(str)

    if SIDE_COL not in df.columns:
        if SAMPLE_COL in df.columns:
            sample_numeric = pd.to_numeric(df[SAMPLE_COL], errors="coerce")
            df[SIDE_COL] = np.where(sample_numeric.mod(2).eq(1), "right", "left")
        else:
            df[SIDE_COL] = df.groupby(PATIENT_ID_COL).cumcount().map({0: "right", 1: "left"}).fillna("unknown")

    required = [TARGET_COL, PATIENT_ID_COL]
    if require_all_candidates:
        required += CONTINUOUS_CANDIDATES + [
            "sex_male", "jaw_deformity_type3", "third_molar_yes"
        ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after canonicalization: {missing}")

    return df


def read_cohort(path: str | Path, sheet_name: int | str = 0, require_all_candidates: bool = False) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, sheet_name=sheet_name)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raise ValueError("Supported input formats are .xlsx, .xls, and .csv.")
    return canonicalize_dataframe(raw, require_all_candidates=require_all_candidates)


def validate_bilateral_structure(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    counts = df.groupby(PATIENT_ID_COL).size().rename("n_sides").reset_index()
    invalid = counts[counts["n_sides"] != 2]
    if strict and not invalid.empty:
        raise ValueError(
            "Every patient must contribute exactly two mandibular sides. "
            f"Invalid patient counts: {invalid.to_dict(orient='records')[:10]}"
        )
    return counts


def representative_patient_label(group: pd.DataFrame) -> str:
    labels = group[TARGET_COL].astype(str)
    modes = labels.mode()
    if not modes.empty:
        return str(modes.iloc[0])
    return str(labels.iloc[0])


@dataclass(frozen=True)
class CohortSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    train_patient_ids: list[str]
    test_patient_ids: list[str]


def patient_level_stratified_split(
    df: pd.DataFrame,
    test_fraction: float = 0.30,
    random_state: int = 70,
) -> CohortSplit:
    """Replicate the manuscript's 7:3 patient-level stratified split."""
    patient_info = (
        df.groupby(PATIENT_ID_COL, sort=True)[TARGET_COL]
        .apply(lambda labels: str(labels.mode().iloc[0]) if not labels.mode().empty else str(labels.iloc[0]))
        .rename("fracture_type")
        .reset_index()
    )

    rng = np.random.RandomState(random_state)
    target_test_size = int(len(patient_info) * test_fraction)
    proportions = patient_info["fracture_type"].value_counts(normalize=True).to_dict()

    train_ids: list[str] = []
    test_ids: list[str] = []
    # Preserve the legacy class order and random-number generator used in the original analysis.
    for label in patient_info["fracture_type"].unique():
        ids = patient_info.loc[patient_info["fracture_type"] == label, PATIENT_ID_COL].tolist()
        n_test = int(target_test_size * proportions[label])
        n_test = max(1, n_test) if len(ids) > 1 else len(ids)
        selected = set(rng.choice(ids, size=n_test, replace=False).tolist())
        test_ids.extend([x for x in ids if x in selected])
        train_ids.extend([x for x in ids if x not in selected])

    train = df[df[PATIENT_ID_COL].isin(train_ids)].copy().reset_index(drop=True)
    test = df[df[PATIENT_ID_COL].isin(test_ids)].copy().reset_index(drop=True)
    if set(train[PATIENT_ID_COL]) & set(test[PATIENT_ID_COL]):
        raise RuntimeError("Patient leakage detected between training and internal test sets.")
    return CohortSplit(train=train, test=test, train_patient_ids=train_ids, test_patient_ids=test_ids)


def save_dataset_manifest(
    cohorts: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, df in cohorts.items():
        row: dict[str, object] = {
            "cohort": name,
            "patients": int(df[PATIENT_ID_COL].nunique()),
            "sides": int(len(df)),
        }
        for label, count in df[TARGET_COL].value_counts().sort_index().items():
            row[f"type_{label}_sides"] = int(count)
        rows.append(row)
    manifest = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest
