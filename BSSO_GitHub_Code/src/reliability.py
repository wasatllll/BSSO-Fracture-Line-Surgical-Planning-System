from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa


def icc_2k(values: np.ndarray) -> float:
    """Two-way random-effects, absolute-agreement, average-measures ICC(2,k)."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("ICC requires at least two targets and two raters.")
    if np.isnan(matrix).any():
        raise ValueError("ICC input contains missing values.")
    n_targets, n_raters = matrix.shape
    grand_mean = matrix.mean()
    target_means = matrix.mean(axis=1)
    rater_means = matrix.mean(axis=0)
    ss_targets = n_raters * np.sum((target_means - grand_mean) ** 2)
    ss_raters = n_targets * np.sum((rater_means - grand_mean) ** 2)
    residual = matrix - target_means[:, None] - rater_means[None, :] + grand_mean
    ss_error = np.sum(residual**2)
    ms_targets = ss_targets / (n_targets - 1)
    ms_raters = ss_raters / (n_raters - 1)
    ms_error = ss_error / ((n_targets - 1) * (n_raters - 1))
    denominator = ms_targets + (ms_raters - ms_error) / n_targets
    return float((ms_targets - ms_error) / denominator) if denominator != 0 else np.nan


def bootstrap_icc_2k(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> dict[str, float | int]:
    matrix = np.asarray(values, dtype=float)
    point = icc_2k(matrix)
    rng = np.random.default_rng(random_state)
    estimates = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, matrix.shape[0], matrix.shape[0])
        try:
            estimates.append(icc_2k(matrix[indices]))
        except Exception:
            continue
    array = np.asarray(estimates, dtype=float)
    return {
        "icc_2k": point,
        "ci_lower": float(np.quantile(array, 0.025)) if len(array) else np.nan,
        "ci_upper": float(np.quantile(array, 0.975)) if len(array) else np.nan,
        "bootstrap_successful": int(len(array)),
    }


def fleiss_kappa_wide(ratings: pd.DataFrame) -> float:
    """Fleiss' kappa for a wide target-by-rater categorical rating table."""
    if ratings.isna().any().any():
        raise ValueError("Categorical rating table contains missing values.")
    # Encode jointly so that the same category has the same code across raters.
    categories = sorted(pd.unique(ratings.astype(str).to_numpy().ravel()), key=str)
    mapping = {category: index for index, category in enumerate(categories)}
    integer_ratings = ratings.astype(str).apply(lambda column: column.map(mapping)).to_numpy(dtype=int)
    count_table, _ = aggregate_raters(integer_ratings, n_cat=len(categories))
    return float(fleiss_kappa(count_table, method="fleiss"))


def run_reliability_analysis(
    continuous_tables: dict[str, pd.DataFrame],
    categorical_table: pd.DataFrame | None,
    output_dir: str | Path,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    continuous_rows = []
    for index, (measurement, table) in enumerate(continuous_tables.items()):
        numeric = table.apply(pd.to_numeric, errors="coerce").dropna()
        result = bootstrap_icc_2k(
            numeric.to_numpy(),
            n_bootstrap=n_bootstrap,
            random_state=random_state + index,
        )
        continuous_rows.append(
            {
                "measurement": measurement,
                "n_targets": int(len(numeric)),
                "n_raters": int(numeric.shape[1]),
                "model": "ICC(2,k)",
                **result,
            }
        )
    continuous_output = pd.DataFrame(continuous_rows)
    continuous_output.to_csv(output_dir / "continuous_measurement_icc.csv", index=False)

    categorical_output = None
    if categorical_table is not None:
        categorical_clean = categorical_table.dropna().copy()
        categorical_output = {
            "n_targets": int(len(categorical_clean)),
            "n_raters": int(categorical_clean.shape[1]),
            "statistic": "Fleiss kappa",
            "kappa": fleiss_kappa_wide(categorical_clean),
        }
        pd.DataFrame([categorical_output]).to_csv(
            output_dir / "categorical_interobserver_kappa.csv", index=False
        )
    return {
        "continuous": continuous_output,
        "categorical": categorical_output,
    }
