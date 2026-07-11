from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, kruskal, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from .constants import CONTINUOUS_CANDIDATES, TARGET_COL


DEFAULT_CATEGORICAL_VARIABLES = [
    "sex",
    "type of jaw deformity",
    "third molar presence",
]


def _sort_levels(values) -> list[str]:
    def key(value):
        try:
            return (0, float(value))
        except Exception:
            return (1, str(value))

    return [str(value) for value in sorted(pd.unique(values), key=key)]


def _median_iqr(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    return (
        f"{numeric.median():.2f} "
        f"[{numeric.quantile(0.25):.2f}–{numeric.quantile(0.75):.2f}]"
    )


def _count_percent(count: int, denominator: int) -> str:
    if denominator <= 0:
        return "0 (0.0%)"
    return f"{int(count)} ({100 * count / denominator:.1f}%)"


def _dunn_pairwise(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    adjust_method: str = "bonferroni",
) -> pd.DataFrame:
    frame = data[[value_col, group_col]].dropna().copy()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna()
    groups = _sort_levels(frame[group_col].astype(str))
    if len(groups) < 2:
        return pd.DataFrame()

    frame[group_col] = frame[group_col].astype(str)
    frame["rank"] = stats.rankdata(frame[value_col].to_numpy())
    n_total = len(frame)
    _, tie_counts = np.unique(frame[value_col].to_numpy(), return_counts=True)
    tie_correction = (
        1 - np.sum(tie_counts**3 - tie_counts) / (n_total**3 - n_total)
        if n_total > 1
        else 1.0
    )
    tie_correction = max(float(tie_correction), 1e-12)

    rows = []
    for group_a, group_b in itertools.combinations(groups, 2):
        a = frame[frame[group_col] == group_a]
        b = frame[frame[group_col] == group_b]
        if a.empty or b.empty:
            continue
        standard_error = np.sqrt(
            n_total
            * (n_total + 1)
            / 12
            * (1 / len(a) + 1 / len(b))
            * tie_correction
        )
        if standard_error <= 0:
            z_statistic = np.nan
            p_value = np.nan
        else:
            z_statistic = (a["rank"].mean() - b["rank"].mean()) / standard_error
            p_value = 2 * stats.norm.sf(abs(z_statistic))
        rows.append(
            {
                "comparison": f"Type {group_a} vs Type {group_b}",
                "test": "Dunn test",
                "statistic": z_statistic,
                "raw_p": p_value,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        valid = result["raw_p"].notna()
        result.loc[valid, "adjusted_p"] = multipletests(
            result.loc[valid, "raw_p"], method=adjust_method
        )[1]
        result["adjust_method"] = adjust_method
    return result


def continuous_baseline_table(
    df: pd.DataFrame,
    variables: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Continuous-variable table with group-wise Shapiro testing and post hoc comparisons."""
    variables = variables or CONTINUOUS_CANDIDATES
    classes = _sort_levels(df[TARGET_COL].dropna().astype(str))
    main_rows = []
    posthoc_rows = []

    for variable in variables:
        if variable not in df.columns:
            continue
        valid = df[[variable, TARGET_COL]].dropna().copy()
        valid[variable] = pd.to_numeric(valid[variable], errors="coerce")
        valid[TARGET_COL] = valid[TARGET_COL].astype(str)
        valid = valid.dropna()
        groups = [
            valid.loc[valid[TARGET_COL] == class_label, variable].dropna()
            for class_label in classes
        ]
        if any(len(group) == 0 for group in groups):
            continue

        shapiro_values = []
        all_normal = True
        for class_label, group in zip(classes, groups):
            if len(group) < 3:
                p_value = np.nan
                all_normal = False
            else:
                try:
                    p_value = float(shapiro(group).pvalue)
                    all_normal = all_normal and p_value >= 0.05
                except Exception:
                    p_value = np.nan
                    all_normal = False
            shapiro_values.append((class_label, p_value))

        if all_normal:
            statistic, p_value = stats.f_oneway(*groups)
            test_name = "one-way ANOVA"
            tukey = pairwise_tukeyhsd(
                endog=valid[variable], groups=valid[TARGET_COL], alpha=0.05
            )
            posthoc = pd.DataFrame(
                tukey.summary().data[1:], columns=tukey.summary().data[0]
            )
            for _, row in posthoc.iterrows():
                posthoc_rows.append(
                    {
                        "variable": variable,
                        "comparison": f"Type {row['group1']} vs Type {row['group2']}",
                        "test": "Tukey HSD",
                        "statistic": np.nan,
                        "raw_p": np.nan,
                        "adjusted_p": float(row["p-adj"]),
                        "adjust_method": "Tukey HSD",
                    }
                )
        else:
            statistic, p_value = kruskal(*groups)
            test_name = "Kruskal-Wallis"
            posthoc = _dunn_pairwise(valid, variable, TARGET_COL)
            if not posthoc.empty:
                posthoc.insert(0, "variable", variable)
                posthoc_rows.extend(posthoc.to_dict(orient="records"))

        relevant_posthoc = [row for row in posthoc_rows if row["variable"] == variable]
        pairwise_text = "; ".join(
            f"{row['comparison']}: adjusted P={row['adjusted_p']:.3f}"
            for row in relevant_posthoc
            if pd.notna(row.get("adjusted_p"))
        )
        main_row: dict[str, object] = {
            "variable": variable,
            "overall": _median_iqr(valid[variable]),
            "test": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "pairwise_adjusted_p": pairwise_text,
            "normality_rule": "all outcome groups Shapiro-Wilk P >= 0.05",
        }
        for class_label, shapiro_p in shapiro_values:
            main_row[f"type_{class_label}"] = _median_iqr(
                valid.loc[valid[TARGET_COL] == class_label, variable]
            )
            main_row[f"shapiro_p_type_{class_label}"] = shapiro_p
        main_rows.append(main_row)

    return pd.DataFrame(main_rows), pd.DataFrame(posthoc_rows)


def _expected_counts_adequate(expected: np.ndarray) -> bool:
    expected = np.asarray(expected, dtype=float)
    return bool(np.all(expected >= 1) and np.mean(expected >= 5) >= 0.80)


def _chi_square_statistic(table: pd.DataFrame) -> float:
    statistic, _, _, _ = chi2_contingency(table, correction=False)
    return float(statistic)


def _monte_carlo_chi_square(
    category: pd.Series,
    outcome: pd.Series,
    n_permutations: int,
    random_state: int,
) -> tuple[float, float, int]:
    frame = pd.DataFrame({"category": category, "outcome": outcome}).dropna()
    observed = pd.crosstab(frame["category"], frame["outcome"])
    observed_statistic = _chi_square_statistic(observed)
    rng = np.random.default_rng(random_state)
    simulated = []
    category_values = frame["category"].to_numpy()
    outcome_values = frame["outcome"].to_numpy()
    for _ in range(n_permutations):
        permuted = rng.permutation(outcome_values)
        table = pd.crosstab(category_values, permuted).reindex(
            index=observed.index, columns=observed.columns, fill_value=0
        )
        try:
            simulated.append(_chi_square_statistic(table))
        except Exception:
            continue
    simulated_array = np.asarray(simulated, dtype=float)
    p_value = (
        (np.sum(simulated_array >= observed_statistic) + 1)
        / (len(simulated_array) + 1)
        if len(simulated_array)
        else np.nan
    )
    return observed_statistic, float(p_value), int(len(simulated_array))


def _categorical_test(
    frame: pd.DataFrame,
    variable: str,
    n_permutations: int,
    random_state: int,
) -> dict[str, object]:
    table = pd.crosstab(frame[variable], frame[TARGET_COL])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "test": "Not applicable",
            "statistic": np.nan,
            "p_value": np.nan,
            "n_permutations_successful": 0,
        }

    statistic, p_chi, _, expected = chi2_contingency(table, correction=False)
    if _expected_counts_adequate(expected):
        return {
            "test": "Pearson chi-square",
            "statistic": float(statistic),
            "p_value": float(p_chi),
            "n_permutations_successful": 0,
        }
    if table.shape == (2, 2):
        odds_ratio, p_value = fisher_exact(table.to_numpy())
        return {
            "test": "Fisher exact",
            "statistic": float(odds_ratio),
            "p_value": float(p_value),
            "n_permutations_successful": 0,
        }
    statistic, p_value, successful = _monte_carlo_chi_square(
        frame[variable],
        frame[TARGET_COL],
        n_permutations=n_permutations,
        random_state=random_state,
    )
    return {
        "test": "Monte Carlo chi-square",
        "statistic": statistic,
        "p_value": p_value,
        "n_permutations_successful": successful,
    }


def _categorical_pairwise(
    frame: pd.DataFrame,
    variable: str,
    classes: list[str],
    n_permutations: int,
    random_state: int,
    adjust_method: str = "bonferroni",
) -> pd.DataFrame:
    rows = []
    for pair_index, (class_a, class_b) in enumerate(itertools.combinations(classes, 2)):
        pair = frame[frame[TARGET_COL].isin([class_a, class_b])].copy()
        result = _categorical_test(
            pair,
            variable,
            n_permutations=n_permutations,
            random_state=random_state + pair_index,
        )
        rows.append(
            {
                "variable": variable,
                "comparison": f"Type {class_a} vs Type {class_b}",
                "test": result["test"],
                "statistic": result["statistic"],
                "raw_p": result["p_value"],
                "n_permutations_successful": result["n_permutations_successful"],
            }
        )
    output = pd.DataFrame(rows)
    if not output.empty:
        valid = output["raw_p"].notna()
        output.loc[valid, "adjusted_p"] = multipletests(
            output.loc[valid, "raw_p"], method=adjust_method
        )[1]
        output["adjust_method"] = adjust_method
    return output


def categorical_baseline_table(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = variables or DEFAULT_CATEGORICAL_VARIABLES
    classes = _sort_levels(df[TARGET_COL].dropna().astype(str))
    main_rows = []
    posthoc_tables = []

    for variable_index, variable in enumerate(variables):
        if variable not in df.columns:
            continue
        frame = df[[variable, TARGET_COL]].dropna().copy()
        frame[TARGET_COL] = frame[TARGET_COL].astype(str)
        overall = _categorical_test(
            frame,
            variable,
            n_permutations=n_permutations,
            random_state=random_state + 1000 * variable_index,
        )
        posthoc = _categorical_pairwise(
            frame,
            variable,
            classes,
            n_permutations=n_permutations,
            random_state=random_state + 10000 + 1000 * variable_index,
        )
        posthoc_tables.append(posthoc)
        pairwise_text = "; ".join(
            f"{row['comparison']}: adjusted P={row['adjusted_p']:.3f}"
            for _, row in posthoc.iterrows()
            if pd.notna(row.get("adjusted_p"))
        )

        for level in sorted(pd.unique(frame[variable]), key=str):
            row: dict[str, object] = {
                "variable": variable,
                "level": str(level),
                "overall": _count_percent(int((frame[variable] == level).sum()), len(frame)),
                "test": overall["test"],
                "statistic": overall["statistic"],
                "p_value": overall["p_value"],
                "pairwise_adjusted_p": pairwise_text,
                "n_permutations_successful": overall["n_permutations_successful"],
            }
            for class_label in classes:
                subset = frame[frame[TARGET_COL] == class_label]
                row[f"type_{class_label}"] = _count_percent(
                    int((subset[variable] == level).sum()), len(subset)
                )
            main_rows.append(row)

    posthoc_output = (
        pd.concat(posthoc_tables, ignore_index=True)
        if posthoc_tables
        else pd.DataFrame()
    )
    return pd.DataFrame(main_rows), posthoc_output


def run_descriptive_tables(
    df: pd.DataFrame,
    output_dir: str | Path,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    continuous, continuous_posthoc = continuous_baseline_table(df)
    categorical, categorical_posthoc = categorical_baseline_table(
        df,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    continuous.to_csv(output_dir / "baseline_continuous.csv", index=False)
    continuous_posthoc.to_csv(output_dir / "continuous_posthoc_results.csv", index=False)
    categorical.to_csv(output_dir / "baseline_categorical.csv", index=False)
    categorical_posthoc.to_csv(output_dir / "categorical_posthoc_results.csv", index=False)
    return {
        "continuous": continuous,
        "continuous_posthoc": continuous_posthoc,
        "categorical": categorical,
        "categorical_posthoc": categorical_posthoc,
    }
