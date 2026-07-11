from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .constants import (
    BINARY_CANDIDATES,
    CANDIDATE_FEATURES,
    CONTINUOUS_CANDIDATES,
    PATIENT_ID_COL,
    TARGET_COL,
    TOP50_THRESHOLD,
    TOP_N_SUBSETS,
)


def _safe_n_splits(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    class_group_counts = []
    for label in np.unique(y):
        class_group_counts.append(len(pd.unique(groups[y == label])))
    n_splits = min(requested, min(class_group_counts), len(pd.unique(groups)))
    if n_splits < 2:
        raise ValueError("Insufficient grouped observations for cross-validation.")
    return int(n_splits)


def make_group_cv(y: np.ndarray, groups: np.ndarray, n_splits: int = 5, random_state: int = 42):
    return StratifiedGroupKFold(
        n_splits=_safe_n_splits(y, groups, n_splits),
        shuffle=True,
        random_state=random_state,
    )


def _multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    one_hot = label_binarize(y_true, classes=np.arange(n_classes))
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _align_probability_columns(model, probabilities: np.ndarray, n_classes: int) -> np.ndarray:
    classes = None
    if hasattr(model, "classes_"):
        classes = np.asarray(model.classes_).astype(int)
    elif hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                classes = np.asarray(step.classes_).astype(int)
                break
    if classes is None:
        return probabilities
    aligned = np.zeros((len(probabilities), n_classes), dtype=float)
    for column, class_id in enumerate(classes):
        aligned[:, class_id] = probabilities[:, column]
    aligned += 1e-12
    return aligned / aligned.sum(axis=1, keepdims=True)


def _subset_estimator(features: list[str], random_state: int) -> Pipeline:
    continuous = [f for f in features if f in CONTINUOUS_CANDIDATES]
    binary = [f for f in features if f in BINARY_CANDIDATES]
    transformers = []
    if continuous:
        transformers.append(("continuous", StandardScaler(), continuous))
    if binary:
        transformers.append(("binary", "passthrough", binary))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _evaluate_subset(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    random_state: int,
) -> dict[str, float]:
    n_classes = len(np.unique(y))
    predictions = np.full(len(y), -1, dtype=int)
    probabilities = np.zeros((len(y), n_classes), dtype=float)
    estimator = _subset_estimator(features, random_state)

    for train_index, validation_index in splits:
        model = clone(estimator)
        model.fit(X.iloc[train_index][features], y[train_index])
        predictions[validation_index] = model.predict(X.iloc[validation_index][features]).astype(int)
        fold_probabilities = model.predict_proba(X.iloc[validation_index][features])
        probabilities[validation_index] = _align_probability_columns(model, fold_probabilities, n_classes)

    one_hot = label_binarize(y, classes=np.arange(n_classes))
    return {
        "Log_Loss": float(log_loss(y, probabilities, labels=np.arange(n_classes))),
        "Accuracy": float(accuracy_score(y, predictions)),
        "Macro_F1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, predictions)),
        "AUC_OVR_Macro": float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro")),
        "Brier_Multiclass": _multiclass_brier(y, probabilities, n_classes),
        "AP_Macro": float(average_precision_score(one_hot, probabilities, average="macro")),
    }


def _rank(values: pd.Series, higher_is_better: bool) -> np.ndarray:
    clean = pd.to_numeric(values, errors="coerce")
    clean = clean.fillna(clean.median())
    return rankdata(-clean if higher_is_better else clean, method="average")


def exhaustive_best_subset(
    training_df: pd.DataFrame,
    output_dir: str | Path,
    candidate_features: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    top_n: int = TOP_N_SUBSETS,
    threshold: float = TOP50_THRESHOLD,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_features = candidate_features or CANDIDATE_FEATURES

    required = [PATIENT_ID_COL, TARGET_COL] + candidate_features
    data = training_df[required].dropna().copy().reset_index(drop=True)
    encoder = LabelEncoder()
    y = encoder.fit_transform(data[TARGET_COL].astype(str))
    groups = data[PATIENT_ID_COL].to_numpy()
    X = data[candidate_features]

    cv = make_group_cv(y, groups, n_splits=n_splits, random_state=random_state)
    splits = list(cv.split(X, y, groups))

    rows: list[dict[str, object]] = []
    feature_indices = range(len(candidate_features))
    for subset_size in range(1, len(candidate_features) + 1):
        for indices in itertools.combinations(feature_indices, subset_size):
            subset = [candidate_features[index] for index in indices]
            metrics = _evaluate_subset(X, y, groups, subset, splits, random_state)
            rows.append(
                {
                    "n_features": subset_size,
                    "features": ", ".join(subset),
                    "feature_key": "|".join(sorted(subset)),
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    results["rank_log_loss"] = _rank(results["Log_Loss"], higher_is_better=False)
    results["rank_accuracy"] = _rank(results["Accuracy"], higher_is_better=True)
    results["rank_macro_f1"] = _rank(results["Macro_F1"], higher_is_better=True)
    results["rank_balanced_accuracy"] = _rank(results["Balanced_Accuracy"], higher_is_better=True)
    results["rank_auc"] = _rank(results["AUC_OVR_Macro"], higher_is_better=True)
    results["rank_brier"] = _rank(results["Brier_Multiclass"], higher_is_better=False)
    rank_columns = [
        "rank_log_loss",
        "rank_accuracy",
        "rank_macro_f1",
        "rank_balanced_accuracy",
        "rank_auc",
        "rank_brier",
    ]
    results["multi_objective_mean_rank"] = results[rank_columns].mean(axis=1)
    results = results.sort_values(
        ["multi_objective_mean_rank", "rank_macro_f1", "rank_accuracy"]
    ).reset_index(drop=True)
    results["multi_objective_rank"] = np.arange(1, len(results) + 1)
    results.to_csv(output_dir / "exhaustive_all_subsets_multi_objective_rank.csv", index=False)

    top = results.head(min(top_n, len(results)))
    recurrence_rows = []
    for feature in candidate_features:
        count = int(top["features"].apply(lambda text: feature in text.split(", ")).sum())
        recurrence_rows.append(
            {
                "feature": feature,
                "count": count,
                "inclusion_frequency": count / len(top),
                "operational_threshold": threshold,
                "selected": count / len(top) > threshold,
            }
        )
    recurrence = pd.DataFrame(recurrence_rows).sort_values(
        ["inclusion_frequency", "feature"], ascending=[False, True]
    )
    recurrence.to_csv(output_dir / "top50_recurrence_decision.csv", index=False)

    recurrence_plot = recurrence.sort_values("inclusion_frequency")
    fig, axis = plt.subplots(figsize=(8.5, 6.2))
    axis.barh(recurrence_plot["feature"], recurrence_plot["inclusion_frequency"])
    axis.axvline(threshold, linestyle="--", linewidth=1.4, label=f"Strict threshold > {threshold:.2f}")
    axis.set_xlim(0, 1.05)
    axis.set_xlabel(f"Inclusion frequency among Top-{top_n} subsets")
    axis.set_title("Top-50 high-performing subset recurrence")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_4a_top50_recurrence.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "figure_4a_top50_recurrence.pdf", bbox_inches="tight")
    plt.close(fig)

    best_by_size = (
        results.sort_values("multi_objective_rank")
        .groupby("n_features", as_index=False)
        .first()
        .sort_values("n_features")
    )
    best_by_size.to_csv(output_dir / "best_subset_by_feature_number.csv", index=False)
    fig, axis_left = plt.subplots(figsize=(9, 6))
    axis_left.plot(best_by_size["n_features"], best_by_size["Log_Loss"], marker="o", label="Log loss")
    axis_left.set_xlabel("Number of predictors")
    axis_left.set_ylabel("Log loss")
    axis_right = axis_left.twinx()
    axis_right.plot(best_by_size["n_features"], best_by_size["Macro_F1"], marker="s", linestyle="--", label="Macro-F1")
    axis_right.plot(best_by_size["n_features"], best_by_size["AUC_OVR_Macro"], marker="^", linestyle=":", label="Macro-AUC")
    axis_right.set_ylabel("Classification metric")
    lines_left, labels_left = axis_left.get_legend_handles_labels()
    lines_right, labels_right = axis_right.get_legend_handles_labels()
    axis_left.legend(lines_left + lines_right, labels_left + labels_right, frameon=False, loc="best")
    axis_left.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "supplementary_figure_s4_performance_complexity.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "supplementary_figure_s4_performance_complexity.pdf", bbox_inches="tight")
    plt.close(fig)

    selected_features = recurrence.loc[recurrence["selected"], "feature"].tolist()
    selected_key = "|".join(sorted(selected_features))
    selected_row = results.loc[results["feature_key"] == selected_key]
    if len(selected_row) != 1:
        raise RuntimeError("The Top-50-selected feature set was not uniquely identified in exhaustive results.")
    selected_row = selected_row.iloc[0]

    best_same_size = (
        results.loc[results["n_features"] == len(selected_features)]
        .sort_values("multi_objective_rank")
        .iloc[0]
    )
    summary = {
        "decision_rule": f"Top-{top_n} inclusion frequency strictly greater than {threshold:.2f}",
        "selected_features": selected_features,
        "selected_k": len(selected_features),
        "selected_subset_rank": int(selected_row["multi_objective_rank"]),
        "best_subset_of_same_size": bool(
            int(selected_row["multi_objective_rank"]) == int(best_same_size["multi_objective_rank"])
        ),
        "accuracy": float(selected_row["Accuracy"]),
        "macro_f1": float(selected_row["Macro_F1"]),
        "balanced_accuracy": float(selected_row["Balanced_Accuracy"]),
        "macro_auc": float(selected_row["AUC_OVR_Macro"]),
        "log_loss": float(selected_row["Log_Loss"]),
        "multiclass_brier": float(selected_row["Brier_Multiclass"]),
    }
    with open(output_dir / "feature_selection_decision.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    pd.DataFrame([summary]).to_csv(output_dir / "feature_selection_decision.csv", index=False)
    return {
        "summary": summary,
        "recurrence": recurrence,
        "exhaustive": results,
        "encoder": encoder,
    }


def correlation_and_vif(training_df: pd.DataFrame, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = training_df[CANDIDATE_FEATURES].dropna().copy()
    spearman = data.corr(method="spearman")
    spearman.to_csv(output_dir / "training_spearman_correlation.csv")

    scaled = StandardScaler().fit_transform(data)
    vif_rows = []
    for index, feature in enumerate(CANDIDATE_FEATURES):
        try:
            value = float(variance_inflation_factor(scaled, index))
        except Exception:
            value = np.nan
        vif_rows.append({"feature": feature, "vif": value})
    vif = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)
    vif.to_csv(output_dir / "training_variance_inflation_factors.csv", index=False)

    pair_data = training_df[CONTINUOUS_CANDIDATES + [TARGET_COL]].dropna().copy()
    pair_grid = sns.pairplot(
        pair_data,
        vars=CONTINUOUS_CANDIDATES,
        hue=TARGET_COL,
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.65, "s": 22, "edgecolor": "none"},
    )
    pair_grid.fig.suptitle("Training-set pairwise distributions of continuous predictors", y=1.01)
    pair_grid.fig.savefig(
        output_dir / "supplementary_figure_s2_pairplot.png", dpi=600, bbox_inches="tight"
    )
    plt.close(pair_grid.fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(
        spearman,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="RdBu_r",
        square=True,
        ax=axes[0],
        cbar_kws={"label": "Spearman correlation"},
    )
    axes[0].set_title("Spearman correlation matrix")
    ordered_vif = vif.sort_values("vif")
    axes[1].barh(ordered_vif["feature"], ordered_vif["vif"])
    axes[1].axvline(5, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Variance inflation factor")
    axes[1].set_title("Variance inflation factor analysis")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "supplementary_figure_s3_correlation_vif.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "supplementary_figure_s3_correlation_vif.pdf", bbox_inches="tight")
    plt.close(fig)

    return {"spearman": spearman, "vif": vif}


def _patient_subsample_indices(
    y: np.ndarray,
    groups: np.ndarray,
    fraction: float,
    random_state: int,
    max_attempts: int = 300,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    unique_groups = np.asarray(pd.unique(groups))
    n_select = max(2, int(round(len(unique_groups) * fraction)))
    all_classes = set(np.unique(y))
    indices = np.arange(len(y))
    for _ in range(max_attempts):
        sampled_groups = rng.choice(unique_groups, size=n_select, replace=False)
        indices = np.flatnonzero(np.isin(groups, sampled_groups))
        if set(np.unique(y[indices])) == all_classes:
            break
    return indices


def _mrmr_select(X: pd.DataFrame, y: np.ndarray, k: int, random_state: int) -> list[str]:
    values = X.to_numpy(dtype=float)
    relevance = mutual_info_classif(values, y, discrete_features=False, random_state=random_state)
    if np.ptp(relevance) > 0:
        relevance = (relevance - relevance.min()) / np.ptp(relevance)
    correlation = X.corr().abs().fillna(0).to_numpy()
    selected: list[int] = []
    remaining = list(range(X.shape[1]))
    for _ in range(min(k, X.shape[1])):
        scores = []
        for index in remaining:
            redundancy = np.mean([correlation[index, chosen] for chosen in selected]) if selected else 0.0
            scores.append((float(relevance[index] - redundancy), index))
        _, best = max(scores, key=lambda item: item[0])
        selected.append(best)
        remaining.remove(best)
    return [X.columns[index] for index in selected]


def complementary_stability_analyses(
    training_df: pd.DataFrame,
    output_dir: str | Path,
    n_resampling: int = 300,
    n_rf_resampling: int = 100,
    fraction: float = 0.80,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run LASSO, Elastic Net, mRMR, RF-RFE, and Boruta-style supporting analyses."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = training_df[[PATIENT_ID_COL, TARGET_COL] + CANDIDATE_FEATURES].dropna().reset_index(drop=True)
    X = data[CANDIDATE_FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(data[TARGET_COL].astype(str))
    groups = data[PATIENT_ID_COL].to_numpy()
    p = len(CANDIDATE_FEATURES)

    frequencies = {
        "LASSO": np.zeros(p),
        "ElasticNet": np.zeros(p),
        "mRMR": np.zeros(p),
        "RF_RFE": np.zeros(p),
        "Boruta": np.zeros(p),
    }
    successes = Counter()

    lasso_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    elastic_grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    l1_ratios = [0.2, 0.5, 0.8]

    for iteration in range(n_resampling):
        indices = _patient_subsample_indices(y, groups, fraction, random_state + iteration)
        X_sub, y_sub, groups_sub = X.iloc[indices], y[indices], groups[indices]
        cv = make_group_cv(y_sub, groups_sub, n_splits=3, random_state=random_state + iteration)

        for name, penalty, grid in [
            ("LASSO", "l1", {"classifier__C": lasso_grid}),
            (
                "ElasticNet",
                "elasticnet",
                {"classifier__C": elastic_grid, "classifier__l1_ratio": l1_ratios},
            ),
        ]:
            estimator = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            penalty=penalty,
                            solver="saga",
                            max_iter=8000,
                            class_weight="balanced",
                            random_state=random_state + iteration,
                        ),
                    ),
                ]
            )
            search = GridSearchCV(
                estimator,
                grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
                error_score=np.nan,
            )
            try:
                search.fit(X_sub, y_sub, groups=groups_sub)
                coefficients = search.best_estimator_.named_steps["classifier"].coef_
                selected = np.any(np.abs(coefficients) > 1e-6, axis=0)
                frequencies[name] += selected.astype(float)
                successes[name] += 1
            except Exception:
                pass

        try:
            selected = _mrmr_select(X_sub, y_sub, k=5, random_state=random_state + 20000 + iteration)
            frequencies["mRMR"] += np.array([feature in selected for feature in CANDIDATE_FEATURES], dtype=float)
            successes["mRMR"] += 1
        except Exception:
            pass

        try:
            rng = np.random.default_rng(random_state + 40000 + iteration)
            shadow = X_sub.copy()
            for column in shadow.columns:
                shadow[column] = rng.permutation(shadow[column].to_numpy())
            shadow.columns = [f"{column}_shadow" for column in shadow.columns]
            combined = pd.concat([X_sub.reset_index(drop=True), shadow.reset_index(drop=True)], axis=1)
            forest = RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state + 40000 + iteration,
                n_jobs=-1,
            )
            forest.fit(combined, y_sub)
            importances = forest.feature_importances_
            selected = importances[:p] > np.max(importances[p:])
            frequencies["Boruta"] += selected.astype(float)
            successes["Boruta"] += 1
        except Exception:
            pass

    for iteration in range(n_rf_resampling):
        indices = _patient_subsample_indices(y, groups, fraction, random_state + 30000 + iteration)
        X_sub = X.iloc[indices].reset_index(drop=True)
        y_sub = y[indices]
        groups_sub = groups[indices]
        cv = make_group_cv(y_sub, groups_sub, n_splits=3, random_state=random_state + 30000 + iteration)
        splits = list(cv.split(X_sub, y_sub, groups_sub))
        forest = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state + 30000 + iteration,
            n_jobs=-1,
        )
        selector = RFECV(
            estimator=forest,
            step=1,
            min_features_to_select=2,
            cv=splits,
            scoring="f1_macro",
            n_jobs=-1,
        )
        try:
            selector.fit(X_sub, y_sub)
            frequencies["RF_RFE"] += selector.support_.astype(float)
            successes["RF_RFE"] += 1
        except Exception:
            pass

    rows = []
    for index, feature in enumerate(CANDIDATE_FEATURES):
        row = {"feature": feature}
        for method in frequencies:
            row[method] = frequencies[method][index] / max(successes[method], 1)
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "complementary_feature_stability.csv", index=False)
    pd.DataFrame(
        [{"method": method, "successful_iterations": count} for method, count in successes.items()]
    ).to_csv(output_dir / "complementary_method_success_counts.csv", index=False)

    heat = result.set_index("feature")[["LASSO", "ElasticNet", "mRMR", "RF_RFE", "Boruta"]]
    fig, axis = plt.subplots(figsize=(8.5, max(5, 0.45 * len(heat) + 2)))
    sns.heatmap(heat, vmin=0, vmax=1, cmap="YlOrRd", annot=True, fmt=".2f", ax=axis)
    axis.set_title("Complementary feature-selection stability")
    axis.set_xlabel("Method")
    axis.set_ylabel("Predictor")
    fig.tight_layout()
    fig.savefig(output_dir / "supplementary_figure_s4_complementary_stability.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "supplementary_figure_s4_complementary_stability.pdf", bbox_inches="tight")
    plt.close(fig)
    build_supplementary_s4_composite(
        training_df, output_dir, result, random_state=random_state
    )
    return result


def build_supplementary_s4_composite(
    training_df: pd.DataFrame,
    output_dir: str | Path,
    stability: pd.DataFrame,
    random_state: int = 42,
) -> None:
    """Reconstruct the 12-panel complementary feature-selection diagnostic figure."""
    import warnings

    output_dir = Path(output_dir)
    diagnostic_dir = output_dir / "method_specific_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    exhaustive_path = output_dir / "exhaustive_all_subsets_multi_objective_rank.csv"
    recurrence_path = output_dir / "top50_recurrence_decision.csv"
    best_size_path = output_dir / "best_subset_by_feature_number.csv"
    if not all(path.exists() for path in [exhaustive_path, recurrence_path, best_size_path]):
        raise FileNotFoundError(
            "Exhaustive subset outputs are required before building Supplementary Fig. S4."
        )

    exhaustive = pd.read_csv(exhaustive_path)
    recurrence = pd.read_csv(recurrence_path)
    best_by_size = pd.read_csv(best_size_path)
    data = training_df[[PATIENT_ID_COL, TARGET_COL] + CANDIDATE_FEATURES].dropna().reset_index(drop=True)
    X = data[CANDIDATE_FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(data[TARGET_COL].astype(str))
    groups = data[PATIENT_ID_COL].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    c_path = np.logspace(-3, 2, 24)
    lasso_matrix = []
    elastic_matrix = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c_value in c_path:
            lasso = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=float(c_value),
                max_iter=3000,
                tol=1e-3,
                class_weight="balanced",
                random_state=random_state,
            )
            elastic = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=float(c_value),
                max_iter=3000,
                tol=1e-3,
                class_weight="balanced",
                random_state=random_state,
            )
            try:
                lasso.fit(X_scaled, y)
                lasso_matrix.append(np.sqrt(np.sum(lasso.coef_**2, axis=0)))
            except Exception:
                lasso_matrix.append(np.full(len(CANDIDATE_FEATURES), np.nan))
            try:
                elastic.fit(X_scaled, y)
                elastic_matrix.append(np.sqrt(np.sum(elastic.coef_**2, axis=0)))
            except Exception:
                elastic_matrix.append(np.full(len(CANDIDATE_FEATURES), np.nan))
    lasso_matrix = np.asarray(lasso_matrix)
    elastic_matrix = np.asarray(elastic_matrix)

    path_rows = []
    for method, matrix in [("LASSO", lasso_matrix), ("ElasticNet", elastic_matrix)]:
        for path_index, c_value in enumerate(c_path):
            for feature_index, feature in enumerate(CANDIDATE_FEATURES):
                path_rows.append(
                    {
                        "method": method,
                        "C": float(c_value),
                        "log10_C": float(np.log10(c_value)),
                        "feature": feature,
                        "coefficient_l2_norm": float(matrix[path_index, feature_index]),
                    }
                )
    pd.DataFrame(path_rows).to_csv(
        diagnostic_dir / "penalized_regression_coefficient_paths.csv", index=False
    )

    relevance = mutual_info_classif(
        X.to_numpy(dtype=float), y, discrete_features=False, random_state=random_state
    )
    if np.ptp(relevance) > 0:
        relevance_scaled = (relevance - relevance.min()) / np.ptp(relevance)
    else:
        relevance_scaled = np.zeros_like(relevance)
    redundancy = X.corr().abs().to_numpy()
    mean_redundancy = (redundancy.sum(axis=1) - 1) / max(len(CANDIDATE_FEATURES) - 1, 1)
    mrmr_profile = pd.DataFrame(
        {
            "feature": CANDIDATE_FEATURES,
            "relevance_scaled": relevance_scaled,
            "mean_absolute_redundancy": mean_redundancy,
        }
    ).merge(stability[["feature", "mRMR"]], on="feature", how="left")
    mrmr_profile.to_csv(diagnostic_dir / "mrmr_relevance_redundancy_profile.csv", index=False)

    rf = RandomForestClassifier(
        n_estimators=30,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    cv = make_group_cv(y, groups, n_splits=5, random_state=random_state)
    splits = list(cv.split(X, y, groups))
    rfecv = RFECV(
        estimator=rf,
        step=1,
        min_features_to_select=2,
        cv=splits,
        scoring="f1_macro",
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfecv.fit(X, y)
    cv_results = rfecv.cv_results_
    mean_score = np.asarray(cv_results["mean_test_score"])
    sd_score = np.asarray(cv_results.get("std_test_score", np.full_like(mean_score, np.nan)))
    if "n_features" in cv_results:
        n_features_path = np.asarray(cv_results["n_features"]).astype(int)
    else:
        n_features_path = np.arange(2, 2 + len(mean_score))
    rf_path = pd.DataFrame(
        {
            "n_features": n_features_path,
            "mean_macro_f1": mean_score,
            "sd_macro_f1": sd_score,
            "selected_n_features": int(rfecv.n_features_),
        }
    ).sort_values("n_features")
    rf_path.to_csv(diagnostic_dir / "rf_rfe_performance_path.csv", index=False)

    top30 = exhaustive.head(min(30, len(exhaustive))).copy()
    membership = np.zeros((len(top30), len(CANDIDATE_FEATURES)), dtype=int)
    for row_index, feature_text in enumerate(top30["features"]):
        included = set(str(feature_text).split(", "))
        membership[row_index] = [int(feature in included) for feature in CANDIDATE_FEATURES]

    frequency_map = stability.set_index("feature")
    recurrence_map = recurrence.set_index("feature")["inclusion_frequency"]

    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    axes = axes.ravel()

    axes[0].imshow(membership, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    axes[0].set_xticks(np.arange(len(CANDIDATE_FEATURES)))
    axes[0].set_xticklabels(CANDIDATE_FEATURES, rotation=75, ha="right", fontsize=8)
    axes[0].set_ylabel("Top-ranked subset")
    axes[0].set_title("a  Top-30 subset membership", loc="left", fontweight="bold")

    axes[1].plot(best_by_size["n_features"], best_by_size["Log_Loss"], marker="o", label="Log loss")
    axes[1].plot(best_by_size["n_features"], best_by_size["Macro_F1"], marker="s", label="Macro-F1")
    axes[1].plot(best_by_size["n_features"], best_by_size["AUC_OVR_Macro"], marker="^", label="Macro-AUC")
    axes[1].axvline(4, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Number of predictors")
    axes[1].set_title("b  Performance–complexity", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)

    for feature_index, feature in enumerate(CANDIDATE_FEATURES):
        axes[2].plot(np.log10(c_path), lasso_matrix[:, feature_index], linewidth=1.4, label=feature)
    axes[2].set_xlabel("log10(C)")
    axes[2].set_ylabel("Coefficient L2 norm")
    axes[2].set_title("c  LASSO coefficient path", loc="left", fontweight="bold")
    axes[2].grid(alpha=0.2)

    ordered = stability.sort_values("LASSO")
    axes[3].barh(ordered["feature"], ordered["LASSO"])
    axes[3].set_xlim(0, 1)
    axes[3].set_title("d  LASSO selection frequency", loc="left", fontweight="bold")

    for feature_index, feature in enumerate(CANDIDATE_FEATURES):
        axes[4].plot(np.log10(c_path), elastic_matrix[:, feature_index], linewidth=1.4, label=feature)
    axes[4].set_xlabel("log10(C)")
    axes[4].set_ylabel("Coefficient L2 norm")
    axes[4].set_title("e  Elastic Net coefficient path", loc="left", fontweight="bold")
    axes[4].grid(alpha=0.2)

    ordered = stability.sort_values("ElasticNet")
    axes[5].barh(ordered["feature"], ordered["ElasticNet"])
    axes[5].set_xlim(0, 1)
    axes[5].set_title("f  Elastic Net selection frequency", loc="left", fontweight="bold")

    ordered = stability.sort_values("mRMR")
    axes[6].barh(ordered["feature"], ordered["mRMR"])
    axes[6].set_xlim(0, 1)
    axes[6].set_title("g  mRMR selection frequency", loc="left", fontweight="bold")

    scatter = axes[7].scatter(
        mrmr_profile["mean_absolute_redundancy"],
        mrmr_profile["relevance_scaled"],
        s=60 + 240 * mrmr_profile["mRMR"].fillna(0),
        c=mrmr_profile["mRMR"].fillna(0),
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    for _, row in mrmr_profile.iterrows():
        axes[7].annotate(row["feature"], (row["mean_absolute_redundancy"], row["relevance_scaled"]), fontsize=7)
    axes[7].set_xlabel("Mean absolute redundancy")
    axes[7].set_ylabel("Scaled relevance")
    axes[7].set_title("h  mRMR relevance–redundancy", loc="left", fontweight="bold")
    fig.colorbar(scatter, ax=axes[7], label="Selection frequency")

    axes[8].errorbar(
        rf_path["n_features"],
        rf_path["mean_macro_f1"],
        yerr=rf_path["sd_macro_f1"],
        marker="o",
        capsize=3,
    )
    axes[8].axvline(rfecv.n_features_, linestyle="--", linewidth=1)
    axes[8].set_xlabel("Number of retained predictors")
    axes[8].set_ylabel("CV macro-F1")
    axes[8].set_title("i  RF-RFE performance path", loc="left", fontweight="bold")
    axes[8].grid(alpha=0.2)

    ordered = stability.sort_values("RF_RFE")
    axes[9].barh(ordered["feature"], ordered["RF_RFE"])
    axes[9].set_xlim(0, 1)
    axes[9].set_title("j  RF-RFE selection frequency", loc="left", fontweight="bold")

    ordered = stability.sort_values("Boruta")
    axes[10].barh(ordered["feature"], ordered["Boruta"])
    axes[10].set_xlim(0, 1)
    axes[10].set_title("k  Boruta-style confirmed frequency", loc="left", fontweight="bold")

    boruta = frequency_map["Boruta"].reindex(CANDIDATE_FEATURES)
    top50 = recurrence_map.reindex(CANDIDATE_FEATURES)
    axes[11].scatter(boruta, top50, s=55)
    axes[11].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    for feature in CANDIDATE_FEATURES:
        axes[11].annotate(feature, (boruta.loc[feature], top50.loc[feature]), fontsize=7)
    axes[11].set_xlim(-0.03, 1.03)
    axes[11].set_ylim(-0.03, 1.03)
    axes[11].set_xlabel("Boruta confirmed frequency")
    axes[11].set_ylabel("Top-50 recurrence")
    axes[11].set_title("l  Boruta vs Top-50 recurrence", loc="left", fontweight="bold")
    axes[11].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "supplementary_figure_s4_composite.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "supplementary_figure_s4_composite.pdf", bbox_inches="tight")
    plt.close(fig)
