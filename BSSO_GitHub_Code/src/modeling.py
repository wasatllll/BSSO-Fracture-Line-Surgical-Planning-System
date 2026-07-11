from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .constants import FINAL_FEATURES, MODEL_ORDER, PATIENT_ID_COL, TARGET_COL
from .metrics import align_probability_columns, calculate_metrics, per_class_metrics




class CompatibleLogisticRegression(ClassifierMixin, BaseEstimator):
    """Reproduce solver-specific multiclass behavior across scikit-learn versions."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 100,
        solver: str = "lbfgs",
        random_state: int = 42,
    ):
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state

    def fit(self, X, y):
        base = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state,
        )
        if self.solver == "liblinear" and len(np.unique(y)) > 2:
            self.model_ = OneVsRestClassifier(base)
        else:
            self.model_ = base
        self.model_.fit(X, y)
        self.classes_ = np.asarray(self.model_.classes_).astype(int)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).ravel().astype(int)

    def predict_proba(self, X):
        return np.asarray(self.model_.predict_proba(X), dtype=float)

    @property
    def coef_(self):
        if hasattr(self.model_, "estimators_"):
            return np.vstack([estimator.coef_.ravel() for estimator in self.model_.estimators_])
        return np.asarray(self.model_.coef_)


class CatBoostSklearnAdapter(ClassifierMixin, BaseEstimator):
    """Small compatibility wrapper for CatBoost across scikit-learn versions."""

    def __init__(
        self,
        iterations: int = 100,
        depth: int = 5,
        learning_rate: float = 0.01,
        l2_leaf_reg: float = 3.0,
        random_state: int = 42,
        thread_count: int = 1,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.thread_count = thread_count

    def fit(self, X, y):
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_state,
            verbose=False,
            loss_function="MultiClass",
            thread_count=self.thread_count,
        )
        self.model_.fit(X, y)
        self.classes_ = np.asarray(self.model_.classes_).astype(int)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).ravel().astype(int)

    def predict_proba(self, X):
        return np.asarray(self.model_.predict_proba(X), dtype=float)

    def get_feature_importance(self):
        return self.model_.get_feature_importance()


@dataclass
class ModelResult:
    name: str
    estimator: object
    best_parameters: dict[str, object]
    cv_accuracy: float
    cv_auc_micro: float
    cv_auc_micro_sd: float
    cv_auc_micro_oof: float
    oof_predictions: np.ndarray
    oof_probabilities: np.ndarray
    oof_fold_ids: np.ndarray
    fold_metrics: pd.DataFrame
    train_metrics: dict[str, float] | None = None
    test_predictions: np.ndarray | None = None
    test_probabilities: np.ndarray | None = None
    test_metrics: dict[str, float] | None = None




def _clean_parameter_dictionary(parameters: dict[str, object]) -> dict[str, object]:
    cleaned = {}
    for key, value in parameters.items():
        clean_key = key.replace("classifier__estimator__", "").replace("classifier__", "")
        cleaned[clean_key] = value
    return cleaned


def _clean_parameter_grid(grid):
    if isinstance(grid, list):
        return [_clean_parameter_dictionary(item) for item in grid]
    return _clean_parameter_dictionary(grid)


def _model_and_grid(name: str, n_classes: int, random_state: int):
    if name == "Logistic Regression":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    CompatibleLogisticRegression(random_state=random_state),
                ),
            ]
        )
        grid = {
            "classifier__C": [0.001, 0.01, 1, 10.0, 20.0, 100.0],
            "classifier__max_iter": [20, 25, 100],
            "classifier__solver": ["liblinear", "lbfgs", "saga"],
        }
    elif name == "SVM":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(probability=True, random_state=random_state)),
            ]
        )
        grid = {
            "classifier__C": [0.001, 0.01, 0.5, 1.0, 2, 5, 10],
            "classifier__kernel": ["rbf", "linear", "poly"],
            "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0, 10],
            "classifier__class_weight": [None, "balanced"],
        }
    elif name == "Random Forest":
        estimator = RandomForestClassifier(random_state=random_state, n_jobs=1)
        grid = {
            "n_estimators": [50, 100, 200, 300, 500],
            "max_depth": [3, 5, 7, 10, 20, None],
            "min_samples_split": [2, 5, 10, 15],
        }
    elif name == "XGBoost":
        estimator = XGBClassifier(
            random_state=random_state,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            n_jobs=1,
        )
        grid = {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [3, 4, 5, 6, 7, 15],
            "learning_rate": [0.001, 0.01, 0.1, 0.5],
        }
    elif name == "CatBoost":
        estimator = CatBoostSklearnAdapter(
            random_state=random_state,
            thread_count=1,
        )
        grid = {
            "iterations": [100, 200, 500],
            "depth": [3, 5, 7, 10],
            "learning_rate": [0.001, 0.01, 0.1],
            "l2_leaf_reg": [1, 3, 5, 10],
        }
    else:
        raise KeyError(name)
    return estimator, grid


def _legacy_patient_level_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reproduce the patient-level fold allocation used in the original analysis."""
    rng = np.random.RandomState(random_state)
    unique_groups = np.unique(groups)
    patient_labels = {
        patient_id: int(y[np.flatnonzero(groups == patient_id)[0]])
        for patient_id in unique_groups
    }
    class_patients = {
        class_id: [
            patient_id
            for patient_id, label in patient_labels.items()
            if label == class_id
        ]
        for class_id in np.unique(y)
    }
    minimum_class_patients = min(len(values) for values in class_patients.values())
    actual_splits = min(n_splits, minimum_class_patients, len(unique_groups))
    if actual_splits < 2:
        raise ValueError("Insufficient patient groups for patient-level cross-validation.")

    folds: list[list[object]] = [[] for _ in range(actual_splits)]
    fold_counts = {
        fold: {class_id: 0 for class_id in class_patients}
        for fold in range(actual_splits)
    }
    for class_id in class_patients:
        shuffled = list(class_patients[class_id])
        rng.shuffle(shuffled)
        for patient_id in shuffled:
            target_fold = min(
                range(actual_splits),
                key=lambda fold: fold_counts[fold][class_id],
            )
            folds[target_fold].append(patient_id)
            fold_counts[target_fold][class_id] += 1

    splits = []
    for fold_index in range(actual_splits):
        validation_patients = set(folds[fold_index])
        validation_index = np.flatnonzero(np.isin(groups, list(validation_patients)))
        training_index = np.flatnonzero(~np.isin(groups, list(validation_patients)))
        if len(np.unique(y[training_index])) < len(np.unique(y)):
            raise ValueError(f"Fold {fold_index + 1} training partition lacks an outcome class.")
        splits.append((training_index, validation_index))
    return splits


def _oof_evaluate(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    predictions = np.full(len(y), -1, dtype=int)
    probabilities = np.zeros((len(y), n_classes), dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_rows = []
    for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
        model = clone(estimator)
        model.fit(X.iloc[train_index], y[train_index])
        fold_predictions = np.asarray(model.predict(X.iloc[validation_index])).astype(int).ravel()
        fold_probabilities = align_probability_columns(
            model, model.predict_proba(X.iloc[validation_index]), n_classes
        )
        predictions[validation_index] = fold_predictions
        probabilities[validation_index] = fold_probabilities
        fold_ids[validation_index] = fold_number
        fold_rows.append(
            {
                "fold": fold_number,
                **calculate_metrics(
                    y[validation_index], fold_predictions, fold_probabilities, n_classes
                ),
            }
        )
    if np.any(fold_ids < 1):
        raise RuntimeError("At least one training observation did not receive an out-of-fold prediction.")
    return predictions, probabilities, fold_ids, pd.DataFrame(fold_rows)


def train_and_compare_models(
    training_df: pd.DataFrame,
    internal_test_df: pd.DataFrame,
    output_dir: str | Path,
    features: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, object]:
    """Tune models in the training set and lock the algorithm before test-set evaluation."""
    features = features or FINAL_FEATURES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)

    required = [PATIENT_ID_COL, TARGET_COL] + features
    train = training_df[required].dropna().reset_index(drop=True)
    test = internal_test_df[required].dropna().reset_index(drop=True)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train[TARGET_COL].astype(str))
    y_test = encoder.transform(test[TARGET_COL].astype(str))
    class_labels = [str(value) for value in encoder.classes_]
    n_classes = len(class_labels)

    X_train = train[features]
    X_test = test[features]
    groups = train[PATIENT_ID_COL].to_numpy()
    splits = _legacy_patient_level_splits(y_train, groups, n_splits, random_state)

    results: dict[str, ModelResult] = {}
    search_ranges: dict[str, object] = {}
    for name in MODEL_ORDER:
        estimator, parameter_grid = _model_and_grid(name, n_classes, random_state)
        search_ranges[name] = _clean_parameter_grid(parameter_grid)
        search = GridSearchCV(
            estimator,
            parameter_grid,
            scoring="accuracy",
            cv=splits,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=True,
            error_score=np.nan,
        )
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        oof_pred, oof_proba, oof_fold_ids, fold_metrics = _oof_evaluate(
            best_estimator, X_train, y_train, splits, n_classes
        )
        cv_accuracy = float(search.best_score_)
        cv_auc_micro = float(fold_metrics["auc_micro"].mean())
        cv_auc_micro_sd = float(fold_metrics["auc_micro"].std(ddof=1))
        cv_auc_micro_oof = float(
            calculate_metrics(y_train, oof_pred, oof_proba, n_classes)["auc_micro"]
        )
        best_estimator.fit(X_train, y_train)
        train_predictions = np.asarray(best_estimator.predict(X_train)).astype(int).ravel()
        train_probabilities = align_probability_columns(
            best_estimator, best_estimator.predict_proba(X_train), n_classes
        )
        train_metrics = calculate_metrics(
            y_train, train_predictions, train_probabilities, n_classes
        )

        results[name] = ModelResult(
            name=name,
            estimator=best_estimator,
            best_parameters=_clean_parameter_dictionary(search.best_params_),
            cv_accuracy=cv_accuracy,
            cv_auc_micro=cv_auc_micro,
            cv_auc_micro_sd=cv_auc_micro_sd,
            cv_auc_micro_oof=cv_auc_micro_oof,
            oof_predictions=oof_pred,
            oof_probabilities=oof_proba,
            oof_fold_ids=oof_fold_ids,
            fold_metrics=fold_metrics,
            train_metrics=train_metrics,
        )
        joblib.dump(best_estimator, model_dir / f"{name.lower().replace(' ', '_')}.joblib")
        fold_metrics.to_csv(
            output_dir / f"{name.lower().replace(' ', '_')}_cv_fold_metrics.csv", index=False
        )

    with open(output_dir / "hyperparameter_search_ranges.json", "w", encoding="utf-8") as handle:
        json.dump(search_ranges, handle, indent=2)

    pd.DataFrame(
        [
            {"model": name, **result.best_parameters}
            for name, result in results.items()
        ]
    ).to_csv(output_dir / "best_hyperparameters.csv", index=False)

    # Algorithm selection is restricted to training-set CV.
    selection_table = pd.DataFrame(
        [
            {
                "model": name,
                "cv_accuracy": result.cv_accuracy,
                "cv_auc_micro": result.cv_auc_micro,
                "cv_auc_micro_sd": result.cv_auc_micro_sd,
                "cv_auc_micro_oof": result.cv_auc_micro_oof,
                "best_parameters": json.dumps(result.best_parameters, sort_keys=True),
            }
            for name, result in results.items()
        ]
    ).sort_values(["cv_accuracy", "cv_auc_micro"], ascending=[False, False])
    selection_table["selected_by_training_cv"] = False
    locked_name = str(selection_table.iloc[0]["model"])
    selection_table.loc[selection_table["model"] == locked_name, "selected_by_training_cv"] = True
    selection_table.to_csv(output_dir / "model_selection_training_cv_only.csv", index=False)

    # The internal test set is evaluated only after the algorithm has been locked.
    test_rows = []
    per_class_tables = []
    for name, result in results.items():
        predictions = np.asarray(result.estimator.predict(X_test)).astype(int).ravel()
        probabilities = align_probability_columns(
            result.estimator, result.estimator.predict_proba(X_test), n_classes
        )
        metrics = calculate_metrics(y_test, predictions, probabilities, n_classes)
        result.test_predictions = predictions
        result.test_probabilities = probabilities
        result.test_metrics = metrics
        test_rows.append({"model": name, **metrics})
        class_table = per_class_metrics(y_test, predictions, probabilities, class_labels)
        class_table.insert(0, "model", name)
        per_class_tables.append(class_table)

    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(output_dir / "internal_test_performance_all_models.csv", index=False)
    pd.concat(per_class_tables, ignore_index=True).to_csv(
        output_dir / "internal_test_per_class_metrics.csv", index=False
    )

    overall_rows = []
    for name in MODEL_ORDER:
        result = results[name]
        overall_rows.append(
            {
                "model": name,
                "train_accuracy": result.train_metrics["accuracy"],
                "cv_accuracy": result.cv_accuracy,
                "train_auc_micro": result.train_metrics["auc_micro"],
                "cv_auc_micro_mean": result.cv_auc_micro,
                "cv_auc_micro_sd": result.cv_auc_micro_sd,
                "internal_test_accuracy": result.test_metrics["accuracy"],
                "internal_test_auc_micro": result.test_metrics["auc_micro"],
                "internal_test_auc_macro_ovr": result.test_metrics["auc_macro_ovr"],
                "internal_test_f1_macro": result.test_metrics["f1_macro"],
            }
        )
    pd.DataFrame(overall_rows).to_csv(
        output_dir / "overall_model_performance.csv", index=False
    )

    locked_result = results[locked_name]
    locked_manifest = {
        "selection_dataset": "training set only",
        "primary_selection_metric": "cross-validated accuracy",
        "tie_breaker": "cross-validated micro-average AUC",
        "locked_model": locked_name,
        "features": features,
        "classes": class_labels,
        "internal_test_used_for_selection": False,
    }
    with open(output_dir / "locked_model_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(locked_manifest, handle, indent=2)

    return {
        "results": results,
        "locked_model_name": locked_name,
        "locked_model": locked_result.estimator,
        "encoder": encoder,
        "class_labels": class_labels,
        "features": features,
        "training_df": train,
        "test_df": test,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "selection_table": selection_table,
        "test_table": test_table,
    }


def evaluate_external_cohort(
    cohort_df: pd.DataFrame,
    estimator,
    encoder: LabelEncoder,
    features: list[str],
) -> dict[str, object]:
    required = [PATIENT_ID_COL, TARGET_COL] + features
    cohort = cohort_df[required].dropna().reset_index(drop=True)
    X = cohort[features]
    y = encoder.transform(cohort[TARGET_COL].astype(str))
    predictions = np.asarray(estimator.predict(X)).astype(int).ravel()
    probabilities = align_probability_columns(estimator, estimator.predict_proba(X), len(encoder.classes_))
    metrics = calculate_metrics(y, predictions, probabilities, len(encoder.classes_))
    class_table = per_class_metrics(y, predictions, probabilities, [str(x) for x in encoder.classes_])
    return {
        "data": cohort,
        "X": X,
        "y": y,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
        "per_class": class_table,
    }
