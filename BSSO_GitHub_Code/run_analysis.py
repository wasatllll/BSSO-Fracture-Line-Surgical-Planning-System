from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.causal import run_causal_analysis
from src.constants import (
    FINAL_FEATURES,
    MODEL_ORDER,
    PATIENT_ID_COL,
    RANDOM_STATE_MODEL,
    RANDOM_STATE_SPLIT,
    TARGET_COL,
)
from src.descriptive import run_descriptive_tables
from src.data import (
    patient_level_stratified_split,
    read_cohort,
    save_dataset_manifest,
    validate_bilateral_structure,
)
from src.feasibility import analyze_feasibility, read_feasibility_data
from src.feature_selection import (
    complementary_stability_analyses,
    correlation_and_vif,
    exhaustive_best_subset,
)
from src.metrics import (
    classwise_brier_scores,
    hosmer_lemeshow_binary,
    multiclass_micro_auc,
    paired_delong_micro_multiclass,
    patient_cluster_bootstrap_ci,
)
from src.modeling import evaluate_external_cohort, train_and_compare_models
from src.plotting import (
    plot_cv_roc_panels,
    plot_external_calibration,
    plot_external_confusion_matrices,
    plot_external_pr,
    plot_external_roc,
    plot_test_calibration_all_models,
    plot_test_confusion_matrices,
    plot_test_pr_all_models,
    plot_test_roc_all_models,
    plot_test_radar,
    plot_feature_importance,
)
from src.reverse_planning import (
    ReversePlanner,
    build_reverse_planning_manifest,
    plot_contours,
    plot_surfaces,
    save_deployment_bundle,
)
from src.shap_analysis import run_shap_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce BSSO feature selection, model development, external validation, and reverse planning."
    )
    parser.add_argument("--development", required=True, help="Development cohort (.xlsx or .csv).")
    parser.add_argument("--temporal", required=True, help="Temporal external cohort (.xlsx or .csv).")
    parser.add_argument("--geographical", required=True, help="Geographical external cohort (.xlsx or .csv).")
    parser.add_argument(
        "--feasibility",
        default=None,
        help="Optional guide-assisted clinical feasibility cohort (.xlsx or .csv).",
    )
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--feature-resampling", type=int, default=300)
    parser.add_argument("--rf-rfe-resampling", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=150)
    parser.add_argument("--example-mrt", type=float, default=None)
    parser.add_argument("--example-pmbt", type=float, default=None)
    parser.add_argument("--skip-feature-selection", action="store_true")
    parser.add_argument("--skip-complementary-selection", action="store_true")
    parser.add_argument("--run-causal", action="store_true")
    parser.add_argument("--causal-bootstrap", type=int, default=1000)
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--shap-evaluation-size", type=int, default=None)
    parser.add_argument(
        "--local-shap-case-index",
        type=int,
        default=92,
        help="Fallback zero-based row position for a local SHAP explanation.",
    )
    parser.add_argument(
        "--local-shap-sample-id",
        default="92",
        help="Preferred sample identifier for the local SHAP explanation when a sample column is available.",
    )
    return parser.parse_args()


def _ensure_expected_features(selected_features: list[str]) -> None:
    if set(selected_features) != set(FINAL_FEATURES):
        raise RuntimeError(
            "The Top-50 rule did not reproduce the prespecified four-feature set. "
            f"Observed: {selected_features}; expected: {FINAL_FEATURES}. "
            "Do not continue without reconciling the data and manuscript."
        )


def _test_uncertainty_and_comparisons(modeling: dict[str, object], output_dir: Path, n_bootstrap: int) -> None:
    results = modeling["results"]
    y_test = modeling["y_test"]
    test_patient_ids = modeling["test_df"][PATIENT_ID_COL].to_numpy()
    n_classes = len(modeling["class_labels"])

    ci_rows = []
    for model_name in MODEL_ORDER:
        probabilities = results[model_name].test_probabilities
        point = multiclass_micro_auc(y_test, probabilities, n_classes)
        _, lower, upper, _ = patient_cluster_bootstrap_ci(
            y_test,
            probabilities,
            test_patient_ids,
            metric=lambda y, p: multiclass_micro_auc(y, p, n_classes),
            n_bootstrap=n_bootstrap,
            random_state=RANDOM_STATE_MODEL,
        )
        ci_rows.append(
            {
                "model": model_name,
                "auc_micro": point,
                "ci_lower": lower,
                "ci_upper": upper,
                "bootstrap_unit": "patient",
                "n_bootstrap": n_bootstrap,
            }
        )
    pd.DataFrame(ci_rows).to_csv(output_dir / "internal_test_auc_cluster_bootstrap_ci.csv", index=False)

    delong_rows = []
    for model_a, model_b in itertools.combinations(MODEL_ORDER, 2):
        comparison = paired_delong_micro_multiclass(
            y_test,
            results[model_a].test_probabilities,
            results[model_b].test_probabilities,
            n_classes,
        )
        delong_rows.append({"model_a": model_a, "model_b": model_b, **comparison})
    pd.DataFrame(delong_rows).to_csv(output_dir / "paired_delong_micro_auc.csv", index=False)

    calibration_rows = []
    for model_name in MODEL_ORDER:
        probabilities = results[model_name].test_probabilities
        brier = classwise_brier_scores(y_test, probabilities, modeling["class_labels"])
        for class_index, row in brier.iterrows():
            binary = (y_test == class_index).astype(int)
            hl = hosmer_lemeshow_binary(binary, probabilities[:, class_index], n_bins=10)
            calibration_rows.append(
                {
                    "model": model_name,
                    "class": row["class"],
                    "brier_score": row["brier_score"],
                    **hl,
                }
            )
    pd.DataFrame(calibration_rows).to_csv(output_dir / "internal_test_calibration_summary.csv", index=False)


def _external_outputs(
    modeling: dict[str, object],
    temporal: pd.DataFrame,
    geographical: pd.DataFrame,
    output_dir: Path,
    n_bootstrap: int,
) -> dict[str, dict[str, object]]:
    external_results = {
        "Temporal external cohort": evaluate_external_cohort(
            temporal, modeling["locked_model"], modeling["encoder"], modeling["features"]
        ),
        "Geographical external cohort": evaluate_external_cohort(
            geographical, modeling["locked_model"], modeling["encoder"], modeling["features"]
        ),
    }

    overall_rows = []
    class_rows = []
    n_classes = len(modeling["class_labels"])
    for cohort_name, result in external_results.items():
        patient_ids = result["data"][PATIENT_ID_COL].to_numpy()
        point = multiclass_micro_auc(result["y"], result["probabilities"], n_classes)
        _, lower, upper, _ = patient_cluster_bootstrap_ci(
            result["y"],
            result["probabilities"],
            patient_ids,
            metric=lambda y, p: multiclass_micro_auc(y, p, n_classes),
            n_bootstrap=n_bootstrap,
            random_state=RANDOM_STATE_MODEL,
        )
        overall_rows.append(
            {
                "cohort": cohort_name,
                **result["metrics"],
                "auc_micro_ci_lower": lower,
                "auc_micro_ci_upper": upper,
                "patients": int(result["data"][PATIENT_ID_COL].nunique()),
                "sides": int(len(result["data"])),
                "bootstrap_unit": "patient",
            }
        )
        class_table = result["per_class"].copy()
        class_table.insert(0, "cohort", cohort_name)
        class_rows.append(class_table)

    pd.DataFrame(overall_rows).to_csv(output_dir / "external_validation_metrics.csv", index=False)
    pd.concat(class_rows, ignore_index=True).to_csv(
        output_dir / "external_validation_per_class_metrics.csv", index=False
    )
    return external_results


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    artifact_dir = output_dir / "artifacts"
    feature_dir = output_dir / "feature_selection"
    model_dir = output_dir / "modeling"
    reverse_dir = output_dir / "reverse_planning"
    for directory in [figure_dir, artifact_dir, feature_dir, model_dir, reverse_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    development = read_cohort(args.development, require_all_candidates=True)
    temporal = read_cohort(args.temporal, require_all_candidates=False)
    geographical = read_cohort(args.geographical, require_all_candidates=False)
    run_descriptive_tables(development, output_dir / "descriptive" / "development")
    run_descriptive_tables(temporal, output_dir / "descriptive" / "temporal_external")
    run_descriptive_tables(geographical, output_dir / "descriptive" / "geographical_external")
    validate_bilateral_structure(development, strict=True).to_csv(
        output_dir / "development_bilateral_structure_check.csv", index=False
    )
    validate_bilateral_structure(temporal, strict=True).to_csv(
        output_dir / "temporal_bilateral_structure_check.csv", index=False
    )
    validate_bilateral_structure(geographical, strict=True).to_csv(
        output_dir / "geographical_bilateral_structure_check.csv", index=False
    )

    split = patient_level_stratified_split(
        development, test_fraction=0.30, random_state=RANDOM_STATE_SPLIT
    )
    save_dataset_manifest(
        {
            "development_full": development,
            "training": split.train,
            "internal_test": split.test,
            "temporal_external": temporal,
            "geographical_external": geographical,
        },
        output_dir / "dataset_manifest.csv",
    )
    pd.DataFrame({PATIENT_ID_COL: split.train_patient_ids}).to_csv(
        output_dir / "training_patient_ids.csv", index=False
    )
    pd.DataFrame({PATIENT_ID_COL: split.test_patient_ids}).to_csv(
        output_dir / "internal_test_patient_ids.csv", index=False
    )

    correlation_and_vif(split.train, feature_dir)
    if not args.skip_feature_selection:
        selection = exhaustive_best_subset(split.train, feature_dir)
        _ensure_expected_features(selection["summary"]["selected_features"])
        if not args.skip_complementary_selection:
            complementary_stability_analyses(
                split.train,
                feature_dir,
                n_resampling=args.feature_resampling,
                n_rf_resampling=args.rf_rfe_resampling,
                random_state=RANDOM_STATE_MODEL,
            )

    modeling = train_and_compare_models(
        split.train,
        split.test,
        model_dir,
        features=FINAL_FEATURES,
        random_state=RANDOM_STATE_MODEL,
    )
    joblib.dump(modeling["locked_model"], artifact_dir / "locked_model.joblib")
    joblib.dump(modeling["encoder"], artifact_dir / "label_encoder.joblib")

    _test_uncertainty_and_comparisons(modeling, model_dir, args.n_bootstrap)
    external_results = _external_outputs(
        modeling, temporal, geographical, model_dir, args.n_bootstrap
    )

    plot_cv_roc_panels(
        modeling["results"],
        modeling["y_train"],
        modeling["class_labels"],
        figure_dir / "supplementary_figure_s5_cv_roc.png",
    )
    plot_test_confusion_matrices(
        modeling["results"],
        modeling["y_test"],
        modeling["class_labels"],
        figure_dir / "supplementary_figure_s6_test_confusion_matrices.png",
    )
    plot_test_radar(
        modeling["results"],
        figure_dir / "figure_4b_internal_test_radar.png",
    )
    plot_test_roc_all_models(
        modeling["results"],
        modeling["y_test"],
        modeling["class_labels"],
        figure_dir / "figure_4c_internal_test_roc.png",
    )
    plot_test_pr_all_models(
        modeling["results"],
        modeling["y_test"],
        modeling["class_labels"],
        figure_dir / "figure_4d_internal_test_pr.png",
    )
    plot_test_calibration_all_models(
        modeling["results"],
        modeling["y_test"],
        modeling["class_labels"],
        figure_dir / "figure_4e_internal_test_calibration.png",
    )
    plot_feature_importance(
        modeling["results"],
        modeling["features"],
        figure_dir / "supplementary_figure_s7_feature_importance.png",
    )
    plot_external_confusion_matrices(
        external_results,
        modeling["class_labels"],
        figure_dir / "supplementary_figure_s8_external_confusion_matrices.png",
    )
    plot_external_roc(
        external_results,
        modeling["class_labels"],
        figure_dir / "figure_4f_external_roc.png",
        n_bootstrap=args.n_bootstrap,
    )
    plot_external_pr(
        external_results,
        modeling["class_labels"],
        figure_dir / "figure_4g_external_pr.png",
    )
    plot_external_calibration(
        external_results,
        modeling["class_labels"],
        figure_dir / "figure_4h_external_calibration.png",
    )

    reverse_manifest = build_reverse_planning_manifest(
        split.train,
        modeling["class_labels"],
        reverse_dir / "reverse_planning_manifest.json",
    )
    bundle_path = artifact_dir / "model_bundle.joblib"
    save_deployment_bundle(
        modeling["locked_model"],
        modeling["encoder"],
        FINAL_FEATURES,
        reverse_manifest,
        bundle_path,
        metadata={
            "locked_model_name": modeling["locked_model_name"],
            "algorithm_selection_source": "training-set five-fold patient-level cross-validation",
            "internal_test_used_for_selection": False,
        },
    )

    planner = ReversePlanner(joblib.load(bundle_path), grid_size=args.grid_size)
    fixed_domain = reverse_manifest["fixed_input_domain"]
    example_mrt = args.example_mrt if args.example_mrt is not None else fixed_domain["MRT"]["median"]
    example_pmbt = args.example_pmbt if args.example_pmbt is not None else fixed_domain["PMBT"]["median"]
    planning = planner.plan_all(example_mrt, example_pmbt)
    planning["summary"].to_csv(reverse_dir / "example_reverse_planning_summary.csv", index=False)
    plot_contours(
        planning["selected_results"],
        figure_dir / "supplementary_figure_s16_reverse_planning_contours.png",
    )
    plot_surfaces(
        planning["selected_results"],
        figure_dir / "figure_6a_reverse_planning_surfaces.png",
    )

    if args.run_shap:
        run_shap_analysis(
            modeling["locked_model"],
            development,
            FINAL_FEATURES,
            modeling["class_labels"],
            output_dir / "shap",
            evaluation_size=args.shap_evaluation_size,
            local_case_index=args.local_shap_case_index,
            local_sample_id=args.local_shap_sample_id,
        )

    if args.run_causal:
        run_causal_analysis(
            development,
            output_dir / "causal",
            n_bootstrap=args.causal_bootstrap,
            random_state=RANDOM_STATE_MODEL,
        )

    if args.feasibility:
        feasibility = read_feasibility_data(args.feasibility)
        analyze_feasibility(feasibility, output_dir / "clinical_feasibility")

    s4_relative_file = (
        "feature_selection/supplementary_figure_s4_composite.png"
        if (feature_dir / "supplementary_figure_s4_composite.png").exists()
        else "feature_selection/supplementary_figure_s4_performance_complexity.png"
    )
    figure_manifest = pd.DataFrame(
        [
            {"manuscript_item": "Fig. 4a", "file": "feature_selection/figure_4a_top50_recurrence.png"},
            {"manuscript_item": "Fig. S2", "file": "feature_selection/supplementary_figure_s2_pairplot.png"},
            {"manuscript_item": "Fig. S3", "file": "feature_selection/supplementary_figure_s3_correlation_vif.png"},
            {"manuscript_item": "Fig. S4", "file": s4_relative_file},
            {"manuscript_item": "Fig. S5", "file": "figures/supplementary_figure_s5_cv_roc.png"},
            {"manuscript_item": "Fig. S6", "file": "figures/supplementary_figure_s6_test_confusion_matrices.png"},
            {"manuscript_item": "Fig. S7", "file": "figures/supplementary_figure_s7_feature_importance.png"},
            {"manuscript_item": "Fig. S8", "file": "figures/supplementary_figure_s8_external_confusion_matrices.png"},
            {"manuscript_item": "Fig. S16", "file": "figures/supplementary_figure_s16_reverse_planning_contours.png"},
            {"manuscript_item": "Fig. 4b", "file": "figures/figure_4b_internal_test_radar.png"},
            {"manuscript_item": "Fig. 4c", "file": "figures/figure_4c_internal_test_roc.png"},
            {"manuscript_item": "Fig. 4d", "file": "figures/figure_4d_internal_test_pr.png"},
            {"manuscript_item": "Fig. 4e", "file": "figures/figure_4e_internal_test_calibration.png"},
            {"manuscript_item": "Fig. 4f", "file": "figures/figure_4f_external_roc.png"},
            {"manuscript_item": "Fig. 4g", "file": "figures/figure_4g_external_pr.png"},
            {"manuscript_item": "Fig. 4h", "file": "figures/figure_4h_external_calibration.png"},
            {"manuscript_item": "Fig. 6a", "file": "figures/figure_6a_reverse_planning_surfaces.png"},
        ]
    )
    figure_manifest.to_csv(output_dir / "figure_manifest.csv", index=False)

    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "development_file": "user-supplied path",
                "temporal_file": "user-supplied path",
                "geographical_file": "user-supplied path",
                "locked_model": modeling["locked_model_name"],
                "selected_features": FINAL_FEATURES,
                "model_selection": "training-set CV accuracy; CV micro-AUC tie-breaker",
                "internal_test_used_for_selection": False,
                "reverse_planning_domain_source": "training set only",
                "reverse_planning_fallback": "IQR strict -> P10-P90 strict -> no recommendation",
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
