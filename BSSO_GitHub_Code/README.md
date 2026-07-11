# BSSO Lingual Fracture-Line Prediction and Reverse-Planning Analysis

This repository contains the manuscript-aligned Python analysis code for conditional prediction and parameterized planning of lingual fracture-line patterns after bilateral sagittal split osteotomy (BSSO).

## Repository contents

```text
.
├── run_analysis.py
├── README.md
├── KNOWN_LIMITATIONS.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── causal.py
    ├── constants.py
    ├── data.py
    ├── descriptive.py
    ├── feasibility.py
    ├── feature_selection.py
    ├── legacy.py
    ├── metrics.py
    ├── modeling.py
    ├── plotting.py
    ├── reliability.py
    ├── reverse_planning.py
    └── shap_analysis.py
```

## Analytical workflow

The code implements the following manuscript-aligned workflow:

1. Patient-level 7:3 split of the retrospective development cohort into training and internal test sets.
2. Feature selection performed only in the training set.
3. Exhaustive evaluation of all 4,095 non-empty subsets of the 12 candidate predictors using multinomial logistic regression and five-fold patient-level stratified group cross-validation.
4. Primary predictor selection using inclusion frequency strictly greater than 0.60 among the 50 highest-ranking subsets.
5. LASSO, Elastic Net, mRMR, RF-RFE, and Boruta-style shadow-feature screening as complementary analyses.
6. Training and comparison of logistic regression, support vector machine, random forest, XGBoost, and CatBoost models.
7. Hyperparameter optimization by five-fold patient-level cross-validation in the training set.
8. Final algorithm locking according to training-set cross-validated accuracy, with cross-validated micro-AUC used as the tie-breaker.
9. Independent evaluation of the locked model in the internal test set and unchanged application to temporal and geographical external validation cohorts.
10. SHAP-based interpretation, exploratory GEE/E-value/g-computation analyses, reverse-planning probability mapping, and clinical feasibility summaries when requested.
11. Reverse-planning domains derived only from the predefined training set, using the class-specific IQR first and the 10th–90th percentile range second.
12. Suppression of planning recommendations when the target class is not the highest-probability class or no connected near-optimal region is supported.

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Input data

Patient-level clinical and imaging data are not included because they are subject to ethical and privacy restrictions. Input paths are supplied at runtime; no local absolute paths or institution-specific filenames are embedded in the code.

Each analytical row should represent one mandibular side. A de-identified `patient_id` shared by both sides of the same patient is required for patient-level grouping and resampling.

The development dataset should contain the model outcome and the candidate predictors used in the manuscript, including:

```text
patient_id
fracture_type
LLBCE
PMBT
MRT
Depth of A
RAPL
ART
RH
LSND
age
sex
jaw deformity type
third molar status
```

Column-name aliases are normalized by the data-loading functions. Review `src/constants.py` and `src/data.py` before running the analysis with a new data dictionary.

## Run the complete analysis

```bash
python run_analysis.py \
  --development path/to/development_cohort.xlsx \
  --temporal path/to/temporal_external_cohort.xlsx \
  --geographical path/to/geographical_external_cohort.xlsx \
  --output outputs
```

Optional SHAP, causal-support, and clinical-feasibility analyses:

```bash
python run_analysis.py \
  --development path/to/development_cohort.xlsx \
  --temporal path/to/temporal_external_cohort.xlsx \
  --geographical path/to/geographical_external_cohort.xlsx \
  --feasibility path/to/clinical_feasibility_cohort.xlsx \
  --output outputs \
  --run-shap \
  --run-causal
```

For a model-only verification run after feature-selection outputs have already been confirmed:

```bash
python run_analysis.py \
  --development path/to/development_cohort.xlsx \
  --temporal path/to/temporal_external_cohort.xlsx \
  --geographical path/to/geographical_external_cohort.xlsx \
  --output outputs \
  --skip-feature-selection
```

Display all command-line options:

```bash
python run_analysis.py --help
```

## Principal outputs

Depending on the selected options, the pipeline generates:

- training and internal-test patient identifiers;
- descriptive statistics and candidate-predictor diagnostics;
- correlation matrices and variance inflation factors;
- exhaustive best-subset and Top-50 recurrence results;
- complementary feature-selection outputs;
- cross-validated model-selection records and optimized hyperparameters;
- internal-test and external-validation metrics;
- ROC, precision–recall, calibration, and confusion-matrix figures;
- patient-clustered bootstrap confidence intervals;
- SHAP outputs;
- exploratory GEE, E-value, and g-computation outputs;
- reverse-planning domains, probability surfaces, contour plots, and recommendation summaries;
- clinical-feasibility agreement and parameter-transfer error summaries.

## Reproducibility notes

- Feature selection, standardization, and hyperparameter optimization are restricted to the training set.
- Bilateral observations are grouped by patient during splitting, cross-validation, and clustered resampling.
- The internal test set is not used to define the reverse-planning empirical domain.
- Temporal and geographical external cohorts are not used for model tuning or recommendation-domain construction.
- Random seeds are defined centrally in `src/constants.py`.

## Verification before upload

Run a syntax check:

```bash
python -m compileall -q run_analysis.py src
```

Inspect the command-line interface:

```bash
python run_analysis.py --help
```

Numerical reproduction of the manuscript requires the original de-identified analytical datasets and the exact software environment used for the final locked analysis.

## Intended use

This repository is provided for research and reproducibility purposes. The reverse-planning outputs are probabilistic model-derived candidate values and are not deterministic surgical instructions. See `KNOWN_LIMITATIONS.md` before reuse or clinical interpretation.
