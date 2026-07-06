import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import joblib
import warnings
import os
from collections import Counter
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
if not os.path.exists('result'):
    os.makedirs('result')
df = pd.read_excel('your_external_validation_data.xlsx')
features = ['LLBCE', 'PMBT', 'MRT', 'Depth  of A']
target = 'Type of fracture line'
df['patient_id'] = (df['sample'] - 1) // 2 + 1
patient_info = df.groupby('patient_id').agg({target: lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]}).reset_index()
patient_info.columns = ['patient_id', 'fracture_type']
pass
pass
pass

def patient_level_stratified_split(patient_info, test_size=0.3, random_state=70, tolerance=0.15):
    np.random.seed(random_state)
    total_patients = len(patient_info)
    target_test_size = int(total_patients * test_size)
    class_patients = {}
    for cls in patient_info['fracture_type'].unique():
        class_patients[cls] = patient_info[patient_info['fracture_type'] == cls]['patient_id'].tolist()
    class_proportions = patient_info['fracture_type'].value_counts(normalize=True).to_dict()
    test_patients = []
    train_patients = []
    for cls, patients in class_patients.items():
        n_test = int(target_test_size * class_proportions[cls])
        n_test = max(1, n_test) if len(patients) > 1 else len(patients)
        selected = np.random.choice(patients, size=n_test, replace=False).tolist()
        test_patients.extend(selected)
        train_patients.extend([p for p in patients if p not in selected])
    test_dist = patient_info[patient_info['patient_id'].isin(test_patients)]['fracture_type'].value_counts(normalize=True)
    train_dist = patient_info[patient_info['patient_id'].isin(train_patients)]['fracture_type'].value_counts(normalize=True)
    pass
    pass
    pass
    for cls in class_proportions.keys():
        train_prop = train_dist.get(cls, 0)
        target_prop = class_proportions[cls]
        if abs(train_prop - target_prop) > tolerance:
            pass
    return (train_patients, test_patients)
train_patients, test_patients = patient_level_stratified_split(patient_info, test_size=0.3, random_state=70)
train_df = df[df['patient_id'].isin(train_patients)].copy()
test_df = df[df['patient_id'].isin(test_patients)].copy()
pass
pass
X_train = train_df[features]
X_test = test_df[features]
y_train = train_df[target].values
y_test = test_df[target].values
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
class_names = [str(name) for name in le.classes_]
pass
pass
pass
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pass
train_export = train_df.copy()
test_export = test_df.copy()
train_export['dataset'] = ''
test_export['dataset'] = ''
train_export['target_encoded'] = y_train_encoded
test_export['target_encoded'] = y_test_encoded
with pd.ExcelWriter('your_output_path/your_output.xlsx') as writer:
    train_export.to_excel(writer, sheet_name='', index=False)
    test_export.to_excel(writer, sheet_name='', index=False)
pass
pass
pass
import os
import re
import json
import time
import itertools
import warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score, log_loss
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
DATA_PATH = Path('your_data_path/your_training_data.xlsx')
SHEET_NAME = 0
SAMPLE_ID_COL = 'sample'
TARGET_COL = 'Type.of.fracture.line'
CONTINUOUS_FEATURES = ['LLBCE', 'PMBT', 'MRT', 'Depth.of.A', 'RAPL', 'ART', 'RH', 'LSND', 'age']
RAW_CATEGORICAL_FEATURES = ['sex', 'type of jaw deformity', 'third molar presence']
CANDIDATE_FEATURES = ['LLBCE', 'PMBT', 'MRT', 'Depth.of.A', 'RAPL', 'ART', 'RH', 'LSND', 'age', 'sex_male', 'jaw_deformity_type3', 'third_molar_yes']
OUTPUT_DIR = Path('your_output_path')
TABLE_DIR = OUTPUT_DIR / 'tables'
FIG_DIR = OUTPUT_DIR / 'figures'
METHOD_TABLE_DIR = TABLE_DIR / 'method_specific_diagnostics'
METHOD_FIG_DIR = FIG_DIR / 'method_specific_diagnostics'
for d in [OUTPUT_DIR, TABLE_DIR, FIG_DIR, METHOD_TABLE_DIR, METHOD_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_RESAMPLING = 300
N_RF_RFE_RESAMPLING = 100
SUBSAMPLE_FRACTION = 0.8
CV_SPLITS = 5
INNER_CV_SPLITS = 3
MRMR_TOP_K = 5
RF_RFE_MIN_FEATURES = 2
BORUTA_N_ESTIMATORS = 500
TOP_SUBSET_N = 50
TOP_SUBSET_SENSITIVITY = [25, 50, 100]
MAX_EXHAUSTIVE_FEATURES = 15
TOPK_MIN_FEATURES = 2
PRIMARY_SIZE_METRIC = 'Macro_F1'
ONE_SE_RULE = True
RUN_METHOD_DIAGNOSTICS = True
DPI = 600
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def savefig(fig, filename):
    fig.savefig(FIG_DIR / f'{filename}.png', dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close(fig)

def save_method_fig(fig, filename):
    fig.savefig(METHOD_FIG_DIR / f'{filename}.png', dpi=DPI, bbox_inches='tight')
    fig.savefig(METHOD_FIG_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close(fig)

def normalize_colnames(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def normalize_depth_column(df):
    df = df.copy()
    aliases = {'Depth of A': 'Depth.of.A', 'Depth  of A': 'Depth.of.A', 'Depth_of_A': 'Depth.of.A', 'Depth.of.A': 'Depth.of.A'}
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    return df

def make_patient_id_from_sample_id(sample_ids):
    return ((pd.Series(sample_ids).astype(int) - 1) // 2 + 1).values

def get_safe_n_splits(y, groups, requested_splits=5):
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    n_groups = len(pd.unique(groups))
    n_splits = min(requested_splits, min_class_count, n_groups)
    if n_splits < 2:
        raise ValueError('Not enough samples/classes/groups for cross-validation.')
    return int(n_splits)

def make_group_stratified_cv(y, groups, requested_splits=5, random_state=42):
    n_splits = get_safe_n_splits(y, groups, requested_splits=requested_splits)
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def patient_level_subsample_indices(y, groups, fraction=0.8, random_state=None, max_try=300):
    rng = np.random.default_rng(random_state)
    unique_groups = np.array(pd.unique(groups))
    n_select = max(2, int(round(len(unique_groups) * fraction)))
    all_classes = set(np.unique(y))
    for _ in range(max_try):
        sampled_groups = rng.choice(unique_groups, size=n_select, replace=False)
        idx = np.where(np.isin(groups, sampled_groups))[0]
        if set(np.unique(y[idx])) == all_classes:
            return idx
    return idx

def align_proba_to_classes(model, proba, n_classes):
    proba = np.asarray(proba)
    if hasattr(model, 'classes_'):
        classes = np.asarray(model.classes_).astype(int)
    elif hasattr(model, 'named_steps'):
        classes = None
        for step in reversed(model.named_steps.values()):
            if hasattr(step, 'classes_'):
                classes = np.asarray(step.classes_).astype(int)
                break
    else:
        classes = None
    if classes is None:
        if proba.shape[1] == n_classes:
            return proba
        raise ValueError('Cannot infer model classes.')
    aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
    for col_idx, cls in enumerate(classes):
        if 0 <= int(cls) < n_classes:
            aligned[:, int(cls)] = proba[:, col_idx]
    aligned = aligned + 1e-12
    aligned = aligned / aligned.sum(axis=1, keepdims=True)
    return aligned

def multiclass_brier_score(y_true, y_proba, n_classes):
    y_onehot = label_binarize(y_true, classes=np.arange(n_classes))
    return np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))

def calculate_metrics(y_true, y_pred, y_proba, n_classes):
    out = {}
    out['Accuracy'] = accuracy_score(y_true, y_pred)
    out['Precision_Macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    out['Recall_Macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    out['Macro_F1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    out['Balanced_Accuracy'] = balanced_accuracy_score(y_true, y_pred)
    try:
        out['AUC_OVR_Macro'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    except Exception:
        out['AUC_OVR_Macro'] = np.nan
    try:
        y_onehot = label_binarize(y_true, classes=np.arange(n_classes))
        out['AP_Macro'] = average_precision_score(y_onehot, y_proba, average='macro')
    except Exception:
        out['AP_Macro'] = np.nan
    try:
        out['Log_Loss'] = log_loss(y_true, y_proba, labels=np.arange(n_classes))
    except Exception:
        out['Log_Loss'] = np.nan
    out['Brier_Multiclass'] = multiclass_brier_score(y_true, y_proba, n_classes)
    return out

def cv_oof_evaluate(estimator, X, y, groups, features, n_splits=5, random_state=42):
    X_sub = X[features].copy()
    n_classes = len(np.unique(y))
    cv = make_group_stratified_cv(y, groups, requested_splits=n_splits, random_state=random_state)
    y_pred_all = np.full(len(y), -1, dtype=int)
    y_proba_all = np.zeros((len(y), n_classes), dtype=float)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_sub, y, groups), start=1):
        model = clone(estimator)
        model.fit(X_sub.iloc[tr_idx], y[tr_idx])
        pred = np.asarray(model.predict(X_sub.iloc[va_idx])).ravel().astype(int)
        proba = model.predict_proba(X_sub.iloc[va_idx])
        proba = align_proba_to_classes(model, proba, n_classes)
        y_pred_all[va_idx] = pred
        y_proba_all[va_idx, :] = proba
        fold_metrics = calculate_metrics(y[va_idx], pred, proba, n_classes)
        fold_metrics['Fold'] = fold
        fold_rows.append(fold_metrics)
    overall = calculate_metrics(y, y_pred_all, y_proba_all, n_classes)
    fold_df = pd.DataFrame(fold_rows)
    summary = overall.copy()
    for col in ['Accuracy', 'Macro_F1', 'Balanced_Accuracy', 'AUC_OVR_Macro', 'AP_Macro', 'Log_Loss', 'Brier_Multiclass']:
        if col in fold_df.columns:
            summary[f'{col}_Fold_Mean'] = fold_df[col].mean()
            summary[f'{col}_Fold_SD'] = fold_df[col].std(ddof=1)
            summary[f'{col}_Fold_SE'] = fold_df[col].std(ddof=1) / np.sqrt(len(fold_df))
    return (summary, fold_df, y_pred_all, y_proba_all)

def make_selection_estimator(random_state=42):
    return Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(solver='lbfgs', max_iter=5000, class_weight='balanced', multi_class='auto', random_state=random_state))])

def safe_rank(values, higher_is_better=True):
    values = pd.Series(values).astype(float)
    if values.isna().all():
        values = pd.Series(np.zeros(len(values)))
    values = values.fillna(values.median())
    if higher_is_better:
        return rankdata(-values.values, method='average')
    return rankdata(values.values, method='average')

def minmax_scale(x):
    x = np.asarray(x, dtype=float)
    if np.nanmax(x) - np.nanmin(x) < 1e-12:
        return np.zeros_like(x)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def coef_norm_from_multiclass(model):
    coef = model.coef_
    if coef.ndim == 1:
        return np.abs(coef)
    return np.sqrt(np.sum(coef ** 2, axis=0))

def rfecv_results_to_dataframe(rfecv, total_features, min_features_to_select):
    if not hasattr(rfecv, 'cv_results_'):
        if hasattr(rfecv, 'grid_scores_'):
            scores = np.asarray(rfecv.grid_scores_)
            if scores.ndim == 2:
                mean_scores = scores.mean(axis=1)
                std_scores = scores.std(axis=1, ddof=1)
            else:
                mean_scores = scores
                std_scores = np.full_like(mean_scores, np.nan, dtype=float)
            n_steps = len(mean_scores)
            if n_steps == total_features - min_features_to_select + 1:
                n_features_seq = np.arange(min_features_to_select, total_features + 1)
            else:
                n_features_seq = np.arange(1, n_steps + 1)
            return pd.DataFrame({'n_features': n_features_seq, 'mean_test_score': mean_scores, 'std_test_score': std_scores})
        raise ValueError('RFECV object has neither cv_results_ nor grid_scores_.')
    cv_results = rfecv.cv_results_
    mean_scores = np.asarray(cv_results['mean_test_score']).ravel()
    n_steps = len(mean_scores)
    if 'std_test_score' in cv_results:
        std_scores = np.asarray(cv_results['std_test_score']).ravel()
    else:
        std_scores = np.full(n_steps, np.nan)
    if 'n_features' in cv_results:
        n_features_raw = np.asarray(cv_results['n_features'])
        if n_features_raw.ndim == 1 and len(n_features_raw) == n_steps:
            n_features_seq = n_features_raw.astype(int)
        else:
            n_features_seq = np.arange(min_features_to_select, min_features_to_select + n_steps)
    elif n_steps == total_features - min_features_to_select + 1:
        n_features_seq = np.arange(min_features_to_select, total_features + 1)
    else:
        n_features_seq = np.arange(1, n_steps + 1)
    clean_df = pd.DataFrame({'n_features': n_features_seq, 'mean_test_score': mean_scores, 'std_test_score': std_scores})
    return clean_df.sort_values('n_features').reset_index(drop=True)
start_time = time.time()
if not DATA_PATH.exists():
    raise FileNotFoundError(f'：{DATA_PATH}')
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
df = normalize_colnames(df)
df = normalize_depth_column(df)
pass
pass
required_raw_cols = [SAMPLE_ID_COL, TARGET_COL] + CONTINUOUS_FEATURES + RAW_CATEGORICAL_FEATURES
missing_raw_cols = [c for c in required_raw_cols if c not in df.columns]
if missing_raw_cols:
    raise ValueError(f'：{missing_raw_cols}')
for col in CONTINUOUS_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['sex_male'] = df['sex'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0, 'm': 1, 'f': 0, '': 1, '': 0, '1': 1, '0': 0})
df['jaw_deformity_type3'] = pd.to_numeric(df['type of jaw deformity'], errors='coerce').eq(3).astype(int)
df['third_molar_yes'] = df['third molar presence'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0, 'y': 1, 'n': 0, 'present': 1, 'absent': 0, '': 1, '': 0, '1': 1, '0': 0})
for col in ['sex_male', 'third_molar_yes']:
    if df[col].isna().any():
        pass
df['patient_id'] = make_patient_id_from_sample_id(df[SAMPLE_ID_COL])
analysis_cols = [SAMPLE_ID_COL, 'patient_id', TARGET_COL] + CANDIDATE_FEATURES
df_model = df[analysis_cols].dropna().copy()
X_train = df_model[CANDIDATE_FEATURES].reset_index(drop=True)
y_raw = df_model[TARGET_COL].reset_index(drop=True)
groups_train = df_model['patient_id'].values
le = LabelEncoder()
y_train = le.fit_transform(y_raw)
class_names = [str(c) for c in le.classes_]
n_classes = len(class_names)
pass
pass
pass
pass
pass
training_dist_rows = []
for k, cname in enumerate(class_names):
    training_dist_rows.append({'Class': cname, 'Encoded': k, 'Count': int(np.sum(y_train == k)), 'Proportion': float(np.mean(y_train == k))})
training_dist_df = pd.DataFrame(training_dist_rows)
training_dist_df.to_excel(TABLE_DIR / 'predefined_training_set_distribution.xlsx', index=False)
features = CANDIDATE_FEATURES
p = len(features)
frequency_records = {'LASSO': np.zeros(p), 'ElasticNet': np.zeros(p), 'mRMR': np.zeros(p), 'RF_RFE': np.zeros(p), 'Boruta': np.zeros(p)}
success_counts = {'LASSO': 0, 'ElasticNet': 0, 'mRMR': 0, 'RF_RFE': 0, 'Boruta': 0}
pass
lasso_C_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
for i in range(N_RESAMPLING):
    idx = patient_level_subsample_indices(y_train, groups_train, fraction=SUBSAMPLE_FRACTION, random_state=RANDOM_STATE + i)
    X_sub = X_train.iloc[idx]
    y_sub = y_train[idx]
    g_sub = groups_train[idx]
    estimator = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(penalty='l1', solver='saga', max_iter=8000, class_weight='balanced', multi_class='auto', random_state=RANDOM_STATE + i))])
    cv_inner = make_group_stratified_cv(y_sub, g_sub, requested_splits=INNER_CV_SPLITS, random_state=RANDOM_STATE + i)
    gs = GridSearchCV(estimator, param_grid={'clf__C': lasso_C_grid}, scoring='f1_macro', cv=cv_inner, n_jobs=-1, refit=True, error_score=np.nan)
    try:
        gs.fit(X_sub, y_sub, groups=g_sub)
        model = gs.best_estimator_
        coef = model.named_steps['clf'].coef_
        selected = np.any(np.abs(coef) > 1e-06, axis=0)
        frequency_records['LASSO'] += selected.astype(float)
        success_counts['LASSO'] += 1
    except Exception as e:
        pass
frequency_records['LASSO'] /= max(success_counts['LASSO'], 1)
pass
elastic_C_grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
elastic_l1_grid = [0.2, 0.5, 0.8]
for i in range(N_RESAMPLING):
    idx = patient_level_subsample_indices(y_train, groups_train, fraction=SUBSAMPLE_FRACTION, random_state=RANDOM_STATE + 10000 + i)
    X_sub = X_train.iloc[idx]
    y_sub = y_train[idx]
    g_sub = groups_train[idx]
    estimator = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(penalty='elasticnet', solver='saga', max_iter=8000, class_weight='balanced', multi_class='auto', random_state=RANDOM_STATE + 10000 + i))])
    cv_inner = make_group_stratified_cv(y_sub, g_sub, requested_splits=INNER_CV_SPLITS, random_state=RANDOM_STATE + 10000 + i)
    gs = GridSearchCV(estimator, param_grid={'clf__C': elastic_C_grid, 'clf__l1_ratio': elastic_l1_grid}, scoring='f1_macro', cv=cv_inner, n_jobs=-1, refit=True, error_score=np.nan)
    try:
        gs.fit(X_sub, y_sub, groups=g_sub)
        model = gs.best_estimator_
        coef = model.named_steps['clf'].coef_
        selected = np.any(np.abs(coef) > 1e-06, axis=0)
        frequency_records['ElasticNet'] += selected.astype(float)
        success_counts['ElasticNet'] += 1
    except Exception as e:
        pass
frequency_records['ElasticNet'] /= max(success_counts['ElasticNet'], 1)
pass

def mrmr_select_features(X_df, y, feature_names, k=5, random_state=42):
    X_values = X_df[feature_names].values
    relevance = mutual_info_classif(X_values, y, discrete_features=False, random_state=random_state)
    relevance_norm = minmax_scale(relevance)
    corr = pd.DataFrame(X_values, columns=feature_names).corr().abs().fillna(0).values
    selected = []
    remaining = list(range(len(feature_names)))
    for step in range(min(k, len(feature_names))):
        best_idx = None
        best_score = -np.inf
        for j in remaining:
            if len(selected) == 0:
                redundancy = 0
            else:
                redundancy = np.mean([corr[j, s] for s in selected])
            score = relevance_norm[j] - redundancy
            if score > best_score:
                best_score = score
                best_idx = j
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [feature_names[i] for i in selected]
for i in range(N_RESAMPLING):
    idx = patient_level_subsample_indices(y_train, groups_train, fraction=SUBSAMPLE_FRACTION, random_state=RANDOM_STATE + 20000 + i)
    X_sub = X_train.iloc[idx]
    y_sub = y_train[idx]
    try:
        selected_features = mrmr_select_features(X_sub, y_sub, features, k=MRMR_TOP_K, random_state=RANDOM_STATE + 20000 + i)
        selected_mask = np.array([f in selected_features for f in features], dtype=float)
        frequency_records['mRMR'] += selected_mask
        success_counts['mRMR'] += 1
    except Exception as e:
        pass
frequency_records['mRMR'] /= max(success_counts['mRMR'], 1)
pass
for i in range(N_RF_RFE_RESAMPLING):
    idx = patient_level_subsample_indices(y_train, groups_train, fraction=SUBSAMPLE_FRACTION, random_state=RANDOM_STATE + 30000 + i)
    X_sub = X_train.iloc[idx].reset_index(drop=True)
    y_sub = y_train[idx]
    g_sub = groups_train[idx]
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, class_weight='balanced', random_state=RANDOM_STATE + 30000 + i, n_jobs=-1)
    cv_inner = make_group_stratified_cv(y_sub, g_sub, requested_splits=INNER_CV_SPLITS, random_state=RANDOM_STATE + 30000 + i)
    cv_splits = list(cv_inner.split(X_sub, y_sub, g_sub))
    selector = RFECV(estimator=rf, step=1, min_features_to_select=RF_RFE_MIN_FEATURES, cv=cv_splits, scoring='f1_macro', n_jobs=-1)
    try:
        selector.fit(X_sub, y_sub)
        selected = selector.support_.astype(float)
        frequency_records['RF_RFE'] += selected
        success_counts['RF_RFE'] += 1
    except Exception as e:
        pass
frequency_records['RF_RFE'] /= max(success_counts['RF_RFE'], 1)
pass
for i in range(N_RESAMPLING):
    idx = patient_level_subsample_indices(y_train, groups_train, fraction=SUBSAMPLE_FRACTION, random_state=RANDOM_STATE + 40000 + i)
    X_sub = X_train.iloc[idx].reset_index(drop=True)
    y_sub = y_train[idx]
    try:
        rng = np.random.default_rng(RANDOM_STATE + 40000 + i)
        X_shadow = X_sub.copy()
        for col in X_shadow.columns:
            X_shadow[col] = rng.permutation(X_shadow[col].values)
        X_shadow.columns = [f'{c}_shadow' for c in X_shadow.columns]
        X_boruta = pd.concat([X_sub, X_shadow], axis=1)
        rf = RandomForestClassifier(n_estimators=BORUTA_N_ESTIMATORS, max_depth=None, min_samples_leaf=2, class_weight='balanced', random_state=RANDOM_STATE + 40000 + i, n_jobs=-1)
        rf.fit(X_boruta, y_sub)
        importances = rf.feature_importances_
        original_importance = importances[:p]
        shadow_importance = importances[p:]
        shadow_threshold = np.max(shadow_importance)
        confirmed = original_importance > shadow_threshold
        frequency_records['Boruta'] += confirmed.astype(float)
        success_counts['Boruta'] += 1
    except Exception as e:
        pass
frequency_records['Boruta'] /= max(success_counts['Boruta'], 1)
pass
if p > MAX_EXHAUSTIVE_FEATURES:
    raise ValueError(f'p={p}，。 MAX_EXHAUSTIVE_FEATURES。')
estimator_for_subset = make_selection_estimator()
subset_rows = []
all_feature_indices = list(range(p))
for r in range(1, p + 1):
    pass
    for combo in itertools.combinations(all_feature_indices, r):
        subset_features = [features[i] for i in combo]
        try:
            metrics, fold_df, _, _ = cv_oof_evaluate(estimator=estimator_for_subset, X=X_train, y=y_train, groups=groups_train, features=subset_features, n_splits=CV_SPLITS, random_state=RANDOM_STATE + r)
            row = {'n_features': r, 'features': ', '.join(subset_features), 'feature_key': '|'.join(sorted(subset_features))}
            row.update(metrics)
            subset_rows.append(row)
        except Exception as e:
            pass
exhaustive_df = pd.DataFrame(subset_rows)
exhaustive_df['rank_log_loss'] = safe_rank(exhaustive_df['Log_Loss'], higher_is_better=False)
exhaustive_df['rank_accuracy'] = safe_rank(exhaustive_df['Accuracy'], higher_is_better=True)
exhaustive_df['rank_macro_f1'] = safe_rank(exhaustive_df['Macro_F1'], higher_is_better=True)
exhaustive_df['rank_balanced_accuracy'] = safe_rank(exhaustive_df['Balanced_Accuracy'], higher_is_better=True)
exhaustive_df['rank_auc'] = safe_rank(exhaustive_df['AUC_OVR_Macro'], higher_is_better=True)
exhaustive_df['rank_brier'] = safe_rank(exhaustive_df['Brier_Multiclass'], higher_is_better=False)
rank_cols = ['rank_log_loss', 'rank_accuracy', 'rank_macro_f1', 'rank_balanced_accuracy', 'rank_auc', 'rank_brier']
exhaustive_df['multi_objective_mean_rank'] = exhaustive_df[rank_cols].mean(axis=1)
exhaustive_df = exhaustive_df.sort_values(['multi_objective_mean_rank', 'rank_macro_f1', 'rank_accuracy'], ascending=True).reset_index(drop=True)
exhaustive_df['multi_objective_rank'] = np.arange(1, len(exhaustive_df) + 1)
exhaustive_df['multi_objective_percentile'] = exhaustive_df['multi_objective_rank'] / len(exhaustive_df)
exhaustive_df.to_excel(TABLE_DIR / 'exhaustive_all_subsets_multi_objective_rank.xlsx', index=False)
topN_freq_tables = {}
for top_n in TOP_SUBSET_SENSITIVITY:
    top_n_actual = min(top_n, len(exhaustive_df))
    top_df = exhaustive_df.head(top_n_actual).copy()
    rows = []
    for f in features:
        count = int(top_df['features'].apply(lambda s: f in str(s).split(', ')).sum())
        freq = count / top_n_actual
        rows.append({'Feature': f, f'Top{top_n}_Count': count, f'Top{top_n}_High_Performing_Subset_Recurrence': freq})
    freq_df = pd.DataFrame(rows)
    topN_freq_tables[top_n] = freq_df
    freq_df.to_excel(TABLE_DIR / f'top{top_n}_high_performing_subset_recurrence.xlsx', index=False)
top50_freq_df = topN_freq_tables[TOP_SUBSET_N]
top50_col = f'Top{TOP_SUBSET_N}_High_Performing_Subset_Recurrence'
top50_map = dict(zip(top50_freq_df['Feature'], top50_freq_df[top50_col]))
stability_df = pd.DataFrame({'Feature': features, 'LASSO': frequency_records['LASSO'], 'ElasticNet': frequency_records['ElasticNet'], 'mRMR': frequency_records['mRMR'], 'RF_RFE': frequency_records['RF_RFE'], 'Boruta': frequency_records['Boruta'], f'Top{TOP_SUBSET_N}_High_Performing_Subset_Recurrence': [top50_map.get(f, 0.0) for f in features]})
stability_df['Penalized_Score'] = stability_df[['LASSO', 'ElasticNet']].mean(axis=1)
stability_df['Filter_Score'] = stability_df['mRMR']
stability_df['Tree_Score'] = stability_df[['RF_RFE', 'Boruta']].mean(axis=1)
stability_df['Subset_Score'] = stability_df[f'Top{TOP_SUBSET_N}_High_Performing_Subset_Recurrence']
stability_df['Overall_Stability_Score'] = stability_df[['Penalized_Score', 'Filter_Score', 'Tree_Score', 'Subset_Score']].mean(axis=1)

def evidence_level(score):
    if score >= 0.8:
        return 'High stability'
    if score >= 0.6:
        return 'Moderate stability'
    if score >= 0.4:
        return 'Borderline stability'
    return 'Low stability'
stability_df['Evidence_Level'] = stability_df['Overall_Stability_Score'].apply(evidence_level)
stability_df = stability_df.sort_values('Overall_Stability_Score', ascending=False).reset_index(drop=True)
stability_df['Stability_Rank'] = np.arange(1, len(stability_df) + 1)
success_count_df = pd.DataFrame([{'Method': k, 'Successful_iterations': v} for k, v in success_counts.items()])
stability_df.to_excel(TABLE_DIR / 'six_method_domain_balanced_stability_score.xlsx', index=False)
success_count_df.to_excel(TABLE_DIR / 'method_success_counts.xlsx', index=False)
pass
pass
pass
ranked_features = stability_df['Feature'].tolist()
topk_rows = []
topk_fold_rows = []
topk_estimator = make_selection_estimator()
for k in range(TOPK_MIN_FEATURES, len(ranked_features) + 1):
    topk_features = ranked_features[:k]
    metrics, fold_df, y_pred_oof, y_proba_oof = cv_oof_evaluate(estimator=topk_estimator, X=X_train, y=y_train, groups=groups_train, features=topk_features, n_splits=CV_SPLITS, random_state=RANDOM_STATE + 50000 + k)
    row = {'K': k, 'Features': ', '.join(topk_features)}
    row.update(metrics)
    topk_rows.append(row)
    fold_df = fold_df.copy()
    fold_df['K'] = k
    fold_df['Features'] = ', '.join(topk_features)
    topk_fold_rows.append(fold_df)
topk_df = pd.DataFrame(topk_rows)
topk_fold_df = pd.concat(topk_fold_rows, axis=0, ignore_index=True)
lower_is_better_metrics = ['Log_Loss', 'Brier_Multiclass']
if PRIMARY_SIZE_METRIC in lower_is_better_metrics:
    best_idx = topk_df[PRIMARY_SIZE_METRIC].idxmin()
    best_value = topk_df.loc[best_idx, PRIMARY_SIZE_METRIC]
    se_col = f'{PRIMARY_SIZE_METRIC}_Fold_SE'
    best_se = topk_df.loc[best_idx, se_col] if se_col in topk_df.columns else 0
    threshold = best_value + best_se
    if ONE_SE_RULE:
        eligible = topk_df[topk_df[PRIMARY_SIZE_METRIC] <= threshold]
        selected_k = int(eligible.sort_values('K').iloc[0]['K'])
    else:
        selected_k = int(topk_df.loc[best_idx, 'K'])
else:
    best_idx = topk_df[PRIMARY_SIZE_METRIC].idxmax()
    best_value = topk_df.loc[best_idx, PRIMARY_SIZE_METRIC]
    se_col = f'{PRIMARY_SIZE_METRIC}_Fold_SE'
    best_se = topk_df.loc[best_idx, se_col] if se_col in topk_df.columns else 0
    threshold = best_value - best_se
    if ONE_SE_RULE:
        eligible = topk_df[topk_df[PRIMARY_SIZE_METRIC] >= threshold]
        selected_k = int(eligible.sort_values('K').iloc[0]['K'])
    else:
        selected_k = int(topk_df.loc[best_idx, 'K'])
selected_features = ranked_features[:selected_k]
topk_df['Selected_By_OneSE_Rule'] = topk_df['K'] == selected_k
topk_df['Best_Primary_Metric_Model'] = topk_df['K'] == int(topk_df.loc[best_idx, 'K'])
decision_summary = {'Primary metric': PRIMARY_SIZE_METRIC, 'One-SE rule used': ONE_SE_RULE, 'Best K by primary metric': int(topk_df.loc[best_idx, 'K']), 'Best primary metric value': float(best_value), 'Best model SE': float(best_se), 'One-SE threshold': float(threshold), 'Selected K': int(selected_k), 'Selected features': selected_features}
topk_df.to_excel(TABLE_DIR / 'topk_feature_number_performance.xlsx', index=False)
topk_fold_df.to_excel(TABLE_DIR / 'topk_feature_number_fold_metrics.xlsx', index=False)
pd.DataFrame([decision_summary]).to_excel(TABLE_DIR / 'feature_number_decision_summary.xlsx', index=False)
with open(TABLE_DIR / 'feature_number_decision_summary.json', 'w', encoding='utf-8') as f:
    json.dump(decision_summary, f, ensure_ascii=False, indent=4)
pass
pass
pass
method_cols = ['LASSO', 'ElasticNet', 'mRMR', 'RF_RFE', 'Boruta', f'Top{TOP_SUBSET_N}_High_Performing_Subset_Recurrence']
heat_df = stability_df.set_index('Feature')[method_cols]
fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(heat_df) + 2)))
im = ax.imshow(heat_df.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(np.arange(len(method_cols)))
ax.set_xticklabels(['LASSO', 'Elastic Net', 'mRMR', 'RF-RFE', 'Boruta', f'Top{TOP_SUBSET_N}\nrecurrence'], rotation=35, ha='right')
ax.set_yticks(np.arange(len(heat_df.index)))
ax.set_yticklabels(heat_df.index)
for i in range(heat_df.shape[0]):
    for j in range(heat_df.shape[1]):
        value = heat_df.iloc[i, j]
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='black' if value < 0.65 else 'white', fontsize=9)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Selection frequency / recurrence')
ax.set_title('Six-method feature stability heatmap', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
savefig(fig, 'figure_1_six_method_stability_heatmap')
domain_cols = ['Penalized_Score', 'Filter_Score', 'Tree_Score', 'Subset_Score', 'Overall_Stability_Score']
domain_df = stability_df.set_index('Feature')[domain_cols]
fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(domain_df) + 2)))
im = ax.imshow(domain_df.values, aspect='auto', cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(np.arange(len(domain_cols)))
ax.set_xticklabels(['Penalized', 'Filter', 'Tree', 'Subset', 'Overall'], rotation=30, ha='right')
ax.set_yticks(np.arange(len(domain_df.index)))
ax.set_yticklabels(domain_df.index)
for i in range(domain_df.shape[0]):
    for j in range(domain_df.shape[1]):
        value = domain_df.iloc[i, j]
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white' if value < 0.55 else 'black', fontsize=9)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Domain-balanced score')
ax.set_title('Domain-balanced feature stability scores', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
savefig(fig, 'figure_2_domain_balanced_stability_heatmap')
plot_df = stability_df.sort_values('Overall_Stability_Score', ascending=True)
fig, ax = plt.subplots(figsize=(9, max(5, 0.42 * len(plot_df) + 2)))
colors = []
for score in plot_df['Overall_Stability_Score']:
    if score >= 0.8:
        colors.append('#b2182b')
    elif score >= 0.6:
        colors.append('#ef8a62')
    elif score >= 0.4:
        colors.append('#67a9cf')
    else:
        colors.append('#2166ac')
ax.barh(plot_df['Feature'], plot_df['Overall_Stability_Score'], color=colors, alpha=0.9)
ax.axvline(0.8, color='gray', linestyle='--', linewidth=1.2)
ax.axvline(0.6, color='gray', linestyle=':', linewidth=1.2)
ax.axvline(0.4, color='gray', linestyle=':', linewidth=1.2)
for i, v in enumerate(plot_df['Overall_Stability_Score']):
    ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)
ax.set_xlim(0, 1.05)
ax.set_xlabel('Overall stability score')
ax.set_title('Overall feature stability ranking', fontsize=15, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.25)
plt.tight_layout()
savefig(fig, 'figure_3_overall_stability_barplot')
recurrence_plot = pd.DataFrame({'Feature': features})
for top_n, freq_df in topN_freq_tables.items():
    recurrence_plot = recurrence_plot.merge(freq_df[['Feature', f'Top{top_n}_High_Performing_Subset_Recurrence']], on='Feature', how='left')
recurrence_plot['Overall_Stability_Score'] = recurrence_plot['Feature'].map(dict(zip(stability_df['Feature'], stability_df['Overall_Stability_Score'])))
recurrence_plot = recurrence_plot.sort_values('Overall_Stability_Score', ascending=False)
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(recurrence_plot))
width = 0.22
for i, top_n in enumerate(TOP_SUBSET_SENSITIVITY):
    col = f'Top{top_n}_High_Performing_Subset_Recurrence'
    ax.bar(x + (i - 1) * width, recurrence_plot[col], width=width, label=f'Top {top_n}')
ax.set_xticks(x)
ax.set_xticklabels(recurrence_plot['Feature'], rotation=45, ha='right')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Recurrence frequency')
ax.set_title('Feature recurrence in top high-performing subsets', fontsize=15, fontweight='bold', pad=15)
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
savefig(fig, 'figure_4_top_subset_recurrence_sensitivity')
best_by_size = exhaustive_df.sort_values('multi_objective_mean_rank', ascending=True).groupby('n_features', as_index=False).first()
best_by_size.to_excel(TABLE_DIR / 'best_multi_objective_subset_by_feature_number.xlsx', index=False)
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(best_by_size['n_features'], best_by_size['Log_Loss'], marker='o', linewidth=2.2, color='#2166ac', label='Best CV log loss')
ax1.set_xlabel('Number of features')
ax1.set_ylabel('Best CV log loss', color='#2166ac')
ax1.tick_params(axis='y', labelcolor='#2166ac')
ax1.grid(alpha=0.25)
ax2 = ax1.twinx()
ax2.plot(best_by_size['n_features'], best_by_size['Macro_F1'], marker='s', linewidth=2.2, linestyle='--', color='#b2182b', label='Best CV macro-F1')
ax2.set_ylabel('Best CV macro-F1', color='#b2182b')
ax2.tick_params(axis='y', labelcolor='#b2182b')
ax1.axvline(selected_k, color='black', linestyle=':', linewidth=1.5)
ax1.text(selected_k + 0.1, ax1.get_ylim()[1] - 0.05 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]), f'Selected K={selected_k}', fontsize=11, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc='best')
plt.title('Subset-size performance plateau', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
savefig(fig, 'figure_5_best_subset_by_size')
metrics_to_plot = [('Accuracy', 'Accuracy'), ('Macro_F1', 'Macro-F1'), ('AUC_OVR_Macro', 'Macro-AUC'), ('Balanced_Accuracy', 'Balanced accuracy')]
fig, ax = plt.subplots(figsize=(10, 6))
for metric, label in metrics_to_plot:
    y_values = topk_df[metric].values
    se_col = f'{metric}_Fold_SE'
    y_err = topk_df[se_col].values if se_col in topk_df.columns else None
    ax.errorbar(topk_df['K'], y_values, yerr=y_err, marker='o', linewidth=2, capsize=3, label=label)
ax.axvline(selected_k, color='black', linestyle=':', linewidth=1.5, label=f'Selected K={selected_k}')
ax.set_xlabel('Number of top-ranked features')
ax.set_ylabel('Cross-validated performance')
ax.set_ylim(0, 1.05)
ax.set_title('Top-k feature-set performance', fontsize=15, fontweight='bold', pad=15)
ax.legend(frameon=False)
ax.grid(alpha=0.25)
plt.tight_layout()
savefig(fig, 'figure_6_topk_performance_curves')
fig, ax = plt.subplots(figsize=(9, 6))
metric = PRIMARY_SIZE_METRIC
se_col = f'{metric}_Fold_SE'
err = topk_df[se_col].values if se_col in topk_df.columns else None
ax.errorbar(topk_df['K'], topk_df[metric], yerr=err, marker='o', linewidth=2.5, capsize=4, color='#b2182b')
ax.axvline(selected_k, color='black', linestyle=':', linewidth=1.5, label=f'Selected K={selected_k}')
ax.axhline(threshold, color='gray', linestyle='--', linewidth=1.2, label='1-SE threshold')
ax.scatter([topk_df.loc[best_idx, 'K']], [topk_df.loc[best_idx, metric]], s=110, color='#2166ac', zorder=5, label='Best primary metric')
ax.set_xlabel('Number of top-ranked features')
ax.set_ylabel(metric)
ax.set_title(f'Feature-number selection by one-standard-error rule\nPrimary metric: {metric}', fontsize=14, fontweight='bold', pad=15)
ax.legend(frameon=False)
ax.grid(alpha=0.25)
plt.tight_layout()
savefig(fig, 'figure_7_one_se_feature_number_decision')
fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(exhaustive_df['Log_Loss'], exhaustive_df['Macro_F1'], c=exhaustive_df['n_features'], cmap='viridis', alpha=0.65, s=35, edgecolor='none')
top10 = exhaustive_df.head(10)
ax.scatter(top10['Log_Loss'], top10['Macro_F1'], s=95, facecolor='none', edgecolor='red', linewidth=1.5, label='Top 10 multi-objective subsets')
ax.set_xlabel('CV log loss')
ax.set_ylabel('CV macro-F1')
ax.set_title('Exhaustive subset performance landscape', fontsize=15, fontweight='bold', pad=15)
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Number of features')
ax.legend(frameon=False)
ax.grid(alpha=0.25)
plt.tight_layout()
savefig(fig, 'figure_8_exhaustive_subset_performance_landscape')
if RUN_METHOD_DIAGNOSTICS:
    pass
    X_scaled = StandardScaler().fit_transform(X_train[features])
    C_path = np.logspace(-3, 2, 60)
    lasso_path_rows = []
    lasso_coef_matrix = []
    for C in C_path:
        clf = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=10000, class_weight='balanced', multi_class='auto', random_state=RANDOM_STATE)
        try:
            clf.fit(X_scaled, y_train)
            coef_norm = coef_norm_from_multiclass(clf)
        except Exception:
            coef_norm = np.full(len(features), np.nan)
        lasso_coef_matrix.append(coef_norm)
        for f, coef_value in zip(features, coef_norm):
            lasso_path_rows.append({'C': C, 'log10_C': np.log10(C), 'Feature': f, 'Coefficient_L2_Norm': coef_value, 'Selected': bool(coef_value > 1e-06) if np.isfinite(coef_value) else False})
    lasso_path_df = pd.DataFrame(lasso_path_rows)
    lasso_path_df.to_excel(METHOD_TABLE_DIR / 'lasso_coefficient_path.xlsx', index=False)
    lasso_coef_matrix = np.vstack(lasso_coef_matrix)
    fig, ax = plt.subplots(figsize=(10, 7))
    for j, f in enumerate(features):
        ax.plot(np.log10(C_path), lasso_coef_matrix[:, j], linewidth=2, label=f)
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_xlabel('log10(C)')
    ax.set_ylabel('Coefficient L2 norm across classes')
    ax.set_title('LASSO coefficient path', fontsize=16, fontweight='bold', pad=15)
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=9)
    plt.tight_layout()
    save_method_fig(fig, 'method_1_lasso_coefficient_path')
    elastic_l1_ratios = [0.2, 0.5, 0.8]
    elastic_path_rows = []
    fig, axes = plt.subplots(1, len(elastic_l1_ratios), figsize=(6 * len(elastic_l1_ratios), 6), sharey=True)
    if len(elastic_l1_ratios) == 1:
        axes = [axes]
    for ax, l1_ratio in zip(axes, elastic_l1_ratios):
        coef_matrix = []
        for C in C_path:
            clf = LogisticRegression(penalty='elasticnet', solver='saga', C=C, l1_ratio=l1_ratio, max_iter=10000, class_weight='balanced', multi_class='auto', random_state=RANDOM_STATE)
            try:
                clf.fit(X_scaled, y_train)
                coef_norm = coef_norm_from_multiclass(clf)
            except Exception:
                coef_norm = np.full(len(features), np.nan)
            coef_matrix.append(coef_norm)
            for f, coef_value in zip(features, coef_norm):
                elastic_path_rows.append({'l1_ratio': l1_ratio, 'C': C, 'log10_C': np.log10(C), 'Feature': f, 'Coefficient_L2_Norm': coef_value, 'Selected': bool(coef_value > 1e-06) if np.isfinite(coef_value) else False})
        coef_matrix = np.vstack(coef_matrix)
        for j, f in enumerate(features):
            ax.plot(np.log10(C_path), coef_matrix[:, j], linewidth=1.8, label=f)
        ax.axhline(0, linestyle='--', linewidth=1)
        ax.set_xlabel('log10(C)')
        ax.set_title(f'Elastic Net path\nl1_ratio={l1_ratio}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('Coefficient L2 norm across classes')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=9)
    fig.suptitle('Elastic Net coefficient paths', fontsize=17, fontweight='bold', y=1.03)
    plt.tight_layout()
    save_method_fig(fig, 'method_2_elastic_net_coefficient_paths')
    elastic_path_df = pd.DataFrame(elastic_path_rows)
    elastic_path_df.to_excel(METHOD_TABLE_DIR / 'elastic_net_coefficient_paths.xlsx', index=False)
    rf_for_rfe = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    rf_rfe_cv = make_group_stratified_cv(y_train, groups_train, requested_splits=CV_SPLITS, random_state=RANDOM_STATE)
    rf_rfe_splits = list(rf_rfe_cv.split(X_train[features], y_train, groups_train))
    rfecv = RFECV(estimator=rf_for_rfe, step=1, min_features_to_select=RF_RFE_MIN_FEATURES, cv=rf_rfe_splits, scoring='f1_macro', n_jobs=-1)
    rfecv.fit(X_train[features], y_train)
    rf_rfe_path_df = rfecv_results_to_dataframe(rfecv=rfecv, total_features=len(features), min_features_to_select=RF_RFE_MIN_FEATURES)
    rf_rfe_path_df['Selected_n_features'] = int(rfecv.n_features_)
    rf_rfe_path_df.to_excel(METHOD_TABLE_DIR / 'rf_rfe_cv_performance_path.xlsx', index=False)
    rf_rfe_selected_df = pd.DataFrame({'Feature': features, 'RF_RFE_Selected_FullTraining': rfecv.support_, 'RF_RFE_Ranking_FullTraining': rfecv.ranking_}).sort_values('RF_RFE_Ranking_FullTraining')
    rf_rfe_selected_df.to_excel(METHOD_TABLE_DIR / 'rf_rfe_full_training_selected_features.xlsx', index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    if rf_rfe_path_df['std_test_score'].notna().any():
        ax.errorbar(rf_rfe_path_df['n_features'], rf_rfe_path_df['mean_test_score'], yerr=rf_rfe_path_df['std_test_score'], marker='o', linewidth=2.2, capsize=4, color='#55A868')
    else:
        ax.plot(rf_rfe_path_df['n_features'], rf_rfe_path_df['mean_test_score'], marker='o', linewidth=2.2, color='#55A868')
    ax.axvline(rfecv.n_features_, linestyle='--', linewidth=1.5, color='black', label=f'Selected n={rfecv.n_features_}')
    ax.set_xlabel('Number of retained features')
    ax.set_ylabel('Cross-validated macro-F1')
    ax.set_title('RF-RFE performance path', fontsize=16, fontweight='bold', pad=15)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_method_fig(fig, 'method_3_rf_rfe_cv_performance_path')
combined_workbook = OUTPUT_DIR / 'your_feature_selection_results.xlsx'
with pd.ExcelWriter(combined_workbook) as writer:
    training_dist_df.to_excel(writer, sheet_name='Training_Distribution', index=False)
    success_count_df.to_excel(writer, sheet_name='Success_Counts', index=False)
    stability_df.to_excel(writer, sheet_name='Six_Method_Stability', index=False)
    topk_df.to_excel(writer, sheet_name='TopK_Performance', index=False)
    topk_fold_df.to_excel(writer, sheet_name='TopK_Fold_Metrics', index=False)
    pd.DataFrame([decision_summary]).to_excel(writer, sheet_name='Decision_Summary', index=False)
    exhaustive_df.to_excel(writer, sheet_name='Exhaustive_All_Subsets', index=False)
    best_by_size.to_excel(writer, sheet_name='Best_By_Size', index=False)
    for top_n, freq_df in topN_freq_tables.items():
        freq_df.to_excel(writer, sheet_name=f'Top{top_n}_Recurrence', index=False)
    if RUN_METHOD_DIAGNOSTICS:
        lasso_path_df.to_excel(writer, sheet_name='LASSO_Path', index=False)
        elastic_path_df.to_excel(writer, sheet_name='ElasticNet_Path', index=False)
        rf_rfe_path_df.to_excel(writer, sheet_name='RF_RFE_Path', index=False)
        rf_rfe_selected_df.to_excel(writer, sheet_name='RF_RFE_Selected', index=False)
interpretation = f"\nFeature selection was performed exclusively within the predefined training set using a fast six-method stability framework.\nNo additional train-test split was performed because your_training_data.xlsx already represented the training cohort.\n\nThe six methods included LASSO, Elastic Net, mRMR, RF-RFE, Boruta-style shadow feature screening, and Top-{TOP_SUBSET_N} high-performing subset recurrence from exhaustive subset analysis.\nA domain-balanced overall stability score was calculated by averaging four evidence domains:\npenalized regression, filter-based relevance-redundancy, tree-based wrapper/all-relevant selection, and high-performing subset recurrence.\n\nVariables were ranked by the overall stability score. Top-k feature sets were then evaluated using patient-level cross-validation.\nThe primary metric for feature-number selection was {PRIMARY_SIZE_METRIC}.\nThe best K by the primary metric was {int(topk_df.loc[best_idx, 'K'])}.\nUsing the one-standard-error rule, the selected number of features was K={selected_k}.\nThe selected features were: {', '.join(selected_features)}. ： 。 your_training_data.xlsx ，/。， LASSO、Elastic Net、mRMR、RF-RFE、Boruta-style ， Top-{TOP_SUBSET_N}。，、filter -、 wrapper/all-relevant 。， Top-k 。{PRIMARY_SIZE_METRIC}one-standard-error rule{selected_k}：{', '.join(selected_features)}。\n"
with open(OUTPUT_DIR / 'manuscript_ready_interpretation.txt', 'w', encoding='utf-8') as f:
    f.write(interpretation)
elapsed = time.time() - start_time
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass

class PatientLevelStratifiedKFold:

    def __init__(self, n_splits=5, random_state=42, tolerance=0.15, max_iter=1000):
        self.n_splits = n_splits
        self.random_state = random_state
        self.tolerance = tolerance
        self.max_iter = max_iter

    def split(self, X, y, groups):
        np.random.seed(self.random_state)
        patient_labels = {}
        for patient_id in np.unique(groups):
            patient_labels[patient_id] = y[groups == patient_id][0]
        class_patients = {}
        for cls in np.unique(y):
            class_patients[cls] = [p for p, label in patient_labels.items() if label == cls]
        total_patients = len(patient_labels)
        target_per_fold = {cls: len(patients) / self.n_splits for cls, patients in class_patients.items()}
        folds = [[] for _ in range(self.n_splits)]
        fold_counts = {i: {cls: 0 for cls in class_patients.keys()} for i in range(self.n_splits)}
        for cls in class_patients:
            np.random.shuffle(class_patients[cls])
        for cls, patients in class_patients.items():
            for i, patient in enumerate(patients):
                min_fold = min(range(self.n_splits), key=lambda f: fold_counts[f][cls])
                folds[min_fold].append(patient)
                fold_counts[min_fold][cls] += 1
        for i in range(self.n_splits):
            fold_dist = {}
            for cls in class_patients.keys():
                fold_dist[cls] = fold_counts[i][cls] / len(folds[i]) if folds[i] else 0
            expected_dist = {cls: len(patients) / total_patients for cls, patients in class_patients.items()}
            for cls in class_patients.keys():
                if abs(fold_dist.get(cls, 0) - expected_dist[cls]) > self.tolerance:
                    pass
        for i in range(self.n_splits):
            val_patients = set(folds[i])
            train_patients_cv = set([p for f in folds[:i] + folds[i + 1:] for p in f])
            train_idx = np.where(np.isin(groups, list(train_patients_cv)))[0]
            val_idx = np.where(np.isin(groups, list(val_patients)))[0]
            yield (train_idx, val_idx)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
train_patient_ids = train_df['patient_id'].values
param_grids = {'Logistic Regression': {'C': [0.001, 0.01, 1, 10.0, 20.0, 100.0], 'max_iter': [20, 25, 100], 'solver': ['liblinear', 'lbfgs', 'saga']}, 'SVM': {'C': [0.001, 0.01, 0.5, 1.0, 2, 5, 10], 'kernel': ['rbf', 'linear', 'poly'], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0, 10], 'class_weight': [None, 'balanced']}, 'Random Forest': {'n_estimators': [50, 100, 200, 300, 500], 'max_depth': [3, 5, 7, 10, 20, None], 'min_samples_split': [2, 5, 10, 15]}, 'XGBoost': {'n_estimators': [50, 100, 200, 500], 'max_depth': [3, 4, 5, 6, 7, 15], 'learning_rate': [0.001, 0.01, 0.1, 0.5]}, 'CatBoost': {'iterations': [100, 200, 500], 'depth': [3, 5, 7, 10], 'learning_rate': [0.001, 0.01, 0.1], 'l2_leaf_reg': [1, 3, 5, 10]}}
final_results = {}
pass
n_folds = 5
cv_stratified = PatientLevelStratifiedKFold(n_splits=n_folds, random_state=42, tolerance=0.15)
for model_name, param_grid in param_grids.items():
    pass
    pass
    pass
    if model_name in ['Logistic Regression', 'SVM']:
        X_tr_data = X_train_scaled
        X_te_data = X_test_scaled
    else:
        X_tr_data = X_train.values
        X_te_data = X_test.values
    if model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=1)
    elif model_name == 'SVM':
        model = SVC(probability=True, random_state=1)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=1)
    elif model_name == 'XGBoost':
        model = XGBClassifier(random_state=1)
    else:
        model = CatBoostClassifier(random_state=1, verbose=False, loss_function='MultiClass')
    pass
    cv_splitter = list(cv_stratified.split(X_tr_data, y_train_encoded, train_patient_ids))
    grid_search = GridSearchCV(model, param_grid, cv=cv_splitter, scoring='accuracy', n_jobs=-1, verbose=1, return_train_score=True)
    grid_search.fit(X_tr_data, y_train_encoded)
    best_model = grid_search.best_estimator_
    best_model.fit(X_tr_data, y_train_encoded)
    y_pred = best_model.predict(X_te_data)
    y_pred_proba = best_model.predict_proba(X_te_data)
    test_accuracy = accuracy_score(y_test_encoded, y_pred)
    test_precision = precision_score(y_test_encoded, y_pred, average='macro')
    test_recall = recall_score(y_test_encoded, y_pred, average='macro')
    test_f1 = f1_score(y_test_encoded, y_pred, average='macro')
    test_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr')
    final_results[model_name] = {'model': best_model, 'accuracy': test_accuracy, 'precision': test_precision, 'recall': test_recall, 'f1': test_f1, 'auc': test_auc, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba, 'best_params': grid_search.best_params_, 'cv_score': grid_search.best_score_, 'grid_search': grid_search}
    pass
    pass
    pass
    pass
pass
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
axes = axes.flatten()
for idx, (model_name, res) in enumerate(final_results.items()):
    ax = axes[idx]
    grid_search = res['grid_search']
    if model_name in ['Logistic Regression', 'SVM']:
        X_data = X_train_scaled
    else:
        X_data = X_train.values
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    accuracies = []
    cv_splitter = list(PatientLevelStratifiedKFold(n_splits=n_folds, random_state=42, tolerance=0.15).split(X_data, y_train_encoded, train_patient_ids))
    for fold, (train_idx, val_idx) in enumerate(cv_splitter):
        X_tr, X_val = (X_data[train_idx], X_data[val_idx])
        y_tr, y_val = (y_train_encoded[train_idx], y_train_encoded[val_idx])
        best_params = res['best_params']
        if model_name == 'Logistic Regression':
            model_clone = LogisticRegression(**best_params, random_state=1)
        elif model_name == 'SVM':
            model_clone = SVC(**best_params, probability=True, random_state=1)
        elif model_name == 'Random Forest':
            model_clone = RandomForestClassifier(**best_params, random_state=1)
        elif model_name == 'XGBoost':
            model_clone = XGBClassifier(**best_params, random_state=1)
        else:
            model_clone = CatBoostClassifier(**best_params, random_state=1, verbose=False, loss_function='MultiClass')
        model_clone.fit(X_tr, y_tr)
        y_score = model_clone.predict_proba(X_val)
        y_val_pred = model_clone.predict(X_val)
        fold_accuracy = accuracy_score(y_val, y_val_pred)
        accuracies.append(fold_accuracy)
        y_val_binarized = label_binarize(y_val, classes=np.arange(len(class_names)))
        fpr, tpr, _ = roc_curve(y_val_binarized.ravel(), y_score.ravel())
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    mean_accuracy = np.mean(accuracies)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.plot(mean_fpr, mean_tpr, color='b', linewidth=2.5, label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3, label=f'±1 std dev')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random Classifier')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f"{model_name}\nCV Acc: {res['cv_score']:.3f} | Test Acc: {res['accuracy']:.3f}", fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    pass
if len(final_results) < 6:
    for idx in range(len(final_results), 6):
        axes[idx].axis('off')
plt.tight_layout()
plt.savefig('result/cv_roc_correct.png', dpi=600, bbox_inches='tight')
plt.show()
pass
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
axes = axes.flatten()
colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
for idx, (model_name, res) in enumerate(final_results.items()):
    ax = axes[idx]
    y_score = res['y_pred_proba']
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve((y_test_encoded == i).astype(int), y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=f'Class {class_names[i]} (AUC = {roc_auc:.3f})')
    y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(len(class_names)))
    fpr_micro, tpr_micro, _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color='black', linestyle=':', linewidth=3, label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random Classifier')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f"{model_name}\nTest Acc: {res['accuracy']:.3f}, Test AUC: {res['auc']:.3f}", fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
if len(final_results) < 6:
    for idx in range(len(final_results), 6):
        axes[idx].axis('off')
plt.tight_layout()
plt.savefig('result/test_roc_curves.png', dpi=600, bbox_inches='tight')
plt.show()
pass
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
axes = axes.flatten()
colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
for idx, (model_name, res) in enumerate(final_results.items()):
    ax = axes[idx]
    y_score = res['y_pred_proba']
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve((y_test_encoded == i).astype(int), y_score[:, i])
        average_precision = average_precision_score((y_test_encoded == i).astype(int), y_score[:, i])
        ax.plot(recall, precision, color=colors[i], linewidth=2, label=f'Class {class_names[i]} (AP = {average_precision:.3f})')
    y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(len(class_names)))
    precision_micro, recall_micro, _ = precision_recall_curve(y_test_binarized.ravel(), y_score.ravel())
    average_precision_micro = average_precision_score(y_test_binarized, y_score, average='micro')
    ax.plot(recall_micro, precision_micro, color='black', linestyle=':', linewidth=3, label=f'Micro-average (AP = {average_precision_micro:.3f})')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f"{model_name}\nTest Acc: {res['accuracy']:.3f}, Test AP: {average_precision_micro:.3f}", fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
if len(final_results) < 6:
    for idx in range(len(final_results), 6):
        axes[idx].axis('off')
plt.tight_layout()
plt.savefig('result/test_pr_curves.png', dpi=600, bbox_inches='tight')
plt.show()
pass
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
axes = axes.flatten()
for idx, (model_name, res) in enumerate(final_results.items()):
    ax = axes[idx]
    y_pred = res['y_pred']
    cm = confusion_matrix(y_test_encoded, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar_kws={'shrink': 0.8})
    ax.set_title(f"{model_name}\nTest Accuracy: {res['accuracy']:.3f}", fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
if len(final_results) < 6:
    for idx in range(len(final_results), 6):
        axes[idx].axis('off')
plt.tight_layout()
plt.savefig('result/confusion_matrices.png', dpi=600, bbox_inches='tight')
plt.show()
pass
feature_importance_models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'CatBoost']
for model_name in feature_importance_models:
    if model_name not in final_results:
        continue
    pass
    fig, ax = plt.subplots(figsize=(10, 6))
    model = final_results[model_name]['model']
    if model_name == 'Logistic Regression':
        importance = np.mean(np.abs(model.coef_), axis=0)
        importance_name = 'Coefficient Magnitude'
        color = 'skyblue'
    elif model_name == 'Random Forest':
        importance = model.feature_importances_
        importance_name = 'Feature Importance'
        color = 'lightgreen'
    elif model_name == 'XGBoost':
        importance = model.feature_importances_
        importance_name = 'Feature Importance'
        color = 'lightcoral'
    else:
        importance = model.get_feature_importance()
        importance_name = 'Feature Importance'
        color = 'gold'
    feature_imp_df = pd.DataFrame({'feature': features, 'importance': importance}).sort_values('importance', ascending=True)
    bars = ax.barh(range(len(feature_imp_df)), feature_imp_df['importance'], color=color, edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(feature_imp_df)))
    ax.set_yticklabels(feature_imp_df['feature'])
    ax.set_xlabel(importance_name, fontweight='bold')
    ax.set_ylabel('Features', fontweight='bold')
    ax.set_title(f"{model_name} - Feature Importance\nTest Accuracy: {final_results[model_name]['accuracy']:.3f}", fontweight='bold', fontsize=14)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01 * max(importance), bar.get_y() + bar.get_height() / 2, f'{width:.3f}', ha='left', va='center', fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"result/feature_importance_{model_name.replace(' ', '_').lower()}.png", dpi=600, bbox_inches='tight')
    plt.show()
pass
models = list(final_results.keys())
performance_data = []
for model_name in models:
    res = final_results[model_name]
    performance_data.append({'Model': model_name, 'CV Accuracy': res['cv_score'], 'Test Accuracy': res['accuracy'], 'Test AUC': res['auc'], 'Test Precision': res['precision'], 'Test Recall': res['recall'], 'Test F1': res['f1']})
performance_df = pd.DataFrame(performance_data)
pass
pass
performance_df.to_csv('result/performance_summary.csv', index=False)
joblib.dump(scaler, 'result/scaler.pkl')
joblib.dump(le, 'result/label_encoder.pkl')
for model_name, res in final_results.items():
    safe_name = model_name.replace(' ', '_').lower()
    joblib.dump(res['model'], f'result/{safe_name}_model.pkl')
pass
pass

pass
for model_name in list(final_results.keys()):
    try:
        pass
        try:
            shap_data = joblib.load(f"result/{model_name.replace(' ', '_')}_shap_values.pkl")
            shap_values = shap_data['shap_values']
            feature_names = shap_data['feature_names']
            X_data = shap_data['X_data']
        except:
            pass
            df_clean = df[features + [target]].dropna()
            if model_name in ['Logistic Regression', 'SVM']:
                X_data = scaler.transform(df_clean[features])
            else:
                X_data = df_clean[features].values
            feature_names = features
            model_obj = final_results[model_name]['model']
            if model_name == 'CatBoost':
                try:
                    import catboost
                    from catboost import Pool
                    shap_values = model_obj.get_feature_importance(data=Pool(X_data, feature_names=feature_names), type='ShapValues')
                    shap_values = shap_values[:, :-1]
                except:
                    explainer = shap.TreeExplainer(model_obj)
                    shap_values = explainer.shap_values(X_data)
            elif model_name == 'Random Forest':
                explainer = shap.TreeExplainer(model_obj)
                shap_values = explainer.shap_values(X_data)
            elif model_name == 'XGBoost':
                explainer = shap.TreeExplainer(model_obj)
                shap_values = explainer.shap_values(X_data)
            elif model_name == 'Logistic Regression':
                explainer = shap.LinearExplainer(model_obj, X_data)
                shap_values = explainer.shap_values(X_data)
            else:

                def model_predict(X):
                    return model_obj.predict_proba(X)
                background = shap.sample(X_data, 100)
                explainer = shap.KernelExplainer(model_predict, background)
                shap_values = explainer.shap_values(X_data)
            shap_data = {'shap_values': shap_values, 'feature_names': feature_names, 'class_names': class_names, 'X_data': X_data, 'model_name': model_name}
            joblib.dump(shap_data, f"result/{model_name.replace(' ', '_')}_shap_values.pkl")
        pass
        if hasattr(shap_values, 'shape'):
            pass
        sample_indices = [92, 97, 98]
        for idx in sample_indices:
            pass
            model_obj = final_results[model_name]['model']
            pred_proba = model_obj.predict_proba([X_data[idx]])[0]
            pred_class = np.argmax(pred_proba)
            class_name = class_names[pred_class]
            pred_prob = pred_proba[pred_class]
            pass
            if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                class_shap_values = shap_values[idx, :, pred_class]
                base_value = np.mean(shap_values[:, :, pred_class])
            elif isinstance(shap_values, list):
                class_shap_values = shap_values[pred_class][idx]
                base_value = np.mean(shap_values[pred_class])
            else:
                class_shap_values = shap_values[idx]
                base_value = np.mean(shap_values)
            explanation = shap.Explanation(values=class_shap_values, base_values=base_value, data=X_data[idx], feature_names=feature_names)
            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(explanation, max_display=10, show=False)
            plt.title(f'{model_name} - Sample {idx}\nPredicted: {class_name} (prob: {pred_prob:.3f})', fontweight='bold', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"result/{model_name.replace(' ', '_')}_waterfall_sample_{idx}.png", dpi=600, bbox_inches='tight')
            plt.show()
        pass
    except Exception as e:
        pass
        import traceback
        traceback.print_exc()

pass
pass
pass
detailed_performance = {}
for model_name, res in final_results.items():
    pass
    y_pred = res['y_pred']
    y_pred_proba = res['y_pred_proba']
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision_per_class = precision_score(y_test_encoded, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test_encoded, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test_encoded, y_pred, average=None, zero_division=0)
    precision_macro = precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)
    precision_micro = precision_score(y_test_encoded, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_test_encoded, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_test_encoded, y_pred, average='micro', zero_division=0)
    if len(class_names) == 2:
        auc_score = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
        auc_per_class = [auc_score, auc_score]
    else:
        auc_score = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr')
        auc_per_class = []
        for i in range(len(class_names)):
            auc_class = roc_auc_score((y_test_encoded == i).astype(int), y_pred_proba[:, i])
            auc_per_class.append(auc_class)
    cm = confusion_matrix(y_test_encoded, y_pred)
    detailed_performance[model_name] = {'accuracy': accuracy, 'precision_per_class': precision_per_class, 'recall_per_class': recall_per_class, 'f1_per_class': f1_per_class, 'precision_macro': precision_macro, 'recall_macro': recall_macro, 'f1_macro': f1_macro, 'precision_micro': precision_micro, 'recall_micro': recall_micro, 'f1_micro': f1_micro, 'auc_ovr': auc_score, 'auc_per_class': auc_per_class, 'confusion_matrix': cm, 'best_params': res['best_params'], 'cv_score': res['cv_score']}
    pass
    pass
    pass
    pass
    pass
    pass
    for i, class_name in enumerate(class_names):
        pass
    pass
    pass
    pass
    pass
    pass
    pass
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    pass
overall_summary = []
for model_name, perf in detailed_performance.items():
    overall_summary.append({'Model': model_name, 'CV_Accuracy': f"{perf['cv_score']:.4f}", 'Test_Accuracy': f"{perf['accuracy']:.4f}", 'Precision_Macro': f"{perf['precision_macro']:.4f}", 'Recall_Macro': f"{perf['recall_macro']:.4f}", 'F1_Macro': f"{perf['f1_macro']:.4f}", 'Precision_Micro': f"{perf['precision_micro']:.4f}", 'Recall_Micro': f"{perf['recall_micro']:.4f}", 'F1_Micro': f"{perf['f1_micro']:.4f}", 'AUC_OVR': f"{perf['auc_ovr']:.4f}", 'Best_Parameters': str(perf['best_params'])})
overall_df = pd.DataFrame(overall_summary)
pass
pass
pass
pass
overall_df.to_csv('result/model_performance_overall.csv', index=False, encoding='utf-8-sig')
class_performance_data = []
for model_name, perf in detailed_performance.items():
    for i, class_name in enumerate(class_names):
        class_performance_data.append({'Model': model_name, 'Class': class_name, 'Precision': f"{perf['precision_per_class'][i]:.4f}", 'Recall': f"{perf['recall_per_class'][i]:.4f}", 'F1_Score': f"{perf['f1_per_class'][i]:.4f}", 'AUC': f"{perf['auc_per_class'][i]:.4f}"})
class_performance_df = pd.DataFrame(class_performance_data)
pass
pass
pass
pass
class_performance_df.to_csv('result/model_performance_per_class.csv', index=False, encoding='utf-8-sig')
for model_name, perf in detailed_performance.items():
    cm_df = pd.DataFrame(perf['confusion_matrix'], index=[f'True_{name}' for name in class_names], columns=[f'Pred_{name}' for name in class_names])
    cm_df.to_csv(f"result/{model_name.replace(' ', '_').lower()}_confusion_matrix.csv", encoding='utf-8-sig')
import json
performance_for_json = {}
for model_name, perf in detailed_performance.items():
    performance_for_json[model_name] = {'accuracy': float(perf['accuracy']), 'precision_per_class': [float(x) for x in perf['precision_per_class']], 'recall_per_class': [float(x) for x in perf['recall_per_class']], 'f1_per_class': [float(x) for x in perf['f1_per_class']], 'precision_macro': float(perf['precision_macro']), 'recall_macro': float(perf['recall_macro']), 'f1_macro': float(perf['f1_macro']), 'precision_micro': float(perf['precision_micro']), 'recall_micro': float(perf['recall_micro']), 'f1_micro': float(perf['f1_micro']), 'auc_ovr': float(perf['auc_ovr']), 'auc_per_class': [float(x) for x in perf['auc_per_class']], 'cv_score': float(perf['cv_score']), 'best_params': perf['best_params'], 'confusion_matrix': perf['confusion_matrix'].tolist()}
with open('result/detailed_performance_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(performance_for_json, f, indent=2, ensure_ascii=False)
pass
pass
pass
best_by_accuracy = max(detailed_performance.keys(), key=lambda x: detailed_performance[x]['accuracy'])
best_by_f1_macro = max(detailed_performance.keys(), key=lambda x: detailed_performance[x]['f1_macro'])
best_by_auc = max(detailed_performance.keys(), key=lambda x: detailed_performance[x]['auc_ovr'])
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
final_best_model = best_by_accuracy
pass
joblib.dump(final_results[final_best_model]['model'], 'result/best_model.pkl')
joblib.dump(le, 'result/label_encoder.pkl')
joblib.dump(scaler, 'result/scaler.pkl')
best_model_results = {'y_true': y_test_encoded, 'y_pred': final_results[final_best_model]['y_pred'], 'y_pred_proba': final_results[final_best_model]['y_pred_proba'], 'feature_names': features, 'class_names': class_names}
joblib.dump(best_model_results, 'result/best_model_predictions.pkl')
if 'CatBoost' in final_results:
    pass
    pass
    pass
    catboost_model = final_results['CatBoost']['model']
    catboost_model.save_model('result/catboost_model.cbm')
    pass
    feature_importance = catboost_model.get_feature_importance()
    pass
    for i, feat in enumerate(features):
        pass
    catboost_imp_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance}).sort_values('Importance', ascending=False)
    catboost_imp_df.to_csv('result/catboost_feature_importance.csv', index=False, encoding='utf-8-sig')
pass
pass
pass
pass
pass
pass
pass
pass
if 'CatBoost' in final_results:
    pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
if 'CatBoost' in final_results:
    pass

pass
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
axes = axes.flatten()
for idx, (model_name, res) in enumerate(final_results.items()):
    ax = axes[idx]
    y_pred_proba = res['y_pred_proba']
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    brier_scores = []
    for i in range(len(class_names)):
        y_true_binary = (y_test_encoded == i).astype(int)
        y_prob_binary = y_pred_proba[:, i]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true_binary, y_prob_binary, n_bins=10, strategy='quantile')
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', color=colors[i], linewidth=2, markersize=6, label=f'Class {class_names[i]}')
        from sklearn.metrics import brier_score_loss
        brier_score = brier_score_loss(y_true_binary, y_prob_binary)
        brier_scores.append(brier_score)
    ax.plot([0, 1], [0, 1], 'k:', linewidth=2, label='Perfectly calibrated')
    ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontweight='bold')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    mean_brier_score = np.mean(brier_scores)
    ax.set_title(f"{model_name}\nTest Accuracy: {res['accuracy']:.3f} | Mean Brier Score: {mean_brier_score:.3f}", fontweight='bold')
    final_results[model_name]['brier_score'] = mean_brier_score
if len(final_results) < 6:
    for idx in range(len(final_results), 6):
        axes[idx].axis('off')
plt.tight_layout()
plt.savefig('result/calibration_curves.png', dpi=600, bbox_inches='tight')
plt.show()
pass
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
axes = axes.flatten()
for idx, (model_name, res) in enumerate(final_results.items()):
    ax = axes[idx]
    y_pred_proba = res['y_pred_proba']
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    for i in range(len(class_names)):
        class_probs = y_pred_proba[y_test_encoded == i, i]
        ax.hist(class_probs, bins=20, range=(0, 1), alpha=0.6, color=colors[i], label=f'Class {class_names[i]}', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Predicted Probability', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_xlim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{model_name}\nProbability Distribution', fontweight='bold')
if len(final_results) < 6:
    for idx in range(len(final_results), 6):
        axes[idx].axis('off')
plt.tight_layout()
plt.savefig('result/probability_distributions.png', dpi=600, bbox_inches='tight')
plt.show()
pass
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
axes = axes.flatten()
for idx, (model_name, res) in enumerate(final_results.items()):
    ax = axes[idx]
    y_pred_proba = res['y_pred_proba']
    y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(len(class_names)))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test_binarized.ravel(), y_pred_proba.ravel(), n_bins=10, strategy='quantile')
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', linewidth=2, markersize=8, color='red', label='Calibration curve')
    ax.plot([0, 1], [0, 1], 'k:', linewidth=2, label='Perfect calibration')
    bin_counts = np.histogram(y_pred_proba.ravel(), bins=10, range=(0, 1))[0]
    for i, (frac, mean) in enumerate(zip(fraction_of_positives, mean_predicted_value)):
        ax.text(mean, frac + 0.02, f'n={bin_counts[i]}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Mean Predicted Probability (All Classes)', fontweight='bold')
    ax.set_ylabel('Fraction of Positives (All Classes)', fontweight='bold')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    from sklearn.metrics import brier_score_loss
    brier_overall = brier_score_loss(y_test_binarized.ravel(), y_pred_proba.ravel())
    ax.set_title(f'{model_name}\nOverall Brier Score: {brier_overall:.3f}', fontweight='bold')
if len(final_results) < 6:
    for idx in range(len(final_results), 6):
        axes[idx].axis('off')
plt.tight_layout()
plt.savefig('result/reliability_diagrams.png', dpi=600, bbox_inches='tight')
plt.show()
pass
calibration_data = []
for model_name, res in final_results.items():
    calibration_data.append({'Model': model_name, 'Brier Score': res.get('brier_score', np.nan), 'Test Accuracy': res['accuracy'], 'Test AUC': res['auc']})
calibration_df = pd.DataFrame(calibration_data)
pass
pass
calibration_df.to_csv('result/calibration_summary.csv', index=False)
plt.figure(figsize=(14, 6))
models = [data['Model'] for data in calibration_data]
brier_scores = [data['Brier Score'] for data in calibration_data]
colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
bars = plt.bar(models, brier_scores, color=colors, edgecolor='black', alpha=0.8)
plt.xlabel('Models', fontweight='bold')
plt.ylabel('Brier Score', fontweight='bold')
plt.title('Model Calibration Comparison (Lower Brier Score = Better Calibration)', fontweight='bold', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, brier_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('result/brier_score_comparison.png', dpi=600, bbox_inches='tight')
plt.show()
pass

pass
from scipy import stats
import numpy as np

def delong_test(y_true, y_score1, y_score2):
    n = len(y_true)
    auc1 = roc_auc_score(y_true, y_score1, multi_class='ovr')
    auc2 = roc_auc_score(y_true, y_score2, multi_class='ovr')
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    auc1_boot = []
    auc2_boot = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        try:
            auc1_boot.append(roc_auc_score(y_true[indices], y_score1[indices], multi_class='ovr'))
            auc2_boot.append(roc_auc_score(y_true[indices], y_score2[indices], multi_class='ovr'))
        except:
            continue
    if len(auc1_boot) > 0 and len(auc2_boot) > 0:
        diff = np.array(auc1_boot) - np.array(auc2_boot)
        se = np.std(diff, ddof=1)
        z_stat = (auc1 - auc2) / (se + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        return (auc1, auc2, z_stat, p_value, se)
    else:
        return (auc1, auc2, 0, 1.0, float('inf'))
model_names = list(final_results.keys())
n_models = len(model_names)
delong_results = pd.DataFrame(index=model_names, columns=model_names)
for i in range(n_models):
    for j in range(i + 1, n_models):
        model1 = model_names[i]
        model2 = model_names[j]
        y_score1 = final_results[model1]['y_pred_proba']
        y_score2 = final_results[model2]['y_pred_proba']
        auc1, auc2, z_stat, p_value, se = delong_test(y_test_encoded, y_score1, y_score2)
        result_text = f'z={z_stat:.3f}, p={p_value:.4f}'
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        delong_results.loc[model1, model2] = f'{result_text} {sig}'
        delong_results.loc[model2, model1] = f'{result_text} {sig}'
        pass
        pass
        pass
        pass
        pass
delong_results.to_csv('result/delong_test_results.csv')
pass
pass

def bootstrap_auc_ci(y_true, y_score, n_bootstraps=1000, ci=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    bootstrapped_aucs = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue
        try:
            auc = roc_auc_score(y_true[indices], y_score[indices], multi_class='ovr')
            bootstrapped_aucs.append(auc)
        except:
            continue
    alpha = 1 - ci
    lower = np.percentile(bootstrapped_aucs, alpha / 2 * 100)
    upper = np.percentile(bootstrapped_aucs, (1 - alpha / 2) * 100)
    mean_auc = np.mean(bootstrapped_aucs)
    return (mean_auc, lower, upper, bootstrapped_aucs)
auc_ci_results = []
for model_name, res in final_results.items():
    y_score = res['y_pred_proba']
    mean_auc, lower, upper, _ = bootstrap_auc_ci(y_test_encoded, y_score)
    class_cis = {}
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_test_encoded == i).astype(int)
        y_prob_binary = y_score[:, i]
        mean_auc_cls, lower_cls, upper_cls, _ = bootstrap_auc_ci(y_true_binary, y_prob_binary, n_bootstraps=500)
        class_cis[class_name] = {'mean': mean_auc_cls, 'ci_lower': lower_cls, 'ci_upper': upper_cls}
    auc_ci_results.append({'Model': model_name, 'AUC_Mean': mean_auc, 'AUC_CI_Lower': lower, 'AUC_CI_Upper': upper, 'Class_CIs': class_cis})
    pass
    pass
    for class_name, ci_info in class_cis.items():
        pass
auc_ci_df = pd.DataFrame([{'Model': r['Model'], 'AUC_Mean': r['AUC_Mean'], 'AUC_CI_Lower': r['AUC_CI_Lower'], 'AUC_CI_Upper': r['AUC_CI_Upper']} for r in auc_ci_results])
auc_ci_df.to_csv('result/auc_confidence_intervals.csv', index=False)
pass
pass
pass
detailed_metrics_output = []
for model_name, res in final_results.items():
    model = res['model']
    if model_name in ['Logistic Regression', 'SVM']:
        X_tr = X_train_scaled
        X_te = X_test_scaled
    else:
        X_tr = X_train.values
        X_te = X_test.values
    y_train_pred = model.predict(X_tr)
    y_train_proba = model.predict_proba(X_tr)
    train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
    train_precision = precision_score(y_train_encoded, y_train_pred, average='macro', zero_division=0)
    train_recall = recall_score(y_train_encoded, y_train_pred, average='macro', zero_division=0)
    train_f1 = f1_score(y_train_encoded, y_train_pred, average='macro', zero_division=0)
    train_auc = roc_auc_score(y_train_encoded, y_train_proba, multi_class='ovr')
    train_precision_per_class = precision_score(y_train_encoded, y_train_pred, average=None, zero_division=0)
    train_recall_per_class = recall_score(y_train_encoded, y_train_pred, average=None, zero_division=0)
    train_f1_per_class = f1_score(y_train_encoded, y_train_pred, average=None, zero_division=0)
    test_accuracy = res['accuracy']
    test_precision = res['precision']
    test_recall = res['recall']
    test_f1 = res['f1']
    test_auc = res['auc']
    test_precision_per_class = precision_score(y_test_encoded, res['y_pred'], average=None, zero_division=0)
    test_recall_per_class = recall_score(y_test_encoded, res['y_pred'], average=None, zero_division=0)
    test_f1_per_class = f1_score(y_test_encoded, res['y_pred'], average=None, zero_division=0)
    cv_accuracy = res['cv_score']
    model_metrics = {'Model': model_name, 'Train_Accuracy': train_accuracy, 'CV_Accuracy': cv_accuracy, 'Test_Accuracy': test_accuracy, 'Train_Precision': train_precision, 'Test_Precision': test_precision, 'Train_Recall': train_recall, 'Test_Recall': test_recall, 'Train_F1': train_f1, 'Test_F1': test_f1, 'Train_AUC': train_auc, 'Test_AUC': test_auc, 'Train_Precision_Per_Class': train_precision_per_class, 'Train_Recall_Per_Class': train_recall_per_class, 'Train_F1_Per_Class': train_f1_per_class, 'Test_Precision_Per_Class': test_precision_per_class, 'Test_Recall_Per_Class': test_recall_per_class, 'Test_F1_Per_Class': test_f1_per_class}
    detailed_metrics_output.append(model_metrics)
    pass
    pass
    pass
    overall_df = pd.DataFrame({'Dataset': ['Train', 'CV', 'Test'], 'Accuracy': [train_accuracy, cv_accuracy, test_accuracy], 'Precision': [train_precision, '-', test_precision], 'Recall': [train_recall, '-', test_recall], 'F1-Score': [train_f1, '-', test_f1], 'AUC': [train_auc, '-', test_auc]})
    pass
    pass
    pass
    class_metrics_df = pd.DataFrame({'Class': class_names, 'Train_Precision': train_precision_per_class, 'Test_Precision': test_precision_per_class, 'Train_Recall': train_recall_per_class, 'Test_Recall': test_recall_per_class, 'Train_F1': train_f1_per_class, 'Test_F1': test_f1_per_class})
    pass
    overall_df.to_csv(f"result/{model_name.replace(' ', '_').lower()}_overall_metrics.csv", index=False)
    class_metrics_df.to_csv(f"result/{model_name.replace(' ', '_').lower()}_per_class_metrics.csv", index=False)
summary_df = pd.DataFrame([{'Model': m['Model'], 'Train_Acc': f"{m['Train_Accuracy']:.4f}", 'CV_Acc': f"{m['CV_Accuracy']:.4f}", 'Test_Acc': f"{m['Test_Accuracy']:.4f}", 'Train_AUC': f"{m['Train_AUC']:.4f}", 'Test_AUC': f"{m['Test_AUC']:.4f}", 'Test_F1': f"{m['Test_F1']:.4f}"} for m in detailed_metrics_output])
pass
pass
pass
pass
summary_df.to_csv('result/all_models_metrics_summary.csv', index=False)
pass
from math import pi
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'AP']
N = len(categories)
radar_data = {}
for model_name, res in final_results.items():
    y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(len(class_names)))
    ap_score = average_precision_score(y_test_binarized, res['y_pred_proba'], average='micro')
    radar_data[model_name] = [res['accuracy'], res['precision'], res['recall'], res['f1'], res['auc'], ap_score]
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
for idx, (model_name, values) in enumerate(radar_data.items()):
    values_plot = values + values[:1]
    ax.plot(angles, values_plot, 'o-', linewidth=2, label=model_name, color=colors[idx])
    ax.fill(angles, values_plot, alpha=0.15, color=colors[idx])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=24, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=20)
ax.grid(True)
plt.title('Test Set Performance Comparison\n(Radar Chart)', size=16, fontweight='bold', y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=22)
plt.tight_layout()
plt.savefig('result/test_set_radar_chart.png', dpi=600, bbox_inches='tight')
plt.show()
radar_df = pd.DataFrame(radar_data, index=categories).T
pass
pass
radar_df.to_csv('result/radar_chart_data.csv')
fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
axes = axes.flatten()
for idx, (model_name, values) in enumerate(radar_data.items()):
    ax = axes[idx]
    values_plot = values + values[:1]
    ax.plot(angles, values_plot, 'o-', linewidth=2, color=colors[idx])
    ax.fill(angles, values_plot, alpha=0.25, color=colors[idx])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_title(model_name, size=12, fontweight='bold', pad=20)
    ax.grid(True)
if len(radar_data) < 6:
    for idx in range(len(radar_data), 6):
        axes[idx].axis('off')
plt.tight_layout()
plt.savefig('result/test_set_radar_chart_individual.png', dpi=600, bbox_inches='tight')
plt.show()
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass

pass
colors = plt.cm.Set1(np.linspace(0, 1, len(final_results)))
model_colors = {model_name: colors[idx] for idx, model_name in enumerate(final_results.keys())}
line_styles = ['-', '--', '-.', ':', '-']
markers = ['o', 's', '^', 'D', 'v']
pass
plt.figure(figsize=(10, 8))
for idx, (model_name, res) in enumerate(final_results.items()):
    y_score = res['y_pred_proba']
    y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(len(class_names)))
    fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=model_colors[model_name], linestyle=line_styles[idx % len(line_styles)], linewidth=2.5, marker=markers[idx % len(markers)], markersize=4, markevery=20, label=f'{model_name} (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, label='Random Classifier')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
plt.title('Test Set ROC Curves - All Models Comparison', fontsize=15, fontweight='bold')
plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('result/test_roc_all_models_summary.png', dpi=600, bbox_inches='tight')
plt.show()
pass
plt.figure(figsize=(10, 8))
for idx, (model_name, res) in enumerate(final_results.items()):
    y_score = res['y_pred_proba']
    y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(len(class_names)))
    precision_micro, recall_micro, _ = precision_recall_curve(y_test_binarized.ravel(), y_score.ravel())
    ap_micro = average_precision_score(y_test_binarized, y_score, average='micro')
    plt.plot(recall_micro, precision_micro, color=model_colors[model_name], linestyle=line_styles[idx % len(line_styles)], linewidth=2.5, marker=markers[idx % len(markers)], markersize=4, markevery=20, label=f'{model_name} (AP = {ap_micro:.3f})')
baseline = np.mean(y_test_encoded == 1) if len(class_names) == 2 else 1.0 / len(class_names)
plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Baseline ({baseline:.3f})')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('Recall', fontsize=13, fontweight='bold')
plt.ylabel('Precision', fontsize=13, fontweight='bold')
plt.title('Test Set Precision-Recall Curves - All Models Comparison', fontsize=15, fontweight='bold')
plt.legend(loc='lower left', fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('result/test_pr_all_models_summary.png', dpi=600, bbox_inches='tight')
plt.show()
pass
plt.figure(figsize=(10, 8))
for idx, (model_name, res) in enumerate(final_results.items()):
    y_pred_proba = res['y_pred_proba']
    y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(len(class_names)))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test_binarized.ravel(), y_pred_proba.ravel(), n_bins=10, strategy='quantile')
    brier = brier_score_loss(y_test_binarized.ravel(), y_pred_proba.ravel())
    plt.plot(mean_predicted_value, fraction_of_positives, color=model_colors[model_name], linestyle=line_styles[idx % len(line_styles)], linewidth=2.5, marker=markers[idx % len(markers)], markersize=8, label=f'{model_name} (Brier = {brier:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6, label='Perfect Calibration')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('Mean Predicted Probability', fontsize=13, fontweight='bold')
plt.ylabel('Fraction of Positives', fontsize=13, fontweight='bold')
plt.title('Test Set Calibration Curves - All Models Comparison', fontsize=15, fontweight='bold')
plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('result/test_calibration_all_models_summary.png', dpi=600, bbox_inches='tight')
plt.show()
pass
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
ax1 = axes[0]
for idx, (model_name, res) in enumerate(final_results.items()):
    y_score = res['y_pred_proba']
    y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(len(class_names)))
    fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color=model_colors[model_name], linestyle=line_styles[idx % len(line_styles)], linewidth=2, label=f'{model_name} ({roc_auc:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax1.set_xlabel('False Positive Rate', fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontweight='bold')
ax1.set_title('ROC Curves', fontweight='bold', fontsize=12)
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax2 = axes[1]
for idx, (model_name, res) in enumerate(final_results.items()):
    y_score = res['y_pred_proba']
    y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(len(class_names)))
    precision_micro, recall_micro, _ = precision_recall_curve(y_test_binarized.ravel(), y_score.ravel())
    ap_micro = average_precision_score(y_test_binarized, y_score, average='micro')
    ax2.plot(recall_micro, precision_micro, color=model_colors[model_name], linestyle=line_styles[idx % len(line_styles)], linewidth=2, label=f'{model_name} ({ap_micro:.3f})')
baseline = 1.0 / len(class_names)
ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Recall', fontweight='bold')
ax2.set_ylabel('Precision', fontweight='bold')
ax2.set_title('Precision-Recall Curves', fontweight='bold', fontsize=12)
ax2.legend(loc='lower left', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.02])
ax3 = axes[2]
for idx, (model_name, res) in enumerate(final_results.items()):
    y_pred_proba = res['y_pred_proba']
    y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(len(class_names)))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test_binarized.ravel(), y_pred_proba.ravel(), n_bins=10, strategy='quantile')
    brier = brier_score_loss(y_test_binarized.ravel(), y_pred_proba.ravel())
    ax3.plot(mean_predicted_value, fraction_of_positives, color=model_colors[model_name], linestyle=line_styles[idx % len(line_styles)], linewidth=2, label=f'{model_name} ({brier:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax3.set_xlabel('Mean Predicted Probability', fontweight='bold')
ax3.set_ylabel('Fraction of Positives', fontweight='bold')
ax3.set_title('Calibration Curves', fontweight='bold', fontsize=12)
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([-0.02, 1.02])
ax3.set_ylim([-0.02, 1.02])
plt.suptitle('Test Set Performance Comparison - All Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('result/test_all_metrics_summary_combined.png', dpi=600, bbox_inches='tight')
plt.show()
pass
pass
pass
pass
pass
pass
pass

pass
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy.stats import chi2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def hosmer_lemeshow_test(y_true, y_pred_proba, n_bins=10):
    data = pd.DataFrame({'observed': y_true.astype(int), 'predicted': y_pred_proba})
    try:
        data['bin'] = pd.qcut(data['predicted'], q=n_bins, labels=False, duplicates='drop')
        actual_bins = data['bin'].nunique()
    except ValueError:
        n_unique = len(np.unique(y_pred_proba))
        n_bins = min(n_bins, n_unique)
        data['bin'] = pd.qcut(data['predicted'], q=n_bins, labels=False, duplicates='drop')
        actual_bins = data['bin'].nunique()
    bin_stats = []
    for bin_idx in range(actual_bins):
        bin_data = data[data['bin'] == bin_idx]
        n = len(bin_data)
        observed_pos = bin_data['observed'].sum()
        observed_neg = n - observed_pos
        expected_pos = bin_data['predicted'].sum()
        expected_neg = n - expected_pos
        bin_stats.append({'bin': bin_idx, 'n': n, 'observed_pos': observed_pos, 'observed_neg': observed_neg, 'expected_pos': expected_pos, 'expected_neg': expected_neg, 'observed_prop': observed_pos / n if n > 0 else 0, 'expected_prop': expected_pos / n if n > 0 else 0})
    bin_stats = pd.DataFrame(bin_stats)
    hl_stat = 0
    for _, row in bin_stats.iterrows():
        n = row['n']
        o = row['observed_pos']
        e = row['expected_pos']
        if n > 0 and e > 0 and (e < n):
            variance = n * (e / n) * (1 - e / n)
            hl_stat += (o - e) ** 2 / variance
    dof = max(actual_bins - 2, 1)
    p_value = 1 - chi2.cdf(hl_stat, dof) if hl_stat > 0 else 1.0
    return {'hl_statistic': hl_stat, 'p_value': p_value, 'dof': dof, 'n_bins': actual_bins, 'bin_details': bin_stats, 'calibration_status': 'Good' if p_value > 0.05 else 'Poor'}
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
axes = axes.flatten()
calibration_results = {}
for idx, (model_name, res) in enumerate(final_results.items()):
    ax = axes[idx]
    y_pred_proba = res['y_pred_proba']
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    class_results = []
    brier_scores = []
    hl_results_all = []
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_test_encoded == i).astype(int)
        y_prob_binary = y_pred_proba[:, i]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true_binary, y_prob_binary, n_bins=10, strategy='quantile')
        brier = brier_score_loss(y_true_binary, y_prob_binary)
        brier_scores.append(brier)
        hl_result = hosmer_lemeshow_test(y_true_binary, y_prob_binary, n_bins=10)
        hl_results_all.append({'model': model_name, 'class': class_name, **hl_result})
        line = ax.plot(mean_predicted_value, fraction_of_positives, 's-', color=colors[i], linewidth=2, markersize=6, label=f"{class_name} (Brier={brier:.3f}, HL-P={hl_result['p_value']:.3f})")[0]
        bin_details = hl_result['bin_details']
        for j, (frac, mean) in enumerate(zip(fraction_of_positives, mean_predicted_value)):
            if j < len(bin_details):
                n_bin = int(bin_details.iloc[j]['n'])
                if j % 2 == 0:
                    ax.annotate(f'n={n_bin}', xy=(mean, frac), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=7, alpha=0.6, color=colors[i])
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration', alpha=0.5)
    ax.set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontweight='bold', fontsize=11)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    mean_brier = np.mean(brier_scores)
    ax.set_title(f"{model_name}\nAccuracy: {res['accuracy']:.3f} | Mean Brier: {mean_brier:.3f}", fontweight='bold', fontsize=12)
    calibration_results[model_name] = {'brier_scores': brier_scores, 'mean_brier': mean_brier, 'hl_results': hl_results_all}
    pass
    pass
    pass
    pass
    pass
    for hl in hl_results_all:
        status = '✓ Good' if hl['p_value'] > 0.05 else '✗ Poor'
        pass
for idx in range(len(final_results), 6):
    axes[idx].axis('off')
plt.tight_layout()
plt.savefig('result/calibration_curves_with_hl.png', dpi=600, bbox_inches='tight')
plt.show()
pass
pass
pass
calibration_table = []
for model_name, results in calibration_results.items():
    for i, (brier, hl) in enumerate(zip(results['brier_scores'], results['hl_results'])):
        calibration_table.append({'Model': model_name, 'Fracture Type': f"Type {hl['class']}", 'Brier Score': f'{brier:.4f}', 'HL Statistic': f"{hl['hl_statistic']:.3f}", 'P-value': f"{hl['p_value']:.3f}", 'Calibration': 'Good' if hl['p_value'] > 0.05 else 'Poor', 'DOF': hl['dof']})
calibration_df = pd.DataFrame(calibration_table)
pass
calibration_df.to_csv('result/calibration_hosmer_lemeshow_summary.csv', index=False)
pass
pass
latex_table = calibration_df.to_latex(index=False, float_format='%.3f')
pass
pass
fig, ax = plt.subplots(figsize=(12, 6))
models = list(calibration_results.keys())
mean_briers = [calibration_results[m]['mean_brier'] for m in models]
colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
bars = ax.bar(models, mean_briers, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
for bar, score in zip(bars, mean_briers):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.005, f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Random guess (0.25)')
ax.axhline(y=0.125, color='green', linestyle='--', alpha=0.5, label='Good calibration threshold (0.125)')
ax.set_xlabel('Models', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Brier Score', fontweight='bold', fontsize=12)
ax.set_title('Model Calibration Comparison\nLower Brier Score = Better Calibration', fontweight='bold', fontsize=14)
ax.set_ylim([0, max(mean_briers) * 1.2])
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('result/brier_score_comparison.png', dpi=600, bbox_inches='tight')
plt.show()
pass
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
axes = axes.flatten()
for idx, (model_name, res) in enumerate(final_results.items()):
    ax = axes[idx]
    y_pred_proba = res['y_pred_proba']
    y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(len(class_names)))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test_binarized.ravel(), y_pred_proba.ravel(), n_bins=10, strategy='quantile')
    brier_micro = brier_score_loss(y_test_binarized.ravel(), y_pred_proba.ravel())
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', linewidth=2.5, markersize=8, color='darkred', label=f'Micro-average (Brier={brier_micro:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration', alpha=0.5)
    bin_counts = np.histogram(y_pred_proba.ravel(), bins=len(mean_predicted_value), range=(0, 1))[0]
    for i, (frac, mean) in enumerate(zip(fraction_of_positives, mean_predicted_value)):
        ax.text(mean, frac + 0.02, f'n={bin_counts[i]}', ha='center', va='bottom', fontsize=8, alpha=0.7)
    ax.set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontweight='bold', fontsize=11)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{model_name} - Micro-average Calibration\nOverall Brier: {brier_micro:.3f}', fontweight='bold', fontsize=12)
for idx in range(len(final_results), 6):
    axes[idx].axis('off')
plt.tight_layout()
plt.savefig('result/reliability_diagrams_micro.png', dpi=600, bbox_inches='tight')
plt.show()
pass

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
FEATURES = ['LLBCE', 'PMBT', 'MRT', 'Depth  of A']
TARGET = 'Type of fracture line'
SAMPLE_COL = 'sample'
EXPOSURES = ['Depth  of A', 'LLBCE']
ADJUSTERS = ['MRT', 'PMBT']
DELTA_MM = 0.5
os.makedirs('result', exist_ok=True)
os.makedirs('result/figures', exist_ok=True)
sns.set(style='whitegrid', font='Arial', font_scale=1.0)
data_base = df.copy()
data_base = data_base[[SAMPLE_COL] + FEATURES + [TARGET]].dropna().copy()
data_base['patient_id'] = ((data_base[SAMPLE_COL].astype(int) + 1) // 2).astype(int)
data_base['side'] = np.where(data_base[SAMPLE_COL].astype(int) % 2 == 1, 'right', 'left')
classes = sorted(data_base[TARGET].unique().tolist())

def evalue_rr(rr: float) -> float:
    rr = float(rr)
    if rr < 1:
        rr = 1.0 / rr
    if rr <= 1:
        return 1.0
    return rr + np.sqrt(rr * (rr - 1))

def compute_evalue_from_rr(rr_est: float, ci_low: float, ci_high: float):
    point_ev = evalue_rr(rr_est)
    if rr_est >= 1:
        ci_bound = ci_low
    else:
        ci_bound = ci_high
    if rr_est >= 1 and ci_bound <= 1 or (rr_est < 1 and ci_bound >= 1):
        ci_ev = 1.0
    else:
        ci_ev = evalue_rr(ci_bound)
    return {'E_value_point': point_ev, 'E_value_CI': ci_ev}

def fit_evalue_continuous_one_vs_rest_gee_poisson(data: pd.DataFrame, outcome_class, exposure: str, adjusters=('MRT', 'PMBT'), delta_mm=0.5):
    d = data.copy()
    y = (d[TARGET] == outcome_class).astype(int)
    X = d[[exposure] + list(adjusters)].copy()
    X = sm.add_constant(X, has_constant='add')
    model = sm.GEE(endog=y, exog=X, groups=d['patient_id'], family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence())
    fit = model.fit()
    beta = fit.params[exposure]
    se = fit.bse[exposure]
    beta_delta = beta * delta_mm
    se_delta = se * delta_mm
    rr_est = np.exp(beta_delta)
    ci_low = np.exp(beta_delta - 1.96 * se_delta)
    ci_high = np.exp(beta_delta + 1.96 * se_delta)
    ev = compute_evalue_from_rr(rr_est, ci_low, ci_high)
    return {'outcome_class': outcome_class, 'exposure': exposure, 'effect_unit_mm': delta_mm, 'n_lines': len(d), 'n_patients': d['patient_id'].nunique(), 'event_rate': float(y.mean()), 'beta_per_1mm': float(beta), 'se_per_1mm': float(se), 'RR_per_delta': float(rr_est), 'CI_low': float(ci_low), 'CI_high': float(ci_high), 'p_value': float(fit.pvalues[exposure]), **ev}

def batch_evalue_continuous_gee_poisson(data: pd.DataFrame, outcome_classes, exposures=('Depth  of A', 'LLBCE'), adjusters=('MRT', 'PMBT'), delta_mm=0.5):
    rows = []
    for c in outcome_classes:
        for exp in exposures:
            row = fit_evalue_continuous_one_vs_rest_gee_poisson(data=data, outcome_class=c, exposure=exp, adjusters=adjusters, delta_mm=delta_mm)
            rows.append(row)
    return pd.DataFrame(rows)
evalue_results_cont = batch_evalue_continuous_gee_poisson(data=data_base, outcome_classes=classes, exposures=EXPOSURES, adjusters=ADJUSTERS, delta_mm=DELTA_MM)
pass
pass
evalue_results_cont.to_csv('result/evalue_continuous_per_0.5mm_gee_poisson.csv', index=False)
with pd.ExcelWriter('result/evalue_continuous_per_0.5mm_gee_poisson.xlsx') as writer:
    evalue_results_cont.to_excel(writer, sheet_name='E_value_continuous', index=False)

def plot_evalue_continuous(evalue_df: pd.DataFrame, delta_mm=0.5, save_path='result/figures/evalue_continuous_per_0.5mm_gee_poisson.png'):
    plot_df = evalue_df.copy()
    plot_df['label'] = plot_df['exposure'] + f' (per {delta_mm} mm) -> Type ' + plot_df['outcome_class'].astype(str)
    plot_df = plot_df.sort_values('E_value_point', ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.65 * len(plot_df))))
    y_pos = np.arange(len(plot_df))
    ax.hlines(y=y_pos, xmin=plot_df['E_value_CI'], xmax=plot_df['E_value_point'], linewidth=2)
    ax.scatter(plot_df['E_value_point'], y_pos, s=70, label='Point E-value', zorder=3)
    ax.scatter(plot_df['E_value_CI'], y_pos, s=50, marker='s', label='CI E-value', zorder=3)
    ax.axvline(1.0, linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['label'])
    ax.set_xlabel('E-value')
    ax.set_ylabel('')
    ax.set_title(f'E-value sensitivity analysis (GEE Poisson, per {delta_mm} mm increase)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
plot_evalue_continuous(evalue_results_cont, delta_mm=DELTA_MM, save_path='result/figures/evalue_continuous_per_0.5mm_gee_poisson.png')
manuscript_table = evalue_results_cont.copy()
manuscript_table['RR (95% CI) per 0.5 mm'] = manuscript_table.apply(lambda r: f"{r['RR_per_delta']:.3f} ({r['CI_low']:.3f}, {r['CI_high']:.3f})", axis=1)
manuscript_table['E-value point / CI'] = manuscript_table.apply(lambda r: f"{r['E_value_point']:.3f} / {r['E_value_CI']:.3f}", axis=1)
manuscript_table = manuscript_table[['outcome_class', 'exposure', 'effect_unit_mm', 'n_lines', 'n_patients', 'event_rate', 'RR (95% CI) per 0.5 mm', 'p_value', 'E-value point / CI']]
manuscript_table.to_csv('result/evalue_continuous_per_0.5mm_gee_poisson_manuscript_table.csv', index=False)
pass
pass
pass
pass
pass

import os
import itertools
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, f_oneway, kruskal, chi2_contingency, fisher_exact
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc, average_precision_score, brier_score_loss, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
OUTPUT_DIR = 'result/external_validation_all_pairwise_summary'
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
EXTERNAL_COHORTS = {'Temporal external cohort': 'your_temporal_external_validation_data.xlsx', 'Geographical external cohort': 'your_geographical_external_validation_data.xlsx'}
COHORT_COLORS = {'Temporal external cohort': '#2A6FBB', 'Geographical external cohort': '#C95F2D'}
COHORT_MARKERS = {'Temporal external cohort': 'o', 'Geographical external cohort': 's'}
CANDIDATE_CONTINUOUS_VARS = ['LLBCE', 'PMBT', 'MRT', 'Depth.of.A', 'RAPL', 'ART', 'RH', 'LSND', 'age']
CANDIDATE_CATEGORICAL_VARS = ['sex', 'type of jaw deformity', 'third molar presence']
VARIABLE_LABELS = {'LLBCE': 'LLBCE', 'PMBT': 'PMBT', 'MRT': 'MRT', 'Depth.of.A': 'Depth of A', 'RAPL': 'RAPL', 'ART': 'ART', 'RH': 'RH', 'LSND': 'LSND', 'age': 'Age', 'sex': 'Sex', 'type of jaw deformity': 'Jaw deformity type', 'third molar presence': 'Third molar presence'}
LEVEL_LABELS = {'male': 'Male', 'female': 'Female', 'yes': 'Present', 'no': 'Absent', '2': 'Skeletal Class II', '2.0': 'Skeletal Class II', '3': 'Skeletal Class III', '3.0': 'Skeletal Class III'}
COLUMN_ALIASES = {'Depth.of.A': ['Depth.of.A', 'Depth of A', 'Depth  of A', 'Depth_of_A', 'Depth.of. A', 'Depth .of.A'], 'type of jaw deformity': ['type of jaw deformity', 'type.of.jaw.deformity', 'jaw deformity type', 'type_of_jaw_deformity'], 'third molar presence': ['third molar presence', 'third.molar.presence', 'third_molar_presence', 'third molar']}
DUNN_ADJUST_METHOD = 'bonferroni'
CATEGORICAL_POSTHOC_ADJUST_METHOD = 'bonferroni'
N_PERMUTATIONS = 10000
RANDOM_STATE = 42
required_objects = ['features', 'target', 'le', 'scaler', 'performance_df', 'final_results']
missing_objects = [obj for obj in required_objects if obj not in globals()]
if len(missing_objects) > 0:
    raise NameError(f'：{missing_objects}，。')
try:
    class_names = list(class_names)
except NameError:
    class_names = list(le.classes_)
n_classes = len(le.classes_)
class_indices = np.arange(n_classes)
try:
    best_model_name = performance_df.loc[performance_df['Test Accuracy'].idxmax(), 'Model']
except Exception:
    best_model_name = list(final_results.keys())[0]
best_model = final_results[best_model_name]['model']
pass
pass
pass
pass
pass
pass
pass

def clean_column_names(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace('\xa0', ' ', regex=False).str.replace('\t', ' ', regex=False).str.strip()
    return df

def normalize_name(x):
    return str(x).replace('\xa0', ' ').replace('\t', ' ').strip().lower().replace(' ', '').replace('_', '').replace('.', '')

def find_column(df, canonical_name):
    columns = list(df.columns)
    norm_map = {normalize_name(c): c for c in columns}
    candidate_names = [canonical_name] + COLUMN_ALIASES.get(canonical_name, [])
    for name in candidate_names:
        key = normalize_name(name)
        if key in norm_map:
            return norm_map[key]
    return None

def build_feature_dataframe(df, feature_list):
    X = pd.DataFrame(index=df.index)
    missing = []
    for f in feature_list:
        actual_col = find_column(df, f)
        if actual_col is None:
            missing.append(f)
        else:
            X[f] = pd.to_numeric(df[actual_col], errors='coerce')
    if len(missing) > 0:
        raise ValueError(f'：{missing}Excel ：{df.columns.tolist()}')
    return X

def clean_string_series(s):
    return s.astype(str).str.replace('\xa0', ' ', regex=False).str.replace('\t', ' ', regex=False).str.strip().str.lower()

def sort_levels(levels):

    def key_func(x):
        try:
            return float(x)
        except Exception:
            return str(x)
    return sorted(levels, key=key_func)

def label_variable(var):
    return VARIABLE_LABELS.get(var, var)

def label_level(level):
    level_str = str(level).strip().lower()
    return LEVEL_LABELS.get(level_str, str(level))

def format_p_value(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'

def median_iqr_string(x):
    x = pd.to_numeric(pd.Series(x), errors='coerce').dropna()
    if len(x) == 0:
        return ''
    median = np.median(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    return f'{median:.2f} [{q1:.2f}–{q3:.2f}]'

def count_percent_string(count, denom):
    if denom == 0:
        return '0 (0.0%)'
    return f'{int(count)} ({100 * count / denom:.1f}%)'

def model_requires_scaled_input(model_name):
    scaled_model_names = ['Logistic Regression', 'SVM', 'KNN', 'MLP', 'Neural Network']
    return model_name in scaled_model_names

def safe_predict_proba(model, X_input):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_input)
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X_input)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return proba
    raise ValueError('predict_proba decision_function，。')

def align_proba_to_classes(model, proba):
    proba = np.asarray(proba)
    if hasattr(model, 'classes_'):
        model_classes = np.asarray(model.classes_)
        aligned = np.zeros((proba.shape[0], n_classes))
        for j, cls in enumerate(model_classes):
            try:
                idx = int(cls)
                if idx in class_indices:
                    aligned[:, idx] = proba[:, j]
            except Exception:
                pass
        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        aligned = aligned / row_sums
        return aligned
    return proba

def transform_labels_safely(y_raw):
    try:
        y_encoded = le.transform(y_raw)
        valid_mask = np.ones(len(y_raw), dtype=bool)
        return (y_encoded, y_raw, valid_mask)
    except Exception:
        pass
    y_str = pd.Series(y_raw).astype(str).str.strip().values
    le_classes_str = pd.Series(le.classes_).astype(str).str.strip().values
    valid_mask = np.isin(y_str, le_classes_str)
    mapping = {str(cls).strip(): cls for cls in le.classes_}
    y_mapped = np.array([mapping.get(v, None) for v in y_str[valid_mask]])
    y_encoded = le.transform(y_mapped)
    return (y_encoded, y_mapped, valid_mask)

def bootstrap_micro_roc_band(y_encoded, y_proba, n_bootstraps=1000, ci=0.95, random_state=42, grid_size=201):
    rng = np.random.default_rng(random_state)
    n_samples = len(y_encoded)
    fpr_grid = np.linspace(0, 1, grid_size)
    boot_tprs = []
    boot_aucs = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n_samples, n_samples)
        y_boot = y_encoded[idx]
        proba_boot = y_proba[idx]
        try:
            y_boot_bin = label_binarize(y_boot, classes=class_indices)
            if len(np.unique(y_boot_bin.ravel())) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_boot_bin.ravel(), proba_boot.ravel())
            boot_auc = auc(fpr, tpr)
            interp_tpr = np.interp(fpr_grid, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tpr[-1] = 1.0
            boot_tprs.append(interp_tpr)
            boot_aucs.append(boot_auc)
        except Exception:
            continue
    if len(boot_tprs) == 0:
        return (fpr_grid, np.full_like(fpr_grid, np.nan), np.full_like(fpr_grid, np.nan), np.full_like(fpr_grid, np.nan), np.nan, np.nan, np.nan)
    boot_tprs = np.asarray(boot_tprs)
    boot_aucs = np.asarray(boot_aucs)
    alpha = 1 - ci
    mean_tpr = np.mean(boot_tprs, axis=0)
    lower_tpr = np.percentile(boot_tprs, alpha / 2 * 100, axis=0)
    upper_tpr = np.percentile(boot_tprs, (1 - alpha / 2) * 100, axis=0)
    auc_mean = np.mean(boot_aucs)
    auc_lower = np.percentile(boot_aucs, alpha / 2 * 100)
    auc_upper = np.percentile(boot_aucs, (1 - alpha / 2) * 100)
    return (fpr_grid, mean_tpr, lower_tpr, upper_tpr, auc_mean, auc_lower, auc_upper)

def compute_per_class_metrics(y_encoded, y_pred, y_proba, cohort_name):
    y_bin = label_binarize(y_encoded, classes=class_indices)
    cm = confusion_matrix(y_encoded, y_pred, labels=class_indices)
    rows = []
    total_n = len(y_encoded)
    for k, class_label in enumerate(le.classes_):
        TP = cm[k, k]
        FN = cm[k, :].sum() - TP
        FP = cm[:, k].sum() - TP
        TN = total_n - TP - FN - FP
        class_accuracy = (TP + TN) / total_n if total_n > 0 else np.nan
        recall = TP / (TP + FN) if TP + FN > 0 else np.nan
        specificity = TN / (TN + FP) if TN + FP > 0 else np.nan
        precision = TP / (TP + FP) if TP + FP > 0 else np.nan
        f1 = 2 * precision * recall / (precision + recall) if pd.notna(precision) and pd.notna(recall) and (precision + recall > 0) else np.nan
        try:
            class_auc = roc_auc_score(y_bin[:, k], y_proba[:, k])
        except Exception:
            class_auc = np.nan
        try:
            class_ap = average_precision_score(y_bin[:, k], y_proba[:, k])
        except Exception:
            class_ap = np.nan
        rows.append({'Cohort': cohort_name, 'Class': f'Type {class_label}', 'Support, n': int(cm[k, :].sum()), 'Predicted, n': int(cm[:, k].sum()), 'Class accuracy': class_accuracy, 'Precision': precision, 'Recall / sensitivity': recall, 'Specificity': specificity, 'F1-score': f1, 'OvR-AUC': class_auc, 'Average precision': class_ap})
    return pd.DataFrame(rows)

def check_expected_counts(expected):
    expected = np.asarray(expected)
    if np.any(expected < 1):
        return False
    if np.mean(expected >= 5) < 0.8:
        return False
    return True

def chi_square_stat_from_table(table):
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    return chi2

def monte_carlo_chi_square_test(cat_values, group_values, n_permutations=10000, random_state=42):
    rng = np.random.default_rng(random_state)
    temp = pd.DataFrame({'cat': cat_values, 'group': group_values}).dropna()
    observed_table = pd.crosstab(temp['cat'], temp['group'])
    observed_chi2 = chi_square_stat_from_table(observed_table)
    group_array = temp['group'].values.copy()
    cat_array = temp['cat'].values.copy()
    perm_chi2 = []
    for _ in range(n_permutations):
        permuted_group = rng.permutation(group_array)
        perm_table = pd.crosstab(cat_array, permuted_group)
        perm_table = perm_table.reindex(index=observed_table.index, columns=observed_table.columns, fill_value=0)
        try:
            chi2_val = chi_square_stat_from_table(perm_table)
            perm_chi2.append(chi2_val)
        except Exception:
            continue
    perm_chi2 = np.asarray(perm_chi2)
    p_value = (np.sum(perm_chi2 >= observed_chi2) + 1) / (len(perm_chi2) + 1)
    return (observed_chi2, p_value)

def dunn_test(data, value_col, group_col, adjust_method='bonferroni'):
    temp = data[[value_col, group_col]].dropna().copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors='coerce')
    temp = temp.dropna()
    groups = sort_levels(temp[group_col].unique())
    if len(groups) < 2:
        return pd.DataFrame()
    temp['rank'] = stats.rankdata(temp[value_col].values)
    N = len(temp)
    _, tie_counts = np.unique(temp[value_col].values, return_counts=True)
    if N > 1:
        tie_correction = 1 - np.sum(tie_counts ** 3 - tie_counts) / (N ** 3 - N)
    else:
        tie_correction = 1
    if tie_correction <= 0:
        tie_correction = 1
    group_stats = {}
    for g in groups:
        sub = temp[temp[group_col] == g]
        group_stats[g] = {'n': len(sub), 'mean_rank': sub['rank'].mean()}
    records = []
    for g1, g2 in itertools.combinations(groups, 2):
        n1 = group_stats[g1]['n']
        n2 = group_stats[g2]['n']
        if n1 == 0 or n2 == 0:
            continue
        r1 = group_stats[g1]['mean_rank']
        r2 = group_stats[g2]['mean_rank']
        se = np.sqrt(N * (N + 1) / 12.0 * (1.0 / n1 + 1.0 / n2) * tie_correction)
        if se == 0:
            z = np.nan
            p = np.nan
        else:
            z = (r1 - r2) / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
        records.append({'comparison': f'Type {g1} vs Type {g2}', 'raw_p': p})
    result = pd.DataFrame(records)
    if len(result) > 0:
        mask = result['raw_p'].notna()
        if mask.sum() > 0:
            result.loc[mask, 'adjusted_p'] = multipletests(result.loc[mask, 'raw_p'], method=adjust_method)[1]
    return result

def tukey_posthoc(data, value_col, group_col):
    temp = data[[value_col, group_col]].dropna().copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors='coerce')
    temp = temp.dropna()
    tukey = pairwise_tukeyhsd(endog=temp[value_col], groups=temp[group_col], alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_df = tukey_df.rename(columns={'p-adj': 'adjusted_p'})
    tukey_df['comparison'] = tukey_df.apply(lambda r: f"Type {r['group1']} vs Type {r['group2']}", axis=1)
    return tukey_df[['comparison', 'adjusted_p']]

def categorical_overall_test(data, cat_var, outcome_col):
    temp = data[[cat_var, outcome_col]].dropna().copy()
    table = pd.crosstab(temp[cat_var], temp[outcome_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return ('Not applicable', np.nan)
    chi2, p_chi, dof, expected = chi2_contingency(table, correction=False)
    expected_ok = check_expected_counts(expected)
    if expected_ok:
        return ('Pearson χ² test', p_chi)
    if table.shape == (2, 2):
        odds_ratio, p_fisher = fisher_exact(table.values)
        return ('Fisher exact test', p_fisher)
    chi2_mc, p_mc = monte_carlo_chi_square_test(cat_values=temp[cat_var], group_values=temp[outcome_col], n_permutations=N_PERMUTATIONS, random_state=RANDOM_STATE)
    return ('Monte Carlo χ² test', p_mc)

def pairwise_categorical_tests(data, cat_var, outcome_col, outcome_levels):
    records = []
    for g1, g2 in itertools.combinations(outcome_levels, 2):
        temp = data[data[outcome_col].isin([g1, g2])][[cat_var, outcome_col]].dropna().copy()
        table = pd.crosstab(temp[cat_var], temp[outcome_col])
        if table.shape[0] < 2 or table.shape[1] < 2:
            records.append({'comparison': f'Type {g1} vs Type {g2}', 'raw_p': np.nan})
            continue
        chi2, p_chi, dof, expected = chi2_contingency(table, correction=False)
        expected_ok = check_expected_counts(expected)
        if expected_ok:
            p_value = p_chi
        elif table.shape == (2, 2):
            _, p_value = fisher_exact(table.values)
        else:
            _, p_value = monte_carlo_chi_square_test(cat_values=temp[cat_var], group_values=temp[outcome_col], n_permutations=N_PERMUTATIONS, random_state=RANDOM_STATE)
        records.append({'comparison': f'Type {g1} vs Type {g2}', 'raw_p': p_value})
    result = pd.DataFrame(records)
    if len(result) > 0:
        mask = result['raw_p'].notna()
        if mask.sum() > 0:
            result.loc[mask, 'adjusted_p'] = multipletests(result.loc[mask, 'raw_p'], method=CATEGORICAL_POSTHOC_ADJUST_METHOD)[1]
    return result

def summarize_all_pairwise(posthoc_df):
    if posthoc_df is None or len(posthoc_df) == 0:
        return ''
    if 'comparison' not in posthoc_df.columns or 'adjusted_p' not in posthoc_df.columns:
        return ''
    pieces = []
    for _, r in posthoc_df.iterrows():
        comp = str(r['comparison'])
        p = r['adjusted_p']
        pieces.append(f'{comp}: P_adj={format_p_value(p)}')
    return '; '.join(pieces)

def summarize_significant_pairwise(posthoc_df, alpha=0.05):
    if posthoc_df is None or len(posthoc_df) == 0:
        return ''
    if 'comparison' not in posthoc_df.columns or 'adjusted_p' not in posthoc_df.columns:
        return ''
    pieces = []
    for _, r in posthoc_df.iterrows():
        comp = str(r['comparison'])
        p = r['adjusted_p']
        if pd.notna(p) and p < alpha:
            pieces.append(f'{comp}: P_adj={format_p_value(p)}')
    return '; '.join(pieces) if len(pieces) > 0 else 'None'

def extract_pairwise_p_columns(posthoc_df, outcome_levels):
    result = {}
    for g1, g2 in itertools.combinations(outcome_levels, 2):
        result[f'Type {g1} vs Type {g2} P_adj'] = ''
    if posthoc_df is None or len(posthoc_df) == 0:
        return result
    if 'comparison' not in posthoc_df.columns or 'adjusted_p' not in posthoc_df.columns:
        return result
    for _, r in posthoc_df.iterrows():
        comp = str(r['comparison'])
        p = r['adjusted_p']
        p_text = format_p_value(p)
        for g1, g2 in itertools.combinations(outcome_levels, 2):
            comp1 = f'Type {g1} vs Type {g2}'
            comp2 = f'Type {g2} vs Type {g1}'
            col_name = f'Type {g1} vs Type {g2} P_adj'
            if comp == comp1 or comp == comp2:
                result[col_name] = p_text
    return result

def make_simplified_variable_table(df, cohort_name, outcome_col):
    data = clean_column_names(df.copy())
    actual_outcome_col = find_column(data, outcome_col)
    if actual_outcome_col is None:
        raise ValueError(f'{cohort_name}：{outcome_col}')
    data[outcome_col] = data[actual_outcome_col].astype(str).str.strip()
    outcome_levels = sort_levels(data[outcome_col].dropna().unique())
    rows = []
    for canonical_var in CANDIDATE_CONTINUOUS_VARS:
        actual_col = find_column(data, canonical_var)
        if actual_col is None:
            pass
            continue
        data[canonical_var] = pd.to_numeric(data[actual_col], errors='coerce')
        row = {'Cohort': cohort_name, 'Variable': label_variable(canonical_var), 'Level': '', 'Overall': median_iqr_string(data[canonical_var]), 'Summary': 'Median [Q1–Q3]'}
        group_values = []
        shapiro_ps = []
        for g in outcome_levels:
            x = data.loc[data[outcome_col] == g, canonical_var].dropna()
            row[f'Type {g}'] = median_iqr_string(x)
            group_values.append(x)
            if len(x) >= 3:
                try:
                    _, p_shapiro = shapiro(x)
                except Exception:
                    p_shapiro = np.nan
            else:
                p_shapiro = np.nan
            shapiro_ps.append(p_shapiro)
        all_normal = all((pd.notna(p) for p in shapiro_ps)) and all((p >= 0.05 for p in shapiro_ps))
        valid_group_values = [x for x in group_values if len(x) > 0]
        if len(valid_group_values) < 2:
            test_name = 'Not applicable'
            p_value = np.nan
            posthoc_df = pd.DataFrame()
        elif all_normal:
            try:
                _, p_value = f_oneway(*valid_group_values)
                test_name = 'One-way ANOVA'
                posthoc_df = tukey_posthoc(data, canonical_var, outcome_col)
            except Exception:
                test_name = 'One-way ANOVA failed'
                p_value = np.nan
                posthoc_df = pd.DataFrame()
        else:
            try:
                _, p_value = kruskal(*valid_group_values)
                test_name = 'Kruskal–Wallis test'
                posthoc_df = dunn_test(data, canonical_var, outcome_col, adjust_method=DUNN_ADJUST_METHOD)
            except Exception:
                test_name = 'Kruskal–Wallis failed'
                p_value = np.nan
                posthoc_df = pd.DataFrame()
        row['Statistical test'] = test_name
        row['Overall P value'] = format_p_value(p_value)
        row['All pairwise comparisons'] = summarize_all_pairwise(posthoc_df)
        row.update(extract_pairwise_p_columns(posthoc_df, outcome_levels))
        row['Significant pairwise comparisons'] = summarize_significant_pairwise(posthoc_df)
        rows.append(row)
    for canonical_var in CANDIDATE_CATEGORICAL_VARS:
        actual_col = find_column(data, canonical_var)
        if actual_col is None:
            pass
            continue
        data[canonical_var] = clean_string_series(data[actual_col])
        test_name, p_value = categorical_overall_test(data, canonical_var, outcome_col)
        posthoc_df = pairwise_categorical_tests(data, canonical_var, outcome_col, outcome_levels)
        all_pairwise_text = summarize_all_pairwise(posthoc_df)
        significant_pairwise_text = summarize_significant_pairwise(posthoc_df)
        pairwise_p_cols = extract_pairwise_p_columns(posthoc_df, outcome_levels)
        temp = data[[canonical_var, outcome_col]].dropna().copy()
        levels = sorted(temp[canonical_var].dropna().unique())
        for idx, level in enumerate(levels):
            row = {'Cohort': cohort_name, 'Variable': label_variable(canonical_var) if idx == 0 else '', 'Level': label_level(level), 'Overall': '', 'Summary': 'n (%)'}
            total_n = len(temp)
            total_count = int((temp[canonical_var] == level).sum())
            row['Overall'] = count_percent_string(total_count, total_n)
            for g in outcome_levels:
                sub = temp[temp[outcome_col] == g]
                denom = len(sub)
                count = int((sub[canonical_var] == level).sum())
                row[f'Type {g}'] = count_percent_string(count, denom)
            if idx == 0:
                row['Statistical test'] = test_name
                row['Overall P value'] = format_p_value(p_value)
                row['All pairwise comparisons'] = all_pairwise_text
                row.update(pairwise_p_cols)
                row['Significant pairwise comparisons'] = significant_pairwise_text
            else:
                row['Statistical test'] = ''
                row['Overall P value'] = ''
                row['All pairwise comparisons'] = ''
                for g1, g2 in itertools.combinations(outcome_levels, 2):
                    row[f'Type {g1} vs Type {g2} P_adj'] = ''
                row['Significant pairwise comparisons'] = ''
            rows.append(row)
    return pd.DataFrame(rows)

def load_and_evaluate_external_cohort(cohort_name, file_path):
    pass
    pass
    pass
    pass
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'：{file_path}')
    df = pd.read_excel(file_path)
    df = clean_column_names(df)
    actual_target_col = find_column(df, target)
    if actual_target_col is None:
        raise ValueError(f'{cohort_name}：{target}')
    X_external = build_feature_dataframe(df, features)
    y_raw = df[actual_target_col].values
    valid_notna = X_external.notna().all(axis=1) & pd.Series(y_raw).notna().values
    X_external = X_external.loc[valid_notna].copy()
    y_raw = y_raw[valid_notna]
    y_encoded, y_used, valid_label_mask = transform_labels_safely(y_raw)
    if not np.all(valid_label_mask):
        X_external = X_external.loc[valid_label_mask].copy()
    if model_requires_scaled_input(best_model_name):
        X_input = scaler.transform(X_external)
    else:
        X_input = X_external.values
    y_pred = best_model.predict(X_input)
    y_proba = safe_predict_proba(best_model, X_input)
    y_proba = align_proba_to_classes(best_model, y_proba)
    y_bin = label_binarize(y_encoded, classes=class_indices)
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    micro_auc = auc(fpr, tpr)
    fpr_grid, mean_tpr, lower_tpr, upper_tpr, auc_mean, auc_lower, auc_upper = bootstrap_micro_roc_band(y_encoded=y_encoded, y_proba=y_proba, n_bootstraps=1000, ci=0.95, random_state=RANDOM_STATE)
    pr_precision, pr_recall, _ = precision_recall_curve(y_bin.ravel(), y_proba.ravel())
    micro_ap = average_precision_score(y_bin, y_proba, average='micro')
    fraction_of_positives, mean_predicted_value = calibration_curve(y_bin.ravel(), y_proba.ravel(), n_bins=10, strategy='quantile')
    brier = brier_score_loss(y_bin.ravel(), y_proba.ravel())
    accuracy = accuracy_score(y_encoded, y_pred)
    macro_f1 = f1_score(y_encoded, y_pred, average='macro', zero_division=0)
    try:
        macro_ovr_auc = roc_auc_score(y_encoded, y_proba, multi_class='ovr', average='macro')
    except Exception:
        macro_ovr_auc = np.nan
    per_class_df = compute_per_class_metrics(y_encoded=y_encoded, y_pred=y_pred, y_proba=y_proba, cohort_name=cohort_name)
    cm = confusion_matrix(y_encoded, y_pred, labels=class_indices)
    variable_table = make_simplified_variable_table(df=df, cohort_name=cohort_name, outcome_col=target)
    pass
    pass
    pass
    pass
    pass
    pass
    return {'cohort_name': cohort_name, 'N': len(y_encoded), 'y_encoded': y_encoded, 'y_pred': y_pred, 'y_proba': y_proba, 'cm': cm, 'fpr': fpr, 'tpr': tpr, 'micro_auc': micro_auc, 'fpr_grid': fpr_grid, 'lower_tpr': lower_tpr, 'upper_tpr': upper_tpr, 'auc_lower': auc_lower, 'auc_upper': auc_upper, 'pr_precision': pr_precision, 'pr_recall': pr_recall, 'micro_ap': micro_ap, 'pr_baseline': np.mean(y_bin.ravel()), 'cal_fraction_of_positives': fraction_of_positives, 'cal_mean_predicted_value': mean_predicted_value, 'brier': brier, 'accuracy': accuracy, 'macro_f1': macro_f1, 'macro_ovr_auc': macro_ovr_auc, 'per_class_df': per_class_df, 'variable_table': variable_table}
external_results = {}
for cohort_name, file_path in EXTERNAL_COHORTS.items():
    external_results[cohort_name] = load_and_evaluate_external_cohort(cohort_name=cohort_name, file_path=file_path)
table1_variable_tests = pd.concat([res['variable_table'] for res in external_results.values()], ignore_index=True)
table2_per_class_metrics = pd.concat([res['per_class_df'] for res in external_results.values()], ignore_index=True)
metric_cols = ['Class accuracy', 'Precision', 'Recall / sensitivity', 'Specificity', 'F1-score', 'OvR-AUC', 'Average precision']
for col in metric_cols:
    if col in table2_per_class_metrics.columns:
        table2_per_class_metrics[col] = table2_per_class_metrics[col].apply(lambda x: round(x, 3) if pd.notna(x) else x)
excel_path = os.path.join(OUTPUT_DIR, 'external_validation_two_summary_tables_all_pairwise.xlsx')
with pd.ExcelWriter(excel_path) as writer:
    table1_variable_tests.to_excel(writer, sheet_name='Table1_variable_tests', index=False)
    table2_per_class_metrics.to_excel(writer, sheet_name='Table2_per_class_metrics', index=False)
table1_variable_tests.to_csv(os.path.join(OUTPUT_DIR, 'Table1_variable_tests_all_pairwise.csv'), index=False, encoding='utf-8-sig')
table2_per_class_metrics.to_csv(os.path.join(OUTPUT_DIR, 'Table2_per_class_metrics.csv'), index=False, encoding='utf-8-sig')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
ax_roc, ax_pr, ax_cal = axes
for cohort_name, res in external_results.items():
    color = COHORT_COLORS.get(cohort_name, '#333333')
    ax_roc.plot(res['fpr'], res['tpr'], color=color, linewidth=2.8, label=f"{cohort_name} (AUC = {res['micro_auc']:.3f}, 95% CI [{res['auc_lower']:.3f}, {res['auc_upper']:.3f}])")
    ax_roc.fill_between(res['fpr_grid'], res['lower_tpr'], res['upper_tpr'], color=color, alpha=0.18, linewidth=0)
ax_roc.plot([0, 1], [0, 1], color='#555555', linestyle='--', linewidth=1.5, alpha=0.8, label='Random classifier')
ax_roc.set_xlabel('False positive rate')
ax_roc.set_ylabel('True positive rate')
ax_roc.set_title('A. ROC curves')
ax_roc.set_xlim(-0.02, 1.02)
ax_roc.set_ylim(-0.02, 1.02)
ax_roc.grid(True, linestyle=':', linewidth=0.7, alpha=0.5)
ax_roc.legend(loc='lower right', frameon=True)
for cohort_name, res in external_results.items():
    color = COHORT_COLORS.get(cohort_name, '#333333')
    ax_pr.plot(res['pr_recall'], res['pr_precision'], color=color, linewidth=2.8, label=f"{cohort_name} (AP = {res['micro_ap']:.3f})")
    ax_pr.axhline(y=res['pr_baseline'], color=color, linestyle='--', linewidth=1.2, alpha=0.45)
ax_pr.set_xlabel('Recall')
ax_pr.set_ylabel('Precision')
ax_pr.set_title('B. Precision–recall curves')
ax_pr.set_xlim(-0.02, 1.02)
ax_pr.set_ylim(-0.02, 1.02)
ax_pr.grid(True, linestyle=':', linewidth=0.7, alpha=0.5)
ax_pr.legend(loc='lower left', frameon=True)
for cohort_name, res in external_results.items():
    color = COHORT_COLORS.get(cohort_name, '#333333')
    marker = COHORT_MARKERS.get(cohort_name, 'o')
    ax_cal.plot(res['cal_mean_predicted_value'], res['cal_fraction_of_positives'], marker=marker, color=color, linewidth=2.5, markersize=6.5, label=f"{cohort_name} (Brier = {res['brier']:.3f})")
ax_cal.plot([0, 1], [0, 1], color='#555555', linestyle='--', linewidth=1.5, alpha=0.8, label='Perfect calibration')
ax_cal.set_xlabel('Mean predicted probability')
ax_cal.set_ylabel('Fraction of positives')
ax_cal.set_title('C. Calibration curves')
ax_cal.set_xlim(-0.02, 1.02)
ax_cal.set_ylim(-0.02, 1.02)
ax_cal.grid(True, linestyle=':', linewidth=0.7, alpha=0.5)
ax_cal.legend(loc='upper left', frameon=True)
fig.suptitle(f'External validation performance of {best_model_name}', fontsize=16, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'combined_external_validation_roc_pr_calibration_with_ci.png'), dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'combined_external_validation_roc_pr_calibration_with_ci.pdf'), bbox_inches='tight')
plt.show()

def make_cm_annotation(cm):
    annot = np.empty_like(cm).astype(object)
    row_sums = cm.sum(axis=1, keepdims=True)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if row_sums[i, 0] > 0:
                pct = cm[i, j] / row_sums[i, 0] * 100
            else:
                pct = 0
            annot[i, j] = f'{cm[i, j]}\n({pct:.1f}%)'
    return annot
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
for ax, (cohort_name, res) in zip(axes, external_results.items()):
    cm = res['cm']
    annot = make_cm_annotation(cm)
    sns.heatmap(cm, annot=annot, fmt='', cmap='YlGnBu', linewidths=0.8, linecolor='white', square=True, cbar=True, xticklabels=[f'Type {c}' for c in le.classes_], yticklabels=[f'Type {c}' for c in le.classes_], annot_kws={'fontsize': 11, 'fontweight': 'bold'}, ax=ax)
    ax.set_title(f"{cohort_name}\nAccuracy = {res['accuracy']:.3f}, Macro-F1 = {res['macro_f1']:.3f}", fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
plt.suptitle('Confusion matrices of external validation cohorts', fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'external_validation_confusion_matrices.png'), dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'external_validation_confusion_matrices.pdf'), bbox_inches='tight')
plt.show()
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass

import os
import glob
import joblib
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
SHAP_INPUT_DIR = 'result'
OUTPUT_DIR = os.path.join('result', 'shap_horizontal_all_models_600dpi')
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_NAME_TO_PLOT = None
MAX_DISPLAY = 4
ROSE_TOP_N = 4
SHOW_INSET_ROSE = False
DPI = 600
SAVE_PDF = True
SAVE_TIFF = True
RANDOM_STATE = 42
FIG_WIDTH = 24
FIG_HEIGHT = 6.8
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
FEATURE_LABELS = {'LLBCE': 'LLBCE', 'PMBT': 'PMBT', 'MRT': 'MRT', 'Depth.of.A': 'Depth of A', 'Depth  of A': 'Depth of A', 'Depth of A': 'Depth of A', 'Depth_of_A': 'Depth of A', 'RAPL': 'RAPL', 'ART': 'ART', 'RH': 'RH', 'LSND': 'LSND', 'age': 'Age', 'sex_male': 'Male sex', 'third_molar_yes': 'Third molar presence', 'jaw_deformity_type3': 'Skeletal Class III'}

def pretty_feature_name(x):
    return FEATURE_LABELS.get(str(x), str(x))

def safe_class_name(x):
    x = str(x).strip()
    if x.lower().startswith('type'):
        return x.replace('type', 'Type')
    if x.lower().startswith('class'):
        return x.replace('class', 'Class')
    try:
        val = float(x)
        if val.is_integer():
            return f'Type {int(val)}'
    except Exception:
        pass
    return f'Type {x}'

def safe_filename(x):
    return str(x).replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('|', '').replace(',', '')

def get_shap_pkl_files():
    all_files = sorted(glob.glob(os.path.join(SHAP_INPUT_DIR, '*_shap_values.pkl')))
    if len(all_files) == 0:
        raise FileNotFoundError(f'{SHAP_INPUT_DIR}*_shap_values.pkl。 SHAP 。')
    if MODEL_NAME_TO_PLOT is None:
        return all_files
    key = MODEL_NAME_TO_PLOT.lower().replace(' ', '_')
    selected = []
    for f in all_files:
        base = os.path.basename(f).lower()
        if key in base:
            selected.append(f)
    if len(selected) == 0:
        pass
        for f in all_files:
            pass
        raise FileNotFoundError(f'{MODEL_NAME_TO_PLOT}SHAP pkl 。')
    return selected

def normalize_shap_values_to_class_list(shap_values, n_features, class_names):
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values
    class_entries = []
    if isinstance(shap_values, list):
        for i, arr in enumerate(shap_values):
            arr = np.asarray(arr)
            if arr.ndim != 2:
                continue
            if arr.shape[1] == n_features + 1:
                arr = arr[:, :-1]
            if arr.shape[1] != n_features:
                continue
            label = class_names[i] if i < len(class_names) else i
            class_entries.append({'class_label': label, 'values': arr})
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            if arr.shape[2] == n_features + 1:
                arr = arr[:, :, :-1]
            if arr.shape[1] == n_features + 1:
                arr = arr[:, :-1, :]
            if arr.shape[1] == n_features:
                n_classes = arr.shape[2]
                for i in range(n_classes):
                    label = class_names[i] if i < len(class_names) else i
                    class_entries.append({'class_label': label, 'values': arr[:, :, i]})
            elif arr.shape[2] == n_features:
                n_classes = arr.shape[1]
                for i in range(n_classes):
                    label = class_names[i] if i < len(class_names) else i
                    class_entries.append({'class_label': label, 'values': arr[:, i, :]})
        elif arr.ndim == 2:
            if arr.shape[1] == n_features + 1:
                arr = arr[:, :-1]
            if arr.shape[1] == n_features:
                class_entries.append({'class_label': 'Overall', 'values': arr})
    if len(class_entries) == 0:
        raise ValueError(f'SHAP 。 n_features={n_features}，shap_values ={type(shap_values)}。')
    return class_entries

def get_display_feature_dataframe(shap_data):
    feature_names = list(shap_data['feature_names'])
    X_data = shap_data['X_data']
    if isinstance(X_data, pd.DataFrame):
        X_display = X_data.copy()
    else:
        X_display = pd.DataFrame(np.asarray(X_data), columns=feature_names)
    for col in feature_names:
        X_display[col] = pd.to_numeric(X_display[col], errors='coerce')
    return X_display

def compute_mean_abs_shap(shap_matrix, feature_names):
    mean_abs = np.nanmean(np.abs(shap_matrix), axis=0)
    imp_df = pd.DataFrame({'feature': feature_names, 'feature_label': [pretty_feature_name(f) for f in feature_names], 'mean_abs_shap': mean_abs})
    total = imp_df['mean_abs_shap'].sum()
    if total > 0:
        imp_df['contribution_percent'] = imp_df['mean_abs_shap'] / total * 100
    else:
        imp_df['contribution_percent'] = 0.0
    imp_df = imp_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    return imp_df

def compute_global_top_indices(class_entries, feature_names, top_n=4):
    abs_means = []
    for entry in class_entries:
        arr = np.asarray(entry['values'])
        abs_means.append(np.nanmean(np.abs(arr), axis=0))
    global_mean = np.nanmean(np.vstack(abs_means), axis=0)
    top_indices = np.argsort(global_mean)[::-1][:min(top_n, len(feature_names))]
    global_df = pd.DataFrame({'feature': feature_names, 'feature_label': [pretty_feature_name(f) for f in feature_names], 'global_mean_abs_shap': global_mean}).sort_values('global_mean_abs_shap', ascending=False).reset_index(drop=True)
    total = global_df['global_mean_abs_shap'].sum()
    if total > 0:
        global_df['global_contribution_percent'] = global_df['global_mean_abs_shap'] / total * 100
    else:
        global_df['global_contribution_percent'] = 0.0
    return (top_indices, global_df)

def get_harmonized_colors(n, cmap_name='Spectral_r'):
    cmap = plt.get_cmap(cmap_name)
    vals = np.linspace(0.08, 0.92, n)
    return [mpl.colors.to_hex(cmap(v)) for v in vals]

def build_feature_color_map(global_top_indices, feature_names, cmap_name='Spectral_r'):
    colors = get_harmonized_colors(len(global_top_indices), cmap_name=cmap_name)
    feature_color_map = {}
    for i, idx in enumerate(global_top_indices):
        feature_color_map[feature_names[idx]] = colors[i]
    return feature_color_map

def normalize_feature_values(values):
    values = pd.to_numeric(pd.Series(values), errors='coerce').values
    if np.all(~np.isfinite(values)):
        return np.full(len(values), 0.5)
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return np.full(len(values), 0.5)
    vmin = np.nanpercentile(valid, 1)
    vmax = np.nanpercentile(valid, 99)
    if vmax == vmin:
        return np.full(len(values), 0.5)
    normed = (values - vmin) / (vmax - vmin)
    normed = np.clip(normed, 0, 1)
    normed[~np.isfinite(normed)] = 0.5
    return normed

def simple_beeswarm(x_values, nbins=90, width=0.34):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.zeros_like(x_values, dtype=float)
    finite_mask = np.isfinite(x_values)
    if finite_mask.sum() <= 1:
        return y_values
    x = x_values[finite_mask]
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if xmin == xmax:
        return y_values
    counts, edges = np.histogram(x, bins=nbins, range=(xmin, xmax))
    max_count = np.max(counts) if len(counts) > 0 else 1
    finite_indices = np.where(finite_mask)[0]
    bin_ids = np.digitize(x, edges) - 1
    bin_ids = np.clip(bin_ids, 0, nbins - 1)
    rng = np.random.default_rng(RANDOM_STATE)
    for b in np.unique(bin_ids):
        local_idx = np.where(bin_ids == b)[0]
        global_idx = finite_indices[local_idx]
        count = len(global_idx)
        if count <= 1:
            y_values[global_idx] = 0
            continue
        current_width = counts[b] / max_count * width
        offsets = np.linspace(-current_width, current_width, count)
        rng.shuffle(offsets)
        y_values[global_idx] = offsets
    return y_values

def draw_true_rose_inset(ax, shap_matrix, feature_names, global_top_indices, feature_color_map, inset_position=(0.68, 0.2, 0.22, 0.22)):
    mean_abs = np.nanmean(np.abs(shap_matrix), axis=0)
    values = np.array([mean_abs[idx] for idx in global_top_indices], dtype=float)
    if np.sum(values) <= 0:
        contributions = np.zeros_like(values)
    else:
        contributions = values / np.sum(values)
    n = len(global_top_indices)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.78
    vmax = contributions.max() if contributions.max() > 0 else 1.0
    radii = 0.26 + 0.74 * (contributions / vmax)
    inset_ax = ax.inset_axes(inset_position, projection='polar')
    inset_ax.set_theta_offset(np.pi / 2)
    inset_ax.set_theta_direction(-1)
    for ang, feat_idx, radius, contrib in zip(theta, global_top_indices, radii, contributions):
        feat = feature_names[feat_idx]
        color = feature_color_map.get(feat, '#999999')
        inset_ax.bar(ang, radius, width=width, bottom=0.0, color=color, edgecolor='white', linewidth=0.9, align='center', alpha=0.96)
        if contrib >= 0.04:
            inset_ax.text(ang, radius + 0.09, f'{contrib * 100:.1f}%', ha='center', va='center', fontsize=7.2, fontweight='bold')
    inset_ax.set_ylim(0, 1.18)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.grid(False)
    inset_ax.spines['polar'].set_visible(False)
    inset_ax.set_facecolor('none')

def plot_single_class_panel(ax, shap_matrix, X_display, feature_names, class_label, panel_letter, global_top_indices, feature_color_map, max_display=4, show_feature_title=False, show_inset_rose=False):
    imp_df = compute_mean_abs_shap(shap_matrix, feature_names)
    display_df = imp_df.head(min(max_display, len(imp_df))).copy()
    sorted_features = display_df['feature'].tolist()
    sorted_feature_labels = display_df['feature_label'].tolist()
    cmap = plt.get_cmap('Spectral_r')
    feature_indices = [feature_names.index(f) for f in sorted_features]
    selected_shap = shap_matrix[:, feature_indices]
    xmax = np.nanpercentile(np.abs(selected_shap), 99)
    if not np.isfinite(xmax) or xmax == 0:
        xmax = np.nanmax(np.abs(selected_shap))
    if not np.isfinite(xmax) or xmax == 0:
        xmax = 1.0
    xmax = xmax * 1.24
    for i, feature in enumerate(sorted_features):
        feature_index = feature_names.index(feature)
        sv = np.asarray(shap_matrix[:, feature_index], dtype=float)
        fv = X_display[feature].values
        fv_norm = normalize_feature_values(fv)
        y_offset = simple_beeswarm(sv, nbins=90, width=0.34)
        ax.scatter(sv, i + y_offset, c=fv_norm, cmap=cmap, s=17, alpha=0.94, edgecolors='none', rasterized=True)
    ax.axvline(x=0, color='#9CB832', linestyle='-', linewidth=1.7, alpha=0.95, zorder=0)
    ax.set_xlim(-xmax, xmax)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_feature_labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel('Feature impact on model output (SHAP)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=11.5, fontweight='bold')
    ax.grid(axis='x', linestyle=':', linewidth=0.6, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.11, 1.06, panel_letter, transform=ax.transAxes, fontsize=17, va='top', ha='right', fontweight='bold')
    ax.text(0.97, 0.98, safe_class_name(class_label), transform=ax.transAxes, fontsize=13.5, fontweight='bold', ha='right', va='top')
    if show_feature_title:
        ax.text(-0.18, 1.15, 'Features', transform=ax.transAxes, fontsize=15, fontweight='bold', ha='left')
    if show_inset_rose:
        draw_true_rose_inset(ax=ax, shap_matrix=shap_matrix, feature_names=feature_names, global_top_indices=global_top_indices, feature_color_map=feature_color_map, inset_position=(0.68, 0.2, 0.22, 0.22))
    return imp_df

def draw_multiclass_ring_pie(ax, class_entries, feature_names, global_top_indices, feature_color_map, model_name):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    n_classes = len(class_entries)
    base_radius = 0.25
    ring_width = 0.17
    ring_gap = 0.025
    for class_i, entry in enumerate(class_entries):
        shap_matrix = np.asarray(entry['values'])
        mean_abs = np.nanmean(np.abs(shap_matrix), axis=0)
        raw = np.array([mean_abs[idx] for idx in global_top_indices], dtype=float)
        if np.sum(raw) <= 0:
            contrib = np.zeros_like(raw)
        else:
            contrib = raw / np.sum(raw)
        current_theta = 0.0
        outer_radius = base_radius + (n_classes - class_i) * (ring_width + ring_gap)
        inner_radius = outer_radius - ring_width
        for feat_idx, prob in zip(global_top_indices, contrib):
            width_val = 2 * np.pi * prob
            feat = feature_names[feat_idx]
            color = feature_color_map.get(feat, '#999999')
            ax.bar(current_theta, height=ring_width, width=width_val, bottom=inner_radius, color=color, edgecolor='white', linewidth=1.0, align='edge', alpha=0.96)
            current_theta += width_val
        ax.text(np.deg2rad(0), inner_radius + ring_width / 2, safe_class_name(entry['class_label']), ha='center', va='center', fontsize=8.3, fontweight='bold', color='white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_title(f'Feature contribution\nacross classes\n{model_name}', fontsize=12.5, fontweight='bold', pad=13)
    legend_labels = [pretty_feature_name(feature_names[i]) for i in global_top_indices]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=feature_color_map[feature_names[i]]) for i in global_top_indices]
    ax.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.23), ncol=2, frameon=False, title='Feature contribution', handlelength=1.3, handleheight=0.8, fontsize=8.6, title_fontsize=9.3)

def generate_horizontal_shap_figure(class_entries, X_display, feature_names, model_name, save_path):
    n_classes = len(class_entries)
    global_top_indices, global_importance_df = compute_global_top_indices(class_entries=class_entries, feature_names=feature_names, top_n=ROSE_TOP_N)
    feature_color_map = build_feature_color_map(global_top_indices=global_top_indices, feature_names=feature_names, cmap_name='Spectral_r')
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = gridspec.GridSpec(1, n_classes + 2, figure=fig, width_ratios=[1.18] * n_classes + [0.95, 0.05], wspace=0.35)
    panel_letters = list('abcdefghijklmnopqrstuvwxyz')
    importance_tables = []
    rose_tables = []
    for idx, entry in enumerate(class_entries):
        ax = fig.add_subplot(gs[0, idx])
        imp_df = plot_single_class_panel(ax=ax, shap_matrix=np.asarray(entry['values']), X_display=X_display, feature_names=feature_names, class_label=entry['class_label'], panel_letter=panel_letters[idx], global_top_indices=global_top_indices, feature_color_map=feature_color_map, max_display=MAX_DISPLAY, show_feature_title=idx == 0, show_inset_rose=SHOW_INSET_ROSE)
        imp_df.insert(0, 'model', model_name)
        imp_df.insert(1, 'class', safe_class_name(entry['class_label']))
        importance_tables.append(imp_df)
        shap_matrix = np.asarray(entry['values'])
        mean_abs = np.nanmean(np.abs(shap_matrix), axis=0)
        raw = np.array([mean_abs[i] for i in global_top_indices], dtype=float)
        if np.sum(raw) <= 0:
            contrib = np.zeros_like(raw)
        else:
            contrib = raw / np.sum(raw) * 100
        rose_row = {'model': model_name, 'class': safe_class_name(entry['class_label'])}
        for feat_idx, percent in zip(global_top_indices, contrib):
            rose_row[pretty_feature_name(feature_names[feat_idx])] = percent
        rose_tables.append(rose_row)
    ax_ring = fig.add_subplot(gs[0, n_classes], polar=True)
    draw_multiclass_ring_pie(ax=ax_ring, class_entries=class_entries, feature_names=feature_names, global_top_indices=global_top_indices, feature_color_map=feature_color_map, model_name=model_name)
    cax = fig.add_subplot(gs[0, n_classes + 1])
    sm = mpl.cm.ScalarMappable(cmap=plt.get_cmap('Spectral_r'), norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Feature value', fontsize=10)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle(f'SHAP summary with feature contribution - {model_name}', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    if SAVE_PDF:
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    if SAVE_TIFF:
        plt.savefig(save_path.replace('.png', '.tif'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    final_importance = pd.concat(importance_tables, ignore_index=True)
    rose_df = pd.DataFrame(rose_tables)
    return (final_importance, global_importance_df, rose_df)

def main():
    shap_files = get_shap_pkl_files()
    all_importance_tables = []
    all_global_tables = []
    all_rose_tables = []
    pass
    pass
    pass
    pass
    for f in shap_files:
        pass
    for pkl_file in shap_files:
        try:
            shap_data = joblib.load(pkl_file)
            model_name = shap_data.get('model_name', os.path.basename(pkl_file).replace('_shap_values.pkl', ''))
            feature_names = list(shap_data['feature_names'])
            shap_values = shap_data['shap_values']
            local_class_names = shap_data.get('class_names', [])
            try:
                local_class_names = list(local_class_names)
            except Exception:
                local_class_names = []
            if len(local_class_names) == 0:
                local_class_names = ['Overall']
            X_display = get_display_feature_dataframe(shap_data)
            class_entries = normalize_shap_values_to_class_list(shap_values=shap_values, n_features=len(feature_names), class_names=local_class_names)
            pass
            pass
            pass
            pass
            pass
            save_path = os.path.join(OUTPUT_DIR, f'{safe_filename(model_name)}_horizontal_shap_600dpi.png')
            importance_df, global_df, rose_df = generate_horizontal_shap_figure(class_entries=class_entries, X_display=X_display, feature_names=feature_names, model_name=model_name, save_path=save_path)
            all_importance_tables.append(importance_df)
            global_df.insert(0, 'model', model_name)
            all_global_tables.append(global_df)
            all_rose_tables.append(rose_df)
            pass
            pass
        except Exception as e:
            pass
            pass
            import traceback
            traceback.print_exc()
            continue
    excel_path = os.path.join(OUTPUT_DIR, 'shap_all_models_feature_contribution_summary.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        if len(all_importance_tables) > 0:
            final_importance_df = pd.concat(all_importance_tables, ignore_index=True)
            final_importance_df.to_excel(writer, sheet_name='Class_specific_SHAP', index=False)
            final_importance_df.to_csv(os.path.join(OUTPUT_DIR, 'shap_all_models_class_specific_SHAP.csv'), index=False, encoding='utf-8-sig')
        if len(all_global_tables) > 0:
            final_global_df = pd.concat(all_global_tables, ignore_index=True)
            final_global_df.to_excel(writer, sheet_name='Global_SHAP', index=False)
            final_global_df.to_csv(os.path.join(OUTPUT_DIR, 'shap_all_models_global_SHAP.csv'), index=False, encoding='utf-8-sig')
        if len(all_rose_tables) > 0:
            final_rose_df = pd.concat(all_rose_tables, ignore_index=True)
            final_rose_df.to_excel(writer, sheet_name='Ring_contribution', index=False)
            final_rose_df.to_csv(os.path.join(OUTPUT_DIR, 'shap_all_models_ring_contribution.csv'), index=False, encoding='utf-8-sig')
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()

import os
import re
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import shap
try:
    from catboost import Pool
except:
    Pool = None
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
SHAP_CMAP = LinearSegmentedColormap.from_list('beeswarm_like_cmap', ['#355C9A', '#62B6CB', '#D9E58B', '#F2C572', '#E76F51', '#9B2226'], N=256)

def safe_filename(text):
    text = str(text)
    text = text.replace(' ', '_')
    text = text.replace('/', '_')
    text = text.replace('\\', '_')
    text = text.replace(':', '_')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace(',', '')
    return text

def format_type_label(label):
    s = str(label).strip()
    m = re.search('(\\d+)', s)
    if m:
        return f'Type {m.group(1)}'
    if s.lower().startswith('type'):
        return s
    return f'Type {s}'

def format_type_for_filename(label):
    type_label = format_type_label(label)
    m = re.search('(\\d+)', type_label)
    if m:
        return f'type_{m.group(1)}'
    return safe_filename(type_label.lower())

def get_class_shap_matrix(shap_values, class_idx, n_features):
    if isinstance(shap_values, list):
        return shap_values[class_idx]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        if arr.shape[1] == n_features:
            return arr[:, :, class_idx]
        elif arr.shape[2] == n_features:
            return arr[:, class_idx, :]
        else:
            raise ValueError(f'SHAP ：{arr.shape}')
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f'SHAP ：{arr.ndim}')

def get_n_classes_from_shap(shap_values, n_features):
    if isinstance(shap_values, list):
        return len(shap_values)
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        if arr.shape[1] == n_features:
            return arr.shape[2]
        elif arr.shape[2] == n_features:
            return arr.shape[1]
    return 1
pass
for model_name in list(final_results.keys()):
    try:
        pass
        try:
            shap_data = joblib.load(f"result/{model_name.replace(' ', '_')}_shap_values.pkl")
            shap_values = shap_data['shap_values']
            feature_names = shap_data['feature_names']
            X_data = shap_data['X_data']
        except:
            pass
            df_clean = df[features + [target]].dropna()
            if model_name in ['Logistic Regression', 'SVM']:
                X_data = scaler.transform(df_clean[features])
            else:
                X_data = df_clean[features].values
            feature_names = features
            model = final_results[model_name]['model']
            if model_name == 'CatBoost':
                try:
                    if Pool is None:
                        raise ImportError('catboost.Pool not available')
                    shap_values = model.get_feature_importance(data=Pool(X_data, feature_names=feature_names), type='ShapValues')
                    if shap_values.ndim == 2 and shap_values.shape[1] == len(feature_names) + 1:
                        shap_values = shap_values[:, :-1]
                except:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_data)
            elif model_name == 'Random Forest':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_data)
            elif model_name == 'XGBoost':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_data)
            elif model_name == 'Logistic Regression':
                explainer = shap.LinearExplainer(model, X_data)
                shap_values = explainer.shap_values(X_data)
            else:

                def model_predict(X):
                    return model.predict_proba(X)
                background = shap.sample(X_data, 100)
                explainer = shap.KernelExplainer(model_predict, background)
                shap_values = explainer.shap_values(X_data)
            shap_data = {'shap_values': shap_values, 'feature_names': feature_names, 'class_names': class_names, 'X_data': X_data, 'model_name': model_name}
            joblib.dump(shap_data, f"result/{model_name.replace(' ', '_')}_shap_values.pkl")
        pass
        if hasattr(shap_values, 'shape'):
            pass
        n_features = len(feature_names)
        n_classes = get_n_classes_from_shap(shap_values, n_features)
        for feature_idx, feature_name in enumerate(feature_names):
            pass
            if n_classes > 1:
                pass
                for class_idx in range(n_classes):
                    raw_class_name = class_names[class_idx]
                    type_label = format_type_label(raw_class_name)
                    type_file_label = format_type_for_filename(raw_class_name)
                    pass
                    class_shap_matrix = get_class_shap_matrix(shap_values=shap_values, class_idx=class_idx, n_features=n_features)
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(feature_idx, class_shap_matrix, X_data, feature_names=feature_names, cmap=SHAP_CMAP, alpha=0.9, dot_size=16, show=False)
                    plt.title(f'{model_name} - SHAP Dependence: {feature_name}\n({type_label})', fontweight='bold', fontsize=12)
                    plt.tight_layout()
                    output_path = f"result/{model_name.replace(' ', '_')}_shap_dep_{safe_filename(feature_name)}_{type_file_label}.png"
                    plt.savefig(output_path, dpi=600, bbox_inches='tight')
                    plt.show()
                    pass
            else:
                pass
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature_idx, shap_values, X_data, feature_names=feature_names, cmap=SHAP_CMAP, alpha=0.9, dot_size=16, show=False)
                plt.title(f'{model_name} - SHAP Dependence: {feature_name}', fontweight='bold', fontsize=12)
                plt.tight_layout()
                output_path = f"result/{model_name.replace(' ', '_')}_shap_dep_{safe_filename(feature_name)}.png"
                plt.savefig(output_path, dpi=600, bbox_inches='tight')
                plt.show()
                pass
        pass
    except Exception as e:
        pass
        import traceback
        traceback.print_exc()

import os
import itertools
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, f_oneway, kruskal, chi2_contingency, fisher_exact
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
file_path = 'your_data_path/your_data.xlsx'
output_dir = 'your_output_path'
os.makedirs(output_dir, exist_ok=True)
OUTCOME_COL = 'Type.of.fracture.line'
continuous_vars = ['LLBCE', 'PMBT', 'MRT', 'Depth.of.A', 'RAPL', 'ART', 'RH', 'LSND', 'age']
categorical_vars = ['sex', 'type of jaw deformity', 'third molar presence']
VARIABLE_LABELS = {'LLBCE': 'LLBCE', 'PMBT': 'PMBT', 'MRT': 'MRT', 'Depth.of.A': 'Depth of A', 'RAPL': 'RAPL', 'ART': 'ART', 'RH': 'RH', 'LSND': 'LSND', 'age': 'Age', 'sex': 'Sex', 'type of jaw deformity': 'Jaw deformity type', 'third molar presence': 'Third molar presence'}
LEVEL_LABELS = {'male': 'Male', 'female': 'Female', 'yes': 'Yes', 'no': 'No', '2': 'Type II', '2.0': 'Type II', '3': 'Type III', '3.0': 'Type III'}
DUNN_ADJUST_METHOD = 'bonferroni'
CATEGORICAL_POSTHOC_ADJUST_METHOD = 'bonferroni'
N_PERMUTATIONS = 10000
RANDOM_STATE = 42

def clean_column_names(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace('\xa0', ' ', regex=False).str.replace('\t', ' ', regex=False).str.strip()
    return df

def clean_string_series(s):
    return s.astype(str).str.replace('\xa0', ' ', regex=False).str.replace('\t', ' ', regex=False).str.strip().str.lower()

def sort_outcome_levels(levels):

    def key_func(x):
        try:
            return float(x)
        except Exception:
            return str(x)
    return sorted(levels, key=key_func)

def format_p_value(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'

def median_iqr_string(x):
    x = pd.to_numeric(pd.Series(x), errors='coerce').dropna()
    if len(x) == 0:
        return ''
    median = np.median(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    return f'{median:.2f} [{q1:.2f}–{q3:.2f}]'

def count_percent_string(count, denom):
    if denom == 0:
        return '0 (0.0%)'
    return f'{int(count)} ({100 * count / denom:.1f}%)'

def label_variable(var):
    return VARIABLE_LABELS.get(var, var)

def label_level(level):
    level_str = str(level).strip().lower()
    return LEVEL_LABELS.get(level_str, str(level))

def check_expected_counts(expected):
    expected = np.asarray(expected)
    if np.any(expected < 1):
        return False
    prop_ge_5 = np.mean(expected >= 5)
    if prop_ge_5 < 0.8:
        return False
    return True

def chi_square_stat_from_table(table):
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    return chi2

def monte_carlo_chi_square_test(cat_values, group_values, n_permutations=10000, random_state=42):
    rng = np.random.default_rng(random_state)
    temp = pd.DataFrame({'cat': cat_values, 'group': group_values}).dropna()
    observed_table = pd.crosstab(temp['cat'], temp['group'])
    observed_chi2 = chi_square_stat_from_table(observed_table)
    group_array = temp['group'].values.copy()
    cat_array = temp['cat'].values.copy()
    perm_chi2 = []
    for _ in range(n_permutations):
        permuted_group = rng.permutation(group_array)
        perm_table = pd.crosstab(cat_array, permuted_group)
        perm_table = perm_table.reindex(index=observed_table.index, columns=observed_table.columns, fill_value=0)
        try:
            chi2_val = chi_square_stat_from_table(perm_table)
            perm_chi2.append(chi2_val)
        except Exception:
            continue
    perm_chi2 = np.array(perm_chi2)
    p_value = (np.sum(perm_chi2 >= observed_chi2) + 1) / (len(perm_chi2) + 1)
    return (observed_chi2, p_value)

def categorical_overall_test(data, cat_var, outcome_col, n_permutations=10000):
    temp = data[[cat_var, outcome_col]].dropna().copy()
    table = pd.crosstab(temp[cat_var], temp[outcome_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {'test': 'Not applicable', 'statistic': np.nan, 'p_value': np.nan, 'table': table}
    chi2, p_chi, dof, expected = chi2_contingency(table, correction=False)
    expected_ok = check_expected_counts(expected)
    if expected_ok:
        return {'test': 'Pearson chi-square', 'statistic': chi2, 'p_value': p_chi, 'table': table}
    if table.shape == (2, 2):
        odds_ratio, p_fisher = fisher_exact(table.values)
        return {'test': 'Fisher exact', 'statistic': odds_ratio, 'p_value': p_fisher, 'table': table}
    chi2_mc, p_mc = monte_carlo_chi_square_test(cat_values=temp[cat_var], group_values=temp[outcome_col], n_permutations=n_permutations, random_state=RANDOM_STATE)
    return {'test': 'Monte Carlo chi-square', 'statistic': chi2_mc, 'p_value': p_mc, 'table': table}

def pairwise_categorical_tests(data, cat_var, outcome_col, outcome_levels):
    records = []
    for g1, g2 in itertools.combinations(outcome_levels, 2):
        temp = data[data[outcome_col].isin([g1, g2])][[cat_var, outcome_col]].dropna().copy()
        if temp.empty:
            continue
        table = pd.crosstab(temp[cat_var], temp[outcome_col])
        if table.shape[0] < 2 or table.shape[1] < 2:
            records.append({'variable': cat_var, 'comparison': f'Type {g1} vs Type {g2}', 'test': 'Not applicable', 'statistic': np.nan, 'raw_p': np.nan})
            continue
        chi2, p_chi, dof, expected = chi2_contingency(table, correction=False)
        expected_ok = check_expected_counts(expected)
        if expected_ok:
            test_name = 'Pearson chi-square'
            stat = chi2
            p_value = p_chi
        elif table.shape == (2, 2):
            stat, p_value = fisher_exact(table.values)
            test_name = 'Fisher exact'
        else:
            stat, p_value = monte_carlo_chi_square_test(cat_values=temp[cat_var], group_values=temp[outcome_col], n_permutations=N_PERMUTATIONS, random_state=RANDOM_STATE)
            test_name = 'Monte Carlo chi-square'
        records.append({'variable': cat_var, 'comparison': f'Type {g1} vs Type {g2}', 'test': test_name, 'statistic': stat, 'raw_p': p_value})
    result = pd.DataFrame(records)
    if len(result) > 0:
        mask = result['raw_p'].notna()
        if mask.sum() > 0:
            adjusted = multipletests(result.loc[mask, 'raw_p'], method=CATEGORICAL_POSTHOC_ADJUST_METHOD)[1]
            result.loc[mask, 'adjusted_p'] = adjusted
            result['adjust_method'] = CATEGORICAL_POSTHOC_ADJUST_METHOD
    return result

def dunn_test(data, value_col, group_col, adjust_method='bonferroni'):
    temp = data[[value_col, group_col]].dropna().copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors='coerce')
    temp = temp.dropna()
    groups = sort_outcome_levels(temp[group_col].unique())
    if len(groups) < 2:
        return pd.DataFrame()
    temp['rank'] = stats.rankdata(temp[value_col].values)
    N = len(temp)
    _, tie_counts = np.unique(temp[value_col].values, return_counts=True)
    if N > 1:
        tie_correction = 1 - np.sum(tie_counts ** 3 - tie_counts) / (N ** 3 - N)
    else:
        tie_correction = 1
    if tie_correction <= 0:
        tie_correction = 1
    group_stats = {}
    for g in groups:
        sub = temp[temp[group_col] == g]
        group_stats[g] = {'n': len(sub), 'mean_rank': sub['rank'].mean()}
    records = []
    for g1, g2 in itertools.combinations(groups, 2):
        n1 = group_stats[g1]['n']
        n2 = group_stats[g2]['n']
        if n1 == 0 or n2 == 0:
            continue
        r1 = group_stats[g1]['mean_rank']
        r2 = group_stats[g2]['mean_rank']
        se = np.sqrt(N * (N + 1) / 12.0 * (1.0 / n1 + 1.0 / n2) * tie_correction)
        if se == 0:
            z = np.nan
            p = np.nan
        else:
            z = (r1 - r2) / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
        records.append({'variable': value_col, 'comparison': f'Type {g1} vs Type {g2}', 'mean_rank_1': r1, 'mean_rank_2': r2, 'z': z, 'raw_p': p})
    result = pd.DataFrame(records)
    if len(result) > 0:
        mask = result['raw_p'].notna()
        if mask.sum() > 0:
            adjusted = multipletests(result.loc[mask, 'raw_p'], method=adjust_method)[1]
            result.loc[mask, 'adjusted_p'] = adjusted
            result['adjust_method'] = adjust_method
    return result

def tukey_posthoc(data, value_col, group_col):
    temp = data[[value_col, group_col]].dropna().copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors='coerce')
    temp = temp.dropna()
    tukey = pairwise_tukeyhsd(endog=temp[value_col], groups=temp[group_col], alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_df = tukey_df.rename(columns={'group1': 'group1', 'group2': 'group2', 'p-adj': 'adjusted_p', 'meandiff': 'mean_diff', 'lower': 'ci_low', 'upper': 'ci_high', 'reject': 'reject'})
    tukey_df['variable'] = value_col
    tukey_df['comparison'] = tukey_df.apply(lambda r: f"Type {r['group1']} vs Type {r['group2']}", axis=1)
    tukey_df['raw_p'] = np.nan
    tukey_df['adjust_method'] = 'Tukey HSD'
    keep_cols = ['variable', 'comparison', 'mean_diff', 'ci_low', 'ci_high', 'raw_p', 'adjusted_p', 'adjust_method', 'reject']
    return tukey_df[keep_cols]

def summarize_posthoc_for_main_table(posthoc_df, alpha=0.05):
    if posthoc_df is None or len(posthoc_df) == 0:
        return ('', '')
    df = posthoc_df.copy()
    if 'adjusted_p' not in df.columns:
        return ('', '')
    pieces_all = []
    pieces_sig = []
    for _, r in df.iterrows():
        comp = r.get('comparison', '')
        p = r.get('adjusted_p', np.nan)
        if pd.isna(p):
            p_text = ''
        else:
            p_text = format_p_value(p)
        if comp != '':
            pieces_all.append(f'{comp}: P_adj={p_text}')
            if pd.notna(p) and p < alpha:
                pieces_sig.append(f'{comp}: P_adj={p_text}')
    all_text = '; '.join(pieces_all)
    sig_text = '; '.join(pieces_sig) if len(pieces_sig) > 0 else 'None'
    return (all_text, sig_text)
df = pd.read_excel(file_path)
df = clean_column_names(df)
pass
pass
required_cols = [OUTCOME_COL] + continuous_vars + categorical_vars
missing_cols = [c for c in required_cols if c not in df.columns]
if len(missing_cols) > 0:
    raise ValueError(f'Excel ，：{missing_cols}Excel ：{df.columns.tolist()}')
data = df[required_cols].copy()
data[OUTCOME_COL] = data[OUTCOME_COL].astype(str).str.strip()
outcome_levels = sort_outcome_levels(data[OUTCOME_COL].dropna().unique())
pass
pass
for col in continuous_vars:
    data[col] = pd.to_numeric(data[col], errors='coerce')
for col in categorical_vars:
    data[col] = clean_string_series(data[col])
for col in categorical_vars:
    pass
    pass
main_rows = []
normality_rows = []
continuous_posthoc_tables = []
method_decision_rows = []
for var in continuous_vars:
    var_label = label_variable(var)
    row = {'Variable': var_label, 'Level': '', 'Overall': median_iqr_string(data[var]), 'Summary format': 'Median [Q1–Q3]'}
    group_values = []
    for g in outcome_levels:
        x = data.loc[data[OUTCOME_COL] == g, var].dropna()
        row[f'Type {g}'] = median_iqr_string(x)
        group_values.append(x)
        if len(x) >= 3:
            try:
                W, p_shapiro = shapiro(x)
            except Exception:
                W, p_shapiro = (np.nan, np.nan)
        else:
            W, p_shapiro = (np.nan, np.nan)
        normality_rows.append({'variable': var, 'group': f'Type {g}', 'n': len(x), 'shapiro_W': W, 'shapiro_p': p_shapiro, 'normal_by_Shapiro_p_ge_0.05': bool(pd.notna(p_shapiro) and p_shapiro >= 0.05)})
    group_normal_p = [r['shapiro_p'] for r in normality_rows if r['variable'] == var]
    all_groups_have_shapiro = all((pd.notna(p) for p in group_normal_p))
    all_normal = all_groups_have_shapiro and all((p >= 0.05 for p in group_normal_p))
    valid_group_values = [x for x in group_values if len(x) > 0]
    posthoc_df = pd.DataFrame()
    if len(valid_group_values) < 2:
        test_name = 'Not applicable'
        stat_value = np.nan
        p_value = np.nan
        posthoc_name = ''
    elif all_normal:
        try:
            stat_value, p_value = f_oneway(*valid_group_values)
            test_name = 'One-way ANOVA'
            posthoc_name = 'Tukey HSD'
            posthoc_df = tukey_posthoc(data=data, value_col=var, group_col=OUTCOME_COL)
        except Exception as e:
            test_name = 'One-way ANOVA failed'
            stat_value = np.nan
            p_value = np.nan
            posthoc_name = ''
            pass
    else:
        try:
            stat_value, p_value = kruskal(*valid_group_values)
            test_name = 'Kruskal-Wallis'
            posthoc_name = f'Dunn-{DUNN_ADJUST_METHOD}'
            posthoc_df = dunn_test(data=data, value_col=var, group_col=OUTCOME_COL, adjust_method=DUNN_ADJUST_METHOD)
        except Exception as e:
            test_name = 'Kruskal-Wallis failed'
            stat_value = np.nan
            p_value = np.nan
            posthoc_name = ''
            pass
    if len(posthoc_df) > 0:
        posthoc_df['variable_label'] = var_label
        posthoc_df['overall_test'] = test_name
        posthoc_df['overall_p'] = p_value
        continuous_posthoc_tables.append(posthoc_df)
    posthoc_all_text, posthoc_sig_text = summarize_posthoc_for_main_table(posthoc_df)
    row['Statistical test'] = test_name
    row['Post hoc test'] = posthoc_name
    row['P value'] = format_p_value(p_value)
    row['Raw P value'] = p_value
    row['All pairwise comparisons'] = posthoc_all_text
    row['Significant pairwise comparisons'] = posthoc_sig_text
    main_rows.append(row)
    method_decision_rows.append({'variable': var, 'all_groups_normal_by_Shapiro': all_normal, 'test_used': test_name, 'posthoc_used': posthoc_name, 'overall_p_value': p_value, 'all_pairwise_comparisons': posthoc_all_text, 'significant_pairwise_comparisons': posthoc_sig_text})
if len(continuous_posthoc_tables) > 0:
    continuous_posthoc_df = pd.concat(continuous_posthoc_tables, ignore_index=True)
else:
    continuous_posthoc_df = pd.DataFrame()
categorical_overall_rows = []
categorical_posthoc_tables = []
for var in categorical_vars:
    var_label = label_variable(var)
    test_result = categorical_overall_test(data=data, cat_var=var, outcome_col=OUTCOME_COL, n_permutations=N_PERMUTATIONS)
    p_value = test_result['p_value']
    test_name = test_result['test']
    stat_value = test_result['statistic']
    categorical_overall_rows.append({'variable': var, 'variable_label': var_label, 'test': test_name, 'statistic': stat_value, 'p_value': p_value})
    cat_posthoc = pairwise_categorical_tests(data=data, cat_var=var, outcome_col=OUTCOME_COL, outcome_levels=outcome_levels)
    if len(cat_posthoc) > 0:
        cat_posthoc['variable_label'] = var_label
        cat_posthoc['overall_test'] = test_name
        cat_posthoc['overall_p'] = p_value
        categorical_posthoc_tables.append(cat_posthoc)
    posthoc_all_text, posthoc_sig_text = summarize_posthoc_for_main_table(cat_posthoc)
    temp = data[[var, OUTCOME_COL]].dropna().copy()
    levels = sorted(temp[var].dropna().unique())
    for idx, level in enumerate(levels):
        level_label = label_level(level)
        row = {'Variable': var_label if idx == 0 else '', 'Level': level_label, 'Overall': '', 'Summary format': 'n (%)'}
        total_n = len(temp)
        total_count = (temp[var] == level).sum()
        row['Overall'] = count_percent_string(total_count, total_n)
        for g in outcome_levels:
            sub = temp[temp[OUTCOME_COL] == g]
            denom = len(sub)
            count = (sub[var] == level).sum()
            row[f'Type {g}'] = count_percent_string(count, denom)
        if idx == 0:
            row['Statistical test'] = test_name
            row['Post hoc test'] = f'Pairwise tests with {CATEGORICAL_POSTHOC_ADJUST_METHOD} correction'
            row['P value'] = format_p_value(p_value)
            row['Raw P value'] = p_value
            row['All pairwise comparisons'] = posthoc_all_text
            row['Significant pairwise comparisons'] = posthoc_sig_text
        else:
            row['Statistical test'] = ''
            row['Post hoc test'] = ''
            row['P value'] = ''
            row['Raw P value'] = np.nan
            row['All pairwise comparisons'] = ''
            row['Significant pairwise comparisons'] = ''
        main_rows.append(row)
if len(categorical_posthoc_tables) > 0:
    categorical_posthoc_df = pd.concat(categorical_posthoc_tables, ignore_index=True)
else:
    categorical_posthoc_df = pd.DataFrame()
main_table = pd.DataFrame(main_rows)
group_cols = [f'Type {g}' for g in outcome_levels]
ordered_cols = ['Variable', 'Level', 'Overall'] + group_cols + ['Summary format', 'Statistical test', 'Post hoc test', 'P value', 'Raw P value', 'All pairwise comparisons', 'Significant pairwise comparisons']
main_table = main_table[ordered_cols]
normality_df = pd.DataFrame(normality_rows)
method_decision_df = pd.DataFrame(method_decision_rows)
categorical_overall_df = pd.DataFrame(categorical_overall_rows)
main_excel_path = os.path.join(output_dir, 'main_table_for_copy.xlsx')
main_csv_path = os.path.join(output_dir, 'main_table_for_copy.csv')
main_table.to_excel(main_excel_path, index=False)
main_table.to_csv(main_csv_path, index=False, encoding='utf-8-sig')
normality_df.to_excel(os.path.join(output_dir, 'normality_results.xlsx'), index=False)
method_decision_df.to_excel(os.path.join(output_dir, 'continuous_method_decisions.xlsx'), index=False)
continuous_posthoc_df.to_excel(os.path.join(output_dir, 'continuous_posthoc_results.xlsx'), index=False)
categorical_overall_df.to_excel(os.path.join(output_dir, 'categorical_overall_tests.xlsx'), index=False)
categorical_posthoc_df.to_excel(os.path.join(output_dir, 'categorical_posthoc_results.xlsx'), index=False)
summary_excel_path = os.path.join(output_dir, 'statistical_analysis_summary.xlsx')
with pd.ExcelWriter(summary_excel_path) as writer:
    main_table.to_excel(writer, sheet_name='Main_Table', index=False)
    normality_df.to_excel(writer, sheet_name='Shapiro_Normality', index=False)
    method_decision_df.to_excel(writer, sheet_name='Continuous_Methods', index=False)
    continuous_posthoc_df.to_excel(writer, sheet_name='Continuous_Posthoc', index=False)
    categorical_overall_df.to_excel(writer, sheet_name='Categorical_Tests', index=False)
    categorical_posthoc_df.to_excel(writer, sheet_name='Categorical_Posthoc', index=False)
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
sns.set_theme(style='white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
file_path = 'your_data_path/your_data.xlsx'
output_dir = 'your_output_path'
os.makedirs(output_dir, exist_ok=True)
OUTCOME_COL = 'Type.of.fracture.line'
RUN_FULL_PAIRPLOT = True
RUN_CORE_PAIRPLOT = True
CORE_PAIRPLOT_VARS = ['LLBCE', 'PMBT', 'MRT', 'Depth.of.A', 'RAPL', 'ART']
FEATURE_LABELS = {'LLBCE': 'LLBCE', 'PMBT': 'PMBT', 'MRT': 'MRT', 'Depth.of.A': 'Depth of A', 'RAPL': 'RAPL', 'ART': 'ART', 'RH': 'RH', 'LSND': 'LSND', 'age': 'Age', 'sex_male': 'Male sex', 'jaw_deformity_type3': 'III deformity', 'third_molar_yes': 'Third molar'}

def beautify_labels(labels):
    return [FEATURE_LABELS.get(x, x) for x in labels]
df = pd.read_excel(file_path)
df.columns = df.columns.astype(str).str.replace('\xa0', ' ', regex=False).str.replace('\t', ' ', regex=False).str.strip()
pass
pass
duplicate_cols = df.columns[df.columns.duplicated()].tolist()
pass
pass
if len(duplicate_cols) > 0:
    raise ValueError(f'， Excel ：{duplicate_cols}')
continuous_vars = ['LLBCE', 'PMBT', 'MRT', 'Depth.of.A', 'RAPL', 'ART', 'RH', 'LSND', 'age']
categorical_vars = ['sex', 'type of jaw deformity', 'third molar presence']
all_vars = continuous_vars + categorical_vars
required_cols = all_vars.copy()
if OUTCOME_COL in df.columns:
    required_cols.append(OUTCOME_COL)
else:
    pass
missing_cols = [col for col in all_vars if col not in df.columns]
if len(missing_cols) > 0:
    pass
    pass
    raise ValueError(f'Excel ，：{missing_cols}')
df_analysis = df[required_cols].copy()
for col in continuous_vars:
    df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')

def clean_string_series(s):
    return s.astype(str).str.replace('\xa0', ' ', regex=False).str.replace('\t', ' ', regex=False).str.strip().str.lower()
df_analysis['sex'] = clean_string_series(df_analysis['sex'])
df_analysis['type of jaw deformity'] = clean_string_series(df_analysis['type of jaw deformity'])
df_analysis['third molar presence'] = clean_string_series(df_analysis['third molar presence'])
if OUTCOME_COL in df_analysis.columns:
    df_analysis[OUTCOME_COL] = df_analysis[OUTCOME_COL].astype(str).str.strip()
missing_summary = df_analysis.isna().sum().reset_index()
missing_summary.columns = ['variable', 'missing_count']
missing_summary['missing_rate'] = missing_summary['missing_count'] / len(df_analysis)
pass
pass
missing_summary.to_excel(os.path.join(output_dir, 'missing_summary.xlsx'), index=False)

def spearman_corr_pvalue(data, variables):
    corr_matrix = pd.DataFrame(np.nan, index=variables, columns=variables, dtype=float)
    p_matrix = pd.DataFrame(np.nan, index=variables, columns=variables, dtype=float)
    for i in variables:
        for j in variables:
            if i == j:
                corr_matrix.loc[i, j] = 1.0
                p_matrix.loc[i, j] = 0.0
                continue
            x = pd.to_numeric(data[i], errors='coerce')
            y = pd.to_numeric(data[j], errors='coerce')
            temp = pd.DataFrame({'x': x, 'y': y}).dropna()
            if len(temp) < 3:
                corr_matrix.loc[i, j] = np.nan
                p_matrix.loc[i, j] = np.nan
                continue
            if temp['x'].nunique() <= 1 or temp['y'].nunique() <= 1:
                corr_matrix.loc[i, j] = np.nan
                p_matrix.loc[i, j] = np.nan
                continue
            r, p = spearmanr(temp['x'].values, temp['y'].values)
            corr_matrix.loc[i, j] = float(r)
            p_matrix.loc[i, j] = float(p)
    return (corr_matrix, p_matrix)

def plot_advanced_corr_ellipse(corr_df, title, save_path, p_df=None, show_significance=False, figsize=None):
    corr_df = corr_df.copy()
    variables = corr_df.columns.tolist()
    labels = beautify_labels(variables)
    n = len(variables)
    if figsize is None:
        figsize = (max(8, 0.75 * n), max(7, 0.75 * n))
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect('equal')
    for i in range(n):
        for j in range(n):
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='white', edgecolor='#E6E6E6', linewidth=0.8)
            ax.add_patch(rect)
    for i in range(n):
        for j in range(n):
            value = corr_df.iloc[i, j]
            if pd.isna(value):
                continue
            if i == j:
                ax.text(j, i, labels[i], ha='center', va='center', fontsize=10, fontweight='bold', color='black')
            elif i > j:
                r = float(value)
                abs_r = abs(r)
                major_axis = 0.86
                minor_axis = 0.86 * np.sqrt(max(0.0, 1.0 - abs_r))
                angle = 45 if r >= 0 else -45
                ellipse = Ellipse(xy=(j, i), width=major_axis, height=minor_axis, angle=angle, facecolor=cmap(norm(r)), edgecolor='black', linewidth=0.4, alpha=0.9)
                ax.add_patch(ellipse)
            else:
                sig = ''
                if show_significance and p_df is not None:
                    p_value = p_df.iloc[i, j]
                    if pd.notna(p_value):
                        if p_value < 0.001:
                            sig = '***'
                        elif p_value < 0.01:
                            sig = '**'
                        elif p_value < 0.05:
                            sig = '*'
                ax.text(j, i, f'{value:.2f}{sig}', ha='center', va='center', fontsize=10, color='black')
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Spearman correlation coefficient', fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax.set_title(title, fontsize=16, pad=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

def plot_pairplot(data, variables, save_path, outcome_col=None, title='Pairwise Distribution of Continuous Predictors'):
    pair_vars = [v for v in variables if v in data.columns]
    if len(pair_vars) < 2:
        pass
        return
    pair_df = data[pair_vars].copy()
    for col in pair_vars:
        pair_df[col] = pd.to_numeric(pair_df[col], errors='coerce')
    rename_dict = {v: FEATURE_LABELS.get(v, v) for v in pair_vars}
    pair_df = pair_df.rename(columns=rename_dict)
    plot_vars = list(rename_dict.values())
    hue_col = None
    if outcome_col is not None and outcome_col in data.columns:
        pair_df[outcome_col] = data[outcome_col].astype(str)
        hue_col = outcome_col
    pair_df = pair_df.dropna()
    if len(pair_df) < 5:
        pass
        return
    sns.set_theme(style='white')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    if hue_col is not None:
        g = sns.pairplot(pair_df, vars=plot_vars, hue=hue_col, corner=True, diag_kind='hist', plot_kws={'alpha': 0.75, 's': 32, 'edgecolor': 'white', 'linewidth': 0.3}, diag_kws={'alpha': 0.65, 'edgecolor': 'black', 'linewidth': 0.4})
    else:
        g = sns.pairplot(pair_df, vars=plot_vars, corner=True, diag_kind='hist', plot_kws={'alpha': 0.75, 's': 32, 'edgecolor': 'white', 'linewidth': 0.3}, diag_kws={'alpha': 0.65, 'edgecolor': 'black', 'linewidth': 0.4})
    g.fig.suptitle(title, y=1.02, fontsize=16, fontfamily='Times New Roman')
    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(axis='both', labelsize=9)
            ax.set_xlabel(ax.get_xlabel(), fontsize=10, fontfamily='Times New Roman')
            ax.set_ylabel(ax.get_ylabel(), fontsize=10, fontfamily='Times New Roman')
    if hasattr(g, '_legend') and g._legend is not None:
        g._legend.set_title('Fracture line type')
        for text in g._legend.texts:
            text.set_fontfamily('Times New Roman')
            text.set_fontsize(10)
        g._legend.get_title().set_fontfamily('Times New Roman')
        g._legend.get_title().set_fontsize(10)
    g.savefig(save_path, dpi=600, bbox_inches='tight')
    g.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
continuous_corr, continuous_p = spearman_corr_pvalue(df_analysis, continuous_vars)
continuous_corr.to_excel(os.path.join(output_dir, 'continuous_spearman_correlation_matrix.xlsx'))
continuous_p.to_excel(os.path.join(output_dir, 'continuous_spearman_pvalue_matrix.xlsx'))
pass
pass
pass
pass
plot_advanced_corr_ellipse(corr_df=continuous_corr, p_df=continuous_p, title='Spearman Correlation Matrix of Continuous Variables', save_path=os.path.join(output_dir, 'advanced_continuous_spearman_ellipse_corrplot.png'), show_significance=False, figsize=(9, 8))
if RUN_FULL_PAIRPLOT:
    plot_pairplot(data=df_analysis, variables=continuous_vars, outcome_col=OUTCOME_COL if OUTCOME_COL in df_analysis.columns else None, title='Pairwise Distribution of Continuous Candidate Predictors', save_path=os.path.join(output_dir, 'pairplot_all_continuous_predictors.png'))
if RUN_CORE_PAIRPLOT:
    plot_pairplot(data=df_analysis, variables=CORE_PAIRPLOT_VARS, outcome_col=OUTCOME_COL if OUTCOME_COL in df_analysis.columns else None, title='Pairwise Distribution of Core Candidate Predictors', save_path=os.path.join(output_dir, 'pairplot_core_candidate_predictors.png'))
df_encoded = df_analysis.copy()
sex_map = {'female': 0, 'male': 1, 'f': 0, 'm': 1, '': 0, '': 1}
jaw_map = {'2': 0, '2.0': 0, 'type 2': 0, 'type2': 0, '3': 1, '3.0': 1, 'type 3': 1, 'type3': 1}
third_molar_map = {'no': 0, 'yes': 1, 'n': 0, 'y': 1, '': 0, '': 1, 'absent': 0, 'present': 1}
df_encoded['sex_male'] = df_encoded['sex'].map(sex_map)
df_encoded['jaw_deformity_type3'] = df_encoded['type of jaw deformity'].map(jaw_map)
df_encoded['third_molar_yes'] = df_encoded['third molar presence'].map(third_molar_map)
encoding_check = df_encoded[['sex', 'sex_male', 'type of jaw deformity', 'jaw_deformity_type3', 'third molar presence', 'third_molar_yes']].copy()
pass
pass
encoding_check.to_excel(os.path.join(output_dir, 'categorical_encoding_check.xlsx'), index=False)
encoding_missing = {'sex_male_missing': int(df_encoded['sex_male'].isna().sum()), 'jaw_deformity_type3_missing': int(df_encoded['jaw_deformity_type3'].isna().sum()), 'third_molar_yes_missing': int(df_encoded['third_molar_yes'].isna().sum())}
pass
pass
if any((v > 0 for v in encoding_missing.values())):
    pass
    pass
    pass
    pass
encoded_vars = continuous_vars + ['sex_male', 'jaw_deformity_type3', 'third_molar_yes']
mixed_corr, mixed_p = spearman_corr_pvalue(df_encoded, encoded_vars)
mixed_corr.to_excel(os.path.join(output_dir, 'mixed_spearman_correlation_matrix.xlsx'))
mixed_p.to_excel(os.path.join(output_dir, 'mixed_spearman_pvalue_matrix.xlsx'))
pass
pass
pass
pass
plot_advanced_corr_ellipse(corr_df=mixed_corr, p_df=mixed_p, title='Spearman Correlation Matrix of Candidate Predictors', save_path=os.path.join(output_dir, 'advanced_mixed_spearman_ellipse_corrplot.png'), show_significance=False, figsize=(11, 10))

def get_high_corr_pairs(corr_df, threshold=0.7):
    pairs = []
    cols = corr_df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_df.loc[cols[i], cols[j]]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append({'variable_1': cols[i], 'variable_2': cols[j], 'spearman_r': float(r), 'abs_r': float(abs(r))})
    result = pd.DataFrame(pairs)
    if len(result) > 0:
        result = result.sort_values('abs_r', ascending=False)
    return result
high_corr_pairs_070 = get_high_corr_pairs(mixed_corr, threshold=0.7)
high_corr_pairs_085 = get_high_corr_pairs(mixed_corr, threshold=0.85)
high_corr_pairs_070.to_excel(os.path.join(output_dir, 'high_correlation_pairs_abs_r_ge_0.70.xlsx'), index=False)
high_corr_pairs_085.to_excel(os.path.join(output_dir, 'high_correlation_pairs_abs_r_ge_0.85.xlsx'), index=False)
pass
pass
pass
pass

def calculate_vif(data, variables):
    vif_data = data[variables].copy()
    for col in variables:
        vif_data[col] = pd.to_numeric(vif_data[col], errors='coerce')
    before_n = len(vif_data)
    vif_data = vif_data.dropna()
    after_n = len(vif_data)
    pass
    if after_n < 5:
        raise ValueError('VIF ，。')
    non_constant_vars = []
    constant_vars = []
    for col in variables:
        if vif_data[col].nunique() > 1:
            non_constant_vars.append(col)
        else:
            constant_vars.append(col)
    if len(constant_vars) > 0:
        pass
        pass
    vif_data = vif_data[non_constant_vars].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(vif_data)
    vif_results = []
    for i, var in enumerate(non_constant_vars):
        try:
            vif_value = variance_inflation_factor(X_scaled, i)
        except Exception as e:
            pass
            vif_value = np.nan
        vif_results.append({'variable': var, 'VIF': vif_value})
    vif_df = pd.DataFrame(vif_results)
    vif_df = vif_df.sort_values('VIF', ascending=False).reset_index(drop=True)
    return vif_df
vif_results = calculate_vif(df_encoded, encoded_vars)
pass
pass

def classify_vif(vif):
    if pd.isna(vif):
        return 'VIF calculation failed'
    elif vif < 3:
        return 'Low collinearity'
    elif vif < 5:
        return 'Acceptable / mild collinearity'
    elif vif < 10:
        return 'Moderate collinearity; consider checking redundancy'
    else:
        return 'Severe collinearity; consider removing or combining'
vif_results['interpretation'] = vif_results['VIF'].apply(classify_vif)
vif_results.to_excel(os.path.join(output_dir, 'vif_results_with_interpretation.xlsx'), index=False)

def plot_advanced_vif_lollipop(vif_df, save_path):
    plot_df = vif_df.copy()
    plot_df = plot_df.dropna(subset=['VIF'])
    plot_df = plot_df.sort_values('VIF', ascending=True)
    plot_df['label'] = plot_df['variable'].map(FEATURE_LABELS).fillna(plot_df['variable'])
    max_vif = float(plot_df['VIF'].max())
    x_max = max(10.5, max_vif * 1.25)
    fig_height = max(5.5, 0.45 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    y_pos = np.arange(len(plot_df))
    ax.axvspan(0, 3, color='#E8F5E9', alpha=0.65, zorder=0)
    ax.axvspan(3, 5, color='#FFF8E1', alpha=0.7, zorder=0)
    ax.axvspan(5, 10, color='#FFEBEE', alpha=0.7, zorder=0)
    ax.axvline(3, linestyle='--', linewidth=1.2, color='#777777')
    ax.axvline(5, linestyle='--', linewidth=1.2, color='#777777')
    ax.axvline(10, linestyle='--', linewidth=1.2, color='#777777')
    colors = []
    for v in plot_df['VIF']:
        if v < 3:
            colors.append('#2E86AB')
        elif v < 5:
            colors.append('#F4A261')
        else:
            colors.append('#D62828')
    ax.hlines(y=y_pos, xmin=0, xmax=plot_df['VIF'], color='#A7A7A7', linewidth=1.6, zorder=1)
    ax.scatter(plot_df['VIF'], y_pos, s=95, color=colors, edgecolor='black', linewidth=0.5, zorder=2)
    for i, v in enumerate(plot_df['VIF']):
        ax.text(v + 0.08, i, f'{v:.2f}', va='center', ha='left', fontsize=10, color='black')
    ax.text(1.5, len(plot_df) - 0.25, 'Low', ha='center', va='top', fontsize=10, color='#2E7D32')
    ax.text(4.0, len(plot_df) - 0.25, 'Mild', ha='center', va='top', fontsize=10, color='#B26A00')
    ax.text(7.5, len(plot_df) - 0.25, 'Moderate', ha='center', va='top', fontsize=10, color='#B71C1C')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['label'])
    ax.set_xlim(0, x_max)
    ax.set_xlabel('Variance Inflation Factor (VIF)')
    ax.set_ylabel('')
    ax.set_title('VIF Analysis of Candidate Predictors', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle=':', linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
plot_advanced_vif_lollipop(vif_results, save_path=os.path.join(output_dir, 'advanced_vif_lollipop.png'))
summary_excel_path = os.path.join(output_dir, 'correlation_vif_pairplot_summary.xlsx')
with pd.ExcelWriter(summary_excel_path) as writer:
    missing_summary.to_excel(writer, sheet_name='Missing', index=False)
    continuous_corr.to_excel(writer, sheet_name='Cont_Spearman_R')
    continuous_p.to_excel(writer, sheet_name='Cont_Spearman_P')
    mixed_corr.to_excel(writer, sheet_name='Mixed_Spearman_R')
    mixed_p.to_excel(writer, sheet_name='Mixed_Spearman_P')
    high_corr_pairs_070.to_excel(writer, sheet_name='HighCorr_0.70', index=False)
    high_corr_pairs_085.to_excel(writer, sheet_name='HighCorr_0.85', index=False)
    vif_results.to_excel(writer, sheet_name='VIF', index=False)
    encoding_check.to_excel(writer, sheet_name='Encoding_Check', index=False)
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
file_path = 'your_data_path/your_data.xlsx'
output_dir = 'your_output_path'
os.makedirs(output_dir, exist_ok=True)
OUTCOME_COL = 'Type.of.fracture.line'
all_continuous_vars = ['LLBCE', 'PMBT', 'MRT', 'Depth.of.A', 'RAPL', 'ART', 'RH', 'LSND', 'age']
core_vars = ['LLBCE', 'PMBT', 'MRT', 'Depth.of.A', 'RAPL', 'ART']
FEATURE_LABELS = {'LLBCE': 'LLBCE', 'PMBT': 'PMBT', 'MRT': 'MRT', 'Depth.of.A': 'Depth of A', 'RAPL': 'RAPL', 'ART': 'ART', 'RH': 'RH', 'LSND': 'LSND', 'age': 'Age'}

def clean_column_names(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace('\xa0', ' ', regex=False).str.replace('\t', ' ', regex=False).str.strip()
    return df

def sort_levels(levels):

    def key_func(x):
        try:
            return float(x)
        except Exception:
            return str(x)
    return sorted(levels, key=key_func)

def format_p_value(p):
    if pd.isna(p):
        return 'NA'
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'

def significance_stars(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def corr_bg_color(r):
    if pd.isna(r):
        return (1, 1, 1, 1)
    cmap = plt.cm.RdBu_r
    rgba = cmap((r + 1) / 2.0)
    return (rgba[0], rgba[1], rgba[2], 0.18)

def get_label(var):
    return FEATURE_LABELS.get(var, var)

def get_type_label(x):
    return f'Type {x}'

def safe_numeric(s):
    return pd.to_numeric(s, errors='coerce')

def draw_diag_distribution(ax, data, var, outcome_col, palette_dict):
    x_all = safe_numeric(data[var]).dropna()
    if len(x_all) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=9)
        return
    outcome_levels = sort_levels(data[outcome_col].dropna().unique())
    for level in outcome_levels:
        sub = data[data[outcome_col] == level]
        x = safe_numeric(sub[var]).dropna()
        if len(x) == 0:
            continue
        color = palette_dict[level]
        try:
            ax.hist(x, bins=12, density=True, alpha=0.25, color=color, edgecolor='black', linewidth=0.4)
        except Exception:
            pass
        if len(x) >= 3 and x.nunique() > 1:
            try:
                sns.kdeplot(x=x, ax=ax, color=color, linewidth=1.4, fill=False, warn_singular=False)
            except Exception:
                pass
    ax.set_ylabel('')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def draw_lower_scatter(ax, data, x_var, y_var, outcome_col, palette_dict):
    outcome_levels = sort_levels(data[outcome_col].dropna().unique())
    for level in outcome_levels:
        sub = data[data[outcome_col] == level].copy()
        x = safe_numeric(sub[x_var])
        y = safe_numeric(sub[y_var])
        temp = pd.DataFrame({'x': x, 'y': y}).dropna()
        if len(temp) == 0:
            continue
        ax.scatter(temp['x'], temp['y'], s=24, alpha=0.75, color=palette_dict[level], edgecolor='white', linewidth=0.35)
    temp_all = pd.DataFrame({'x': safe_numeric(data[x_var]), 'y': safe_numeric(data[y_var])}).dropna()
    if len(temp_all) >= 3 and temp_all['x'].nunique() > 1 and (temp_all['y'].nunique() > 1):
        try:
            coef = np.polyfit(temp_all['x'], temp_all['y'], 1)
            x_line = np.linspace(temp_all['x'].min(), temp_all['x'].max(), 100)
            y_line = coef[0] * x_line + coef[1]
            ax.plot(x_line, y_line, color='black', linewidth=1.2, linestyle='-', alpha=0.9)
        except Exception:
            pass
    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def draw_upper_corr_panel(ax, data, x_var, y_var):
    temp = pd.DataFrame({'x': safe_numeric(data[x_var]), 'y': safe_numeric(data[y_var])}).dropna()
    if len(temp) >= 3 and temp['x'].nunique() > 1 and (temp['y'].nunique() > 1):
        r, p = spearmanr(temp['x'], temp['y'])
        n_val = len(temp)
    else:
        r, p, n_val = (np.nan, np.nan, len(temp))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(corr_bg_color(r))
    stars = significance_stars(p)
    if pd.isna(r):
        rho_text = 'rho = NA'
        abs_r = 0.0
    else:
        rho_text = f'ρ = {r:.2f}{stars}'
        abs_r = abs(r)
    p_text = f'P = {format_p_value(p)}'
    n_text = f'n = {n_val}'
    ax.text(0.5, 0.72, rho_text, ha='center', va='center', fontsize=10, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.5, p_text, ha='center', va='center', fontsize=9, transform=ax.transAxes)
    ax.text(0.5, 0.31, n_text, ha='center', va='center', fontsize=9, transform=ax.transAxes)
    ax.add_patch(Rectangle((0.14, 0.09), 0.72, 0.075, transform=ax.transAxes, facecolor='#F2F2F2', edgecolor='#BBBBBB', linewidth=0.5))
    if not pd.isna(r):
        color = plt.cm.RdBu_r((r + 1) / 2.0)
        ax.add_patch(Rectangle((0.14, 0.09), 0.72 * abs_r, 0.075, transform=ax.transAxes, facecolor=color, edgecolor='none'))
    ax.text(0.5, 0.02, '|ρ| strength', ha='center', va='bottom', fontsize=7.5, color='dimgray', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor('#C0C0C0')

def plot_publication_pairplot(data, variables, outcome_col, save_path, title, cell_size=2.0):
    use_vars = [v for v in variables if v in data.columns]
    if len(use_vars) < 2:
        pass
        return
    plot_df = data[use_vars + [outcome_col]].copy()
    for v in use_vars:
        plot_df[v] = safe_numeric(plot_df[v])
    plot_df[outcome_col] = plot_df[outcome_col].astype(str).str.strip()
    plot_df = plot_df.dropna(subset=use_vars, how='all')
    if len(plot_df) < 5:
        pass
        return
    outcome_levels = sort_levels(plot_df[outcome_col].dropna().unique())
    palette = sns.color_palette('Set2', n_colors=len(outcome_levels))
    palette_dict = {level: palette[i] for i, level in enumerate(outcome_levels)}
    n = len(use_vars)
    fig_width = cell_size * n + 1.8
    fig_height = cell_size * n + 0.8
    fig, axes = plt.subplots(n, n, figsize=(fig_width, fig_height), squeeze=False)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            y_var = use_vars[i]
            x_var = use_vars[j]
            if i == j:
                draw_diag_distribution(ax=ax, data=plot_df, var=x_var, outcome_col=outcome_col, palette_dict=palette_dict)
            elif i > j:
                draw_lower_scatter(ax=ax, data=plot_df, x_var=x_var, y_var=y_var, outcome_col=outcome_col, palette_dict=palette_dict)
            else:
                draw_upper_corr_panel(ax=ax, data=plot_df, x_var=x_var, y_var=y_var)
            if i == n - 1:
                ax.set_xlabel(get_label(x_var), fontsize=9)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(get_label(y_var), fontsize=9)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            if i == n - 1:
                for label in ax.get_xticklabels():
                    label.set_rotation(0)
                    label.set_fontsize(8)
            if j == 0:
                for label in ax.get_yticklabels():
                    label.set_fontsize(8)
    legend_handles = []
    for level in outcome_levels:
        legend_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=palette_dict[level], markeredgecolor='white', markeredgewidth=0.6, markersize=7, label=get_type_label(level)))
    fig.legend(handles=legend_handles, title='Fracture line type', loc='center right', bbox_to_anchor=(0.985, 0.5), frameon=False, prop={'family': 'Times New Roman', 'size': 9}, title_fontproperties={'family': 'Times New Roman', 'size': 10, 'weight': 'bold'})
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    plt.subplots_adjust(left=0.06, right=0.88, bottom=0.06, top=0.95, wspace=0.08, hspace=0.08)
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
df = pd.read_excel(file_path)
df = clean_column_names(df)
pass
pass
required_cols = [OUTCOME_COL] + all_continuous_vars
missing_cols = [c for c in required_cols if c not in df.columns]
if len(missing_cols) > 0:
    raise ValueError(f'Excel ：{missing_cols}：{df.columns.tolist()}')
data = df[required_cols].copy()
for col in all_continuous_vars:
    data[col] = safe_numeric(data[col])
data[OUTCOME_COL] = data[OUTCOME_COL].astype(str).str.strip()
pass
pass
plot_publication_pairplot(data=data, variables=all_continuous_vars, outcome_col=OUTCOME_COL, save_path=os.path.join(output_dir, 'publication_pairplot_all_continuous.png'), title='Pairwise Relationships among Continuous Candidate Predictors', cell_size=1.85)
plot_publication_pairplot(data=data, variables=core_vars, outcome_col=OUTCOME_COL, save_path=os.path.join(output_dir, 'publication_pairplot_core_variables.png'), title='Pairwise Relationships among Core Candidate Predictors', cell_size=2.15)
pass
pass
pass
pass
pass
pass
pass
pass
pass

import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
DATA_FILE = 'your_external_validation_data.xlsx'
MODEL_PATH = 'result/best_model.pkl'
SCALER_PATH = 'result/scaler.pkl'
LABEL_ENCODER_PATH = 'result/label_encoder.pkl'
OUTPUT_DIR = 'result/best_model_reverse_hierarchical'
os.makedirs(OUTPUT_DIR, exist_ok=True)
FEATURES = ['LLBCE', 'PMBT', 'MRT', 'Depth  of A']
TARGET_COL = 'Type of fracture line'
MRT_INPUT = 9.5
PMBT_INPUT = 3.2
PATIENT_ID = 'example_patient'
GRID_SIZE = 150
NEAR_OPTIMAL_RATIO = 0.9
LOCAL_DEPTH_TOL = 0.1
LOCAL_LLBCE_TOL = 0.25
DRAW_REFERENCE_THRESHOLD = True
REFERENCE_THRESHOLD = 0.6
POINT_LABEL = 'Recommended stable points'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
SEARCH_LEVELS = [{'level': 1, 'name': 'IQR strict', 'low_q': 0.25, 'high_q': 0.75, 'require_top_class': True, 'near_optimal_ratio': NEAR_OPTIMAL_RATIO, 'grade': 'Primary recommendation'}, {'level': 2, 'name': '10th–90th percentile strict', 'low_q': 0.1, 'high_q': 0.9, 'require_top_class': True, 'near_optimal_ratio': NEAR_OPTIMAL_RATIO, 'grade': 'Expanded empirical recommendation'}, {'level': 3, 'name': '10th–90th percentile relaxed', 'low_q': 0.1, 'high_q': 0.9, 'require_top_class': False, 'near_optimal_ratio': NEAR_OPTIMAL_RATIO, 'grade': 'Probability-guided exploratory recommendation'}]

def normalize_name(name):
    return str(name).replace('\xa0', ' ').replace('\t', ' ').strip().lower().replace(' ', '').replace('_', '').replace('.', '')

def find_column(df, target_name):
    norm_map = {normalize_name(c): c for c in df.columns}
    key = normalize_name(target_name)
    if key in norm_map:
        return norm_map[key]
    depth_keys = [normalize_name('Depth  of A'), normalize_name('Depth.of.A'), normalize_name('Depth of A'), normalize_name('Depth_of_A')]
    if normalize_name(target_name) in depth_keys:
        for c in df.columns:
            if normalize_name(c) in depth_keys:
                return c
    return None

def is_depth_feature(name):
    return normalize_name(name) in [normalize_name('Depth  of A'), normalize_name('Depth.of.A'), normalize_name('Depth of A'), normalize_name('Depth_of_A')]

def is_llbce_feature(name):
    return normalize_name(name) == normalize_name('LLBCE')

def is_pmbt_feature(name):
    return normalize_name(name) == normalize_name('PMBT')

def is_mrt_feature(name):
    return normalize_name(name) == normalize_name('MRT')

def safe_range_string(low, high):
    if pd.isna(low) or pd.isna(high):
        return 'NA'
    return f'{low:.3f}–{high:.3f}'

def connected_component_from_best(mask, best_pos):
    if not mask[best_pos]:
        return np.zeros_like(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    q = deque([best_pos])
    visited[best_pos] = True
    n_row, n_col = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < n_row and 0 <= cc < n_col:
                if mask[rr, cc] and (not visited[rr, cc]):
                    visited[rr, cc] = True
                    q.append((rr, cc))
    return visited

def align_proba_to_classes(model, proba, n_classes):
    proba = np.asarray(proba)
    if proba.ndim == 3:
        proba = np.squeeze(proba)
    if proba.ndim != 2:
        raise ValueError(f'predict_proba :{proba.shape}')
    if proba.shape[1] != n_classes:
        raise ValueError(f': proba.shape={proba.shape}, n_classes={n_classes}')
    if not hasattr(model, 'classes_'):
        return proba
    model_classes = np.asarray(model.classes_)
    try:
        model_classes_int = model_classes.astype(int)
        if list(model_classes_int) == list(range(n_classes)):
            return proba
        aligned = np.zeros((proba.shape[0], n_classes))
        for j, cls in enumerate(model_classes_int):
            if 0 <= cls < n_classes:
                aligned[:, cls] = proba[:, j]
        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        aligned = aligned / row_sums
        return aligned
    except Exception:
        return proba

def build_grid_for_domain(depth_low, depth_high, llbce_low, llbce_high):
    depth_values = np.linspace(depth_low, depth_high, GRID_SIZE)
    llbce_values = np.linspace(llbce_low, llbce_high, GRID_SIZE)
    Depth_grid, LLBCE_grid = np.meshgrid(depth_values, llbce_values)
    grid_df = pd.DataFrame(index=np.arange(Depth_grid.size))
    for f in FEATURES:
        if is_llbce_feature(f):
            grid_df[f] = LLBCE_grid.ravel()
        elif is_pmbt_feature(f):
            grid_df[f] = float(PMBT_INPUT)
        elif is_mrt_feature(f):
            grid_df[f] = float(MRT_INPUT)
        elif is_depth_feature(f):
            grid_df[f] = Depth_grid.ravel()
        else:
            raise ValueError(f':{f}')
    grid_df = grid_df[FEATURES]
    return (grid_df, Depth_grid, LLBCE_grid)

def predict_grid(grid_df, Depth_grid, LLBCE_grid):
    grid_scaled = scaler.transform(grid_df)
    grid_proba = best_model.predict_proba(grid_scaled)
    grid_proba = align_proba_to_classes(best_model, grid_proba, n_classes)
    prob_grids = [grid_proba[:, j].reshape(GRID_SIZE, GRID_SIZE) for j in range(n_classes)]
    top_class_grid = np.argmax(grid_proba, axis=1).reshape(GRID_SIZE, GRID_SIZE)
    return (grid_proba, prob_grids, top_class_grid)

def get_class_quantile_domain(df, cls_name, low_q, high_q):
    sub = df[df['_target_str'] == str(cls_name)].copy()
    if len(sub) < 3:
        raise ValueError(f'Type {cls_name}，。')
    llbce_low = float(sub[llbce_col].quantile(low_q))
    llbce_high = float(sub[llbce_col].quantile(high_q))
    depth_low = float(sub[depth_col].quantile(low_q))
    depth_high = float(sub[depth_col].quantile(high_q))
    return (depth_low, depth_high, llbce_low, llbce_high, len(sub))

def evaluate_search_level(cls_index, cls_name, level_cfg):
    depth_low, depth_high, llbce_low, llbce_high, n_sub = get_class_quantile_domain(df=df, cls_name=cls_name, low_q=level_cfg['low_q'], high_q=level_cfg['high_q'])
    grid_df, Depth_grid, LLBCE_grid = build_grid_for_domain(depth_low=depth_low, depth_high=depth_high, llbce_low=llbce_low, llbce_high=llbce_high)
    grid_proba, prob_grids, top_class_grid = predict_grid(grid_df=grid_df, Depth_grid=Depth_grid, LLBCE_grid=LLBCE_grid)
    P_target = prob_grids[cls_index]
    require_top = level_cfg['require_top_class']
    if require_top:
        valid_for_best = top_class_grid == cls_index
        if np.any(valid_for_best):
            P_search = P_target.copy()
            P_search[~valid_for_best] = -np.inf
            best_pos = np.unravel_index(np.argmax(P_search), P_search.shape)
        else:
            return {'success': False, 'reason': 'No grid point predicted target class as top class', 'level_cfg': level_cfg, 'n_sub': n_sub, 'depth_low': depth_low, 'depth_high': depth_high, 'llbce_low': llbce_low, 'llbce_high': llbce_high, 'grid_df': grid_df, 'Depth_grid': Depth_grid, 'LLBCE_grid': LLBCE_grid, 'grid_proba': grid_proba, 'prob_grids': prob_grids, 'top_class_grid': top_class_grid}
    else:
        best_pos = np.unravel_index(np.argmax(P_target), P_target.shape)
    optimal_depth = float(Depth_grid[best_pos])
    optimal_llbce = float(LLBCE_grid[best_pos])
    pmax = float(P_target[best_pos])
    near_optimal_cutoff = level_cfg['near_optimal_ratio'] * pmax
    near_mask = P_target >= near_optimal_cutoff
    if require_top:
        top_mask = top_class_grid == cls_index
    else:
        top_mask = np.ones_like(P_target, dtype=bool)
    stable_candidate_mask = near_mask & top_mask
    stable_connected_mask = connected_component_from_best(stable_candidate_mask, best_pos)
    success = np.any(stable_connected_mask)
    return {'success': success, 'reason': 'Stable connected region found' if success else 'No connected stable region', 'level_cfg': level_cfg, 'n_sub': n_sub, 'depth_low': depth_low, 'depth_high': depth_high, 'llbce_low': llbce_low, 'llbce_high': llbce_high, 'grid_df': grid_df, 'Depth_grid': Depth_grid, 'LLBCE_grid': LLBCE_grid, 'grid_proba': grid_proba, 'prob_grids': prob_grids, 'top_class_grid': top_class_grid, 'P_target': P_target, 'best_pos': best_pos, 'optimal_depth': optimal_depth, 'optimal_llbce': optimal_llbce, 'pmax': pmax, 'near_optimal_cutoff': near_optimal_cutoff, 'stable_candidate_mask': stable_candidate_mask, 'stable_connected_mask': stable_connected_mask}

def local_window_fallback(cls_index, cls_name):
    level_cfg = {'level': 4, 'name': 'Local tolerance fallback', 'low_q': 0.1, 'high_q': 0.9, 'require_top_class': False, 'near_optimal_ratio': np.nan, 'grade': 'Local tolerance exploratory recommendation'}
    depth_low, depth_high, llbce_low, llbce_high, n_sub = get_class_quantile_domain(df=df, cls_name=cls_name, low_q=0.1, high_q=0.9)
    grid_df, Depth_grid, LLBCE_grid = build_grid_for_domain(depth_low=depth_low, depth_high=depth_high, llbce_low=llbce_low, llbce_high=llbce_high)
    grid_proba, prob_grids, top_class_grid = predict_grid(grid_df=grid_df, Depth_grid=Depth_grid, LLBCE_grid=LLBCE_grid)
    P_target = prob_grids[cls_index]
    best_pos = np.unravel_index(np.argmax(P_target), P_target.shape)
    optimal_depth = float(Depth_grid[best_pos])
    optimal_llbce = float(LLBCE_grid[best_pos])
    pmax = float(P_target[best_pos])
    stable_connected_mask = (np.abs(Depth_grid - optimal_depth) <= LOCAL_DEPTH_TOL) & (np.abs(LLBCE_grid - optimal_llbce) <= LOCAL_LLBCE_TOL)
    return {'success': np.any(stable_connected_mask), 'reason': 'Using local tolerance window around optimal point', 'level_cfg': level_cfg, 'n_sub': n_sub, 'depth_low': depth_low, 'depth_high': depth_high, 'llbce_low': llbce_low, 'llbce_high': llbce_high, 'grid_df': grid_df, 'Depth_grid': Depth_grid, 'LLBCE_grid': LLBCE_grid, 'grid_proba': grid_proba, 'prob_grids': prob_grids, 'top_class_grid': top_class_grid, 'P_target': P_target, 'best_pos': best_pos, 'optimal_depth': optimal_depth, 'optimal_llbce': optimal_llbce, 'pmax': pmax, 'near_optimal_cutoff': np.nan, 'stable_candidate_mask': stable_connected_mask, 'stable_connected_mask': stable_connected_mask}

def summarize_selected_result(cls_index, cls_name, selected):
    class_label = f'Type {cls_name}'
    level_cfg = selected['level_cfg']
    Depth_grid = selected['Depth_grid']
    LLBCE_grid = selected['LLBCE_grid']
    prob_grids = selected['prob_grids']
    P_target = selected['P_target']
    best_pos = selected['best_pos']
    stable_mask = selected['stable_connected_mask']
    optimal_all_probs = {f'P(Type {class_names[j]}) at optimal point': float(prob_grids[j][best_pos]) for j in range(n_classes)}
    row = {'Patient ID': PATIENT_ID, 'Class': class_label, 'MRT': float(MRT_INPUT), 'PMBT': float(PMBT_INPUT), 'Selected search level': level_cfg['level'], 'Selected search name': level_cfg['name'], 'Recommendation grade': level_cfg['grade'], 'Fallback reason': selected['reason'], 'Empirical quantile range': f"{level_cfg['low_q']:.2f}–{level_cfg['high_q']:.2f}", 'Search Depth of A range': safe_range_string(selected['depth_low'], selected['depth_high']), 'Search LLBCE range': safe_range_string(selected['llbce_low'], selected['llbce_high']), 'Near-optimal ratio': level_cfg['near_optimal_ratio'], 'Near-optimal cutoff': selected['near_optimal_cutoff'], 'Require target as top class': level_cfg['require_top_class'], 'Optimal Depth of A': selected['optimal_depth'], 'Optimal LLBCE': selected['optimal_llbce'], 'Max probability': selected['pmax'], 'Stable region points, n': int(np.sum(stable_mask)), 'Stable region area, %': float(np.sum(stable_mask) / P_target.size * 100)}
    row.update(optimal_all_probs)
    if np.any(stable_mask):
        stable_depth = Depth_grid[stable_mask]
        stable_llbce = LLBCE_grid[stable_mask]
        stable_p = P_target[stable_mask]
        depth_mean = float(np.mean(stable_depth))
        depth_sd = float(np.std(stable_depth))
        depth_median = float(np.median(stable_depth))
        depth_q1 = float(np.quantile(stable_depth, 0.25))
        depth_q3 = float(np.quantile(stable_depth, 0.75))
        depth_min = float(np.min(stable_depth))
        depth_max = float(np.max(stable_depth))
        llbce_mean = float(np.mean(stable_llbce))
        llbce_sd = float(np.std(stable_llbce))
        llbce_median = float(np.median(stable_llbce))
        llbce_q1 = float(np.quantile(stable_llbce, 0.25))
        llbce_q3 = float(np.quantile(stable_llbce, 0.75))
        llbce_min = float(np.min(stable_llbce))
        llbce_max = float(np.max(stable_llbce))
        row.update({'Mean probability in stable region': float(np.mean(stable_p)), 'Min probability in stable region': float(np.min(stable_p)), 'Max probability in stable region': float(np.max(stable_p)), 'Depth of A mean': depth_mean, 'Depth of A SD': depth_sd, 'Depth of A median': depth_median, 'Depth of A Q1': depth_q1, 'Depth of A Q3': depth_q3, 'Depth of A min': depth_min, 'Depth of A max': depth_max, 'LLBCE mean': llbce_mean, 'LLBCE SD': llbce_sd, 'LLBCE median': llbce_median, 'LLBCE Q1': llbce_q1, 'LLBCE Q3': llbce_q3, 'LLBCE min': llbce_min, 'LLBCE max': llbce_max, 'Recommended Depth of A, Mean ± SD': f'{depth_mean:.3f} ± {depth_sd:.3f}', 'Recommended LLBCE, Mean ± SD': f'{llbce_mean:.3f} ± {llbce_sd:.3f}', 'Recommended Depth of A, Median [IQR]': f'{depth_median:.3f} [{depth_q1:.3f}–{depth_q3:.3f}]', 'Recommended LLBCE, Median [IQR]': f'{llbce_median:.3f} [{llbce_q1:.3f}–{llbce_q3:.3f}]', 'Recommended Depth of A range': f'{depth_min:.3f}–{depth_max:.3f}', 'Recommended LLBCE range': f'{llbce_min:.3f}–{llbce_max:.3f}'})
    else:
        row.update({'Mean probability in stable region': np.nan, 'Min probability in stable region': np.nan, 'Max probability in stable region': np.nan, 'Depth of A mean': np.nan, 'Depth of A SD': np.nan, 'Depth of A median': np.nan, 'Depth of A Q1': np.nan, 'Depth of A Q3': np.nan, 'Depth of A min': np.nan, 'Depth of A max': np.nan, 'LLBCE mean': np.nan, 'LLBCE SD': np.nan, 'LLBCE median': np.nan, 'LLBCE Q1': np.nan, 'LLBCE Q3': np.nan, 'LLBCE min': np.nan, 'LLBCE max': np.nan, 'Recommended Depth of A, Mean ± SD': 'No stable region', 'Recommended LLBCE, Mean ± SD': 'No stable region', 'Recommended Depth of A, Median [IQR]': 'No stable region', 'Recommended LLBCE, Median [IQR]': 'No stable region', 'Recommended Depth of A range': 'No stable region', 'Recommended LLBCE range': 'No stable region'})
    return row

def print_classwise_recommendation_summary(summary_df, grid_size, output_dir, patient_id='example_patient', point_label='Recommended stable points'):
    total_points = grid_size * grid_size
    lines = []
    lines.append('=' * 70)
    lines.append('Reverse planning recommendation summary')
    lines.append(f'Patient ID: {patient_id}')
    lines.append('=' * 70)
    lines.append('')
    for _, row in summary_df.iterrows():
        class_name = str(row['Class']).replace('Type ', '')
        point_n = int(row['Stable region points, n'])
        area_pct = float(row['Stable region area, %'])
        lines.append(f'Class {class_name}:')
        lines.append(f"  Selected search level: {row['Selected search level']} - {row['Selected search name']}")
        lines.append(f"  Recommendation grade: {row['Recommendation grade']}")
        lines.append(f"  Max probability: {row['Max probability']:.4f}")
        lines.append(f"  Optimal point: Depth of A = {row['Optimal Depth of A']:.3f}, LLBCE = {row['Optimal LLBCE']:.3f}")
        lines.append(f"  Search Depth of A range: {row['Search Depth of A range']}")
        lines.append(f"  Search LLBCE range: {row['Search LLBCE range']}")
        lines.append(f'  {point_label}: {point_n}/{total_points} ({area_pct:.2f}%)')
        if pd.notna(row.get('Depth of A min', np.nan)) and pd.notna(row.get('Depth of A max', np.nan)):
            lines.append(f"  Depth of A range: [{row['Depth of A min']:.3f}, {row['Depth of A max']:.3f}]")
        else:
            lines.append('  Depth of A range: No stable region')
        if pd.notna(row.get('LLBCE min', np.nan)) and pd.notna(row.get('LLBCE max', np.nan)):
            lines.append(f"  LLBCE range: [{row['LLBCE min']:.3f}, {row['LLBCE max']:.3f}]")
        else:
            lines.append('  LLBCE range: No stable region')
        if pd.notna(row.get('Depth of A mean', np.nan)) and pd.notna(row.get('Depth of A SD', np.nan)):
            lines.append(f"  Depth of A mean: {row['Depth of A mean']:.3f} ± {row['Depth of A SD']:.3f}")
        else:
            lines.append('  Depth of A mean: No stable region')
        if pd.notna(row.get('LLBCE mean', np.nan)) and pd.notna(row.get('LLBCE SD', np.nan)):
            lines.append(f"  LLBCE mean: {row['LLBCE mean']:.3f} ± {row['LLBCE SD']:.3f}")
        else:
            lines.append('  LLBCE mean: No stable region')
        if pd.notna(row.get('Depth of A median', np.nan)):
            lines.append(f"  Depth of A median [IQR]: {row['Depth of A median']:.3f} [{row['Depth of A Q1']:.3f}, {row['Depth of A Q3']:.3f}]")
        if pd.notna(row.get('LLBCE median', np.nan)):
            lines.append(f"  LLBCE median [IQR]: {row['LLBCE median']:.3f} [{row['LLBCE Q1']:.3f}, {row['LLBCE Q3']:.3f}]")
        lines.append('')
    lines.append('Analysis completed successfully!')
    lines.append(f"Results saved to '{output_dir}' folder")
    output_text = '\n'.join(lines)
    pass
    txt_path = os.path.join(output_dir, f'best_model_{patient_id}_classwise_recommendation_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(output_text)
    pass
if not os.path.exists(MODEL_PATH):
    pass
    candidates = glob.glob('result/*best*.pkl') + glob.glob('result/**/*best*.pkl', recursive=True)
    if len(candidates) > 0:
        pass
        for c in candidates:
            pass
    raise FileNotFoundError('MODEL_PATH best_model.pkl 。')
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f'scaler :{SCALER_PATH}')
if not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError(f'label encoder :{LABEL_ENCODER_PATH}')
best_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)
class_names = [str(x) for x in le.classes_]
n_classes = len(class_names)
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f':{DATA_FILE}')
df = pd.read_excel(DATA_FILE)
df.columns = df.columns.astype(str).str.strip()
llbce_col = find_column(df, 'LLBCE')
depth_col = find_column(df, 'Depth  of A')
target_col = find_column(df, TARGET_COL)
if llbce_col is None:
    raise ValueError('LLBCE 。')
if depth_col is None:
    raise ValueError('Depth of A 。')
if target_col is None:
    raise ValueError('Type of fracture line 。')
df[llbce_col] = pd.to_numeric(df[llbce_col], errors='coerce')
df[depth_col] = pd.to_numeric(df[depth_col], errors='coerce')
df = df.dropna(subset=[llbce_col, depth_col, target_col]).copy()
df['_target_str'] = df[target_col].astype(str).str.strip()
range_rows = []
for cls_name in class_names:
    sub = df[df['_target_str'] == str(cls_name)].copy()
    row = {'Class': f'Type {cls_name}', 'n': len(sub), 'LLBCE mean': float(sub[llbce_col].mean()), 'LLBCE median': float(sub[llbce_col].median()), 'LLBCE Q10': float(sub[llbce_col].quantile(0.1)), 'LLBCE Q25': float(sub[llbce_col].quantile(0.25)), 'LLBCE Q75': float(sub[llbce_col].quantile(0.75)), 'LLBCE Q90': float(sub[llbce_col].quantile(0.9)), 'LLBCE min': float(sub[llbce_col].min()), 'LLBCE max': float(sub[llbce_col].max()), 'Depth of A mean': float(sub[depth_col].mean()), 'Depth of A median': float(sub[depth_col].median()), 'Depth of A Q10': float(sub[depth_col].quantile(0.1)), 'Depth of A Q25': float(sub[depth_col].quantile(0.25)), 'Depth of A Q75': float(sub[depth_col].quantile(0.75)), 'Depth of A Q90': float(sub[depth_col].quantile(0.9)), 'Depth of A min': float(sub[depth_col].min()), 'Depth of A max': float(sub[depth_col].max())}
    range_rows.append(row)
range_df = pd.DataFrame(range_rows)
range_path = os.path.join(OUTPUT_DIR, 'class_specific_empirical_ranges.xlsx')
range_df.to_excel(range_path, index=False)
pass
pass
summary_rows = []
all_grid_predictions = {}
all_plot_objects = {}
search_log_rows = []
for k, cls_name in enumerate(class_names):
    selected = None
    search_attempts = []
    for level_cfg in SEARCH_LEVELS:
        result = evaluate_search_level(cls_index=k, cls_name=cls_name, level_cfg=level_cfg)
        attempt_row = {'Class': f'Type {cls_name}', 'Level': level_cfg['level'], 'Search name': level_cfg['name'], 'Require target as top class': level_cfg['require_top_class'], 'Low Q': level_cfg['low_q'], 'High Q': level_cfg['high_q'], 'Success': result['success'], 'Reason': result['reason'], 'Search Depth of A range': safe_range_string(result['depth_low'], result['depth_high']), 'Search LLBCE range': safe_range_string(result['llbce_low'], result['llbce_high'])}
        if 'pmax' in result:
            attempt_row['Max probability'] = result['pmax']
            attempt_row['Stable region points, n'] = int(np.sum(result['stable_connected_mask']))
            attempt_row['Stable region area, %'] = float(np.sum(result['stable_connected_mask']) / (GRID_SIZE * GRID_SIZE) * 100)
        else:
            attempt_row['Max probability'] = np.nan
            attempt_row['Stable region points, n'] = 0
            attempt_row['Stable region area, %'] = 0.0
        search_attempts.append(attempt_row)
        if result['success']:
            selected = result
            break
    if selected is None:
        selected = local_window_fallback(cls_index=k, cls_name=cls_name)
        search_attempts.append({'Class': f'Type {cls_name}', 'Level': 4, 'Search name': 'Local tolerance fallback', 'Require target as top class': False, 'Low Q': 0.1, 'High Q': 0.9, 'Success': selected['success'], 'Reason': selected['reason'], 'Search Depth of A range': safe_range_string(selected['depth_low'], selected['depth_high']), 'Search LLBCE range': safe_range_string(selected['llbce_low'], selected['llbce_high']), 'Max probability': selected['pmax'], 'Stable region points, n': int(np.sum(selected['stable_connected_mask'])), 'Stable region area, %': float(np.sum(selected['stable_connected_mask']) / (GRID_SIZE * GRID_SIZE) * 100)})
    search_log_rows.extend(search_attempts)
    row = summarize_selected_result(cls_index=k, cls_name=cls_name, selected=selected)
    summary_rows.append(row)
    class_label = f'Type {cls_name}'
    grid_prediction_df = selected['grid_df'].copy()
    grid_proba = selected['grid_proba']
    for j, c_name in enumerate(class_names):
        grid_prediction_df[f'P(Type {c_name})'] = grid_proba[:, j]
    grid_prediction_df['Target class'] = class_label
    grid_prediction_df['P(target)'] = grid_proba[:, k]
    grid_prediction_df['Predicted top class encoded'] = np.argmax(grid_proba, axis=1)
    grid_prediction_df['Predicted top class'] = [f'Type {class_names[idx]}' for idx in grid_prediction_df['Predicted top class encoded']]
    grid_prediction_df['Connected stable region'] = selected['stable_connected_mask'].ravel().astype(int)
    grid_prediction_df['Optimal point'] = 0
    best_flat_idx = np.ravel_multi_index(selected['best_pos'], selected['P_target'].shape)
    grid_prediction_df.loc[best_flat_idx, 'Optimal point'] = 1
    all_grid_predictions[class_label] = grid_prediction_df
    all_plot_objects[class_label] = selected
summary_df = pd.DataFrame(summary_rows)
search_log_df = pd.DataFrame(search_log_rows)
summary_path = os.path.join(OUTPUT_DIR, f'best_model_{PATIENT_ID}_hierarchical_recommended_values.xlsx')
grid_path = os.path.join(OUTPUT_DIR, f'best_model_{PATIENT_ID}_hierarchical_grid_predictions.xlsx')
with pd.ExcelWriter(summary_path) as writer:
    summary_df.to_excel(writer, sheet_name='Recommended_values', index=False)
    range_df.to_excel(writer, sheet_name='Empirical_ranges', index=False)
    search_log_df.to_excel(writer, sheet_name='Search_log', index=False)
with pd.ExcelWriter(grid_path) as writer:
    for class_label, gdf in all_grid_predictions.items():
        sheet_name = class_label.replace(' ', '_')[:31]
        gdf.to_excel(writer, sheet_name=sheet_name, index=False)
print_classwise_recommendation_summary(summary_df=summary_df, grid_size=GRID_SIZE, output_dir=OUTPUT_DIR, patient_id=PATIENT_ID, point_label=POINT_LABEL)
pass
pass
pass
pass
fig, axes = plt.subplots(1, n_classes, figsize=(5.8 * n_classes, 5.1), constrained_layout=True)
if n_classes == 1:
    axes = [axes]
levels = np.linspace(0, 1, 21)
for k, cls_name in enumerate(class_names):
    class_label = f'Type {cls_name}'
    obj = all_plot_objects[class_label]
    Depth_grid = obj['Depth_grid']
    LLBCE_grid = obj['LLBCE_grid']
    P_target = obj['P_target']
    stable_mask = obj['stable_connected_mask']
    best_pos = obj['best_pos']
    ax = axes[k]
    cf = ax.contourf(Depth_grid, LLBCE_grid, P_target, levels=levels, cmap='RdYlBu_r', alpha=0.95)
    if pd.notna(obj['near_optimal_cutoff']):
        try:
            cs = ax.contour(Depth_grid, LLBCE_grid, P_target, levels=[obj['near_optimal_cutoff']], colors='black', linewidths=1.8)
            ax.clabel(cs, inline=True, fontsize=8, fmt={obj['near_optimal_cutoff']: 'near-optimal'})
        except Exception:
            pass
    if DRAW_REFERENCE_THRESHOLD:
        try:
            cs_ref = ax.contour(Depth_grid, LLBCE_grid, P_target, levels=[REFERENCE_THRESHOLD], colors='gray', linestyles='--', linewidths=1.2)
            ax.clabel(cs_ref, inline=True, fontsize=8, fmt={REFERENCE_THRESHOLD: f'P={REFERENCE_THRESHOLD:.2f}'})
        except Exception:
            pass
    if np.any(stable_mask):
        ax.contour(Depth_grid, LLBCE_grid, stable_mask.astype(float), levels=[0.5], colors='red', linewidths=2.2)
        ax.scatter(Depth_grid[stable_mask], LLBCE_grid[stable_mask], s=7, color='red', alpha=0.18)
    ax.scatter(Depth_grid[best_pos], LLBCE_grid[best_pos], s=100, color='yellow', edgecolor='black', marker='*', zorder=5)
    ax.set_xlabel('Depth of A')
    ax.set_ylabel('LLBCE')
    ax.set_title(class_label, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(f'P({class_label})')
fig.suptitle(f'Hierarchical reverse probability maps | Patient: {PATIENT_ID}', fontsize=15, fontweight='bold', y=1.08)
contour_base = os.path.join(OUTPUT_DIR, f'best_model_{PATIENT_ID}_hierarchical_contour_maps')
plt.savefig(contour_base + '.png', dpi=600, bbox_inches='tight')
plt.savefig(contour_base + '.pdf', bbox_inches='tight')
plt.savefig(contour_base + '.svg', bbox_inches='tight')
plt.show()
plot_df = summary_df.copy()
x_labels = plot_df['Class'].tolist()
x = np.arange(len(x_labels))
depth_mean = pd.to_numeric(plot_df['Depth of A mean'], errors='coerce').values
depth_sd = pd.to_numeric(plot_df['Depth of A SD'], errors='coerce').values
llbce_mean = pd.to_numeric(plot_df['LLBCE mean'], errors='coerce').values
llbce_sd = pd.to_numeric(plot_df['LLBCE SD'], errors='coerce').values
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
ax1, ax2 = axes
ax1.errorbar(x, depth_mean, yerr=depth_sd, fmt='o', capsize=5, markersize=7, linewidth=1.8)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_ylabel('Depth of A')
ax1.set_title('Recommended Depth of A', fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.55)
ax2.errorbar(x, llbce_mean, yerr=llbce_sd, fmt='o', capsize=5, markersize=7, linewidth=1.8)
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels)
ax2.set_ylabel('LLBCE')
ax2.set_title('Recommended LLBCE', fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.55)
fig.suptitle(f'Recommended surgical parameter values, Mean ± SD | Patient: {PATIENT_ID}', fontsize=14, fontweight='bold', y=1.04)
plt.tight_layout()
values_base = os.path.join(OUTPUT_DIR, f'best_model_{PATIENT_ID}_recommended_values_Mean_SD')
plt.savefig(values_base + '.png', dpi=600, bbox_inches='tight')
plt.savefig(values_base + '.pdf', bbox_inches='tight')
plt.savefig(values_base + '.svg', bbox_inches='tight')
plt.show()
depth_median = pd.to_numeric(plot_df['Depth of A median'], errors='coerce').values
depth_q1 = pd.to_numeric(plot_df['Depth of A Q1'], errors='coerce').values
depth_q3 = pd.to_numeric(plot_df['Depth of A Q3'], errors='coerce').values
llbce_median = pd.to_numeric(plot_df['LLBCE median'], errors='coerce').values
llbce_q1 = pd.to_numeric(plot_df['LLBCE Q1'], errors='coerce').values
llbce_q3 = pd.to_numeric(plot_df['LLBCE Q3'], errors='coerce').values
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
ax1, ax2 = axes
ax1.errorbar(x, depth_median, yerr=[depth_median - depth_q1, depth_q3 - depth_median], fmt='o', capsize=5, markersize=7, linewidth=1.8)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_ylabel('Depth of A')
ax1.set_title('Recommended Depth of A, Median [IQR]', fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.55)
ax2.errorbar(x, llbce_median, yerr=[llbce_median - llbce_q1, llbce_q3 - llbce_median], fmt='o', capsize=5, markersize=7, linewidth=1.8)
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels)
ax2.set_ylabel('LLBCE')
ax2.set_title('Recommended LLBCE, Median [IQR]', fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.55)
fig.suptitle(f'Recommended surgical parameter values, Median [IQR] | Patient: {PATIENT_ID}', fontsize=14, fontweight='bold', y=1.04)
plt.tight_layout()
iqr_base = os.path.join(OUTPUT_DIR, f'best_model_{PATIENT_ID}_recommended_values_Median_IQR')
plt.savefig(iqr_base + '.png', dpi=600, bbox_inches='tight')
plt.savefig(iqr_base + '.pdf', bbox_inches='tight')
plt.savefig(iqr_base + '.svg', bbox_inches='tight')
plt.show()
pass
pass

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
OUT_DIR = Path('causal_enhanced_results')
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = OUT_DIR / 'improved_figures'
FIG_DIR.mkdir(exist_ok=True)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
COLOR_M1 = '#1f77b4'
COLOR_M4 = '#ff7f0e'
GEE_M1_NAME = 'Model 1: final 4-variable adjustment'
GEE_M4_NAME = 'Model 4: full DAG-informed adjustment'
GCOMP_M1_NAME = 'Final 4-variable outcome model'
GCOMP_M4_NAME = 'Full DAG-informed outcome model'

def load_gee_results_if_needed():
    if 'gee_results' in globals():
        pass
        return globals()['gee_results'].copy()
    gee_excel = OUT_DIR / 'extended_GEE_sensitivity_analysis.xlsx'
    if not gee_excel.exists():
        raise FileNotFoundError(f'gee_results，：{gee_excel}GEE，Excel。')
    pass
    try:
        df = pd.read_excel(gee_excel, sheet_name='GEE_results_full')
    except Exception:
        df = pd.read_excel(gee_excel, sheet_name=0)
    return df

def load_gcomp_results_if_needed():
    if 'gcomp_results' in globals():
        pass
        return globals()['gcomp_results'].copy()
    gcomp_excel = OUT_DIR / 'g_computation_counterfactual_probability_changes.xlsx'
    if not gcomp_excel.exists():
        raise FileNotFoundError(f'gcomp_results，：{gcomp_excel}g-computation，Excel。')
    pass
    try:
        df = pd.read_excel(gcomp_excel, sheet_name='Gcomp_results_full')
    except Exception:
        df = pd.read_excel(gcomp_excel, sheet_name=0)
    return df
gee_results_plot = load_gee_results_if_needed()
gcomp_results_plot = load_gcomp_results_if_needed()

def outcome_sort_key(x):
    try:
        return int(str(x))
    except Exception:
        return str(x)

def clean_outcome_label(x):
    sx = str(x)
    if sx.lower().startswith('type'):
        return sx
    return f'Type {sx}'

def pretty_exposure_name(x):
    sx = str(x)
    if sx in ['Depth.of.A', 'Depth_of_A', 'Depth  of A']:
        return 'Depth.of.A'
    if sx in ['LLBCE']:
        return 'LLBCE'
    return sx

def pretty_intervention_name(x):
    sx = str(x)
    sx = sx.replace('Depth_of_A', 'Depth.of.A')
    sx = sx.replace('Depth  of A', 'Depth.of.A')
    return sx

def safe_numeric(df, cols):
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out

def draw_dag_improved(output_path):
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    def draw_box(text, x, y, w=0.2, h=0.1, fc='white'):
        rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc, edgecolor='black', linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11)

    def arrow(x1, y1, x2, y2, rad=0.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', lw=1.7, color='black', connectionstyle=f'arc3,rad={rad}'))
    patient_x, patient_y = (0.13, 0.74)
    anatomy_x, anatomy_y = (0.4, 0.74)
    mod_x, mod_y = (0.63, 0.5)
    outcome_x, outcome_y = (0.86, 0.5)
    surgeon_x, surgeon_y = (0.3, 0.24)
    unmeasured_x, unmeasured_y = (0.63, 0.15)
    draw_box('Patient factors\n(age, sex,\njaw deformity,\nthird molar)', patient_x, patient_y, w=0.18, h=0.13, fc='#f7f7f7')
    draw_box('Baseline mandibular anatomy\n(MRT, PMBT,\nRAPL, ART, RH, LSND)', anatomy_x, anatomy_y, w=0.27, h=0.11, fc='#f7f7f7')
    draw_box('Modifiable osteotomy parameters\nLLBCE\nDepth of A', mod_x, mod_y, w=0.25, h=0.12, fc='#eef5ff')
    draw_box('Fracture-line type\n(Type 1 / Type 2 / Type 3)', outcome_x, outcome_y, w=0.2, h=0.1, fc='#fff5ee')
    draw_box('Surgeon / center /\ninstrument / experience', surgeon_x, surgeon_y, w=0.22, h=0.1, fc='#f7f7f7')
    draw_box('Unmeasured intraoperative factors\nbone quality, splitting force,\nchisel direction, stress release', unmeasured_x, unmeasured_y, w=0.32, h=0.1, fc='#f7f7f7')
    arrow(0.22, 0.74, 0.265, 0.74)
    arrow(0.19, 0.68, 0.52, 0.53, rad=-0.08)
    arrow(0.2, 0.8, 0.76, 0.56, rad=-0.18)
    arrow(0.48, 0.7, 0.55, 0.55)
    arrow(0.5, 0.77, 0.77, 0.55, rad=-0.08)
    arrow(0.755, 0.5, 0.76, 0.5)
    arrow(0.4, 0.27, 0.53, 0.45)
    arrow(0.4, 0.27, 0.77, 0.45, rad=0.12)
    arrow(0.63, 0.205, 0.63, 0.44)
    arrow(0.72, 0.19, 0.78, 0.45)
    ax.text(0.5, 0.96, 'Directed acyclic graph for exploratory causal framework', ha='center', va='center', fontsize=18, fontweight='bold')
    ax.text(0.5, 0.045, 'This DAG was constructed a priori based on clinical knowledge and mechanism assumptions. It defines the exposure, outcome, measured candidate confounders, and potential unmeasured confounders.', ha='center', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()
dag_improved_path = FIG_DIR / 'DAG_exploratory_causal_framework_improved.png'
draw_dag_improved(dag_improved_path)
pass

def plot_gee_m1_m4_dualcolor(gee_df, output_path):
    plot_df = gee_df.copy()
    required_cols = ['Outcome class', 'Exposure', 'Adjustment model', 'RR', 'CI lower', 'CI upper']
    missing = [c for c in required_cols if c not in plot_df.columns]
    if missing:
        raise ValueError('GEE：' + '\n'.join(missing) + '：' + '\n'.join(plot_df.columns.astype(str)))
    plot_df = safe_numeric(plot_df, ['RR', 'CI lower', 'CI upper'])
    plot_df = plot_df[plot_df['Adjustment model'].isin([GEE_M1_NAME, GEE_M4_NAME])].copy()
    plot_df['Exposure_clean'] = plot_df['Exposure'].apply(pretty_exposure_name)
    plot_df = plot_df[plot_df['Exposure_clean'].isin(['Depth.of.A', 'LLBCE'])].copy()
    plot_df = plot_df.dropna(subset=['RR', 'CI lower', 'CI upper'])
    if plot_df.empty:
        pass
        return
    outcome_order = sorted(plot_df['Outcome class'].unique(), key=outcome_sort_key)
    exposure_order = ['Depth.of.A', 'LLBCE']
    row_order = []
    for outcome in outcome_order:
        for exposure in exposure_order:
            row_order.append((outcome, exposure))
    row_df = pd.DataFrame({'Outcome class': [r[0] for r in row_order], 'Exposure_clean': [r[1] for r in row_order]})
    row_df['Label'] = row_df.apply(lambda r: f"{clean_outcome_label(r['Outcome class'])} | {r['Exposure_clean']}", axis=1)
    row_df['y_base'] = np.arange(len(row_df))[::-1]
    m1 = plot_df[plot_df['Adjustment model'] == GEE_M1_NAME][['Outcome class', 'Exposure_clean', 'RR', 'CI lower', 'CI upper']].rename(columns={'RR': 'RR_M1', 'CI lower': 'CI_low_M1', 'CI upper': 'CI_high_M1'})
    m4 = plot_df[plot_df['Adjustment model'] == GEE_M4_NAME][['Outcome class', 'Exposure_clean', 'RR', 'CI lower', 'CI upper']].rename(columns={'RR': 'RR_M4', 'CI lower': 'CI_low_M4', 'CI upper': 'CI_high_M4'})
    merged = row_df.merge(m1, on=['Outcome class', 'Exposure_clean'], how='left')
    merged = merged.merge(m4, on=['Outcome class', 'Exposure_clean'], how='left')
    fig_height = max(5.5, len(merged) * 0.62)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    offset = 0.13
    ax.errorbar(merged['RR_M1'], merged['y_base'] - offset, xerr=[merged['RR_M1'] - merged['CI_low_M1'], merged['CI_high_M1'] - merged['RR_M1']], fmt='o', color=COLOR_M1, ecolor=COLOR_M1, elinewidth=2, capsize=4, markersize=8, label='M1: final 4-variable adjustment')
    ax.errorbar(merged['RR_M4'], merged['y_base'] + offset, xerr=[merged['RR_M4'] - merged['CI_low_M4'], merged['CI_high_M4'] - merged['RR_M4']], fmt='o', color=COLOR_M4, ecolor=COLOR_M4, elinewidth=2, capsize=4, markersize=8, label='M4: full DAG-informed adjustment')
    ax.axvline(1, linestyle='--', color='black', linewidth=1.7)
    ax.set_yticks(merged['y_base'])
    ax.set_yticklabels(merged['Label'])
    ax.set_xlabel('Relative risk per 0.5 mm increase')
    ax.set_title('DAG-informed GEE Poisson sensitivity analysis')
    ax.grid(axis='x', alpha=0.25)
    ax.legend(frameon=False, loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()
gee_dualcolor_path = FIG_DIR / 'GEE_RR_forest_plot_M1_M4_dualcolor.png'
plot_gee_m1_m4_dualcolor(gee_results_plot, gee_dualcolor_path)
pass

def plot_gcomp_m1_m4_dualcolor(gcomp_df, output_path):
    plot_df = gcomp_df.copy()
    required_cols = ['G-computation model', 'Intervention', 'Outcome class', 'Mean probability difference', 'CI lower', 'CI upper']
    missing = [c for c in required_cols if c not in plot_df.columns]
    if missing:
        raise ValueError('g-computation：' + '\n'.join(missing) + '：' + '\n'.join(plot_df.columns.astype(str)))
    plot_df = safe_numeric(plot_df, ['Mean probability difference', 'CI lower', 'CI upper'])
    plot_df = plot_df[plot_df['G-computation model'].isin([GCOMP_M1_NAME, GCOMP_M4_NAME])].copy()
    if plot_df.empty:
        pass
        return
    plot_df['Mean_pp'] = plot_df['Mean probability difference'] * 100
    plot_df['CI_low_pp'] = plot_df['CI lower'] * 100
    plot_df['CI_high_pp'] = plot_df['CI upper'] * 100
    plot_df['Intervention_clean'] = plot_df['Intervention'].apply(pretty_intervention_name)
    intervention_order = ['LLBCE +0.5 mm', 'LLBCE -0.5 mm', 'Depth.of.A +0.5 mm', 'Depth.of.A -0.5 mm']
    existing_interventions = plot_df['Intervention_clean'].unique().tolist()
    intervention_order = [x for x in intervention_order if x in existing_interventions]
    if len(intervention_order) == 0:
        intervention_order = existing_interventions
    outcome_order = sorted(plot_df['Outcome class'].unique(), key=outcome_sort_key)
    rows = []
    for intervention in intervention_order:
        for outcome in outcome_order:
            rows.append({'Intervention_clean': intervention, 'Outcome class': outcome, 'Label': f'{intervention} | {clean_outcome_label(outcome)}'})
    row_df = pd.DataFrame(rows)
    row_df['y_base'] = np.arange(len(row_df))[::-1]
    m1 = plot_df[plot_df['G-computation model'] == GCOMP_M1_NAME][['Intervention_clean', 'Outcome class', 'Mean_pp', 'CI_low_pp', 'CI_high_pp']].rename(columns={'Mean_pp': 'Mean_M1', 'CI_low_pp': 'CI_low_M1', 'CI_high_pp': 'CI_high_M1'})
    m4 = plot_df[plot_df['G-computation model'] == GCOMP_M4_NAME][['Intervention_clean', 'Outcome class', 'Mean_pp', 'CI_low_pp', 'CI_high_pp']].rename(columns={'Mean_pp': 'Mean_M4', 'CI_low_pp': 'CI_low_M4', 'CI_high_pp': 'CI_high_M4'})
    merged = row_df.merge(m1, on=['Intervention_clean', 'Outcome class'], how='left')
    merged = merged.merge(m4, on=['Intervention_clean', 'Outcome class'], how='left')
    fig_height = max(7.0, len(merged) * 0.58)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    offset = 0.13
    ax.errorbar(merged['Mean_M1'], merged['y_base'] - offset, xerr=[merged['Mean_M1'] - merged['CI_low_M1'], merged['CI_high_M1'] - merged['Mean_M1']], fmt='o', color=COLOR_M1, ecolor=COLOR_M1, elinewidth=2, capsize=4, markersize=8, label='M1: final 4-variable outcome model')
    ax.errorbar(merged['Mean_M4'], merged['y_base'] + offset, xerr=[merged['Mean_M4'] - merged['CI_low_M4'], merged['CI_high_M4'] - merged['Mean_M4']], fmt='o', color=COLOR_M4, ecolor=COLOR_M4, elinewidth=2, capsize=4, markersize=8, label='M4: full DAG-informed outcome model')
    ax.axvline(0, linestyle='--', color='black', linewidth=1.7)
    ax.set_yticks(merged['y_base'])
    ax.set_yticklabels(merged['Label'])
    ax.set_xlabel('Average probability difference, percentage points')
    ax.set_title('Parametric g-computation')
    ax.grid(axis='x', alpha=0.25)
    ax.legend(frameon=False, loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()
gcomp_dualcolor_path = FIG_DIR / 'g_computation_M1_M4_dualcolor.png'
plot_gcomp_m1_m4_dualcolor(gcomp_results_plot, gcomp_dualcolor_path)
pass

def export_m1_m4_check_tables(gee_df, gcomp_df, output_path):
    gee_check = gee_df.copy()
    gee_check['Exposure_clean'] = gee_check['Exposure'].apply(pretty_exposure_name)
    gee_check = gee_check[gee_check['Adjustment model'].isin([GEE_M1_NAME, GEE_M4_NAME]) & gee_check['Exposure_clean'].isin(['Depth.of.A', 'LLBCE'])].copy()
    gee_check = safe_numeric(gee_check, ['RR', 'CI lower', 'CI upper', 'P value', 'E-value', 'E-value for CI'])
    gee_check['Model short'] = gee_check['Adjustment model'].map({GEE_M1_NAME: 'M1', GEE_M4_NAME: 'M4'})
    gee_cols = ['Outcome class', 'Exposure_clean', 'Model short', 'RR', 'CI lower', 'CI upper', 'P value']
    optional_gee_cols = ['E-value', 'E-value for CI']
    for c in optional_gee_cols:
        if c in gee_check.columns:
            gee_cols.append(c)
    gee_check = gee_check[gee_cols].copy()
    gcomp_check = gcomp_df.copy()
    gcomp_check['Intervention_clean'] = gcomp_check['Intervention'].apply(pretty_intervention_name)
    gcomp_check = gcomp_check[gcomp_check['G-computation model'].isin([GCOMP_M1_NAME, GCOMP_M4_NAME])].copy()
    gcomp_check = safe_numeric(gcomp_check, ['Mean probability difference', 'CI lower', 'CI upper', 'Mean probability difference (%)', 'CI lower (%)', 'CI upper (%)'])
    gcomp_check['Model short'] = gcomp_check['G-computation model'].map({GCOMP_M1_NAME: 'M1', GCOMP_M4_NAME: 'M4'})
    gcomp_cols = ['Intervention_clean', 'Outcome class', 'Model short', 'Mean probability difference', 'CI lower', 'CI upper']
    for c in ['Mean probability difference (%)', 'CI lower (%)', 'CI upper (%)']:
        if c in gcomp_check.columns:
            gcomp_cols.append(c)
    gcomp_check = gcomp_check[gcomp_cols].copy()
    for df_tmp in [gee_check, gcomp_check]:
        num_cols = df_tmp.select_dtypes(include=[np.number]).columns
        df_tmp[num_cols] = df_tmp[num_cols].round(4)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        gee_check.to_excel(writer, sheet_name='GEE_M1_M4_plot_data', index=False)
        gcomp_check.to_excel(writer, sheet_name='Gcomp_M1_M4_plot_data', index=False)
check_table_path = FIG_DIR / 'M1_M4_plot_data_check_tables.xlsx'
export_m1_m4_check_tables(gee_results_plot, gcomp_results_plot, check_table_path)
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass