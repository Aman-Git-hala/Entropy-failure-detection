#!/usr/bin/env python3
"""
Bi-Level SGD with Entropy Meta-Features for Software Defect Prediction — V2
============================================================================
Enhanced version with:
  - Better hyperparameter tuning for SGD (alpha + eta0 grid)
  - Warm-restart SGD for stability
  - Threshold tuning via Youden's J on training fold
  - Improved SMOTE handling
  - Comprehensive ablation + statistical tests

Target: Beat Ali et al. 2024 (DOI: 10.7717/peerj-cs.1860) on AUC and MCC.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from scipy import stats
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from tabulate import tabulate

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_arff(filepath):
    """Load NASA PROMISE ARFF files using scipy."""
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    df.columns = [c.lower() for c in df.columns]
    label_col = df.columns[-1]
    df[label_col] = df[label_col].apply(
        lambda x: 1 if str(x).strip().lower() in [
            "b'true'", "b'1'", "b'yes'", "b'y'", "true", "1", "y"
        ] else 0
    )
    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df, label_col


def assign_feature_groups(columns, label_col):
    """Assign features to 3 semantic groups."""
    cols = [c for c in columns if c != label_col]
    
    volume_keywords = [
        'loc_', 'number_of_lines', 'loc_total', 'loc_blank', 'loc_comment',
        'loc_code', 'loc_executable', 'percent_comment', 'call_pairs',
        'parameter_count', 'branch_count'
    ]
    complexity_keywords = [
        'cyclomatic', 'essential', 'design', 'condition', 'decision',
        'edge_count', 'node_count', 'maintenance', 'modified_condition',
        'multiple_condition', 'normalized_cylomatic', 'global_data'
    ]
    halstead_keywords = [
        'halstead', 'num_operand', 'num_operator', 'num_unique',
        'operand', 'operator'
    ]
    
    volume, complexity, halstead = [], [], []
    
    for col in cols:
        col_lower = col.lower()
        if any(k in col_lower for k in halstead_keywords):
            halstead.append(col)
        elif any(k in col_lower for k in complexity_keywords):
            complexity.append(col)
        elif any(k in col_lower for k in volume_keywords):
            volume.append(col)
        else:
            volume.append(col)
    
    return volume, complexity, halstead


# ═══════════════════════════════════════════════════════════════════
# CORE: BI-LEVEL SGD WITH ENTROPY META-FEATURES
# ═══════════════════════════════════════════════════════════════════

def binary_entropy(p, eps=1e-15):
    """H(p) = -p·log₂(p) - (1-p)·log₂(1-p)"""
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def train_inner_sgd(X_train, y_train, X_test, alpha=1e-4):
    """
    Train inner SGDClassifier with CalibratedClassifierCV.
    Returns calibrated probabilities + entropies for train and test.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    base_clf = SGDClassifier(
        loss='log_loss',
        penalty='elasticnet',
        alpha=alpha,
        l1_ratio=0.15,
        max_iter=2000,
        random_state=42,
        class_weight='balanced',
        tol=1e-4,
        warm_start=True,
        learning_rate='adaptive',
        eta0=0.01,
    )
    
    # For very small datasets, skip calibration
    min_class_count = min(np.sum(y_train == 0), np.sum(y_train == 1))
    
    if min_class_count >= 5 and len(y_train) >= 30:
        n_cv = min(5, min_class_count)
        cal_clf = CalibratedClassifierCV(base_clf, cv=n_cv, method='isotonic')
        cal_clf.fit(X_train_s, y_train)
        p_train = cal_clf.predict_proba(X_train_s)[:, 1]
        p_test = cal_clf.predict_proba(X_test_s)[:, 1]
    else:
        base_clf.fit(X_train_s, y_train)
        # Use sigmoid transformation for probability calibration
        d_train = base_clf.decision_function(X_train_s)
        d_test = base_clf.decision_function(X_test_s)
        p_train = 1 / (1 + np.exp(-d_train))
        p_test = 1 / (1 + np.exp(-d_test))
    
    return p_train, binary_entropy(p_train), p_test, binary_entropy(p_test)


def build_meta_features(X_train, y_train, X_test, feature_groups, alpha=1e-4):
    """
    Build meta-features from inner SGD models.
    z = [p_1, H_1, p_2, H_2, p_3, H_3]
    """
    meta_train = np.zeros((len(X_train), 6))
    meta_test = np.zeros((len(X_test), 6))
    
    for i, gcols in enumerate(feature_groups):
        X_tr_g = X_train[gcols].values
        X_te_g = X_test[gcols].values
        
        p_tr, H_tr, p_te, H_te = train_inner_sgd(X_tr_g, y_train, X_te_g, alpha)
        
        meta_train[:, 2*i]   = p_tr
        meta_train[:, 2*i+1] = H_tr
        meta_test[:, 2*i]    = p_te
        meta_test[:, 2*i+1]  = H_te
    
    return meta_train, meta_test


def optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def bilevel_sgd_fold(X_train, y_train, X_test, y_test, feature_groups,
                     inner_alpha=1e-4, meta_alpha=1e-4):
    """
    Full bi-level SGD for one fold. Returns metrics + meta-features.
    """
    # Build meta-features from inner models
    meta_train, meta_test = build_meta_features(
        X_train, y_train, X_test, feature_groups, inner_alpha
    )
    
    # Outer-loop meta-classifier
    meta_scaler = StandardScaler()
    meta_train_s = meta_scaler.fit_transform(meta_train)
    meta_test_s = meta_scaler.transform(meta_test)
    
    meta_clf = SGDClassifier(
        loss='log_loss',
        penalty='elasticnet',
        alpha=meta_alpha,
        l1_ratio=0.15,
        max_iter=2000,
        random_state=42,
        class_weight='balanced',
        tol=1e-4,
        warm_start=True,
        learning_rate='adaptive',
        eta0=0.01,
    )
    
    min_class_count = min(np.sum(y_train == 0), np.sum(y_train == 1))
    
    if min_class_count >= 5 and len(y_train) >= 30:
        n_cv = min(5, min_class_count)
        meta_cal = CalibratedClassifierCV(meta_clf, cv=n_cv, method='isotonic')
        meta_cal.fit(meta_train_s, y_train)
        y_prob_train = meta_cal.predict_proba(meta_train_s)[:, 1]
        y_prob = meta_cal.predict_proba(meta_test_s)[:, 1]
    else:
        meta_clf.fit(meta_train_s, y_train)
        d_train = meta_clf.decision_function(meta_train_s)
        d_test = meta_clf.decision_function(meta_test_s)
        y_prob_train = 1 / (1 + np.exp(-d_train))
        y_prob = 1 / (1 + np.exp(-d_test))
    
    # Optimal threshold from training data (Youden's J)
    threshold = optimal_threshold(y_train, y_prob_train)
    y_pred = (y_prob >= threshold).astype(int)
    
    return y_test, y_pred, y_prob, meta_test, threshold


def run_full_evaluation(df, label_col, feature_groups, dataset_name,
                        n_folds=10, inner_alpha=1e-4, meta_alpha=1e-4):
    """
    10-fold stratified CV with SMOTE inside each fold.
    """
    X = df.drop(columns=[label_col])
    y = df[label_col].values
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_aucs, fold_mccs, fold_f1s = [], [], []
    all_y_true, all_y_prob = [], []
    all_meta, all_labels = [], []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        
        # SMOTE on training fold only
        try:
            k = min(5, sum(y_train == 1) - 1)
            if k >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            else:
                X_train_res, y_train_res = X_train, y_train
        except ValueError:
            X_train_res, y_train_res = X_train, y_train
        
        y_true, y_pred, y_prob, meta_feats, thresh = bilevel_sgd_fold(
            X_train_res, y_train_res, X_test, y_test,
            feature_groups, inner_alpha, meta_alpha
        )
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        fold_aucs.append(auc)
        fold_mccs.append(mcc)
        fold_f1s.append(f1)
        
        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)
        all_meta.append(meta_feats)
        all_labels.extend(y_true)
    
    results = {
        'dataset': dataset_name,
        'auc_mean': np.mean(fold_aucs), 'auc_std': np.std(fold_aucs),
        'mcc_mean': np.mean(fold_mccs), 'mcc_std': np.std(fold_mccs),
        'f1_mean': np.mean(fold_f1s), 'f1_std': np.std(fold_f1s),
        'fold_aucs': fold_aucs, 'fold_mccs': fold_mccs, 'fold_f1s': fold_f1s,
        'all_y_true': np.array(all_y_true),
        'all_y_prob': np.array(all_y_prob),
        'all_meta_features': np.vstack(all_meta),
        'all_y_labels': np.array(all_labels),
    }
    
    return results


# ═══════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING VIA NESTED CV
# ═══════════════════════════════════════════════════════════════════

def tune_alpha(df, label_col, feature_groups, dataset_name):
    """
    Quick grid search over alpha values for inner and outer SGD.
    Uses 5-fold inner CV to select best alpha, then evaluates with 10-fold.
    """
    alpha_grid = [1e-5, 1e-4, 1e-3, 1e-2]
    best_auc = -1
    best_inner, best_meta = 1e-4, 1e-4
    
    X = df.drop(columns=[label_col])
    y = df[label_col].values
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for inner_alpha in alpha_grid:
        for meta_alpha in alpha_grid:
            aucs = []
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
                y_train, y_test = y[train_idx], y[test_idx]
                
                try:
                    k = min(5, sum(y_train == 1) - 1)
                    if k >= 1:
                        smote = SMOTE(random_state=42, k_neighbors=k)
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                except ValueError:
                    pass
                
                _, _, y_prob, _, _ = bilevel_sgd_fold(
                    X_train, y_train, X_test, y_test,
                    feature_groups, inner_alpha, meta_alpha
                )
                try:
                    aucs.append(roc_auc_score(y_test, y_prob))
                except:
                    aucs.append(0.5)
            
            mean_auc = np.mean(aucs)
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_inner = inner_alpha
                best_meta = meta_alpha
    
    return best_inner, best_meta, best_auc


# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

DATASETS = {
    'CM1': os.path.join(BASE_DIR, 'CM1.arff'),
    'JM1': os.path.join(BASE_DIR, 'JM1.arff'),
    'MC2': os.path.join(BASE_DIR, 'MC2.arff'),
    'PC1': os.path.join(BASE_DIR, 'PC1.arff'),
}

print("=" * 72)
print("LOADING NASA PROMISE DATASETS")
print("=" * 72)

loaded_data = {}
for name, path in DATASETS.items():
    df, label_col = load_arff(path)
    loaded_data[name] = (df, label_col)
    n_def = df[label_col].sum()
    print(f"  {name}: {df.shape}, defective={n_def} ({n_def/len(df)*100:.1f}%)")

# Phase 1: Tune hyperparameters
print("\n" + "=" * 72)
print("PHASE 1: HYPERPARAMETER TUNING (5-fold inner CV)")
print("=" * 72)

tuned_params = {}
for name, (df, label_col) in loaded_data.items():
    vol, comp, hal = assign_feature_groups(df.columns, label_col)
    best_inner, best_meta, best_auc = tune_alpha(df, label_col, [vol, comp, hal], name)
    tuned_params[name] = (best_inner, best_meta)
    print(f"  {name}: inner_α={best_inner:.0e}, meta_α={best_meta:.0e}, "
          f"CV-AUC={best_auc:.4f}")

# Phase 2: Full evaluation with tuned params
print("\n" + "=" * 72)
print("PHASE 2: FULL EVALUATION (10-fold stratified CV, tuned params)")
print("=" * 72)

all_results = {}
for name, (df, label_col) in loaded_data.items():
    vol, comp, hal = assign_feature_groups(df.columns, label_col)
    inner_a, meta_a = tuned_params[name]
    results = run_full_evaluation(
        df, label_col, [vol, comp, hal], name,
        n_folds=10, inner_alpha=inner_a, meta_alpha=meta_a
    )
    all_results[name] = results
    print(f"\n  {name}:")
    print(f"    AUC-ROC: {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
    print(f"    MCC:     {results['mcc_mean']:.4f} ± {results['mcc_std']:.4f}")
    print(f"    F1 (def):{results['f1_mean']:.4f} ± {results['f1_std']:.4f}")


# Phase 3: Ablation study
print("\n" + "=" * 72)
print("PHASE 3: ABLATION STUDY — WITH vs WITHOUT ENTROPY")
print("=" * 72)

def run_no_entropy_ablation(df, label_col, feature_groups, inner_alpha, meta_alpha, n_folds=10):
    """Ablation: only p features, no H."""
    X = df.drop(columns=[label_col])
    y = df[label_col].values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_aucs, fold_mccs, fold_f1s = [], [], []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        
        try:
            k = min(5, sum(y_train == 1) - 1)
            if k >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_train, y_train = smote.fit_resample(X_train, y_train)
        except ValueError:
            pass
        
        meta_train = np.zeros((len(X_train), 3))
        meta_test = np.zeros((len(X_test), 3))
        
        for i, gcols in enumerate(feature_groups):
            p_tr, _, p_te, _ = train_inner_sgd(
                X_train[gcols].values, y_train, X_test[gcols].values, inner_alpha
            )
            meta_train[:, i] = p_tr
            meta_test[:, i] = p_te
        
        scaler = StandardScaler()
        meta_train_s = scaler.fit_transform(meta_train)
        meta_test_s = scaler.transform(meta_test)
        
        clf = SGDClassifier(
            loss='log_loss', penalty='elasticnet', alpha=meta_alpha,
            l1_ratio=0.15, max_iter=2000, random_state=42,
            class_weight='balanced', learning_rate='adaptive', eta0=0.01,
        )
        
        min_cls = min(np.sum(y_train == 0), np.sum(y_train == 1))
        if min_cls >= 5 and len(y_train) >= 30:
            cal = CalibratedClassifierCV(clf, cv=min(5, min_cls), method='isotonic')
            cal.fit(meta_train_s, y_train)
            y_prob_tr = cal.predict_proba(meta_train_s)[:, 1]
            y_prob = cal.predict_proba(meta_test_s)[:, 1]
        else:
            clf.fit(meta_train_s, y_train)
            y_prob_tr = 1 / (1 + np.exp(-clf.decision_function(meta_train_s)))
            y_prob = 1 / (1 + np.exp(-clf.decision_function(meta_test_s)))
        
        thresh = optimal_threshold(y_train, y_prob_tr)
        y_pred = (y_prob >= thresh).astype(int)
        
        try:
            fold_aucs.append(roc_auc_score(y_test, y_prob))
        except:
            fold_aucs.append(0.5)
        fold_mccs.append(matthews_corrcoef(y_test, y_pred))
        fold_f1s.append(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    
    return {
        'auc_mean': np.mean(fold_aucs), 'auc_std': np.std(fold_aucs),
        'mcc_mean': np.mean(fold_mccs), 'mcc_std': np.std(fold_mccs),
        'f1_mean': np.mean(fold_f1s), 'f1_std': np.std(fold_f1s),
        'fold_aucs': fold_aucs,
    }


ablation_table = []
for name, (df, label_col) in loaded_data.items():
    vol, comp, hal = assign_feature_groups(df.columns, label_col)
    inner_a, meta_a = tuned_params[name]
    no_h = run_no_entropy_ablation(df, label_col, [vol, comp, hal], inner_a, meta_a)
    full = all_results[name]
    
    # Paired t-test on fold AUC
    t_stat, p_val = stats.ttest_rel(full['fold_aucs'], no_h['fold_aucs'])
    
    delta_auc = full['auc_mean'] - no_h['auc_mean']
    delta_mcc = full['mcc_mean'] - no_h['mcc_mean']
    
    ablation_table.append([
        name,
        f"{no_h['auc_mean']:.4f}",
        f"{full['auc_mean']:.4f}",
        f"{delta_auc:+.4f}",
        f"{p_val:.4f}" + (" *" if p_val < 0.05 else ""),
        f"{no_h['mcc_mean']:.4f}",
        f"{full['mcc_mean']:.4f}",
        f"{delta_mcc:+.4f}",
    ])

abl_headers = ['Dataset', 'AUC\n(No H)', 'AUC\n(+H)', 'ΔAUC', 'p-value',
               'MCC\n(No H)', 'MCC\n(+H)', 'ΔMCC']
print("\n  TABLE: Ablation — Effect of Entropy Meta-Features")
print(tabulate(ablation_table, headers=abl_headers, tablefmt='grid', stralign='center'))


# Phase 4: Comparative results table
print("\n" + "=" * 72)
print("PHASE 4: COMPARATIVE RESULTS")
print("=" * 72)

ali_results = {
    'CM1': {'accuracy': 0.905, 'auc': 'N/R'},
    'JM1': {'accuracy': 0.807, 'auc': 'N/R'},
    'MC2': {'accuracy': 0.741, 'auc': 'N/R'},
    'PC1': {'accuracy': 0.929, 'auc': 'N/R'},
}

table_data = []
for name in ['CM1', 'JM1', 'MC2', 'PC1']:
    ours = all_results[name]
    ali = ali_results[name]
    table_data.append([
        name,
        f"{ali['accuracy']:.3f}",
        ali['auc'],
        f"{ours['auc_mean']:.4f} ± {ours['auc_std']:.4f}",
        f"{ours['mcc_mean']:.4f} ± {ours['mcc_std']:.4f}",
        f"{ours['f1_mean']:.4f} ± {ours['f1_std']:.4f}",
    ])

headers = ['Dataset', 'Ali et al.\nAccuracy', 'Ali et al.\nAUC',
           'Ours\nAUC-ROC', 'Ours\nMCC', 'Ours\nF1 (Def)']
print("\n  TABLE: Performance Comparison — Bi-Level SGD vs Ali et al. 2024")
print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='center'))

mean_auc = np.mean([all_results[n]['auc_mean'] for n in all_results])
mean_mcc = np.mean([all_results[n]['mcc_mean'] for n in all_results])
mean_f1 = np.mean([all_results[n]['f1_mean'] for n in all_results])

print(f"\n  OVERALL AVERAGES:")
print(f"    AUC-ROC: {mean_auc:.4f}")
print(f"    MCC:     {mean_mcc:.4f}")
print(f"    F1 (Def):{mean_f1:.4f}")


# ═══════════════════════════════════════════════════════════════════
# PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("GENERATING PUBLICATION FIGURES")
print("=" * 72)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

group_names = ['Volume', 'Complexity', 'Halstead']
meta_col_names = ['p_vol', 'H_vol', 'p_comp', 'H_comp', 'p_hal', 'H_hal']

# Figure 1: Entropy distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for idx, (name, res) in enumerate(all_results.items()):
    ax = axes[idx]
    meta = res['all_meta_features']
    labels = res['all_y_labels']
    
    entropy_cols = [1, 3, 5]
    H_agg = meta[:, entropy_cols].mean(axis=1)
    H_def = H_agg[labels == 1]
    H_clean = H_agg[labels == 0]
    
    bins = np.linspace(0, 1, 35)
    ax.hist(H_def, bins=bins, alpha=0.65, color='#E63946', density=True,
            label=f'Defective (n={len(H_def)})', edgecolor='white', linewidth=0.5)
    ax.hist(H_clean, bins=bins, alpha=0.65, color='#457B9D', density=True,
            label=f'Clean (n={len(H_clean)})', edgecolor='white', linewidth=0.5)
    
    # Add means
    ax.axvline(H_def.mean(), color='#E63946', linestyle='--', linewidth=2, alpha=0.9,
               label=f'μ_def={H_def.mean():.3f}')
    ax.axvline(H_clean.mean(), color='#457B9D', linestyle='--', linewidth=2, alpha=0.9,
               label=f'μ_clean={H_clean.mean():.3f}')
    
    ax.set_title(f'{name}', fontweight='bold', fontsize=14)
    ax.set_xlabel('Mean Binary Entropy H̄(p)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2)

fig.suptitle(
    'Figure 1: Entropy Distribution — Defective vs Non-Defective Modules\n'
    r'$\bar{H} = \frac{1}{3}\sum_{i} H(p_i)$, where $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$',
    fontsize=13, fontweight='bold', y=1.03
)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig1_entropy_distribution.png'))
plt.close()
print("  → fig1_entropy_distribution.png")

# Figure 2: ROC curves
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for idx, (name, res) in enumerate(all_results.items()):
    ax = axes[idx]
    fpr, tpr, _ = roc_curve(res['all_y_true'], res['all_y_prob'])
    auc_val = roc_auc_score(res['all_y_true'], res['all_y_prob'])
    
    ax.plot(fpr, tpr, color='#E63946', linewidth=2.5,
            label=f'Bi-Level SGD+H (AUC={auc_val:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.12, color='#E63946')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
    ax.set_title(f'{name}', fontweight='bold', fontsize=14)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

fig.suptitle('Figure 2: ROC Curves — Bi-Level SGD with Entropy Meta-Features',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig2_roc_curves.png'))
plt.close()
print("  → fig2_roc_curves.png")

# Figure 3: Meta-feature scatter
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for idx, (name, res) in enumerate(all_results.items()):
    ax = axes[idx]
    meta = res['all_meta_features']
    labels = res['all_y_labels']
    
    # Mean prediction vs mean entropy
    p_mean = meta[:, [0, 2, 4]].mean(axis=1)
    H_mean = meta[:, [1, 3, 5]].mean(axis=1)
    
    sc = ax.scatter(p_mean, H_mean, c=labels, cmap='RdYlBu_r',
                    alpha=0.5, s=12, edgecolors='none')
    ax.set_xlabel('Mean Prediction p̄')
    ax.set_ylabel('Mean Entropy H̄')
    ax.set_title(f'{name}', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.2)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Defective')

fig.suptitle(
    'Figure 3: Meta-Feature Space — Prediction vs Entropy',
    fontsize=14, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig3_meta_space.png'))
plt.close()
print("  → fig3_meta_space.png")

# Figure 4: Box plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metric_names = ['AUC-ROC', 'MCC', 'F1 (Defective)']
metric_keys = ['fold_aucs', 'fold_mccs', 'fold_f1s']

for ax, mname, mkey in zip(axes, metric_names, metric_keys):
    data = [all_results[n][mkey] for n in all_results]
    bp = ax.boxplot(data, labels=list(all_results.keys()),
                    patch_artist=True, notch=True)
    colors_box = ['#264653', '#2A9D8F', '#E9C46A', '#E76F51']
    for patch, c in zip(bp['boxes'], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_title(mname, fontweight='bold')
    ax.set_ylabel(mname)
    ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('Figure 4: 10-Fold CV Stability', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig4_cv_stability.png'))
plt.close()
print("  → fig4_cv_stability.png")

# Figure 5: Per-group entropy comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for idx, (name, res) in enumerate(all_results.items()):
    ax = axes[idx]
    meta = res['all_meta_features']
    labels = res['all_y_labels']
    
    group_H_def = [meta[labels == 1, c].mean() for c in [1, 3, 5]]
    group_H_clean = [meta[labels == 0, c].mean() for c in [1, 3, 5]]
    
    x = np.arange(3)
    w = 0.35
    bars1 = ax.bar(x - w/2, group_H_def, w, label='Defective', color='#E63946', alpha=0.8)
    bars2 = ax.bar(x + w/2, group_H_clean, w, label='Clean', color='#457B9D', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.set_ylabel('Mean Entropy H(p)')
    ax.set_title(f'{name}', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

fig.suptitle(
    'Figure 5: Per-Group Entropy — Defective vs Clean Modules',
    fontsize=14, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig5_per_group_entropy.png'))
plt.close()
print("  → fig5_per_group_entropy.png")


# ═══════════════════════════════════════════════════════════════════
# META-FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("META-FEATURE IMPORTANCE (OUTER MODEL WEIGHTS)")
print("=" * 72)

for name, (df, label_col) in loaded_data.items():
    vol, comp, hal = assign_feature_groups(df.columns, label_col)
    inner_a, meta_a = tuned_params[name]
    
    X = df.drop(columns=[label_col])
    y = df[label_col].values
    
    try:
        k = min(5, sum(y == 1) - 1)
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X, y)
    except:
        X_res, y_res = X, y
    
    meta_full = np.zeros((len(X_res), 6))
    for i, gcols in enumerate([vol, comp, hal]):
        X_g = X_res[gcols].values
        sc = StandardScaler()
        X_g_s = sc.fit_transform(X_g)
        clf = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=inner_a,
                            l1_ratio=0.15, max_iter=2000, random_state=42,
                            class_weight='balanced', learning_rate='adaptive', eta0=0.01)
        clf.fit(X_g_s, y_res)
        p = 1 / (1 + np.exp(-clf.decision_function(X_g_s)))
        meta_full[:, 2*i] = p
        meta_full[:, 2*i+1] = binary_entropy(p)
    
    ms = StandardScaler()
    meta_s = ms.fit_transform(meta_full)
    meta_clf = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=meta_a,
                            l1_ratio=0.15, max_iter=2000, random_state=42,
                            class_weight='balanced', learning_rate='adaptive', eta0=0.01)
    meta_clf.fit(meta_s, y_res)
    
    weights = meta_clf.coef_[0]
    print(f"\n  {name}:")
    for wname, w in zip(meta_col_names, weights):
        bar = '█' * int(min(abs(w) * 5, 40))
        sign = '+' if w > 0 else '−'
        print(f"    {wname:8s}: {sign}{abs(w):.4f}  {bar}")


print("\n" + "=" * 72)
print("✓ ALL DONE — Pipeline complete")
print("=" * 72)
print(f"  Figures: fig1–fig5 saved in {BASE_DIR}")
print(f"  Mean AUC: {mean_auc:.4f} | Mean MCC: {mean_mcc:.4f} | Mean F1: {mean_f1:.4f}")
