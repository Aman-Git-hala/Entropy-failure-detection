#!/usr/bin/env python3
"""
Entropy-Calibrated Multi-View Failure Prediction for Predictive Maintenance
============================================================================

Bi-level SGD pipeline with entropy + KL-divergence meta-features.

Architecture:
    Raw Sensor Data
          |
          +-- Thermal  [air_temp, proc_temp]       --> SGD_1 --> p1, H(p1)
          +-- Mechanical [rot_speed, torque]        --> SGD_2 --> p2, H(p2)
          +-- Wear [tool_wear, type_encoded]        --> SGD_3 --> p3, H(p3)
                                                          |
                        Meta-features: [p1,H1, p2,H2, p3,H3, KL12,KL13,KL23, dH12,dH13,dH23]
                                                          |
                                                    SGD_outer --> prediction + uncertainty

Dataset: AI4I 2020 Predictive Maintenance (Matzka, 2020)
         10,000 samples, binary failure target

Baselines: Naive Bayes, Logistic Regression, Random Forest, XGBoost
           (all run in same CV folds for fair comparison)

Reference baselines from literature (AI4I 2020):
    - Matzka (2020): RF AUC = 0.954
    - Recent benchmarks: RF = 0.971, XGBoost = 0.974
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    precision_score, recall_score, accuracy_score,
    roc_curve, confusion_matrix, brier_score_loss
)

from imblearn.over_sampling import SMOTE
from tabulate import tabulate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────── paths ───────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "ai4i_2020.csv")

# ─────────────────────────── Semantic Feature Groups ───────────────────────────
GROUPS = {
    "Thermal": ["Air temperature [K]", "Process temperature [K]"],
    "Mechanical": ["Rotational speed [rpm]", "Torque [Nm]"],
    "Wear": ["Tool wear [min]", "Type_encoded"],
}
GROUP_NAMES = list(GROUPS.keys())
N_GROUPS = len(GROUP_NAMES)

# ─────────────────────────── Helper functions ───────────────────────────

def binary_entropy(p, eps=1e-15):
    """H(p) = -p log2(p) - (1-p) log2(1-p)"""
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def kl_divergence(p, q, eps=1e-15):
    """KL(p || q) for Bernoulli distributions."""
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log2(p / q) + (1 - p) * np.log2((1 - p) / (1 - q))


def youdens_j_threshold(y_true, y_prob):
    """Find threshold maximizing Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def tune_alpha(X_train, y_train, alphas=None, n_folds=3):
    """Grid-search alpha for SGDClassifier via inner CV."""
    if alphas is None:
        alphas = [1e-5, 1e-4, 1e-3, 1e-2]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    best_alpha, best_score = alphas[0], -1
    for a in alphas:
        scores = []
        for tr, va in skf.split(X_train, y_train):
            clf = SGDClassifier(
                loss="log_loss", penalty="elasticnet",
                alpha=a, l1_ratio=0.15, max_iter=2000,
                class_weight="balanced", random_state=42
            )
            clf.fit(X_train[tr], y_train[tr])
            prob = clf.decision_function(X_train[va])
            try:
                scores.append(roc_auc_score(y_train[va], prob))
            except ValueError:
                scores.append(0.5)
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = a
    return best_alpha


# ─────────────────────────── Data Loading ───────────────────────────

def load_data():
    """Load and preprocess AI4I 2020 dataset."""
    df = pd.read_csv(DATA_PATH)
    
    # Encode product type: L=0, M=1, H=2
    le = LabelEncoder()
    df["Type_encoded"] = le.fit_transform(df["Type"])
    
    # Collect all feature columns
    feature_cols = []
    for g in GROUP_NAMES:
        feature_cols.extend(GROUPS[g])
    
    X = df[feature_cols].values.astype(float)
    y = df["Machine failure"].values.astype(int)
    
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Failure rate: {y.mean():.1%} ({y.sum()}/{len(y)})")
    print(f"  Feature groups: {', '.join(GROUP_NAMES)}")
    for g in GROUP_NAMES:
        print(f"    {g}: {GROUPS[g]}")
    
    return X, y, feature_cols


# ─────────────────────────── Build Meta-Features ───────────────────────────

def build_meta_features(X_train, y_train, X_test, feature_cols):
    """
    Train per-group SGD models, compute predictions, entropy,
    KL divergence, and entropy contrast meta-features.
    
    Returns:
        meta_train: (n_train, 12) array of meta-features for training
        meta_test:  (n_test, 12)  array of meta-features for test
        group_models: list of trained (scaler, model) per group
        group_alphas: list of tuned alpha per group
    """
    group_models = []
    group_alphas = []
    
    train_preds = []   # per-group probabilities for train
    test_preds = []    # per-group probabilities for test
    
    for g_name in GROUP_NAMES:
        # Get column indices for this group
        g_cols = GROUPS[g_name]
        g_idx = [feature_cols.index(c) for c in g_cols]
        
        Xg_train = X_train[:, g_idx]
        Xg_test = X_test[:, g_idx]
        
        # Scale
        scaler = StandardScaler()
        Xg_train_s = scaler.fit_transform(Xg_train)
        Xg_test_s = scaler.transform(Xg_test)
        
        # Tune alpha
        alpha = tune_alpha(Xg_train_s, y_train)
        group_alphas.append(alpha)
        
        # Train SGD with isotonic calibration
        base = SGDClassifier(
            loss="log_loss", penalty="elasticnet",
            alpha=alpha, l1_ratio=0.15, max_iter=2000,
            class_weight="balanced", random_state=42
        )
        cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal.fit(Xg_train_s, y_train)
        
        # Predict probabilities
        p_train = cal.predict_proba(Xg_train_s)[:, 1]
        p_test = cal.predict_proba(Xg_test_s)[:, 1]
        
        train_preds.append(p_train)
        test_preds.append(p_test)
        group_models.append((scaler, cal))
    
    # ── Compute meta-features ──
    def compute_meta(preds_list):
        n = len(preds_list[0])
        
        # Base: [p1, H1, p2, H2, p3, H3]
        base_feats = []
        entropies = []
        for p in preds_list:
            h = binary_entropy(p)
            base_feats.append(p)
            base_feats.append(h)
            entropies.append(h)
        
        # KL divergence between groups: KL(1||2), KL(1||3), KL(2||3)
        kl_feats = []
        pairs = [(0, 1), (0, 2), (1, 2)]
        for i, j in pairs:
            kl = kl_divergence(preds_list[i], preds_list[j])
            kl_feats.append(kl)
        
        # Entropy contrast: |H1-H2|, |H1-H3|, |H2-H3|
        dh_feats = []
        for i, j in pairs:
            dh = np.abs(entropies[i] - entropies[j])
            dh_feats.append(dh)
        
        # Stack: 6 base + 3 KL + 3 dH = 12 meta-features
        meta = np.column_stack(base_feats + kl_feats + dh_feats)
        return meta
    
    meta_train = compute_meta(train_preds)
    meta_test = compute_meta(test_preds)
    
    return meta_train, meta_test, group_models, group_alphas


def build_meta_features_no_entropy(X_train, y_train, X_test, feature_cols):
    """Ablation: meta-features WITHOUT entropy/KL/dH — only base predictions."""
    group_models = []
    
    train_preds = []
    test_preds = []
    
    for g_name in GROUP_NAMES:
        g_cols = GROUPS[g_name]
        g_idx = [feature_cols.index(c) for c in g_cols]
        
        Xg_train = X_train[:, g_idx]
        Xg_test = X_test[:, g_idx]
        
        scaler = StandardScaler()
        Xg_train_s = scaler.fit_transform(Xg_train)
        Xg_test_s = scaler.transform(Xg_test)
        
        alpha = tune_alpha(Xg_train_s, y_train)
        
        base = SGDClassifier(
            loss="log_loss", penalty="elasticnet",
            alpha=alpha, l1_ratio=0.15, max_iter=2000,
            class_weight="balanced", random_state=42
        )
        cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal.fit(Xg_train_s, y_train)
        
        p_train = cal.predict_proba(Xg_train_s)[:, 1]
        p_test = cal.predict_proba(Xg_test_s)[:, 1]
        
        train_preds.append(p_train)
        test_preds.append(p_test)
        group_models.append((scaler, cal))
    
    # Only base predictions: [p1, p2, p3]
    meta_train = np.column_stack(train_preds)
    meta_test = np.column_stack(test_preds)
    
    return meta_train, meta_test


# ─────────────────────────── Main Experiment ───────────────────────────

def run_experiment():
    """Run full 10-fold CV experiment with baselines and ablation."""
    X, y, feature_cols = load_data()
    
    N_FOLDS = 10
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # ── Results storage ──
    methods = [
        "Ours (Full)",        # bilevel + entropy + KL
        "Ours (No Entropy)",  # ablation: bilevel, predictions only
        "Naive Bayes",
        "Logistic Regression",
        "Random Forest",
        "XGBoost (GB)",
    ]
    results = {m: defaultdict(list) for m in methods}
    
    # For per-group entropy analysis
    all_entropies = {g: {"fail": [], "ok": []} for g in GROUP_NAMES}
    all_meta_test = []
    all_y_test = []
    all_probs_full = []
    
    print(f"\n{'='*70}")
    print(f"  10-Fold Stratified Cross-Validation")
    print(f"{'='*70}")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train_raw, y_test = y[train_idx], y[test_idx]
        
        # SMOTE on training fold only
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_raw, y_train_raw)
        
        # ── 1. Our Full Method (bilevel + entropy + KL) ──
        meta_train, meta_test, g_models, g_alphas = build_meta_features(
            X_train_sm, y_train_sm, X_test_raw, feature_cols
        )
        
        # Scale meta-features
        meta_scaler = StandardScaler()
        meta_train_s = meta_scaler.fit_transform(meta_train)
        meta_test_s = meta_scaler.transform(meta_test)
        
        # Tune outer SGD
        outer_alpha = tune_alpha(meta_train_s, y_train_sm)
        outer_base = SGDClassifier(
            loss="log_loss", penalty="elasticnet",
            alpha=outer_alpha, l1_ratio=0.15, max_iter=2000,
            class_weight="balanced", random_state=42
        )
        outer_cal = CalibratedClassifierCV(outer_base, method="isotonic", cv=3)
        outer_cal.fit(meta_train_s, y_train_sm)
        
        probs_full = outer_cal.predict_proba(meta_test_s)[:, 1]
        thresh_full = youdens_j_threshold(y_test, probs_full)
        preds_full = (probs_full >= thresh_full).astype(int)
        
        results["Ours (Full)"]["auc"].append(roc_auc_score(y_test, probs_full))
        results["Ours (Full)"]["mcc"].append(matthews_corrcoef(y_test, preds_full))
        results["Ours (Full)"]["f1"].append(f1_score(y_test, preds_full))
        results["Ours (Full)"]["precision"].append(precision_score(y_test, preds_full, zero_division=0))
        results["Ours (Full)"]["recall"].append(recall_score(y_test, preds_full, zero_division=0))
        results["Ours (Full)"]["accuracy"].append(accuracy_score(y_test, preds_full))
        results["Ours (Full)"]["brier"].append(brier_score_loss(y_test, probs_full))
        
        # Store for analysis
        all_meta_test.append(meta_test)
        all_y_test.append(y_test)
        all_probs_full.append(probs_full)
        
        # Collect per-group entropies
        for gi, g_name in enumerate(GROUP_NAMES):
            h_col = meta_test[:, 2 * gi + 1]  # H1 at idx 1, H2 at idx 3, H3 at idx 5
            all_entropies[g_name]["fail"].extend(h_col[y_test == 1].tolist())
            all_entropies[g_name]["ok"].extend(h_col[y_test == 0].tolist())
        
        # ── 2. Ablation: No Entropy ──
        meta_train_ne, meta_test_ne = build_meta_features_no_entropy(
            X_train_sm, y_train_sm, X_test_raw, feature_cols
        )
        
        meta_scaler_ne = StandardScaler()
        meta_train_ne_s = meta_scaler_ne.fit_transform(meta_train_ne)
        meta_test_ne_s = meta_scaler_ne.transform(meta_test_ne)
        
        outer_alpha_ne = tune_alpha(meta_train_ne_s, y_train_sm)
        outer_ne = SGDClassifier(
            loss="log_loss", penalty="elasticnet",
            alpha=outer_alpha_ne, l1_ratio=0.15, max_iter=2000,
            class_weight="balanced", random_state=42
        )
        outer_ne_cal = CalibratedClassifierCV(outer_ne, method="isotonic", cv=3)
        outer_ne_cal.fit(meta_train_ne_s, y_train_sm)
        
        probs_ne = outer_ne_cal.predict_proba(meta_test_ne_s)[:, 1]
        thresh_ne = youdens_j_threshold(y_test, probs_ne)
        preds_ne = (probs_ne >= thresh_ne).astype(int)
        
        results["Ours (No Entropy)"]["auc"].append(roc_auc_score(y_test, probs_ne))
        results["Ours (No Entropy)"]["mcc"].append(matthews_corrcoef(y_test, preds_ne))
        results["Ours (No Entropy)"]["f1"].append(f1_score(y_test, preds_ne))
        results["Ours (No Entropy)"]["precision"].append(precision_score(y_test, preds_ne, zero_division=0))
        results["Ours (No Entropy)"]["recall"].append(recall_score(y_test, preds_ne, zero_division=0))
        results["Ours (No Entropy)"]["accuracy"].append(accuracy_score(y_test, preds_ne))
        results["Ours (No Entropy)"]["brier"].append(brier_score_loss(y_test, probs_ne))
        
        # ── 3. Baselines (flat models on raw features) ──
        scaler_flat = StandardScaler()
        X_train_flat = scaler_flat.fit_transform(X_train_sm)
        X_test_flat = scaler_flat.transform(X_test_raw)
        
        baselines = {
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "XGBoost (GB)": GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42
            ),
        }
        
        for name, clf in baselines.items():
            clf.fit(X_train_flat, y_train_sm)
            if hasattr(clf, "predict_proba"):
                probs_b = clf.predict_proba(X_test_flat)[:, 1]
            else:
                probs_b = clf.decision_function(X_test_flat)
            
            thresh_b = youdens_j_threshold(y_test, probs_b)
            preds_b = (probs_b >= thresh_b).astype(int)
            
            results[name]["auc"].append(roc_auc_score(y_test, probs_b))
            results[name]["mcc"].append(matthews_corrcoef(y_test, preds_b))
            results[name]["f1"].append(f1_score(y_test, preds_b))
            results[name]["precision"].append(precision_score(y_test, preds_b, zero_division=0))
            results[name]["recall"].append(recall_score(y_test, preds_b, zero_division=0))
            results[name]["accuracy"].append(accuracy_score(y_test, preds_b))
            results[name]["brier"].append(brier_score_loss(y_test, probs_b))
        
        print(f"  Fold {fold+1:2d}/{N_FOLDS}: "
              f"Full AUC={results['Ours (Full)']['auc'][-1]:.4f}  "
              f"RF AUC={results['Random Forest']['auc'][-1]:.4f}  "
              f"XGB AUC={results['XGBoost (GB)']['auc'][-1]:.4f}")
    
    # ── Print Results ──
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY (Mean ± Std over 10 folds)")
    print(f"{'='*70}")
    
    metrics_to_show = ["auc", "mcc", "f1", "precision", "recall", "accuracy", "brier"]
    headers = ["Method"] + [m.upper() for m in metrics_to_show]
    table = []
    for m in methods:
        row = [m]
        for metric in metrics_to_show:
            vals = results[m][metric]
            row.append(f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")
        table.append(row)
    
    print(tabulate(table, headers=headers, tablefmt="grid"))
    
    # ── Ablation t-test ──
    print(f"\n{'='*70}")
    print("  ABLATION STUDY: Entropy Features Contribution")
    print(f"{'='*70}")
    
    for metric in ["auc", "mcc", "f1"]:
        full_vals = results["Ours (Full)"][metric]
        ne_vals = results["Ours (No Entropy)"][metric]
        t_stat, p_val = stats.ttest_rel(full_vals, ne_vals)
        delta = np.mean(full_vals) - np.mean(ne_vals)
        sig = "✓ YES" if p_val < 0.05 else "✗ NO"
        print(f"  {metric.upper():>6s}: Full={np.mean(full_vals):.4f}, "
              f"NoEntropy={np.mean(ne_vals):.4f}, "
              f"Δ={delta:+.4f}, p={p_val:.4f} [{sig}]")
    
    # ── Interpretability Analysis ──
    print(f"\n{'='*70}")
    print("  INTERPRETABILITY: Per-Group Entropy Analysis")
    print(f"{'='*70}")
    
    for g_name in GROUP_NAMES:
        h_fail = np.mean(all_entropies[g_name]["fail"])
        h_ok = np.mean(all_entropies[g_name]["ok"])
        delta = h_fail - h_ok
        print(f"  {g_name:>12s}: H(fail)={h_fail:.4f}, H(ok)={h_ok:.4f}, "
              f"Δ={delta:+.4f}  {'↑ More uncertain on failures' if delta > 0 else '↓ Less uncertain on failures'}")
    
    # ── Generate Figures ──
    generate_figures(results, all_entropies, all_meta_test, all_y_test, all_probs_full)
    
    return results


# ─────────────────────────── Figures ───────────────────────────

def generate_figures(results, all_entropies, all_meta_test, all_y_test, all_probs_full):
    """Generate 5 publication-quality figures."""
    
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 150,
    })
    
    # ── Figure 1: Per-Group Entropy Distribution ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = {"ok": "#2196F3", "fail": "#F44336"}
    for ax, g_name in zip(axes, GROUP_NAMES):
        h_ok = all_entropies[g_name]["ok"]
        h_fail = all_entropies[g_name]["fail"]
        ax.hist(h_ok, bins=40, alpha=0.6, color=colors["ok"],
                label="Normal", density=True, edgecolor="white")
        ax.hist(h_fail, bins=40, alpha=0.6, color=colors["fail"],
                label="Failure", density=True, edgecolor="white")
        ax.set_title(f"{g_name} Group", fontweight="bold")
        ax.set_xlabel("Binary Entropy H(p)")
        ax.set_ylabel("Density")
        ax.legend(framealpha=0.9)
    fig.suptitle("Entropy Distribution by Signal Group and Failure Status",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "fig1_entropy_distribution.png"),
                bbox_inches="tight", dpi=200)
    plt.close()
    print("  → Saved fig1_entropy_distribution.png")
    
    # ── Figure 2: Method Comparison Bar Chart ──
    fig, ax = plt.subplots(figsize=(12, 6))
    method_names = list(results.keys())
    metrics = ["auc", "mcc", "f1"]
    x = np.arange(len(method_names))
    width = 0.25
    colors_bar = ["#1976D2", "#388E3C", "#F57C00"]
    
    for i, metric in enumerate(metrics):
        means = [np.mean(results[m][metric]) for m in method_names]
        stds = [np.std(results[m][metric]) for m in method_names]
        bars = ax.bar(x + i * width, means, width, yerr=stds,
                      label=metric.upper(), color=colors_bar[i],
                      edgecolor="white", linewidth=0.5, capsize=3)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(method_names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison: AUC / MCC / F1", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.954, color='gray', linestyle='--', alpha=0.5,
               label="Matzka (2020) RF baseline")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "fig2_method_comparison.png"),
                bbox_inches="tight", dpi=200)
    plt.close()
    print("  → Saved fig2_method_comparison.png")
    
    # ── Figure 3: ROC Curves (last fold) ──
    fig, ax = plt.subplots(figsize=(8, 7))
    y_test_last = all_y_test[-1]
    probs_full_last = all_probs_full[-1]
    
    fpr, tpr, _ = roc_curve(y_test_last, probs_full_last)
    auc_val = roc_auc_score(y_test_last, probs_full_last)
    ax.plot(fpr, tpr, color="#1976D2", lw=2.5,
            label=f"Ours (Full) — AUC={auc_val:.3f}")
    
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Entropy-Calibrated Multi-View Model", fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "fig3_roc_curve.png"),
                bbox_inches="tight", dpi=200)
    plt.close()
    print("  → Saved fig3_roc_curve.png")
    
    # ── Figure 4: Ablation Study (AUC per fold) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    folds_x = np.arange(1, 11)
    ax.plot(folds_x, results["Ours (Full)"]["auc"], "o-",
            color="#1976D2", lw=2, markersize=8, label="Full (with entropy + KL)")
    ax.plot(folds_x, results["Ours (No Entropy)"]["auc"], "s--",
            color="#F44336", lw=2, markersize=8, label="No Entropy (predictions only)")
    ax.fill_between(folds_x,
                     results["Ours (Full)"]["auc"],
                     results["Ours (No Entropy)"]["auc"],
                     alpha=0.15, color="#1976D2")
    ax.set_xlabel("CV Fold")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Ablation Study: Entropy + KL Divergence Contribution", fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(folds_x)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "fig4_ablation_study.png"),
                bbox_inches="tight", dpi=200)
    plt.close()
    print("  → Saved fig4_ablation_study.png")
    
    # ── Figure 5: Meta-Feature Importance / KL Divergence Heatmap ──
    meta_all = np.vstack(all_meta_test)
    y_all = np.concatenate(all_y_test)
    
    meta_names = [
        "p_Thermal", "H_Thermal",
        "p_Mechanical", "H_Mechanical",
        "p_Wear", "H_Wear",
        "KL(T||M)", "KL(T||W)", "KL(M||W)",
        "ΔH(T,M)", "ΔH(T,W)", "ΔH(M,W)"
    ]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    # Mean values by class
    data_by_class = []
    for label, label_name in [(0, "Normal"), (1, "Failure")]:
        mask = y_all == label
        means = meta_all[mask].mean(axis=0)
        data_by_class.append(means)
    
    diff = data_by_class[1] - data_by_class[0]  # failure - normal
    colors_diff = ["#F44336" if d > 0 else "#2196F3" for d in diff]
    
    bars = ax.barh(range(len(meta_names)), diff, color=colors_diff,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(meta_names)))
    ax.set_yticklabels(meta_names)
    ax.set_xlabel("Mean Difference (Failure − Normal)")
    ax.set_title("Meta-Feature Shift: What Changes During Failures?", fontweight="bold")
    ax.axvline(x=0, color="black", lw=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "fig5_meta_feature_shift.png"),
                bbox_inches="tight", dpi=200)
    plt.close()
    print("  → Saved fig5_meta_feature_shift.png")
    
    print("\n  All 5 figures saved successfully.")


# ─────────────────────────── Entry ───────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Entropy-Calibrated Multi-View Failure Prediction")
    print("  Dataset: AI4I 2020 Predictive Maintenance")
    print("=" * 70)
    
    results = run_experiment()
