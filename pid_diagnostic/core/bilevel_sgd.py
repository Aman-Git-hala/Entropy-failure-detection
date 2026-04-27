#!/usr/bin/env python3
"""
Bi-Level SGD Pipeline (Dataset-Agnostic)
==========================================

Refactored version of the bi-level SGD classifier that accepts arbitrary
datasets and group definitions. Produces calibrated probabilities and
the full 12D entropy-KL meta-feature vector.

This is the core model evaluated across all datasets in the paper.

Architecture:
    For each group G_i:
        SGD_i: X_Gi → p_i (calibrated probability)
        H_i = binary_entropy(p_i)

    Meta-features:
        z = [p₁, H₁, p₂, H₂, p₃, H₃, KL₁₂, KL₁₃, KL₂₃, ΔH₁₂, ΔH₁₃, ΔH₂₃]

    Outer classifier:
        SGD_outer: z → final prediction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    precision_score, recall_score, accuracy_score,
    roc_curve, brier_score_loss
)
from imblearn.over_sampling import SMOTE

from .entropy_features import build_12d_meta_vector


@dataclass
class FoldResult:
    """Results from a single CV fold."""
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    meta_features: np.ndarray
    threshold: float
    group_probs: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Aggregate results from full cross-validation."""
    dataset_name: str
    method_name: str
    # Per-fold metrics
    fold_aucs: List[float]
    fold_mccs: List[float]
    fold_f1s: List[float]
    fold_precisions: List[float]
    fold_recalls: List[float]
    fold_accuracies: List[float]
    fold_briers: List[float]
    # Aggregated arrays (for ROC curves, etc.)
    all_y_true: np.ndarray
    all_y_prob: np.ndarray
    all_meta: Optional[np.ndarray] = None
    meta_feature_names: Optional[List[str]] = None
    # Per-group probabilities for interpretability analysis
    all_group_probs: Optional[Dict[str, np.ndarray]] = None

    @property
    def auc_mean(self): return float(np.mean(self.fold_aucs))
    @property
    def auc_std(self): return float(np.std(self.fold_aucs))
    @property
    def mcc_mean(self): return float(np.mean(self.fold_mccs))
    @property
    def mcc_std(self): return float(np.std(self.fold_mccs))
    @property
    def f1_mean(self): return float(np.mean(self.fold_f1s))
    @property
    def f1_std(self): return float(np.std(self.fold_f1s))
    @property
    def precision_mean(self): return float(np.mean(self.fold_precisions))
    @property
    def recall_mean(self): return float(np.mean(self.fold_recalls))
    @property
    def accuracy_mean(self): return float(np.mean(self.fold_accuracies))
    @property
    def brier_mean(self): return float(np.mean(self.fold_briers))

    def summary_row(self) -> list:
        """Return a row for tabulate."""
        return [
            self.method_name,
            f"{self.auc_mean:.4f} ± {self.auc_std:.4f}",
            f"{self.mcc_mean:.4f} ± {self.mcc_std:.4f}",
            f"{self.f1_mean:.4f} ± {self.f1_std:.4f}",
            f"{self.precision_mean:.4f}",
            f"{self.recall_mean:.4f}",
            f"{self.brier_mean:.4f}",
        ]


def _tune_alpha(X: np.ndarray, y: np.ndarray, n_folds: int = 3) -> float:
    """Grid-search alpha for SGDClassifier."""
    alphas = [1e-5, 1e-4, 1e-3, 1e-2]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    best_alpha, best_score = alphas[0], -1.0

    for a in alphas:
        scores = []
        for tr, va in skf.split(X, y):
            clf = SGDClassifier(
                loss="log_loss", penalty="elasticnet",
                alpha=a, l1_ratio=0.15, max_iter=2000,
                class_weight="balanced", random_state=42
            )
            clf.fit(X[tr], y[tr])
            try:
                scores.append(roc_auc_score(y[va], clf.decision_function(X[va])))
            except ValueError:
                scores.append(0.5)
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = a

    return best_alpha


def _youdens_j_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold maximizing Youden's J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def _train_group_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a single inner-loop SGD classifier for one sensor group.

    Returns calibrated probabilities for train and test sets.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    if alpha is None:
        alpha = _tune_alpha(X_tr_s, y_train)

    base = SGDClassifier(
        loss="log_loss", penalty="elasticnet",
        alpha=alpha, l1_ratio=0.15, max_iter=2000,
        class_weight="balanced", random_state=42
    )

    min_class = min(np.sum(y_train == 0), np.sum(y_train == 1))

    if min_class >= 5 and len(y_train) >= 30:
        n_cv = min(5, min_class)
        cal = CalibratedClassifierCV(base, method="isotonic", cv=n_cv)
        cal.fit(X_tr_s, y_train)
        p_train = cal.predict_proba(X_tr_s)[:, 1]
        p_test = cal.predict_proba(X_te_s)[:, 1]
    else:
        base.fit(X_tr_s, y_train)
        d_tr = base.decision_function(X_tr_s)
        d_te = base.decision_function(X_te_s)
        p_train = 1 / (1 + np.exp(-d_tr))
        p_test = 1 / (1 + np.exp(-d_te))

    return p_train, p_test


def bilevel_sgd_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups: Dict[str, List[int]],
    feature_names: Optional[List[str]] = None,
    use_entropy: bool = True
) -> FoldResult:
    """
    Run one fold of the bi-level SGD pipeline.

    Args:
        X_train, X_test: Feature arrays.
        y_train, y_test: Labels.
        groups: Dict mapping group names to lists of column indices.
        feature_names: Column names (for debugging).
        use_entropy: If True, use full 12D vector. If False, ablation (3D).

    Returns:
        FoldResult with predictions, meta-features, group probabilities.
    """
    group_names = list(groups.keys())
    group_indices = list(groups.values())

    # Inner loop: train per-group models
    train_probs = []
    test_probs = []
    group_probs_test = {}

    for g_name, g_idx in zip(group_names, group_indices):
        X_tr_g = X_train[:, g_idx]
        X_te_g = X_test[:, g_idx]
        p_tr, p_te = _train_group_model(X_tr_g, y_train, X_te_g)
        train_probs.append(p_tr)
        test_probs.append(p_te)
        group_probs_test[g_name] = p_te

    if use_entropy:
        # Full 12D meta-features
        meta_train, meta_names = build_12d_meta_vector(train_probs, group_names)
        meta_test, _ = build_12d_meta_vector(test_probs, group_names)
    else:
        # Ablation: predictions only (3D)
        meta_train = np.column_stack(train_probs)
        meta_test = np.column_stack(test_probs)
        meta_names = [f"p_{g}" for g in group_names]

    # Outer loop: meta-classifier
    meta_scaler = StandardScaler()
    meta_tr_s = meta_scaler.fit_transform(meta_train)
    meta_te_s = meta_scaler.transform(meta_test)

    outer_alpha = _tune_alpha(meta_tr_s, y_train)
    outer_base = SGDClassifier(
        loss="log_loss", penalty="elasticnet",
        alpha=outer_alpha, l1_ratio=0.15, max_iter=2000,
        class_weight="balanced", random_state=42
    )

    min_class = min(np.sum(y_train == 0), np.sum(y_train == 1))
    if min_class >= 5 and len(y_train) >= 30:
        n_cv = min(5, min_class)
        outer_cal = CalibratedClassifierCV(outer_base, method="isotonic", cv=n_cv)
        outer_cal.fit(meta_tr_s, y_train)
        y_prob_tr = outer_cal.predict_proba(meta_tr_s)[:, 1]
        y_prob = outer_cal.predict_proba(meta_te_s)[:, 1]
    else:
        outer_base.fit(meta_tr_s, y_train)
        y_prob_tr = 1 / (1 + np.exp(-outer_base.decision_function(meta_tr_s)))
        y_prob = 1 / (1 + np.exp(-outer_base.decision_function(meta_te_s)))

    threshold = _youdens_j_threshold(y_train, y_prob_tr)
    y_pred = (y_prob >= threshold).astype(int)

    return FoldResult(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        meta_features=meta_test,
        threshold=threshold,
        group_probs=group_probs_test,
    )


def evaluate_bilevel_sgd(
    X: np.ndarray,
    y: np.ndarray,
    groups: Dict[str, List[int]],
    dataset_name: str = "Dataset",
    n_folds: int = 10,
    use_entropy: bool = True,
    use_smote: bool = True,
    verbose: bool = True
) -> EvaluationResult:
    """
    Run full k-fold cross-validated evaluation of the bi-level SGD pipeline.

    Args:
        X: Feature array, shape (n_samples, n_features).
        y: Binary labels, shape (n_samples,).
        groups: Dict mapping group names to column index lists.
        dataset_name: For reporting.
        n_folds: Number of CV folds.
        use_entropy: If True, full 12D. If False, ablation 3D.
        use_smote: Apply SMOTE inside each fold.
        verbose: Print per-fold progress.

    Returns:
        EvaluationResult with all metrics and aggregated predictions.
    """
    method_name = f"Ours ({'Full' if use_entropy else 'No Entropy'})"
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_aucs, fold_mccs, fold_f1s = [], [], []
    fold_precs, fold_recs, fold_accs, fold_briers = [], [], [], []
    all_y_true, all_y_prob = [], []
    all_meta = []
    all_group_probs = {g: [] for g in groups.keys()}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # SMOTE inside fold
        if use_smote:
            try:
                k = min(5, int(np.sum(y_train == 1)) - 1)
                if k >= 1:
                    smote = SMOTE(random_state=42, k_neighbors=k)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
            except ValueError:
                pass

        result = bilevel_sgd_fold(
            X_train, y_train, X_test, y_test,
            groups, use_entropy=use_entropy
        )

        # Metrics
        try:
            auc = roc_auc_score(result.y_true, result.y_prob)
        except ValueError:
            auc = 0.5
        mcc = matthews_corrcoef(result.y_true, result.y_pred)
        f1 = f1_score(result.y_true, result.y_pred, zero_division=0)
        prec = precision_score(result.y_true, result.y_pred, zero_division=0)
        rec = recall_score(result.y_true, result.y_pred, zero_division=0)
        acc = accuracy_score(result.y_true, result.y_pred)
        brier = brier_score_loss(result.y_true, result.y_prob)

        fold_aucs.append(auc)
        fold_mccs.append(mcc)
        fold_f1s.append(f1)
        fold_precs.append(prec)
        fold_recs.append(rec)
        fold_accs.append(acc)
        fold_briers.append(brier)

        all_y_true.extend(result.y_true)
        all_y_prob.extend(result.y_prob)
        all_meta.append(result.meta_features)

        for g_name, g_probs in result.group_probs.items():
            all_group_probs[g_name].extend(g_probs)

        if verbose:
            print(f"    Fold {fold+1:2d}/{n_folds}: "
                  f"AUC={auc:.4f}  MCC={mcc:.4f}  F1={f1:.4f}")

    # Build group names for meta-features
    group_names = list(groups.keys())
    if use_entropy:
        meta_names = []
        for g in group_names:
            meta_names.extend([f"p_{g}", f"H_{g}"])
        for i, j in [(0,1), (0,2), (1,2)]:
            meta_names.append(f"KL({group_names[i]}||{group_names[j]})")
        for i, j in [(0,1), (0,2), (1,2)]:
            meta_names.append(f"ΔH({group_names[i]},{group_names[j]})")
    else:
        meta_names = [f"p_{g}" for g in group_names]

    return EvaluationResult(
        dataset_name=dataset_name,
        method_name=method_name,
        fold_aucs=fold_aucs,
        fold_mccs=fold_mccs,
        fold_f1s=fold_f1s,
        fold_precisions=fold_precs,
        fold_recalls=fold_recs,
        fold_accuracies=fold_accs,
        fold_briers=fold_briers,
        all_y_true=np.array(all_y_true),
        all_y_prob=np.array(all_y_prob),
        all_meta=np.vstack(all_meta),
        meta_feature_names=meta_names,
        all_group_probs={g: np.array(v) for g, v in all_group_probs.items()},
    )
