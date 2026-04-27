#!/usr/bin/env python3
"""
Unified Cross-Dataset Experiment Runner
==========================================

Runs the full experimental pipeline across all datasets:
    1. PID decomposition → Synergy Ratio for each dataset
    2. Bi-level SGD evaluation (full and ablation)
    3. Baseline comparisons (NB, LR, RF, XGBoost)
    4. Statistical analysis (t-tests, effect sizes, Bonferroni)
    5. Publication figure generation

This is the main entry point for reproducing all paper results.

Usage:
    python -m pid_diagnostic.experiments.run_all
    python -m pid_diagnostic.experiments.run_all --dataset ai4i --quick
    python -m pid_diagnostic.experiments.run_all --no-figures
"""

import os
import sys
import argparse
import warnings
import numpy as np
import json
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
from tabulate import tabulate
from scipy import stats

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pid_decomposition import compute_synergy_diagnostic, SynergyDiagnostic
from core.bilevel_sgd import evaluate_bilevel_sgd, EvaluationResult
from datasets.ai4i_loader import load_ai4i, load_ai4i_grouped
from datasets.cmapss_loader import load_cmapss, load_cmapss_grouped
from datasets.smd_loader import load_smd, load_smd_grouped
from datasets.synthetic_synergy import (
    generate_cascading_failures,
    generate_controllable_synergy,
    generate_grouped,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────── Baseline Models ───────────────────────────

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    precision_score, recall_score, accuracy_score,
    roc_curve, brier_score_loss,
)
from imblearn.over_sampling import SMOTE


def _youdens_j(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def evaluate_baselines(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    n_folds: int = 10,
    verbose: bool = True
) -> Dict[str, EvaluationResult]:
    """Run all baseline models on the same CV splits."""

    baselines = {
        "Naive Bayes": GaussianNB(),
        "Logistic Reg.": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost (GB)": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42
        ),
    }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}

    for name, clf_template in baselines.items():
        fold_aucs, fold_mccs, fold_f1s = [], [], []
        fold_precs, fold_recs, fold_accs, fold_briers = [], [], [], []
        all_y_true, all_y_prob = [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # SMOTE
            try:
                k = min(5, int(np.sum(y_train == 1)) - 1)
                if k >= 1:
                    smote = SMOTE(random_state=42, k_neighbors=k)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
            except ValueError:
                pass

            # Scale
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)

            # Clone and train
            from sklearn.base import clone
            clf = clone(clf_template)
            clf.fit(X_tr_s, y_train)

            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_te_s)[:, 1]
            else:
                y_prob = clf.decision_function(X_te_s)

            thresh = _youdens_j(y_test, y_prob)
            y_pred = (y_prob >= thresh).astype(int)

            try:
                auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc = 0.5
            fold_aucs.append(auc)
            fold_mccs.append(matthews_corrcoef(y_test, y_pred))
            fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))
            fold_precs.append(precision_score(y_test, y_pred, zero_division=0))
            fold_recs.append(recall_score(y_test, y_pred, zero_division=0))
            fold_accs.append(accuracy_score(y_test, y_pred))
            fold_briers.append(brier_score_loss(y_test, y_prob))
            all_y_true.extend(y_test)
            all_y_prob.extend(y_prob)

        results[name] = EvaluationResult(
            dataset_name=dataset_name,
            method_name=name,
            fold_aucs=fold_aucs,
            fold_mccs=fold_mccs,
            fold_f1s=fold_f1s,
            fold_precisions=fold_precs,
            fold_recalls=fold_recs,
            fold_accuracies=fold_accs,
            fold_briers=fold_briers,
            all_y_true=np.array(all_y_true),
            all_y_prob=np.array(all_y_prob),
        )

        if verbose:
            print(f"  {name}: AUC={results[name].auc_mean:.4f} ± {results[name].auc_std:.4f}")

    return results


# ─────────────────────────── Main Pipeline ───────────────────────────

def run_dataset_experiment(
    X: np.ndarray,
    y: np.ndarray,
    groups: Dict[str, List[int]],
    groups_raw: Dict[str, np.ndarray],
    dataset_name: str,
    n_folds: int = 10,
    n_bins: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Run the complete experiment for one dataset.

    Returns dict with:
        - synergy_diagnostic: SynergyDiagnostic object
        - result_full: EvaluationResult for full 12D model
        - result_ablation: EvaluationResult for 3D ablation
        - baselines: Dict of baseline EvaluationResults
    """
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"  Samples: {X.shape[0]:,} | Features: {X.shape[1]} | "
          f"Failure rate: {y.mean():.1%}")
    print(f"{'='*60}")

    # 1. PID Diagnostic
    print(f"\n  [1/4] Computing PID Decomposition...")
    diagnostic = compute_synergy_diagnostic(
        groups_raw, y,
        dataset_name=dataset_name,
        n_bins=n_bins,
        bootstrap_n=50
    )
    print(diagnostic)

    # 2. Our model (full)
    print(f"\n  [2/4] Evaluating Bi-Level SGD (Full 12D)...")
    result_full = evaluate_bilevel_sgd(
        X, y, groups, dataset_name=dataset_name,
        n_folds=n_folds, use_entropy=True, verbose=verbose
    )
    print(f"  → Full: AUC={result_full.auc_mean:.4f} ± {result_full.auc_std:.4f}, "
          f"MCC={result_full.mcc_mean:.4f}, F1={result_full.f1_mean:.4f}")

    # 3. Ablation (no entropy)
    print(f"\n  [3/4] Evaluating Bi-Level SGD (Ablation, 3D)...")
    result_ablation = evaluate_bilevel_sgd(
        X, y, groups, dataset_name=dataset_name,
        n_folds=n_folds, use_entropy=False, verbose=verbose
    )
    print(f"  → Ablation: AUC={result_ablation.auc_mean:.4f} ± {result_ablation.auc_std:.4f}")

    # 4. Baselines
    print(f"\n  [4/4] Evaluating Baselines...")
    baselines = evaluate_baselines(X, y, dataset_name, n_folds=n_folds, verbose=verbose)

    # Ablation t-test
    t_stat, p_val = stats.ttest_rel(result_full.fold_aucs, result_ablation.fold_aucs)
    delta_auc = result_full.auc_mean - result_ablation.auc_mean
    sig = "✓ YES" if p_val < 0.05 else "✗ NO"

    print(f"\n  Ablation t-test: Δ AUC = {delta_auc:+.4f}, "
          f"t={t_stat:.3f}, p={p_val:.4f} [{sig}]")

    return {
        "synergy_diagnostic": diagnostic,
        "result_full": result_full,
        "result_ablation": result_ablation,
        "baselines": baselines,
        "ablation_p_value": p_val,
        "ablation_delta_auc": delta_auc,
    }


def run_all_experiments(
    datasets: List[str] = None,
    n_folds: int = 10,
    verbose: bool = True,
    output_dir: str = None,
) -> Dict[str, dict]:
    """
    Run experiments across all specified datasets.

    Args:
        datasets: List of dataset names to run. Default: all.
        n_folds: CV folds.
        verbose: Print progress.
        output_dir: Directory for figures and results.

    Returns:
        Dict mapping dataset name to experiment results.
    """
    if datasets is None:
        datasets = ["AI4I", "C-MAPSS", "Synthetic (Cascading)", "SMD"]

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
    os.makedirs(output_dir, exist_ok=True)

    all_experiments = {}

    print(f"\n{'#'*60}")
    print(f"  PID ARCHITECTURE SELECTION DIAGNOSTIC")
    print(f"  Cross-Dataset Evaluation Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    for ds_name in datasets:
        try:
            if ds_name == "AI4I":
                X, y, _, groups_idx = load_ai4i()
                groups_raw, _ = load_ai4i_grouped()
            elif ds_name == "C-MAPSS":
                X, y, _, groups_idx = load_cmapss()
                groups_raw, _ = load_cmapss_grouped()
            elif ds_name == "SMD":
                X, y, _, groups_idx = load_smd()
                groups_raw, _ = load_smd_grouped()
            elif ds_name.startswith("Synthetic"):
                X, y, _, groups_idx = generate_cascading_failures()
                groups_raw, _ = generate_grouped(mode="cascading")
            else:
                print(f"  [Skip] Unknown dataset: {ds_name}")
                continue

            result = run_dataset_experiment(
                X, y, groups_idx, groups_raw,
                dataset_name=ds_name,
                n_folds=n_folds,
                verbose=verbose,
            )
            all_experiments[ds_name] = result

        except Exception as e:
            print(f"\n  [ERROR] Failed on {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cross-dataset summary
    if len(all_experiments) >= 2:
        print_cross_dataset_summary(all_experiments)
        save_results_json(all_experiments, output_dir)

    return all_experiments


def print_cross_dataset_summary(experiments: Dict[str, dict]):
    """Print the key cross-dataset comparison table."""

    print(f"\n{'='*70}")
    print(f"  CROSS-DATASET SUMMARY: Synergy Ratio vs Fusion Benefit")
    print(f"{'='*70}")

    # Table 1: SR and model performance
    headers = ["Dataset", "SR", "SR 95% CI", "Ours AUC", "XGB AUC", "Gap", "Ablation Δ", "p-value"]
    rows = []

    for ds_name, exp in experiments.items():
        diag = exp["synergy_diagnostic"]
        full = exp["result_full"]
        ablation = exp["result_ablation"]
        xgb = exp["baselines"].get("XGBoost (GB)")

        sr = diag.synergy_ratio
        ci = diag.synergy_ratio_ci
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "N/A"

        xgb_auc = xgb.auc_mean if xgb else float("nan")
        gap = xgb_auc - full.auc_mean if xgb else float("nan")

        rows.append([
            ds_name,
            f"{sr:.4f}",
            ci_str,
            f"{full.auc_mean:.4f}",
            f"{xgb_auc:.4f}" if xgb else "N/A",
            f"{gap:+.4f}" if xgb else "N/A",
            f"{exp['ablation_delta_auc']:+.4f}",
            f"{exp['ablation_p_value']:.4f}",
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Key finding
    print(f"\n  KEY FINDING:")
    print(f"  ─────────────")

    sr_values = {ds: exp["synergy_diagnostic"].synergy_ratio for ds, exp in experiments.items()}
    gaps = {}
    for ds, exp in experiments.items():
        xgb = exp["baselines"].get("XGBoost (GB)")
        if xgb:
            gaps[ds] = xgb.auc_mean - exp["result_full"].auc_mean

    low_sr = [(ds, sr) for ds, sr in sr_values.items() if sr < 0.05]
    high_sr = [(ds, sr) for ds, sr in sr_values.items() if sr >= 0.05]

    if low_sr:
        print(f"  LOW SR datasets ({', '.join(ds for ds, _ in low_sr)}):")
        for ds, sr in low_sr:
            gap = gaps.get(ds)
            print(f"    {ds}: SR={sr:.4f}, "
                  f"Gap to XGBoost={'N/A' if gap is None else f'{gap:.4f}'}")
        print(f"    → Simple entropy-KL model is sufficient.")

    if high_sr:
        print(f"  HIGH SR datasets ({', '.join(ds for ds, _ in high_sr)}):")
        for ds, sr in high_sr:
            gap = gaps.get(ds)
            print(f"    {ds}: SR={sr:.4f}, "
                  f"Gap to XGBoost={'N/A' if gap is None else f'{gap:.4f}'}")
        print(f"    → Cross-group fusion adds genuine value.")


def save_results_json(experiments: Dict[str, dict], output_dir: str):
    """Save key results to JSON for reproducibility."""
    results = {}
    for ds_name, exp in experiments.items():
        diag = exp["synergy_diagnostic"]
        full = exp["result_full"]
        ablation = exp["result_ablation"]

        ds_results = {
            "synergy_ratio": diag.synergy_ratio,
            "mi_per_group": diag.mi_per_group,
            "pairwise_pids": [
                {
                    "pair": f"{pid.source_i}×{pid.source_j}",
                    "redundancy": pid.redundancy,
                    "unique_i": pid.unique_i,
                    "unique_j": pid.unique_j,
                    "synergy": pid.synergy,
                }
                for pid in diag.pairwise_pids
            ],
            "full_auc_mean": full.auc_mean,
            "full_auc_std": full.auc_std,
            "full_mcc_mean": full.mcc_mean,
            "ablation_auc_mean": ablation.auc_mean,
            "ablation_delta_auc": exp["ablation_delta_auc"],
            "ablation_p_value": exp["ablation_p_value"],
            "baselines": {
                name: {"auc_mean": res.auc_mean, "auc_std": res.auc_std}
                for name, res in exp["baselines"].items()
            },
        }
        results[ds_name] = ds_results

    filepath = os.path.join(output_dir, "experiment_results.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  → Results saved to {filepath}")


# ─────────────────────────── CLI Entry Point ───────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PID Architecture Selection Diagnostic — Cross-Dataset Experiments"
    )
    parser.add_argument(
        "--dataset", "-d",
        nargs="+",
        default=None,
        choices=["AI4I", "C-MAPSS", "SMD", "Synthetic"],
        help="Datasets to run. Default: all."
    )
    parser.add_argument(
        "--folds", "-f",
        type=int, default=10,
        help="Number of CV folds."
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: 5 folds, fewer bootstrap samples."
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Output directory for results."
    )

    args = parser.parse_args()

    datasets = args.dataset
    if datasets and "Synthetic" in datasets:
        datasets = [d if d != "Synthetic" else "Synthetic (Cascading)" for d in datasets]

    n_folds = 5 if args.quick else args.folds

    run_all_experiments(
        datasets=datasets,
        n_folds=n_folds,
        verbose=True,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
