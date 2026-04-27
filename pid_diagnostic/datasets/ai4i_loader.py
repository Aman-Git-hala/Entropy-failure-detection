#!/usr/bin/env python3
"""
AI4I 2020 Predictive Maintenance Dataset Loader
=================================================

Loads the AI4I 2020 dataset (Matzka, 2020) and defines semantic sensor groups:
    - Thermal:    Air temperature, Process temperature
    - Mechanical: Rotational speed, Torque
    - Wear:       Tool wear, Product type

Dataset: 10,000 samples, 6 features, binary failure label (~3.4% failure rate).

Reference:
    Matzka, S. (2020). Explainable Artificial Intelligence for Predictive
    Maintenance Applications. IEEE 3rd Int. Conf. AI for Industries, pp. 391-395.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


# Semantic sensor groups with physical interpretation
GROUPS = {
    "Thermal": ["Air temperature [K]", "Process temperature [K]"],
    "Mechanical": ["Rotational speed [rpm]", "Torque [Nm]"],
    "Wear": ["Tool wear [min]", "Type_encoded"],
}


def load_ai4i(
    data_dir: str = None,
    filename: str = "ai4i_2020.csv"
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[int]]]:
    """
    Load and preprocess the AI4I 2020 dataset.

    Args:
        data_dir: Directory containing the CSV file.
                  Defaults to otel_failure_prediction/ sibling dir.
        filename: CSV filename.

    Returns:
        X: Feature array, shape (n_samples, 6).
        y: Binary failure labels, shape (n_samples,).
        feature_names: List of 6 feature column names.
        groups: Dict mapping group names to column index lists.
    """
    if data_dir is None:
        # Default: look in sibling otel_failure_prediction directory
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent = os.path.dirname(base)
        data_dir = os.path.join(parent, "otel_failure_prediction")

    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"AI4I dataset not found at {filepath}. "
            f"Download from UCI ML Repository or run otel_failure_prediction/download_dataset.py"
        )

    df = pd.read_csv(filepath)

    # Encode product type: L=0, M=1, H=2
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["Type_encoded"] = le.fit_transform(df["Type"])

    # Collect all feature columns in group order
    feature_cols = []
    for g_name in GROUPS:
        feature_cols.extend(GROUPS[g_name])

    X = df[feature_cols].values.astype(float)
    y = df["Machine failure"].values.astype(int)

    # Build column index mapping for groups
    groups_idx = {}
    for g_name, g_cols in GROUPS.items():
        groups_idx[g_name] = [feature_cols.index(c) for c in g_cols]

    return X, y, feature_cols, groups_idx


def load_ai4i_grouped(
    data_dir: str = None
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Load AI4I and return feature arrays grouped for PID analysis.

    Returns:
        groups: Dict mapping group names to feature arrays.
        y: Binary labels.
    """
    X, y, feature_names, groups_idx = load_ai4i(data_dir)

    groups = {}
    for g_name, g_idx in groups_idx.items():
        groups[g_name] = X[:, g_idx]

    return groups, y


def print_ai4i_summary(data_dir: str = None):
    """Print dataset summary statistics."""
    X, y, features, groups = load_ai4i(data_dir)

    print(f"\n  AI4I 2020 Predictive Maintenance Dataset")
    print(f"  {'─'*45}")
    print(f"  Samples: {X.shape[0]:,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Failure rate: {y.mean():.1%} ({y.sum()}/{len(y)})")
    print(f"  Groups:")
    for g_name, g_idx in groups.items():
        g_features = [features[i] for i in g_idx]
        print(f"    {g_name}: {g_features}")
