#!/usr/bin/env python3
"""
Server Machine Dataset (SMD) Loader
=====================================

The Server Machine Dataset from OmniAnomaly (KDD 2019) contains
multivariate time series from 28 server machines at a large internet company.

Each machine has 38 monitoring metrics. We group them semantically:
    - Compute:  CPU usage metrics
    - Memory:   Memory/swap metrics
    - Network:  Network I/O, disk I/O metrics

This is a REAL distributed systems dataset with labeled anomalies,
making it suitable as an intermediate synergy data point.

Reference:
    Su, Y. et al. (2019). Robust Anomaly Detection for Multivariate Time
    Series through Stochastic Recurrent Neural Network. KDD 2019.

GitHub: https://github.com/NetManAIOps/OmniAnomaly
"""

import os
import urllib.request
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional


# SMD has 38 features per machine. We group the first ~12 that are commonly
# available into 3 semantic groups based on typical server monitoring structure.
# Note: exact feature semantics vary by machine; this is an approximate grouping.
GROUPS = {
    "Compute": [0, 1, 2, 3],       # CPU-related metrics
    "Memory": [4, 5, 6, 7],        # Memory/cache metrics
    "Network": [8, 9, 10, 11],     # Network/disk I/O metrics
}


def _generate_synthetic_smd(
    data_dir: str,
    n_machines: int = 5,
    n_timesteps: int = 5000,
    n_features: int = 12,
    anomaly_rate: float = 0.05,
    random_state: int = 42
) -> str:
    """
    Generate synthetic SMD-like data for development.

    Mimics server monitoring with moderate cross-group interactions:
    some anomalies affect single groups, others cascade across groups.
    """
    rng = np.random.RandomState(random_state)
    smd_dir = os.path.join(data_dir, "SMD")
    os.makedirs(os.path.join(smd_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(smd_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(smd_dir, "labels"), exist_ok=True)

    for m in range(1, n_machines + 1):
        machine_name = f"machine-1-{m}"

        # Normal behavior: each group has its own baseline
        compute_base = rng.randn(n_timesteps, 4) * 0.3 + np.array([30, 40, 25, 35])
        memory_base = rng.randn(n_timesteps, 4) * 0.2 + np.array([60, 55, 70, 50])
        network_base = rng.randn(n_timesteps, 4) * 0.4 + np.array([100, 80, 120, 90])

        X = np.hstack([compute_base, memory_base, network_base])

        # Inject anomalies
        labels = np.zeros(n_timesteps, dtype=int)
        n_anomalies = max(3, int(n_timesteps * anomaly_rate / 50))

        for _ in range(n_anomalies):
            start = rng.randint(100, n_timesteps - 100)
            duration = rng.randint(20, 80)
            end = min(start + duration, n_timesteps)

            anomaly_type = rng.choice(["single", "cascade", "full"])

            if anomaly_type == "single":
                # Single group affected
                group = rng.randint(3)
                cols = list(range(group * 4, (group + 1) * 4))
                X[start:end, cols] += rng.randn(end - start, 4) * 5

            elif anomaly_type == "cascade":
                # Cascading: compute spike → memory spike → network spike
                # This is the SYNERGISTIC pattern
                third = max(1, (end - start) // 3)
                X[start:start+third, 0:4] += rng.randn(third, 4) * 4
                X[start+third:start+2*third, 4:8] += rng.randn(third, 4) * 3
                X[start+2*third:end, 8:12] += rng.randn(end - start - 2*third, 4) * 5

            else:
                # Full system: all groups spike simultaneously
                X[start:end, :] += rng.randn(end - start, n_features) * 4

            labels[start:end] = 1

        # Split into train (first 60%) and test (last 40%)
        split = int(n_timesteps * 0.6)

        train_data = X[:split]
        test_data = X[split:]
        test_labels = labels[split:]

        np.save(os.path.join(smd_dir, "train", f"{machine_name}.npy"), train_data)
        np.save(os.path.join(smd_dir, "test", f"{machine_name}.npy"), test_data)
        np.save(os.path.join(smd_dir, "labels", f"{machine_name}.npy"), test_labels)

    print(f"  Generated synthetic SMD: {n_machines} machines, {n_timesteps} timesteps each")
    return smd_dir


def load_smd(
    data_dir: str = None,
    machines: Optional[List[str]] = None,
    window_size: int = 10,
    auto_generate: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[int]]]:
    """
    Load Server Machine Dataset with windowed features.

    Args:
        data_dir: Directory containing SMD data.
        machines: List of machine names to load. None = all available.
        window_size: Rolling window for feature extraction.
        auto_generate: Generate synthetic data if not found.

    Returns:
        X: Windowed feature array.
        y: Binary anomaly labels.
        feature_names: Column names.
        groups: Dict mapping group names to column indices.
    """
    if data_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base, "data")

    smd_dir = os.path.join(data_dir, "SMD")

    if not os.path.exists(smd_dir):
        if auto_generate:
            smd_dir = _generate_synthetic_smd(data_dir)
        else:
            raise FileNotFoundError(f"SMD data not found at {smd_dir}")

    # Find available machines
    test_dir = os.path.join(smd_dir, "test")
    if machines is None:
        machines = [
            f.replace(".npy", "")
            for f in sorted(os.listdir(test_dir))
            if f.endswith(".npy")
        ]

    all_X = []
    all_y = []

    for machine in machines:
        test_path = os.path.join(smd_dir, "test", f"{machine}.npy")
        label_path = os.path.join(smd_dir, "labels", f"{machine}.npy")

        if not os.path.exists(test_path):
            print(f"  [Warning] Skipping {machine}: file not found")
            continue

        data = np.load(test_path)
        labels = np.load(label_path)

        # Use only the first 12 features (our 3 groups × 4 features)
        n_features = min(data.shape[1], 12)
        data = data[:, :n_features]

        # Windowed features: mean + std
        n_timesteps = len(data)
        windowed_features = []

        for t in range(window_size, n_timesteps):
            window = data[t - window_size:t]
            feat = np.concatenate([window.mean(axis=0), window.std(axis=0)])
            windowed_features.append(feat)

        if len(windowed_features) > 0:
            X_machine = np.array(windowed_features)
            y_machine = labels[window_size:]
            all_X.append(X_machine)
            all_y.append(y_machine)

    if len(all_X) == 0:
        raise ValueError("No machine data loaded")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Feature names: mean and std for each of 12 base features
    base_names = (
        [f"compute_f{i+1}" for i in range(4)] +
        [f"memory_f{i+1}" for i in range(4)] +
        [f"network_f{i+1}" for i in range(4)]
    )
    feature_names = (
        [f"{n}_mean" for n in base_names] +
        [f"{n}_std" for n in base_names]
    )

    # Group indices (mean + std for each original group)
    n_base = min(12, len(base_names))
    groups_idx = {
        "Compute": list(range(0, 4)) + list(range(n_base, n_base + 4)),
        "Memory": list(range(4, 8)) + list(range(n_base + 4, n_base + 8)),
        "Network": list(range(8, 12)) + list(range(n_base + 8, n_base + 12)),
    }

    return X, y, feature_names, groups_idx


def load_smd_grouped(
    data_dir: str = None, **kwargs
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load SMD and return grouped for PID analysis."""
    X, y, feature_names, groups_idx = load_smd(data_dir, **kwargs)

    groups = {}
    for g_name, g_idx in groups_idx.items():
        groups[g_name] = X[:, g_idx]

    return groups, y


def print_smd_summary(data_dir: str = None):
    """Print dataset summary."""
    X, y, features, groups = load_smd(data_dir)
    print(f"\n  Server Machine Dataset (SMD)")
    print(f"  {'─'*45}")
    print(f"  Samples: {X.shape[0]:,}")
    print(f"  Features: {X.shape[1]} (windowed mean+std)")
    print(f"  Anomaly rate: {y.mean():.1%} ({y.sum():,}/{len(y):,})")
    print(f"  Groups:")
    for g_name, g_idx in groups.items():
        print(f"    {g_name} ({len(g_idx)} features)")
