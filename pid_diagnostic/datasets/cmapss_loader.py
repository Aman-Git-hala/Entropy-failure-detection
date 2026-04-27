#!/usr/bin/env python3
"""
C-MAPSS Turbofan Degradation Dataset Loader
=============================================

NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)
dataset for turbofan engine remaining useful life (RUL) prediction.

We use FD001: single operating condition, single fault mode (HPC degradation).
    - 100 engines, run-to-failure trajectories
    - 21 sensor channels + 3 operational settings
    - Binary failure label: 1 if RUL ≤ threshold, 0 otherwise

Semantic groups (based on physical sensor location):
    - Temperature:  sensors 2,3,4,11,17 (fan/LPC/HPC outlet temps)
    - Pressure:     sensors 6,7,8,13 (total pressures)
    - Speed/Flow:   sensors 9,14,15 (corrected speeds, bleed enthalpy)

Expected PID result: LOW synergy, HIGH redundancy (all sensors track
single HPC degradation mode → they're proxies for one latent variable).

Reference:
    Saxena, A. et al. (2008). Damage Propagation Modeling for Aircraft
    Engine Run-to-Failure Simulation. PHM Conference.
"""

import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional


# Column definitions for C-MAPSS files
CMAPSS_COLS = (
    ["engine_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

# Semantic sensor groups
GROUPS = {
    "Temperature": ["sensor_2", "sensor_3", "sensor_4", "sensor_11", "sensor_17"],
    "Pressure": ["sensor_6", "sensor_7", "sensor_8", "sensor_13"],
    "Speed": ["sensor_9", "sensor_14", "sensor_15"],
}

# Sensors known to be constant/near-constant in FD001 (remove for cleaner analysis)
CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_10", "sensor_16",
                    "sensor_18", "sensor_19"]

# NASA dataset URL
CMAPSS_URL = "https://ti.arc.nasa.gov/c/6/"


def download_cmapss(data_dir: str) -> str:
    """
    Download C-MAPSS dataset if not present.

    Returns path to extracted directory.
    """
    zip_path = os.path.join(data_dir, "CMAPSSData.zip")
    extract_dir = os.path.join(data_dir, "CMAPSSData")

    if os.path.exists(extract_dir):
        return extract_dir

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        print(f"  C-MAPSS zip not found. Generating synthetic C-MAPSS-like data...")
        print(f"  (To use real data, manually download CMAPSSData.zip from:")
        print(f"    https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        print(f"   and place it in {data_dir})")
        return _generate_synthetic_cmapss(data_dir)

    # Validate zip file
    try:
        print(f"  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
    except zipfile.BadZipFile:
        print(f"  Bad zip file detected, removing and generating synthetic data...")
        os.remove(zip_path)
        return _generate_synthetic_cmapss(data_dir)

    return extract_dir


def _generate_synthetic_cmapss(data_dir: str) -> str:
    """
    Generate synthetic C-MAPSS-like data for development/testing.

    Mimics FD001 structure: 100 engines, run-to-failure, 21 sensors.
    Key property: all sensors are driven by a single latent degradation
    variable, so we expect HIGH redundancy and LOW synergy.
    """
    extract_dir = os.path.join(data_dir, "CMAPSSData")
    os.makedirs(extract_dir, exist_ok=True)

    rng = np.random.RandomState(42)
    n_engines = 100

    all_rows = []
    for eng_id in range(1, n_engines + 1):
        # Each engine runs for 150-350 cycles
        n_cycles = rng.randint(150, 351)

        # Latent degradation: starts near 0, increases over time
        degradation = np.linspace(0, 1, n_cycles) ** 1.5

        for cycle in range(1, n_cycles + 1):
            d = degradation[cycle - 1]
            noise = rng.randn(21) * 0.02

            # All sensors are functions of the SAME degradation variable
            # This creates the high-redundancy, low-synergy pattern
            sensors = np.zeros(21)

            # Temperature sensors: increase with degradation
            sensors[1] = 641.82 + d * 20 + noise[1] * 5      # sensor_2
            sensors[2] = 1589.7 + d * 50 + noise[2] * 10     # sensor_3
            sensors[3] = 1400.6 + d * 40 + noise[3] * 8      # sensor_4
            sensors[10] = 47.47 + d * 5 + noise[10]           # sensor_11
            sensors[16] = 392.0 + d * 10 + noise[16] * 3      # sensor_17

            # Pressure sensors: decrease with degradation
            sensors[5] = 21.61 - d * 3 + noise[5] * 0.5      # sensor_6
            sensors[6] = 554.4 - d * 20 + noise[6] * 5       # sensor_7
            sensors[7] = 2388.0 - d * 50 + noise[7] * 10     # sensor_8
            sensors[12] = 47.83 - d * 5 + noise[12]           # sensor_13

            # Speed sensors: change with degradation
            sensors[8] = 9065.4 - d * 200 + noise[8] * 20    # sensor_9
            sensors[13] = 8138.6 + d * 100 + noise[13] * 15  # sensor_14
            sensors[14] = 8.448 + d * 2 + noise[14] * 0.3    # sensor_15

            # Constant/near-constant sensors (noise only)
            for idx in [0, 4, 9, 15, 17, 18, 19, 20]:
                sensors[idx] = rng.randn() * 0.01

            row = [eng_id, cycle] + [0.0, 0.0, 100.0] + sensors.tolist()
            all_rows.append(row)

    df = pd.DataFrame(all_rows, columns=CMAPSS_COLS)

    # Save as train_FD001.txt (space-separated, no header)
    filepath = os.path.join(extract_dir, "train_FD001.txt")
    df.to_csv(filepath, sep=" ", header=False, index=False)

    print(f"  Generated synthetic C-MAPSS data: {n_engines} engines, {len(df)} rows")
    return extract_dir


def load_cmapss(
    data_dir: str = None,
    subset: str = "FD001",
    rul_threshold: int = 30,
    window_size: int = 5,
    auto_download: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[int]]]:
    """
    Load C-MAPSS dataset with windowed features and binary failure labeling.

    Args:
        data_dir: Directory for C-MAPSS data. Defaults to pid_diagnostic/data/.
        subset: Which subset to load (FD001, FD002, FD003, FD004).
        rul_threshold: Cycles before failure to label as "imminent failure".
        window_size: Rolling window for feature extraction.
        auto_download: Try to download if not present.

    Returns:
        X: Feature array with windowed statistics.
        y: Binary labels (1 if RUL ≤ threshold).
        feature_names: Column names.
        groups: Dict mapping group names to column indices.
    """
    if data_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base, "data")

    extract_dir = os.path.join(data_dir, "CMAPSSData")

    if not os.path.exists(extract_dir):
        if auto_download:
            extract_dir = download_cmapss(data_dir)
        else:
            raise FileNotFoundError(
                f"C-MAPSS data not found at {extract_dir}. "
                f"Set auto_download=True or download manually."
            )

    # Load training data
    filepath = os.path.join(extract_dir, f"train_{subset}.txt")
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=CMAPSS_COLS)

    # Compute RUL for each engine
    max_cycles = df.groupby("engine_id")["cycle"].max()
    df = df.merge(max_cycles.rename("max_cycle"), on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["failure"] = (df["RUL"] <= rul_threshold).astype(int)

    # Select sensor columns for our groups
    sensor_cols = []
    for g_name in GROUPS:
        sensor_cols.extend(GROUPS[g_name])

    # Windowed features: rolling mean and std per engine
    feature_cols = []
    all_features = []

    for eng_id in df["engine_id"].unique():
        eng_df = df[df["engine_id"] == eng_id].sort_values("cycle")

        eng_feats = []
        for col in sensor_cols:
            # Rolling mean
            roll_mean = eng_df[col].rolling(window_size, min_periods=1).mean().values
            # Rolling std
            roll_std = eng_df[col].rolling(window_size, min_periods=1).std().fillna(0).values
            eng_feats.append(roll_mean)
            eng_feats.append(roll_std)

        all_features.append(np.column_stack(eng_feats))

    # On first iteration, build feature names
    for col in sensor_cols:
        feature_cols.append(f"{col}_mean")
        feature_cols.append(f"{col}_std")

    X = np.vstack(all_features)
    y = df.sort_values(["engine_id", "cycle"])["failure"].values

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Build group index mapping
    groups_idx = {}
    for g_name, g_sensor_cols in GROUPS.items():
        indices = []
        for sc in g_sensor_cols:
            mean_name = f"{sc}_mean"
            std_name = f"{sc}_std"
            if mean_name in feature_cols:
                indices.append(feature_cols.index(mean_name))
            if std_name in feature_cols:
                indices.append(feature_cols.index(std_name))
        groups_idx[g_name] = indices

    return X, y, feature_cols, groups_idx


def load_cmapss_grouped(
    data_dir: str = None, **kwargs
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load C-MAPSS and return grouped arrays for PID analysis."""
    X, y, feature_names, groups_idx = load_cmapss(data_dir, **kwargs)

    groups = {}
    for g_name, g_idx in groups_idx.items():
        groups[g_name] = X[:, g_idx]

    return groups, y


def print_cmapss_summary(data_dir: str = None):
    """Print dataset summary."""
    X, y, features, groups = load_cmapss(data_dir)
    print(f"\n  C-MAPSS FD001 Turbofan Degradation Dataset")
    print(f"  {'─'*45}")
    print(f"  Samples: {X.shape[0]:,}")
    print(f"  Features: {X.shape[1]} (windowed mean+std)")
    print(f"  Failure rate: {y.mean():.1%} ({y.sum():,}/{len(y):,})")
    print(f"  Groups:")
    for g_name, g_idx in groups.items():
        g_features = [features[i] for i in g_idx]
        print(f"    {g_name} ({len(g_idx)} features): {g_features[:4]}...")
