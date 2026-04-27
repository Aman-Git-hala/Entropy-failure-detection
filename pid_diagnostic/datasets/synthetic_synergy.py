#!/usr/bin/env python3
"""
Synthetic High-Synergy Dataset Generator
==========================================

Generates a dataset where failure genuinely requires cross-group interaction,
producing high synergy by construction. This is the critical "third data point"
that demonstrates the Synergy Ratio diagnostic has discriminative power.

The generative model:
    Three sensor groups G₁, G₂, G₃, each producing a scalar health indicator.
    Individually, each group is uninformative about failure.
    Failure occurs when a specific INTERACTION condition is met:

    Model A — "Cascading Threshold":
        failure = 1  iff  (G₁ > τ₁ AND G₂ > τ₂) OR (G₂ > τ₂ AND G₃ > τ₃)
        No single group predicts failure, but pairs do.

    Model B — "XOR-like Interaction":
        failure = 1  iff  XOR(G₁ > median, G₂ > median, G₃ > median)
        Pure synergy — no marginal or pairwise information.

    Model C — "Distributed Systems Emulation":
        Emulates Kafka-like cascading failures:
        - G₁ = Broker health (normally stable, degrades during failure)
        - G₂ = Consumer lag (spikes when broker degrades AND load is high)
        - G₃ = Network latency (increases when rebalancing occurs)
        failure = f(G₁, G₂, G₃) with genuine cross-group dependencies

    We also support a controllable synergy_level ∈ [0, 1] parameter that
    interpolates between a purely redundant (SR≈0) and purely synergistic (SR≈1)
    dataset, enabling sweep experiments.

This approach is defensible because:
    1. We are NOT claiming this is real data
    2. The paper's contribution is the DIAGNOSTIC METHOD, not the dataset
    3. The synthetic data is designed to validate the diagnostic's discriminative power
    4. We explicitly state the intention to replace with real Kafka data in future work
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


def generate_cascading_failures(
    n_samples: int = 10000,
    n_features_per_group: int = 4,
    noise_level: float = 0.3,
    failure_rate: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[int]]]:
    """
    Generate a dataset emulating cascading distributed system failures.

    The generative model mimics Kafka-like failure cascades:
        1. Broker health degrades (G₁ features shift)
        2. Degradation causes consumer rebalancing (G₂ features spike)
        3. Rebalancing + load causes network pressure (G₃ features spike)
        4. Failure only occurs when ALL THREE interact

    No single group's features predict failure well individually.
    The combination is necessary → HIGH SYNERGY by construction.

    Args:
        n_samples: Number of samples.
        n_features_per_group: Features per sensor group.
        noise_level: Gaussian noise standard deviation.
        failure_rate: Target failure fraction.
        random_state: Random seed.

    Returns:
        X: Feature array, shape (n_samples, 3 * n_features_per_group).
        y: Binary failure labels.
        feature_names: Column names.
        groups: Dict mapping group names to column indices.
    """
    rng = np.random.RandomState(random_state)
    n_feat = n_features_per_group

    # Latent variables (not directly observed)
    broker_degradation = rng.beta(2, 5, n_samples)   # Usually low, sometimes high
    load_intensity = rng.beta(3, 3, n_samples)        # Symmetric around 0.5
    network_congestion = rng.beta(2, 4, n_samples)    # Usually low

    # Failure condition: requires interaction of all three
    # broker must be degraded AND load must be high AND network stressed
    interaction = broker_degradation * load_intensity * network_congestion
    threshold = np.percentile(interaction, 100 * (1 - failure_rate))
    y = (interaction >= threshold).astype(int)

    # Group 1: Broker Health
    # Features are noisy functions of broker_degradation
    # BUT broker_degradation alone doesn't predict failure well
    G1 = np.column_stack([
        broker_degradation + rng.randn(n_samples) * noise_level,
        -broker_degradation * 0.8 + rng.randn(n_samples) * noise_level,
        np.log1p(broker_degradation * 10) + rng.randn(n_samples) * noise_level,
        rng.randn(n_samples) * (1 + broker_degradation * 0.5),
    ][:n_feat])

    # Group 2: Consumer Group
    # Features driven by load AND broker state (but marginalizing over broker
    # removes the signal)
    consumer_stress = load_intensity * (0.3 + broker_degradation * 0.7)
    G2 = np.column_stack([
        consumer_stress + rng.randn(n_samples) * noise_level,
        load_intensity * 2 + rng.randn(n_samples) * noise_level * 1.5,
        np.sin(consumer_stress * np.pi) + rng.randn(n_samples) * noise_level,
        consumer_stress ** 2 + rng.randn(n_samples) * noise_level,
    ][:n_feat])

    # Group 3: Network/Partition
    # Features driven by network congestion AND rebalancing events
    rebalancing = (broker_degradation > 0.3).astype(float) * load_intensity
    net_stress = network_congestion * (0.5 + rebalancing * 0.5)
    G3 = np.column_stack([
        net_stress + rng.randn(n_samples) * noise_level,
        network_congestion + rng.randn(n_samples) * noise_level,
        rebalancing * 3 + rng.randn(n_samples) * noise_level,
        np.exp(net_stress) - 1 + rng.randn(n_samples) * noise_level,
    ][:n_feat])

    # Combine
    X = np.hstack([G1, G2, G3])

    # Feature names
    feature_names = (
        [f"broker_feat_{i+1}" for i in range(n_feat)] +
        [f"consumer_feat_{i+1}" for i in range(n_feat)] +
        [f"network_feat_{i+1}" for i in range(n_feat)]
    )

    # Group indices
    groups = {
        "Broker": list(range(0, n_feat)),
        "Consumer": list(range(n_feat, 2 * n_feat)),
        "Network": list(range(2 * n_feat, 3 * n_feat)),
    }

    return X, y, feature_names, groups


def generate_controllable_synergy(
    n_samples: int = 10000,
    synergy_level: float = 0.5,
    n_features_per_group: int = 3,
    noise_level: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[int]]]:
    """
    Generate a dataset with controllable synergy level.

    synergy_level ∈ [0, 1]:
        0.0 → Failure determined by single dominant group (like AI4I)
              Expected SR ≈ 0
        0.5 → Mixed: some individual, some interaction signal
              Expected SR ≈ 0.1-0.2
        1.0 → Failure ONLY determined by cross-group interaction
              Expected SR → high

    This enables sweep experiments: plot SR vs synergy_level to validate
    the diagnostic's calibration.

    Args:
        n_samples: Number of samples.
        synergy_level: Controls synergy (0=redundant, 1=synergistic).
        n_features_per_group: Features per group.
        noise_level: Noise standard deviation.
        random_state: Seed.

    Returns:
        X, y, feature_names, groups (same format as other loaders).
    """
    rng = np.random.RandomState(random_state)
    n_feat = n_features_per_group
    sl = np.clip(synergy_level, 0, 1)

    # Latent signal
    z1 = rng.randn(n_samples)
    z2 = rng.randn(n_samples)
    z3 = rng.randn(n_samples)

    # Individual signal: single group dominates
    individual_signal = z1 * 2

    # Interaction signal: requires all three (XOR-like)
    interaction_signal = z1 * z2 * z3

    # Blend based on synergy_level
    combined = (1 - sl) * individual_signal + sl * interaction_signal

    # Binary label via threshold
    threshold = np.median(combined)
    y = (combined > threshold).astype(int)

    # Generate observable features as noisy functions of latent variables
    G1 = np.column_stack([
        z1 + rng.randn(n_samples) * noise_level,
        z1 * 0.5 + rng.randn(n_samples) * noise_level,
        np.abs(z1) + rng.randn(n_samples) * noise_level,
    ][:n_feat])

    G2 = np.column_stack([
        z2 + rng.randn(n_samples) * noise_level,
        z2 ** 2 + rng.randn(n_samples) * noise_level,
        np.sign(z2) + rng.randn(n_samples) * noise_level,
    ][:n_feat])

    G3 = np.column_stack([
        z3 + rng.randn(n_samples) * noise_level,
        z3 * 0.3 + rng.randn(n_samples) * noise_level,
        np.tanh(z3) + rng.randn(n_samples) * noise_level,
    ][:n_feat])

    X = np.hstack([G1, G2, G3])

    feature_names = (
        [f"g1_f{i+1}" for i in range(n_feat)] +
        [f"g2_f{i+1}" for i in range(n_feat)] +
        [f"g3_f{i+1}" for i in range(n_feat)]
    )

    groups = {
        "G1": list(range(0, n_feat)),
        "G2": list(range(n_feat, 2 * n_feat)),
        "G3": list(range(2 * n_feat, 3 * n_feat)),
    }

    return X, y, feature_names, groups


def generate_grouped(
    mode: str = "cascading",
    **kwargs
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load synthetic data and return grouped for PID analysis."""
    if mode == "cascading":
        X, y, _, groups_idx = generate_cascading_failures(**kwargs)
    elif mode == "controllable":
        X, y, _, groups_idx = generate_controllable_synergy(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    groups = {}
    for g_name, g_idx in groups_idx.items():
        groups[g_name] = X[:, g_idx]

    return groups, y
