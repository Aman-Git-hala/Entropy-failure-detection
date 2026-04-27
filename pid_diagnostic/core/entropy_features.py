#!/usr/bin/env python3
"""
Information-Theoretic Feature Functions
========================================

Core entropy and divergence computations used throughout the pipeline.

Provides:
    - Binary Shannon entropy H(p)
    - KL divergence for Bernoulli distributions
    - Entropy contrast (absolute difference)
    - Joint entropy and mutual information estimators

All functions are vectorized (NumPy) for performance on 10,000+ samples.

References:
    Shannon, C.E. (1948). A Mathematical Theory of Communication.
    Kullback, S. & Leibler, R.A. (1951). On Information and Sufficiency.
"""

import numpy as np
from typing import Tuple, Optional


def binary_entropy(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Compute binary Shannon entropy for Bernoulli random variables.

    H(p) = -p·log₂(p) - (1-p)·log₂(1-p)

    Properties:
        - H(0) = H(1) = 0  (complete certainty)
        - H(0.5) = 1.0     (maximum uncertainty)
        - H(p) = H(1-p)    (symmetric)

    Args:
        p: Array of probabilities in [0, 1].
        eps: Small constant for numerical stability.

    Returns:
        Array of entropy values in [0, 1].
    """
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    KL divergence between two Bernoulli distributions.

    D_KL(p || q) = p·log₂(p/q) + (1-p)·log₂((1-p)/(1-q))

    Properties:
        - D_KL(p || q) ≥ 0 (Gibbs' inequality)
        - D_KL(p || q) = 0  iff  p = q
        - Asymmetric: D_KL(p || q) ≠ D_KL(q || p) in general

    Args:
        p: Array of probabilities (reference distribution).
        q: Array of probabilities (comparison distribution).
        eps: Small constant for numerical stability.

    Returns:
        Array of KL divergence values in [0, ∞).
    """
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log2(p / q) + (1 - p) * np.log2((1 - p) / (1 - q))


def entropy_contrast(h_i: np.ndarray, h_j: np.ndarray) -> np.ndarray:
    """
    Absolute entropy difference between two groups.

    ΔH_ij = |H(p_i) - H(p_j)|

    Captures confidence asymmetry: large ΔH means one group is much more
    confident than the other, which is itself a diagnostic signal.

    Args:
        h_i: Entropy values for group i.
        h_j: Entropy values for group j.

    Returns:
        Array of absolute entropy differences.
    """
    return np.abs(h_i - h_j)


def build_12d_meta_vector(
    probs: list[np.ndarray],
    group_names: Optional[list[str]] = None
) -> Tuple[np.ndarray, list[str]]:
    """
    Construct the 12-dimensional meta-feature vector from 3 group probabilities.

    z = [p₁, H₁, p₂, H₂, p₃, H₃, KL₁₂, KL₁₃, KL₂₃, ΔH₁₂, ΔH₁₃, ΔH₂₃]

    Args:
        probs: List of 3 arrays, each shape (n_samples,), containing
               calibrated failure probabilities from inner models.
        group_names: Optional list of 3 group names for labeling.

    Returns:
        meta_features: (n_samples, 12) array of meta-features.
        feature_names: List of 12 feature name strings.
    """
    assert len(probs) == 3, "Exactly 3 groups required"
    n = len(probs[0])

    if group_names is None:
        group_names = ["G1", "G2", "G3"]

    # Base features: [p_i, H_i] for each group
    entropies = [binary_entropy(p) for p in probs]

    base = []
    for p, h in zip(probs, entropies):
        base.append(p)
        base.append(h)

    # KL divergences: pairwise
    pairs = [(0, 1), (0, 2), (1, 2)]
    kl_feats = [kl_divergence(probs[i], probs[j]) for i, j in pairs]

    # Entropy contrasts: pairwise
    dh_feats = [entropy_contrast(entropies[i], entropies[j]) for i, j in pairs]

    # Stack
    meta = np.column_stack(base + kl_feats + dh_feats)

    # Feature names
    names = []
    for g in group_names:
        names.extend([f"p_{g}", f"H_{g}"])
    for i, j in pairs:
        names.append(f"KL({group_names[i]}||{group_names[j]})")
    for i, j in pairs:
        names.append(f"ΔH({group_names[i]},{group_names[j]})")

    return meta, names


def discretize_features(
    X: np.ndarray,
    n_bins: int = 8,
    strategy: str = "quantile"
) -> np.ndarray:
    """
    Discretize continuous features for information-theoretic computation.

    PID requires discrete random variables. We use equal-frequency (quantile)
    binning by default as it maximizes entropy of the marginal distribution,
    avoiding degenerate bins with extreme-valued features.

    Args:
        X: Continuous feature array, shape (n_samples, n_features).
        n_bins: Number of discretization bins.
        strategy: 'quantile' for equal-frequency, 'uniform' for equal-width.

    Returns:
        Discretized array of same shape, values in {0, ..., n_bins-1}.
    """
    X_disc = np.zeros_like(X, dtype=int)

    for j in range(X.shape[1]):
        col = X[:, j]
        if strategy == "quantile":
            # Equal-frequency binning
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.unique(np.percentile(col, percentiles))
            X_disc[:, j] = np.clip(
                np.digitize(col, bin_edges[1:-1]), 0, n_bins - 1
            )
        elif strategy == "uniform":
            # Equal-width binning
            col_min, col_max = col.min(), col.max()
            if col_max == col_min:
                X_disc[:, j] = 0
            else:
                X_disc[:, j] = np.clip(
                    ((col - col_min) / (col_max - col_min) * n_bins).astype(int),
                    0, n_bins - 1
                )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return X_disc


def mutual_information_discrete(
    X: np.ndarray,
    Y: np.ndarray,
    n_bins_x: Optional[int] = None
) -> float:
    """
    Compute mutual information I(X; Y) between discrete multivariate X and Y.

    Uses the plugin estimator with joint frequency counting.
    X can be multi-dimensional — we hash feature vectors to joint states.

    Args:
        X: Discrete feature array, shape (n_samples,) or (n_samples, n_features).
        Y: Discrete label array, shape (n_samples,).

    Returns:
        Mutual information in bits (log base 2).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n = len(Y)

    # Hash multi-dimensional X into single integer states
    # Use a simple cantor-like pairing
    x_states = np.zeros(n, dtype=int)
    max_val = 1
    for j in range(X.shape[1]):
        col = X[:, j].astype(int)
        x_states = x_states * (col.max() + 1) + col
        max_val *= (col.max() + 1)

    # Joint distribution P(X, Y)
    unique_x = np.unique(x_states)
    unique_y = np.unique(Y)

    mi = 0.0
    for x in unique_x:
        for y in unique_y:
            p_xy = np.sum((x_states == x) & (Y == y)) / n
            p_x = np.sum(x_states == x) / n
            p_y = np.sum(Y == y) / n

            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log2(p_xy / (p_x * p_y))

    return mi
