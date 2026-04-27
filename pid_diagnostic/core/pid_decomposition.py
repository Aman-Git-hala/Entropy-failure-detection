#!/usr/bin/env python3
"""
Partial Information Decomposition (PID) Framework
===================================================

Decomposes the mutual information I(X₁, X₂, ..., Xₖ ; Y) between
k source variables (sensor groups) and a target (failure label) into
non-negative atoms: Unique, Redundancy, and Synergy.

For the pairwise case (2 sources):
    I(X₁, X₂ ; Y) = Red(X₁, X₂; Y) + Unq(X₁; Y) + Unq(X₂; Y) + Syn(X₁, X₂; Y)

This module implements PID computation using two approaches:
    1. The `dit` library (Ibroja estimator) — theoretically rigorous
    2. A fallback Williams-Beer I_min estimator — simpler, works without `dit`

References:
    Williams, P.L. & Beer, R.D. (2010). Nonnegative decomposition of
        multivariate information. arXiv:1004.2515.
    Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N. (2014).
        Quantifying unique information. Entropy, 16(4), 2161-2183.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from .entropy_features import discretize_features, mutual_information_discrete


@dataclass
class PIDResult:
    """Results of a pairwise PID decomposition."""
    source_i: str
    source_j: str
    target: str
    # The four PID atoms
    redundancy: float    # Information shared by both sources
    unique_i: float      # Information only in source i
    unique_j: float      # Information only in source j
    synergy: float       # Information only in the joint (i, j)
    # Derived quantities
    mi_i: float          # I(X_i; Y)
    mi_j: float          # I(X_j; Y)
    mi_joint: float      # I(X_i, X_j; Y)

    @property
    def total_check(self) -> float:
        """Verify: Red + Unq_i + Unq_j + Syn should equal I(Xi, Xj; Y)."""
        return self.redundancy + self.unique_i + self.unique_j + self.synergy

    def __repr__(self):
        return (
            f"PID({self.source_i}, {self.source_j} → {self.target}):\n"
            f"  I({self.source_i}; Y) = {self.mi_i:.4f} bits\n"
            f"  I({self.source_j}; Y) = {self.mi_j:.4f} bits\n"
            f"  I({self.source_i},{self.source_j}; Y) = {self.mi_joint:.4f} bits\n"
            f"  Redundancy = {self.redundancy:.4f} bits\n"
            f"  Unique({self.source_i}) = {self.unique_i:.4f} bits\n"
            f"  Unique({self.source_j}) = {self.unique_j:.4f} bits\n"
            f"  Synergy = {self.synergy:.4f} bits\n"
            f"  Check: {self.total_check:.4f} ≈ {self.mi_joint:.4f}"
        )


@dataclass
class SynergyDiagnostic:
    """Full diagnostic report for a dataset's synergy structure."""
    dataset_name: str
    group_names: List[str]
    # Per-group MI
    mi_per_group: Dict[str, float]
    # Pairwise PID results
    pairwise_pids: List[PIDResult]
    # Aggregate synergy ratio
    synergy_ratio: float
    synergy_ratio_ci: Optional[Tuple[float, float]] = None

    def __repr__(self):
        lines = [
            f"\n{'='*60}",
            f"  SYNERGY DIAGNOSTIC: {self.dataset_name}",
            f"{'='*60}",
            f"\n  Per-Group Mutual Information:",
        ]
        for g, mi in self.mi_per_group.items():
            lines.append(f"    I({g}; Y) = {mi:.4f} bits")

        lines.append(f"\n  Pairwise PID:")
        total_syn = 0
        total_red = 0
        for pid in self.pairwise_pids:
            lines.append(
                f"    {pid.source_i} × {pid.source_j}: "
                f"Red={pid.redundancy:.4f}, "
                f"Unq_i={pid.unique_i:.4f}, "
                f"Unq_j={pid.unique_j:.4f}, "
                f"Syn={pid.synergy:.4f}"
            )
            total_syn += pid.synergy
            total_red += pid.redundancy

        lines.extend([
            f"\n  Aggregates:",
            f"    Total Synergy:    {total_syn:.4f} bits",
            f"    Total Redundancy: {total_red:.4f} bits",
            f"    Synergy Ratio:    {self.synergy_ratio:.4f}",
        ])

        if self.synergy_ratio_ci is not None:
            lines.append(
                f"    SR 95% CI:        [{self.synergy_ratio_ci[0]:.4f}, "
                f"{self.synergy_ratio_ci[1]:.4f}]"
            )

        # Interpretation
        lines.append(f"\n  Interpretation:")
        if self.synergy_ratio < 0.05:
            lines.append(
                "    ⬇ LOW SYNERGY — Individual groups or redundancy dominate."
            )
            lines.append(
                "    → Simple interpretable model (entropy-KL framework) is sufficient."
            )
            lines.append(
                "    → Cross-group fusion (XGBoost, neural nets) adds complexity without benefit."
            )
        elif self.synergy_ratio < 0.15:
            lines.append(
                "    ↔ MODERATE SYNERGY — Some cross-group interaction exists."
            )
            lines.append(
                "    → Entropy-KL framework competitive; fusion may give marginal gains."
            )
        else:
            lines.append(
                "    ⬆ HIGH SYNERGY — Failure genuinely emerges from group interactions."
            )
            lines.append(
                "    → Cross-group fusion is justified and likely necessary."
            )
            lines.append(
                "    → Consider interaction-capturing architectures."
            )

        lines.append(f"{'='*60}")
        return "\n".join(lines)


def _imin_redundancy(
    X_i: np.ndarray, X_j: np.ndarray, Y: np.ndarray
) -> float:
    """
    Williams-Beer I_min redundancy measure.

    I_min(X_i, X_j; Y) = Σ_y p(y) min[ I_spec(X_i; y), I_spec(X_j; y) ]

    where I_spec(X; y) = Σ_x p(x|y) log₂(p(y|x) / p(y))
                       = Σ_x p(x|y) [log₂ p(x,y) - log₂ p(x) - log₂ p(y) + log₂ p(y)]
                       = specific information about outcome y from X

    This is the original Williams-Beer (2010) proposal. Known to have
    limitations (e.g., may overestimate redundancy) but is simple and
    widely implemented.
    """
    n = len(Y)
    unique_y = np.unique(Y)

    imin = 0.0

    for y_val in unique_y:
        p_y = np.sum(Y == y_val) / n
        mask_y = (Y == y_val)

        # Specific information from X_i about y
        si_i = _specific_information(X_i, Y, y_val, n)
        si_j = _specific_information(X_j, Y, y_val, n)

        imin += p_y * min(si_i, si_j)

    return max(0.0, imin)


def _specific_information(
    X: np.ndarray, Y: np.ndarray, y_val: int, n: int
) -> float:
    """
    Compute specific information I_spec(X; y) = Σ_x p(x|y) log₂(p(y|x)/p(y)).

    This measures how much a specific outcome y is predicted by knowing X.
    """
    mask_y = (Y == y_val)
    p_y = np.sum(mask_y) / n

    if p_y == 0 or p_y == 1:
        return 0.0

    # Hash X if multidimensional
    if X.ndim == 1:
        x_vals = X
    else:
        x_vals = np.zeros(n, dtype=int)
        for j in range(X.shape[1]):
            col = X[:, j].astype(int)
            x_vals = x_vals * (col.max() + 1) + col

    unique_x = np.unique(x_vals)
    si = 0.0

    for x_val in unique_x:
        mask_x = (x_vals == x_val)
        p_xy = np.sum(mask_x & mask_y) / n
        p_x = np.sum(mask_x) / n

        if p_xy > 0 and p_x > 0:
            p_x_given_y = p_xy / p_y
            p_y_given_x = p_xy / p_x
            si += p_x_given_y * np.log2(p_y_given_x / p_y)

    return max(0.0, si)


def compute_pairwise_pid(
    X_i: np.ndarray,
    X_j: np.ndarray,
    Y: np.ndarray,
    name_i: str = "G1",
    name_j: str = "G2",
    n_bins: int = 8,
    use_dit: bool = False,
    max_samples: int = 2000
) -> PIDResult:
    """
    Compute PID decomposition for two source groups and binary target.

    Steps:
        1. Discretize continuous features
        2. Compute marginal MI: I(X_i; Y), I(X_j; Y)
        3. Compute joint MI: I(X_i, X_j; Y)
        4. Compute redundancy via I_min
        5. Derive: Unique_i = I(X_i; Y) - Red
                   Unique_j = I(X_j; Y) - Red
                   Synergy = I(X_i, X_j; Y) - Unique_i - Unique_j - Red

    Args:
        X_i: Features for group i, shape (n_samples, n_features_i).
        X_j: Features for group j, shape (n_samples, n_features_j).
        Y: Binary labels, shape (n_samples,).
        name_i: Name of group i.
        name_j: Name of group j.
        n_bins: Number of bins for discretization.
        use_dit: If True, attempt to use the `dit` library for Ibroja estimator.

    Returns:
        PIDResult with all four atoms and MI values.
    """
    if X_i.ndim == 1:
        X_i = X_i.reshape(-1, 1)
    if X_j.ndim == 1:
        X_j = X_j.reshape(-1, 1)

    # Subsample for speed (PID converges well before 10k samples)
    n = len(Y)
    if max_samples > 0 and n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_i = X_i[idx]
        X_j = X_j[idx]
        Y = Y[idx]

    # Discretize
    X_i_d = discretize_features(X_i, n_bins=n_bins)
    X_j_d = discretize_features(X_j, n_bins=n_bins)
    Y_d = Y.astype(int)

    # Marginal MI
    mi_i = mutual_information_discrete(X_i_d, Y_d)
    mi_j = mutual_information_discrete(X_j_d, Y_d)

    # Joint MI
    X_joint = np.hstack([X_i_d, X_j_d])
    mi_joint = mutual_information_discrete(X_joint, Y_d)

    # PID via I_min
    if use_dit:
        try:
            red, unq_i, unq_j, syn = _pid_via_dit(X_i_d, X_j_d, Y_d)
        except (ImportError, Exception) as e:
            print(f"  [Warning] dit library failed ({e}), falling back to I_min")
            red = _imin_redundancy(X_i_d, X_j_d, Y_d)
            unq_i = max(0.0, mi_i - red)
            unq_j = max(0.0, mi_j - red)
            syn = max(0.0, mi_joint - unq_i - unq_j - red)
    else:
        red = _imin_redundancy(X_i_d, X_j_d, Y_d)
        unq_i = max(0.0, mi_i - red)
        unq_j = max(0.0, mi_j - red)
        syn = max(0.0, mi_joint - unq_i - unq_j - red)

    return PIDResult(
        source_i=name_i,
        source_j=name_j,
        target="Failure",
        redundancy=red,
        unique_i=unq_i,
        unique_j=unq_j,
        synergy=syn,
        mi_i=mi_i,
        mi_j=mi_j,
        mi_joint=mi_joint,
    )


def _pid_via_dit(
    X_i_d: np.ndarray, X_j_d: np.ndarray, Y_d: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute PID using the `dit` library's Ibroja estimator.

    The Ibroja measure (Bertschinger et al., 2014) is considered more
    theoretically justified than I_min for many scenarios.
    """
    import dit
    from dit.pid import PID_BROJA

    n = len(Y_d)

    # Hash multi-dim arrays to single int states
    def hash_array(X):
        if X.ndim == 1:
            return X.astype(int)
        vals = np.zeros(len(X), dtype=int)
        max_v = 1
        for j in range(X.shape[1]):
            col = X[:, j].astype(int)
            vals = vals * (col.max() + 1) + col
        return vals

    x_i_states = hash_array(X_i_d)
    x_j_states = hash_array(X_j_d)

    # Build joint distribution p(x_i, x_j, y)
    outcomes = []
    probs = []
    counts = {}
    for idx in range(n):
        key = (str(x_i_states[idx]), str(x_j_states[idx]), str(Y_d[idx]))
        counts[key] = counts.get(key, 0) + 1

    for key, count in counts.items():
        outcomes.append(key)
        probs.append(count / n)

    d = dit.Distribution(outcomes, probs)
    d.set_rv_names("XYZ")

    pid = PID_BROJA(d, [[0], [1]], [2])

    red = float(pid._pis[((0,), (1,))])
    unq_i = float(pid._pis[((0,),)])
    unq_j = float(pid._pis[((1,),)])
    syn = float(pid._pis[((0, 1),)])

    return red, unq_i, unq_j, syn


def compute_synergy_diagnostic(
    groups: Dict[str, np.ndarray],
    Y: np.ndarray,
    dataset_name: str = "Dataset",
    n_bins: int = 8,
    use_dit: bool = False,
    bootstrap_n: int = 50,
    bootstrap_ci: float = 0.95,
    max_samples: int = 2000
) -> SynergyDiagnostic:
    """
    Run full synergy diagnostic on a dataset with semantic groups.

    Computes:
        1. Per-group mutual information I(G_i; Y)
        2. All pairwise PID decompositions
        3. Synergy Ratio SR = Σ Syn / Σ I

    Args:
        groups: Dict mapping group names to feature arrays.
                Each value has shape (n_samples, n_features_in_group).
        Y: Binary failure labels, shape (n_samples,).
        dataset_name: Name for reporting.
        n_bins: Discretization bins.
        use_dit: Whether to use dit library.
        bootstrap_n: Number of bootstrap samples for CI.
        bootstrap_ci: Confidence level for CI.

    Returns:
        SynergyDiagnostic with full report.
    """
    group_names = list(groups.keys())
    group_arrays = list(groups.values())

    assert len(group_names) == 3, "Exactly 3 groups required for current framework"

    # Subsample for PID computation (converges well at 2000 samples)
    n_total = len(Y)
    if max_samples > 0 and n_total > max_samples:
        rng_sub = np.random.RandomState(42)
        sub_idx = rng_sub.choice(n_total, size=max_samples, replace=False)
        Y_sub = Y[sub_idx]
        group_arrays_sub = [arr[sub_idx] for arr in group_arrays]
    else:
        Y_sub = Y
        group_arrays_sub = group_arrays

    # 1. Per-group MI
    mi_per_group = {}
    for name, X_g in zip(group_names, group_arrays_sub):
        if X_g.ndim == 1:
            X_g = X_g.reshape(-1, 1)
        X_g_d = discretize_features(X_g, n_bins=n_bins)
        mi = mutual_information_discrete(X_g_d, Y_sub.astype(int))
        mi_per_group[name] = mi

    # 2. Pairwise PID
    pairs = [(0, 1), (0, 2), (1, 2)]
    pairwise_pids = []
    for i, j in pairs:
        pid = compute_pairwise_pid(
            group_arrays_sub[i], group_arrays_sub[j], Y_sub,
            name_i=group_names[i], name_j=group_names[j],
            n_bins=n_bins, use_dit=use_dit,
            max_samples=0  # already subsampled
        )
        pairwise_pids.append(pid)

    # 3. Synergy Ratio
    total_synergy = sum(pid.synergy for pid in pairwise_pids)
    total_mi = sum(mi_per_group.values())
    sr = total_synergy / total_mi if total_mi > 0 else 0.0

    # 4. Bootstrap CI for SR
    sr_ci = None
    boot_sample_size = min(1000, len(Y_sub))  # small subsample for bootstrap speed
    if bootstrap_n > 0:
        sr_boots = []
        n_boot = len(Y_sub)
        rng = np.random.RandomState(42)

        for _ in range(bootstrap_n):
            idx = rng.choice(n_boot, size=boot_sample_size, replace=True)
            Y_b = Y_sub[idx]
            arrays_b = [arr[idx] for arr in group_arrays_sub]

            try:
                mi_b = {}
                for name, X_g in zip(group_names, arrays_b):
                    if X_g.ndim == 1:
                        X_g = X_g.reshape(-1, 1)
                    X_g_d = discretize_features(X_g, n_bins=n_bins)
                    mi_b[name] = mutual_information_discrete(X_g_d, Y_b.astype(int))

                syn_b = 0.0
                for ii, jj in pairs:
                    pid_b = compute_pairwise_pid(
                        arrays_b[ii], arrays_b[jj], Y_b,
                        n_bins=n_bins, use_dit=False,
                        max_samples=0  # already subsampled
                    )
                    syn_b += pid_b.synergy

                mi_total_b = sum(mi_b.values())
                sr_b = syn_b / mi_total_b if mi_total_b > 0 else 0.0
                sr_boots.append(sr_b)
            except Exception:
                continue

        if len(sr_boots) > 10:
            alpha = (1 - bootstrap_ci) / 2
            sr_ci = (
                float(np.percentile(sr_boots, 100 * alpha)),
                float(np.percentile(sr_boots, 100 * (1 - alpha)))
            )

    return SynergyDiagnostic(
        dataset_name=dataset_name,
        group_names=group_names,
        mi_per_group=mi_per_group,
        pairwise_pids=pairwise_pids,
        synergy_ratio=sr,
        synergy_ratio_ci=sr_ci,
    )
