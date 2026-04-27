#!/usr/bin/env python3
"""Generate publication figures from experiment_results.json. CPU-only, fast."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─── Load results ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
with open(os.path.join(RESULTS_DIR, "experiment_results.json")) as f:
    data = json.load(f)

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Color palette ───
C_BLUE = "#2563EB"
C_RED = "#DC2626"
C_GREEN = "#059669"
C_ORANGE = "#D97706"
C_PURPLE = "#7C3AED"
C_GRAY = "#6B7280"
BG = "#FAFBFC"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.facecolor': BG,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

datasets = list(data.keys())
short_names = {"AI4I": "AI4I", "C-MAPSS": "C-MAPSS", "Synthetic (Cascading)": "Synthetic", "SMD": "SMD"}

# ════════════════════════════════════════════════
# FIG 1: PID Decomposition — Stacked Bars
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(datasets))
width = 0.22

for ds_idx, ds in enumerate(datasets):
    pids = data[ds]["pairwise_pids"]
    pairs = [p["pair"].replace("×", "×\n") for p in pids]

    for p_idx, pid in enumerate(pids):
        offset = ds_idx * width * 0.0
        bottom = 0
        total = pid["redundancy"] + pid["unique_i"] + pid["unique_j"] + pid["synergy"]

        vals = [
            ("Redundancy", pid["redundancy"], C_BLUE),
            ("Unique₁", pid["unique_i"], C_GREEN),
            ("Unique₂", pid["unique_j"], C_ORANGE),
            ("Synergy", pid["synergy"], C_RED),
        ]

# New approach: one subplot per dataset
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=False)
fig.suptitle("PID Decomposition: Information Atoms per Dataset", fontsize=14, fontweight='bold', y=1.02)

for ax, ds in zip(axes, datasets):
    pids = data[ds]["pairwise_pids"]
    pair_labels = [p["pair"].replace("×", " ×\n") for p in pids]

    x_pos = np.arange(len(pids))
    bottom = np.zeros(len(pids))

    components = [
        ("Redundancy", [p["redundancy"] for p in pids], C_BLUE),
        ("Unique₁", [p["unique_i"] for p in pids], C_GREEN),
        ("Unique₂", [p["unique_j"] for p in pids], C_ORANGE),
        ("Synergy", [p["synergy"] for p in pids], C_RED),
    ]

    for label, vals, color in components:
        vals = np.array(vals)
        ax.bar(x_pos, vals, bottom=bottom, color=color, width=0.6, label=label, edgecolor='white', linewidth=0.5)
        bottom += vals

    ax.set_xticks(x_pos)
    ax.set_xticklabels(pair_labels, fontsize=7, ha='center')
    sr = data[ds]["synergy_ratio"]
    ax.set_title(f"{short_names[ds]}\nSR = {sr:.3f}", fontsize=11, fontweight='bold')
    ax.set_ylabel("bits" if ds == datasets[0] else "")

handles = [mpatches.Patch(color=c, label=l) for l, _, c in components]
fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig1_pid_decomposition.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig1_pid_decomposition.png")


# ════════════════════════════════════════════════
# FIG 2: SR vs Gap to XGBoost (THE KEY FIGURE)
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5.5))

srs, gaps, names, colors = [], [], [], []
color_map = {"AI4I": C_RED, "C-MAPSS": C_BLUE, "Synthetic (Cascading)": C_PURPLE, "SMD": C_GREEN}

for ds in datasets:
    sr = data[ds]["synergy_ratio"]
    our_auc = data[ds]["full_auc_mean"]
    xgb_auc = data[ds]["baselines"]["XGBoost (GB)"]["auc_mean"]
    gap = xgb_auc - our_auc
    srs.append(sr)
    gaps.append(gap)
    names.append(short_names[ds])
    colors.append(color_map[ds])

for i in range(len(srs)):
    ax.scatter(srs[i], gaps[i], s=200, c=colors[i], zorder=5, edgecolors='white', linewidth=2)
    offset_x = 0.03 if srs[i] < 1 else -0.15
    offset_y = 0.003
    ax.annotate(names[i], (srs[i], gaps[i]),
                textcoords="offset points", xytext=(12, 8),
                fontsize=12, fontweight='bold', color=colors[i])

# Add trend arrow
ax.annotate('', xy=(2.2, 0.09), xytext=(0.15, 0.005),
            arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=2, ls='--'))
ax.text(1.0, 0.055, 'Higher SR →\nLarger gap', fontsize=9, color=C_GRAY,
        ha='center', style='italic')

# Regions
ax.axhspan(-0.01, 0.01, alpha=0.08, color=C_GREEN, zorder=0)
ax.axhspan(0.01, 0.15, alpha=0.05, color=C_RED, zorder=0)
ax.text(0.01, 0.003, '✓ Simple model sufficient', fontsize=9, color=C_GREEN, style='italic')
ax.text(0.01, 0.08, '⚠ Fusion needed', fontsize=9, color=C_RED, style='italic')

ax.set_xlabel("Synergy Ratio (SR)", fontsize=13)
ax.set_ylabel("Gap to XGBoost (AUC)", fontsize=13)
ax.set_title("Synergy Ratio Predicts When Fusion Is Needed", fontsize=14, fontweight='bold')
ax.set_xlim(-0.05, 2.6)
ax.set_ylim(-0.005, 0.12)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig2_sr_vs_gap.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig2_sr_vs_gap.png")


# ════════════════════════════════════════════════
# FIG 3: Model Comparison Bars per Dataset
# ════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
fig.suptitle("AUC-ROC Comparison Across Datasets", fontsize=14, fontweight='bold', y=1.02)

methods = ["Ours (Full)", "Ours (Ablation)", "Naive Bayes", "Logistic Reg.", "Random Forest", "XGBoost (GB)"]
method_colors = [C_RED, C_ORANGE, C_GRAY, C_GRAY, C_BLUE, C_PURPLE]

for ax, ds in zip(axes, datasets):
    aucs = [
        data[ds]["full_auc_mean"],
        data[ds]["ablation_auc_mean"],
        data[ds]["baselines"]["Naive Bayes"]["auc_mean"],
        data[ds]["baselines"]["Logistic Reg."]["auc_mean"],
        data[ds]["baselines"]["Random Forest"]["auc_mean"],
        data[ds]["baselines"]["XGBoost (GB)"]["auc_mean"],
    ]
    stds = [
        data[ds]["full_auc_std"],
        0,  # no std for ablation in JSON
        data[ds]["baselines"]["Naive Bayes"]["auc_std"],
        data[ds]["baselines"]["Logistic Reg."]["auc_std"],
        data[ds]["baselines"]["Random Forest"]["auc_std"],
        data[ds]["baselines"]["XGBoost (GB)"]["auc_std"],
    ]

    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, aucs, xerr=stds, color=method_colors, height=0.6,
                   edgecolor='white', linewidth=0.5, capsize=3)

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{auc:.3f}', va='center', fontsize=8, color='#374151')

    sr = data[ds]["synergy_ratio"]
    ax.set_title(f"{short_names[ds]} (SR={sr:.3f})", fontsize=11, fontweight='bold')
    ax.set_xlim(0.65, 1.02)
    if ds == datasets[0]:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods, fontsize=9)
    else:
        ax.set_yticks([])

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig3_model_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig3_model_comparison.png")


# ════════════════════════════════════════════════
# FIG 4: Ablation Effect — Entropy Adds Value?
# ════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Delta AUC bars with significance
deltas = [data[ds]["ablation_delta_auc"] for ds in datasets]
pvals = [data[ds]["ablation_p_value"] for ds in datasets]
bar_colors = [C_GREEN if p < 0.05 else C_GRAY for p in pvals]

y_pos = np.arange(len(datasets))
bars = ax1.barh(y_pos, deltas, color=bar_colors, height=0.5, edgecolor='white')

for i, (bar, delta, pval) in enumerate(zip(bars, deltas, pvals)):
    sig = "✓ p<0.05" if pval < 0.05 else f"p={pval:.2f}"
    x = max(delta, 0) + 0.001
    ax1.text(x, bar.get_y() + bar.get_height()/2,
             f'Δ={delta:+.4f} ({sig})', va='center', fontsize=10,
             fontweight='bold' if pval < 0.05 else 'normal')

ax1.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
ax1.set_yticks(y_pos)
ax1.set_yticklabels([short_names[ds] for ds in datasets], fontsize=11)
ax1.set_xlabel("Δ AUC (Full − Ablation)", fontsize=12)
ax1.set_title("Entropy Feature Contribution", fontsize=13, fontweight='bold')
ax1.set_xlim(-0.005, 0.025)

# Right: SR vs Ablation Delta
for i, ds in enumerate(datasets):
    sr = data[ds]["synergy_ratio"]
    delta = data[ds]["ablation_delta_auc"]
    ax2.scatter(sr, delta, s=180, c=list(color_map.values())[i], zorder=5,
                edgecolors='white', linewidth=2)
    ax2.annotate(short_names[ds], (sr, delta),
                textcoords="offset points", xytext=(10, 6),
                fontsize=11, fontweight='bold', color=list(color_map.values())[i])

ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax2.set_xlabel("Synergy Ratio", fontsize=12)
ax2.set_ylabel("Δ AUC (Full − Ablation)", fontsize=12)
ax2.set_title("SR Predicts Entropy Feature Value", fontsize=13, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig4_ablation.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig4_ablation.png")


# ════════════════════════════════════════════════
# FIG 5: Synergy Ratio Overview — The Hero Figure
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

sr_vals = [data[ds]["synergy_ratio"] for ds in datasets]
bar_colors_sr = [C_RED if sr > 0.15 else (C_ORANGE if sr > 0.05 else C_BLUE) for sr in sr_vals]

bars = ax.bar(range(len(datasets)), sr_vals, color=bar_colors_sr, width=0.5,
              edgecolor='white', linewidth=1.5)

# Add value labels
for bar, sr in zip(bars, sr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'SR = {sr:.3f}', ha='center', fontsize=12, fontweight='bold')

# Threshold lines
ax.axhline(y=0.05, color=C_ORANGE, linewidth=1.5, linestyle='--', alpha=0.6)
ax.axhline(y=0.15, color=C_RED, linewidth=1.5, linestyle='--', alpha=0.6)
ax.text(3.7, 0.06, 'Moderate\nthreshold', fontsize=8, color=C_ORANGE, ha='center')
ax.text(3.7, 0.16, 'High\nthreshold', fontsize=8, color=C_RED, ha='center')

# Zone labels
ax.axhspan(-0.1, 0.05, alpha=0.04, color=C_BLUE)
ax.axhspan(0.05, 0.15, alpha=0.04, color=C_ORANGE)

ax.set_xticks(range(len(datasets)))
ax.set_xticklabels([short_names[ds] for ds in datasets], fontsize=12, fontweight='bold')
ax.set_ylabel("Synergy Ratio", fontsize=13)
ax.set_title("Cross-Dataset Synergy Ratio Comparison", fontsize=14, fontweight='bold')
ax.set_ylim(0, max(sr_vals) * 1.15)

# Legend
legend_patches = [
    mpatches.Patch(color=C_BLUE, label='Low SR — Simple model sufficient'),
    mpatches.Patch(color=C_ORANGE, label='Moderate SR — Fusion optional'),
    mpatches.Patch(color=C_RED, label='High SR — Fusion justified'),
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig5_synergy_overview.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ fig5_synergy_overview.png")

print(f"\n  All 5 figures saved to {FIGURES_DIR}/")
