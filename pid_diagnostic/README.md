# PID Architecture Selection Diagnostic

**Information-Theoretic Diagnostic for Multi-View Failure Prediction**

> *When should you bother with cross-group sensor fusion? We answer this question before you train a single model.*

---

## The Problem

Modern failure detection systems use multiple sensor groups (thermal, mechanical, wear, network, etc.). Two architectural choices exist:

1. **Simple interpretable models** — train per-group classifiers, combine with entropy-weighted meta-features. Fast, interpretable, explainable.
2. **Complex fusion models** — XGBoost, neural nets, attention mechanisms. Higher accuracy potential, but black-box.

Researchers default to (2) without asking: *does the data actually contain cross-group synergistic information that justifies the complexity?*

## Our Contribution

We introduce the **Synergy Ratio (SR)** — a Partial Information Decomposition (PID) metric that answers this question *before* model selection:

```
SR = (Syn₁₂ + Syn₁₃ + Syn₂₃ + Syn₁₂₃) / (I₁ + I₂ + I₃)
```

- **Low SR** (< 0.05): Failure signal is dominated by individual groups or shared redundancy. Simple models suffice. XGBoost adds complexity without benefit.
- **High SR** (> 0.15): Failure genuinely emerges from cross-group interaction. Complex fusion is justified.

## Datasets

| Dataset | Domain | Samples | Groups | Expected SR |
|---------|--------|--------:|:------:|:-----------:|
| AI4I 2020 | CNC machines | 10,000 | Thermal/Mechanical/Wear | LOW (~0.02) |
| C-MAPSS FD001 | Turbofan engines | ~20,000 | Temperature/Pressure/Speed | LOW (~0.01) |
| Synthetic (Cascading) | Distributed systems | 10,000 | Broker/Consumer/Network | HIGH (>0.1) |
| SMD | Server machines | ~10,000 | Compute/Memory/Network | MODERATE |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on a single dataset
python -m pid_diagnostic.experiments.run_all --dataset AI4I

# Run all experiments (takes 5-10 minutes)
python -m pid_diagnostic.experiments.run_all

# Quick mode (5-fold, faster)
python -m pid_diagnostic.experiments.run_all --quick
```

## Project Structure

```
pid_diagnostic/
├── core/
│   ├── entropy_features.py     # H(p), KL divergence, 12D meta-vector
│   ├── pid_decomposition.py    # PID computation, Synergy Ratio
│   └── bilevel_sgd.py          # Dataset-agnostic bi-level SGD pipeline
├── datasets/
│   ├── ai4i_loader.py          # AI4I 2020 predictive maintenance
│   ├── cmapss_loader.py        # NASA C-MAPSS turbofan degradation
│   ├── smd_loader.py           # Server Machine Dataset
│   └── synthetic_synergy.py    # Controllable synergy generators
├── experiments/
│   └── run_all.py              # Unified experiment pipeline
├── docs/
│   └── theorem.md              # Formal proposition for SR properties
├── tests/
├── data/                       # Auto-downloaded datasets
├── results/                    # Generated figures and result JSONs
├── references.md               # Full bibliography (24 references)
├── requirements.txt
└── README.md                   # This file
```

## Key Results (Expected)

| Dataset | SR | Our AUC | XGB AUC | Gap | Interpretation |
|---------|:--:|:-------:|:-------:|:---:|:--------------|
| AI4I | ~0.02 | ~0.884 | ~0.976 | ~0.09 | Low SR → gap exists but simple model is interpretable |
| C-MAPSS | ~0.01 | ~0.85 | ~0.90 | ~0.05 | Low SR → minimal fusion benefit |
| Synthetic | ~0.15+ | ~0.80 | ~0.85 | ~0.05 | High SR → fusion helps, gap smaller |

**The key figure**: Plot SR (x-axis) vs "fusion benefit" (y-axis, defined as improvement from adding entropy/KL features). Should show positive correlation — demonstrating the diagnostic predicts when fusion is worth it.

## Paper Target

- **Primary**: FSE 2027 / ICSE 2027 (CORE A*)
- **Backup**: DSN 2027 / Middleware 2027 (CORE A, Q1)
- **Journal fallback**: ESWA / RESS / IST (Q1, rolling submission)

## Future Work

The synthetic high-synergy dataset demonstrates the diagnostic's discriminative power, but the paper will be significantly strengthened by a **real Kafka failure dataset**:

- 3-broker cluster with JMX + Prometheus scraping at 1-second granularity
- Fault injection: broker kill, network partition, disk throttle, consumer crash
- 150-200 fault episodes across 4 fault types
- First public labeled Kafka failure dataset

This is planned for a follow-up study once lab infrastructure is available.

---

## Citation

*(To be updated with paper details after submission)*

## License

MIT
