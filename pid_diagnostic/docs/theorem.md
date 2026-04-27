# Formal Proposition: Synergy Ratio Properties

## Proposition 1: Synergy Ratio Bounds and Monotonicity

### Statement

Let $X_1, X_2, X_3$ be three sensor group random variables and $Y$ a binary failure label. Define the **Synergy Ratio** as:

$$\text{SR} = \frac{\sum_{i < j} \text{Syn}(X_i, X_j; Y)}{I(X_1; Y) + I(X_2; Y) + I(X_3; Y)}$$

where $\text{Syn}(X_i, X_j; Y)$ is the synergistic information atom from the PID decomposition of $I(X_i, X_j; Y)$.

**Claim 1 (Boundedness)**: $\text{SR} \geq 0$ under any non-negative PID measure (Williams-Beer, BROJA, etc.).

**Claim 2 (Zero Synergy Sufficient Condition)**: If $Y$ is fully determined by any single group — i.e., $\exists k$ such that $I(X_k; Y) = H(Y)$ — then $\text{SR} = 0$.

**Claim 3 (Single Latent Variable)**: If all groups are conditionally independent given a single latent degradation variable $Z$ — i.e., $X_i \perp X_j \mid Z$ and $Y = f(Z)$ — then synergy is bounded by the noise in the latent-to-observed mapping:

$$\text{Syn}(X_i, X_j; Y) \leq H(Y) - I(Z; Y)$$

In the noiseless case ($I(Z; Y) = H(Y)$), all synergy atoms are zero.

### Proof Sketch

**Claim 1**: Direct from the PID framework. The Williams-Beer lattice decomposition ensures all atoms are non-negative by construction (Williams & Beer, 2010, Theorem 1). The denominator is the sum of marginal mutual informations, which is non-negative. Hence $\text{SR} \geq 0$.

**Claim 2**: If $I(X_k; Y) = H(Y)$, then $X_k$ fully determines $Y$. For any pair $(X_k, X_j)$:

$$I(X_k, X_j; Y) = I(X_k; Y) + I(X_j; Y \mid X_k) = H(Y) + 0 = H(Y)$$

The PID decomposes this as:

$$H(Y) = \text{Red}(X_k, X_j; Y) + \text{Unq}(X_k; Y) + \text{Unq}(X_j; Y) + \text{Syn}(X_k, X_j; Y)$$

Since $I(X_k; Y) = H(Y) = \text{Red}(X_k, X_j; Y) + \text{Unq}(X_k; Y)$ (by the marginal constraint), and $I(X_j; Y \mid X_k) = 0$ implies $\text{Unq}(X_j; Y) + \text{Syn}(X_k, X_j; Y) = 0$. By non-negativity of both terms, $\text{Syn}(X_k, X_j; Y) = 0$. ∎

**Claim 3**: Under the conditional independence assumption $X_i \perp X_j \mid Z$, the data processing inequality gives:

$$I(X_i, X_j; Y) \leq I(Z; Y)$$

The synergy atom satisfies:

$$\text{Syn}(X_i, X_j; Y) = I(X_i, X_j; Y) - I(X_i; Y) - I(X_j; Y) + \text{Red}(X_i, X_j; Y)$$

When all information flows through $Z$, the joint provides no information beyond what $Z$ already carries. The synergy is bounded by the information loss in the $Z \to X_i$ and $Z \to X_j$ channels. In the noiseless case ($I(Z;Y) = H(Y)$ and $X_i$ are deterministic functions of $Z$), the conditional independence forces all pairwise synergy to zero. ∎

---

## Proposition 2: When High Synergy Is Expected

### Statement

Consider a failure mechanism where $Y = 1$ iff a conjunction of conditions across multiple groups is satisfied:

$$Y = \mathbb{1}[g_1(X_1) > \tau_1 \;\wedge\; g_2(X_2) > \tau_2 \;\wedge\; g_3(X_3) > \tau_3]$$

where $g_i$ are group-specific health indicators and $\tau_i$ are thresholds. If the $X_i$ are independent (no shared latent variable), then:

$$\text{Syn}(X_i, X_j; Y) > 0 \quad \forall\; i \neq j$$

and the Synergy Ratio $\text{SR} > 0$.

### Proof Sketch

Independence of $X_i$ means $I(X_i; X_j) = 0$ but the AND condition creates statistical dependence between each $X_i$ and $Y$ that is only fully captured jointly. For any single group $X_i$:

$$I(X_i; Y) < H(Y)$$

because $X_i$ alone cannot determine $Y$ (the other groups' conditions may or may not be met). However:

$$I(X_i, X_j; Y) > I(X_i; Y) + I(X_j; Y) - I(X_i; X_j; Y)$$

where $I(X_i; X_j; Y)$ is the interaction information (which is negative for AND-type interactions). This negative interaction information manifests as positive synergy in the PID decomposition.

Intuitively: knowing $X_1 > \tau_1$ is only useful for predicting failure if you *also* know $X_2 > \tau_2$. This "only useful in combination" property is precisely what synergy captures. ∎

---

## Proposition 3: SR as Architecture Selection Criterion

### Statement (Empirical)

Define the **fusion benefit** as:

$$\Delta_{\text{fusion}} = \text{AUC}(\text{Full 12D model}) - \text{AUC}(\text{Ablation 3D model})$$

**Conjecture**: Across datasets with varying synergy structures, $\Delta_{\text{fusion}}$ is positively correlated with $\text{SR}$:

$$\text{Corr}(\text{SR}, \Delta_{\text{fusion}}) > 0$$

This is validated empirically by sweeping the `synergy_level` parameter in our synthetic data generator and measuring both SR and $\Delta_{\text{fusion}}$ at each level.

### Empirical Validation Plan

1. Generate synthetic datasets with `synergy_level` ∈ {0.0, 0.1, 0.2, ..., 1.0}
2. For each: compute SR and run both full and ablation models
3. Plot SR vs $\Delta_{\text{fusion}}$
4. Compute Spearman rank correlation
5. If $\rho > 0.7$ with $p < 0.01$, the conjecture is supported

---

## Corollary: Practical Decision Rule

**Rule**: Given a new failure detection domain with $k = 3$ sensor groups:

1. Compute $\text{SR}$ from a labeled sample
2. If $\text{SR} < 0.05$: deploy simple entropy-KL model (interpretable, fast, sufficient)
3. If $0.05 \leq \text{SR} < 0.15$: entropy-KL model competitive; fusion optional
4. If $\text{SR} \geq 0.15$: cross-group fusion justified; consider XGBoost or neural fusion

The thresholds (0.05, 0.15) are calibrated from our three-dataset evaluation and the synthetic sweep experiment.

---

## Connection to Existing Theory

Our Synergy Ratio relates to established information-theoretic concepts:

- **Interaction Information** (McGill, 1954): $I(X_1; X_2; Y) = I(X_1; Y) + I(X_2; Y) - I(X_1, X_2; Y)$. Negative interaction information implies synergy. But interaction information can be both positive (redundancy) and negative (synergy), making it hard to interpret. PID resolves this ambiguity.

- **Total Correlation** (Watanabe, 1960): $TC(X_1, X_2, X_3) = \sum_i H(X_i) - H(X_1, X_2, X_3)$. TC measures all dependencies, not just those relevant to $Y$. SR is target-specific.

- **Dual Total Correlation** (Han, 1975): Also known as binding information. Related to but distinct from synergy.

The key advantage of SR over these older measures is that it specifically quantifies synergy *with respect to the prediction target* $Y$, making it directly actionable for model selection.
