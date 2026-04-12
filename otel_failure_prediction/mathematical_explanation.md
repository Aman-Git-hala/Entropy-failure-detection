# Mathematical Foundation: Entropy-Calibrated Multi-View Failure Prediction

> A complete mathematical walkthrough with numerical examples for every formula used in the pipeline.

---

## Table of Contents

1. [Feature Grouping](#1-feature-grouping)
2. [StandardScaler Normalization](#2-standardscaler-normalization)
3. [SGD Classifier (Inner Loop)](#3-sgd-classifier-inner-loop)
4. [SMOTE Oversampling](#4-smote-oversampling)
5. [Isotonic Calibration](#5-isotonic-calibration)
6. [Binary Shannon Entropy](#6-binary-shannon-entropy)
7. [KL Divergence (Cross-Group Disagreement)](#7-kl-divergence)
8. [Entropy Contrast](#8-entropy-contrast)
9. [Meta-Feature Vector Construction](#9-meta-feature-vector-construction)
10. [Outer Classifier](#10-outer-classifier)
11. [Youden's J Threshold](#11-youdens-j-threshold)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Ablation Study (Paired t-test)](#13-ablation-study)
14. [Full Worked Example (End-to-End)](#14-full-worked-example)

---

## 1. Feature Grouping

We split 6 raw sensor features into 3 **semantic groups** based on the physical failure mechanism they relate to:

```
Group 1 — THERMAL:     {Air temperature [K], Process temperature [K]}
Group 2 — MECHANICAL:  {Rotational speed [rpm], Torque [Nm]}
Group 3 — WEAR:        {Tool wear [min], Product type (encoded)}
```

### Why these groups?

| Group | Physics | Related Failure Mode |
|-------|---------|---------------------|
| Thermal | Heat generation & dissipation | HDF: fails when temp diff < 8.6K AND speed < 1380 |
| Mechanical | Rotational dynamics & force | PWF: fails when Power < 3500W or > 9000W |
| Wear | Cumulative material degradation | TWF: fails when wear > 200 min; OSF: wear × torque > threshold |

### Example: One sample from the dataset

```
Sample #4:
  Air temperature     = 297.7 K
  Process temperature = 306.8 K
  Rotational speed    = 1335 rpm
  Torque              = 39.3 Nm
  Tool wear           = 88 min
  Type                = M (encoded as 1)
  Machine failure     = 0 (no failure)
```

Grouped:
```
  Thermal:    x₁ = [297.7, 306.8]
  Mechanical: x₂ = [1335, 39.3]
  Wear:       x₃ = [88, 1]
```

---

## 2. StandardScaler Normalization

Each feature in each group is **z-score normalized**:

$$z = \frac{x - \mu}{\sigma}$$

where $\mu$ = mean of that feature in the training set, $\sigma$ = standard deviation.

### Worked Example (Thermal group)

Suppose the training set statistics are:

| Feature | $\mu$ (mean) | $\sigma$ (std) |
|---------|:---:|:---:|
| Air temperature | 300.0 K | 2.0 K |
| Process temperature | 310.0 K | 2.25 K |

For sample #4:

$$z_{\text{air}} = \frac{297.7 - 300.0}{2.0} = \frac{-2.3}{2.0} = -1.15$$

$$z_{\text{proc}} = \frac{306.8 - 310.0}{2.25} = \frac{-3.2}{2.25} = -1.42$$

So the **scaled** Thermal input becomes:

$$\mathbf{x}_1^{(scaled)} = [-1.15, -1.42]$$

### Why normalize?

SGD is sensitive to feature scales. Without normalization, Rotational speed (range 1168–2886) would dominate Torque (range 3.8–76.8). Z-scoring puts all features on the same scale.

---

## 3. SGD Classifier (Inner Loop)

Each group gets its own **SGDClassifier** — essentially logistic regression trained via stochastic gradient descent.

### 3.1 Linear Model

The model computes a **decision score**:

$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b$$

where $d$ is the number of features in the group, $\mathbf{w}$ are learned weights, and $b$ is the bias.

### 3.2 Sigmoid Activation

The score is converted to a probability via the **sigmoid function**:

$$\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Properties of sigmoid:
- $\sigma(0) = 0.5$ (uncertain)
- $\sigma(\infty) = 1.0$ (certain positive)
- $\sigma(-\infty) = 0.0$ (certain negative)

### 3.3 Loss Function (Log-Loss with ElasticNet)

During training, the model minimizes:

$$\mathcal{L} = \underbrace{-\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\Big]}_{\text{Binary Cross-Entropy}} + \underbrace{\alpha\Big[\frac{1-\rho}{2}\|\mathbf{w}\|_2^2 + \rho\|\mathbf{w}\|_1\Big]}_{\text{ElasticNet Regularization}}$$

| Symbol | Value | Meaning |
|--------|:-----:|---------|
| $\alpha$ | tuned (e.g., 0.001) | Regularization strength |
| $\rho$ | 0.15 | L1 ratio (mix of L1 and L2) |
| $\|\mathbf{w}\|_1$ | $\sum|w_j|$ | L1 norm (promotes sparsity) |
| $\|\mathbf{w}\|_2^2$ | $\sum w_j^2$ | L2 norm (prevents large weights) |

### 3.4 SGD Weight Update

At each training step, for a single sample $(x_i, y_i)$:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla \mathcal{L}_i$$

where $\eta$ is the learning rate (decreasing schedule) and:

$$\nabla \mathcal{L}_i = (\hat{p}_i - y_i)\mathbf{x}_i + \alpha\big[(1-\rho)\mathbf{w} + \rho \cdot \text{sign}(\mathbf{w})\big]$$

### Worked Example (Mechanical group)

Suppose after training, the Mechanical model has:
- $w_1 = -0.85$ (weight for speed)
- $w_2 = +1.20$ (weight for torque)  
- $b = +0.35$ (bias)

For sample #4 with scaled features $\mathbf{x}_2^{(scaled)} = [-0.19, 0.99]$:

$$z = (-0.85)(-0.19) + (1.20)(0.99) + 0.35$$

$$z = 0.1615 + 1.188 + 0.35 = 1.6995$$

$$\hat{p}_2 = \sigma(1.6995) = \frac{1}{1 + e^{-1.6995}} = \frac{1}{1 + 0.1827} = 0.8454$$

**Interpretation**: The Mechanical model predicts an 84.5% probability of failure for this sample.

---

## 4. SMOTE Oversampling

**Problem**: The dataset has ~22% failures and ~78% normal — imbalanced.

**SMOTE** (Synthetic Minority Oversampling TEchnique) creates synthetic failure samples.

### Algorithm

For each minority class sample $\mathbf{x}_i$:

1. Find its $k=5$ nearest neighbors in the minority class
2. Randomly pick one neighbor $\mathbf{x}_{nn}$
3. Generate a synthetic sample:

$$\mathbf{x}_{new} = \mathbf{x}_i + \lambda \cdot (\mathbf{x}_{nn} - \mathbf{x}_i), \quad \lambda \sim U(0, 1)$$

### Worked Example

Suppose we have two failure samples:

$$\mathbf{x}_i = [298.5, 307.2] \quad (\text{Thermal features})$$
$$\mathbf{x}_{nn} = [301.3, 310.8] \quad (\text{a nearest neighbor})$$

With $\lambda = 0.4$:

$$\mathbf{x}_{new} = [298.5, 307.2] + 0.4 \cdot ([301.3, 310.8] - [298.5, 307.2])$$

$$= [298.5, 307.2] + 0.4 \cdot [2.8, 3.6]$$

$$= [298.5 + 1.12, \; 307.2 + 1.44]$$

$$= [299.62, \; 308.64]$$

This creates a **new synthetic failure sample** at $[299.62, 308.64]$ — a point along the line between the original sample and its neighbor.

**Critical rule**: SMOTE is applied **after** the train/test split to prevent data leakage. Synthetic samples never appear in the test set.

---

## 5. Isotonic Calibration

Raw SGD probabilities are often poorly calibrated (e.g., a prediction of 0.80 might actually correspond to 60% true failure rate).

**Isotonic calibration** learns a non-decreasing step function $f$ that maps raw scores to calibrated probabilities:

$$p_{calibrated} = f(z_{raw})$$

It is fit on a held-out calibration set by minimizing:

$$\min_f \sum_{i=1}^{n} (y_i - f(z_i))^2 \quad \text{subject to} \quad f(z_i) \leq f(z_j) \text{ whenever } z_i \leq z_j$$

### Worked Example

| Raw SGD score $z$ | Isotonic calibrated $p$ |
|:-:|:-:|
| -2.0 | 0.05 |
| -1.0 | 0.12 |
| 0.0 | 0.28 |
| 0.5 | 0.41 |
| 1.0 | 0.63 |
| 1.7 | 0.85 |
| 2.5 | 0.94 |

For our Mechanical model's score of $z = 1.6995$, the calibrated probability might be:

$$p_2^{cal} = f(1.6995) \approx 0.845$$

The calibration step ensures that when the model says "85% failure probability," roughly 85% of similar cases are indeed failures.

---

## 6. Binary Shannon Entropy

This is the **core innovation**. For each inner model's calibrated prediction $p$, we compute:

$$H(p) = -p \log_2(p) - (1-p) \log_2(1-p)$$

### Properties

| Property | Value |
|----------|-------|
| Domain | $p \in (0, 1)$ |
| Range | $H \in [0, 1]$ |
| Maximum | $H(0.5) = 1.0$ (maximum uncertainty) |
| Minimum | $H(0) = H(1) = 0$ (complete certainty) |
| Symmetry | $H(p) = H(1-p)$ |

### Detailed Numerical Examples

**Example A**: Confident prediction $p = 0.92$

$$H(0.92) = -(0.92)\log_2(0.92) - (0.08)\log_2(0.08)$$

Step by step:
- $\log_2(0.92) = \frac{\ln(0.92)}{\ln(2)} = \frac{-0.0834}{0.6931} = -0.1203$
- $\log_2(0.08) = \frac{\ln(0.08)}{\ln(2)} = \frac{-2.5257}{0.6931} = -3.6439$

$$H = -(0.92)(-0.1203) - (0.08)(-3.6439)$$

$$H = 0.1107 + 0.2915 = \boxed{0.4022}$$

**Interpretation**: Low entropy → the model is fairly confident.

---

**Example B**: Uncertain prediction $p = 0.55$

$$H(0.55) = -(0.55)\log_2(0.55) - (0.45)\log_2(0.45)$$

- $\log_2(0.55) = -0.8625$
- $\log_2(0.45) = -1.1520$

$$H = -(0.55)(-0.8625) - (0.45)(-1.1520)$$

$$H = 0.4744 + 0.5184 = \boxed{0.9928}$$

**Interpretation**: Very high entropy → the model is almost guessing (close to a coin flip).

---

**Example C**: Very confident prediction $p = 0.99$

$$H(0.99) = -(0.99)\log_2(0.99) - (0.01)\log_2(0.01)$$

- $\log_2(0.99) = -0.01449$
- $\log_2(0.01) = -6.6439$

$$H = -(0.99)(-0.01449) - (0.01)(-6.6439) = 0.01435 + 0.06644 = \boxed{0.0808}$$

**Interpretation**: Very low entropy → almost certain.

### Reference Table

| $p$ | $H(p)$ | Confidence Level |
|:---:|:------:|:----------------|
| 0.50 | 1.000 | Total uncertainty (coin flip) |
| 0.55 | 0.993 | Almost uncertain |
| 0.60 | 0.971 | Slightly leaning |
| 0.70 | 0.881 | Mild confidence |
| 0.80 | 0.722 | Moderate confidence |
| 0.90 | 0.469 | High confidence |
| 0.95 | 0.286 | Very high confidence |
| 0.99 | 0.081 | Near certain |

---

## 7. KL Divergence

The **Kullback-Leibler divergence** measures how much two groups' predictions disagree.

For two Bernoulli distributions with parameters $p_i$ and $p_j$:

$$D_{KL}(p_i \| p_j) = p_i \log_2\frac{p_i}{p_j} + (1-p_i)\log_2\frac{1-p_i}{1-p_j}$$

### Properties

| Property | Value |
|----------|-------|
| Range | $[0, \infty)$ — unbounded! |
| Zero when | $p_i = p_j$ (groups agree perfectly) |
| Asymmetric | $D_{KL}(p \| q) \neq D_{KL}(q \| p)$ in general |
| Large when | Groups strongly disagree |

### Worked Example

Suppose for one sample:
- Thermal predicts: $p_1 = 0.55$ (uncertain)
- Mechanical predicts: $p_2 = 0.92$ (confident failure)

$$D_{KL}(p_1 \| p_2) = (0.55)\log_2\frac{0.55}{0.92} + (0.45)\log_2\frac{0.45}{0.08}$$

Step by step:

**Term 1**: $0.55 \cdot \log_2(0.55 / 0.92) = 0.55 \cdot \log_2(0.5978)$

$$\log_2(0.5978) = \frac{\ln(0.5978)}{\ln(2)} = \frac{-0.5145}{0.6931} = -0.7425$$

$$\text{Term 1} = 0.55 \times (-0.7425) = -0.4084$$

**Term 2**: $0.45 \cdot \log_2(0.45 / 0.08) = 0.45 \cdot \log_2(5.625)$

$$\log_2(5.625) = \frac{\ln(5.625)}{\ln(2)} = \frac{1.7272}{0.6931} = 2.4919$$

$$\text{Term 2} = 0.45 \times 2.4919 = 1.1214$$

$$D_{KL}(0.55 \| 0.92) = -0.4084 + 1.1214 = \boxed{0.7130}$$

**Interpretation**: High KL divergence (0.713) means Thermal and Mechanical **strongly disagree** about this sample. Thermal says "maybe fail" while Mechanical says "definitely fail." This disagreement is itself a useful signal — it flags samples where the failure pathway is ambiguous.

### All Three Pairs

We compute KL divergence for all 3 group pairs:

$$D_{KL}^{12} = D_{KL}(p_{\text{Thermal}} \| p_{\text{Mechanical}})$$

$$D_{KL}^{13} = D_{KL}(p_{\text{Thermal}} \| p_{\text{Wear}})$$

$$D_{KL}^{23} = D_{KL}(p_{\text{Mechanical}} \| p_{\text{Wear}})$$

---

## 8. Entropy Contrast

The **absolute difference in entropy** between two groups:

$$\Delta H_{ij} = |H(p_i) - H(p_j)|$$

This tells us: "Is one group much more confident than another?"

### Worked Example

From examples above:
- $H_1 = H(0.55) = 0.993$ (Thermal — very uncertain)
- $H_2 = H(0.92) = 0.402$ (Mechanical — fairly confident)
- Suppose $H_3 = H(0.70) = 0.881$ (Wear — mildly confident)

$$\Delta H_{12} = |0.993 - 0.402| = \boxed{0.591}$$

$$\Delta H_{13} = |0.993 - 0.881| = \boxed{0.112}$$

$$\Delta H_{23} = |0.402 - 0.881| = \boxed{0.479}$$

**Interpretation**:
- $\Delta H_{12} = 0.591$: Large gap → Mechanical is MUCH more confident than Thermal
- $\Delta H_{13} = 0.112$: Small gap → Thermal and Wear have similar uncertainty
- $\Delta H_{23} = 0.479$: Medium gap → Mechanical more confident than Wear

---

## 9. Meta-Feature Vector Construction

For each sample, we assemble a **12-dimensional meta-feature vector**:

$$\mathbf{m} = \begin{bmatrix} p_1 \\ H_1 \\ p_2 \\ H_2 \\ p_3 \\ H_3 \\ D_{KL}^{12} \\ D_{KL}^{13} \\ D_{KL}^{23} \\ \Delta H_{12} \\ \Delta H_{13} \\ \Delta H_{23} \end{bmatrix}$$

### Worked Example (Complete)

Using values from all examples above:

| Index | Name | Value | Source |
|:-----:|------|:-----:|--------|
| 0 | $p_1$ | 0.550 | Thermal prediction |
| 1 | $H_1$ | 0.993 | Thermal entropy |
| 2 | $p_2$ | 0.920 | Mechanical prediction |
| 3 | $H_2$ | 0.402 | Mechanical entropy |
| 4 | $p_3$ | 0.700 | Wear prediction |
| 5 | $H_3$ | 0.881 | Wear entropy |
| 6 | $D_{KL}^{12}$ | 0.713 | Thermal↔Mechanical disagreement |
| 7 | $D_{KL}^{13}$ | 0.128 | Thermal↔Wear disagreement |
| 8 | $D_{KL}^{23}$ | 0.389 | Mechanical↔Wear disagreement |
| 9 | $\Delta H_{12}$ | 0.591 | Thermal–Mechanical entropy gap |
| 10 | $\Delta H_{13}$ | 0.112 | Thermal–Wear entropy gap |
| 11 | $\Delta H_{23}$ | 0.479 | Mechanical–Wear entropy gap |

$$\mathbf{m} = [0.55, 0.993, 0.92, 0.402, 0.70, 0.881, 0.713, 0.128, 0.389, 0.591, 0.112, 0.479]$$

### Ablation Variant (No Entropy)

In the ablation, we use only 3 features (predictions only):

$$\mathbf{m}_{ablation} = [p_1, p_2, p_3] = [0.55, 0.92, 0.70]$$

The 9 additional information-theoretic features ($H$, $D_{KL}$, $\Delta H$) are what we prove are statistcially significant via the paired t-test.

---

## 10. Outer Classifier

The outer SGD meta-classifier operates on the 12-dimensional meta-feature space:

$$z_{outer} = \mathbf{w}_{outer}^T \mathbf{m} + b_{outer} = \sum_{k=0}^{11} w_k m_k + b$$

$$\hat{p}_{final} = \sigma(z_{outer})$$

### What the Outer Weights Learn

| Weight | What it learns |
|--------|---------------|
| $w_0$ (for $p_1$) | How much to trust Thermal group's prediction |
| $w_1$ (for $H_1$) | How to weigh Thermal uncertainty |
| $w_2$ (for $p_2$) | How much to trust Mechanical group |
| $w_3$ (for $H_2$) | How to weigh Mechanical uncertainty |
| $w_6$ (for $D_{KL}^{12}$) | How disagreement between Thermal and Mechanical affects the prediction |

### Worked Example

Suppose the outer model has weights:

$$\mathbf{w}_{outer} = [0.3, -0.5, 1.8, -0.9, 0.6, -0.3, 0.7, 0.2, 0.4, 0.5, 0.1, 0.3], \quad b = -1.2$$

Then for our meta-vector:

$$z = (0.3)(0.55) + (-0.5)(0.993) + (1.8)(0.92) + (-0.9)(0.402) + (0.6)(0.70) + (-0.3)(0.881)$$
$$\quad + (0.7)(0.713) + (0.2)(0.128) + (0.4)(0.389) + (0.5)(0.591) + (0.1)(0.112) + (0.3)(0.479) - 1.2$$

Computing each term:

| Term | $w_k \times m_k$ |
|------|:-:|
| $0.3 \times 0.55$ | 0.165 |
| $-0.5 \times 0.993$ | -0.497 |
| $1.8 \times 0.92$ | 1.656 |
| $-0.9 \times 0.402$ | -0.362 |
| $0.6 \times 0.70$ | 0.420 |
| $-0.3 \times 0.881$ | -0.264 |
| $0.7 \times 0.713$ | 0.499 |
| $0.2 \times 0.128$ | 0.026 |
| $0.4 \times 0.389$ | 0.156 |
| $0.5 \times 0.591$ | 0.296 |
| $0.1 \times 0.112$ | 0.011 |
| $0.3 \times 0.479$ | 0.144 |
| bias | -1.200 |
| **Total** | **1.050** |

$$\hat{p}_{final} = \sigma(1.050) = \frac{1}{1+e^{-1.050}} = \frac{1}{1+0.3499} = \boxed{0.741}$$

**Final prediction**: 74.1% probability of failure.

**Interpretation of key weights**:
- $w_2 = 1.8$ (Mechanical prediction): The model trusts the Mechanical group the most
- $w_3 = -0.9$ (Mechanical entropy): **Negative** — low entropy (high confidence) pushes prediction higher. When the Mechanical model is sure, the outer model listens more.
- $w_6 = 0.7$ (KL Thermal↔Mechanical): Disagreement between groups adds evidence for failure — conflicting signals are a warning sign

---

## 11. Youden's J Threshold

Instead of using the default threshold of $t = 0.5$, we find the **optimal threshold** by maximizing **Youden's J statistic**:

$$J(t) = \text{TPR}(t) - \text{FPR}(t) = \text{Sensitivity}(t) + \text{Specificity}(t) - 1$$

$$t^* = \arg\max_{t} J(t)$$

### Worked Example

Suppose from the ROC curve:

| Threshold $t$ | TPR | FPR | $J = \text{TPR} - \text{FPR}$ |
|:---:|:---:|:---:|:---:|
| 0.30 | 0.95 | 0.40 | 0.55 |
| 0.40 | 0.90 | 0.25 | 0.65 |
| 0.50 | 0.82 | 0.15 | **0.67** |
| 0.60 | 0.70 | 0.08 | 0.62 |
| 0.70 | 0.55 | 0.04 | 0.51 |

The maximum $J = 0.67$ occurs at $t^* = 0.50$.

For our example sample with $\hat{p}_{final} = 0.741 > t^* = 0.50$:

$$\hat{y} = 1 \quad (\text{FAILURE predicted})$$

---

## 12. Evaluation Metrics

### 12.1 Confusion Matrix

For a binary classifier:

|  | Predicted Failure | Predicted Normal |
|--|:-:|:-:|
| **Actual Failure** | TP (True Positive) | FN (False Negative) |
| **Actual Normal** | FP (False Positive) | TN (True Negative) |

### 12.2 AUC-ROC

$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))$$

**Probabilistic interpretation**: If you randomly pick one failure sample and one normal sample, the AUC is the probability that the model assigns a higher score to the failure sample.

$$\text{AUC} = P(\hat{p}_{\text{failure}} > \hat{p}_{\text{normal}})$$

Example: AUC = 0.884 means in 88.4% of random pairs, the model correctly ranks the failure sample higher.

### 12.3 Matthews Correlation Coefficient (MCC)

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

**Range**: [-1, 1]. Perfectly correct = +1, random = 0, perfectly wrong = -1.

**Example**: Suppose over 1000 test samples:

| | Pred Fail | Pred Normal | Total |
|--|:-:|:-:|:-:|
| Actual Fail | TP=180 | FN=40 | 220 |
| Actual Normal | FP=100 | TN=680 | 780 |

$$\text{MCC} = \frac{(180)(680) - (100)(40)}{\sqrt{(280)(220)(780)(720)}}$$

Numerator: $122400 - 4000 = 118400$

Denominator: $\sqrt{280 \times 220 \times 780 \times 720} = \sqrt{34,594,560,000} = 185,998$

$$\text{MCC} = \frac{118400}{185998} = \boxed{0.636}$$

### 12.4 F1 Score

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Where:

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{180}{280} = 0.643$$

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{180}{220} = 0.818$$

$$F_1 = 2 \cdot \frac{0.643 \times 0.818}{0.643 + 0.818} = 2 \cdot \frac{0.526}{1.461} = \boxed{0.720}$$

### 12.5 Brier Score (Calibration Quality)

$$\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{p}_i)^2$$

**Range**: [0, 1]. Lower = better calibrated.

**Example** for 5 samples:

| $y_i$ (actual) | $\hat{p}_i$ (predicted) | $(y_i - \hat{p}_i)^2$ |
|:-:|:-:|:-:|
| 1 | 0.90 | 0.0100 |
| 0 | 0.15 | 0.0225 |
| 1 | 0.70 | 0.0900 |
| 0 | 0.30 | 0.0900 |
| 1 | 0.85 | 0.0225 |

$$\text{Brier} = \frac{0.01 + 0.0225 + 0.09 + 0.09 + 0.0225}{5} = \frac{0.235}{5} = \boxed{0.047}$$

---

## 13. Ablation Study

### 13.1 Paired t-test

We compare AUC scores from 10 CV folds. The **null hypothesis** $H_0$ is: "Entropy + KL features provide no improvement" (i.e., $\mu_d = 0$).

We compare AUC from the **Full model** (12 meta-features) vs the **No-Entropy baseline** (3 meta-features):

| Fold | $A_{\text{full}}$ | $A_{\text{no-entropy}}$ | $d_i = A_{\text{full}} - A_{\text{no-entropy}}$ |
|:----:|:-:|:-:|:-:|
| 1 | 0.884 | 0.858 | +0.026 |
| 2 | 0.891 | 0.876 | +0.015 |
| 3 | 0.886 | 0.878 | +0.008 |
| 4 | 0.868 | 0.880 | −0.012 |
| 5 | 0.898 | 0.877 | +0.021 |
| 6 | 0.895 | 0.868 | +0.027 |
| 7 | 0.886 | 0.884 | +0.002 |
| 8 | 0.879 | 0.882 | −0.003 |
| 9 | 0.872 | 0.852 | +0.020 |
| 10 | 0.877 | 0.858 | +0.019 |

Note: Some folds show **negative** differences — this is normal. The question is whether the *average* improvement is significantly non-zero.

**Step 1**: Compute mean difference

$$\bar{d} = \frac{\sum d_i}{n} = \frac{0.026 + 0.015 + 0.008 + (-0.012) + 0.021 + 0.027 + 0.002 + (-0.003) + 0.020 + 0.019}{10}$$

$$\bar{d} = \frac{0.123}{10} = 0.0123$$

**Step 2**: Compute standard deviation of differences

First, compute each $(d_i - \bar{d})^2$:

| Fold | $d_i$ | $d_i - \bar{d}$ | $(d_i - \bar{d})^2$ |
|:----:|:-----:|:-----:|:-----:|
| 1 | +0.026 | +0.0137 | 0.000188 |
| 2 | +0.015 | +0.0027 | 0.000007 |
| 3 | +0.008 | −0.0043 | 0.000018 |
| 4 | −0.012 | −0.0243 | 0.000590 |
| 5 | +0.021 | +0.0087 | 0.000076 |
| 6 | +0.027 | +0.0147 | 0.000216 |
| 7 | +0.002 | −0.0103 | 0.000106 |
| 8 | −0.003 | −0.0153 | 0.000234 |
| 9 | +0.020 | +0.0077 | 0.000059 |
| 10 | +0.019 | +0.0067 | 0.000045 |
| | | **Sum** | **0.001539** |

$$s_d = \sqrt{\frac{0.001539}{10 - 1}} = \sqrt{\frac{0.001539}{9}} = \sqrt{0.000171} = 0.01308$$

**Step 3**: Compute t-statistic

$$t = \frac{\bar{d}}{s_d / \sqrt{n}} = \frac{0.0123}{0.01308 / \sqrt{10}} = \frac{0.0123}{0.004136} = 2.974$$

**Step 4**: Determine p-value and compare to critical value

With $df = n - 1 = 9$ degrees of freedom and $\alpha = 0.05$ (two-tailed):

$$t_{critical}(df=9, \alpha=0.05) = 2.262$$

Since $|t| = 2.974 > t_{critical} = 2.262$:

$$\boxed{p = 0.0156 < 0.05 \quad \Rightarrow \quad \text{Reject } H_0}$$

**Conclusion**: The entropy and KL divergence meta-features provide a **statistically significant improvement** in AUC (paired t-test, $t(9) = 2.974$, $p = 0.016$).

**Interpretation**: Even though 2 out of 10 folds showed the no-entropy model slightly edging ahead, the overall trend strongly favors the full model. The probability of observing this improvement by pure chance is only 1.6%.

---

## 14. Full Worked Example (End-to-End)

Let's trace one sample through the complete pipeline:

### Input

```
Air temperature     = 303.5 K
Process temperature = 312.8 K
Rotational speed    = 1250 rpm
Torque              = 55.0 Nm
Tool wear           = 210 min
Type                = L (encoded as 0)
```

### Step 1: Group & Scale

| Group | Raw | Scaled (z-score) |
|-------|-----|:-:|
| Thermal | [303.5, 312.8] | [+1.75, +1.24] |
| Mechanical | [1250, 55.0] | [-1.48, +1.50] |
| Wear | [210, 0] | [+1.31, -1.00] |

### Step 2: Inner-loop predictions

| Group | SGD Score $z$ | Calibrated $p$ |
|-------|:---:|:---:|
| Thermal | 0.42 | $p_1 = 0.60$ |
| Mechanical | 2.85 | $p_2 = 0.95$ |
| Wear | 1.10 | $p_3 = 0.75$ |

### Step 3: Compute entropy

| Group | $p$ | $H(p)$ |
|-------|:---:|:------:|
| Thermal | 0.60 | 0.971 |
| Mechanical | 0.95 | 0.286 |
| Wear | 0.75 | 0.811 |

### Step 4: Compute KL divergence

$$D_{KL}^{12} = (0.60)\log_2\frac{0.60}{0.95} + (0.40)\log_2\frac{0.40}{0.05} = (0.60)(-0.664) + (0.40)(3.0) = -0.398 + 1.200 = 0.802$$

$$D_{KL}^{13} = (0.60)\log_2\frac{0.60}{0.75} + (0.40)\log_2\frac{0.40}{0.25} = (0.60)(-0.322) + (0.40)(0.678) = -0.193 + 0.271 = 0.078$$

$$D_{KL}^{23} = (0.95)\log_2\frac{0.95}{0.75} + (0.05)\log_2\frac{0.05}{0.25} = (0.95)(0.341) + (0.05)(-2.322) = 0.324 - 0.116 = 0.208$$

### Step 5: Compute entropy contrast

$$\Delta H_{12} = |0.971 - 0.286| = 0.685$$

$$\Delta H_{13} = |0.971 - 0.811| = 0.160$$

$$\Delta H_{23} = |0.286 - 0.811| = 0.525$$

### Step 6: Assemble meta-vector

$$\mathbf{m} = [0.60, 0.971, 0.95, 0.286, 0.75, 0.811, 0.802, 0.078, 0.208, 0.685, 0.160, 0.525]$$

### Step 7: Outer model → Final prediction

$$z_{outer} = \mathbf{w}_{outer}^T \mathbf{m} + b_{outer} = 2.15$$

$$\hat{p}_{final} = \sigma(2.15) = \frac{1}{1+e^{-2.15}} = 0.896$$

### Step 8: Apply threshold

With $t^* = 0.50$: since $0.896 > 0.50$:

$$\boxed{\hat{y} = 1 \quad \text{(FAILURE PREDICTED)}}$$

### Step 9: Interpretability Report

```
FAILURE PREDICTION — Confidence: 89.6%
┌──────────────────────────────────────────────┐
│  Signal Group    Prediction  Entropy  Status │
│  ─────────────   ──────────  ───────  ────── │
│  🌡️ Thermal      60.0%       0.971   ⚠️ UNCERTAIN   │
│  ⚙️ Mechanical    95.0%       0.286   ✅ CONFIDENT   │
│  🔧 Wear          75.0%       0.811   ⚠️ MODERATE    │
│                                               │
│  Cross-group disagreement:                    │
│  Thermal ↔ Mechanical: 0.802 (HIGH)          │
│  Thermal ↔ Wear: 0.078 (low)                 │
│  Mechanical ↔ Wear: 0.208 (moderate)         │
│                                               │
│  ➡️ ACTION: Check mechanical subsystem         │
│    (speed/torque anomaly detected)            │
│    Thermal signals are inconclusive.          │
│    Tool wear is high (210 min) — replace soon │
└──────────────────────────────────────────────┘
```

**This is the unique value proposition**: No black-box model (RF, XGBoost, neural network) can generate this breakdown without post-hoc explainability tools like SHAP. Our method produces it **natively**.

---

## Summary of All Formulas

| # | Formula | Purpose |
|:-:|---------|---------|
| 1 | $z_j = (x_j - \mu_j) / \sigma_j$ | Feature normalization |
| 2 | $\hat{p} = \sigma(\mathbf{w}^T\mathbf{x} + b) = 1/(1+e^{-z})$ | SGD prediction |
| 3 | $\mathcal{L} = -\frac{1}{N}\sum[y\log\hat{p} + (1-y)\log(1-\hat{p})] + \alpha R(\mathbf{w})$ | Training loss |
| 4 | $\mathbf{x}_{new} = \mathbf{x}_i + \lambda(\mathbf{x}_{nn} - \mathbf{x}_i)$ | SMOTE synthesis |
| 5 | $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$ | Binary entropy |
| 6 | $D_{KL}(p_i \| p_j) = p_i\log_2\frac{p_i}{p_j} + (1-p_i)\log_2\frac{1-p_i}{1-p_j}$ | KL divergence |
| 7 | $\Delta H_{ij} = \|H(p_i) - H(p_j)\|$ | Entropy contrast |
| 8 | $J(t) = \text{TPR}(t) - \text{FPR}(t)$ | Youden's J |
| 9 | $\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | Matthews Correlation |
| 10 | $F_1 = 2PR/(P+R)$ | F1 score |
| 11 | $\text{Brier} = \frac{1}{N}\sum(y_i - \hat{p}_i)^2$ | Calibration quality |
| 12 | $t = \bar{d} / (s_d / \sqrt{n})$ | Paired t-test |
