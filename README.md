# Entropy-Calibrated Multi-View Failure Prediction Pipeline

This repository contains the code and mathematical proofs for our predictive maintenance pipeline, demonstrating how information-theoretic signals (Entropy & KL Divergence) can make lightweight diagnostic models outperform traditional linear approaches on industrial sensor data.

## System Configuration & Hardware Requirements

This code is written to be extremely lightweight, utilizing Stochastic Gradient Descent (SGD). **No GPU or specialized hardware is required.**

- **OS:** macOS, Linux, or Windows (WSL recommended)
- **CPU:** Standard multi-core processor (Intel i5/i7 or Apple Silicon M1/M2)
- **RAM:** 8 GB minimum (16 GB recommended for faster cross-validation execution)
- **Disk Space:** ~50 MB for the repository, datasets, and generated image artifacts.
- **Python Version:** 3.9, 3.10, or 3.11

---

## Installation & Setup

We recommend running this pipeline inside an isolated virtual environment. 

### 1. Clone the Repository
```bash
git clone https://github.com/Aman-Git-hala/Entropy-failure-detection.git
cd Entropy-failure-detection
```

### 2. Create and Activate a Virtual Environment
**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
All necessary libraries are listed in `requirements.txt`.
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Pipeline

The entire pipeline is self-contained. You do not need to download the dataset manually; the pipeline will automatically generate or load the required AI4I 2020 Predictive Maintenance dataset.

### Step 1: Run The Pipeline
Execute the main Python script located in the `otel_failure_prediction` directory:

```bash
cd otel_failure_prediction
python bilevel_entropy_maintenance.py
```

### Step 2: What Happens During Execution?
Once you hit enter, the system will automatically:
1. **Load data:** Download/generate the 10,000-sample `ai4i_2020.csv` dataset.
2. **Execute SMOTE Strategy:** Dynamically handle the 22% class imbalance.
3. **Train Models (10-fold CV):** Train the inner specialist SGD models (Thermal, Mechanical, Wear) and the outer Meta-Classifier.
4. **Grid-Search Tuning:** Optimize hyperparameters (like alpha) per fold.
5. **Run Baselines:** Simultaneously train Naive Bayes, Logistic Regression, Random Forest, and XGBoost on exactly the same data splits to ensure a fair comparison.
6. **Perform Statistical Testing:** Run the paired t-test against the ablation (no-entropy) model.

*(Execution time usually ranges from 15 to 45 seconds depending on CPU power).*

### Step 3: Analyze the Output
When execution finishes, check the terminal output for the performance tables (AUC, MCC, Brier) and the statistical significance results.

Five publication-ready PNG figures will be auto-generated inside the `otel_failure_prediction/` directory:
- `fig1_entropy_distribution.png`: Why Mechanical entropy reveals hidden insight.
- `fig2_method_comparison.png`: The ROC curves comparing us against baselines.
- `fig3_roc_curve.png`: How we separate the classes.
- `fig4_ablation_study.png`: Visual proof of the impact of Entropy & KL features.
- `fig5_meta_feature_shift.png`: The exact weights and shifts of the 12-dimensional vector.

---

## Directory Documentation

If you want to read deeper into the logic of what is happening under the hood, we have prepared detailed Markdown files:
- **`shashi.md`** - A casual, high-level summary of the architectural pivot and thought processes behind the project.
- **`otel_failure_prediction/mathematical_explanation.md`** - A comprehensive breakdown of every algebraic formula, complete with a step-by-step numerical walkthrough of the 12-D vector.
- **`otel_failure_prediction/explanation.md`** - A high-level technical overview of the semantic groupings and pipeline design.
- **`otel_failure_prediction/comparison.md`** - The baseline benchmarking results and where our algorithm holds its unique interpretability advantages.
