# SoundPulse — End‑to‑End ML on Spotify: From EDA to Explainable Predictions

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python\&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg?logo=scikitlearn\&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-GBDT-66CC66.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-8A2BE2.svg)](https://shap.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg?logo=Jupyter\&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-9cf.svg)](#license)

> **Executive summary.** This repository delivers a complete ML workflow on Spotify tracks: **EDA → Outlier Detection → Exploratory Clustering → Multiclass Genre Classification (10 classes from 114 raw genres, including a deep learning model: TensorFlow/Keras MLP with LayerNorm + GELU, decaying Dropout 0.45→0.10, AdamW with warm-up + cosine annealing, class-weighted training, early stopping & checkpoints) → Popularity Regression**, with **explainability** (global & local) and **production-minded** evaluation.


---

## Table of Contents

* [Why it matters](#why-it-matters)
* [Project scope](#project-scope)
* [Data & EDA](#data--eda)
* [Outlier Detection](#outlier-detection)
* [Exploratory Clustering](#exploratory-clustering)
* [Genre Classification (10 classes)](#genre-classification-10-classes)
* [Explainability](#explainability)
* [Popularity Regression](#popularity-regression)
* [Imbalanced Learning](#imbalanced-learning)
* [Project structure](#project-structure)
* [Reproducible setup](#reproducible-setup)
* [How to run](#how-to-run)
* [Models & evaluation](#models-and-evaluation)
* [Highlights](#highlights)
* [Roadmap](#roadmap)
* [Hire me for](#hire-me-for)
* [License](#license)

---

## Why it matters

* **Predict what resonates.** Anticipate track **popularity** to inform promo, playlisting, and spend.
* **Understand the why.** Feature‑level drivers via SHAP & permutation importance translate models into **actionable levers** (danceability, energy, tempo, etc.).
* **De‑risk** with **anomaly screening** and **segment discovery** before committing budget.
* **Operational rigor.** Reproducible CV, artifact persistence, consistent reporting, clean handoff.

---

## Project scope

This repo covers the full ML lifecycle on Spotify audio features:

1. **Data Understanding & Preparation** — rigorous cleaning, typing, scaling; stratified splits with seed control.
2. **Outlier Detection** — LOF to identify anomalous tracks and guide data quality decisions.
3. **Exploratory Clustering** — discover latent structures and micro‑genres to inform marketing personas.
4. **Multiclass Genre Classification (10 classes)** — compress 114 raw genres into **10 consolidated** classes and train interpretable classifiers.
5. **Popularity Regression** — compare linear vs. non‑linear regressors under a uniform cross‑validation and diagnostics protocol.

---

## Data & EDA

**Source.** Spotify track‑level audio features (e.g., `danceability`, `energy`, `acousticness`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`, `duration_ms`, plus metadata like `key`, `mode`, `time_signature`).

**EDA focus.**

* Sanity checks (missingness, typing), scaling/standardization for model comparability.
* Distributional analysis and pairwise relationships to detect leakage or degenerate features.
* Target exploration: class counts for the **10‑genre taxonomy** and distribution of **popularity** for regression tasks.

See: `notebooks/1_data_preparation.ipynb`.

---

## Outlier Detection

**Method.** Local Outlier Factor (LOF) on standardized features.

**Why.** Early anomaly flagging improves data reliability and helps A\&R/QA triage unusual tracks (production artifacts, metadata errors, extreme audio profiles).

**Deliverables.**

* Outlier scores & flags.
* Visual diagnostics: density plots, neighborhood consistency.

See: `notebooks/2_outlier_detection.ipynb`.

---

## Exploratory Clustering

**Goal.** Uncover latent segments/micro‑genres to support curation and targeted campaigns.

**Approach.** K‑means (and variants where relevant) on standardized features; selection guided by inertia/`silhouette_score` and business plausibility.

**Outputs.**

* Cluster assignments and centroids.
* Profile cards per cluster (top features, representative tracks).

See: `notebooks/3_clustering.ipynb`.

---

## Genre Classification (10 classes)

**From 114 → 10.** Raw Spotify genres were consolidated into **10 macro‑classes** via frequency thresholds and semantic grouping to balance class sizes and interpretability.

**Models.**

* **RandomForestClassifier** — strong non‑linear baseline, robust to feature scaling and outliers; used with stratified splits and consistent reporting.
* **Advanced Neural Network (Deep MLP, TensorFlow/Keras)** — designed for class‑imbalance, stability, and expressive power:

  * **Input:** standardized feature vector; optional `GaussianNoise(0.1)` for regularization.
  * **Hidden stack (representative):** `1536 → 768 → 768 → 768 → 512 → 384 → 384 → 256 → 128 → 64` units.
  * **Blocks:** each dense layer is followed by `LayerNormalization → GELU → Dropout`, with a **decaying dropout schedule** `0.45 → 0.40 → 0.35 → 0.30 → 0.25 → 0.20 → 0.15 → 0.10`.
  * **Regularization & inits:** `HeNormal` on most hidden layers; pervasive **L2=3e‑4** (and selective **L1/L2** on the output layer); one **sigmoid gating** dense at 768 units for additional non‑linearity control; final **Softmax** over 10 classes.
  * **Optimizer:** **AdamW** (lr **5e‑4**, β=(0.9, 0.999), ε=1e‑7) with **gradient clipping** (`clipnorm=1.0`).
  * **Schedule:** **Warm‑up 10 epochs → Cosine annealing** for the remaining training.
  * **Training setup:** `epochs_total=350`, `batch_size=4096`, `validation_split=0.2`, **class weights** via `compute_class_weight('balanced', ...)`.
  * **Callbacks:** `EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True)`, `ReduceLROnPlateau(patience=15, min_lr=1e‑7)`, `ModelCheckpoint(save_best_only=True, monitor='val_loss')`.
  * **Loss & metrics:** `sparse_categorical_crossentropy`; metrics include **accuracy**, **Top‑3** and **Top‑5 accuracy**.

**Protocol.**

* Classical models use stratified splits and, where applicable, **RepeatedKFold** CV with multi‑metric scoring (Accuracy, Macro‑F1, ROC‑AUC OvR when appropriate).
* The neural model uses a held‑out **validation split (20%)** with early stopping and LR scheduling; artifacts and plots are persisted like the other models.

**Reporting.**

* Confusion matrices and per‑class metrics; Macro‑F1 and precision/recall breakdowns.
* SHAP **waterfall plots** for local explanations on correctly/incorrectly classified examples.

See: `notebooks/4_classification_explainability.ipynb`.

## Explainability

**Global.** Permutation importance to rank universal drivers; SHAP summary plots to visualize feature impact distribution.

**Local.** SHAP waterfall plots for per‑track narratives: why this song is predicted as a given genre or why its popularity is high/low.

**Value.** Moves beyond accuracy into **decision support** for A\&R, curation, and marketing.

---

## Popularity Regression

**Objective.** Predict track **popularity** as a continuous target to prioritize releases and campaigns.

**Models (uniform protocol).**

* **Ridge Regression**, **Linear SVR** (linear baselines).
* **RandomForestRegressor**, **XGBoostRegressor**, **GradientBoostingRegressor** (non‑linear ensembles).

**Evaluation.**

* RepeatedKFold CV with multi‑metric scoring (**R²**, **RMSE**, **MAE**), parity plots, residual diagnostics, and permutation importance.
* Best estimators and CV logs persisted for auditability and rapid iteration.

See: `notebooks/5_regression.ipynb`.

---

## Imbalanced Learning

**Objective** Predict very imbalanced "explicit" binary variable (91% - 9%) using advanced techniques like SMOTE / Tomek-Links

**Models** SMOTE / SMOTE-Tomek / undersampling pipelines; class weights and sample weights; 

**Focal Loss** for the deep MLP; 

**metrics**: **Macro-F1** and **Balanced Accuracy**.

---

## Project structure

```text
.
├── notebooks/
│   ├── 1_data_preparation.ipynb
│   ├── 2_outlier_detection.ipynb
│   ├── 3_clustering.ipynb
│   ├── 4_classification_explainability.ipynb
│   └── 5_regression.ipynb
├── results/
│   ├── *_cv_results.csv
│   ├── *_best_params.json
│   └── metrics_snapshots/*.json
├── models/
│   └── *.joblib
├── plots/
│   └── *.png
├── README.md
└── requirements.txt
```

> Notebooks are modular and write downstream artifacts to ensure incremental reproducibility.

---

## Reproducible setup

```bash
# 1) Create environment
conda create -n spotify-ml python=3.10 -y
conda activate spotify-ml

# 2) Install deps
pip install -r requirements.txt

# 3) Start notebooks
jupyter lab
```

**Minimal `requirements.txt`:**

```txt
numpy
pandas
scikit-learn
xgboost
shap
matplotlib
joblib
```

---

## How to run

1. Run `notebooks/1_data_preparation.ipynb` to produce cleaned splits (`X_train`, `X_test`, `y_train`, `y_test`).
2. Run `notebooks/2_outlier_detection.ipynb` to flag anomalies (optional but recommended for QA).
3. Run `notebooks/3_clustering.ipynb` for latent segment discovery.
4. Run `notebooks/4_classification_explainability.ipynb` for 10‑class genre models + SHAP.
5. Run `notebooks/5_regression.ipynb` for popularity forecasting (RF/XGB/GBR/Ridge/SVR).

All stages persist models (`models/`), CV logs (`results/`), and plots (`plots/`).

---

## Models and evaluation

| Task           | Models                                                                 | Primary metrics                                        | Notes                                                          |
| -------------- | ---------------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------- |
| Classification | Linear SVC, RandomForestClassifier, **Deep MLP (Keras: LayerNorm+GELU, decaying Dropout 0.45→0.10, AdamW warm-up+cosine)** with **Imbalanced Learning** (SMOTE/SMOTE-Tomek/undersampling, class/sample weights, Focal Loss) | Accuracy, **Balanced Accuracy**, **Macro-F1**, Top-3/Top-5, ROC-AUC (OvR) | 10 classes (114 → 10); SHAP waterfalls; confusion matrices     |
| Regression     | Ridge, Linear SVR, RF, XGB, GBR                                        | R², RMSE, MAE                                           | Uniform CV, parity/residual diagnostics, permutation importance |

---

## Highlights

* **Full‑stack ML**: from EDA to decisions, not just a single model.
* **Explainability‑first**: SHAP (global & local) + permutation importance embedded in the workflow.
* **Engineering discipline**: deterministic splits, RepeatedKFold CV, artifact persistence, clean plots.
* **Stakeholder alignment**: outputs framed for A\&R, curation, and marketing decisions.

---

## Roadmap

* Time‑aware evaluation across release windows.
* Model calibration & conformal prediction for uncertainty‑aware decisions.
* Package regression/classification pipelines as a **CLI** with config management.
* Optional **Streamlit** dashboard for non‑technical stakeholders.

---

## Hire me for

- **End-to-end Data Science & ML** — from EDA to deployment-ready artifacts: reproducible CV, persisted models/metrics/plots, and clean hand-offs.
- **Explainable classification at scale** — 10-class taxonomy (114 → 10) with a class-weighted **Deep MLP** (LayerNorm + GELU, decaying Dropout 0.45→0.10, AdamW warm-up + cosine) alongside strong baselines (Linear SVC, RandomForest).
- **Robust regression & forecasting** — Ridge / Linear SVR / RF / XGB / GBR with parity & residual diagnostics and permutation importance for driver analysis.
- **MLOps-minded delivery** — deterministic seeds, modular repo, `joblib` models, `results/*.csv` logs, and CI-friendly structure; GPU/Apple-Silicon aware.
- **Stakeholder-ready storytelling** — SHAP waterfalls and summary plots distilled into concise, decision-oriented reports for A&R, curation, and marketing.
- **Fast iteration with measurable ROI** — tight feedback loops, metric tracking, and experiment logs that make progress visible and comparable.


**Giuseppe Muschetta**
*M.Sc. in Computer Science (Data Science & Business Informatics), University of Pisa*

* GitHub: https://github.com/peppe212
* LinkedIn: https://www.linkedin.com/in/giuseppemuschetta/
* Email: peppe212@gmail.com

---

## License

This project is licensed under the [MIT License](LICENSE).

