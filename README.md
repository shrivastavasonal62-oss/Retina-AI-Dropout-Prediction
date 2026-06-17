# RETINA AI — Multimodal Deep Learning for Student Dropout Prediction

A multimodal machine learning system that predicts student dropout risk by fusing three heterogeneous data sources: academic records, attendance time-series, and unstructured counsellor notes. Built for the RETINA AI Hackathon hosted by ABES Engineering College.

## Problem

Most dropout-prediction systems rely on a single data source, usually grades. This misses two critical signals: behavioral consistency over time, and the qualitative context that academic counsellors observe but never gets digitized into a model. RETINA AI fuses all three into one prediction pipeline, classifying students into Low, Medium, or High dropout risk.

## Architecture

| Module | Data source | Model | Purpose |
|---|---|---|---|
| A | Academic tabular data (CGPA, backlogs, demographics) | LightGBM, 5-fold stratified CV | Structured academic signal |
| B | Weekly attendance across 3 semesters | Bidirectional LSTM | Temporal/behavioral signal |
| C | Counsellor notes (free text) | TF-IDF + sentiment + Logistic Regression | Qualitative/contextual signal |
| D | Outputs of A, B, C | Empirically searched weighted ensemble | Final fused prediction, weights chosen by grid search against validation F1 (not assumed) |

A diagram of the full pipeline is in `images/architecture.png`.

## Key engineering decisions

**Bidirectional LSTM over standard LSTM.** Attendance sequences were reshaped into `(24 timesteps × 4 features)` per student — 3 semesters × 8 weeks, with per-subject and mean attendance as features. A bidirectional architecture reads each sequence forward and backward, which matters here: a student who started strong and declined late in the semester needs to be flagged differently than one who was inconsistent from week one. A unidirectional LSTM weights recent timesteps more heavily by construction and would miss the first pattern.

**Engineered features over raw columns.** Rather than feeding raw semester-wise CGPA into the tabular model, I derived `cgpa_trend` (sem4 − sem1), `total_backlogs`, `backlog_trend`, and a composite `academic_stress` score (`total_backlogs × (10 − mean_cgpa)`). These compress multi-column relationships into single signals that LightGBM's tree splits can exploit directly, which mattered more here than just throwing more raw columns at the model.

**Why fusion weights are searched, not fixed.** The first version of this pipeline used fixed weights (50% LightGBM / 30% LSTM / 20% NLP) chosen by intuition about which modality "should" matter most. That assumption was wrong: NLP was actually the strongest standalone module by a wide margin, so giving it the smallest weight pulled the fused prediction below NLP's own standalone score. The fix was a grid search over weight combinations evaluated against out-of-fold weighted F1, which is the only way to know the right weighting without guessing. See the Results section below for the full before/after.

**TF-IDF + handcrafted lexicon over raw embeddings.** Given a relatively small, domain-specific text corpus (counsellor notes), TF-IDF bigrams combined with a manually curated risk/positive keyword lexicon outperformed generic embeddings in early testing — the lexicon directly encodes domain vocabulary ("backlog," "irregular," "withdraw") that generic embeddings don't prioritize.

## Results

5-fold stratified cross-validation was used throughout to avoid data leakage and get a reliable performance estimate before touching the test set.

Per-module weighted F1 (out-of-fold):

| Module | Weighted F1 |
|---|---|
| LightGBM alone | 0.6005 |
| Bidirectional LSTM alone | 0.5654 |
| NLP (TF-IDF + Logistic Regression) alone | 0.7481 |
| **Fused ensemble (final)** | **0.7503** |

The first fusion attempt used fixed, intuition-based weights (50% LightGBM / 30% LSTM / 20% NLP) and scored 0.6152 — *worse* than the NLP module alone. Weighted averaging of probabilities is only guaranteed to fall between the best and worst component when the weights reflect actual component strength; here the strongest model (NLP) was given the smallest weight, so its accurate predictions were diluted by the two weaker models on every example. This was caught by comparing the fused score against each standalone module rather than only reporting the ensemble number.

The fix was an empirical grid search over weight combinations (step 0.1) maximizing out-of-fold weighted F1, rather than assuming weights upfront. The search converged on LightGBM ≈ 0.0, LSTM ≈ 0.1, NLP ≈ 0.9, raising weighted F1 to 0.7503 and, more importantly, fixing Medium Risk recall from 0.10 to 0.53 — under the original weights the model was misclassifying 2427 of 3000 Medium Risk students as Low Risk; under the corrected weights that drops to 804.

This also means the attendance LSTM contributes very little to the final prediction once tabular and text signals are combined — most likely because semester-level CGPA and backlog trends already proxy for the same behavioral decline that attendance sequences capture. That's a legitimate finding about this dataset, not a failure of the LSTM module itself, and is reported here rather than hidden behind an aggregate score.

See `notebooks/retina_ai_dropout_prediction.ipynb` for the full classification report and confusion matrix.

## Tech stack

Python · LightGBM · TensorFlow/Keras · scikit-learn · TextBlob · pandas · NumPy

## Repository structure

```
retina-ai-dropout-prediction/
├── notebooks/
│   └── retina_ai_dropout_prediction.ipynb   # Full pipeline: EDA → 4 modules → fusion → submission
├── images/
│   └── architecture.png                      # Pipeline diagram
├── docs/
│   └── methodology.md                        # Extended writeup (problem framing, design tradeoffs)
└── README.md
```

## What I'd improve with more time

- SMOTE or class-weighted loss to address the 15% High-Risk class imbalance
- Replace TF-IDF with a fine-tuned distilled transformer for the text module
- Optuna-based hyperparameter search for LightGBM instead of manual tuning
- A learned meta-model (stacking) in place of fixed fusion weights
- SHAP-based explainability layer so the High Risk flag is interpretable by counsellors, not just a probability

## Author

Built for the RETINA AI Hackathon, ABES Engineering College, June 2026.
