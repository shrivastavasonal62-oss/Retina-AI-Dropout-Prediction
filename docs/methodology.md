# RETINA AI — Multimodal Deep Learning for Student Dropout Risk Prediction

## 1. Problem Understanding

Student dropout is one of the most critical challenges in higher education. Early identification of at-risk students allows institutions to intervene before it's too late. This competition required classifying 3,000 students into three dropout risk levels — Low (0), Medium (1), and High (2) — using three heterogeneous data sources simultaneously.

The key challenge was that no single data source tells the complete story:
- Academic records show *what* grades a student has
- Attendance sequences show *how consistently* they show up over time
- Counsellor notes reveal *why* a student might be struggling

My solution combines all three into one unified multimodal deep learning system called **RETINA AI**.

---

## 2. Solution Architecture

My approach uses four modules working together:

### Module A — LightGBM on Academic Tabular Data
The academic data (CGPA per semester, backlogs, demographics, family income, etc.) was processed using **LightGBM**, a gradient boosting algorithm well-suited for structured data. I used **5-fold stratified cross-validation** to ensure reliable, unbiased evaluation.

**Feature Engineering:**
- `cgpa_trend` = cgpa_sem4 − cgpa_sem1 → captures whether the student is improving or declining
- `cgpa_mean`, `cgpa_min`, `cgpa_std` → aggregate academic performance statistics
- `total_backlogs` = sum of backlogs across all 3 semesters
- `backlog_trend` = backlogs_sem3 − backlogs_sem1 → is the backlog problem getting worse?
- `academic_stress` = total_backlogs × (10 − mean_cgpa) → combined stress indicator

Missing values in `commute_time_mins` were filled with the median; `parent_education` was filled with 'Unknown'.

### Module B — Bidirectional LSTM on Attendance Time-Series
The attendance data was reshaped into sequences of shape **(24 timesteps × 4 features)** per student:
- 24 timesteps = 3 semesters × 8 weeks
- 4 features = Core_1 attendance, Core_2 attendance, Elective attendance, mean attendance

A **Bidirectional LSTM** was used because it reads sequences both forward AND backward — this helps detect patterns like "student was regular early but dropped off later" as well as "student recovered after a bad patch." The model uses two Bidirectional LSTM layers with Dropout regularization to prevent overfitting, and Early Stopping to find the optimal number of epochs automatically.

### Module C — TF-IDF + Sentiment Analysis on Counsellor Notes
Counsellor notes are unstructured text containing rich signals about student behaviour. I extracted the following features:

**Statistical features:**
- Word count, character count, average word length
- Sentiment polarity (TextBlob): −1 = very negative, +1 = very positive
- Sentiment subjectivity score
- Risk keyword count: words like 'absent', 'fail', 'struggle', 'concern', 'withdraw'
- Positive keyword count: words like 'excellent', 'motivated', 'consistent', 'progress'
- Net keyword score = positive keywords − risk keywords

**TF-IDF Vectorization:**
- 300 features, unigrams + bigrams, English stop words removed
- Captures important phrases like "action plan", "needs improvement", "performing well"

A **Logistic Regression** model was trained on the combined TF-IDF + statistical features.

### Module D — Weighted Average Fusion
The final prediction combines all three models using a weighted average of their class probabilities:

| Model | Weight | Reason |
|-------|--------|--------|
| LightGBM | 50% | Best overall performance on structured academic data |
| Bidirectional LSTM | 30% | Strong temporal pattern detection in attendance |
| NLP (Logistic Regression) | 20% | Useful supplementary signal from counsellor text |

Final class = argmax(weighted average probabilities across Low, Medium, High)

---

## 3. Why This Approach Works

Most approaches to this problem use only tabular data. What makes RETINA AI different is the genuine multimodal fusion — each data source captures a different dimension of student risk:

- A student might have decent grades but rapidly declining attendance → LSTM catches this
- A student might have average CGPA but counsellor notes full of risk keywords → NLP catches this
- The LightGBM captures the overall academic trajectory through feature engineering

By combining all three, the model is much harder to fool than any single modality alone.

---

## 4. Technical Stack

- **Python 3.12** on Kaggle GPU (Tesla T4 x2)
- **LightGBM** — gradient boosting for tabular data
- **TensorFlow / Keras** — Bidirectional LSTM
- **scikit-learn** — preprocessing, TF-IDF, Logistic Regression, evaluation
- **TextBlob** — sentiment analysis
- **pandas, numpy** — data processing

---

## 5. Key Results

- LightGBM alone: strong baseline with 5-fold CV
- LSTM adds temporal attendance signal on top
- NLP adds counsellor sentiment signal on top
- Final fusion outperforms any single modality

The confusion matrix shows the model is especially strong at identifying High Risk students — the most important category for real-world intervention.

---

## 6. What I Would Improve With More Time

1. **SMOTE oversampling** — High Risk students are only 15% of the dataset; oversampling would help the model learn this minority class better
2. **BERT embeddings** instead of TF-IDF for richer text understanding
3. **Optuna hyperparameter tuning** for LightGBM to automatically find optimal settings
4. **Rolling attendance features** — 3-week moving average, consecutive absence streaks
5. **Meta-learner** — instead of fixed weights, train a small model to learn optimal fusion weights

---

## 7. Conclusion

RETINA AI demonstrates that student dropout risk is best understood through multiple lenses simultaneously. By combining structured academic data, temporal attendance patterns, and natural language from counsellor interactions, the model achieves a holistic view of each student's situation — much like how a real academic counsellor would make their assessment.

This multimodal approach is not just effective for this competition — it represents the future of intelligent student success systems in higher education.
