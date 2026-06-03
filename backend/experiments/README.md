# Emotion Method Comparison Experiments

Use this folder to compare the dissertation model against simpler and stronger
baselines on the same GoEmotions-style dataset.

Run from the repository root:

```sh
venv/bin/python -m backend.experiments.compare_methods --sample-size 5000
```

The default experiment dataset is `backend/data/cleaned_journal_data.csv`.
That dataset has no positive `neutral` labels, so the runner excludes zero-support
labels by default. To prepare a cleaner dataset that keeps a controlled number
of neutral examples, run:

```sh
venv/bin/python -m backend.data.prepare_goemotions_dataset
```

Then evaluate that dataset explicitly:

```sh
venv/bin/python -m backend.experiments.compare_methods --data backend/data/cleaned_journal_data_improved.csv
```

The script writes:

- `backend/experiments/results/method_comparison.csv`
- `backend/experiments/results/method_comparison.json`
- `backend/experiments/results/per_emotion_metrics.csv`
- `backend/experiments/results/per_emotion_metrics.json`
- `backend/experiments/results/per_emotion_summary.csv`
- `backend/experiments/results/per_emotion_summary.json`
- `backend/experiments/results/micro_f1_comparison.svg`
- `backend/experiments/results/macro_f1_comparison.svg`
- `backend/experiments/results/latency_comparison.svg`

Recommended dissertation table columns:

- Micro F1
- Macro F1
- Precision
- Recall
- Hamming loss
- Subset accuracy
- Fit time
- Inference latency per sample

Implemented methods:

- `dummy_majority`: majority-label baseline
- `lexicon`: transparent keyword/emotion dictionary baseline
- `tfidf_logreg`: TF-IDF + one-vs-rest logistic regression
- `tfidf_svm`: TF-IDF + one-vs-rest linear SVM
- `tfidf_nb`: TF-IDF + one-vs-rest Complement Naive Bayes
- `tfidf_rf`: TF-IDF + one-vs-rest random forest
- `local_distilbert`: saved local DistilBERT model using sigmoid multi-label scores
- `roberta_goemotions`: optional Hugging Face RoBERTa GoEmotions transformer baseline
- `trained_distilbert`: local model trained with `backend.train_distilbert`
- `trained_roberta`: local model trained with `backend.train_roberta`
- `roberta_weighted_focal`: local RoBERTa trained with weighted focal loss
- `roberta_class_balanced_focal`: local RoBERTa trained with class-balanced focal loss

By default, the runner tunes the DistilBERT threshold on a validation split
before evaluating on the held-out test split. You can change this with:

```sh
venv/bin/python -m backend.experiments.compare_methods --threshold-strategy fixed --threshold 0.25
venv/bin/python -m backend.experiments.compare_methods --threshold-strategy per_label
```

Use `per_emotion_metrics.csv` for the dissertation error-analysis section,
especially to discuss emotions with low support or low F1.

For more robust dissertation reporting, run several seeds:

```sh
venv/bin/python -m backend.experiments.compare_methods --seeds 13 42 101
```

This additionally saves `method_comparison_by_seed_summary.csv` with mean and
standard deviation metrics.

To include bootstrap confidence intervals:

```sh
venv/bin/python -m backend.experiments.compare_methods --bootstrap-iterations 200 --output-dir backend/experiments/results/bootstrap
```

To compare against a stronger pretrained transformer baseline:

```sh
venv/bin/python -m backend.experiments.compare_methods --methods local_distilbert roberta_goemotions tfidf_logreg --output-dir backend/experiments/results/roberta_baseline
```

The first RoBERTa run may need internet access to download
`SamLowe/roberta-base-go_emotions`. After it is cached, it can run locally.

To statistically compare repeated-seed results:

```sh
venv/bin/python -m backend.experiments.statistical_compare --results backend/experiments/results/multiseed/method_comparison.csv
```

To summarize human-centered prediction feedback collected through the frontend:

```sh
venv/bin/python -m backend.experiments.analyze_feedback
```

To retrain DistilBERT on the improved dataset:

```sh
venv/bin/python -m backend.train_distilbert --data backend/data/cleaned_journal_data_improved.csv --output-dir backend/emotion_model_improved
```

To train RoBERTa on the same improved dataset:

```sh
venv/bin/python -m backend.train_roberta --data backend/data/cleaned_journal_data_improved.csv --output-dir backend/emotion_model_roberta
```

To train RoBERTa with weighted focal loss:

```sh
venv/bin/python -m backend.train_roberta --data backend/data/cleaned_journal_data_improved.csv --output-dir backend/emotion_model_roberta_weighted_focal --loss weighted_focal
```

To train RoBERTa with class-balanced focal loss:

```sh
venv/bin/python -m backend.train_roberta --data backend/data/cleaned_journal_data_improved.csv --output-dir backend/emotion_model_roberta_class_balanced_focal --loss class_balanced_focal
```

Then compare the locally trained models:

```sh
venv/bin/python -m backend.experiments.compare_methods --data backend/data/cleaned_journal_data_improved.csv --methods dummy_majority lexicon tfidf_logreg tfidf_svm tfidf_nb trained_distilbert trained_roberta roberta_weighted_focal roberta_class_balanced_focal --distilbert-model-path backend/emotion_model_improved --roberta-model-path backend/emotion_model_roberta --roberta-weighted-focal-path backend/emotion_model_roberta_weighted_focal --roberta-class-balanced-focal-path backend/emotion_model_roberta_class_balanced_focal --threshold-strategy per_label --bootstrap-iterations 200 --output-dir backend/experiments/results/final_model_comparison
```

After reviewing the comparison, run the app with the winning model:

```sh
EMOTION_MODEL_PATH=backend/emotion_model_roberta venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001 --reload
```

or:

```sh
EMOTION_MODEL_PATH=backend/emotion_model_improved venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001 --reload
```
