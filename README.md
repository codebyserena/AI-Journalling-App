<<<<<<< HEAD
# ReflectAI Journal

## Run the app

Open a terminal in the project root:

```bash
cd "/Users/serenamendanha/Desktop/tcd/Semester 1/dissertation"
```

Start the backend with the final RoBERTa model:

```bash
EMOTION_MODEL_PATH=backend/emotion_model_roberta \
venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001 --reload
```

Open a second terminal and start the frontend:

```bash
cd "/Users/serenamendanha/Desktop/tcd/Semester 1/dissertation/quiet-mind-journal"
npm run dev
```

Open the app at:

```text
http://localhost:8080
```

## Useful commands

Prepare the improved dataset:

```bash
cd "/Users/serenamendanha/Desktop/tcd/Semester 1/dissertation"
venv/bin/python -m backend.data.prepare_goemotions_dataset
```

Train DistilBERT:

```bash
venv/bin/python -m backend.train_distilbert \
  --data backend/data/cleaned_journal_data_improved.csv \
  --output-dir backend/emotion_model_improved \
  --epochs 3 \
  --batch-size 16
```

Train RoBERTa:

```bash
venv/bin/python -m backend.train_roberta \
  --data backend/data/cleaned_journal_data_improved.csv \
  --output-dir backend/emotion_model_roberta \
  --epochs 3 \
  --batch-size 8
```

Train RoBERTa with weighted focal loss:

```bash
venv/bin/python -m backend.train_roberta \
  --data backend/data/cleaned_journal_data_improved.csv \
  --output-dir backend/emotion_model_roberta_weighted_focal \
  --epochs 3 \
  --batch-size 8 \
  --loss weighted_focal
```

Train RoBERTa with class-balanced focal loss:

```bash
venv/bin/python -m backend.train_roberta \
  --data backend/data/cleaned_journal_data_improved.csv \
  --output-dir backend/emotion_model_roberta_class_balanced_focal \
  --epochs 3 \
  --batch-size 8 \
  --loss class_balanced_focal
```

Run the final comparison:

```bash
venv/bin/python -m backend.experiments.compare_methods \
  --data backend/data/cleaned_journal_data_improved.csv \
  --methods dummy_majority lexicon tfidf_logreg tfidf_svm tfidf_nb trained_distilbert trained_roberta roberta_weighted_focal roberta_class_balanced_focal \
  --distilbert-model-path backend/emotion_model_improved \
  --roberta-model-path backend/emotion_model_roberta \
  --roberta-weighted-focal-path backend/emotion_model_roberta_weighted_focal \
  --roberta-class-balanced-focal-path backend/emotion_model_roberta_class_balanced_focal \
  --threshold-strategy per_label \
  --bootstrap-iterations 200 \
  --output-dir backend/experiments/results/final_model_comparison
```

Regenerate dissertation graphs and tables:

```bash
venv/bin/python -m backend.experiments.build_dissertation_evidence
```

Analyze user feedback:

```bash
venv/bin/python -m backend.experiments.analyze_feedback
```

## Troubleshooting

If the frontend shows `Failed to fetch`, make sure the backend is running on `127.0.0.1:8001`.

If you want to run the backend with DistilBERT instead:

```bash
EMOTION_MODEL_PATH=backend/emotion_model_improved \
venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001 --reload
```
=======
# AI-Journalling-App
>>>>>>> c28a62a80d4a0370661c95af3adb64b7df2cde57
