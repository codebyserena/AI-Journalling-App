# ============================================================
# File: evaluate_model_improved.py
# Purpose: Evaluate DistilBERT Multi-Label Emotion Model
# Improvements:
#   - Global threshold tuning
#   - Per-class threshold optimization
#   - Clean metric reporting
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, classification_report

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.abspath("emotion_model")
DATA_PATH = "/Users/serenamendanha/Desktop/tcd/Semester 1/dissertation/backend/data/cleaned_journal_data.csv"
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

print("Model loaded successfully.\n")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

TEXT_COLUMN = "text"

emotion_labels = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval',
    'disgust','embarrassment','excitement','fear','gratitude',
    'grief','joy','love','nervousness','optimism','pride',
    'realization','relief','remorse','sadness','surprise','neutral'
]

texts = df[TEXT_COLUMN].tolist()
y_true = df[emotion_labels].values

# ============================================================
# PREDICTION
# ============================================================

def predict(texts):
    probs_list = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN
            ).to(DEVICE)

            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs[0])

    return np.array(probs_list)

print("Running inference...")
probs = predict(texts)
print("Inference complete.\n")

# ============================================================
# 1️⃣ GLOBAL THRESHOLD TUNING
# ============================================================

print("Optimizing global threshold...")

best_f1 = 0
best_threshold = 0.5

for t in np.arange(0.1, 0.6, 0.05):
    preds = (probs > t).astype(int)
    f1 = f1_score(y_true, preds, average="micro", zero_division=0)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best Global Threshold: {best_threshold}")
print(f"Best Micro F1 (Global): {best_f1:.4f}\n")

global_preds = (probs > best_threshold).astype(int)

# ============================================================
# 2️⃣ PER-CLASS THRESHOLD TUNING
# ============================================================

print("Optimizing per-class thresholds...")

class_thresholds = []
per_class_f1 = []

for i in range(len(emotion_labels)):
    best_f1 = 0
    best_t = 0.5

    for t in np.arange(0.1, 0.6, 0.05):
        preds = (probs[:, i] > t).astype(int)
        f1 = f1_score(y_true[:, i], preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    class_thresholds.append(best_t)
    per_class_f1.append(best_f1)

print("Per-class thresholds optimized.\n")

per_class_preds = np.zeros_like(probs)

for i in range(len(class_thresholds)):
    per_class_preds[:, i] = (probs[:, i] > class_thresholds[i]).astype(int)

# ============================================================
# FINAL METRICS (PER-CLASS TUNED)
# ============================================================

print("Final Evaluation (Per-Class Thresholds)")
print("========================================")

micro_f1 = f1_score(y_true, per_class_preds, average="micro", zero_division=0)
macro_f1 = f1_score(y_true, per_class_preds, average="macro", zero_division=0)
precision = precision_score(y_true, per_class_preds, average="micro", zero_division=0)
recall = recall_score(y_true, per_class_preds, average="micro", zero_division=0)
hamming = hamming_loss(y_true, per_class_preds)

print(f"Micro F1 Score: {micro_f1:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Hamming Loss: {hamming:.4f}\n")

print("Classification Report:")
print("========================================")
print(classification_report(
    y_true,
    per_class_preds,
    target_names=emotion_labels,
    zero_division=0
))