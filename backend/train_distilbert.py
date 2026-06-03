from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from backend.emotions import EMOTION_LABELS

BACKEND_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = BACKEND_ROOT / "data" / "cleaned_journal_data_improved.csv"
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "emotion_model_improved"


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer: DistilBertTokenizerFast, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float)
        return item


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on the improved GoEmotions dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.25)
    return parser.parse_args()


def load_data(path: Path, sample_size: int | None, seed: int):
    df = pd.read_csv(path).dropna(subset=["text"])
    missing = [label for label in EMOTION_LABELS if label not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing emotion columns: {missing}")

    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=seed)

    label_counts = df[EMOTION_LABELS].sum()
    keep_labels = [label for label in EMOTION_LABELS if int(label_counts[label]) > 0]
    removed_labels = [label for label in EMOTION_LABELS if label not in keep_labels]
    if removed_labels:
        print(f"Excluding zero-support labels: {', '.join(removed_labels)}")

    texts = df["text"].astype(str).tolist()
    labels = df[keep_labels].astype(int).values
    return texts, labels, keep_labels


def compute_metrics_factory(threshold: float):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probabilities = 1 / (1 + np.exp(-logits))
        predictions = (probabilities >= threshold).astype(int)
        empty_rows = np.where(predictions.sum(axis=1) == 0)[0]
        if len(empty_rows):
            top_indices = probabilities[empty_rows].argmax(axis=1)
            predictions[empty_rows, top_indices] = 1
        return {
            "micro_f1": f1_score(labels, predictions, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
            "precision_micro": precision_score(labels, predictions, average="micro", zero_division=0),
            "recall_micro": recall_score(labels, predictions, average="micro", zero_division=0),
        }

    return compute_metrics


def main():
    args = parse_args()
    texts, labels, label_names = load_data(args.data, args.sample_size, args.seed)

    train_validation_texts, test_texts, y_train_validation, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
    )
    validation_fraction = args.validation_size / (1 - args.test_size)
    train_texts, validation_texts, y_train, y_validation = train_test_split(
        train_validation_texts,
        y_train_validation,
        test_size=validation_fraction,
        random_state=args.seed,
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.base_model)
    model = DistilBertForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
    )
    model.config.id2label = {index: label for index, label in enumerate(label_names)}
    model.config.label2id = {label: index for index, label in enumerate(label_names)}

    train_dataset = EmotionDataset(train_texts, y_train, tokenizer, args.max_length)
    validation_dataset = EmotionDataset(validation_texts, y_validation, tokenizer, args.max_length)
    test_dataset = EmotionDataset(test_texts, y_test, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_factory(args.threshold),
    )
    trainer.train()

    metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "training_summary.json").write_text(json.dumps({
        "data": str(args.data),
        "base_model": args.base_model,
        "labels": label_names,
        "threshold": args.threshold,
        "test_metrics": metrics,
    }, indent=2))

    print("Saved improved model to", args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
