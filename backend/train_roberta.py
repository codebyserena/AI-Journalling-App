from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from backend.emotions import EMOTION_LABELS

BACKEND_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = BACKEND_ROOT / "data" / "cleaned_journal_data_improved.csv"
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "emotion_model_roberta"


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int):
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


class ImbalanceAwareTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_name: str = "bce",
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name
        self.class_weights = class_weights
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None

        if self.loss_name == "bce":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        elif self.loss_name in {"weighted_bce", "class_balanced"}:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=weights,
            )
        elif self.loss_name in {"focal", "weighted_focal", "class_balanced_focal"}:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )
            probabilities = torch.sigmoid(logits)
            pt = torch.where(labels == 1, probabilities, 1 - probabilities)
            focal_factor = (1 - pt).pow(self.gamma)
            loss_values = focal_factor * bce
            if weights is not None:
                positive_weights = torch.where(labels == 1, weights.view(1, -1), torch.ones_like(labels))
                loss_values = loss_values * positive_weights
            loss = loss_values.mean()
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")

        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for multi-label GoEmotions classification.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default="roberta-base")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument(
        "--loss",
        choices=["bce", "weighted_bce", "focal", "weighted_focal", "class_balanced", "class_balanced_focal"],
        default="bce",
        help="Training loss. Use weighted_focal or class_balanced_focal for imbalanced emotion labels.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--class-balanced-beta",
        type=float,
        default=0.9999,
        help="Beta for effective-number class-balanced weights.",
    )
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


def make_class_weights(y_train: np.ndarray, loss_name: str, beta: float):
    if loss_name in {"weighted_bce", "weighted_focal"}:
        positives = y_train.sum(axis=0)
        negatives = len(y_train) - positives
        weights = negatives / np.maximum(positives, 1)
    elif loss_name in {"class_balanced", "class_balanced_focal"}:
        positives = y_train.sum(axis=0)
        effective_num = 1 - np.power(beta, np.maximum(positives, 1))
        weights = (1 - beta) / np.maximum(effective_num, 1e-12)
        weights = weights / np.mean(weights)
    else:
        return None

    weights = np.clip(weights, 0.1, 25.0)
    return torch.tensor(weights, dtype=torch.float)


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

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
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

    class_weights = make_class_weights(y_train, args.loss, args.class_balanced_beta)
    trainer = ImbalanceAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_factory(args.threshold),
        loss_name=args.loss,
        class_weights=class_weights,
        gamma=args.focal_gamma,
    )
    trainer.train()

    metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "training_summary.json").write_text(json.dumps({
        "data": str(args.data),
        "base_model": args.base_model,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "class_balanced_beta": args.class_balanced_beta,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "labels": label_names,
        "threshold": args.threshold,
        "test_metrics": metrics,
    }, indent=2))

    print("Saved RoBERTa model to", args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
