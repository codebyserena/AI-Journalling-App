from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from backend.emotions import EMOTION_LABELS

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = BACKEND_ROOT / "data" / "cleaned_journal_data.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
THRESHOLD_GRID = np.arange(0.05, 0.76, 0.05)


class EmotionLexiconClassifier(BaseEstimator, ClassifierMixin):
    """Transparent baseline using small emotion keyword dictionaries."""

    def __init__(self, labels: list[str]):
        self.labels = labels
        self.lexicon = {
            "admiration": {"admire", "respect", "impressive", "inspiring", "brilliant"},
            "amusement": {"funny", "laugh", "lol", "hilarious", "joke"},
            "anger": {"angry", "furious", "rage", "mad", "hate"},
            "annoyance": {"annoyed", "irritated", "bothered", "frustrating"},
            "approval": {"agree", "approved", "right", "yes", "valid"},
            "caring": {"care", "support", "kind", "help", "gentle"},
            "confusion": {"confused", "unclear", "lost", "unsure", "why"},
            "curiosity": {"curious", "wonder", "question", "interested", "learn"},
            "desire": {"want", "wish", "need", "crave", "hope"},
            "disappointment": {"disappointed", "letdown", "failed", "unfortunate"},
            "disapproval": {"wrong", "disagree", "bad", "unacceptable", "shouldn't"},
            "disgust": {"disgusting", "gross", "awful", "repulsive", "nasty"},
            "embarrassment": {"embarrassed", "awkward", "ashamed", "humiliated"},
            "excitement": {"excited", "thrilled", "amazing", "can't wait"},
            "fear": {"afraid", "scared", "terrified", "fear", "panic"},
            "gratitude": {"thanks", "thankful", "grateful", "appreciate"},
            "grief": {"grief", "mourning", "loss", "heartbroken"},
            "joy": {"happy", "joy", "delighted", "smile", "glad"},
            "love": {"love", "beloved", "affection", "adore", "cherish"},
            "nervousness": {"nervous", "anxious", "worried", "tense"},
            "optimism": {"optimistic", "hopeful", "better", "positive"},
            "pride": {"proud", "accomplished", "achievement", "earned"},
            "realization": {"realized", "noticed", "understood", "learned"},
            "relief": {"relieved", "finally", "safe", "easier"},
            "remorse": {"sorry", "regret", "guilty", "apologize"},
            "sadness": {"sad", "cry", "hurt", "lonely", "depressed"},
            "surprise": {"surprised", "unexpected", "shocked", "wow"},
            "neutral": {"okay", "fine", "normal", "ordinary"},
        }

    def fit(self, texts: Iterable[str], y=None):
        return self

    def predict(self, texts: Iterable[str]):
        rows = []
        for text in texts:
            text_lower = str(text).lower()
            row = np.zeros(len(self.labels), dtype=int)
            for label_index, label in enumerate(self.labels):
                terms = self.lexicon.get(label, set())
                if any(term in text_lower for term in terms):
                    row[label_index] = 1

            if not row.any() and "neutral" in self.labels:
                row[self.labels.index("neutral")] = 1
            rows.append(row)
        return np.array(rows)


def resolve_labels(df: pd.DataFrame, label_policy: str) -> list[str]:
    if label_policy == "all":
        return EMOTION_LABELS

    nonzero_labels = [label for label in EMOTION_LABELS if int(df[label].sum()) > 0]
    removed_labels = [label for label in EMOTION_LABELS if label not in nonzero_labels]
    if removed_labels:
        print(f"Excluding zero-support labels from metrics: {', '.join(removed_labels)}")
    return nonzero_labels


def load_dataset(path: Path, sample_size: int | None, seed: int, label_policy: str) -> tuple[pd.Series, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    missing = [label for label in EMOTION_LABELS if label not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing emotion columns: {missing}")

    df = df.dropna(subset=["text"])
    label_names = resolve_labels(df, label_policy)
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=seed)

    texts = df["text"].astype(str)
    labels = df[label_names].astype(int).values
    return texts, labels, label_names


def drop_untrainable_labels(
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train_support = y_train.sum(axis=0)
    keep_indices = [index for index, support in enumerate(train_support) if support > 0]
    removed_labels = [label for index, label in enumerate(label_names) if index not in keep_indices]
    if removed_labels:
        print(f"Excluding labels with no positives in this training split: {', '.join(removed_labels)}")

    if not keep_indices:
        raise ValueError("No trainable emotion labels remain after filtering.")

    kept_labels = [label_names[index] for index in keep_indices]
    return (
        y_train[:, keep_indices],
        y_validation[:, keep_indices],
        y_test[:, keep_indices],
        kept_labels,
    )


def build_method(name: str, label_names: list[str], seed: int):
    if name == "dummy_majority":
        return OneVsRestClassifier(DummyClassifier(strategy="most_frequent"))
    if name == "lexicon":
        return EmotionLexiconClassifier(label_names)
    if name == "tfidf_logreg":
        return Pipeline([
            ("tfidf", TfidfVectorizer(max_features=80000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", OneVsRestClassifier(LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=2.0,
                solver="liblinear",
                random_state=seed,
            ))),
        ])
    if name == "tfidf_svm":
        return Pipeline([
            ("tfidf", TfidfVectorizer(max_features=80000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", OneVsRestClassifier(LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed))),
        ])
    if name == "tfidf_nb":
        return Pipeline([
            ("tfidf", TfidfVectorizer(max_features=80000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", OneVsRestClassifier(ComplementNB())),
        ])
    if name == "tfidf_rf":
        return Pipeline([
            ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", OneVsRestClassifier(RandomForestClassifier(
                n_estimators=120,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ))),
        ])
    raise ValueError(f"Unknown method: {name}")


def ensure_one_label(y_pred: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    empty_rows = np.where(y_pred.sum(axis=1) == 0)[0]
    if len(empty_rows):
        top_indices = probabilities[empty_rows].argmax(axis=1)
        y_pred[empty_rows, top_indices] = 1
    return y_pred


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    row_count = len(y_true)
    for _ in range(iterations):
        indices = rng.integers(0, row_count, row_count)
        scores.append(f1_score(y_true[indices], y_pred[indices], average=average, zero_division=0))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def evaluate_predictions(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fit_seconds: float,
    predict_seconds: float,
    threshold_strategy: str = "not_applicable",
    threshold: float | str = "not_applicable",
    bootstrap_iterations: int = 0,
    seed: int = 42,
):
    result = {
        "method": name,
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "latency_ms_per_sample": (predict_seconds / max(len(y_true), 1)) * 1000,
        "threshold_strategy": threshold_strategy,
        "threshold": threshold,
    }
    if bootstrap_iterations > 0:
        micro_low, micro_high = bootstrap_metric_ci(y_true, y_pred, "micro", bootstrap_iterations, seed)
        macro_low, macro_high = bootstrap_metric_ci(y_true, y_pred, "macro", bootstrap_iterations, seed + 1)
        result.update({
            "micro_f1_ci_low": micro_low,
            "micro_f1_ci_high": micro_high,
            "macro_f1_ci_low": macro_low,
            "macro_f1_ci_high": macro_high,
            "bootstrap_iterations": bootstrap_iterations,
        })
    return result


def evaluate_per_label(name: str, y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> list[dict]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    return [
        {
            "method": name,
            "emotion": label,
            "precision": precision[index],
            "recall": recall[index],
            "f1": f1[index],
            "support": int(support[index]),
        }
        for index, label in enumerate(label_names)
    ]


def run_sklearn_method(name: str, train_texts, test_texts, y_train, y_test, label_names: list[str], seed: int, bootstrap_iterations: int):
    model = build_method(name, label_names, seed)

    start = time.perf_counter()
    model.fit(train_texts, y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(test_texts)
    predict_seconds = time.perf_counter() - start

    summary = evaluate_predictions(
        name,
        y_test,
        y_pred,
        fit_seconds,
        predict_seconds,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    per_label = evaluate_per_label(name, y_test, y_pred, label_names)
    return summary, per_label


def predict_distilbert_probabilities(texts, batch_size: int):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = BACKEND_ROOT / "emotion_model"
    return predict_local_model_probabilities(texts, model_path, batch_size)


def predict_local_model_probabilities(texts, model_path: Path, batch_size: int):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model_labels = [
        model.config.id2label[index].lower()
        for index in range(len(model.config.id2label))
    ] if getattr(model.config, "id2label", None) and len(model.config.id2label) == model.config.num_labels else EMOTION_LABELS

    probabilities = []
    with torch.no_grad():
        for index in range(0, len(texts), batch_size):
            batch = list(texts.iloc[index:index + batch_size])
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            outputs = model(**inputs)
            probabilities.append(torch.sigmoid(outputs.logits).cpu().numpy())
    return np.vstack(probabilities), model_labels


def filter_probability_columns(probabilities: np.ndarray, label_names: list[str]) -> np.ndarray:
    indices = [EMOTION_LABELS.index(label) for label in label_names]
    return probabilities[:, indices]


def filter_model_probability_columns(
    probabilities: np.ndarray,
    label_names: list[str],
    model_labels: list[str],
) -> np.ndarray:
    missing = [label for label in label_names if label not in model_labels]
    if missing:
        raise ValueError(f"Local model is missing labels needed for this experiment: {missing}")
    indices = [model_labels.index(label) for label in label_names]
    return probabilities[:, indices]


def predict_hf_goemotions_probabilities(texts, model_name: str, batch_size: int, label_names: list[str]):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    id2label = {int(index): label.lower() for index, label in model.config.id2label.items()}
    model_labels = [id2label[index] for index in range(len(id2label))]
    missing = [label for label in label_names if label not in model_labels]
    if missing:
        raise ValueError(
            f"{model_name} does not expose these labels: {missing}. "
            "Use a GoEmotions-compatible checkpoint such as SamLowe/roberta-base-go_emotions."
        )

    probabilities = []
    with torch.no_grad():
        for index in range(0, len(texts), batch_size):
            batch = list(texts.iloc[index:index + batch_size])
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            outputs = model(**inputs)
            probabilities.append(torch.sigmoid(outputs.logits).cpu().numpy())

    all_probabilities = np.vstack(probabilities)
    keep_indices = [model_labels.index(label) for label in label_names]
    return all_probabilities[:, keep_indices]


def apply_thresholds(probabilities: np.ndarray, thresholds: float | np.ndarray) -> np.ndarray:
    y_pred = (probabilities >= thresholds).astype(int)
    return ensure_one_label(y_pred, probabilities)


def tune_global_threshold(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    best_threshold = 0.25
    best_f1 = -1.0
    for threshold in THRESHOLD_GRID:
        y_pred = apply_thresholds(probabilities, threshold)
        score = f1_score(y_true, y_pred, average="micro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def tune_per_label_thresholds(probabilities: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], 0.25)
    for label_index in range(y_true.shape[1]):
        best_threshold = 0.25
        best_f1 = -1.0
        for threshold in THRESHOLD_GRID:
            y_pred = (probabilities[:, label_index] >= threshold).astype(int)
            score = f1_score(y_true[:, label_index], y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thresholds[label_index] = best_threshold
    return thresholds


def run_local_distilbert(
    validation_texts,
    test_texts,
    y_validation,
    y_test,
    threshold: float,
    threshold_strategy: str,
    batch_size: int,
    label_names: list[str],
    bootstrap_iterations: int,
    seed: int,
):
    start = time.perf_counter()
    raw_test_probabilities, model_labels = predict_local_model_probabilities(
        test_texts,
        BACKEND_ROOT / "emotion_model",
        batch_size,
    )
    test_probabilities = filter_model_probability_columns(raw_test_probabilities, label_names, model_labels)
    predict_seconds = time.perf_counter() - start

    selected_threshold: float | np.ndarray = threshold
    threshold_value: float | str = threshold
    if threshold_strategy == "global":
        raw_validation_probabilities, validation_model_labels = predict_local_model_probabilities(
            validation_texts,
            BACKEND_ROOT / "emotion_model",
            batch_size,
        )
        validation_probabilities = filter_model_probability_columns(raw_validation_probabilities, label_names, validation_model_labels)
        selected_threshold = tune_global_threshold(validation_probabilities, y_validation)
        threshold_value = round(float(selected_threshold), 4)
    elif threshold_strategy == "per_label":
        raw_validation_probabilities, validation_model_labels = predict_local_model_probabilities(
            validation_texts,
            BACKEND_ROOT / "emotion_model",
            batch_size,
        )
        validation_probabilities = filter_model_probability_columns(raw_validation_probabilities, label_names, validation_model_labels)
        selected_threshold = tune_per_label_thresholds(validation_probabilities, y_validation)
        threshold_value = "per_label"

    y_pred = apply_thresholds(test_probabilities, selected_threshold)
    summary = evaluate_predictions(
        "local_distilbert",
        y_test,
        y_pred,
        0.0,
        predict_seconds,
        threshold_strategy=threshold_strategy,
        threshold=threshold_value,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    per_label = evaluate_per_label("local_distilbert", y_test, y_pred, label_names)
    return summary, per_label


def run_local_transformer(
    method_name: str,
    model_path: Path,
    validation_texts,
    test_texts,
    y_validation,
    y_test,
    threshold: float,
    threshold_strategy: str,
    batch_size: int,
    label_names: list[str],
    bootstrap_iterations: int,
    seed: int,
):
    start = time.perf_counter()
    raw_test_probabilities, model_labels = predict_local_model_probabilities(test_texts, model_path, batch_size)
    test_probabilities = filter_model_probability_columns(raw_test_probabilities, label_names, model_labels)
    predict_seconds = time.perf_counter() - start

    selected_threshold: float | np.ndarray = threshold
    threshold_value: float | str = threshold
    if threshold_strategy == "global":
        raw_validation_probabilities, validation_model_labels = predict_local_model_probabilities(validation_texts, model_path, batch_size)
        validation_probabilities = filter_model_probability_columns(raw_validation_probabilities, label_names, validation_model_labels)
        selected_threshold = tune_global_threshold(validation_probabilities, y_validation)
        threshold_value = round(float(selected_threshold), 4)
    elif threshold_strategy == "per_label":
        raw_validation_probabilities, validation_model_labels = predict_local_model_probabilities(validation_texts, model_path, batch_size)
        validation_probabilities = filter_model_probability_columns(raw_validation_probabilities, label_names, validation_model_labels)
        selected_threshold = tune_per_label_thresholds(validation_probabilities, y_validation)
        threshold_value = "per_label"

    y_pred = apply_thresholds(test_probabilities, selected_threshold)
    summary = evaluate_predictions(
        method_name,
        y_test,
        y_pred,
        0.0,
        predict_seconds,
        threshold_strategy=threshold_strategy,
        threshold=threshold_value,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    per_label = evaluate_per_label(method_name, y_test, y_pred, label_names)
    return summary, per_label


def run_roberta_goemotions(
    validation_texts,
    test_texts,
    y_validation,
    y_test,
    threshold: float,
    threshold_strategy: str,
    batch_size: int,
    label_names: list[str],
    model_name: str,
    bootstrap_iterations: int,
    seed: int,
):
    start = time.perf_counter()
    test_probabilities = predict_hf_goemotions_probabilities(test_texts, model_name, batch_size, label_names)
    predict_seconds = time.perf_counter() - start

    selected_threshold: float | np.ndarray = threshold
    threshold_value: float | str = threshold
    if threshold_strategy == "global":
        validation_probabilities = predict_hf_goemotions_probabilities(validation_texts, model_name, batch_size, label_names)
        selected_threshold = tune_global_threshold(validation_probabilities, y_validation)
        threshold_value = round(float(selected_threshold), 4)
    elif threshold_strategy == "per_label":
        validation_probabilities = predict_hf_goemotions_probabilities(validation_texts, model_name, batch_size, label_names)
        selected_threshold = tune_per_label_thresholds(validation_probabilities, y_validation)
        threshold_value = "per_label"

    y_pred = apply_thresholds(test_probabilities, selected_threshold)
    summary = evaluate_predictions(
        "roberta_goemotions",
        y_test,
        y_pred,
        0.0,
        predict_seconds,
        threshold_strategy=threshold_strategy,
        threshold=threshold_value,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    per_label = evaluate_per_label("roberta_goemotions", y_test, y_pred, label_names)
    return summary, per_label


def summarize_per_label(per_label_df: pd.DataFrame, output_dir: Path):
    summary_rows = []
    for method, group in per_label_df.groupby("method"):
        nonzero = group[group["support"] > 0]
        summary_rows.append({
            "method": method,
            "best_emotions_by_f1": "; ".join(
                f"{row.emotion} ({row.f1:.3f})"
                for row in nonzero.sort_values("f1", ascending=False).head(5).itertuples()
            ),
            "weakest_emotions_by_f1": "; ".join(
                f"{row.emotion} ({row.f1:.3f}, n={row.support})"
                for row in nonzero.sort_values(["f1", "support"], ascending=[True, True]).head(5).itertuples()
            ),
            "lowest_support_emotions": "; ".join(
                f"{row.emotion} (n={row.support})"
                for row in nonzero.sort_values("support", ascending=True).head(5).itertuples()
            ),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "per_emotion_summary.csv", index=False)
    (output_dir / "per_emotion_summary.json").write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2)
    )


def write_bar_chart(results_df: pd.DataFrame, column: str, title: str, path: Path):
    chart_df = results_df.sort_values(column, ascending=True)
    width = 900
    row_height = 34
    margin_left = 190
    margin_right = 40
    margin_top = 56
    height = margin_top + row_height * len(chart_df) + 40
    chart_width = width - margin_left - margin_right
    max_value = max(float(chart_df[column].max()), 0.001)

    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{margin_left}" y="30" font-family="Arial" font-size="20" font-weight="700" fill="#24312e">{title}</text>',
    ]
    for row_index, row in enumerate(chart_df.itertuples(index=False)):
        y = margin_top + row_index * row_height
        value = float(getattr(row, column))
        bar_width = chart_width * (value / max_value)
        method = getattr(row, "method")
        rows.extend([
            f'<text x="20" y="{y + 20}" font-family="Arial" font-size="13" fill="#39423f">{method}</text>',
            f'<rect x="{margin_left}" y="{y}" width="{bar_width:.2f}" height="22" rx="4" fill="#5b927f"/>',
            f'<text x="{margin_left + bar_width + 8}" y="{y + 16}" font-family="Arial" font-size="13" fill="#39423f">{value:.3f}</text>',
        ])
    rows.append("</svg>")
    path.write_text("\n".join(rows))


def parse_args():
    parser = argparse.ArgumentParser(description="Compare emotion detection methods.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Run multiple random seeds and save mean/std aggregate metrics.",
    )
    parser.add_argument(
        "--label-policy",
        choices=["nonzero", "all"],
        default="nonzero",
        help="Use nonzero to exclude labels with no positives in the dataset.",
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument(
        "--threshold-strategy",
        choices=["fixed", "global", "per_label"],
        default="global",
        help="How to choose DistilBERT thresholds. Global/per-label use the validation split.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=0,
        help="Add bootstrap 95 percent confidence intervals for Micro-F1 and Macro-F1.",
    )
    parser.add_argument(
        "--roberta-model",
        default="SamLowe/roberta-base-go_emotions",
        help="Hugging Face GoEmotions-compatible checkpoint for the roberta_goemotions method.",
    )
    parser.add_argument(
        "--distilbert-model-path",
        type=Path,
        default=BACKEND_ROOT / "emotion_model_improved",
        help="Local trained DistilBERT path for the trained_distilbert method.",
    )
    parser.add_argument(
        "--roberta-model-path",
        type=Path,
        default=BACKEND_ROOT / "emotion_model_roberta",
        help="Local trained RoBERTa path for the trained_roberta method.",
    )
    parser.add_argument(
        "--roberta-weighted-focal-path",
        type=Path,
        default=BACKEND_ROOT / "emotion_model_roberta_weighted_focal",
        help="Local RoBERTa path for the roberta_weighted_focal method.",
    )
    parser.add_argument(
        "--roberta-class-balanced-focal-path",
        type=Path,
        default=BACKEND_ROOT / "emotion_model_roberta_class_balanced_focal",
        help="Local RoBERTa path for the roberta_class_balanced_focal method.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["dummy_majority", "lexicon", "tfidf_logreg", "tfidf_svm", "tfidf_nb", "local_distilbert"],
        help="Methods to run. Add tfidf_rf, roberta_goemotions, trained_distilbert, trained_roberta, roberta_weighted_focal, or roberta_class_balanced_focal as needed.",
    )
    return parser.parse_args()


def run_experiment(args, seed: int):
    texts, labels, label_names = load_dataset(args.data, args.sample_size, seed, args.label_policy)

    train_validation_texts, test_texts, y_train_validation, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=seed,
    )
    validation_fraction = args.validation_size / (1 - args.test_size)
    train_texts, validation_texts, y_train, y_validation = train_test_split(
        train_validation_texts,
        y_train_validation,
        test_size=validation_fraction,
        random_state=seed,
    )
    y_train, y_validation, y_test, label_names = drop_untrainable_labels(
        y_train,
        y_validation,
        y_test,
        label_names,
    )

    results = []
    per_label_results = []
    for method in args.methods:
        print(f"Running {method}...")
        if method == "local_distilbert":
            result, per_label = run_local_distilbert(
                validation_texts,
                test_texts,
                y_validation,
                y_test,
                args.threshold,
                args.threshold_strategy,
                args.batch_size,
                label_names,
                args.bootstrap_iterations,
                seed,
            )
        elif method == "roberta_goemotions":
            result, per_label = run_roberta_goemotions(
                validation_texts,
                test_texts,
                y_validation,
                y_test,
                args.threshold,
                args.threshold_strategy,
                args.batch_size,
                label_names,
                args.roberta_model,
                args.bootstrap_iterations,
                seed,
            )
        elif method == "trained_distilbert":
            result, per_label = run_local_transformer(
                "trained_distilbert",
                args.distilbert_model_path,
                validation_texts,
                test_texts,
                y_validation,
                y_test,
                args.threshold,
                args.threshold_strategy,
                args.batch_size,
                label_names,
                args.bootstrap_iterations,
                seed,
            )
        elif method == "trained_roberta":
            result, per_label = run_local_transformer(
                "trained_roberta",
                args.roberta_model_path,
                validation_texts,
                test_texts,
                y_validation,
                y_test,
                args.threshold,
                args.threshold_strategy,
                args.batch_size,
                label_names,
                args.bootstrap_iterations,
                seed,
            )
        elif method == "roberta_weighted_focal":
            result, per_label = run_local_transformer(
                "roberta_weighted_focal",
                args.roberta_weighted_focal_path,
                validation_texts,
                test_texts,
                y_validation,
                y_test,
                args.threshold,
                args.threshold_strategy,
                args.batch_size,
                label_names,
                args.bootstrap_iterations,
                seed,
            )
        elif method == "roberta_class_balanced_focal":
            result, per_label = run_local_transformer(
                "roberta_class_balanced_focal",
                args.roberta_class_balanced_focal_path,
                validation_texts,
                test_texts,
                y_validation,
                y_test,
                args.threshold,
                args.threshold_strategy,
                args.batch_size,
                label_names,
                args.bootstrap_iterations,
                seed,
            )
        else:
            result, per_label = run_sklearn_method(method, train_texts, test_texts, y_train, y_test, label_names, seed, args.bootstrap_iterations)
        result["seed"] = seed
        results.append(result)
        for row in per_label:
            row["seed"] = seed
        per_label_results.extend(per_label)
        print(
            f"{method}: micro_f1={result['micro_f1']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}, "
            f"latency={result['latency_ms_per_sample']:.2f}ms/sample"
        )

    return results, per_label_results, label_names


def main():
    args = parse_args()
    seeds = args.seeds or [args.seed]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    all_per_label_results = []
    label_names = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        results, per_label_results, label_names = run_experiment(args, seed)
        all_results.extend(results)
        all_per_label_results.extend(per_label_results)

    results_df = pd.DataFrame(all_results).sort_values(["micro_f1", "method"], ascending=[False, True])
    csv_path = args.output_dir / "method_comparison.csv"
    json_path = args.output_dir / "method_comparison.json"
    per_label_csv_path = args.output_dir / "per_emotion_metrics.csv"
    per_label_json_path = args.output_dir / "per_emotion_metrics.json"
    results_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results_df.to_dict(orient="records"), indent=2))
    per_label_df = pd.DataFrame(all_per_label_results).sort_values(["method", "f1"], ascending=[True, False])
    per_label_df.to_csv(per_label_csv_path, index=False)
    per_label_json_path.write_text(json.dumps(per_label_df.to_dict(orient="records"), indent=2))
    summarize_per_label(per_label_df, args.output_dir)

    if len(seeds) > 1:
        metric_columns = [
            "micro_f1", "macro_f1", "precision_micro", "recall_micro",
            "hamming_loss", "subset_accuracy", "latency_ms_per_sample",
        ]
        aggregate_df = results_df.groupby("method")[metric_columns].agg(["mean", "std"]).reset_index()
        aggregate_df.columns = [
            "_".join(column).rstrip("_") if isinstance(column, tuple) else column
            for column in aggregate_df.columns
        ]
        aggregate_df.to_csv(args.output_dir / "method_comparison_by_seed_summary.csv", index=False)

    chart_df = results_df
    if len(seeds) > 1:
        chart_df = results_df.groupby("method", as_index=False)[
            ["micro_f1", "macro_f1", "latency_ms_per_sample"]
        ].mean()
    write_bar_chart(chart_df, "micro_f1", "Micro-F1 by Method", args.output_dir / "micro_f1_comparison.svg")
    write_bar_chart(chart_df, "macro_f1", "Macro-F1 by Method", args.output_dir / "macro_f1_comparison.svg")
    write_bar_chart(chart_df, "latency_ms_per_sample", "Latency per Sample (ms)", args.output_dir / "latency_comparison.svg")

    print("\nResults:")
    print(results_df.to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {json_path}")
    print(f"Saved {per_label_csv_path}")
    print(f"Saved {per_label_json_path}")
    print(f"Saved per-emotion summaries in {args.output_dir}")
    print(f"Saved SVG charts in {args.output_dir}")
    print(f"Labels used ({len(label_names)}): {', '.join(label_names)}")


if __name__ == "__main__":
    main()
