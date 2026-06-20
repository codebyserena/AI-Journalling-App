from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.experiments.compare_methods import (
    BACKEND_ROOT,
    apply_thresholds,
    drop_untrainable_labels,
    filter_model_probability_columns,
    load_dataset,
    predict_local_model_probabilities,
    tune_per_label_thresholds,
)


DEFAULT_DATA = BACKEND_ROOT / "data" / "cleaned_journal_data_improved.csv"
DEFAULT_MODEL = BACKEND_ROOT / "emotion_model_roberta"
DEFAULT_OUTPUT = BACKEND_ROOT / "experiments" / "results" / "pairwise_emotion_analysis"
RARE_EMOTIONS = {"grief", "relief", "pride", "nervousness", "embarrassment"}


def parse_args():
    parser = argparse.ArgumentParser(description="Pairwise emotion confusion and co-occurrence analysis for RoBERTa.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def upper_pairs(matrix: np.ndarray, labels: list[str], value_name: str) -> pd.DataFrame:
    rows = []
    for i, first in enumerate(labels):
        for j in range(i + 1, len(labels)):
            value = int(matrix[i, j])
            if value > 0:
                rows.append({"emotion_a": first, "emotion_b": labels[j], value_name: value})
    return pd.DataFrame(rows).sort_values(value_name, ascending=False)


def confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    rows = []
    for true_index, true_label in enumerate(labels):
        true_mask = y_true[:, true_index] == 1
        support = int(true_mask.sum())
        if support == 0:
            continue
        false_negative_mask = true_mask & (y_pred[:, true_index] == 0)
        for pred_index, pred_label in enumerate(labels):
            if pred_index == true_index:
                continue
            count = int((false_negative_mask & (y_pred[:, pred_index] == 1)).sum())
            if count > 0:
                rows.append({
                    "true_emotion": true_label,
                    "predicted_instead_or_extra": pred_label,
                    "count": count,
                    "true_support": support,
                    "rate_within_true_label": count / support,
                    "rare_true_label": true_label in RARE_EMOTIONS,
                })
    return pd.DataFrame(rows).sort_values(["count", "rate_within_true_label"], ascending=[False, False])


def cooccurrence_lift(y: np.ndarray, labels: list[str]) -> pd.DataFrame:
    n = len(y)
    supports = y.sum(axis=0)
    co = y.T @ y
    rows = []
    for i, first in enumerate(labels):
        for j in range(i + 1, len(labels)):
            count = int(co[i, j])
            if count == 0:
                continue
            expected = (supports[i] / n) * (supports[j] / n) * n
            rows.append({
                "emotion_a": first,
                "emotion_b": labels[j],
                "cooccurrence_count": count,
                "support_a": int(supports[i]),
                "support_b": int(supports[j]),
                "jaccard": count / max(int(supports[i] + supports[j] - count), 1),
                "lift": count / max(expected, 1e-9),
            })
    return pd.DataFrame(rows).sort_values(["cooccurrence_count", "lift"], ascending=[False, False])


def top_extra_labels(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    false_positive = ((y_true == 0) & (y_pred == 1)).sum(axis=0)
    false_negative = ((y_true == 1) & (y_pred == 0)).sum(axis=0)
    true_positive = ((y_true == 1) & (y_pred == 1)).sum(axis=0)
    rows = []
    for index, label in enumerate(labels):
        rows.append({
            "emotion": label,
            "true_positive": int(true_positive[index]),
            "false_positive": int(false_positive[index]),
            "false_negative": int(false_negative[index]),
            "over_prediction_ratio": int(false_positive[index]) / max(int(true_positive[index] + false_positive[index]), 1),
            "miss_rate": int(false_negative[index]) / max(int(true_positive[index] + false_negative[index]), 1),
        })
    return pd.DataFrame(rows).sort_values(["false_positive", "false_negative"], ascending=False)


def write_heatmap(matrix: pd.DataFrame, title: str, path: Path, max_labels: int = 18):
    labels = list(matrix.index[:max_labels])
    data = matrix.loc[labels, labels]
    cell = 34
    left = 150
    top = 120
    width = left + cell * len(labels) + 40
    height = top + cell * len(labels) + 40
    max_value = max(float(data.values.max()), 1.0)
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#1f2933">{title}</text>',
    ]
    for j, label in enumerate(labels):
        x = left + j * cell + 20
        rows.append(f'<text x="{x}" y="{top - 8}" transform="rotate(-50 {x},{top - 8})" font-family="Arial" font-size="11" fill="#25313b">{label}</text>')
    for i, row_label in enumerate(labels):
        y = top + i * cell
        rows.append(f'<text x="24" y="{y + 22}" font-family="Arial" font-size="11" fill="#25313b">{row_label}</text>')
        for j, col_label in enumerate(labels):
            value = float(data.loc[row_label, col_label])
            intensity = value / max_value
            red = int(245 - intensity * 150)
            green = int(247 - intensity * 80)
            blue = int(250 - intensity * 45)
            x = left + j * cell
            rows.append(f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" fill="rgb({red},{green},{blue})" stroke="#eef1f3"/>')
            if value > 0 and intensity > 0.35:
                rows.append(f'<text x="{x + 8}" y="{y + 21}" font-family="Arial" font-size="10" fill="#111827">{int(value)}</text>')
    rows.append("</svg>")
    path.write_text("\n".join(rows))


def markdown_table(df: pd.DataFrame, columns: list[str], n: int = 15) -> str:
    view = df[columns].head(n).copy()
    for col in view.columns:
        if view[col].dtype.kind in "fc":
            view[col] = view[col].map(lambda value: f"{value:.3f}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    texts, labels, label_names = load_dataset(args.data, None, args.seed, "nonzero")
    train_validation_texts, test_texts, y_train_validation, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
    )
    validation_fraction = args.validation_size / (1 - args.test_size)
    _train_texts, validation_texts, y_train, y_validation = train_test_split(
        train_validation_texts,
        y_train_validation,
        test_size=validation_fraction,
        random_state=args.seed,
    )
    y_train, y_validation, y_test, label_names = drop_untrainable_labels(
        y_train,
        y_validation,
        y_test,
        label_names,
    )

    raw_val_prob, model_labels = predict_local_model_probabilities(validation_texts, args.model_path, args.batch_size)
    val_prob = filter_model_probability_columns(raw_val_prob, label_names, model_labels)
    thresholds = tune_per_label_thresholds(val_prob, y_validation)

    raw_test_prob, test_model_labels = predict_local_model_probabilities(test_texts, args.model_path, args.batch_size)
    test_prob = filter_model_probability_columns(raw_test_prob, label_names, test_model_labels)
    y_pred = apply_thresholds(test_prob, thresholds)

    true_co_matrix = y_test.T @ y_test
    pred_co_matrix = y_pred.T @ y_pred
    missed_then_extra = confusion_pairs(y_test, y_pred, label_names)
    true_co = upper_pairs(true_co_matrix, label_names, "true_cooccurrence_count")
    pred_co = upper_pairs(pred_co_matrix, label_names, "predicted_cooccurrence_count")
    true_lift = cooccurrence_lift(y_test, label_names)
    error_profile = top_extra_labels(y_test, y_pred, label_names)

    missed_then_extra.to_csv(args.output_dir / "pairwise_confused_emotion_pairs.csv", index=False)
    true_co.to_csv(args.output_dir / "true_cooccurring_emotion_pairs.csv", index=False)
    pred_co.to_csv(args.output_dir / "predicted_cooccurring_emotion_pairs.csv", index=False)
    true_lift.to_csv(args.output_dir / "true_cooccurrence_lift_pairs.csv", index=False)
    error_profile.to_csv(args.output_dir / "per_emotion_fp_fn_profile.csv", index=False)
    pd.DataFrame({"emotion": label_names, "threshold": thresholds}).to_csv(args.output_dir / "per_label_thresholds.csv", index=False)

    true_matrix_df = pd.DataFrame(true_co_matrix, index=label_names, columns=label_names)
    pred_matrix_df = pd.DataFrame(pred_co_matrix, index=label_names, columns=label_names)
    confusion_matrix = np.zeros((len(label_names), len(label_names)), dtype=int)
    for row in missed_then_extra.itertuples(index=False):
        confusion_matrix[label_names.index(row.true_emotion), label_names.index(row.predicted_instead_or_extra)] = row.count
    confusion_df = pd.DataFrame(confusion_matrix, index=label_names, columns=label_names)
    true_matrix_df.to_csv(args.output_dir / "true_cooccurrence_matrix.csv")
    pred_matrix_df.to_csv(args.output_dir / "predicted_cooccurrence_matrix.csv")
    confusion_df.to_csv(args.output_dir / "pairwise_confusion_matrix.csv")

    order = list(true_matrix_df.sum(axis=1).sort_values(ascending=False).index)
    write_heatmap(true_matrix_df.loc[order, order], "True Emotion Co-occurrence Heatmap", args.output_dir / "figure_true_cooccurrence_heatmap.svg")
    write_heatmap(pred_matrix_df.loc[order, order], "Predicted Emotion Co-occurrence Heatmap", args.output_dir / "figure_predicted_cooccurrence_heatmap.svg")
    write_heatmap(confusion_df.loc[order, order], "Pairwise Confusion Heatmap", args.output_dir / "figure_pairwise_confusion_heatmap.svg")

    doc = f"""# Pairwise Emotion Confusion Analysis

This analysis evaluates pairwise emotion behaviour for the final RoBERTa model using the same improved dataset split and per-label thresholding strategy as the final model comparison. Pairwise analysis is important because ReflectAI Journal is a multi-label system: journal entries may validly contain several emotions at once.

## Most Frequently Confused Emotion Pairs

The table below reports cases where a true emotion was missed while another emotion was predicted for the same entry. This is not a single-label confusion matrix; it is a multi-label error view that identifies likely semantic substitutions or over-predicted neighbouring labels.

{markdown_table(missed_then_extra, ["true_emotion", "predicted_instead_or_extra", "count", "true_support", "rate_within_true_label", "rare_true_label"], args.top_n)}

## Most Frequent True Co-occurring Emotion Pairs

The following pairs appeared together most often in the ground-truth labels. These pairs represent natural emotional co-occurrence patterns in the dataset.

{markdown_table(true_co, ["emotion_a", "emotion_b", "true_cooccurrence_count"], args.top_n)}

## Strongest Co-occurrence Pairs by Lift

Raw counts favour frequent emotions. Lift highlights pairs that co-occur more often than expected from their individual frequencies.

{markdown_table(true_lift, ["emotion_a", "emotion_b", "cooccurrence_count", "jaccard", "lift"], args.top_n)}

## Most Frequent Predicted Co-occurring Emotion Pairs

The following pairs were most commonly predicted together by RoBERTa. Comparing this table against true co-occurrence indicates whether the model reproduces dataset label relationships or over-combines certain emotions.

{markdown_table(pred_co, ["emotion_a", "emotion_b", "predicted_cooccurrence_count"], args.top_n)}

## Per-Emotion False Positive / False Negative Profile

This table identifies emotions most affected by over-prediction and under-prediction.

{markdown_table(error_profile, ["emotion", "true_positive", "false_positive", "false_negative", "over_prediction_ratio", "miss_rate"], args.top_n)}

## Discussion Points

- Pairwise confusion should be interpreted as semantic overlap rather than a strict error matrix because the task is multi-label.
- High confusion between related labels is expected for emotion categories that share valence or arousal, such as sadness/disappointment/grief, fear/nervousness/confusion, anger/annoyance/disapproval, and joy/gratitude/admiration.
- True co-occurrence pairs show which emotions naturally appear together in the dataset. These relationships can justify future label-correlation modelling.
- Predicted co-occurrence pairs show whether RoBERTa learned these relationships or tended to over-predict common emotion clusters.
- Rare emotions may appear in confusion pairs because they are missed and replaced by broader neighbouring labels.

## Publication-Ready Figures

- `backend/experiments/results/pairwise_emotion_analysis/figure_true_cooccurrence_heatmap.svg`
- `backend/experiments/results/pairwise_emotion_analysis/figure_predicted_cooccurrence_heatmap.svg`
- `backend/experiments/results/pairwise_emotion_analysis/figure_pairwise_confusion_heatmap.svg`

## Recommended Dissertation Paragraph

Pairwise analysis showed that the emotion classification problem is not only imbalanced but also semantically entangled. Several emotion labels frequently co-occurred, and the model sometimes confused or jointly predicted neighbouring categories. This supports the decision to frame the task as multi-label classification rather than single-label classification. It also motivates future work on explicit label-correlation modelling, such as classifier chains, label-wise attention, or graph-based emotion dependency models.
"""
    (docs_dir / "pairwise_emotion_confusion_analysis.md").write_text(doc)
    print(f"Saved pairwise analysis to {args.output_dir}")
    print("Saved docs/pairwise_emotion_confusion_analysis.md")


if __name__ == "__main__":
    main()
