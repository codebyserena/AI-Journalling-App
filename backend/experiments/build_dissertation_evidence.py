from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHOD_LABELS = {
    "dummy_majority": "Dummy majority",
    "lexicon": "Lexicon baseline",
    "tfidf_logreg": "TF-IDF Logistic Regression",
    "tfidf_svm": "TF-IDF SVM",
    "tfidf_nb": "TF-IDF Naive Bayes",
    "trained_distilbert": "DistilBERT",
    "trained_roberta": "RoBERTa",
    "roberta_weighted_focal": "RoBERTa + weighted focal loss",
    "roberta_class_balanced_focal": "RoBERTa + class-balanced focal loss",
}

RARE_LABELS = ["grief", "pride", "relief", "nervousness", "embarrassment"]
TRANSFORMER_METHODS = [
    "trained_distilbert",
    "trained_roberta",
    "roberta_weighted_focal",
    "roberta_class_balanced_focal",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build dissertation tables, charts, and evidence summary.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("backend/experiments/results/final_model_comparison"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/experiments/results/dissertation_evidence"),
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
    )
    return parser.parse_args()


def method_name(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_bar_chart(df: pd.DataFrame, column: str, title: str, output: Path):
    chart_df = df.sort_values(column, ascending=True)
    width = 980
    row_height = 34
    margin_left = 260
    margin_top = 58
    margin_right = 90
    chart_width = width - margin_left - margin_right
    height = margin_top + row_height * len(chart_df) + 44
    max_value = max(float(chart_df[column].max()), 0.001)

    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#24312e">{title}</text>',
    ]
    for index, row in enumerate(chart_df.itertuples(index=False)):
        y = margin_top + index * row_height
        value = float(getattr(row, column))
        bar_width = chart_width * (value / max_value)
        label = method_name(getattr(row, "method"))
        fill = "#3f7f6c" if getattr(row, "method") == "trained_roberta" else "#7aa899"
        rows.extend([
            f'<text x="24" y="{y + 18}" font-family="Arial" font-size="13" fill="#39423f">{label}</text>',
            f'<rect x="{margin_left}" y="{y}" width="{bar_width:.2f}" height="22" rx="4" fill="{fill}"/>',
            f'<text x="{margin_left + bar_width + 8}" y="{y + 16}" font-family="Arial" font-size="13" fill="#39423f">{value:.3f}</text>',
        ])
    rows.append("</svg>")
    output.write_text("\n".join(rows))


def write_grouped_metric_chart(df: pd.DataFrame, output: Path):
    methods = ["trained_distilbert", "trained_roberta", "roberta_weighted_focal", "roberta_class_balanced_focal"]
    metrics = ["micro_f1", "macro_f1", "precision_micro", "recall_micro"]
    colors = ["#446f64", "#6f9f8d", "#d6a35d", "#9c6f9f"]
    chart_df = df[df["method"].isin(methods)].set_index("method").loc[methods]
    width = 1100
    height = 430
    margin_left = 80
    margin_bottom = 92
    margin_top = 52
    chart_height = height - margin_top - margin_bottom
    chart_width = width - margin_left - 50
    group_width = chart_width / len(methods)
    bar_width = 34

    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="24" y="32" font-family="Arial" font-size="22" font-weight="700" fill="#24312e">Transformer Metric Trade-Offs</text>',
    ]
    for tick in [0, 0.25, 0.5, 0.75]:
        y = margin_top + chart_height - tick * chart_height
        rows.extend([
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#e4dfd4" stroke-width="1"/>',
            f'<text x="34" y="{y + 4:.1f}" font-family="Arial" font-size="12" fill="#6f746f">{tick:.2f}</text>',
        ])

    for group_index, method in enumerate(methods):
        x0 = margin_left + group_index * group_width + 22
        for metric_index, metric in enumerate(metrics):
            value = float(chart_df.loc[method, metric])
            x = x0 + metric_index * (bar_width + 6)
            bar_height = value * chart_height
            y = margin_top + chart_height - bar_height
            rows.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="4" fill="{colors[metric_index]}"/>')
        rows.append(
            f'<text x="{x0 - 8:.1f}" y="{height - 54}" font-family="Arial" font-size="12" fill="#39423f">{method_name(method)}</text>'
        )

    legend_x = 690
    for index, metric in enumerate(metrics):
        y = 20 + index * 20
        label = metric.replace("_", " ").title()
        rows.extend([
            f'<rect x="{legend_x}" y="{y}" width="12" height="12" fill="{colors[index]}"/>',
            f'<text x="{legend_x + 18}" y="{y + 11}" font-family="Arial" font-size="12" fill="#39423f">{label}</text>',
        ])
    rows.append("</svg>")
    output.write_text("\n".join(rows))


def write_rare_label_chart(per_emotion: pd.DataFrame, output: Path):
    methods = ["trained_roberta", "roberta_weighted_focal", "roberta_class_balanced_focal"]
    chart_df = per_emotion[
        per_emotion["method"].isin(methods) & per_emotion["emotion"].isin(RARE_LABELS)
    ].copy()
    pivot = chart_df.pivot(index="emotion", columns="method", values="f1").loc[RARE_LABELS]
    width = 1080
    height = 460
    margin_left = 90
    margin_top = 58
    margin_bottom = 78
    chart_height = height - margin_top - margin_bottom
    chart_width = width - margin_left - 44
    group_width = chart_width / len(RARE_LABELS)
    colors = {
        "trained_roberta": "#446f64",
        "roberta_weighted_focal": "#d6a35d",
        "roberta_class_balanced_focal": "#9c6f9f",
    }
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#24312e">Rare Emotion F1 Comparison</text>',
    ]
    for tick in [0, 0.25, 0.5, 0.75]:
        y = margin_top + chart_height - tick * chart_height
        rows.extend([
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#e4dfd4" stroke-width="1"/>',
            f'<text x="34" y="{y + 4:.1f}" font-family="Arial" font-size="12" fill="#6f746f">{tick:.2f}</text>',
        ])
    for label_index, emotion in enumerate(RARE_LABELS):
        x0 = margin_left + label_index * group_width + 30
        for method_index, method in enumerate(methods):
            value = float(pivot.loc[emotion, method])
            x = x0 + method_index * 38
            bar_height = value * chart_height
            y = margin_top + chart_height - bar_height
            rows.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="30" height="{bar_height:.1f}" rx="4" fill="{colors[method]}"/>')
            rows.append(f'<text x="{x - 2:.1f}" y="{y - 5:.1f}" font-family="Arial" font-size="10" fill="#39423f">{value:.2f}</text>')
        rows.append(f'<text x="{x0 - 8:.1f}" y="{height - 46}" font-family="Arial" font-size="13" fill="#39423f">{emotion}</text>')

    legend_x = 690
    for index, method in enumerate(methods):
        y = 18 + index * 20
        rows.extend([
            f'<rect x="{legend_x}" y="{y}" width="12" height="12" fill="{colors[method]}"/>',
            f'<text x="{legend_x + 18}" y="{y + 11}" font-family="Arial" font-size="12" fill="#39423f">{method_name(method)}</text>',
        ])
    rows.append("</svg>")
    output.write_text("\n".join(rows))


def write_ci_chart(df: pd.DataFrame, output: Path):
    chart_df = df[df["method"].isin(TRANSFORMER_METHODS)].sort_values("micro_f1", ascending=False)
    width = 980
    height = 300
    margin_left = 260
    margin_top = 58
    row_height = 44
    x_min = 0.38
    x_max = 0.52
    chart_width = width - margin_left - 70

    def x(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * chart_width

    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#24312e">Micro-F1 95% Confidence Intervals</text>',
    ]
    for tick in [0.40, 0.45, 0.50]:
        xt = x(tick)
        rows.extend([
            f'<line x1="{xt:.1f}" y1="48" x2="{xt:.1f}" y2="{height - 46}" stroke="#e4dfd4" stroke-width="1"/>',
            f'<text x="{xt - 12:.1f}" y="{height - 20}" font-family="Arial" font-size="12" fill="#6f746f">{tick:.2f}</text>',
        ])
    for index, row in enumerate(chart_df.itertuples(index=False)):
        y = margin_top + index * row_height
        rows.extend([
            f'<text x="24" y="{y + 5}" font-family="Arial" font-size="13" fill="#39423f">{method_name(row.method)}</text>',
            f'<line x1="{x(row.micro_f1_ci_low):.1f}" y1="{y:.1f}" x2="{x(row.micro_f1_ci_high):.1f}" y2="{y:.1f}" stroke="#446f64" stroke-width="4"/>',
            f'<circle cx="{x(row.micro_f1):.1f}" cy="{y:.1f}" r="7" fill="#d6a35d" stroke="#24312e" stroke-width="1"/>',
            f'<text x="{x(row.micro_f1) + 12:.1f}" y="{y + 4:.1f}" font-family="Arial" font-size="12" fill="#39423f">{row.micro_f1:.3f}</text>',
        ])
    rows.append("</svg>")
    output.write_text("\n".join(rows))


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    for column in view.columns:
        if view[column].dtype.kind in "fc":
            view[column] = view[column].map(lambda value: fmt(value, 4))
    header = "| " + " | ".join(view.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def build_summary_doc(results: pd.DataFrame, per_emotion: pd.DataFrame, output: Path):
    display_results = results.copy()
    if "display_name" not in display_results.columns:
        display_results.insert(0, "display_name", display_results["method"].map(method_name))
    rare = per_emotion[
        per_emotion["method"].isin(["trained_roberta", "roberta_weighted_focal", "roberta_class_balanced_focal"])
        & per_emotion["emotion"].isin(RARE_LABELS)
    ].copy()
    rare["display_name"] = rare["method"].map(method_name)
    rare_pivot = rare.pivot(index="emotion", columns="display_name", values="f1").reset_index()

    roberta = results.set_index("method").loc["trained_roberta"]
    cb = results.set_index("method").loc["roberta_class_balanced_focal"]
    weighted = results.set_index("method").loc["roberta_weighted_focal"]
    distilbert = results.set_index("method").loc["trained_distilbert"]

    top_roberta = (
        per_emotion[per_emotion["method"] == "trained_roberta"]
        .sort_values("f1", ascending=False)
        .head(5)
    )
    worst_roberta = (
        per_emotion[(per_emotion["method"] == "trained_roberta") & (per_emotion["support"] > 0)]
        .sort_values(["f1", "support"], ascending=[True, True])
        .head(5)
    )

    content = f"""# Dissertation Evidence Pack

## Final Model Decision

The final deployed model should be **RoBERTa** (`trained_roberta`). It achieved the strongest overall performance, with Micro-F1 {roberta.micro_f1:.4f}, Macro-F1 {roberta.macro_f1:.4f}, hamming loss {roberta.hamming_loss:.4f}, subset accuracy {roberta.subset_accuracy:.4f}, and latency {roberta.latency_ms_per_sample:.2f} ms/sample.

`roberta_class_balanced_focal` achieved the highest Macro-F1 ({cb.macro_f1:.4f}), which makes it important for the rare-label fairness discussion. However, it reduced Micro-F1 ({cb.micro_f1:.4f}), precision ({cb.precision_micro:.4f}), hamming loss ({cb.hamming_loss:.4f}), and subset accuracy ({cb.subset_accuracy:.4f}) compared with standard RoBERTa. For a journaling app, this suggests more recall and better rare-label coverage, but also more false positives and noisier predictions.

`roberta_weighted_focal` is a middle option. It improved Macro-F1 over standard RoBERTa ({weighted.macro_f1:.4f} vs {roberta.macro_f1:.4f}) and increased recall ({weighted.recall_micro:.4f} vs {roberta.recall_micro:.4f}), but it still reduced Micro-F1 and subset accuracy.

## Final Method Comparison

{markdown_table(display_results, ["display_name", "micro_f1", "macro_f1", "precision_micro", "recall_micro", "hamming_loss", "subset_accuracy", "latency_ms_per_sample"])}

## Confidence Intervals

Standard RoBERTa achieved Micro-F1 {roberta.micro_f1:.4f}, 95% CI [{roberta.micro_f1_ci_low:.4f}, {roberta.micro_f1_ci_high:.4f}], and Macro-F1 {roberta.macro_f1:.4f}, 95% CI [{roberta.macro_f1_ci_low:.4f}, {roberta.macro_f1_ci_high:.4f}].

DistilBERT achieved Micro-F1 {distilbert.micro_f1:.4f}, 95% CI [{distilbert.micro_f1_ci_low:.4f}, {distilbert.micro_f1_ci_high:.4f}], and Macro-F1 {distilbert.macro_f1:.4f}, 95% CI [{distilbert.macro_f1_ci_low:.4f}, {distilbert.macro_f1_ci_high:.4f}].

This supports the conclusion that RoBERTa outperformed DistilBERT on the final improved dataset.

## Best and Weakest RoBERTa Emotions

Best RoBERTa emotions by F1:

{markdown_table(top_roberta, ["emotion", "precision", "recall", "f1", "support"])}

Weakest RoBERTa emotions by F1:

{markdown_table(worst_roberta, ["emotion", "precision", "recall", "f1", "support"])}

## Rare Emotion Comparison

{markdown_table(rare_pivot, list(rare_pivot.columns))}

The rare-label analysis shows why focal loss is useful as a research variant. Class-balanced focal loss substantially improved labels such as grief, pride, and relief, but at the cost of lower overall Micro-F1 and subset accuracy.

## Threshold Strategy

The final comparison used **per-label thresholding**. This is appropriate because each emotion has a different frequency and difficulty level. A single global threshold can favour frequent or easy labels, while rare labels may require more sensitive thresholds. Per-label thresholding therefore supports fairer multi-label prediction across all 28 emotions.

## Runtime and Deployment

TF-IDF baselines were much faster than transformer methods, but their F1 scores were lower. RoBERTa was slower than DistilBERT ({roberta.latency_ms_per_sample:.2f} ms/sample vs {distilbert.latency_ms_per_sample:.2f} ms/sample), but this latency is acceptable for a journaling application because predictions are made one entry at a time after submission.

## Errors, Trials, and Fixes Completed

- The original experiment results were being overwritten. This was fixed by adding explicit output directories for separate runs.
- The neutral label caused warnings when it had no positive examples in a split. This was addressed through improved dataset preparation and label filtering.
- The prediction pipeline was corrected to use sigmoid multi-label outputs instead of treating emotion prediction like a single-label softmax task.
- Classical baselines were strengthened with TF-IDF n-grams, larger feature limits, class balancing, and improved SVM iteration settings.
- Per-emotion metrics and summaries were added to support error analysis.
- Bootstrap confidence intervals were added for stronger statistical reporting.
- DistilBERT and RoBERTa training scripts were added so both transformer models could be trained on the same improved dataset.
- Weighted focal loss and class-balanced focal loss were added to test rare-label improvements.
- Frontend errors such as `Failed to fetch` were addressed by aligning the frontend API URL with the backend port.
- Account creation errors were improved so users see meaningful messages such as email already registered.
- Entry deletion and data export were added for privacy and user control.
- The frontend was changed to avoid overconfident emotional wording and now presents predictions as reflective estimates.
- A feedback mechanism was added so users can mark predictions as right or submit corrections for human-centered evaluation.
- A distress-sensitive support message was added for ethically safer journaling feedback.

## Final Dissertation Argument

The project demonstrates that transformer-based multi-label emotion classification is more effective than lexicon and TF-IDF baselines for emotion-aware journaling. RoBERTa achieved the best overall predictive performance and was selected for deployment. Class-balanced focal loss achieved the best Macro-F1 and improved rare-emotion handling, but it introduced a trade-off by lowering precision, Micro-F1, and subset accuracy. Therefore, standard RoBERTa is the best final model for the application, while focal-loss variants provide valuable research evidence about class imbalance and rare-label fairness.

## Graphs Generated

Use these files in the dissertation:

- `backend/experiments/results/dissertation_evidence/overall_micro_f1.svg`
- `backend/experiments/results/dissertation_evidence/overall_macro_f1.svg`
- `backend/experiments/results/dissertation_evidence/overall_latency.svg`
- `backend/experiments/results/dissertation_evidence/transformer_metric_tradeoffs.svg`
- `backend/experiments/results/dissertation_evidence/rare_emotion_f1.svg`
- `backend/experiments/results/dissertation_evidence/micro_f1_confidence_intervals.svg`
- Existing final comparison charts in `backend/experiments/results/final_model_comparison/`
"""
    output.write_text(content)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.results_dir / "method_comparison.csv")
    per_emotion = pd.read_csv(args.results_dir / "per_emotion_metrics.csv")

    results["display_name"] = results["method"].map(method_name)
    comparison_columns = [
        "display_name", "method", "micro_f1", "macro_f1", "precision_micro", "recall_micro",
        "hamming_loss", "subset_accuracy", "latency_ms_per_sample", "micro_f1_ci_low",
        "micro_f1_ci_high", "macro_f1_ci_low", "macro_f1_ci_high",
    ]
    results[comparison_columns].to_csv(args.output_dir / "final_model_comparison_table.csv", index=False)

    decision_rows = [
        {
            "model": "TF-IDF Logistic Regression",
            "strength": "Strongest classical baseline; very fast",
            "weakness": "Lower Micro-F1 and Macro-F1 than transformer models",
            "decision": "Baseline only",
        },
        {
            "model": "DistilBERT",
            "strength": "Faster and lighter transformer",
            "weakness": "Lower Micro-F1 and Macro-F1 than RoBERTa",
            "decision": "Not selected",
        },
        {
            "model": "RoBERTa",
            "strength": "Best Micro-F1, hamming loss, and subset accuracy",
            "weakness": "Slower than DistilBERT",
            "decision": "Selected for deployment",
        },
        {
            "model": "RoBERTa + weighted focal loss",
            "strength": "Improved recall and Macro-F1 compared with standard RoBERTa",
            "weakness": "Lower precision, Micro-F1, and subset accuracy",
            "decision": "Research variant",
        },
        {
            "model": "RoBERTa + class-balanced focal loss",
            "strength": "Best Macro-F1 and rare-label performance",
            "weakness": "More false positives and lower exact-match performance",
            "decision": "Rare-label fairness variant",
        },
    ]
    pd.DataFrame(decision_rows).to_csv(args.output_dir / "final_model_decision_table.csv", index=False)

    rare = per_emotion[
        per_emotion["method"].isin(TRANSFORMER_METHODS)
        & per_emotion["emotion"].isin(RARE_LABELS)
    ].copy()
    rare["display_name"] = rare["method"].map(method_name)
    rare.to_csv(args.output_dir / "rare_emotion_comparison.csv", index=False)

    roberta_per_emotion = per_emotion[per_emotion["method"] == "trained_roberta"].copy()
    pd.concat([
        roberta_per_emotion.sort_values("f1", ascending=False).head(8).assign(group="best"),
        roberta_per_emotion[roberta_per_emotion["support"] > 0].sort_values(["f1", "support"], ascending=[True, True]).head(8).assign(group="weakest"),
    ]).to_csv(args.output_dir / "trained_roberta_best_worst_emotions.csv", index=False)

    write_bar_chart(results, "micro_f1", "Micro-F1 by Method", args.output_dir / "overall_micro_f1.svg")
    write_bar_chart(results, "macro_f1", "Macro-F1 by Method", args.output_dir / "overall_macro_f1.svg")
    write_bar_chart(results, "latency_ms_per_sample", "Latency per Sample by Method", args.output_dir / "overall_latency.svg")
    write_grouped_metric_chart(results, args.output_dir / "transformer_metric_tradeoffs.svg")
    write_rare_label_chart(per_emotion, args.output_dir / "rare_emotion_f1.svg")
    write_ci_chart(results, args.output_dir / "micro_f1_confidence_intervals.svg")

    build_summary_doc(results, per_emotion, args.docs_dir / "dissertation_evidence_pack.md")
    print(f"Saved evidence outputs to {args.output_dir}")
    print(f"Saved summary document to {args.docs_dir / 'dissertation_evidence_pack.md'}")


if __name__ == "__main__":
    main()
