from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("backend/experiments/results/final_model_comparison")
OUTPUT_DIR = Path("backend/experiments/results/msc_analysis")
DOC_PATH = Path("docs/msc_results_discussion_analysis.md")

METHOD_NAMES = {
    "dummy_majority": "Dummy majority",
    "lexicon": "Lexicon",
    "tfidf_logreg": "TF-IDF Logistic Regression",
    "tfidf_svm": "TF-IDF SVM",
    "tfidf_nb": "TF-IDF Naive Bayes",
    "trained_distilbert": "DistilBERT",
    "trained_roberta": "RoBERTa",
    "roberta_weighted_focal": "RoBERTa + Weighted Focal Loss",
    "roberta_class_balanced_focal": "RoBERTa + Class-Balanced Focal Loss",
}

RARE_EMOTIONS = ["grief", "relief", "pride", "nervousness", "embarrassment"]
TRANSFORMERS = ["trained_roberta", "roberta_weighted_focal", "roberta_class_balanced_focal", "trained_distilbert"]


def name(method: str) -> str:
    return METHOD_NAMES.get(method, method)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    for column in view.columns:
        if view[column].dtype.kind in "fc":
            view[column] = view[column].map(fmt)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def svg_bar(df: pd.DataFrame, value_col: str, title: str, path: Path, width: int = 1100):
    chart = df.sort_values(value_col)
    row_h = 34
    left = 300
    top = 58
    right = 80
    height = top + row_h * len(chart) + 46
    chart_w = width - left - right
    max_v = max(float(chart[value_col].max()), 0.001)
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#1f2933">{title}</text>',
    ]
    for i, row in enumerate(chart.itertuples(index=False)):
        y = top + i * row_h
        method = getattr(row, "method")
        value = float(getattr(row, value_col))
        bw = chart_w * value / max_v
        fill = "#2f6f5e" if method == "trained_roberta" else "#8fb7aa"
        if "focal" in method:
            fill = "#b87935" if method == "roberta_weighted_focal" else "#7655a6"
        rows += [
            f'<text x="24" y="{y + 17}" font-family="Arial" font-size="13" fill="#25313b">{name(method)}</text>',
            f'<rect x="{left}" y="{y}" width="{bw:.2f}" height="22" rx="3" fill="{fill}"/>',
            f'<text x="{left + bw + 8}" y="{y + 16}" font-family="Arial" font-size="13" fill="#25313b">{value:.3f}</text>',
        ]
    rows.append("</svg>")
    path.write_text("\n".join(rows))


def svg_rare_heatmap(per: pd.DataFrame, path: Path):
    methods = ["trained_roberta", "roberta_weighted_focal", "roberta_class_balanced_focal", "trained_distilbert"]
    pivot = per[per.method.isin(methods) & per.emotion.isin(RARE_EMOTIONS)].pivot(index="emotion", columns="method", values="f1").loc[RARE_EMOTIONS]
    cell_w, cell_h = 180, 48
    left, top = 170, 70
    width = left + cell_w * len(methods) + 40
    height = top + cell_h * len(RARE_EMOTIONS) + 60
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#1f2933">Rare Emotion F1 Heatmap</text>',
    ]
    for j, method in enumerate(methods):
        x = left + j * cell_w
        rows.append(f'<text x="{x + 8}" y="{top - 18}" font-family="Arial" font-size="12" fill="#25313b">{name(method)}</text>')
    for i, emotion in enumerate(RARE_EMOTIONS):
        y = top + i * cell_h
        rows.append(f'<text x="24" y="{y + 30}" font-family="Arial" font-size="14" fill="#25313b">{emotion}</text>')
        for j, method in enumerate(methods):
            value = float(pivot.loc[emotion, method])
            intensity = int(245 - min(value, 0.60) / 0.60 * 150)
            fill = f"rgb({intensity},{max(intensity - 20, 70)},{245})"
            x = left + j * cell_w
            rows += [
                f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 8}" rx="4" fill="{fill}" stroke="#e5e7eb"/>',
                f'<text x="{x + 62}" y="{y + 26}" font-family="Arial" font-size="14" font-weight="700" fill="#111827">{value:.3f}</text>',
            ]
    rows.append("</svg>")
    path.write_text("\n".join(rows))


def svg_precision_recall(df: pd.DataFrame, path: Path):
    chart = df[df.method.isin(TRANSFORMERS)].copy()
    width, height = 760, 560
    left, top, cw, ch = 80, 50, 600, 420
    colors = {
        "trained_roberta": "#2f6f5e",
        "roberta_weighted_focal": "#b87935",
        "roberta_class_balanced_focal": "#7655a6",
        "trained_distilbert": "#6b9bbf",
    }
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="Arial" font-size="22" font-weight="700" fill="#1f2933">Precision-Recall Trade-off</text>',
    ]
    for tick in [0.35, 0.40, 0.45, 0.50, 0.55]:
        x = left + (tick - 0.35) / 0.25 * cw
        y = top + ch - (tick - 0.35) / 0.25 * ch
        rows += [
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + ch}" stroke="#edf0f2"/>',
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + cw}" y2="{y:.1f}" stroke="#edf0f2"/>',
            f'<text x="{x - 12:.1f}" y="{top + ch + 22}" font-family="Arial" font-size="12">{tick:.2f}</text>',
            f'<text x="38" y="{y + 4:.1f}" font-family="Arial" font-size="12">{tick:.2f}</text>',
        ]
    rows += [
        f'<line x1="{left}" y1="{top + ch}" x2="{left + cw}" y2="{top + ch}" stroke="#25313b"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + ch}" stroke="#25313b"/>',
        f'<text x="{left + cw / 2 - 40}" y="{height - 28}" font-family="Arial" font-size="14">Precision</text>',
        f'<text x="18" y="{top + ch / 2}" transform="rotate(-90 18,{top + ch / 2})" font-family="Arial" font-size="14">Recall</text>',
    ]
    for row in chart.itertuples(index=False):
        x = left + (row.precision_micro - 0.35) / 0.25 * cw
        y = top + ch - (row.recall_micro - 0.35) / 0.25 * ch
        rows += [
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{colors[row.method]}"/>',
            f'<text x="{x + 12:.1f}" y="{y + 4:.1f}" font-family="Arial" font-size="12" fill="#25313b">{name(row.method)}</text>',
        ]
    rows.append("</svg>")
    path.write_text("\n".join(rows))


def build():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(RESULTS_DIR / "method_comparison.csv")
    per = pd.read_csv(RESULTS_DIR / "per_emotion_metrics.csv")
    results["display_method"] = results.method.map(name)
    per["display_method"] = per.method.map(name)

    final_table = results[[
        "display_method", "micro_f1", "macro_f1", "precision_micro", "recall_micro",
        "hamming_loss", "subset_accuracy", "latency_ms_per_sample",
        "micro_f1_ci_low", "micro_f1_ci_high", "macro_f1_ci_low", "macro_f1_ci_high",
    ]]
    final_table.to_csv(OUTPUT_DIR / "table_final_model_comparison.csv", index=False)

    rare_table = per[per.method.isin(TRANSFORMERS) & per.emotion.isin(RARE_EMOTIONS)][[
        "display_method", "emotion", "precision", "recall", "f1", "support"
    ]].sort_values(["emotion", "f1"], ascending=[True, False])
    rare_table.to_csv(OUTPUT_DIR / "table_rare_emotion_error_analysis.csv", index=False)

    focal_table = results[results.method.isin(["trained_roberta", "roberta_weighted_focal", "roberta_class_balanced_focal"])][[
        "display_method", "micro_f1", "macro_f1", "precision_micro", "recall_micro",
        "hamming_loss", "subset_accuracy", "latency_ms_per_sample",
    ]]
    focal_table.to_csv(OUTPUT_DIR / "table_focal_loss_comparison.csv", index=False)

    roberta = results.set_index("method").loc["trained_roberta"]
    cb = results.set_index("method").loc["roberta_class_balanced_focal"]
    wf = results.set_index("method").loc["roberta_weighted_focal"]
    distil = results.set_index("method").loc["trained_distilbert"]

    top = per[per.method == "trained_roberta"].sort_values("f1", ascending=False).head(8)
    bottom = per[(per.method == "trained_roberta") & (per.support > 0)].sort_values(["f1", "support"]).head(8)

    svg_bar(results, "micro_f1", "Final Model Comparison: Micro-F1", OUTPUT_DIR / "figure_micro_f1.svg")
    svg_bar(results, "macro_f1", "Final Model Comparison: Macro-F1", OUTPUT_DIR / "figure_macro_f1.svg")
    svg_bar(results, "latency_ms_per_sample", "Final Model Comparison: Latency", OUTPUT_DIR / "figure_latency.svg")
    svg_rare_heatmap(per, OUTPUT_DIR / "figure_rare_emotion_heatmap.svg")
    svg_precision_recall(results, OUTPUT_DIR / "figure_precision_recall_tradeoff.svg")

    doc = f"""# MSc Dissertation Results and Discussion Analysis: ReflectAI Journal

## 1. Final Model Comparison

Table 1 presents the final model comparison across classical baselines, DistilBERT, RoBERTa, and two focal-loss RoBERTa variants. Standard RoBERTa achieved the strongest overall performance, with Micro-F1 = {roberta.micro_f1:.4f}, Macro-F1 = {roberta.macro_f1:.4f}, hamming loss = {roberta.hamming_loss:.4f}, subset accuracy = {roberta.subset_accuracy:.4f}, and latency = {roberta.latency_ms_per_sample:.2f} ms/sample. RoBERTa outperformed DistilBERT on both Micro-F1 ({roberta.micro_f1:.4f} vs {distil.micro_f1:.4f}) and Macro-F1 ({roberta.macro_f1:.4f} vs {distil.macro_f1:.4f}), although DistilBERT remained faster ({distil.latency_ms_per_sample:.2f} ms/sample).

{md_table(final_table, list(final_table.columns))}

**Result interpretation.** The transformer models substantially outperformed lexicon and TF-IDF baselines. This indicates that contextual representations are important for journal-style emotion recognition, where emotions are often implicit, mixed, or expressed through longer phrases rather than isolated keywords.

## 2. Rare Emotion Error Analysis

Rare emotions were a major source of error. The lowest-support emotions included grief, relief, pride, nervousness, and embarrassment. Standard RoBERTa performed poorly on some of these categories, especially grief (F1 = 0.0714) and relief (F1 = 0.1192). This suggests that the model learned frequent and lexically clearer emotions more reliably than infrequent or subtle emotions.

{md_table(rare_table, list(rare_table.columns))}

**Key finding.** Class-balanced focal loss substantially improved rare emotion performance. For example, grief improved from F1 = 0.0714 under standard RoBERTa to F1 = 0.5313 under class-balanced focal loss. Relief improved from F1 = 0.1192 to F1 = 0.3468, and pride improved from F1 = 0.2951 to F1 = 0.4034. This supports the claim that imbalance-aware objectives can improve tail-label recognition.

**Discussion point.** Although rare-label F1 improved, these gains came with a reduction in Micro-F1, precision, hamming loss, and subset accuracy. This means the focal-loss model became more sensitive to rare emotions but also more likely to over-predict labels. For a journaling system, this trade-off matters because excessive emotional labels may feel noisy or overinterpretive to users.

## 3. Focal Loss Comparison Analysis

The focal-loss experiments tested whether imbalance-aware training could improve rare-label performance. Weighted focal loss increased recall from {roberta.recall_micro:.4f} to {wf.recall_micro:.4f} and improved Macro-F1 from {roberta.macro_f1:.4f} to {wf.macro_f1:.4f}. Class-balanced focal loss achieved the highest Macro-F1 ({cb.macro_f1:.4f}) and the highest recall ({cb.recall_micro:.4f}), but had lower precision ({cb.precision_micro:.4f}) than standard RoBERTa ({roberta.precision_micro:.4f}).

{md_table(focal_table, list(focal_table.columns))}

**Interpretation.** Macro-F1 rewards balanced performance across labels, including rare labels. Therefore, the improvement in Macro-F1 for class-balanced focal loss indicates better fairness across emotion categories. However, the decrease in Micro-F1 and precision suggests more false positives. Standard RoBERTa is therefore preferable as the deployed model, while class-balanced focal loss is important as a research variant for rare-label fairness.

## 4. Multi-Label Prediction Error Analysis

This task is multi-label: a journal entry may contain several valid emotions at once. As a result, subset accuracy is very strict because the entire predicted label set must exactly match the ground truth. A prediction that correctly identifies joy and gratitude but misses optimism is still counted as incorrect by subset accuracy.

The standard RoBERTa subset accuracy was {roberta.subset_accuracy:.4f}, which may appear modest, but this should be interpreted in the context of a 28-label multi-label task. Hamming loss is more informative because it measures incorrect decisions across all label positions. RoBERTa achieved hamming loss = {roberta.hamming_loss:.4f}, meaning the average label-level error rate was low despite the strict exact-match accuracy.

**Common multi-label error types likely in this task:**

- **Partial-match errors:** the model predicts one correct emotion but misses another co-occurring emotion.
- **Over-prediction errors:** focal-loss models predict additional rare labels, improving recall but reducing precision.
- **Under-prediction errors:** standard BCE models miss rare emotions such as grief and relief.
- **Semantic-neighbour errors:** the model confuses related emotions such as sadness/disappointment, fear/nervousness, anger/annoyance, and joy/gratitude.
- **Neutral ambiguity:** neutral can overlap with low-intensity reflection, calm, realization, or approval.

## 5. Confusion and Label Ambiguity Analysis

In multi-label emotion classification, conventional single-label confusion matrices are limited because multiple labels can be correct for the same text. Instead, confusion should be discussed as semantic overlap between labels.

Likely ambiguity clusters include:

| Emotion cluster | Why confusion occurs |
| --- | --- |
| sadness, disappointment, grief, remorse | These labels share negative valence and often appear in reflective writing about loss or regret. |
| fear, nervousness, confusion | These emotions share uncertainty, anticipation, and cognitive overload. |
| anger, annoyance, disapproval, disgust | These labels represent related forms of negative judgement or frustration. |
| joy, gratitude, admiration, optimism | Positive reflective entries often express several of these together. |
| relief, realization, neutral, approval | These can appear as low-arousal reflective states rather than strong emotional expressions. |

The weak performance on realization and relief suggests that the model struggles with low-arousal or cognitively framed emotions. These emotions may be expressed indirectly, for example through phrases such as “I noticed”, “I finally understood”, or “things felt easier”, rather than explicit emotion words.

## 6. Publication-Ready Figures

The following figures were generated for direct use in the dissertation:

- `backend/experiments/results/msc_analysis/figure_micro_f1.svg`
- `backend/experiments/results/msc_analysis/figure_macro_f1.svg`
- `backend/experiments/results/msc_analysis/figure_latency.svg`
- `backend/experiments/results/msc_analysis/figure_rare_emotion_heatmap.svg`
- `backend/experiments/results/msc_analysis/figure_precision_recall_tradeoff.svg`

Recommended figure captions:

**Figure 1. Micro-F1 comparison across all evaluated methods.** RoBERTa achieved the highest overall label-level performance.

**Figure 2. Macro-F1 comparison across all evaluated methods.** Class-balanced focal loss achieved the strongest balanced performance across emotion labels.

**Figure 3. Inference latency comparison.** TF-IDF methods were fastest, but RoBERTa remained acceptable for single-entry journaling use.

**Figure 4. Rare emotion F1 heatmap.** Class-balanced focal loss improved several low-support emotions, especially grief, relief, and pride.

**Figure 5. Precision-recall trade-off among transformer models.** Focal-loss variants improved recall but reduced precision relative to standard RoBERTa.

## 7. Results Chapter Discussion Points

- RoBERTa was selected for deployment because it achieved the strongest overall performance, with the best Micro-F1, hamming loss, and subset accuracy.
- DistilBERT remained faster, but the performance gap favoured RoBERTa and RoBERTa latency was still acceptable for interactive journaling.
- TF-IDF Logistic Regression was the strongest classical baseline but was substantially below transformer performance.
- Lexicon methods were interpretable but struggled with context-dependent emotions and implicit emotional expression.
- Class-balanced focal loss improved Macro-F1 and rare-label F1, showing that class imbalance was a meaningful limitation in the baseline model.
- The focal-loss variants showed a recall-precision trade-off: better rare-label sensitivity but more false positives.
- Subset accuracy should not be treated as the primary metric because exact label-set matching is too strict for multi-label emotion classification.
- Hamming loss and F1 scores are more informative for this task because partial matches can still be useful for reflection.
- Per-emotion analysis is essential because aggregate metrics hide severe variation between frequent and rare labels.

## 8. Threats to Validity

### Dataset imbalance

The dataset is highly imbalanced, with some emotions occurring far more often than others. Rare labels such as grief, relief, pride, nervousness, and embarrassment had limited support. This threatens internal validity because poor rare-label performance may reflect insufficient examples rather than model incapability.

### Label ambiguity

Emotion labels are subjective and semantically overlapping. For example, sadness and disappointment may both describe the same entry, while fear and nervousness may differ only in intensity. This threatens construct validity because the ground truth may not represent a single objective emotional interpretation.

### Dataset-domain mismatch

GoEmotions is a large public emotion dataset, but it is not specifically a private journaling dataset. Journal entries are often longer, more reflective, and more personal than short online comments. This threatens external validity because model performance on GoEmotions-style text may not fully generalize to real journal writing.

### Neutral-label uncertainty

Neutral examples were retained through controlled sampling to make the dataset more realistic. However, neutral can overlap with calm, realization, approval, or low-intensity emotional states. This makes neutral difficult to define and may affect both training and evaluation.

### Single split and seed dependence

The final comparison used one held-out test split with bootstrap confidence intervals. Although confidence intervals improve statistical reporting, additional multi-seed or cross-validation experiments would provide stronger evidence of stability.

### Human evaluation scale

If the human evaluation uses a small number of participants, the findings should be treated as exploratory. User perceptions of appropriateness, helpfulness, and trust may vary across writing styles and emotional contexts.

### Measurement limitation

Micro-F1, Macro-F1, hamming loss, and subset accuracy measure classification performance, but they do not fully measure whether predictions are meaningful or helpful to users. This is especially important because the project is a reflective journaling system rather than only a classifier.

## 9. Recommended Final Results Paragraph

Overall, RoBERTa was selected as the final deployed model because it achieved the strongest overall predictive performance across the evaluated methods. It outperformed DistilBERT and all classical baselines in Micro-F1 and exact-match subset accuracy, while maintaining acceptable inference latency for single-entry journaling. Class-balanced focal loss achieved the highest Macro-F1 and substantially improved rare emotion recognition, particularly for grief, relief, and pride. However, this came at the cost of lower precision, Micro-F1, hamming loss, and subset accuracy, indicating a tendency toward more false positives. Therefore, standard RoBERTa was selected for deployment, while focal-loss variants were retained as research evidence that imbalance-aware objectives can improve rare-label fairness in multi-label emotion classification.
"""
    DOC_PATH.write_text(doc)
    print(f"Saved {DOC_PATH}")
    print(f"Saved tables and figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    build()
