# MSc Dissertation Results and Discussion Analysis: ReflectAI Journal

## 1. Final Model Comparison

Table 1 presents the final model comparison across classical baselines, DistilBERT, RoBERTa, and two focal-loss RoBERTa variants. Standard RoBERTa achieved the strongest overall performance, with Micro-F1 = 0.4938, Macro-F1 = 0.4326, hamming loss = 0.0469, subset accuracy = 0.3225, and latency = 9.61 ms/sample. RoBERTa outperformed DistilBERT on both Micro-F1 (0.4938 vs 0.4686) and Macro-F1 (0.4326 vs 0.4045), although DistilBERT remained faster (5.49 ms/sample).

| display_method | micro_f1 | macro_f1 | precision_micro | recall_micro | hamming_loss | subset_accuracy | latency_ms_per_sample | micro_f1_ci_low | micro_f1_ci_high | macro_f1_ci_low | macro_f1_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RoBERTa | 0.4938 | 0.4326 | 0.4699 | 0.5203 | 0.0469 | 0.3225 | 9.6065 | 0.4863 | 0.5010 | 0.4222 | 0.4424 |
| RoBERTa + Weighted Focal Loss | 0.4796 | 0.4382 | 0.4187 | 0.5613 | 0.0535 | 0.2432 | 9.5759 | 0.4728 | 0.4859 | 0.4276 | 0.4479 |
| RoBERTa + Class-Balanced Focal Loss | 0.4694 | 0.4560 | 0.4030 | 0.5619 | 0.0559 | 0.2281 | 9.6165 | 0.4615 | 0.4758 | 0.4461 | 0.4663 |
| DistilBERT | 0.4686 | 0.4045 | 0.4178 | 0.5335 | 0.0532 | 0.2582 | 5.4861 | 0.4613 | 0.4755 | 0.3954 | 0.4141 |
| TF-IDF Logistic Regression | 0.3730 | 0.3432 | 0.3089 | 0.4704 | 0.0695 | 0.1423 | 0.0138 | 0.3667 | 0.3790 | 0.3339 | 0.3535 |
| TF-IDF SVM | 0.3474 | 0.2987 | 0.3407 | 0.3544 | 0.0585 | 0.1681 | 0.0143 | 0.3402 | 0.3543 | 0.2888 | 0.3072 |
| Lexicon | 0.2968 | 0.2382 | 0.3085 | 0.2860 | 0.0596 | 0.2391 | 0.0279 | 0.2880 | 0.3042 | 0.2268 | 0.2477 |
| TF-IDF Naive Bayes | 0.1294 | 0.0835 | 0.3720 | 0.0783 | 0.0463 | 0.0619 | 0.0182 | 0.1224 | 0.1365 | 0.0781 | 0.0891 |
| Dummy majority | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0440 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Result interpretation.** The transformer models substantially outperformed lexicon and TF-IDF baselines. This indicates that contextual representations are important for journal-style emotion recognition, where emotions are often implicit, mixed, or expressed through longer phrases rather than isolated keywords.

## 2. Rare Emotion Error Analysis

Rare emotions were a major source of error. The lowest-support emotions included grief, relief, pride, nervousness, and embarrassment. Standard RoBERTa performed poorly on some of these categories, especially grief (F1 = 0.0714) and relief (F1 = 0.1192). This suggests that the model learned frequent and lexically clearer emotions more reliably than infrequent or subtle emotions.

| display_method | emotion | precision | recall | f1 | support |
| --- | --- | --- | --- | --- | --- |
| RoBERTa + Class-Balanced Focal Loss | embarrassment | 0.2525 | 0.4146 | 0.3138 | 123 |
| RoBERTa + Weighted Focal Loss | embarrassment | 0.3333 | 0.2927 | 0.3117 | 123 |
| RoBERTa | embarrassment | 0.3659 | 0.2439 | 0.2927 | 123 |
| DistilBERT | embarrassment | 0.2887 | 0.2276 | 0.2545 | 123 |
| RoBERTa + Class-Balanced Focal Loss | grief | 0.7083 | 0.4250 | 0.5312 | 40 |
| RoBERTa + Weighted Focal Loss | grief | 0.3571 | 0.2500 | 0.2941 | 40 |
| RoBERTa | grief | 0.1250 | 0.0500 | 0.0714 | 40 |
| DistilBERT | grief | 0.5000 | 0.0250 | 0.0476 | 40 |
| RoBERTa + Weighted Focal Loss | nervousness | 0.2885 | 0.2857 | 0.2871 | 105 |
| RoBERTa + Class-Balanced Focal Loss | nervousness | 0.3704 | 0.1905 | 0.2516 | 105 |
| RoBERTa | nervousness | 0.4250 | 0.1619 | 0.2345 | 105 |
| DistilBERT | nervousness | 0.1579 | 0.3429 | 0.2162 | 105 |
| RoBERTa + Class-Balanced Focal Loss | pride | 0.5581 | 0.3158 | 0.4034 | 76 |
| RoBERTa | pride | 0.3913 | 0.2368 | 0.2951 | 76 |
| RoBERTa + Weighted Focal Loss | pride | 0.2038 | 0.4211 | 0.2747 | 76 |
| DistilBERT | pride | 0.5714 | 0.0526 | 0.0964 | 76 |
| RoBERTa + Class-Balanced Focal Loss | relief | 0.3125 | 0.3896 | 0.3468 | 77 |
| RoBERTa + Weighted Focal Loss | relief | 0.2174 | 0.1948 | 0.2055 | 77 |
| RoBERTa | relief | 0.1216 | 0.1169 | 0.1192 | 77 |
| DistilBERT | relief | 0.0889 | 0.0519 | 0.0656 | 77 |

**Key finding.** Class-balanced focal loss substantially improved rare emotion performance. For example, grief improved from F1 = 0.0714 under standard RoBERTa to F1 = 0.5313 under class-balanced focal loss. Relief improved from F1 = 0.1192 to F1 = 0.3468, and pride improved from F1 = 0.2951 to F1 = 0.4034. This supports the claim that imbalance-aware objectives can improve tail-label recognition.

**Discussion point.** Although rare-label F1 improved, these gains came with a reduction in Micro-F1, precision, hamming loss, and subset accuracy. This means the focal-loss model became more sensitive to rare emotions but also more likely to over-predict labels. For a journaling system, this trade-off matters because excessive emotional labels may feel noisy or overinterpretive to users.

## 3. Focal Loss Comparison Analysis

The focal-loss experiments tested whether imbalance-aware training could improve rare-label performance. Weighted focal loss increased recall from 0.5203 to 0.5613 and improved Macro-F1 from 0.4326 to 0.4382. Class-balanced focal loss achieved the highest Macro-F1 (0.4560) and the highest recall (0.5619), but had lower precision (0.4030) than standard RoBERTa (0.4699).

| display_method | micro_f1 | macro_f1 | precision_micro | recall_micro | hamming_loss | subset_accuracy | latency_ms_per_sample |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RoBERTa | 0.4938 | 0.4326 | 0.4699 | 0.5203 | 0.0469 | 0.3225 | 9.6065 |
| RoBERTa + Weighted Focal Loss | 0.4796 | 0.4382 | 0.4187 | 0.5613 | 0.0535 | 0.2432 | 9.5759 |
| RoBERTa + Class-Balanced Focal Loss | 0.4694 | 0.4560 | 0.4030 | 0.5619 | 0.0559 | 0.2281 | 9.6165 |

**Interpretation.** Macro-F1 rewards balanced performance across labels, including rare labels. Therefore, the improvement in Macro-F1 for class-balanced focal loss indicates better fairness across emotion categories. However, the decrease in Micro-F1 and precision suggests more false positives. Standard RoBERTa is therefore preferable as the deployed model, while class-balanced focal loss is important as a research variant for rare-label fairness.

## 4. Multi-Label Prediction Error Analysis

This task is multi-label: a journal entry may contain several valid emotions at once. As a result, subset accuracy is very strict because the entire predicted label set must exactly match the ground truth. A prediction that correctly identifies joy and gratitude but misses optimism is still counted as incorrect by subset accuracy.

The standard RoBERTa subset accuracy was 0.3225, which may appear modest, but this should be interpreted in the context of a 28-label multi-label task. Hamming loss is more informative because it measures incorrect decisions across all label positions. RoBERTa achieved hamming loss = 0.0469, meaning the average label-level error rate was low despite the strict exact-match accuracy.

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
