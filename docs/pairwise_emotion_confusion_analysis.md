# Pairwise Emotion Confusion Analysis

This analysis evaluates pairwise emotion behaviour for the final RoBERTa model using the same improved dataset split and per-label thresholding strategy as the final model comparison. Pairwise analysis is important because ReflectAI Journal is a multi-label system: journal entries may validly contain several emotions at once.

## Most Frequently Confused Emotion Pairs

The table below reports cases where a true emotion was missed while another emotion was predicted for the same entry. This is not a single-label confusion matrix; it is a multi-label error view that identifies likely semantic substitutions or over-predicted neighbouring labels.

| true_emotion | predicted_instead_or_extra | count | true_support | rate_within_true_label | rare_true_label |
| --- | --- | --- | --- | --- | --- |
| approval | neutral | 163 | 999 | 0.163 | False |
| neutral | approval | 114 | 2116 | 0.054 | False |
| neutral | annoyance | 106 | 2116 | 0.050 | False |
| annoyance | neutral | 100 | 670 | 0.149 | False |
| realization | neutral | 95 | 462 | 0.206 | False |
| neutral | disapproval | 91 | 2116 | 0.043 | False |
| neutral | confusion | 82 | 2116 | 0.039 | False |
| approval | admiration | 80 | 999 | 0.080 | False |
| neutral | curiosity | 80 | 2116 | 0.038 | False |
| admiration | love | 70 | 1026 | 0.068 | False |
| disappointment | neutral | 69 | 414 | 0.167 | False |
| disapproval | neutral | 68 | 632 | 0.108 | False |
| neutral | admiration | 67 | 2116 | 0.032 | False |
| admiration | approval | 62 | 1026 | 0.060 | False |
| realization | approval | 60 | 462 | 0.130 | False |
| neutral | amusement | 59 | 2116 | 0.028 | False |
| disapproval | annoyance | 58 | 632 | 0.092 | False |
| neutral | disappointment | 58 | 2116 | 0.027 | False |
| disgust | annoyance | 57 | 291 | 0.196 | False |
| anger | annoyance | 55 | 437 | 0.126 | False |
| annoyance | disapproval | 55 | 670 | 0.082 | False |
| optimism | approval | 54 | 454 | 0.119 | False |
| joy | amusement | 54 | 461 | 0.117 | False |
| neutral | anger | 53 | 2116 | 0.025 | False |
| neutral | caring | 51 | 2116 | 0.024 | False |

## Most Frequent True Co-occurring Emotion Pairs

The following pairs appeared together most often in the ground-truth labels. These pairs represent natural emotional co-occurrence patterns in the dataset.

| emotion_a | emotion_b | true_cooccurrence_count |
| --- | --- | --- |
| admiration | approval | 70 |
| admiration | gratitude | 68 |
| anger | annoyance | 68 |
| excitement | joy | 54 |
| admiration | love | 50 |
| approval | optimism | 49 |
| disappointment | sadness | 48 |
| confusion | curiosity | 47 |
| approval | realization | 46 |
| annoyance | disapproval | 46 |
| admiration | joy | 44 |
| admiration | optimism | 38 |
| admiration | excitement | 36 |
| annoyance | disgust | 35 |
| annoyance | disappointment | 35 |
| fear | nervousness | 32 |
| disapproval | disgust | 31 |
| admiration | amusement | 31 |
| gratitude | love | 31 |
| amusement | approval | 31 |
| approval | gratitude | 30 |
| curiosity | surprise | 29 |
| approval | caring | 29 |
| admiration | curiosity | 29 |
| gratitude | joy | 29 |

## Strongest Co-occurrence Pairs by Lift

Raw counts favour frequent emotions. Lift highlights pairs that co-occur more often than expected from their individual frequencies.

| emotion_a | emotion_b | cooccurrence_count | jaccard | lift |
| --- | --- | --- | --- | --- |
| admiration | approval | 70 | 0.036 | 0.714 |
| anger | annoyance | 68 | 0.065 | 2.428 |
| admiration | gratitude | 68 | 0.042 | 1.033 |
| excitement | joy | 54 | 0.075 | 3.951 |
| admiration | love | 50 | 0.034 | 1.064 |
| approval | optimism | 49 | 0.035 | 1.130 |
| disappointment | sadness | 48 | 0.064 | 3.140 |
| confusion | curiosity | 47 | 0.051 | 2.140 |
| annoyance | disapproval | 46 | 0.037 | 1.136 |
| approval | realization | 46 | 0.033 | 1.042 |
| admiration | joy | 44 | 0.030 | 0.973 |
| admiration | optimism | 38 | 0.026 | 0.853 |
| admiration | excitement | 36 | 0.028 | 1.183 |
| annoyance | disgust | 35 | 0.038 | 1.877 |
| annoyance | disappointment | 35 | 0.033 | 1.319 |
| fear | nervousness | 32 | 0.128 | 18.002 |
| disapproval | disgust | 31 | 0.035 | 1.762 |
| gratitude | love | 31 | 0.028 | 1.008 |
| amusement | approval | 31 | 0.021 | 0.600 |
| admiration | amusement | 31 | 0.020 | 0.584 |
| approval | gratitude | 30 | 0.018 | 0.468 |
| curiosity | surprise | 29 | 0.034 | 1.649 |
| approval | caring | 29 | 0.023 | 0.992 |
| gratitude | joy | 29 | 0.026 | 0.980 |
| admiration | curiosity | 29 | 0.019 | 0.535 |

## Most Frequent Predicted Co-occurring Emotion Pairs

The following pairs were most commonly predicted together by RoBERTa. Comparing this table against true co-occurrence indicates whether the model reproduces dataset label relationships or over-combines certain emotions.

| emotion_a | emotion_b | predicted_cooccurrence_count |
| --- | --- | --- |
| confusion | curiosity | 492 |
| approval | neutral | 374 |
| anger | annoyance | 305 |
| confusion | neutral | 219 |
| curiosity | neutral | 209 |
| disapproval | neutral | 182 |
| disappointment | sadness | 109 |
| approval | realization | 100 |
| realization | neutral | 94 |
| excitement | joy | 85 |
| annoyance | disapproval | 82 |
| joy | relief | 56 |
| admiration | gratitude | 56 |
| annoyance | neutral | 51 |
| annoyance | disappointment | 48 |
| admiration | approval | 46 |
| annoyance | disgust | 42 |
| approval | caring | 42 |
| gratitude | joy | 42 |
| amusement | joy | 41 |
| caring | neutral | 40 |
| approval | optimism | 40 |
| admiration | curiosity | 39 |
| disappointment | disapproval | 37 |
| excitement | surprise | 37 |

## Per-Emotion False Positive / False Negative Profile

This table identifies emotions most affected by over-prediction and under-prediction.

| emotion | true_positive | false_positive | false_negative | over_prediction_ratio | miss_rate |
| --- | --- | --- | --- | --- | --- |
| neutral | 1239 | 1201 | 877 | 0.492 | 0.414 |
| approval | 411 | 758 | 588 | 0.648 | 0.589 |
| confusion | 263 | 560 | 153 | 0.680 | 0.368 |
| curiosity | 397 | 519 | 155 | 0.567 | 0.281 |
| annoyance | 249 | 517 | 421 | 0.675 | 0.628 |
| disapproval | 298 | 457 | 334 | 0.605 | 0.528 |
| admiration | 643 | 333 | 383 | 0.341 | 0.373 |
| disappointment | 128 | 326 | 286 | 0.718 | 0.691 |
| anger | 211 | 273 | 226 | 0.564 | 0.517 |
| amusement | 399 | 252 | 142 | 0.387 | 0.262 |
| joy | 218 | 233 | 243 | 0.517 | 0.527 |
| excitement | 117 | 222 | 193 | 0.655 | 0.623 |
| caring | 133 | 222 | 173 | 0.625 | 0.565 |
| realization | 87 | 218 | 375 | 0.715 | 0.812 |
| disgust | 120 | 206 | 171 | 0.632 | 0.588 |
| love | 364 | 196 | 115 | 0.350 | 0.240 |
| optimism | 186 | 170 | 268 | 0.478 | 0.590 |
| sadness | 182 | 146 | 204 | 0.445 | 0.528 |
| surprise | 160 | 145 | 173 | 0.475 | 0.520 |
| gratitude | 545 | 113 | 126 | 0.172 | 0.188 |
| fear | 97 | 105 | 80 | 0.520 | 0.452 |
| remorse | 90 | 104 | 41 | 0.536 | 0.313 |
| desire | 83 | 96 | 101 | 0.536 | 0.549 |
| relief | 9 | 65 | 68 | 0.878 | 0.883 |
| embarrassment | 30 | 52 | 93 | 0.634 | 0.756 |

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
