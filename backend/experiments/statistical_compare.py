from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run paired statistical comparison across repeated experiment seeds.")
    parser.add_argument("--results", type=Path, default=Path("backend/experiments/results/multiseed/method_comparison.csv"))
    parser.add_argument("--baseline", default="tfidf_logreg")
    parser.add_argument("--candidate", default="local_distilbert")
    parser.add_argument("--metric", default="micro_f1")
    parser.add_argument("--output", type=Path, default=Path("backend/experiments/results/multiseed/statistical_comparison.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.results)
    required_columns = {"method", "seed", args.metric}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.results}: {sorted(missing)}")

    pivot = df.pivot_table(index="seed", columns="method", values=args.metric, aggfunc="first")
    paired = pivot[[args.baseline, args.candidate]].dropna()
    if len(paired) < 2:
        raise ValueError("At least two paired seeds are required for statistical comparison.")

    differences = paired[args.candidate] - paired[args.baseline]
    result = {
        "metric": args.metric,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "paired_seeds": [int(seed) for seed in paired.index],
        "baseline_mean": float(paired[args.baseline].mean()),
        "candidate_mean": float(paired[args.candidate].mean()),
        "mean_difference": float(differences.mean()),
        "std_difference": float(differences.std(ddof=1)),
    }

    try:
        from scipy.stats import ttest_rel

        test = ttest_rel(paired[args.candidate], paired[args.baseline])
        result["paired_t_statistic"] = float(test.statistic)
        result["paired_t_p_value"] = float(test.pvalue)
    except Exception as exc:
        result["paired_t_test_error"] = str(exc)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
