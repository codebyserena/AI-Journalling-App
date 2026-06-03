from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from ftfy import fix_text

from backend.emotions import EMOTION_LABELS

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = DATA_DIR / "goemotions.csv"
DEFAULT_OUTPUT = DATA_DIR / "cleaned_journal_data_improved.csv"


def clean_reddit_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = fix_text(text).lower()
    text = re.sub(r"^>+", "", text)
    text = re.sub(r"\s*/(s|jk)\b", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\[.*?\]", " entity ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s!?.,']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin({"TRUE", "1", "YES"})


def build_dataset(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    neutral_ratio: float,
    max_neutral_only: int | None,
    min_chars: int,
    remove_unclear: bool,
    seed: int,
):
    df = pd.read_csv(input_path)
    original_rows = len(df)

    missing = [label for label in EMOTION_LABELS if label not in df.columns]
    if missing:
        raise ValueError(f"Missing emotion columns: {missing}")

    if remove_unclear and "example_very_unclear" in df.columns:
        df = df[~parse_bool_series(df["example_very_unclear"])]

    df["text"] = df["text"].apply(clean_reddit_text)
    df = df[df["text"].str.len() >= min_chars]
    df = df[["text", *EMOTION_LABELS]].drop_duplicates(subset=["text"])

    label_counts = df[EMOTION_LABELS].sum(axis=1)
    neutral_only = df[(label_counts == 1) & (df["neutral"] == 1)]
    emotional = df[~df.index.isin(neutral_only.index)]

    neutral_target = int(len(emotional) * neutral_ratio)
    if max_neutral_only is not None:
        neutral_target = min(neutral_target, max_neutral_only)
    neutral_target = min(neutral_target, len(neutral_only))

    if neutral_target > 0:
        sampled_neutral = neutral_only.sample(neutral_target, random_state=seed)
        df_out = pd.concat([emotional, sampled_neutral], ignore_index=True)
    else:
        df_out = emotional.copy()

    df_out = df_out.sample(frac=1, random_state=seed).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "original_rows": original_rows,
        "rows_after_cleaning_before_neutral_sampling": len(df),
        "final_rows": len(df_out),
        "removed_unclear": remove_unclear,
        "neutral_only_available_after_cleaning": len(neutral_only),
        "neutral_only_kept": int((df_out[EMOTION_LABELS].sum(axis=1).eq(1) & df_out["neutral"].eq(1)).sum()),
        "neutral_ratio": neutral_ratio,
        "max_neutral_only": max_neutral_only,
        "min_chars": min_chars,
        "label_counts": df_out[EMOTION_LABELS].sum().astype(int).to_dict(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare an improved GoEmotions dataset for emotion experiments.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DATA_DIR / "cleaned_journal_data_improved_report.json")
    parser.add_argument(
        "--neutral-ratio",
        type=float,
        default=0.25,
        help="Keep neutral-only rows up to this fraction of emotional rows.",
    )
    parser.add_argument("--max-neutral-only", type=int, default=12000)
    parser.add_argument("--min-chars", type=int, default=4)
    parser.add_argument("--keep-unclear", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_dataset(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        neutral_ratio=args.neutral_ratio,
        max_neutral_only=args.max_neutral_only,
        min_chars=args.min_chars,
        remove_unclear=not args.keep_unclear,
        seed=args.seed,
    )
    print(f"Saved improved dataset: {report['output']}")
    print(f"Final rows: {report['final_rows']}")
    print(f"Neutral-only rows kept: {report['neutral_only_kept']}")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
