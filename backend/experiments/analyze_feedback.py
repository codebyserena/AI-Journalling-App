from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize human feedback on journal emotion predictions.")
    parser.add_argument("--database", default="sqlite:///./predictions.db")
    parser.add_argument("--output-dir", type=Path, default=Path("backend/experiments/results/human_feedback"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(args.database, connect_args={"check_same_thread": False})

    query = """
    SELECT
      f.id AS feedback_id,
      f.prediction_id,
      f.user_id,
      f.rating,
      f.corrected_emotion,
      f.note,
      f.created_at,
      p.text
    FROM prediction_feedback f
    JOIN predictions p ON p.id = f.prediction_id
    """
    try:
        feedback = pd.read_sql_query(query, engine)
    except Exception as exc:
        summary = {
            "feedback_count": 0,
            "message": "No feedback table is available yet. Start the backend once, then submit feedback in the app.",
            "error": str(exc),
        }
        (args.output_dir / "feedback_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return
    if feedback.empty:
        summary = {
            "feedback_count": 0,
            "message": "No feedback has been submitted yet.",
        }
        (args.output_dir / "feedback_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    rating_counts = feedback["rating"].value_counts().rename_axis("rating").reset_index(name="count")
    corrected_counts = (
        feedback.dropna(subset=["corrected_emotion"])["corrected_emotion"]
        .value_counts()
        .rename_axis("corrected_emotion")
        .reset_index(name="count")
    )
    acceptance_rate = float((feedback["rating"] == "right").mean())
    correction_rate = float((feedback["rating"] == "wrong").mean())

    feedback.to_csv(args.output_dir / "feedback_raw.csv", index=False)
    rating_counts.to_csv(args.output_dir / "feedback_rating_counts.csv", index=False)
    corrected_counts.to_csv(args.output_dir / "feedback_corrections.csv", index=False)
    summary = {
        "feedback_count": int(len(feedback)),
        "acceptance_rate": acceptance_rate,
        "correction_rate": correction_rate,
        "rating_counts": rating_counts.to_dict(orient="records"),
        "top_corrections": corrected_counts.head(10).to_dict(orient="records"),
    }
    (args.output_dir / "feedback_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
