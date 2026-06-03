# ============================================================
# GoEmotions Dataset Cleaning Script
# Author: Dissertation Project
# Purpose: Clean Reddit-style emotion dataset for NLP training
# ============================================================

import pandas as pd
import numpy as np
import re
from ftfy import fix_text
from tqdm import tqdm

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

INPUT_FILE = "goemotions.csv"
OUTPUT_FILE = "goemotions_cleaned.csv"

# Emotion label columns (GoEmotions standard)
EMOTION_COLUMNS = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval',
    'disgust','embarrassment','excitement','fear','gratitude','grief',
    'joy','love','nervousness','optimism','pride','realization','relief',
    'remorse','sadness','surprise','neutral'
]

tqdm.pandas()

# ------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------

def fix_encoding(text):
    """Fix broken UTF-8 characters."""
    if isinstance(text, str):
        return fix_text(text)
    return ""

def clean_reddit_text(text):
    """Clean Reddit-specific noise and normalize text."""
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove Reddit quote markers
    text = re.sub(r"^>+", "", text)

    # Remove sarcasm or joke markers
    text = re.sub(r"\s*/(s|jk)\b", "", text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Replace placeholders like [NAME], [RELIGION]
    text = re.sub(r"\[.*?\]", "<ENTITY>", text)

    # Remove excessive punctuation
    text = re.sub(r"[^\w\s!?.,']", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def main():
    print("📥 Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Original shape: {df.shape}")

    # --------------------------------------------------------
    # Fix encoding issues
    # --------------------------------------------------------
    print("🧠 Fixing text encoding...")
    df["text"] = df["text"].progress_apply(fix_encoding)

    # --------------------------------------------------------
    # Clean Reddit artefacts
    # --------------------------------------------------------
    print("🧹 Cleaning Reddit-specific noise...")
    df["text"] = df["text"].progress_apply(clean_reddit_text)

    # --------------------------------------------------------
    # Keep only relevant columns
    # --------------------------------------------------------
    print("📦 Selecting relevant columns...")
    df = df[["text"] + EMOTION_COLUMNS]

    # --------------------------------------------------------
    # Remove empty or very short texts
    # --------------------------------------------------------
    print("✂ Removing empty or short texts...")
    df = df[df["text"].str.len() > 3]

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------
    print("🧽 Removing duplicate texts...")
    df = df.drop_duplicates(subset=["text"])

    # --------------------------------------------------------
    # Remove neutral-only samples (recommended)
    # --------------------------------------------------------
    print("⚖ Removing neutral-only samples...")
    neutral_only_mask = (df[EMOTION_COLUMNS].sum(axis=1) == 1) & (df["neutral"] == 1)
    print(f"Neutral-only samples removed: {neutral_only_mask.sum()}")
    df = df[~neutral_only_mask]

    # --------------------------------------------------------
    # Final sanity check
    # --------------------------------------------------------
    print("✅ Final dataset statistics:")
    print(df[EMOTION_COLUMNS].sum().sort_values(ascending=False))
    print(f"Final shape: {df.shape}")

    # --------------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Cleaned dataset saved as: {OUTPUT_FILE}")

# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
