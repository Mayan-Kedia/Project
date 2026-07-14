"""
MediMatch — Preprocessing Module (Layer 1 foundation)
=====================================================
Handles all text cleaning, tokenization, stopword removal, and stemming.
Also loads/cleans the raw dataset and derives sentiment labels from ratings.

Pipeline: raw text → lowercase → HTML-decode → strip punctuation →
          tokenize → remove stopwords → Porter stem → rejoin
"""

import re
import html
import pickle
from pathlib import Path

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------------------------------------------------------------------
# Ensure NLTK data is available (downloads only if missing)
# ---------------------------------------------------------------------------
nltk.download("stopwords", quiet=True)

# Initialize stemmer and stopword set once (module-level for reuse)
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Text preprocessing pipeline
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single text string.

    Steps:
        1. Lowercase
        2. HTML-decode entities (e.g. &#039; → ')
        3. Strip punctuation and digits
        4. Tokenize on whitespace
        5. Remove English stopwords
        6. Apply Porter stemming
        7. Rejoin into a single string

    Args:
        text: Raw input string (condition name or review text).

    Returns:
        Cleaned, stemmed string with tokens separated by spaces.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1 — lowercase
    text = text.lower()

    # Step 2 — HTML-decode (handles &#039;, &amp;, etc.)
    text = html.unescape(text)

    # Step 3 — remove anything that isn't a letter or space
    text = re.sub(r"[^a-z\s]", "", text)

    # Step 4 — tokenize on whitespace
    tokens = text.split()

    # Step 5 & 6 — remove stopwords and stem
    tokens = [_stemmer.stem(tok) for tok in tokens if tok not in _stop_words]

    # Step 7 — rejoin
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Dataset loading and cleaning
# ---------------------------------------------------------------------------
def load_and_clean_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw Drug Review CSVs and apply all cleaning steps.

    Cleaning steps:
        - Drop rows with null `condition` or empty `review`.
        - Remove corrupted condition rows (scraping artifact containing '</span>').
        - HTML-decode the `review` column.

    Args:
        train_path: Path to drugsComTrain_raw.csv
        test_path:  Path to drugsComTest_raw.csv

    Returns:
        Tuple of (train_df, test_df), both cleaned.
    """
    print("[INFO] Loading raw datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"  Raw train: {len(train_df):,} rows | Raw test: {len(test_df):,} rows")

    for label, df in [("train", train_df), ("test", test_df)]:
        # Track how many rows we drop at each step
        n_before = len(df)

        # Drop null conditions
        df.dropna(subset=["condition"], inplace=True)

        # Drop corrupted conditions (scraping artifact)
        corrupted_mask = df["condition"].str.contains(r"</span>", na=False)
        df.drop(df[corrupted_mask].index, inplace=True)

        # Drop rows with empty review text
        df.dropna(subset=["review"], inplace=True)
        df = df[df["review"].str.strip() != ""]

        # HTML-decode the review text
        df["review"] = df["review"].apply(lambda r: html.unescape(str(r)))

        n_after = len(df)
        print(f"  Cleaned {label}: {n_after:,} rows (dropped {n_before - n_after:,})")

        if label == "train":
            train_df = df.reset_index(drop=True)
        else:
            test_df = df.reset_index(drop=True)

    return train_df, test_df


# ---------------------------------------------------------------------------
# Sentiment label derivation
# ---------------------------------------------------------------------------
def derive_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive binary sentiment labels from the 1–10 `rating` column.

    Mapping (standard for this dataset):
        - rating >= 7  →  sentiment = 1 (positive)
        - rating <= 4  →  sentiment = 0 (negative)
        - rating 5–6   →  dropped (ambiguous / neutral)

    This sharpens the decision boundary and is the documented design
    decision per the project spec (Section 6.2).

    Args:
        df: DataFrame with a `rating` column.

    Returns:
        Filtered DataFrame with a new `sentiment` column (0 or 1).
    """
    df = df.copy()
    df = df[df["rating"].notna()]

    # Apply the labeling thresholds
    positive = df["rating"] >= 7
    negative = df["rating"] <= 4
    keep_mask = positive | negative

    df = df[keep_mask].copy()
    df["sentiment"] = (df["rating"] >= 7).astype(int)

    n_pos = df["sentiment"].sum()
    n_neg = len(df) - n_pos
    print(f"  Sentiment labels: {n_pos:,} positive, {n_neg:,} negative "
          f"({n_pos / len(df) * 100:.1f}% / {n_neg / len(df) * 100:.1f}%)")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_artifact(obj, path: str) -> None:
    """Save any Python object as a pickle file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved: {path}")


def load_artifact(path: str):
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
