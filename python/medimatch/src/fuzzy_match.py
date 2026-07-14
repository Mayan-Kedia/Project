"""
MediMatch — Fuzzy Condition Matching Module (Layer 2, Part A)
=============================================================
Uses TF-IDF cosine similarity to match user queries against known
condition strings. This handles misspellings and partial inputs
(e.g. "anxeity" → "Anxiety", "acne treatment" → "Acne").

The condition TF-IDF vectorizer is separate from the review-text
vectorizer used for sentiment classification (Section 6.2 of spec).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text


def build_condition_vectorizer(conditions: list[str], conditions_preprocessed: list[str]) -> tuple:
    """
    Fit a TF-IDF vectorizer on the list of unique condition strings.

    Uses character n-grams (3-5) in addition to word-level features
    to catch misspellings.

    Args:
        conditions: List of unique, ORIGINAL condition strings (for display).
        conditions_preprocessed: Same conditions, after preprocess_text().

    Returns:
        Tuple of (vectorizers_tuple, tfidf_matrix, conditions_list):
            - vectorizers_tuple: (word_vectorizer, char_vectorizer)
            - tfidf_matrix: combined sparse matrix
            - conditions_list: original condition strings (aligned with rows)
    """
    # Word-level vectorizer on pre-processed (stemmed) text
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
    )

    # Char-level vectorizer on lowercased ORIGINAL text (for typo catching)
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
    )

    # Fit both and combine features horizontally
    from scipy.sparse import hstack

    word_matrix = word_vectorizer.fit_transform(conditions_preprocessed)
    char_matrix = char_vectorizer.fit_transform([c.lower() for c in conditions])

    # Balance word features with character features (char n-grams catch typos)
    combined_matrix = hstack([word_matrix * 1.0, char_matrix * 2.0])

    print(f"  Condition vectorizer: {len(conditions)} conditions, "
          f"{combined_matrix.shape[1]} features "
          f"(word: {word_matrix.shape[1]}, char: {char_matrix.shape[1]})")

    return (word_vectorizer, char_vectorizer), combined_matrix, list(conditions)


def fuzzy_match_conditions(
    query: str,
    vectorizers: tuple,
    condition_matrix,
    condition_list: list[str],
    threshold: float = 0.3,
) -> list[tuple[str, float]]:
    """
    Find conditions similar to the user's query via cosine similarity.

    Args:
        query:            Raw user query string (e.g. "anxeity").
        vectorizers:      Tuple of (word_vectorizer, char_vectorizer).
        condition_matrix: Pre-computed TF-IDF matrix for all conditions.
        condition_list:   Ordered list of condition strings.
        threshold:        Minimum cosine similarity to include (default 0.3).

    Returns:
        List of (condition_string, similarity_score) tuples, sorted
        descending by similarity. Only includes matches >= threshold.
    """
    from scipy.sparse import hstack

    word_vec, char_vec = vectorizers

    # Transform the query through both vectorizers
    query_preprocessed = preprocess_text(query)
    query_word = word_vec.transform([query_preprocessed]) * 1.0
    query_char = char_vec.transform([query.lower()]) * 2.0
    query_combined = hstack([query_word, query_char])

    # Compute cosine similarity against all conditions
    similarities = cosine_similarity(query_combined, condition_matrix).flatten()

    # Filter by threshold and sort descending
    matches = []
    for i, sim in enumerate(similarities):
        if sim >= threshold:
            matches.append((condition_list[i], float(sim)))

    matches.sort(key=lambda x: x[1], reverse=True)

    return matches
