"""
MediMatch — Ranking Module (Layer 3: Score Combiner)
=====================================================
Combines keyword relevance (from fuzzy matching) with sentiment scores
(from the ML classifier) into a single Final Score per drug.

Formula: Final Score = 0.4 × Keyword Relevance + 0.6 × Sentiment Score

The sentiment score for each drug is pre-computed during training as the
proportion of reviews classified positive, avoiding real-time inference.
"""

import pandas as pd
import numpy as np


# Weights for the Final Score formula (spec Section 6.3)
WEIGHT_KEYWORD = 0.4
WEIGHT_SENTIMENT = 0.6


def precompute_sentiment_scores(
    df: pd.DataFrame,
    model,
    vectorizer,
) -> pd.DataFrame:
    """
    Pre-compute sentiment scores for every (drug, condition) pair.

    For each pair, predicts sentiment on all its reviews, then computes:
        sentiment_score = positive_count / total_count

    This is done once at training time so the Streamlit app doesn't
    need to run ML inference at query time.

    Args:
        df:         Cleaned DataFrame with columns: drugName, condition, review.
        model:      Trained sentiment classifier.
        vectorizer: Fitted review TF-IDF vectorizer.

    Returns:
        DataFrame with columns:
            drugName, condition, sentiment_score, review_count, positive_count
    """
    print("[INFO] Pre-computing sentiment scores for all drug-condition pairs...")

    # Use pre-processed text if available, otherwise raw text
    review_col = "review_processed" if "review_processed" in df.columns else "review"

    # Predict sentiment for ALL reviews in one batch (much faster)
    print("  Vectorizing all reviews...")
    X_all = vectorizer.transform(df[review_col])
    print("  Predicting sentiment...")
    df = df.copy()
    df["predicted_sentiment"] = model.predict(X_all)

    # Aggregate by (drugName, condition)
    print("  Aggregating by drug-condition pair...")
    agg = df.groupby(["drugName", "condition"]).agg(
        review_count=("predicted_sentiment", "count"),
        positive_count=("predicted_sentiment", "sum"),
    ).reset_index()

    agg["sentiment_score"] = agg["positive_count"] / agg["review_count"]

    print(f"  Computed scores for {len(agg):,} drug-condition pairs")
    return agg


def compute_final_score(keyword_relevance: float, sentiment_score: float) -> float:
    """
    Compute the Final Score for a drug.

    Formula: 0.4 × keyword_relevance + 0.6 × sentiment_score

    Args:
        keyword_relevance: Cosine similarity score from fuzzy matching (0–1).
        sentiment_score:   Proportion of positive reviews (0–1).

    Returns:
        Final Score in range [0, 1].
    """
    return WEIGHT_KEYWORD * keyword_relevance + WEIGHT_SENTIMENT * sentiment_score


def rank_drugs(
    query: str,
    inverted_index: dict,
    fuzzy_vectorizers: tuple,
    condition_matrix,
    condition_list: list[str],
    sentiment_scores_df: pd.DataFrame,
    fuzzy_threshold: float = 0.15,
    sentiment_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Full ranking pipeline: query → matched conditions → candidate drugs →
    Final Score → sorted, filtered results.

    Combines Layer 1 (inverted index), Layer 2 (fuzzy matching), and
    Layer 3 (sentiment scores) into a single ranked output.

    Args:
        query:               Raw user query string.
        inverted_index:      Layer 1 inverted index.
        fuzzy_vectorizers:   Tuple of (word_vec, char_vec) from fuzzy_match.
        condition_matrix:    Pre-computed condition TF-IDF matrix.
        condition_list:      Ordered list of condition strings.
        sentiment_scores_df: Pre-computed sentiment scores DataFrame.
        fuzzy_threshold:     Min cosine similarity for condition matching.
        sentiment_threshold: Min sentiment score to include in results.

    Returns:
        DataFrame with columns: drugName, condition, finalScore,
        sentimentScore, keywordRelevance, reviewCount, positiveCount
        Sorted descending by finalScore.
    """
    from src.inverted_index import lookup
    from src.fuzzy_match import fuzzy_match_conditions

    # --- Step 1: Inverted index lookup (exact token match) ---
    exact_matches = lookup(query, inverted_index)

    # --- Step 2: Fuzzy match (TF-IDF cosine similarity) ---
    fuzzy_matches = fuzzy_match_conditions(
        query, fuzzy_vectorizers, condition_matrix, condition_list, fuzzy_threshold
    )

    # Build a dict of condition → keyword_relevance score
    # Fuzzy matches provide the similarity scores
    condition_relevance = {}
    for condition, sim_score in fuzzy_matches:
        condition_relevance[condition] = sim_score

    # Exact matches get a boost — set to max of fuzzy score and 1.0
    for condition in exact_matches:
        if condition in condition_relevance:
            condition_relevance[condition] = max(condition_relevance[condition], 1.0)
        else:
            condition_relevance[condition] = 1.0

    if not condition_relevance:
        return pd.DataFrame(columns=[
            "drugName", "condition", "finalScore",
            "sentimentScore", "keywordRelevance", "reviewCount", "positiveCount"
        ])

    # --- Step 3: Get sentiment scores for matched conditions ---
    matched_conditions = set(condition_relevance.keys())
    relevant_drugs = sentiment_scores_df[
        sentiment_scores_df["condition"].isin(matched_conditions)
    ].copy()

    if relevant_drugs.empty:
        return pd.DataFrame(columns=[
            "drugName", "condition", "finalScore",
            "sentimentScore", "keywordRelevance", "reviewCount", "positiveCount"
        ])

    # --- Step 4: Compute Final Score ---
    relevant_drugs["keywordRelevance"] = relevant_drugs["condition"].map(condition_relevance)
    relevant_drugs["finalScore"] = relevant_drugs.apply(
        lambda row: compute_final_score(row["keywordRelevance"], row["sentiment_score"]),
        axis=1,
    )

    # Rename for output clarity
    relevant_drugs = relevant_drugs.rename(columns={
        "sentiment_score": "sentimentScore",
        "review_count": "reviewCount",
        "positive_count": "positiveCount",
    })

    # --- Step 5: Filter by sentiment threshold and sort ---
    if sentiment_threshold > 0:
        relevant_drugs = relevant_drugs[
            relevant_drugs["sentimentScore"] >= sentiment_threshold
        ]

    relevant_drugs = relevant_drugs.sort_values(
        ["finalScore", "reviewCount"], ascending=[False, False]
    ).reset_index(drop=True)

    return relevant_drugs[[
        "drugName", "condition", "finalScore",
        "sentimentScore", "keywordRelevance", "reviewCount", "positiveCount"
    ]]
