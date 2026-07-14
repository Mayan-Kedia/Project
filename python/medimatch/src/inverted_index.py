"""
MediMatch — Inverted Index Module (Layer 1: IR Engine)
======================================================
Builds an inverted index mapping stemmed condition tokens to their
original condition strings and associated drug names.

Structure:
    {
        stemmed_token: {
            original_condition: set(drugName, ...)
        }
    }

This allows fast lookup: query "anxiety" → stem to "anxieti" →
find all conditions containing that stem → retrieve their drugs.
"""

from collections import defaultdict

import pandas as pd

from src.preprocessing import preprocess_text


def build_inverted_index(df: pd.DataFrame) -> dict:
    """
    Build the inverted index from the cleaned dataset.

    For each unique condition in the DataFrame:
        1. Preprocess (stem/tokenize) the condition string.
        2. For each stemmed token, add the original condition and its
           associated drugs to the index.

    Args:
        df: Cleaned DataFrame with `condition` and `drugName` columns.

    Returns:
        Inverted index dict:
            {stemmed_token: {original_condition: set(drugNames)}}
    """
    index = defaultdict(lambda: defaultdict(set))

    # Group by condition to get all drugs for each condition
    condition_drugs = df.groupby("condition")["drugName"].apply(set).to_dict()

    for condition, drugs in condition_drugs.items():
        # Preprocess the condition string
        processed = preprocess_text(condition)
        tokens = processed.split()

        # Add each token → condition → drugs mapping
        for token in tokens:
            if token:  # skip empty tokens
                index[token][condition].update(drugs)

    # Convert defaultdicts to regular dicts for pickling
    index = {token: dict(cond_map) for token, cond_map in index.items()}

    print(f"  Inverted index: {len(index)} unique tokens, "
          f"{len(condition_drugs)} conditions, "
          f"{sum(len(d) for d in condition_drugs.values())} drug-condition pairs")

    return index


def lookup(query: str, index: dict) -> dict:
    """
    Look up a query in the inverted index.

    Preprocesses the query, then for each stemmed token, retrieves all
    matching conditions and their associated drugs. Results are the
    union across all query tokens.

    Args:
        query: Raw user query string (e.g. "anxiety").
        index: The inverted index built by build_inverted_index().

    Returns:
        Dict of {condition: set(drugNames)} for all matching conditions.
    """
    processed = preprocess_text(query)
    tokens = processed.split()

    if not tokens:
        return {}

    # Collect matches across all query tokens (union)
    results = defaultdict(set)
    for token in tokens:
        if token in index:
            for condition, drugs in index[token].items():
                results[condition].update(drugs)

    return dict(results)
