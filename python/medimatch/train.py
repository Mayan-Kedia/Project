"""
MediMatch — End-to-End Training Script
=======================================
Runs the complete pipeline from raw data to saved model artifacts:

    1. Load & clean data
    2. Build inverted index (Layer 1)
    3. Fit condition TF-IDF vectorizer (Layer 2, Part A)
    4. Derive sentiment labels, train classifiers (Layer 2, Part B)
    5. Pre-compute sentiment scores (Layer 3)
    6. Save all artifacts

Usage:
    python train.py

All artifacts are saved under models/ and data/processed/ so the
Streamlit app can load them without retraining.
"""

import sys
import time
import shutil
from pathlib import Path

import pandas as pd
import joblib

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import (
    load_and_clean_data,
    derive_sentiment_labels,
    preprocess_text,
    save_artifact,
)
from src.inverted_index import build_inverted_index, lookup
from src.fuzzy_match import build_condition_vectorizer, fuzzy_match_conditions
from src.sentiment_model import (
    build_review_vectorizer,
    train_classifiers,
    select_and_save_model,
)
from src.ranking import precompute_sentiment_scores, compute_final_score


def main():
    start_time = time.time()
    project_root = Path(__file__).parent

    # Paths
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"

    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # Step 1: Copy raw data files into data/raw/ (if not already there)
    # ===================================================================
    print("=" * 60)
    print("STEP 1: Data Setup & Cleaning")
    print("=" * 60)

    parent_dir = project_root.parent  # c:\Users\mayan\Programming\python
    for fname in ["drugsComTrain_raw.csv", "drugsComTest_raw.csv"]:
        src = parent_dir / fname
        dst = raw_dir / fname
        if not dst.exists() and src.exists():
            shutil.copy2(str(src), str(dst))
            print(f"  Copied {fname} -> data/raw/")
        elif dst.exists():
            print(f"  Found {fname} in data/raw/")
        else:
            print(f"  [ERROR] Cannot find {fname} in {parent_dir}")
            print("  Please place the CSV files in data/raw/ and re-run.")
            sys.exit(1)

    # Load and clean
    train_df, test_df = load_and_clean_data(
        str(raw_dir / "drugsComTrain_raw.csv"),
        str(raw_dir / "drugsComTest_raw.csv"),
    )

    # Combine for building the index (we use all data for IR, not just train)
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  Combined dataset: {len(all_df):,} rows")

    # Save cleaned data
    save_artifact(train_df, str(processed_dir / "train_cleaned.pkl"))
    save_artifact(test_df, str(processed_dir / "test_cleaned.pkl"))
    save_artifact(all_df, str(processed_dir / "all_cleaned.pkl"))

    # ===================================================================
    # Step 2: Build Inverted Index (Layer 1)
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Building Inverted Index (Layer 1)")
    print("=" * 60)

    inv_index = build_inverted_index(all_df)
    save_artifact(inv_index, str(processed_dir / "inverted_index.pkl"))

    # Verification: test lookups
    print("\n  --- Verification Lookups ---")
    for test_query in ["anxiety", "acne", "depression", "birth control"]:
        results = lookup(test_query, inv_index)
        n_drugs = sum(len(drugs) for drugs in results.values())
        print(f"  Query '{test_query}': {len(results)} conditions, {n_drugs} drugs")
        for cond, drugs in list(results.items())[:2]:
            print(f"    -> {cond}: {sorted(list(drugs))[:5]}...")

    # ===================================================================
    # Step 3: Fit Condition TF-IDF Vectorizer (Layer 2, Part A)
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Building Condition TF-IDF (Fuzzy Matcher)")
    print("=" * 60)

    unique_conditions = sorted(all_df["condition"].unique().tolist())
    print(f"  Unique conditions: {len(unique_conditions)}")

    # Pre-process conditions for the word vectorizer
    conditions_preprocessed = [preprocess_text(c) for c in unique_conditions]

    vectorizers, condition_matrix, condition_list = build_condition_vectorizer(
        unique_conditions, conditions_preprocessed
    )

    # Save fuzzy matching artifacts
    joblib.dump(vectorizers, str(models_dir / "condition_vectorizers.joblib"))
    joblib.dump(condition_matrix, str(models_dir / "condition_matrix.joblib"))
    save_artifact(condition_list, str(processed_dir / "condition_list.pkl"))

    # Verification: test fuzzy matching
    print("\n  --- Fuzzy Match Verification ---")
    for test_query in ["anxiety", "anxeity", "acne", "depresion", "birth contol"]:
        matches = fuzzy_match_conditions(
            test_query, vectorizers, condition_matrix, condition_list, threshold=0.15
        )
        print(f"  Query '{test_query}':")
        for cond, score in matches[:3]:
            print(f"    -> {cond} (similarity: {score:.4f})")

    # ===================================================================
    # Step 4: Train Sentiment Classifiers (Layer 2, Part B)
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Training Sentiment Classifiers")
    print("=" * 60)

    # Derive sentiment labels
    print("\n[INFO] Deriving sentiment labels...")
    train_sentiment = derive_sentiment_labels(train_df)
    test_sentiment = derive_sentiment_labels(test_df)

    # Pre-process ALL review text upfront (this is the slow part, but
    # doing it once is much faster than doing it per-doc inside sklearn)
    print("\n[INFO] Pre-processing review text (this takes a few minutes)...")
    sys.stdout.flush()
    train_sentiment = train_sentiment.copy()
    test_sentiment = test_sentiment.copy()

    # Process training reviews with progress indicator
    n_total = len(train_sentiment)
    processed_reviews = []
    for i, review in enumerate(train_sentiment["review"]):
        processed_reviews.append(preprocess_text(review))
        if (i + 1) % 25000 == 0:
            print(f"  Processed {i + 1:,}/{n_total:,} training reviews...")
            sys.stdout.flush()
    train_sentiment["review_processed"] = processed_reviews
    print(f"  Processed {n_total:,} training reviews (done)")
    sys.stdout.flush()

    # Process test reviews
    n_test = len(test_sentiment)
    processed_test = []
    for i, review in enumerate(test_sentiment["review"]):
        processed_test.append(preprocess_text(review))
        if (i + 1) % 25000 == 0:
            print(f"  Processed {i + 1:,}/{n_test:,} test reviews...")
            sys.stdout.flush()
    test_sentiment["review_processed"] = processed_test
    print(f"  Processed {n_test:,} test reviews (done)")
    sys.stdout.flush()

    # Build review TF-IDF vectorizer (fit on pre-processed training data only)
    print("\n[INFO] Building review TF-IDF vectorizer...")
    review_vectorizer = build_review_vectorizer(train_sentiment["review_processed"])

    # Transform train and test reviews
    print("  Transforming training reviews...")
    X_train = review_vectorizer.transform(train_sentiment["review_processed"])
    y_train = train_sentiment["sentiment"].values
    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]:,} features")

    print("  Transforming test reviews...")
    X_test = review_vectorizer.transform(test_sentiment["review_processed"])
    y_test = test_sentiment["sentiment"].values
    print(f"  Test:  {X_test.shape[0]:,} samples, {X_test.shape[1]:,} features")

    # Train and evaluate both classifiers
    results = train_classifiers(X_train, y_train, X_test, y_test)

    # Select and save the best model
    best_model_name = select_and_save_model(results, review_vectorizer, str(models_dir))

    # ===================================================================
    # Step 5: Pre-compute Sentiment Scores (Layer 3)
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Pre-computing Sentiment Scores (Layer 3)")
    print("=" * 60)

    # Load the saved model for scoring
    best_model = results[best_model_name]["model"]

    # Pre-process ALL reviews for scoring (not just sentiment-labeled ones)
    print("[INFO] Pre-processing all reviews for scoring...")
    sys.stdout.flush()
    all_df = all_df.copy()
    n_all = len(all_df)
    processed_all = []
    for i, review in enumerate(all_df["review"]):
        processed_all.append(preprocess_text(review))
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i + 1:,}/{n_all:,} reviews...")
            sys.stdout.flush()
    all_df["review_processed"] = processed_all
    print(f"  Processed {n_all:,} reviews (done)")
    sys.stdout.flush()

    # Score ALL reviews (not just train/test sentiment subset) so we
    # have coverage for every drug-condition pair
    sentiment_scores_df = precompute_sentiment_scores(all_df, best_model, review_vectorizer)
    save_artifact(sentiment_scores_df, str(processed_dir / "sentiment_scores.pkl"))

    # ===================================================================
    # Step 6: Verification — Final Score Worked Example
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Verification - Worked Example & Sample Queries")
    print("=" * 60)

    # Verify the Final Score formula with the spec's worked example
    print("\n  --- Worked Example from Spec ---")
    print("  Query: 'anxiety', cosine similarity = 0.92")
    print("  Drug X: 180/200 positive -> sentiment = 0.90")
    expected_score = 0.4 * 0.92 + 0.6 * 0.90
    computed_score = compute_final_score(0.92, 0.90)
    print(f"  Expected Final Score: {expected_score:.3f}")
    print(f"  Computed Final Score: {computed_score:.3f}")
    assert abs(computed_score - 0.908) < 0.001, "Final Score formula mismatch!"
    print("  [OK] Formula verified!")

    # Sample full-pipeline queries
    print("\n  --- Full Pipeline Queries ---")
    from src.ranking import rank_drugs

    for test_query in ["anxiety", "acne", "anxeity"]:
        print(f"\n  Query: '{test_query}'")
        ranked = rank_drugs(
            test_query,
            inv_index,
            vectorizers,
            condition_matrix,
            condition_list,
            sentiment_scores_df,
            fuzzy_threshold=0.15,
            sentiment_threshold=0.0,
        )
        if ranked.empty:
            print("    No results found.")
        else:
            print(f"    Found {len(ranked)} drugs. Top 5:")
            for _, row in ranked.head(5).iterrows():
                print(f"      {row['drugName']:25s} | Final: {row['finalScore']:.3f} "
                      f"| Sentiment: {row['sentimentScore']:.3f} "
                      f"| Relevance: {row['keywordRelevance']:.3f} "
                      f"| Reviews: {row['reviewCount']}")

    # ===================================================================
    # Summary
    # ===================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Time elapsed: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
    print(f"\n  Artifacts saved:")
    print(f"    models/sentiment_model.joblib       - Trained {best_model_name}")
    print(f"    models/review_vectorizer.joblib      - Review TF-IDF vectorizer")
    print(f"    models/condition_vectorizers.joblib   - Condition TF-IDF vectorizers")
    print(f"    models/condition_matrix.joblib        - Condition TF-IDF matrix")
    print(f"    data/processed/inverted_index.pkl     - Inverted index")
    print(f"    data/processed/sentiment_scores.pkl   - Pre-computed scores")
    print(f"    data/processed/condition_list.pkl     - Condition list")
    print(f"\n  To launch the app:")
    print(f"    streamlit run app.py")


if __name__ == "__main__":
    main()
