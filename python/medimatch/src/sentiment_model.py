"""
MediMatch — Sentiment Classification Module (Layer 2, Part B)
==============================================================
Trains and evaluates binary sentiment classifiers on drug review text.

Two classifiers are compared:
    - Logistic Regression (typically stronger on this dataset)
    - Multinomial Naïve Bayes (faster, serves as a baseline)

Both use TF-IDF features from review text. The better-performing model
is selected and saved for use in the Streamlit app.

Sentiment labels are derived from ratings:
    rating >= 7  →  positive (1)
    rating <= 4  →  negative (0)
    rating 5–6   →  dropped (ambiguous)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.preprocessing import preprocess_text


def build_review_vectorizer(
    train_reviews_preprocessed: pd.Series,
    max_features: int = 50000,
) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on ALREADY-PREPROCESSED review text.

    IMPORTANT: pass in text that has already been run through
    preprocess_text(). This avoids the massive overhead of calling
    the stemmer inside sklearn's vectorizer loop.

    This is a SEPARATE vectorizer from the condition-matching one —
    different vocabulary, different purpose (spec Section 6.2).

    Args:
        train_reviews_preprocessed: Series of pre-processed review strings.
        max_features: Maximum vocabulary size (default 50,000).

    Returns:
        Fitted TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),       # unigrams + bigrams for "side effects" etc.
        min_df=3,                 # ignore very rare terms
        max_df=0.95,              # ignore terms in >95% of docs
        sublinear_tf=True,        # apply log normalization to TF
    )
    vectorizer.fit(train_reviews_preprocessed)
    print(f"  Review vectorizer: {len(vectorizer.vocabulary_)} features")
    return vectorizer


def train_classifiers(
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    """
    Train both Logistic Regression and Multinomial Naïve Bayes,
    evaluate each, and return a results dictionary.

    Args:
        X_train: TF-IDF sparse matrix for training reviews.
        y_train: Binary sentiment labels for training data.
        X_test:  TF-IDF sparse matrix for test reviews.
        y_test:  Binary sentiment labels for test data.

    Returns:
        Dict with keys 'logistic_regression' and 'naive_bayes', each
        mapping to a dict with 'model', 'accuracy', 'precision',
        'recall', 'f1', 'confusion_matrix', 'report'.
    """
    results = {}

    # --- Logistic Regression ---
    print("\n  Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    results["logistic_regression"] = _evaluate(lr, lr_preds, y_test, "Logistic Regression")

    # --- Multinomial Naïve Bayes ---
    print("\n  Training Multinomial Naïve Bayes...")
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)
    nb_preds = nb.predict(X_test)
    results["naive_bayes"] = _evaluate(nb, nb_preds, y_test, "Naïve Bayes")

    return results


def _evaluate(model, predictions, y_true, name: str) -> dict:
    """Compute and print evaluation metrics for a classifier."""
    acc = accuracy_score(y_true, predictions)
    prec = precision_score(y_true, predictions, zero_division=0)
    rec = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    cm = confusion_matrix(y_true, predictions)
    report = classification_report(y_true, predictions, target_names=["Negative", "Positive"])

    print(f"\n  === {name} Results ===")
    print(f"  Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    print(f"\n{report}")

    return {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }


def select_and_save_model(results: dict, vectorizer, models_dir: str) -> str:
    """
    Select the better-performing classifier and save it with joblib.

    Saves:
        - models/sentiment_model.joblib    (the chosen classifier)
        - models/review_vectorizer.joblib  (the fitted TF-IDF vectorizer)
        - models/model_comparison.txt      (comparison summary)

    Args:
        results:    Dict from train_classifiers().
        vectorizer: The fitted review TF-IDF vectorizer.
        models_dir: Path to the models/ directory.

    Returns:
        Name of the selected model ('logistic_regression' or 'naive_bayes').
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # Compare by accuracy (primary) then F1 (tiebreaker)
    lr = results["logistic_regression"]
    nb = results["naive_bayes"]

    if lr["accuracy"] >= nb["accuracy"]:
        best_name = "logistic_regression"
        best = lr
    else:
        best_name = "naive_bayes"
        best = nb

    print(f"\n  Selected model: {best_name} (accuracy: {best['accuracy']:.4f})")

    # Save the model and vectorizer
    joblib.dump(best["model"], str(models_path / "sentiment_model.joblib"))
    joblib.dump(vectorizer, str(models_path / "review_vectorizer.joblib"))
    print(f"  Saved: {models_path / 'sentiment_model.joblib'}")
    print(f"  Saved: {models_path / 'review_vectorizer.joblib'}")

    # Write a human-readable comparison file
    comparison = (
        "MediMatch — Sentiment Model Comparison\n"
        "=" * 45 + "\n\n"
        f"{'Metric':<15} {'Logistic Regression':>20} {'Naïve Bayes':>20}\n"
        f"{'-' * 55}\n"
        f"{'Accuracy':<15} {lr['accuracy']:>20.4f} {nb['accuracy']:>20.4f}\n"
        f"{'Precision':<15} {lr['precision']:>20.4f} {nb['precision']:>20.4f}\n"
        f"{'Recall':<15} {lr['recall']:>20.4f} {nb['recall']:>20.4f}\n"
        f"{'F1-Score':<15} {lr['f1']:>20.4f} {nb['f1']:>20.4f}\n"
        f"\nSelected: {best_name}\n"
    )
    (models_path / "model_comparison.txt").write_text(comparison, encoding="utf-8")

    return best_name
