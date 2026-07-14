# MediMatch — Condition-Based Drug Search & Sentiment-Ranked Recommendation System

**Using Information Retrieval and Machine Learning**

> B.Tech 1st Year — IR & ML Project  
> Author: Mayan Kedia (Enrollment No. 2501010050)  
> Jaypee Institute of Information Technology, Noida — Sector 62

---

## Overview

MediMatch is a condition-based drug search engine combining classical Information Retrieval with a Machine Learning sentiment classifier. A user types a medical condition (e.g. "anxiety", "acne"); the system returns a ranked list of medications scored by aggregated real-patient sentiment drawn from ~215,000 reviews.

**⚠️ Disclaimer:** This system is developed strictly for educational and informational purposes. It does not constitute medical advice. Always consult a qualified healthcare professional before making any medication-related decisions.

---

## Architecture

MediMatch uses a four-layer architecture:

| Layer | Component | Technique |
|-------|-----------|-----------|
| **Layer 1 — IR Engine** | Inverted index | Tokenization → stopword removal → Porter stemming → inverted index |
| **Layer 2 — ML Engine** | TF-IDF + Classifier | TF-IDF cosine similarity (fuzzy match) · Logistic Regression / Naïve Bayes (sentiment) |
| **Layer 3 — Ranking** | Score combiner | `Final Score = 0.4 × Keyword Relevance + 0.6 × Sentiment Score` |
| **Layer 4 — UI** | Streamlit app | Search bar · sentiment slider · ranked result cards |

**Data pipeline:** `CSV Dataset → Preprocess → Build Index → Train Model → Rank & Serve`

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the dataset

Download the UCI ML Drug Review dataset (ID 462) and place the CSV files in `data/raw/`:

```
medimatch/data/raw/drugsComTrain_raw.csv
medimatch/data/raw/drugsComTest_raw.csv
```

Alternatively, if the CSV files are in the parent directory (`../drugsComTrain_raw.csv`), the training script will auto-copy them.

### 3. Train the models

```bash
python train.py
```

This runs the full pipeline (~5–10 minutes on a laptop):
- Cleans the data (drops nulls, corrupted conditions, HTML-decodes reviews)
- Builds the inverted index
- Fits condition and review TF-IDF vectorizers
- Trains and evaluates Logistic Regression and Naïve Bayes
- Pre-computes sentiment scores for all drug-condition pairs
- Saves all artifacts to `models/` and `data/processed/`

### 4. Launch the app

```bash
streamlit run app.py
```

---

## Project Structure

```
medimatch/
├── data/
│   ├── raw/                     # Raw CSV files (drugsComTrain_raw.csv, drugsComTest_raw.csv)
│   └── processed/               # Cleaned DataFrames, inverted index, pre-computed scores
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Layer 1: text cleaning, tokenization, stemming
│   ├── inverted_index.py        # Layer 1: inverted index build & lookup
│   ├── fuzzy_match.py           # Layer 2: condition TF-IDF + cosine similarity
│   ├── sentiment_model.py       # Layer 2: classifier training & evaluation
│   └── ranking.py               # Layer 3: Final Score formula & full-pipeline ranking
├── models/                      # Saved vectorizers & trained classifier (joblib)
├── train.py                     # End-to-end training script
├── app.py                       # Streamlit web app (Layer 4)
├── requirements.txt
├── README.md
└── DEMO_SCRIPT.md
```

---

## Design Decisions & Assumptions

| Decision | Rationale |
|----------|-----------|
| **Sentiment threshold:** rating ≥ 7 → positive, ≤ 4 → negative, 5–6 dropped | Standard for this dataset; sharpens the decision boundary and is documented practice for the UCI Drug Review corpus |
| **Porter Stemmer** (not Snowball) | Simpler, sufficient for condition names; avoids over-stemming |
| **TF-IDF `max_features=50000`, `ngram_range=(1,2)`** | Bigrams capture multi-word phrases like "side effects" and "birth control"; 50K features balances coverage with laptop memory |
| **Character n-grams (3–5) in fuzzy matcher** | Combined with word-level TF-IDF for typo tolerance (e.g. "anxeity" → "Anxiety") |
| **Pre-computed sentiment scores** | Avoids running ML inference at query time; the Streamlit app loads pre-computed scores instantly |
| **Raw proportion for sentiment score** (no Wilson interval) | Simpler for a 1st-year project; a drug with 2 reviews could rank oddly, but this is documented as a known limitation |
| **Fuzzy match threshold: 0.3** | Empirically chosen starting point; catches typos without too many false positives |
| **CSV format** (not TSV) | The dataset files on disk are comma-separated despite the spec mentioning TSV |

---

## Dataset

**Source:** UCI Machine Learning Repository — *"Drug Reviews (Drugs.com)"*, dataset ID 462  
**Authors:** Kallumadi & Grer (2018)  
**License:** CC BY 4.0  
**Size:** 215,063 reviews (161,297 train + 53,766 test)

---

## How to Retrain

If you modify any source module or want to experiment with different parameters:

```bash
# Full retrain from raw data
python train.py
```

Key parameters to tune:
- **Sentiment threshold** in `src/preprocessing.py` → `derive_sentiment_labels()`
- **TF-IDF settings** in `src/sentiment_model.py` → `build_review_vectorizer()`
- **Fuzzy match threshold** in `src/ranking.py` → `rank_drugs()` (default 0.3)
- **Final Score weights** in `src/ranking.py` → `WEIGHT_KEYWORD` and `WEIGHT_SENTIMENT`

---

## Tech Stack

- Python 3.x
- Pandas · NumPy
- Scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `MultinomialNB`, `cosine_similarity`)
- NLTK (stopwords, Porter stemmer)
- Streamlit
- joblib (model persistence)

---

## References

- Manning, Raghavan & Schütze — *Introduction to Information Retrieval* (2008)
- Salton & Buckley — *Term-Weighting Approaches in Automatic Text Retrieval* (1988)
- Pang, Lee & Vaithyanathan — *Thumbs Up? Sentiment Classification Using Machine Learning Techniques* (2002)
- Pedregosa et al. — *Scikit-learn: Machine Learning in Python* (2011)
