"""
MediMatch — Streamlit Web Application (Layer 4: UI)
====================================================
Interactive condition-based drug search with sentiment-ranked results.

Features:
    - Search bar for medical conditions
    - Sentiment threshold slider to filter low-scoring drugs
    - Ranked result cards with Final Score breakdown
    - Medical disclaimer on every page (non-negotiable)
    - Sidebar with system info and model metrics

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import load_artifact
from src.ranking import rank_drugs


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediMatch — Drug Search & Sentiment Ranking",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---- Import Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---- Global styles ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Main header ---- */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1rem;
        color: #6b7280;
        font-weight: 400;
        margin-top: 0;
    }

    /* ---- Medical disclaimer banner ---- */
    .disclaimer-banner {
        background: linear-gradient(90deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.8rem 0 1.5rem 0;
        font-size: 0.82rem;
        color: #92400e;
        line-height: 1.5;
        text-align: center;
    }
    .disclaimer-banner strong {
        color: #b45309;
    }

    /* ---- Drug result card ---- */
    .drug-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .drug-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }

    .drug-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.4rem;
    }
    .drug-condition {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.8rem;
    }

    /* ---- Score metrics row ---- */
    .score-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 0.8rem;
    }
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-final {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    .badge-sentiment {
        background: #d1fae5;
        color: #065f46;
    }
    .badge-relevance {
        background: #dbeafe;
        color: #1e40af;
    }
    .badge-reviews {
        background: #f3f4f6;
        color: #4b5563;
    }

    /* ---- Score bar ---- */
    .score-bar-container {
        width: 100%;
        height: 6px;
        background: #e5e7eb;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 0.3rem;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.4s ease;
    }

    /* ---- Review snippet ---- */
    .review-snippet {
        font-size: 0.82rem;
        color: #6b7280;
        font-style: italic;
        border-left: 3px solid #e5e7eb;
        padding-left: 0.8rem;
        margin-top: 0.6rem;
        line-height: 1.5;
    }

    /* ---- Search summary ---- */
    .search-summary {
        font-size: 0.95rem;
        color: #4b5563;
        margin-bottom: 1rem;
        padding: 0.6rem 1rem;
        background: #f9fafb;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }

    /* ---- Sidebar styling ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    /* ---- Empty state ---- */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #9ca3af;
    }
    .empty-state .emoji {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .empty-state h3 {
        color: #6b7280;
        font-weight: 600;
    }

    /* ---- Hide Streamlit defaults for cleaner look ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load cached artifacts (runs only once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading MediMatch models...")
def load_artifacts():
    """Load all pre-trained model artifacts into memory."""
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"

    artifacts = {
        "inverted_index": load_artifact(str(processed_dir / "inverted_index.pkl")),
        "condition_vectorizers": joblib.load(str(models_dir / "condition_vectorizers.joblib")),
        "condition_matrix": joblib.load(str(models_dir / "condition_matrix.joblib")),
        "condition_list": load_artifact(str(processed_dir / "condition_list.pkl")),
        "sentiment_scores_df": load_artifact(str(processed_dir / "sentiment_scores.pkl")),
    }

    # Load model comparison if available
    comparison_path = models_dir / "model_comparison.txt"
    if comparison_path.exists():
        artifacts["model_comparison"] = comparison_path.read_text(encoding="utf-8")
    else:
        artifacts["model_comparison"] = "No comparison file found."

    # Load a sample of reviews for snippets
    all_data_path = processed_dir / "all_cleaned.pkl"
    if all_data_path.exists():
        artifacts["all_df"] = load_artifact(str(all_data_path))
    else:
        artifacts["all_df"] = None

    return artifacts


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_review_snippet(all_df, drug_name: str, condition: str) -> str:
    """Get a representative review snippet for a drug-condition pair."""
    if all_df is None:
        return ""

    mask = (all_df["drugName"] == drug_name) & (all_df["condition"] == condition)
    subset = all_df[mask]

    if subset.empty:
        return ""

    # Pick the highest-rated review as the representative snippet
    best_review = subset.sort_values("rating", ascending=False).iloc[0]["review"]

    # Truncate to ~200 chars for display
    if len(best_review) > 200:
        best_review = best_review[:197] + "..."

    # Clean up any stray quotes
    best_review = best_review.strip('"').strip("'")
    return best_review


def get_score_color(score: float) -> str:
    """Return a gradient color based on score (0–1)."""
    if score >= 0.8:
        return "linear-gradient(90deg, #10b981, #34d399)"
    elif score >= 0.6:
        return "linear-gradient(90deg, #3b82f6, #60a5fa)"
    elif score >= 0.4:
        return "linear-gradient(90deg, #f59e0b, #fbbf24)"
    else:
        return "linear-gradient(90deg, #ef4444, #f87171)"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(artifacts):
    """Render the sidebar with system info and model metrics."""
    with st.sidebar:
        st.markdown("### 💊 MediMatch")
        st.markdown("**Condition-Based Drug Search &  \nSentiment-Ranked Recommendations**")

        st.divider()

        st.markdown("### ℹ️ How It Works")
        st.markdown("""
        1. **Type a condition** (e.g. "anxiety", "acne")
        2. The **IR Engine** finds matching conditions using an inverted index
        3. **Fuzzy matching** catches typos via TF-IDF cosine similarity
        4. Drugs are **ranked** by a blend of keyword relevance and patient sentiment
        5. Use the **slider** to filter by minimum sentiment score
        """)

        st.divider()

        st.markdown("### 📊 Model Performance")
        st.code(artifacts["model_comparison"], language=None)

        st.divider()

        st.markdown("### 🔢 Ranking Formula")
        st.latex(r"\text{Final Score} = 0.4 \times R_k + 0.6 \times S_s")
        st.caption("R_k = Keyword Relevance, S_s = Sentiment Score")

        st.divider()

        st.markdown("### 📚 Dataset")
        st.caption(
            "UCI ML Repository — Drug Reviews (Drugs.com)  \n"
            "Kallumadi & Grer (2018), CC BY 4.0  \n"
            "~215,000 patient reviews"
        )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
def main():
    # Load all artifacts
    artifacts = load_artifacts()

    # Render sidebar
    render_sidebar(artifacts)

    # ---- Header ----
    st.markdown("""
    <div class="main-header">
        <h1>💊 MediMatch</h1>
        <p>Condition-Based Drug Search & Sentiment-Ranked Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Medical Disclaimer (ALWAYS visible — non-negotiable) ----
    st.markdown("""
    <div class="disclaimer-banner">
        <strong>⚠️ Medical Disclaimer:</strong> This system is developed strictly for
        <strong>educational and informational purposes</strong>. It does not constitute
        medical advice. Always consult a <strong>qualified healthcare professional</strong>
        before making any medication-related decisions.
    </div>
    """, unsafe_allow_html=True)

    # ---- Search Bar ----
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "🔍 Search for a medical condition",
            placeholder="e.g. anxiety, acne, depression, insomnia...",
            key="search_input",
            label_visibility="collapsed",
        )
    with col2:
        sentiment_threshold = st.slider(
            "Min Sentiment",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="sentiment_slider",
            help="Filter out drugs below this sentiment score",
        )

    # ---- Results ----
    if query and query.strip():
        with st.spinner("Searching..."):
            results = rank_drugs(
                query=query.strip(),
                inverted_index=artifacts["inverted_index"],
                fuzzy_vectorizers=artifacts["condition_vectorizers"],
                condition_matrix=artifacts["condition_matrix"],
                condition_list=artifacts["condition_list"],
                sentiment_scores_df=artifacts["sentiment_scores_df"],
                fuzzy_threshold=0.15,
                sentiment_threshold=sentiment_threshold,
            )

        if results.empty:
            st.markdown("""
            <div class="empty-state">
                <div class="emoji">🔍</div>
                <h3>No results found</h3>
                <p>Try a different condition name or lower the sentiment threshold.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Search summary
            unique_conditions = results["condition"].nunique()
            st.markdown(
                f'<div class="search-summary">'
                f'Found <strong>{len(results)}</strong> drugs across '
                f'<strong>{unique_conditions}</strong> matching condition(s) '
                f'for "<strong>{query}</strong>"'
                f'{"  •  Filtered to sentiment ≥ " + f"{sentiment_threshold:.0%}" if sentiment_threshold > 0 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Render result cards
            for idx, row in results.iterrows():
                score_color = get_score_color(row["finalScore"])

                # Get a review snippet
                snippet = get_review_snippet(
                    artifacts["all_df"], row["drugName"], row["condition"]
                )
                snippet_html = (
                    f'<div class="review-snippet">"{snippet}"</div>'
                    if snippet else ""
                )

                st.markdown(f"""
                <div class="drug-card">
                    <div class="drug-name">{row['drugName']}</div>
                    <div class="drug-condition">Condition: {row['condition']}</div>
                    <div class="score-row">
                        <span class="score-badge badge-final">
                            Final Score: {row['finalScore']:.3f}
                        </span>
                        <span class="score-badge badge-sentiment">
                            Sentiment: {row['sentimentScore']:.1%}
                        </span>
                        <span class="score-badge badge-relevance">
                            Relevance: {row['keywordRelevance']:.3f}
                        </span>
                        <span class="score-badge badge-reviews">
                            {int(row['reviewCount'])} reviews
                        </span>
                    </div>
                    <div class="score-bar-container">
                        <div class="score-bar-fill" style="width: {row['finalScore'] * 100:.1f}%; background: {score_color};"></div>
                    </div>
                    {snippet_html}
                </div>
                """, unsafe_allow_html=True)

                # Limit to top 50 results for performance
                if idx >= 49:
                    st.info(f"Showing top 50 of {len(results)} results. "
                            f"Increase the sentiment threshold to narrow results.")
                    break

    else:
        # Empty state — no query entered yet
        st.markdown("""
        <div class="empty-state">
            <div class="emoji">💊</div>
            <h3>Search for a medical condition to get started</h3>
            <p>Enter a condition like "anxiety", "acne", or "depression" to see ranked drug recommendations based on real patient reviews.</p>
        </div>
        """, unsafe_allow_html=True)

    # ---- Footer disclaimer (always present) ----
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This system is for educational and informational purposes only. "
        "It does not constitute medical advice. Always consult a qualified healthcare "
        "professional before making any medication-related decisions.  \n"
        "**MediMatch** — B.Tech 1st Year IR & ML Project by Mayan Kedia "
        "(Enrollment No. 2501010050), JIIT Noida"
    )


if __name__ == "__main__":
    main()
