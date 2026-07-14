# MediMatch — Demo Script

**For viva / project demonstration walkthrough**

---

## Setup Verification

Before starting the demo, confirm:
```bash
# From the medimatch/ directory
streamlit run app.py
```
The app should launch in a browser at `http://localhost:8501`.

---

## Demo Walkthrough

### 1. First Impression — Medical Disclaimer

**What to show:** The medical disclaimer is visible immediately on page load — both as a yellow banner at the top and in the footer. This is present on every view, not just a landing page.

**Talking point:** *"The disclaimer is non-negotiable and appears on every page because this is an educational tool, not clinical advice."*

---

### 2. Query: "anxiety"

**Action:** Type `anxiety` in the search bar.

**Expected behavior:**
- Multiple drugs appear, ranked by Final Score
- Top results should include well-known anxiety medications (e.g., Lexapro, Buspar, Effexor, Klonopin)
- Each card shows: Drug Name, Condition ("Anxiety"), Final Score, Sentiment Score, Relevance, Review Count
- The Final Score combines 0.4 × Keyword Relevance + 0.6 × Sentiment Score
- A review snippet from a real patient appears on each card

**Talking point:** *"The query 'anxiety' matches exactly via the inverted index. Drugs are ranked by a weighted combination of how relevant the condition match is and what proportion of patient reviews are positive."*

---

### 3. Query: "acne"

**Action:** Clear the search bar and type `acne`.

**Expected behavior:**
- Acne-specific drugs appear (e.g., Accutane/Isotretinoin, Doxycycline, Tretinoin)
- Results are ranked with the same scoring formula
- Different drugs and sentiment scores than the anxiety query

**Talking point:** *"This demonstrates the system works across different medical conditions with the same underlying pipeline."*

---

### 4. Typo Test: "anxeity"

**Action:** Clear the search bar and type `anxeity` (deliberate misspelling).

**Expected behavior:**
- The system STILL returns anxiety-related drugs
- The fuzzy matching (TF-IDF cosine similarity with character n-grams) catches the misspelling
- Keyword Relevance scores may be slightly lower than the exact "anxiety" query

**Talking point:** *"This is the fuzzy matching in action — Layer 2 uses TF-IDF cosine similarity with character n-grams to handle misspellings and abbreviations. The system doesn't require exact spelling."*

---

### 5. Sentiment Slider Interaction

**Action:** With "anxiety" results showing, slowly move the "Min Sentiment" slider from 0.0 upward to 0.5, 0.7, 0.9.

**Expected behavior:**
- As the threshold increases, drugs with lower sentiment scores disappear
- The number of results decreases visibly
- Only the highest-rated drugs remain at 0.9+

**Talking point:** *"The sentiment slider lets users filter by minimum patient satisfaction. A drug needs at least this proportion of positive reviews to appear. This is practical — a user might only want drugs that most patients rated highly."*

---

### 6. Sidebar — Model Info

**Action:** Point to the sidebar.

**Expected behavior:**
- "How It Works" explanation of the 5-step pipeline
- Model Performance comparison showing Logistic Regression vs Naïve Bayes accuracy, precision, recall, F1
- The ranking formula: `Final Score = 0.4 × R_k + 0.6 × S_s`
- Dataset source attribution

**Talking point:** *"The sidebar shows model transparency — users can see which classifier was selected and how it performed. The Logistic Regression model typically achieves around 89–92% accuracy on this dataset."*

---

### 7. Other Queries to Try

| Query | What it demonstrates |
|-------|---------------------|
| `depression` | Large result set, well-known condition |
| `birth control` | Multi-word condition matching |
| `insomnia` | Common condition with many drugs |
| `depresion` | Typo tolerance (Spanish-influenced misspelling) |
| `migrane` | Typo tolerance for "migraine" |
| `ADHD` | Acronym matching |
| `weight loss` | Multi-word query |

---

## Technical Questions to Prepare For

1. **"What is an inverted index?"** — A data structure mapping terms to the documents containing them. Here, stemmed condition tokens map to condition strings and their associated drugs.

2. **"Why TF-IDF and not just exact match?"** — TF-IDF captures term importance and enables fuzzy matching via cosine similarity, handling misspellings and partial queries.

3. **"Why Logistic Regression over Naïve Bayes?"** — Both were trained and evaluated. Logistic Regression typically performs better on this dataset because it models feature interactions better than the conditional independence assumption of Naïve Bayes.

4. **"Why drop ratings 5–6?"** — They're ambiguous — neither clearly positive nor negative. Dropping them sharpens the decision boundary and is standard practice for this dataset in published work.

5. **"What does the Final Score mean?"** — It's a weighted blend: 40% how well the condition matched, 60% how positive patient reviews are. The 0.6 weight on sentiment reflects that patient experience matters more than just text matching.

6. **"Could this be used in production?"** — No. It's a static snapshot, not clinically validated, and uses only one dataset. The disclaimer exists for this reason. It demonstrates IR + ML concepts, not clinical decision support.

---

## Retraining (if asked)

```bash
python train.py
```
Takes ~5–10 minutes. Prints full classification reports, confusion matrices, and sample queries.
