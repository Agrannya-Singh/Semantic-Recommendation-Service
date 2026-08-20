# LLM-as-a-Judge Evaluation & RAG Retrieval Audit Framework

## 1. Overview
The **ScreenScout Recommendation System** utilizes an unsupervised dense vector search architecture paired with Google Gemini RAG reasoning. 

Because unsupervised vector retrieval scales across un-annotated catalogs, evaluating recommendation accuracy requires an automated, objective, and reproducible evaluation method. This repository implements a **Hybrid LLM-as-a-Judge & Vector Cosine Evaluation Framework** to measure retrieval precision, mathematical vector alignment, and qualitative recommendation quality.

---

## 2. Retrieval & RAG Pipeline Architecture

```
[ User Preference / Seed Movie Context ]
                  │
                  ▼
[ SentenceTransformer ('all-MiniLM-L6-v2') ]
         (Encodes 384-d Dense Vector)
                  │
                  ▼
[ Pinecone Vector Search Index ]
         (Fetches Top 15 Candidates via ANN)
                  │
                  ▼
[ Google Gemini 3 Flash (RAG Reasoner) ]
         (Ranks & Filters Top 5 Final Movies)
                  │
                  ▼
[ LLM Auditor & Vector Evaluation Stage ]
```

### Retrieval Parameters
- **Pinecone ANN Retrieval**: Fetches **Top 15 candidate movies** based on 384-dimensional dense vector embeddings.
- **Gemini 3 Flash RAG Selection**: Evaluates the 15 candidate movies against the user prompt/preferences and selects the **Top 5 most relevant films** with custom reasoning.

---

## 3. Dual-Layer Evaluation Methodology

To prevent AI hallucination ("free hand") and ensure mathematical ground-truth rigor, the audit framework combines **deterministic vector math** with **LLM semantic judging**.

### Layer 1: Deterministic Vector Cosine Similarity
For each seed query $V_{\text{query}}$ and each retrieved movie candidate $V_{\text{candidate}}$:

$$\text{Cosine Similarity}(V_{\text{query}}, V_{\text{candidate}}) = \frac{V_{\text{query}} \cdot V_{\text{candidate}}}{\|V_{\text{query}}\| \|V_{\text{candidate}}\|}$$

- **Mean Vector Cosine Similarity**: Average mathematical vector similarity across all returned candidates.
- **Vector Precision %**: Percentage of candidate recommendations achieving Cosine Similarity $\ge 0.45$.

### Layer 2: LLM Auditor Judging (Gemini 3 Flash)
Gemini 3 Flash acts as an independent auditor evaluating each retrieved candidate on a 1–5 scale:
- **5 (Exceptional Match)**: Perfect thematic, genre, and semantic alignment.
- **4 (Strong Match)**: High relevance and strong stylistic similarity.
- **3 (Moderate Match)**: Acceptable genre or stylistic overlap.
- **2 (Weak Match)**: Distant or superficial similarity.
- **1 (Irrelevant)**: Completely unrelated.

- **LLM Precision @ K**: Percentage of retrieved recommendations scoring $\ge 3$.
- **Hit Rate @ K**: Percentage of test queries producing at least one candidate scoring $\ge 4$.

### Layer 3: Hybrid Accuracy Metric

$$\text{Hybrid Accuracy} = 50\% \times \text{Vector Cosine Precision} + 50\% \times \text{LLM Judge Precision}$$

---

## 4. Environment Configuration

All AI and vector operations rely on the unified `GOOGLE_API_KEY` defined in `.env.example`:

```bash
# Core Gemini Key
GOOGLE_API_KEY="AIzaSyYourActualGoogleGeminiApiKeyHere"

# Pinecone Vector DB Key
PINECONE_KEY="your-pinecone-api-key-here"
```

---

## 5. Running the Audit Framework

### CLI Script Execution
```bash
python eval_llm_judge.py --samples 5 --output eval_report.json
```

### HTTP Endpoint Trigger
```http
GET /recommend/evaluate?samples=5
```

### Non-Blocking Background Startup Task
When FastAPI starts (`app/main.py`), a non-blocking background task triggers after 3 seconds. Server boot remains instantaneous (0ms delay), while the background worker evaluates performance and writes persistent audit records to:
1. **Pinecone Cloud Metadata** (`namespace="audit_logs"`, `id="sys_audit_latest"`).
2. **Local SQLite Database** (`movies.db` -> `evaluation_history` table).
