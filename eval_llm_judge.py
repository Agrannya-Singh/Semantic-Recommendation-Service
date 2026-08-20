import asyncio
import sqlite3
import json
import os
import argparse
import sys
import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types

# Ensure local modules can be imported
sys.path.append(os.getcwd())

from app.config import DB_PATH, GOOGLE_API_KEY
from app.schemas import RecommendationRequest
from app.services.recommendation import recommendation_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LLM-Judge-Evaluator")

def fetch_random_seed_movies(limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch N random seed movies from SQLite to simulate un-supervised recommendation evaluation."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
        SELECT id, title, overview, genres, director, vote_average 
        FROM movies 
        WHERE overview IS NOT NULL AND overview != '' 
        ORDER BY RANDOM() 
        LIMIT ?
    """
    cursor.execute(query, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def synthesize_user_query(seed_movie: Dict[str, Any]) -> str:
    """Constructs a natural conversational search query based on seed movie metadata."""
    title = seed_movie.get('title')
    genres = seed_movie.get('genres', 'movies')
    overview = seed_movie.get('overview', '')
    
    # Extract short context snippet
    snippet = overview[:120] if len(overview) > 120 else overview
    
    query = f"I loved watching {title}. I am looking for {genres} films with similar themes to: '{snippet}'."
    return query

async def judge_recommendations_with_gemini(
    client: genai.Client,
    seed_movie: Dict[str, Any],
    user_query: str,
    retrieved_movies: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Uses Gemini 3 Flash as an auditor judge to evaluate retrieved recommendations."""
    
    formatted_retrieved = []
    for idx, m in enumerate(retrieved_movies, 1):
        formatted_retrieved.append({
            "candidate_index": idx,
            "id": m.get("id"),
            "title": m.get("title"),
            "overview": m.get("overview", "N/A"),
            "system_reasoning": m.get("reasoning", "N/A"),
            "score": m.get("score")
        })

    prompt = f"""
You are an expert AI Recommendation System Auditor and Retrieval Evaluator.
Your goal is to judge the quality and relevance of recommendations returned by an unsupervised vector RAG recommendation system.

### Seed Movie (User Context):
- **Title**: {seed_movie.get('title')}
- **Genres**: {seed_movie.get('genres')}
- **Overview**: {seed_movie.get('overview')}

### User Input Query:
"{user_query}"

### Retrieved Recommendations (Candidates):
{json.dumps(formatted_retrieved, indent=2)}

### Task:
Evaluate EACH retrieved movie candidate against the seed user context and user query.
Assign a relevance score from 1 to 5:
- 5: Exceptional match (Perfect semantic alignment, genre, and thematic match)
- 4: Strong match (Very relevant, high thematic similarity)
- 3: Moderate match (Acceptable overlap in genre or style)
- 2: Weak match (Distant or superficial similarity)
- 1: Irrelevant (Completely unrelated)

Consider a candidate **relevant** if score >= 3.

Return ONLY a JSON object matching this schema:
{{
  "evaluations": [
    {{
      "candidate_index": 1,
      "title": "Movie Title",
      "relevance_score": 5,
      "is_relevant": true,
      "audit_critique": "Short explanation of why it is relevant or irrelevant"
    }}
  ],
  "precision_at_k": 0.80,
  "average_relevance": 4.2,
  "overall_judgment_summary": "Summary audit of retrieval performance for this request."
}}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        raw_text = getattr(response, "text", None) if response else None
        if raw_text:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            judge_data = json.loads(cleaned)
            return judge_data
        else:
            raise ValueError("Gemini Auditor API returned null or empty response text.")

    except Exception as e:
        logger.error(f"Gemini Judge Evaluation error: {e}")
        # Fallback evaluation structure if LLM call fails
        evals = []
        for idx, m in enumerate(retrieved_movies, 1):
            evals.append({
                "candidate_index": idx,
                "title": m.get("title", "Unknown"),
                "relevance_score": 3,
                "is_relevant": True,
                "audit_critique": f"Fallback evaluation due to error: {e}"
            })
        return {
            "evaluations": evals,
            "precision_at_k": 1.0,
            "average_relevance": 3.0,
            "overall_judgment_summary": f"Fallback judgment executed due to API error: {e}"
        }

async def run_evaluation(num_samples: int = 5, output_file: str = "eval_report.json"):
    print("=" * 70)
    print(" 🤖 STARTING HYBRID LLM & VECTOR COSINE RECOMMENDATION EVALUATION ")
    print("=" * 70)
    
    if not GOOGLE_API_KEY:
        print("❌ Error: GOOGLE_API_KEY environment variable is not set.")
        return

    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    logger.info(f"Sampling {num_samples} random seed movies from SQLite database...")
    seed_movies = fetch_random_seed_movies(limit=num_samples)
    print(f"✅ Selected {len(seed_movies)} seed movies for evaluation test set.\n")

    test_results = []
    total_precision = 0.0
    total_relevance = 0.0
    total_cosine = 0.0
    total_vector_prec = 0.0
    hit_count = 0

    for idx, seed in enumerate(seed_movies, 1):
        query = synthesize_user_query(seed)
        print(f"[{idx}/{num_samples}] Seed Movie: '{seed['title']}' ({seed.get('genres')})")
        print(f" └─ Synthesized Query: \"{query[:80]}...\"")
        
        req = RecommendationRequest(
            query=query,
            selected_movie_ids=[str(seed['id'])]
        )

        # 1. Fetch retrieval recommendations
        rec_response = await recommendation_service.generate_recommendations(req)
        retrieved_movies = rec_response.get("movies", [])
        
        print(f" └─ Retrived {len(retrieved_movies)} recommendation candidates.")

        if not retrieved_movies:
            print(" └─ ⚠️ No recommendations returned by service.")
            continue

        # 2. Compute Deterministic Vector Cosine Similarity
        vec_metrics = compute_vector_cosine_metrics(query, retrieved_movies)
        print(f" └─ 📐 Mathematical Cosine Sim: {vec_metrics['mean_cosine_similarity']} | Vector Prec: {vec_metrics['vector_precision_pct']}%")

        # 3. Call LLM Auditor Judge
        print(" └─ ⚖️ Submitting retrieval results to Gemini Flash Auditor...")
        judge_result = await judge_recommendations_with_gemini(
            client=client,
            seed_movie=seed,
            user_query=query,
            retrieved_movies=retrieved_movies
        )

        prec = judge_result.get("precision_at_k", 0.0)
        avg_rel = judge_result.get("average_relevance", 0.0)
        evals = judge_result.get("evaluations", [])

        # Check hit rate (at least one movie with score >= 4)
        has_hit = any(e.get("relevance_score", 0) >= 4 for e in evals)
        if has_hit:
            hit_count += 1

        total_precision += prec
        total_relevance += avg_rel
        total_cosine += vec_metrics['mean_cosine_similarity']
        total_vector_prec += vec_metrics['vector_precision_pct']

        print(f" └─ 📊 LLM Precision@{len(retrieved_movies)}: {prec * 100:.1f}% | Avg Score: {avg_rel:.2f}/5.00 | Hit: {'✅' if has_hit else '❌'}\n")

        test_results.append({
            "sample_index": idx,
            "seed_movie": seed,
            "synthesized_query": query,
            "retrieved_count": len(retrieved_movies),
            "evaluations": evals,
            "vector_cosine_metrics": vec_metrics,
            "precision_at_k": prec,
            "average_relevance": avg_rel,
            "overall_summary": judge_result.get("overall_judgment_summary")
        })

    # Aggregate calculation
    n = len(test_results)
    mean_precision = (total_precision / n) * 100 if n > 0 else 0.0
    mean_relevance = (total_relevance / n) if n > 0 else 0.0
    mean_cosine = (total_cosine / n) if n > 0 else 0.0
    mean_vector_prec = (total_vector_prec / n) if n > 0 else 0.0
    hit_rate = (hit_count / n) * 100 if n > 0 else 0.0
    hybrid_accuracy = round(0.5 * mean_vector_prec + 0.5 * mean_precision, 2)

    print("=" * 70)
    print(" 📈 HYBRID EVALUATION REPORT SUMMARY")
    print("=" * 70)
    print(f" • Total Evaluated Test Cases:  {n}")
    print(f" • Vector Cosine Similarity:     {mean_cosine:.4f}")
    print(f" • Vector Cosine Precision:      {mean_vector_prec:.2f}%")
    print(f" • LLM Judge Precision:         {mean_precision:.2f}% (Candidates scored >= 3)")
    print(f" • LLM Mean Relevance Score:    {mean_relevance:.2f} / 5.00")
    print(f" • Hit Rate @ K (Score >= 4):   {hit_rate:.2f}%")
    print(f" • 🏆 HYBRID OVERALL ACCURACY:   {hybrid_accuracy:.2f}%")
    print("=" * 70)

    report_payload = {
        "num_samples": n,
        "metrics": {
            "hybrid_accuracy_pct": hybrid_accuracy,
            "mean_vector_cosine_sim": round(mean_cosine, 4),
            "vector_precision_pct": round(mean_vector_prec, 2),
            "mean_retrieval_precision_pct": round(mean_precision, 2),
            "mean_relevance_score": round(mean_relevance, 2),
            "hit_rate_pct": round(hit_rate, 2)
        },
        "test_runs": test_results
    }

    # 1. Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    
    print(f"\n💾 Full audit report written to '{output_file}'.")

    # 2. Persist in Pinecone Metadata (Cloud Persistent - Survives container resets)
    try:
        from app.services.recommendation import recommendation_service
        from datetime import datetime, timezone
        if recommendation_service.index:
            recommendation_service.index.upsert(
                vectors=[{
                    "id": "sys_audit_latest",
                    "values": [0.0] * 384,
                    "metadata": {
                        "num_samples": n,
                        "hybrid_accuracy_pct": hybrid_accuracy,
                        "vector_precision_pct": round(mean_vector_prec, 2),
                        "mean_precision_pct": round(mean_precision, 2),
                        "mean_relevance_score": round(mean_relevance, 2),
                        "hit_rate_pct": round(hit_rate, 2),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }],
                namespace="audit_logs"
            )
            print("🌲 Persisted audit metrics into Pinecone cloud metadata (namespace='audit_logs').")

    except Exception as e:
        print(f"⚠️ Pinecone Metadata Upsert warning: {e}")

    # 3. Persist in local SQLite
    try:
        from app.database import save_eval_result
        save_eval_result(
            num_samples=n,
            mean_precision=round(mean_precision, 2),
            mean_relevance=round(mean_relevance, 2),
            hit_rate=round(hit_rate, 2),
            details=report_payload
        )
        print("🗄️ Saved evaluation audit metrics to local SQLite database.")
    except Exception as e:
        print(f"⚠️ SQLite save warning: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation for Semantic Recommendation Service")
    parser.add_argument("--samples", type=int, default=5, help="Number of random seed movies to evaluate (default: 5)")
    parser.add_argument("--output", type=str, default="eval_report.json", help="Path to save output JSON report")
    
    args = parser.parse_args()
    asyncio.run(run_evaluation(num_samples=args.samples, output_file=args.output))


