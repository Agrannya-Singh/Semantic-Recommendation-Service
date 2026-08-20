from fastapi import APIRouter, Query
from app.schemas import RecommendationRequest
from app.services.recommendation import recommendation_service

router = APIRouter()

@router.post("/recommend")
async def recommend_movies(req: RecommendationRequest):
    return await recommendation_service.generate_recommendations(req)

@router.get("/recommend/evaluate")
async def evaluate_recommendations(samples: int = Query(default=5, ge=1, le=20)):
    """Runs LLM-as-a-Judge evaluation framework using Gemini 3 Flash Auditor."""
    from eval_llm_judge import fetch_random_seed_movies, synthesize_user_query, judge_recommendations_with_gemini
    from google import genai
    from app.config import GOOGLE_API_KEY

    if not GOOGLE_API_KEY:
        return {"error": "GOOGLE_API_KEY not configured"}

    client = genai.Client(api_key=GOOGLE_API_KEY)
    seed_movies = fetch_random_seed_movies(limit=samples)
    
    results = []
    total_precision = 0.0
    total_relevance = 0.0

    for seed in seed_movies:
        query = synthesize_user_query(seed)
        req = RecommendationRequest(query=query, selected_movie_ids=[str(seed['id'])])
        rec_res = await recommendation_service.generate_recommendations(req)
        movies = rec_res.get("movies", [])
        
        if movies:
            judge_res = await judge_recommendations_with_gemini(client, seed, query, movies)
            prec = judge_res.get("precision_at_k", 0.0)
            avg_rel = judge_res.get("average_relevance", 0.0)
            total_precision += prec
            total_relevance += avg_rel
            results.append({
                "seed_title": seed.get("title"),
                "precision_at_k": prec,
                "average_relevance": avg_rel,
                "summary": judge_res.get("overall_judgment_summary")
            })

    n = len(results)
    return {
        "samples_evaluated": n,
        "metrics": {
            "mean_precision_pct": round((total_precision / n) * 100, 2) if n > 0 else 0.0,
            "mean_relevance_score": round(total_relevance / n, 2) if n > 0 else 0.0
        },
        "details": results
    }

