from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routers import movies, recommend
from app.services.recommendation import recommendation_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warmup the SentenceTransformer model to prevent OOM on first request
    if recommendation_service.embed_model:
        recommendation_service.embed_model.encode("warmup")
    
    # Log latest LLM Judge audit metrics (Pinecone Cloud -> SQLite fallback)
    import logging
    logger = logging.getLogger("uvicorn")
    
    logged_pinecone = False
    if recommendation_service.index:
        try:
            res = recommendation_service.index.fetch(ids=["sys_audit_latest"], namespace="audit_logs")
            if res and hasattr(res, "vectors") and "sys_audit_latest" in res.vectors:
                meta = res.vectors["sys_audit_latest"].metadata
                logger.info(
                    f"[AUDIT LOG - Pinecone Cloud] Timestamp={meta.get('timestamp')} | "
                    f"Precision={meta.get('mean_precision_pct')}% | "
                    f"Relevance={meta.get('mean_relevance_score')}/5.00 | "
                    f"HitRate={meta.get('hit_rate_pct')}%"
                )
                logged_pinecone = True
        except Exception as e:
            logger.warning(f"[Pinecone Audit Fetch Warning]: {e}")

    if not logged_pinecone:
        try:
            from app.database import get_latest_eval_result
            latest_eval = get_latest_eval_result()
            if latest_eval:
                logger.info(
                    f"[AUDIT LOG - Local SQLite] Timestamp={latest_eval.get('timestamp')} | "
                    f"Precision={latest_eval.get('mean_precision_pct')}% | "
                    f"Relevance={latest_eval.get('mean_relevance_score')}/5.00 | "
                    f"HitRate={latest_eval.get('hit_rate_pct')}%"
                )
        except Exception:
            pass
    # Trigger non-blocking evaluation task in the background
    async def run_background_eval():
        try:
            await asyncio.sleep(3)
            logger.info("[Background Audit] Starting non-blocking evaluation on startup...")
            from eval_llm_judge import run_evaluation
            await run_evaluation(num_samples=3, output_file="eval_report.json")
            logger.info("[Background Audit] Non-blocking audit completed successfully.")
        except Exception as eval_err:
            logger.warning(f"[Background Audit Warning]: {eval_err}")

    asyncio.create_task(run_background_eval())
    yield




# --- APP CONFIGURATION ---
app = FastAPI(title="ScreenScout Intelligence Engine", version="PRODUCTION", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ROUTERS ---
app.include_router(movies.router)
app.include_router(recommend.router)

@app.get("/")
def health_check():
    return {"status": "online", "mode": "Secure Production"}
