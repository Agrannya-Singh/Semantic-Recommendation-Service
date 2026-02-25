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
