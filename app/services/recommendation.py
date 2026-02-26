import asyncio
import traceback
from fastapi import HTTPException
import json
from pinecone import Pinecone
from google import genai
from google.genai import types
from app.config import PINECONE_KEY, GOOGLE_API_KEY
import os
import httpx
from app.database import get_titles_from_ids, secure_poster_url
from app.schemas import RecommendationRequest
import logging

OMDB_API_KEY = os.getenv("OMDB_API_KEY")
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self):
        self.pc = None
        self.index = None
        self.chat_client = None
        self.embed_model = None
        self.init_ai_services()

    def init_ai_services(self):
        try:
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            if PINECONE_KEY:
                self.pc = Pinecone(api_key=PINECONE_KEY)
                self.index = self.pc.Index("semantic-recommendation-service") 
                logger.info("[Service] Connected to Pinecone.")
            
            if GOOGLE_API_KEY:
                self.chat_client = genai.Client(api_key=GOOGLE_API_KEY)
                logger.info("[Service] Connected to Gemini via google-genai.")
        except Exception as e:
            logger.error(f"[Service] Discovery Error: {e}")

    async def generate_recommendations(self, req: RecommendationRequest):
        try:
            # 1. SETUP
            selected_titles = get_titles_from_ids(req.selected_movie_ids)
            augmented_query = f"Movies similar to {', '.join(selected_titles)}. Context: {req.query}" if selected_titles else req.query
            logger.debug(f"[DEBUG] Embedding Query -> {augmented_query[:50]}...")

            # 2. EMBED
            try:
                # Generate 384-dimension vector locally (in a separate thread to avoid blocking)
                query_vec = await asyncio.to_thread(self.embed_model.encode, augmented_query, convert_to_numpy=True)
                query_vec = query_vec.tolist()
            except Exception as embed_err:
                raise HTTPException(status_code=500, detail=f"EMBEDDING FAILED: {str(embed_err)}")
            
            # 3. SEARCH PINECONE
            try:
                results = self.index.query(
                    vector=query_vec, 
                    top_k=40,
                    include_metadata=True
                )
            except Exception as pinecone_err:
                 raise HTTPException(status_code=500, detail=f"PINECONE SEARCH FAILED: {str(pinecone_err)}")

            # 4. CHECK RESULTS
            if not results['matches']:
                 return {"ai_reasoning": "I couldn't find any matches. Try a broader search.", "movies": []}

            # 5. PREPARE AI CONTEXT
            context_text = ""
            for match in results['matches']:
                m = match['metadata']
                context_text += f"ID: {match['id']} | Title: {m.get('title')} | Overview: {m.get('overview')}\n"

            # 6. ASK GEMINI (RAG)
            prompt = f"""
            User Query: "{req.query}"
            User Likes: {", ".join(selected_titles)}
            
            Candidates:
            {context_text}
            
            Pick top 5. Return JSON:
            {{
                "reasoning": "Short explanation",
                "movie_ids": ["id1", "id2"]
            }}
            """
            
            ai_data = {}
            try:
                response = self.chat_client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
                    )
                )
                ai_data = json.loads(response.text)
            except Exception as ai_err:
                 logger.error(f"AI Generation Error: {ai_err}")
                 ai_data = {
                     "reasoning": "Here are the most relevant movies from our database.",
                     "movie_ids": [m['id'] for m in results['matches'][:5]]
                 }

            # 7. ASSEMBLE RESPONSE
            target_ids = ai_data.get("movie_ids", [])
            if not target_ids: target_ids = [m['id'] for m in results['matches'][:5]]
            
            ai_reasonings = ai_data.get("reasoning", {})

            # Helper for OMDB
            async def fetch_omdb_metadata(client: httpx.AsyncClient, title: str) -> dict:
                if not OMDB_API_KEY: return {}
                try:
                    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
                    response = await client.get(url, timeout=5.0)
                    if response.status_code == 200:
                            data = response.json()
                            if data.get("Response") == "True":
                                return {
                                    "poster_url": data.get("Poster"),
                                    "year": data.get("Year"),
                                    "rating": data.get("imdbRating")
                                }
                except Exception as e:
                    logger.warning(f"OMDB Error for '{title}': {e}")
                return {}

            # Async enrichment for recommendations
            async def process_recommendation(client, match):
                m = match['metadata']
                title = m.get('title')
                
                # Enrich with OMDB metadata
                omdb_data = await fetch_omdb_metadata(client, title)
                
                # Update poster logic with OMDB fallback
                movie_dict = {
                    "poster_path": omdb_data.get("poster_url") or m.get('poster_path')
                }
                movie_dict = secure_poster_url(movie_dict)
                
                # Get reasoning
                reasoning = ""
                if isinstance(ai_reasonings, dict):
                    reasoning = ai_reasonings.get(match['id'], "Recommended based on your preferences.")
                else:
                    reasoning = str(ai_reasonings)

                return {
                    "id": match['id'],
                    "title": title,
                    "overview": m.get('overview'),
                    "poster_url": movie_dict.get("poster_url"),
                    "score": match['score'],
                    "year": omdb_data.get("year"),
                    "imdb_rating": omdb_data.get("rating"),
                    "reasoning": reasoning
                }

            selected_matches = [m for m in results['matches'] if m['id'] in target_ids]
            
            # Execute concurrently
            async with httpx.AsyncClient() as client:
                final_movies = await asyncio.gather(*(process_recommendation(client, m) for m in selected_matches))
            
            return {
                "ai_reasoning": "Here are my top selections for you.",
                "movies": final_movies
            }

        except HTTPException:
            raise # Re-raise HTTP exceptions
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"SERVER ERROR: {str(e)}")

# Singleton instance
recommendation_service = RecommendationService()
