import os

# ---  CREDENTIALS ---
PINECONE_KEY = os.getenv("PINECONE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- PATHS ---
# Assumes this config.py is inside app/, so we go up one level to root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "movies.db")
