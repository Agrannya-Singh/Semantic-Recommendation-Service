#!/bin/bash
echo "Pre-downloading model weights to cache..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
echo "Starting FastAPI server..."
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
