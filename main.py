import logging
from app.main import app

logging.warning("DEPRECATION WARNING: Please run the server using 'uvicorn app.main:app --reload' instead of 'uvicorn main:app --reload'.")
