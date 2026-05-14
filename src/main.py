import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.core.logging_config import setup_logging
from fastapi.staticfiles import StaticFiles
from src.core.limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import os
import google.generativeai as genai

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Modal Product Discovery API",
    version="1.0.0",
    description="Vector search using OpenCLIP and Qdrant with LLM metadata filtering."
)


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Multi-Modal Search Engine...")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")