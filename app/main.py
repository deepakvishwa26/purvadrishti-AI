"""
PURVADRISHTI — FastAPI Application Entry Point

Run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Swagger:  http://localhost:8000/docs
ReDoc:    http://localhost:8000/redoc
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.predictor import PredictionService
from app.core.config import APP_TITLE, APP_DESCRIPTION, APP_VERSION


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Phase 7 model ONCE at startup. Never reload per request."""
    print("\n" + "="*60)
    print("  PURVADRISHTI — Predictive Intelligence")
    print("  Cyber-Fraud Cash-Out Response System")
    print("="*60)
    print("\n[STARTUP] Loading Phase 7 XGBoost (frozen)...")
    svc = PredictionService()
    svc.load()
    app.state.predictor = svc
    print(f"[STARTUP] Model loaded. {len(svc.known_complaints):,} complaints available.")
    print(f"[STARTUP] Dashboard: http://localhost:8000")
    print(f"[STARTUP] Swagger:   http://localhost:8000/docs")
    print(f"[STARTUP] ReDoc:     http://localhost:8000/redoc\n")
    yield
    print("\n[SHUTDOWN] PURVADRISHTI shutting down.")


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Static files
STATIC_DIR = os.path.join(ROOT, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Routes
app.include_router(router)
