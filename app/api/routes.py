"""
PURVADRISHTI — FastAPI Routes
"""
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.schemas import (
    PredictRequest, PredictResponse,
    HealthResponse, ModelInfoResponse, ComplaintsResponse, ErrorResponse,
)
from app.core.config import MODEL_INFO, APP_TITLE

router = APIRouter()
templates = Jinja2Templates(directory="templates")


# ── Dashboard ─────────────────────────────────────────────────────────
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── Health ────────────────────────────────────────────────────────────
@router.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["System"],
)
async def health(request: Request):
    svc = request.app.state.predictor
    return HealthResponse(
        status="ok",
        model=MODEL_INFO["model"],
        model_frozen=True,
        model_loaded=svc.is_loaded,
    )


# ── Model info ────────────────────────────────────────────────────────
@router.get(
    "/api/model-info",
    response_model=ModelInfoResponse,
    summary="Frozen model metadata",
    tags=["Model"],
)
async def model_info():
    return ModelInfoResponse(
        model=MODEL_INFO["model"],
        version=MODEL_INFO["version"],
        features=MODEL_INFO["features"],
        objective=MODEL_INFO["objective"],
        t3_hit_at_5=MODEL_INFO["t3_hit_at_5"],
        mrr=MODEL_INFO["mrr"],
        ndcg_at_5=MODEL_INFO["ndcg_at_5"],
        kde_primary=MODEL_INFO["kde_primary"],
        kde_note=MODEL_INFO["kde_note"],
    )


# ── Complaints ────────────────────────────────────────────────────────
@router.get(
    "/api/complaints",
    response_model=ComplaintsResponse,
    summary="Available demo complaint IDs",
    tags=["Data"],
)
async def complaints(request: Request):
    svc = request.app.state.predictor
    all_cids = sorted(svc.known_complaints)
    # Put the Phase 10 validated demo complaints first for better demo experience
    DEMO_FIRST = [
        "CMP00007160","CMP00009397","CMP00009051","CMP00008633","CMP00008419",
        "CMP00001290","CMP00006278","CMP00008031","CMP00001297","CMP00001904",
    ]
    available = svc.known_complaints
    demo = [c for c in DEMO_FIRST if c in available]
    rest = [c for c in all_cids if c not in set(demo)]
    cids = (demo + rest)[:200]
    return ComplaintsResponse(complaints=cids, total=len(cids))


# ── Predict ───────────────────────────────────────────────────────────
@router.post(
    "/api/predict",
    response_model=PredictResponse,
    summary="Run PURVADRISHTI prediction for a complaint",
    tags=["Prediction"],
    responses={
        200: {"description": "Prediction result"},
        404: {"model": ErrorResponse, "description": "Complaint not found"},
        500: {"model": ErrorResponse, "description": "Prediction error"},
    },
)
async def predict(body: PredictRequest, request: Request):
    svc = request.app.state.predictor
    cid = body.complaint_id.strip()

    if cid not in svc.known_complaints:
        raise HTTPException(
            status_code=404,
            detail=f"Complaint '{cid}' not found in pre-computed candidate set",
        )
    try:
        result = svc.predict(cid)
        return result
    except AssertionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        # Never expose Python tracebacks to the client
        raise HTTPException(status_code=500, detail="Prediction failed. Please try another complaint.")
