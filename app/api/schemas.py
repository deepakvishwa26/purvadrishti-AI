"""
PURVADRISHTI — Pydantic Request/Response Schemas
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


# ── Request ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    complaint_id: str

    model_config = {"json_schema_extra": {"example": {"complaint_id": "CMP00007160"}}}


# ── Sub-models ────────────────────────────────────────────────────────
class ATMCandidate(BaseModel):
    atm_id: str
    state: str
    city: str
    lat: float
    lon: float
    dist_to_h3_centroid_km: float
    dist_to_victim_km: float


class SHAPFactor(BaseModel):
    feature: str
    display_name: str
    shap_value: float
    feature_value: float
    direction: str  # "increases_score" | "decreases_score"


class SHAPExplanation(BaseModel):
    bias: float
    top_factors: List[SHAPFactor]
    positive_factors: List[SHAPFactor]
    negative_factors: List[SHAPFactor]


class H3Prediction(BaseModel):
    rank: int
    h3_cell: str
    xgb_raw_score: float
    relative_model_score: int          # 0-100, normalized within complaint
    dist_from_victim_km: float
    atm_count_in_zone: int
    in_victim_state: bool
    centroid_lat: float
    centroid_lon: float
    boundary: List[List[float]]        # [[lat, lon], ...]
    nearby_atms: List[ATMCandidate]
    shap: Optional[SHAPExplanation]
    # Ground truth — demo/validation only, NOT used in scoring
    validation_is_actual_h3: bool


class ComplaintInfo(BaseModel):
    fraud_type: str
    fraud_amount: float
    victim_state: str
    victim_lat: float
    victim_lon: float
    incident_datetime: str
    complaint_registered_at: str


class ModelMetadata(BaseModel):
    model_config = {"protected_namespaces": ()}
    model: str
    version: str
    features: int
    objective: str
    t3_hit_at_5: float
    mrr: float
    ndcg_at_5: float
    kde_primary: bool
    kde_note: str
    model_frozen: bool


# ── Primary response ──────────────────────────────────────────────────
class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    complaint_id: str
    complaint_info: ComplaintInfo
    feature_cutoff: str
    n_candidates: int

    # Prediction output
    top_predictions: List[H3Prediction]
    alert_priority: str                 # HIGH | MEDIUM | LOW
    alert_heuristic_description: str
    alert_heuristic_note: str
    score_margin: float                 # raw margin used for priority heuristic

    # Ground truth — demo/validation only, clearly separated
    validation_actual_h3_cells: List[str]
    validation_actual_rank: Optional[int]
    validation_top5_hit: bool

    model_metadata: ModelMetadata


# ── Other responses ───────────────────────────────────────────────────
class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model: str
    model_frozen: bool
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model: str
    version: str
    features: int
    objective: str
    t3_hit_at_5: float
    mrr: float
    ndcg_at_5: float
    kde_primary: bool
    kde_note: str


class ComplaintsResponse(BaseModel):
    complaints: List[str]
    total: int


class ErrorResponse(BaseModel):
    error: str
