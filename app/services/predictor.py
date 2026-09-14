"""
PURVADRISHTI — Prediction Service
Thin wrapper around the existing HIVEPredictor.

Responsibilities:
  1. Call HIVEPredictor.predict()
  2. Apply the corrected alert-priority heuristic (margin-based)
  3. Compute relative model score (0-100 int) for display
  4. Build Pydantic response objects
  5. Cleanly separate ground-truth from prediction output

The HIVEPredictor class is NOT renamed (per spec).
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from typing import Optional
from app.api.schemas import (
    PredictResponse, ComplaintInfo, H3Prediction, ATMCandidate,
    SHAPExplanation, SHAPFactor, ModelMetadata,
)
from app.core.config import MODEL_INFO, ALERT_THRESHOLDS, ALERT_DESCRIPTIONS, ALERT_HEURISTIC_NOTE, SEED
from src.phase10_pipeline import HIVEPredictor


def _compute_alert_priority(sorted_raw_scores: list) -> tuple[str, float]:
    """
    UI Prioritization Heuristic — NOT model probability.

    Rule:
        score_margin = (rank1_score - rank2_score) / max(rank1_score, 1e-9)
        HIGH:   margin > 0.20  (strong score separation)
        MEDIUM: margin > 0.05
        LOW:    otherwise

    This heuristic is documented as a transparent UI prototype rule.
    It is NOT calibrated model confidence.
    """
    if len(sorted_raw_scores) < 2:
        return "MEDIUM", 0.0
    s1 = float(sorted_raw_scores[0])
    s2 = float(sorted_raw_scores[1])
    margin = (s1 - s2) / max(abs(s1), 1e-9)
    if margin > ALERT_THRESHOLDS["HIGH"]:   return "HIGH",   round(margin, 4)
    if margin > ALERT_THRESHOLDS["MEDIUM"]: return "MEDIUM", round(margin, 4)
    return "LOW", round(margin, 4)


def _relative_model_score(raw_scores: list) -> list:
    """
    Normalize raw XGBoost scores to [0, 100] integer per complaint.
    min score → 0, max score → 100.
    Used for display only — NOT a probability.
    """
    if not raw_scores:
        return []
    mn = min(raw_scores); mx = max(raw_scores); rng = mx - mn
    if rng < 1e-9:
        return [100] * len(raw_scores)
    return [round((s - mn) / rng * 100) for s in raw_scores]


def _build_atm(atm: dict) -> ATMCandidate:
    def _s(v) -> str:
        """NaN-safe string: float NaN → empty string."""
        if v is None: return ""
        try:
            f = float(v)
            if f != f: return ""  # NaN check
        except (TypeError, ValueError): pass
        return str(v)
    return ATMCandidate(
        atm_id=_s(atm.get("atm_id")),
        state=_s(atm.get("state")),
        city=_s(atm.get("city")),
        lat=float(atm.get("lat") or 0),
        lon=float(atm.get("lon") or 0),
        dist_to_h3_centroid_km=float(atm.get("dist_to_h3_centroid_km") or 0),
        dist_to_victim_km=float(atm.get("dist_to_victim_km") or 0),
    )


def _build_shap(shap: Optional[dict]) -> Optional[SHAPExplanation]:
    if not shap:
        return None
    def _factor(f: dict) -> SHAPFactor:
        return SHAPFactor(
            feature=f["feature"],
            display_name=f["display_name"],
            shap_value=float(f["shap_value"]),
            feature_value=float(f["feature_value"]),
            direction=f["direction"],
        )
    return SHAPExplanation(
        bias=float(shap.get("bias", 0)),
        top_factors=[_factor(f) for f in shap.get("top_factors", [])],
        positive_factors=[_factor(f) for f in shap.get("positive_factors", [])],
        negative_factors=[_factor(f) for f in shap.get("negative_factors", [])],
    )


class PredictionService:
    """Singleton prediction service. Load once at startup."""

    def __init__(self):
        self._predictor = HIVEPredictor(seed=SEED)
        self._loaded = False

    def load(self):
        self._predictor.load()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def known_complaints(self) -> set:
        return self._predictor._known_complaints

    def predict(self, complaint_id: str) -> PredictResponse:
        assert self._loaded, "Service not loaded"

        # ── Run core prediction (unchanged HIVEPredictor) ─────────────
        raw = self._predictor.predict(complaint_id, top_n=5, seed=SEED)

        top = raw["top_predictions"]

        # ── Compute relative model scores (0-100, display only) ────────
        raw_scores_sorted = [p["xgb_score"] for p in top]
        rel_scores = _relative_model_score(raw_scores_sorted)

        # ── Compute alert priority (margin-based heuristic) ─────────────
        all_scores = [p["xgb_score"] for p in top]
        priority, margin = _compute_alert_priority(all_scores)

        # ── Build H3Prediction list ────────────────────────────────────
        h3_preds = []
        for i, p in enumerate(top):
            h3_preds.append(H3Prediction(
                rank=p["rank"],
                h3_cell=p["h3_cell"],
                xgb_raw_score=round(float(p["xgb_score"]), 6),
                relative_model_score=rel_scores[i],
                dist_from_victim_km=float(p.get("dist_from_victim_km", 0)),
                atm_count_in_zone=int(p.get("atm_count_in_zone", 0)),
                in_victim_state=bool(p.get("in_victim_state", False)),
                centroid_lat=float(p["centroid_lat"]),
                centroid_lon=float(p["centroid_lon"]),
                boundary=[[float(c[0]), float(c[1])] for c in p["boundary"]],
                nearby_atms=[_build_atm(a) for a in (p.get("nearby_atms") or [])],
                shap=_build_shap(p.get("shap")),
                # Ground truth — demo only
                validation_is_actual_h3=bool(p.get("is_actual_h3", False)),
            ))

        ci = raw["complaint_info"]
        mm = MODEL_INFO

        return PredictResponse(
            complaint_id=complaint_id,
            complaint_info=ComplaintInfo(
                fraud_type=ci.get("fraud_type", ""),
                fraud_amount=float(ci.get("fraud_amount", 0)),
                victim_state=ci.get("victim_state", ""),
                victim_lat=float(ci.get("victim_lat", 0)),
                victim_lon=float(ci.get("victim_lon", 0)),
                incident_datetime=str(ci.get("incident_datetime", "")),
                complaint_registered_at=str(ci.get("complaint_registered_at", "")),
            ),
            feature_cutoff=str(raw.get("feature_cutoff", "")),
            n_candidates=int(raw.get("n_candidates", 0)),
            top_predictions=h3_preds,
            alert_priority=priority,
            alert_heuristic_description=ALERT_DESCRIPTIONS[priority],
            alert_heuristic_note=ALERT_HEURISTIC_NOTE,
            score_margin=margin,
            # Ground truth — clearly labelled as demo/validation only
            validation_actual_h3_cells=list(raw.get("actual_h3_cells", [])),
            validation_actual_rank=raw.get("actual_rank_in_candidates"),
            validation_top5_hit=bool(raw.get("top5_hit", False)),
            model_metadata=ModelMetadata(
                model=mm["model"],
                version=mm["version"],
                features=mm["features"],
                objective=mm["objective"],
                t3_hit_at_5=mm["t3_hit_at_5"],
                mrr=mm["mrr"],
                ndcg_at_5=mm["ndcg_at_5"],
                kde_primary=mm["kde_primary"],
                kde_note=mm["kde_note"],
                model_frozen=mm["model_frozen"],
            ),
        )
