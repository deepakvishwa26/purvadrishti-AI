"""
PURVADRISHTI — Configuration
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────
ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT_DIR, "data", "output", "phase7")
DATA_DIR  = os.path.join(ROOT_DIR, "data", "output")
P6V2_DIR  = os.path.join(ROOT_DIR, "data", "output", "phase6_v2")

MODEL_PATH = os.path.join(MODEL_DIR, "model.ubj")

# ── Model metadata (frozen) ───────────────────────────────────────────
MODEL_INFO = {
    "model":       "Phase 7 XGBoost",
    "version":     "Phase 7 Frozen",
    "features":    22,
    "objective":   "rank:ndcg",
    "t3_hit_at_5": 0.6526,
    "mrr":         0.4971,
    "ndcg_at_5":   0.5135,
    "kde_primary": False,
    "kde_note":    (
        "KDE evaluated experimentally in Phase 9 under temporal leakage controls. "
        "Provided no demonstrated utility on the current synthetic dataset. "
        "XGBoost is the sole primary scoring model."
    ),
    "phase8_note": "Phase 8 mule-network features: experimental only — no verified end-to-end improvement.",
    "model_frozen": True,
}

# ── Alert priority — UI heuristic (NOT model probability) ────────────
# Based on relative score separation between rank-1 and rank-2 candidate.
# score_margin = (score_rank1 - score_rank2) / score_rank1
# This is a transparent UI prioritization heuristic, not calibrated confidence.
ALERT_THRESHOLDS = {
    "HIGH":   0.20,   # relative margin > 20% → strong separation
    "MEDIUM": 0.05,   # relative margin > 5%
    # else LOW
}
ALERT_DESCRIPTIONS = {
    "HIGH":   "Strong score separation — warrants immediate review",
    "MEDIUM": "Moderate score separation — review recommended",
    "LOW":    "Weak score separation — low-confidence zone",
}
ALERT_HEURISTIC_NOTE = (
    "Prototype UI prioritization heuristic — NOT model probability, "
    "NOT calibrated confidence, NOT an operational law-enforcement threshold. "
    "Human review required."
)

# ── API ───────────────────────────────────────────────────────────────
APP_TITLE       = "PURVADRISHTI"
APP_DESCRIPTION = (
    "Predictive Intelligence for Cyber-Fraud Cash-Out Response.\n\n"
    "Decision-support prototype using Phase 7 XGBoost LTR (frozen). "
    "Official held-out T3 Hit@5 = 65.26%.\n\n"
    "⚠️ This is a human-in-the-loop decision-support prototype. "
    "Predictions do not trigger automatic enforcement or account freezing."
)
APP_VERSION     = "10.0.0"
SEED            = 42
