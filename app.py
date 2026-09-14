"""
HIVE-Predict Demo — Flask Application
Serves the SIH-style dashboard with live prediction via Phase 7 XGBoost.

Run:
    python app.py
Then open: http://localhost:5000
"""
import os, sys, json
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from flask import Flask, render_template, request, jsonify
from src.phase10_pipeline import HIVEPredictor, P6V2

app = Flask(__name__)

# ── Load predictor once at startup ────────────────────────────────────
predictor = HIVEPredictor(seed=42)
predictor.load()

# ── Complaint list for dropdown ───────────────────────────────────────
_gi = json.load(open(f"{P6V2}/group_info.json"))
TEST_COMPLAINTS = [d["complaint_id"] if isinstance(d,dict) else d
                   for d in _gi["test"]]

@app.route("/")
def index():
    return render_template("index.html",
                            complaints=TEST_COMPLAINTS[:100])

@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    cid  = body.get("complaint_id","").strip()
    if not cid:
        return jsonify({"error": "complaint_id required"}), 400
    try:
        result = predictor.predict(cid, top_n=5, seed=42)
        return jsonify(result)
    except AssertionError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/complaints")
def complaints():
    return jsonify({"complaints": TEST_COMPLAINTS[:100]})

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  HIVE-Predict Demo Server")
    print("  Open: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
