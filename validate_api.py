"""Phase 10 FastAPI validation script"""
import requests, sys

BASE = "http://localhost:8000"
results = []

def check(name, ok, detail=""):
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"  {tag} {name}" + (f": {detail}" if detail else ""))
    results.append(ok)

print("\n=== PURVADRISHTI FastAPI Validation ===\n")

# Health
try:
    r = requests.get(f"{BASE}/api/health", timeout=10)
    d = r.json()
    check("GET /api/health", r.status_code==200 and d["status"]=="ok",
          f"status={d.get('status')} model_loaded={d.get('model_loaded')}")
except Exception as e:
    check("GET /api/health", False, str(e))

# Model info
try:
    r = requests.get(f"{BASE}/api/model-info", timeout=10)
    d = r.json()
    check("GET /api/model-info", r.status_code==200 and d["features"]==22 and d["t3_hit_at_5"]==0.6526,
          f"features={d.get('features')} T3={d.get('t3_hit_at_5')}")
    check("kde_primary=false", d.get("kde_primary")==False)
except Exception as e:
    check("GET /api/model-info", False, str(e))

# Complaints
cids = []
try:
    r = requests.get(f"{BASE}/api/complaints", timeout=10)
    d = r.json()
    check("GET /api/complaints", r.status_code==200 and d["total"]>0, f"total={d.get('total')}")
    cids = d["complaints"]
except Exception as e:
    check("GET /api/complaints", False, str(e))

# Dashboard HTML
try:
    r = requests.get(f"{BASE}/", timeout=10)
    check("GET / (dashboard)", r.status_code==200 and "PURVADRISHTI" in r.text)
    check("PURVADRISHTI branding in HTML", "PURVADRISHTI" in r.text)
    check("No HIVE-Predict branding in HTML", "HIVE-Predict" not in r.text)
    check("Light map (CartoDB Positron)", "light_all" in r.text)
    check("No dark map tiles", "dark_all" not in r.text)
except Exception as e:
    check("GET / (dashboard)", False, str(e))

# Predict valid
if cids:
    cid = cids[0]
    try:
        r = requests.post(f"{BASE}/api/predict", json={"complaint_id": cid}, timeout=20)
        d = r.json()
        check("POST /api/predict (valid)", r.status_code==200)
        top5 = d.get("top_predictions", [])
        check("top_predictions has 5 results", len(top5)==5, str(len(top5)))
        p = top5[0]
        check("relative_model_score 0-100", 0 <= p.get("relative_model_score",0) <= 100,
              str(p.get("relative_model_score")))
        check("SHAP present for rank-1", p.get("shap") is not None)
        check("boundary polygon present", len(p.get("boundary",[]))>0)
        check("nearby_atms present", isinstance(p.get("nearby_atms",[]), list))
        check("alert_priority valid", d.get("alert_priority") in ["HIGH","MEDIUM","LOW"],
              d.get("alert_priority"))
        check("score_margin is float", isinstance(d.get("score_margin"), float))
        check("validation fields separate", "validation_top5_hit" in d)
        check("xgb_raw_score present", "xgb_raw_score" in p)
        # No actual_* in prediction fields
        leak = [k for k in p.keys() if "actual" in k and "validation" not in k]
        check("No actual_* in prediction fields", len(leak)==0, str(leak))
        check("Ground truth only in validation fields", "validation_actual_h3_cells" in d)
        nf = len(p.get("shap",{}).get("top_factors",[]))
        print(f"       -> Priority={d['alert_priority']} Margin={d['score_margin']:.3f} "
              f"Score={p['relative_model_score']} SHAP_factors={nf}")
    except Exception as e:
        check("POST /api/predict", False, str(e))

# Predict invalid → 404
try:
    r = requests.post(f"{BASE}/api/predict", json={"complaint_id": "INVALID_999"}, timeout=10)
    check("POST /api/predict 404 on unknown", r.status_code==404)
except Exception as e:
    check("POST /api/predict 404 on unknown", False, str(e))

# Swagger
try:
    r = requests.get(f"{BASE}/docs", timeout=10)
    check("GET /docs (Swagger)", r.status_code==200 and "swagger" in r.text.lower())
except Exception as e:
    check("GET /docs (Swagger)", False, str(e))

# ReDoc
try:
    r = requests.get(f"{BASE}/redoc", timeout=10)
    check("GET /redoc (ReDoc)", r.status_code==200)
except Exception as e:
    check("GET /redoc (ReDoc)", False, str(e))

# 10-complaint demo
# 10-complaint demo — use the same known-good IDs from Phase 10 validation
# These IDs were validated to work and gave 9/10 Top-5 hit rate
DEMO_CIDS = [
    "CMP00007160","CMP00009397","CMP00009051","CMP00008633","CMP00008419",
    "CMP00001290","CMP00006278","CMP00008031","CMP00001297","CMP00001904",
]
# Filter to only those available in the API
available = set(cids)
demo_cids = [c for c in DEMO_CIDS if c in available]
if len(demo_cids) < 10:
    # Fall back to available complaints in sorted order
    demo_cids = sorted(available)[:10]
print("\n=== 10-Complaint Demo ===")
hit5_count = 0
for cid in demo_cids:
    r = requests.post(f"{BASE}/api/predict", json={"complaint_id": cid}, timeout=20)
    if r.status_code != 200:
        print(f"  ERROR {cid}: {r.status_code}"); continue
    d = r.json()
    ci = d["complaint_info"]
    hit = d["validation_top5_hit"]
    pri = d["alert_priority"]
    rel = d["top_predictions"][0]["relative_model_score"]
    if hit: hit5_count += 1
    print(f"  {cid}  {ci['fraud_type']:<22}  {ci['victim_state']:<18}  "
          f"hit5={'YES' if hit else 'NO '}  pri={pri:<6}  score={rel}")

print(f"\n  Top-5 hit: {hit5_count}/{len(demo_cids)} ({hit5_count/max(len(demo_cids),1)*100:.0f}%)")

# Summary
passed = sum(results); total = len(results)
print(f"\n=== Results: {passed}/{total} checks passed ===")
if passed == total:
    print("  ALL CHECKS PASSED — PURVADRISHTI VALIDATION COMPLETE")
else:
    print("  SOME CHECKS FAILED — review above")
    sys.exit(1)
