# pip install fastapi uvicorn "pydantic>=2" joblib numpy pandas shap scikit-learn
import os, time, json
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd
import shap

# we have not used shap yet
# -------- Config
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
EXPLAINER_PATH = os.getenv("EXPLAINER_PATH", "explainer.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "feature_names.pkl")
SCORE_SCALE = int(os.getenv("SCORE_SCALE", "1000"))
LOW_PD = float(os.getenv("LOW_PD", "0.07"))
MED_PD = float(os.getenv("MED_PD", "0.15"))
HI_PD = float(os.getenv("HI_PD", "0.30"))

# -------- We our Load artifacts once
model = joblib.load(MODEL_PATH)
explainer = joblib.load(EXPLAINER_PATH)
feature_names: List[str] = joblib.load(FEATURES_PATH)
MODEL_VERSION = os.getenv("MODEL_VERSION", "gbm-2025-11-10")

# -------- Schemas
class FeatureVector(BaseModel):
    tel_topups_cnt_90d: conint(ge=0) = 0
    tel_inactive_days_90d: conint(ge=0) = 0
    util_late_cnt_90d: conint(ge=0) = 0
    pos_txn_cnt_90d: conint(ge=0) = 0
    pos_volume_90d_ngn: confloat(ge=0) = 0.0
    salary_inflow_90d_ngn: confloat(ge=0) = 0.0
    net_inflow_90d_ngn: float = 0.0
    age_years: conint(ge=18, le=90) = 32

class ScoreRequest(BaseModel):
    features: FeatureVector
    application_id: str | None = None
    customer_id: str | None = None

class BatchScoreRequest(BaseModel):
    items: List[ScoreRequest]

class ScoreResponse(BaseModel):
    credit_trust_score: conint(ge=0, le=SCORE_SCALE)
    pd_90d: confloat(ge=0.0, le=1.0)
    risk_band: Literal["A","B","C","D"]
    top_feature_impacts: List[Dict[str, Any]]
    model_version: str
    
# start below Francis

# -------- Helpers
def risk_band(pd_prob: float) -> str:
    if pd_prob < LOW_PD: return "A"
    if pd_prob < MED_PD: return "B"
    if pd_prob < HI_PD:  return "C"
    return "D"

def align_row(f: FeatureVector) -> pd.DataFrame:
    row = pd.DataFrame([{k: getattr(f, k) for k in feature_names}], columns=feature_names)
    # We Handle missing values (simple hackathon default)
    row = row.replace([np.inf, -np.inf], np.nan).fillna(0)
    return row

def explain_row(row: pd.DataFrame):
    # TreeExplainer returns ndarray for tree models
    vals = explainer.shap_values(row)
    shap_row = vals[0] if isinstance(vals, list) else vals[0]
    contribs = [
        {"feature": fname, "value": float(row.iloc[0][fname]), "impact": float(s)}
        for fname, s in zip(feature_names, shap_row.tolist())
    ]
    contribs.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return contribs[:5]

# -------- App
app = FastAPI(title="Zenith IntelliScore API", version="1.0")

@app.middleware("http")
async def timing_mw(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        dur_ms = int((time.time() - start) * 1000)
        path = request.url.path
        # (we can swap to proper logging here)
        if path not in ("/health", "/metadata"):
            print(json.dumps({"path": path, "ms": dur_ms}))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "version": MODEL_VERSION}

@app.get("/metadata")
def metadata():
    return {"model_version": MODEL_VERSION, "features": feature_names, "bands": {"A": LOW_PD, "B": MED_PD, "C": HI_PD}}

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    row = align_row(req.features)
    try:
        proba = float(model.predict_proba(row)[0, 1])
    except Exception as e:
        raise HTTPException(400, f"Scoring failed: {e}")

    score_val = int((1.0 - proba) * SCORE_SCALE)
    impacts = explain_row(row)
    return {
        "credit_trust_score": score_val,
        "pd_90d": proba,
        "risk_band": risk_band(proba),
        "top_feature_impacts": impacts,
        "model_version": MODEL_VERSION
    }

@app.post("/score/batch")
def score_batch(req: BatchScoreRequest):
    results = []
    for item in req.items:
        row = align_row(item.features)
        proba = float(model.predict_proba(row)[0,1])
        results.append({
            "application_id": item.application_id,
            "customer_id": item.customer_id,
            "credit_trust_score": int((1-proba)*SCORE_SCALE),
            "pd_90d": proba,
            "risk_band": risk_band(proba),
            "top_feature_impacts": explain_row(row),
            "model_version": MODEL_VERSION
        })
    return {"count": len(results), "items": results}

@app.exception_handler(Exception)
async def all_ex_handler(_req, exc):
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})