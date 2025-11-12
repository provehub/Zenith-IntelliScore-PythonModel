# pip install fastapi uvicorn "pydantic>=2" joblib numpy pandas shap scikit-learn
import os, time, json
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd
import shap

# -------- Config
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
EXPLAINER_PATH = os.getenv("EXPLAINER_PATH", "explainer.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "feature_names.pkl")
SCORE_SCALE = int(os.getenv("SCORE_SCALE", "1000"))
LOW_PD = float(os.getenv("LOW_PD", "0.07"))
MED_PD = float(os.getenv("MED_PD", "0.15"))
HI_PD = float(os.getenv("HI_PD", "0.30"))

# -------- Load artifacts once
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
app = FastAPI(title="Zenith IntelliScore API")

model = joblib.load("model.pkl")
explainer = joblib.load("explainer.pkl")
feature_names = joblib.load("feature_names.pkl")

class ScoreRequest(BaseModel):
    tel_topups_cnt_90d: int
    tel_inactive_days_90d: int
    util_late_cnt_90d: int
    pos_txn_cnt_90d: int
    pos_volume_90d_ngn: float
    salary_inflow_90d_ngn: float
    net_inflow_90d_ngn: float
    age_years: int

def band(pd_prob):
    if pd_prob < 0.07: return "A"
    if pd_prob < 0.15: return "B"
    if pd_prob < 0.30: return "C"
    return "D"

@app.post("/score")
def score(req: ScoreRequest):
    # Align to training feature order
    row = pd.DataFrame([{k: getattr(req, k) for k in feature_names}], columns=feature_names)

    proba = float(model.predict_proba(row)[0,1])  # PD (0..1)
    credit_trust_score = int((1 - proba) * 1000)  # 0..1000
    risk_band = band(proba)

    # SHAP values for transparency
    shap_vals = explainer.shap_values(row)
    # For GBC, shap_values is array-like; take first row
    sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    contribs = []
    for fname, val, s in zip(feature_names, row.iloc[0].tolist(), sv[0].tolist()):
        contribs.append({"feature": fname, "value": val, "impact": float(s)})
    # Top 5 absolute impacts
    top_impacts = sorted(contribs, key=lambda x: abs(x["impact"]), reverse=True)[:5]

    return {
        "credit_trust_score": credit_trust_score,
        "pd_90d": proba,
        "risk_band": risk_band,
        "top_feature_impacts": top_impacts
    }
