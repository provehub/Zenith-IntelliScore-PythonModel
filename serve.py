# pip install fastapi uvicorn pandas numpy shap joblib pydantic
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd

app = FastAPI(title="IntelliScore API")

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
