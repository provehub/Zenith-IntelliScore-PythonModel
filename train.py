# pip install scikit-learn pandas numpy shap joblib
import numpy as np, pandas as pd, shap, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

# ---- 1) Create a quick synthetic dataset (replace with your features_snapshot later)
np.random.seed(42)
N = 800
df = pd.DataFrame({
    "tel_topups_cnt_90d": np.random.poisson(10, N),
    "tel_inactive_days_90d": np.random.randint(0, 20, N),
    "util_late_cnt_90d": np.random.randint(0, 6, N),
    "pos_txn_cnt_90d": np.random.poisson(30, N),
    "pos_volume_90d_ngn": np.random.normal(250000, 80000, N).clip(20000, 800000),
    "salary_inflow_90d_ngn": np.random.normal(300000, 120000, N).clip(0, 1000000),
    "net_inflow_90d_ngn": np.random.normal(120000, 150000, N).clip(-200000, 800000),
    "age_years": np.random.randint(21, 60, N),
})

# A simple synthetic target: higher late counts & inactivity -> higher default; higher inflows/pos -> lower default
logit = (
    0.02*df["tel_inactive_days_90d"] +
    0.35*df["util_late_cnt_90d"] -
    0.000004*df["pos_volume_90d_ngn"] -
    0.000003*df["salary_inflow_90d_ngn"] -
    0.000002*df["net_inflow_90d_ngn"] -
    0.01*df["pos_txn_cnt_90d"] +
    0.01*np.random.randn(N)
)
p = 1/(1+np.exp(-logit))
df["default_90d"] = (np.random.rand(N) < p).astype(int)

X = df.drop(columns=["default_90d"])
y = df["default_90d"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

# ---- 2) Train model
model = GradientBoostingClassifier(random_state=7)
model.fit(X_train, y_train)

# ---- 3) Eval
auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:,1])
print("Valid AUC:", round(auc, 3))

# ---- 4) Save artifacts
joblib.dump(model, "model.pkl")

# SHAP explainer for tree-based model
explainer = shap.TreeExplainer(model)
joblib.dump(explainer, "explainer.pkl")

# Save feature order for runtime alignment
joblib.dump(list(X.columns), "feature_names.pkl")
print("Artifacts saved: model.pkl, explainer.pkl, feature_names.pkl")
