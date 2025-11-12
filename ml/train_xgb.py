# pip install xgboost scikit-learn joblib shap
import joblib, shap, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

df = pd.read_csv("features_snapshot.csv")  # your real export
y = df.pop("label_default_90d").astype(int)
X = df.copy()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.8, random_state=7, n_jobs=4
    ))
])

# Calibrate probabilities
cal = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
cal.fit(X_tr, y_tr)
pred = cal.predict_proba(X_te)[:,1]
auc = roc_auc_score(y_te, pred)
print("Valid AUC:", round(auc, 3))

joblib.dump(cal, "model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

# SHAP explainer (use TreeExplainer on underlying XGB if needed)
base_model = cal.base_estimator.named_steps["clf"]
explainer = shap.TreeExplainer(base_model)
joblib.dump(explainer, "explainer.pkl")
