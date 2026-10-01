"""
Quick stacking re-check on the NEW (nested-TE + orig-dataset) feature
set. Experiments 008/009 found CatBoost/XGBoost OOF correlation ~0.994-
0.996 on the OLD feature set (too tight to stack). The features changed
materially in Experiment 010 and CatBoost gained less than XGBoost from
them, which could mean the two models now disagree more - worth one
cheap check before writing off stacking again.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

BASE = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases"
TARGET = "Will_Buy_EV"

train = pd.read_csv(f"{BASE}/Data/train_nested_te.csv")
y = train[TARGET].map({"Yes": 1, "No": 0}).astype("int8")

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold_idx = list(skf.split(train, y))

cat_cols = train.select_dtypes(include=["object", "category", "str"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]
X = train.copy()
for c in cat_cols:
    X[c] = X[c].astype("category")
feature_cols = [c for c in train.columns if c not in ("id", TARGET)]

xgb_oof = np.zeros(len(X))
t0 = time.time()
for i, (tr_idx, va_idx) in enumerate(fold_idx):
    X_tr, X_va = X.iloc[tr_idx][feature_cols], X.iloc[va_idx][feature_cols]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=3000,
        early_stopping_rounds=50, device="cuda",
        enable_categorical=True, eval_metric="auc", random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_oof[va_idx] = model.predict_proba(X_va)[:, 1]
    print(f"fold {i+1} done, {time.time()-t0:.0f}s", flush=True)

np.save(f"{BASE}/tmp_results/xgb_nestedte_oof.npy", xgb_oof)
xgb_auc = roc_auc_score(y, xgb_oof)
print(f"XGBoost OOF (new features): {xgb_auc:.6f}")

cb_oof = np.load(f"{BASE}/tmp_results/cb_nestedte_oof_partial.npy")
cb_auc = roc_auc_score(y, cb_oof)
print(f"CatBoost OOF (new features): {cb_auc:.6f}")

corr = np.corrcoef(xgb_oof, cb_oof)[0, 1]
print(f"Pearson correlation of the two OOF vectors: {corr:.6f}")
print("(Experiments 008/009 on the OLD feature set measured 0.994-0.996)")

# --- LR stack OOF check (same protocol as stack_phase4.py) ---
oof_matrix = np.column_stack([cb_oof, xgb_oof])
folds2 = list(skf.split(oof_matrix, y))
meta_oof = np.zeros(len(y))
coefs = []
for tr_idx, va_idx in folds2:
    meta = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42)
    meta.fit(oof_matrix[tr_idx], y.iloc[tr_idx])
    meta_oof[va_idx] = meta.predict_proba(oof_matrix[va_idx])[:, 1]
    coefs.append(meta.coef_[0])
meta_auc = roc_auc_score(y, meta_oof)
avg_auc = roc_auc_score(y, oof_matrix.mean(axis=1))
best_single = max(cb_auc, xgb_auc)

print(f"\nLR stack (CB+XGB) OOF AUC: {meta_auc:.6f}  mean coefs: {np.mean(coefs, axis=0)}")
print(f"Simple average OOF AUC:    {avg_auc:.6f}")
print(f"Best single model OOF:     {best_single:.6f}")
print(f"Stack delta vs best single: {meta_auc - best_single:+.6f}")
