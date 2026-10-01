"""
Experiment 014b: nearest-original-distance feature - the one piece of
the external "per-value frequency/lift/novelty vs original dataset"
bucket not built in Experiment 010 (we built frequency+lift+novelty,
skipped distance). Cheap vectorized nearest-neighbor lookup via sorted
array + searchsorted. Tested on top of the Experiment 010 feature set,
XGBoost, default (non-subsampled) config to isolate this one effect.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import time

DATA_DIR = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases/Data/"
TARGET = "Will_Buy_EV"

train = pd.read_csv(DATA_DIR + "train_nested_te.csv")
test = pd.read_csv(DATA_DIR + "test_nested_te.csv")
orig = pd.read_csv(DATA_DIR + "EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = train[TARGET].map({"Yes": 1, "No": 0}).astype("int8")

def nearest_dist(values, ref_sorted):
    idx = np.searchsorted(ref_sorted, values)
    idx = np.clip(idx, 1, len(ref_sorted) - 1)
    left = ref_sorted[idx - 1]
    right = ref_sorted[idx]
    return np.minimum(np.abs(values - left), np.abs(values - right))

inc_ref = np.sort(orig["Annual_Income_USD"].unique())
com_ref = np.sort(orig["Daily_Commute_km"].unique())

train["inc_dist_orig"] = nearest_dist(train["Annual_Income_USD"].values, inc_ref)
train["com_dist_orig"] = nearest_dist(train["Daily_Commute_km"].values, com_ref)
test["inc_dist_orig"] = nearest_dist(test["Annual_Income_USD"].values, inc_ref)
test["com_dist_orig"] = nearest_dist(test["Daily_Commute_km"].values, com_ref)
print("inc_dist_orig stats:", train["inc_dist_orig"].describe().to_dict())

cat_cols = train.select_dtypes(include=["object", "category", "str"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]
X = train.copy()
for c in cat_cols:
    X[c] = X[c].astype("category")

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold_idx = list(skf.split(train, y))

base_feature_cols = [c for c in train.columns if c not in ("id", TARGET, "inc_dist_orig", "com_dist_orig")]
new_feature_cols = base_feature_cols + ["inc_dist_orig", "com_dist_orig"]

def run_cv(feature_cols, label):
    oof = np.zeros(len(X))
    aucs = []
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
        pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = pred
        fold_auc = roc_auc_score(y_va, pred)
        aucs.append(fold_auc)
        print(f"  [{label}] fold {i+1}: {fold_auc:.6f}")
    oof_auc = roc_auc_score(y, oof)
    print(f"[{label}] OOF={oof_auc:.6f} time={time.time()-t0:.0f}s")
    return oof_auc, aucs

base_auc, base_folds = run_cv(base_feature_cols, "baseline (Exp. 010, no distance)")
dist_auc, dist_folds = run_cv(new_feature_cols, "+ nearest-original-distance")

print("=" * 60)
print(f"Baseline OOF: {base_auc:.6f}")
print(f"+ distance:   {dist_auc:.6f}")
print(f"Delta:        {dist_auc - base_auc:+.6f}")

deltas = np.array(dist_folds) - np.array(base_folds)
se = deltas.std(ddof=1) / np.sqrt(5)
print(f"Fold deltas: {deltas}")
print(f"z = {deltas.mean()/se:.3f}")
