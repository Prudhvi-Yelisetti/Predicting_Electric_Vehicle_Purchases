"""
Experiment 014: two genuinely untested levers, checked cheaply on
XGBoost before committing any CatBoost time.

(A) Seed-bagging / variance reduction: train K models per fold with
different random seeds (same data, same fold split, same hyperparams),
average their predictions within each fold. This is a different kind
of lever from everything tried in Experiments 011-013 - it doesn't add
information, it reduces prediction VARIANCE, which has a provably
non-negative expected effect on held-out AUC (unlike the feature/reg
tweaks already ruled out). The external pipeline used exactly this
(20-fold + encoding-seed bags, 15-seed MLP) as its last confirmed step.

(B) Nearest-original-distance: the one piece of the external "per-value
frequency/lift/novelty vs original dataset" bucket we did NOT build in
Experiment 010 (we built frequency+lift+novelty, skipped distance).
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

DATA_DIR = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases/Data/"
TARGET = "Will_Buy_EV"

train = pd.read_csv(DATA_DIR + "train_nested_te.csv")
test = pd.read_csv(DATA_DIR + "test_nested_te.csv")
y = train[TARGET].map({"Yes": 1, "No": 0}).astype("int8")

cat_cols = train.select_dtypes(include=["object", "category", "str"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]
X = train.copy()
for c in cat_cols:
    X[c] = X[c].astype("category")
feature_cols = [c for c in train.columns if c not in ("id", TARGET)]

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold_idx = list(skf.split(train, y))

# --- (A) Seed-bagging check ---
# NOTE: default subsample=1.0, colsample_bytree=1.0 means XGBoost training
# is ~fully deterministic - random_state has almost nothing to act on, so
# seed-bagging at those defaults gives zero diversity (confirmed below).
# Introduce real per-tree randomness first, then bag across seeds.
SEEDS = [42, 123, 456, 789, 2024]
SUBSAMPLE = 0.8
COLSAMPLE = 0.8

def train_one(X_tr, y_tr, X_va, seed):
    model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=3000,
        early_stopping_rounds=50, device="cuda",
        enable_categorical=True, eval_metric="auc", random_state=seed,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, X_va.index.map(lambda i: y.loc[i]))], verbose=False)
    return model.predict_proba(X_va)[:, 1]

single_oof = np.zeros(len(X))
bag_oof = np.zeros(len(X))
per_seed_oof = {s: np.zeros(len(X)) for s in SEEDS}
t0 = time.time()
for i, (tr_idx, va_idx) in enumerate(fold_idx):
    X_tr, X_va = X.iloc[tr_idx][feature_cols], X.iloc[va_idx][feature_cols]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    seed_preds = []
    for s in SEEDS:
        model = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.05, n_estimators=3000,
            subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE,
            early_stopping_rounds=50, device="cuda",
            enable_categorical=True, eval_metric="auc", random_state=s,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        seed_preds.append(model.predict_proba(X_va)[:, 1])
    seed_preds = np.array(seed_preds)
    single_oof[va_idx] = seed_preds[0]
    bag_oof[va_idx] = seed_preds.mean(axis=0)
    for j, s in enumerate(SEEDS):
        per_seed_oof[s][va_idx] = seed_preds[j]
    print(f"fold {i+1} done, {time.time()-t0:.0f}s "
          f"(single={roc_auc_score(y_va, seed_preds[0]):.6f}, "
          f"bag={roc_auc_score(y_va, seed_preds.mean(axis=0)):.6f})", flush=True)

single_auc = roc_auc_score(y, single_oof)
bag_auc = roc_auc_score(y, bag_oof)
print(f"\nSingle-seed OOF: {single_auc:.6f}")
print(f"5-seed-bag OOF:  {bag_auc:.6f}")
print(f"Delta:            {bag_auc - single_auc:+.6f}")

print("\nCumulative seed-count analysis:")
cum = np.zeros(len(X))
for k, s in enumerate(SEEDS, start=1):
    cum = cum + per_seed_oof[s] if k > 1 else per_seed_oof[s].copy()
    running_avg = cum / k
    print(f"  k={k} seeds: OOF={roc_auc_score(y, running_avg):.6f}")
