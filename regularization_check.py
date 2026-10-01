"""
Experiment 013: heavier regularization check, XGBoost only (cheap
screen before deciding whether a CatBoost run is warranted). Inspired
by Other_Solutions/evehicle-stacked-..., whose unrun template used a
CatBoost config with l2_leaf_reg=31, min_data_in_leaf=448, depth=5 -
much heavier than our tuned depth=5/l2=3. Translated to XGBoost's
regularization knobs on the accepted Experiment 010 feature set
(nested TE + orig-dataset features, OOF 0.945733).
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

def run_cv(params, label):
    oof = np.zeros(len(X))
    aucs = []
    t0 = time.time()
    for i, (tr_idx, va_idx) in enumerate(fold_idx):
        X_tr, X_va = X.iloc[tr_idx][feature_cols], X.iloc[va_idx][feature_cols]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = xgb.XGBClassifier(
            n_estimators=3000, early_stopping_rounds=50, device="cuda",
            enable_categorical=True, eval_metric="auc", random_state=42,
            **params,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = pred
        fold_auc = roc_auc_score(y_va, pred)
        aucs.append(fold_auc)
        print(f"  [{label}] fold {i+1}: {fold_auc:.6f}")
    oof_auc = roc_auc_score(y, oof)
    print(f"[{label}] mean={np.mean(aucs):.6f} std={np.std(aucs):.6f} OOF={oof_auc:.6f} time={time.time()-t0:.0f}s")
    return oof_auc

baseline_params = dict(max_depth=6, learning_rate=0.05)
heavy_params = dict(
    max_depth=5, learning_rate=0.03, reg_lambda=15.0, reg_alpha=1.0,
    min_child_weight=50, subsample=0.8, colsample_bytree=0.6,
)

base_auc = run_cv(baseline_params, "baseline (Exp. 010 config)")
heavy_auc = run_cv(heavy_params, "heavier regularization")

print("=" * 60)
print(f"Baseline OOF:  {base_auc:.6f}")
print(f"Heavy-reg OOF: {heavy_auc:.6f}")
print(f"Delta:         {heavy_auc - base_auc:+.6f}")
