"""
Regenerate XGBoost OOF/test predictions on the preprocessed feature set
(income digits + frequency encoding) - these were never saved to disk
in Experiment 005, only the submission CSV. Needed for Phase 4 stacking
alongside cb_tuned_oof/test.npy and lgb_oof/test.npy.

Same setup as Experiment 005 / Notebooks/05_feature_engineering.ipynb:
XGBClassifier, max_depth=6, learning_rate=0.05, up to 3000 trees,
early_stopping_rounds=50/fold, device="cuda", identical 5-fold
StratifiedKFold (random_state=42) to every other experiment.

Fast on GPU (~17s/fold per Experiment 004/005 notes) - safe to run
through a remote connection, unlike the CatBoost runs.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

BASE = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases"
EXPECTED_OOF = 0.944174

train = pd.read_csv(f"{BASE}/Data/train_preprocessed.csv")
test = pd.read_csv(f"{BASE}/Data/test_preprocessed.csv")
TARGET = "Will_Buy_EV"
y_full = train[TARGET].map({"No": 0, "Yes": 1})

digit_cols = [f"inc_d{i}" for i in range(6)]
base_cat_cols = ["Gender", "City_Type", "Current_Car_Type",
                 "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
cat_cols = base_cat_cols + digit_cols

X_full = train.drop(columns=[TARGET, "id"]).copy()
Xt_full = test.drop(columns=["id"]).copy()
for c in cat_cols:
    X_full[c] = X_full[c].astype("category")
    Xt_full[c] = Xt_full[c].astype("category")
print(f"Full data: {X_full.shape}, test: {Xt_full.shape}")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(X_full, y_full))

oof_pred = np.zeros(len(X_full))
test_preds = np.zeros((N_SPLITS, len(Xt_full)))
fold_aucs = []
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(folds):
    X_tr, X_va = X_full.iloc[tr_idx], X_full.iloc[va_idx]
    y_tr, y_va = y_full.iloc[tr_idx], y_full.iloc[va_idx]

    model = XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=3000,
        early_stopping_rounds=50, eval_metric="auc",
        enable_categorical=True, device="cuda", random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    va_pred = model.predict_proba(X_va)[:, 1]
    oof_pred[va_idx] = va_pred
    fold_auc = roc_auc_score(y_va, va_pred)
    fold_aucs.append(fold_auc)
    test_preds[fold] = model.predict_proba(Xt_full)[:, 1]
    print(f"  Fold {fold}: AUC={fold_auc:.6f}  best_iter={model.best_iteration}  "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

oof_auc = roc_auc_score(y_full, oof_pred)
print(f"\nMean fold AUC: {np.mean(fold_aucs):.6f}  Std: {np.std(fold_aucs):.6f}")
print(f"OOF AUC: {oof_auc:.6f}  (Experiment 005 recorded: {EXPECTED_OOF:.6f}, "
      f"delta {oof_auc - EXPECTED_OOF:+.6f})")

np.save(f"{BASE}/tmp_results/xgb_oof.npy", oof_pred)
np.save(f"{BASE}/tmp_results/xgb_test.npy", test_preds.mean(axis=0))
print("\nSaved tmp_results/xgb_oof.npy and tmp_results/xgb_test.npy")
