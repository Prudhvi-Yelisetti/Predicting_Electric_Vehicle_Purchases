"""
Phase 4 continued - isotonic calibration before stacking. The raw-
probability stack (stack_phase4.py) failed to beat the single best
model (CatBoost tuned, 0.945243) because the three models are too
correlated (0.994-0.996) on the preprocessed feature set. This tries
putting each model's OOF predictions on a calibrated probability scale
before combining, in case that's enough to let the meta-learner extract
real complementary signal instead of just rediscovering CatBoost.

Isotonic regression is fit with the SAME 5-fold CV as the base models
(fit on train-fold OOF -> y, apply to valid-fold OOF) to keep the
calibrated OOF honest - not fit on the same rows it's evaluated on.
For the final test-set blend, isotonic is refit on ALL OOF data per
model and applied to that model's test predictions.
"""
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases"
CB_OOF_BASELINE = 0.945243
RAW_STACK_OOF = 0.945239

train = pd.read_csv(f"{BASE}/Data/train_preprocessed.csv")
test = pd.read_csv(f"{BASE}/Data/test_preprocessed.csv")
TARGET = "Will_Buy_EV"
y_full = train[TARGET].map({"No": 0, "Yes": 1}).to_numpy()

cb_oof = np.load(f"{BASE}/tmp_results/cb_tuned_oof.npy")
cb_test = np.load(f"{BASE}/tmp_results/cb_tuned_test.npy").mean(axis=0)
xgb_oof = np.load(f"{BASE}/tmp_results/xgb_oof.npy")
xgb_test = np.load(f"{BASE}/tmp_results/xgb_test.npy")
lgb_oof = np.load(f"{BASE}/tmp_results/lgb_oof.npy")
lgb_test = np.load(f"{BASE}/tmp_results/lgb_test.npy")

oof_cols = {"cb": cb_oof, "xgb": xgb_oof, "lgb": lgb_oof}
test_cols = {"cb": cb_test, "xgb": xgb_test, "lgb": lgb_test}

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(cb_oof.reshape(-1, 1), y_full))

# --- Step 1: per-model isotonic calibration of OOF, via CV to stay honest ---
calibrated_oof = {name: np.zeros(len(y_full)) for name in oof_cols}

for name, oof in oof_cols.items():
    for tr_idx, va_idx in folds:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof[tr_idx], y_full[tr_idx])
        calibrated_oof[name][va_idx] = iso.predict(oof[va_idx])

print("Individual OOF AUC, raw vs. isotonic-calibrated:")
for name, oof in oof_cols.items():
    raw_auc = roc_auc_score(y_full, oof)
    cal_auc = roc_auc_score(y_full, calibrated_oof[name])
    print(f"  {name}: raw={raw_auc:.6f}  calibrated={cal_auc:.6f}  delta={cal_auc-raw_auc:+.6f}")

# --- Step 2: calibrated stack, same meta-learner CV as before ---
cal_oof_matrix = np.column_stack([calibrated_oof["cb"], calibrated_oof["xgb"], calibrated_oof["lgb"]])

meta_oof = np.zeros(len(y_full))
coefs = []
for tr_idx, va_idx in folds:
    meta = LogisticRegression(C=1.0, random_state=42)
    meta.fit(cal_oof_matrix[tr_idx], y_full[tr_idx])
    meta_oof[va_idx] = meta.predict_proba(cal_oof_matrix[va_idx])[:, 1]
    coefs.append(meta.coef_[0])

meta_auc = roc_auc_score(y_full, meta_oof)
mean_coefs = np.mean(coefs, axis=0)
print(f"\nCalibrated stack (LR) 5-fold OOF AUC: {meta_auc:.6f}")
print(f"Mean fold coefficients (CB, XGB, LGB): {mean_coefs}")
print(f"Raw-probability stack (prior attempt): {RAW_STACK_OOF:.6f}")
print(f"Best single model (tuned CatBoost): {CB_OOF_BASELINE:.6f}")
print(f"Calibrated stack delta vs best single model: {meta_auc - CB_OOF_BASELINE:+.6f}")
print(f"Calibrated stack delta vs raw stack: {meta_auc - RAW_STACK_OOF:+.6f}")

# --- Step 3: simple average of calibrated columns, as a sanity check ---
avg_cal_auc = roc_auc_score(y_full, cal_oof_matrix.mean(axis=1))
print(f"\nSimple average of calibrated OOF columns: {avg_cal_auc:.6f}")

# --- Step 4: refit isotonic on ALL OOF data per model, apply to test columns ---
calibrated_test = {}
for name, oof in oof_cols.items():
    iso_full = IsotonicRegression(out_of_bounds="clip")
    iso_full.fit(oof, y_full)
    calibrated_test[name] = iso_full.predict(test_cols[name])

cal_test_matrix = np.column_stack([calibrated_test["cb"], calibrated_test["xgb"], calibrated_test["lgb"]])

final_meta = LogisticRegression(C=1.0, random_state=42)
final_meta.fit(cal_oof_matrix, y_full)
final_test_pred = final_meta.predict_proba(cal_test_matrix)[:, 1]
print(f"\nFinal calibrated-stack meta-learner coefficients (CB, XGB, LGB): {final_meta.coef_[0]}")

best_so_far = max(CB_OOF_BASELINE, RAW_STACK_OOF)
if meta_auc > best_so_far:
    submission = pd.DataFrame({"id": test["id"], TARGET: final_test_pred})
    out_path = f"{BASE}/submissions/stack_calibrated_lr.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nIMPROVED - submission written to: {out_path}")
    print(submission.head())
else:
    print(f"\nCalibrated stack did NOT beat the best result so far ({best_so_far:.6f}) - "
          "no submission written.")

with open(f"{BASE}/tmp_results/calibrated_stack_summary.json", "w") as f:
    json.dump({
        "individual_calibrated_oof": {
            name: float(roc_auc_score(y_full, calibrated_oof[name])) for name in oof_cols
        },
        "calibrated_stack_oof_auc": float(meta_auc),
        "raw_stack_oof_auc": RAW_STACK_OOF,
        "best_single_model_oof": CB_OOF_BASELINE,
        "avg_calibrated_oof_auc": float(avg_cal_auc),
        "final_meta_coefs": final_meta.coef_[0].tolist(),
    }, f, indent=2)
print("\nSaved tmp_results/calibrated_stack_summary.json")
