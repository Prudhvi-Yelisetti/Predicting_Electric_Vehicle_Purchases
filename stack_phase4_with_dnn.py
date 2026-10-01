"""
Phase 4, round 2 - stacking WITH the entity-embedding DNN included.
The 3-tree-model stack (stack_phase4.py) failed because CatBoost/
XGBoost/LightGBM were too correlated (0.994-0.996). The DNN
(Experiment 009) showed real diversity - correlation 0.987-0.989 with
the tree models, meaningfully lower - so this checks whether a 4-way
stack can finally extract something a 3-way tree-only stack couldn't.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases"
CB_OOF_BASELINE = 0.945243
PRIOR_3WAY_STACK_OOF = 0.945239

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
dnn_oof = np.load(f"{BASE}/tmp_results/dnn_oof.npy")
dnn_test = np.load(f"{BASE}/tmp_results/dnn_test.npy").mean(axis=0)

print("Individual OOF AUCs:")
for name, oof in [("CatBoost (tuned)", cb_oof), ("XGBoost", xgb_oof),
                   ("LightGBM", lgb_oof), ("DNN", dnn_oof)]:
    print(f"  {name}: {roc_auc_score(y_full, oof):.6f}")

oof_matrix = np.column_stack([cb_oof, xgb_oof, lgb_oof, dnn_oof])
test_matrix = np.column_stack([cb_test, xgb_test, lgb_test, dnn_test])
corr = np.corrcoef(oof_matrix.T)
print(f"\nOOF correlation matrix (CB, XGB, LGB, DNN):\n{corr}")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(oof_matrix, y_full))

meta_oof = np.zeros(len(y_full))
meta_test_preds = np.zeros((N_SPLITS, len(test)))
coefs = []

for fold, (tr_idx, va_idx) in enumerate(folds):
    meta = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42)
    meta.fit(oof_matrix[tr_idx], y_full[tr_idx])
    meta_oof[va_idx] = meta.predict_proba(oof_matrix[va_idx])[:, 1]
    meta_test_preds[fold] = meta.predict_proba(test_matrix)[:, 1]
    coefs.append(meta.coef_[0])

meta_auc = roc_auc_score(y_full, meta_oof)
mean_coefs = np.mean(coefs, axis=0)
print(f"\n4-way meta-learner (LR, L1) 5-fold OOF AUC: {meta_auc:.6f}")
print(f"Mean fold coefficients (CB, XGB, LGB, DNN): {mean_coefs}")
print(f"Best single model (tuned CatBoost) OOF: {CB_OOF_BASELINE:.6f}")
print(f"Prior 3-way tree-only stack OOF: {PRIOR_3WAY_STACK_OOF:.6f}")
print(f"4-way stack delta vs best single model: {meta_auc - CB_OOF_BASELINE:+.6f}")
print(f"4-way stack delta vs 3-way tree stack: {meta_auc - PRIOR_3WAY_STACK_OOF:+.6f}")

final_meta = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42)
final_meta.fit(oof_matrix, y_full)
final_test_pred = final_meta.predict_proba(test_matrix)[:, 1]

best_so_far = max(CB_OOF_BASELINE, PRIOR_3WAY_STACK_OOF)
if meta_auc > best_so_far:
    submission = pd.DataFrame({"id": test["id"], TARGET: final_test_pred})
    out_path = f"{BASE}/submissions/stack_lr_4way_with_dnn.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nIMPROVED - submission written to: {out_path}")
    print(submission.head())
else:
    print(f"\n4-way stack did NOT beat the best result so far ({best_so_far:.6f}) - "
          "no submission written.")

with open(f"{BASE}/tmp_results/stack_phase4_dnn_summary.json", "w") as f:
    json.dump({
        "individual_oof": {
            "catboost_tuned": float(roc_auc_score(y_full, cb_oof)),
            "xgboost": float(roc_auc_score(y_full, xgb_oof)),
            "lightgbm": float(roc_auc_score(y_full, lgb_oof)),
            "dnn": float(roc_auc_score(y_full, dnn_oof)),
        },
        "oof_correlation": corr.tolist(),
        "meta_4way_oof_auc": float(meta_auc),
        "prior_3way_stack_oof": PRIOR_3WAY_STACK_OOF,
        "best_single_model_oof": CB_OOF_BASELINE,
        "final_meta_coefs": final_meta.coef_[0].tolist(),
    }, f, indent=2)
print("\nSaved tmp_results/stack_phase4_dnn_summary.json")
