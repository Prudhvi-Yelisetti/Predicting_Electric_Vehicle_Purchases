"""
Heavier CatBoost-NATIVE regularization check - the other untried,
lowest-priority item from the execution plan (previously only screened
on XGBoost, where it was non-significant, z~1.71). Inspired by the
same unrun Other_Solutions template that motivated the XGBoost check:
l2_leaf_reg=31, min_data_in_leaf=448 (vs. our tuned l2_leaf_reg~3,
no min_data_in_leaf floor), same depth=5.

Baseline is NOT recomputed - reuses the existing single-seed tuned
nested-TE OOF (tmp_results/cb_nestedte_oof_partial.npy, OOF 0.945943,
the exact same BEST_PARAMS/folds as run_catboost_nested_te.py) so this
is a clean apples-to-apples comparison for one new CatBoost run
(~10-15 min CPU-only) instead of two.

Significance: row-level paired bootstrap (2000 resamples) on the OOF
predictions, z = mean(delta)/std(delta) - consistent with the
project's noise floor (~SE 0.00003) and adopted z>=3 acceptance bar.
"""
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

BASE = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases"

train = pd.read_csv(f"{BASE}/Data/train_nested_te.csv")
test = pd.read_csv(f"{BASE}/Data/test_nested_te.csv")
TARGET = "Will_Buy_EV"
y = train[TARGET].map({"No": 0, "Yes": 1})

digit_cols = [f"inc_d{i}" for i in range(6)]
base_cat_cols = ["Gender", "City_Type", "Current_Car_Type",
                 "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
cat_cols = base_cat_cols + digit_cols

# Same tuned base config, just the two regularization knobs swapped for
# the heavier Other_Solutions-inspired values.
HEAVY_PARAMS = dict(
    learning_rate=0.04304867132714463,
    depth=5,
    l2_leaf_reg=31.0,
    min_data_in_leaf=448,
    bagging_temperature=1.0470518167824743,
    random_strength=2.685717108403337,
    border_count=196,
)

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(train, y))

X = train.drop(columns=[TARGET, "id"]).copy()
Xt = test.drop(columns=["id"]).copy()
for c in cat_cols:
    X[c] = X[c].astype(str)
    Xt[c] = Xt[c].astype(str)

print(f"Feature matrix: {X.shape}, test: {Xt.shape}")
print(f"Heavy-reg params: {HEAVY_PARAMS}", flush=True)

oof_pred = np.zeros(len(X))
test_preds = np.zeros((N_SPLITS, len(Xt)))
fold_aucs = []
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(folds):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = CatBoostClassifier(
        iterations=40000,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        early_stopping_rounds=150,
        **HEAVY_PARAMS,
    )
    model.fit(X_tr, y_tr, cat_features=cat_cols, eval_set=(X_va, y_va))

    va_pred = model.predict_proba(X_va)[:, 1]
    oof_pred[va_idx] = va_pred
    fold_auc = roc_auc_score(y_va, va_pred)
    fold_aucs.append(fold_auc)
    test_preds[fold] = model.predict_proba(Xt)[:, 1]

    np.save(f"{BASE}/tmp_results/cb_heavyreg_oof_partial.npy", oof_pred)
    np.save(f"{BASE}/tmp_results/cb_heavyreg_test_partial.npy", test_preds)

    print(f"Fold {fold}: AUC={fold_auc:.6f}  best_iter={model.get_best_iteration()}  "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

heavy_oof_auc = roc_auc_score(y, oof_pred)
print(f"\nMean fold AUC: {np.mean(fold_aucs):.6f}  Std: {np.std(fold_aucs):.6f}")
print(f"Heavy-reg OOF AUC: {heavy_oof_auc:.6f}", flush=True)

# --- Compare against existing tuned baseline (same folds, same script family) ---
baseline_oof = np.load(f"{BASE}/tmp_results/cb_nestedte_oof_partial.npy")
baseline_auc = roc_auc_score(y, baseline_oof)
print(f"Baseline (tuned, l2=3) OOF AUC: {baseline_auc:.6f}")
print(f"Delta: {heavy_oof_auc - baseline_auc:+.6f}")

# --- Row-level paired bootstrap significance ---
rng = np.random.default_rng(42)
y_arr = y.to_numpy()
n = len(y_arr)
B = 2000
deltas = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n, n)
    auc_heavy = roc_auc_score(y_arr[idx], oof_pred[idx])
    auc_base = roc_auc_score(y_arr[idx], baseline_oof[idx])
    deltas[b] = auc_heavy - auc_base

z = deltas.mean() / deltas.std()
print(f"\nBootstrap ({B} resamples): mean delta={deltas.mean():+.6f}  "
      f"std={deltas.std():.6f}  z={z:.2f}")
print(f"Adopted acceptance bar: z >= 3")
if abs(z) >= 3:
    verdict = "SIGNIFICANT" if z > 0 else "SIGNIFICANTLY WORSE"
else:
    verdict = "NOT significant - reject, do not adopt"
print(f"Verdict: {verdict}")

import json
with open(f"{BASE}/tmp_results/cb_heavyreg_summary.json", "w") as f:
    json.dump({
        "heavy_params": HEAVY_PARAMS,
        "heavy_oof_auc": float(heavy_oof_auc),
        "baseline_oof_auc": float(baseline_auc),
        "delta": float(heavy_oof_auc - baseline_auc),
        "bootstrap_z": float(z),
        "verdict": verdict,
    }, f, indent=2)
print("\nSaved tmp_results/cb_heavyreg_summary.json")
