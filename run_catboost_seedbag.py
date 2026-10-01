"""
Experiment 015: 3-seed bagging for CatBoost on the Experiment 010
feature set. Confirmed significant on XGBoost (z=5.68 for the combined
subsample+bagging effect; 3 seeds captures ~99% of the 5-seed gain, so
3 is used here to keep runtime reasonable - ~3x a single run).

CatBoost's already-tuned config has bagging_temperature=1.0470... (a
Bayesian bootstrap - genuine per-run stochasticity), so unlike
XGBoost's un-subsampled default, no config change is needed here -
just run the same tuned config at 3 different random seeds and average.

CPU-only, ~3x the Experiment 010 runtime (~57-60 min total). Run
directly in your own terminal, not through the remote connection:
    cd ~/Hackathons/Predicting_Electric_Vehicle_Purchases
    .venv/bin/python run_catboost_seedbag.py
Saves progress after every (seed, fold) pair so a drop mid-run loses
at most one fit's worth of time.
"""
import pandas as pd
import numpy as np
import time

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

BEST_PARAMS = dict(
    learning_rate=0.04304867132714463,
    depth=5,
    l2_leaf_reg=2.998747815633837,
    bagging_temperature=1.0470518167824743,
    random_strength=2.685717108403337,
    border_count=196,
)

SEEDS = [42, 123, 456]
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(train, y))

X = train.drop(columns=[TARGET, "id"]).copy()
Xt = test.drop(columns=["id"]).copy()
for c in cat_cols:
    X[c] = X[c].astype(str)
    Xt[c] = Xt[c].astype(str)

print(f"Feature matrix: {X.shape}, test: {Xt.shape}, seeds: {SEEDS}")

per_seed_oof = {s: np.zeros(len(X)) for s in SEEDS}
per_seed_test = {s: np.zeros((N_SPLITS, len(Xt))) for s in SEEDS}
t0 = time.time()

for s in SEEDS:
    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = CatBoostClassifier(
            iterations=40000,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=s,
            verbose=False,
            early_stopping_rounds=150,
            **BEST_PARAMS,
        )
        model.fit(X_tr, y_tr, cat_features=cat_cols, eval_set=(X_va, y_va))

        va_pred = model.predict_proba(X_va)[:, 1]
        per_seed_oof[s][va_idx] = va_pred
        fold_auc = roc_auc_score(y_va, va_pred)
        per_seed_test[s][fold] = model.predict_proba(Xt)[:, 1]

        np.save(f"{BASE}/tmp_results/cb_seedbag_oof_seed{s}.npy", per_seed_oof[s])
        np.save(f"{BASE}/tmp_results/cb_seedbag_test_seed{s}.npy", per_seed_test[s])

        elapsed = time.time() - t0
        print(f"seed={s} fold={fold}: AUC={fold_auc:.6f}  best_iter={model.get_best_iteration()}  "
              f"elapsed={elapsed:.0f}s", flush=True)

    seed_oof_auc = roc_auc_score(y, per_seed_oof[s])
    print(f"--- seed {s} complete: OOF={seed_oof_auc:.6f} ---\n", flush=True)

# --- Cumulative seed-bag analysis + submission ---
print("Cumulative seed-count OOF:")
cum_oof = np.zeros(len(X))
cum_test = np.zeros((N_SPLITS, len(Xt)))
for k, s in enumerate(SEEDS, start=1):
    cum_oof = cum_oof + per_seed_oof[s]
    cum_test = cum_test + per_seed_test[s]
    running_oof_auc = roc_auc_score(y, cum_oof / k)
    print(f"  k={k} seeds ({SEEDS[:k]}): OOF={running_oof_auc:.6f}")

final_oof = cum_oof / len(SEEDS)
final_oof_auc = roc_auc_score(y, final_oof)
print(f"\nFinal {len(SEEDS)}-seed-bag OOF: {final_oof_auc:.6f}")
print(f"(For comparison: Experiment 010 single-seed CatBoost OOF was 0.945943, LB 0.94610)")

final_test_pred = (cum_test / len(SEEDS)).mean(axis=0)
submission = pd.DataFrame({"id": test["id"], TARGET: final_test_pred})
out_path = f"{BASE}/submissions/catboost_5fold_seedbag3.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission written to: {out_path}")
print(submission.head())
