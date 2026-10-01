"""
Stage 2 only: confirm the Optuna-tuned CatBoost params on the real
5-fold CV over full preprocessed data. Picks up best_params.json from
Stage 1 (already run, best subsample-holdout AUC 0.942673) so that
work isn't repeated.

RUN THIS DIRECTLY IN YOUR OWN TERMINAL, not through a remote session -
the last Stage 2 attempt died silently partway through with no error,
almost certainly because it was tied to a remote connection that
dropped (same issue noted in the vault for prior CatBoost runs).

    cd ~/Hackathons/Predicting_Electric_Vehicle_Purchases
    .venv/bin/python -u optuna_stage2_confirm.py

Progress is saved incrementally to tmp_results/ per fold, so even if
this also gets interrupted, completed folds aren't lost - though for
a trustworthy OOF AUC it needs to run all 5 folds without a restart
(restarting would overwrite oof_pred with zeros for already-done
folds unless you resume manually - just let it run uninterrupted).
"""
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases"
BASELINE_OOF = 0.945198

with open(f"{BASE}/tmp_results/optuna_best_params.json") as f:
    best_params = json.load(f)
print(f"Loaded Stage 1 best params: {best_params}")

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
    X_full[c] = X_full[c].astype(str)
    Xt_full[c] = Xt_full[c].astype(str)
print(f"Full data: {X_full.shape}, test: {Xt_full.shape}")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(X_full, y_full))

oof_pred = np.zeros(len(X_full))
test_preds = np.zeros((N_SPLITS, len(Xt_full)))
fold_aucs = []
t1 = time.time()

final_params = {
    **best_params,
    "iterations": 2000,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": False,
}

for fold, (tr_idx, va_idx) in enumerate(folds):
    X_tr2, X_va2 = X_full.iloc[tr_idx], X_full.iloc[va_idx]
    y_tr2, y_va2 = y_full.iloc[tr_idx], y_full.iloc[va_idx]

    model = CatBoostClassifier(**final_params)
    model.fit(X_tr2, y_tr2, cat_features=cat_cols, eval_set=(X_va2, y_va2),
              early_stopping_rounds=50)

    va_pred = model.predict_proba(X_va2)[:, 1]
    oof_pred[va_idx] = va_pred
    fold_auc = roc_auc_score(y_va2, va_pred)
    fold_aucs.append(fold_auc)
    test_preds[fold] = model.predict_proba(Xt_full)[:, 1]

    np.save(f"{BASE}/tmp_results/cb_tuned_oof.npy", oof_pred)
    np.save(f"{BASE}/tmp_results/cb_tuned_test.npy", test_preds)
    print(f"  Fold {fold}: AUC={fold_auc:.6f}  best_iter={model.get_best_iteration()}  "
          f"elapsed={time.time()-t1:.0f}s", flush=True)

oof_auc = roc_auc_score(y_full, oof_pred)
print(f"\nStage 2 mean fold AUC: {np.mean(fold_aucs):.6f}  Std: {np.std(fold_aucs):.6f}")
print(f"Stage 2 OOF AUC (tuned): {oof_auc:.6f}")
print(f"Baseline (Experiment 007, untuned): {BASELINE_OOF:.6f}")
print(f"Delta: {oof_auc - BASELINE_OOF:+.6f}")

if oof_auc > BASELINE_OOF:
    test_pred_avg = test_preds.mean(axis=0)
    submission = pd.DataFrame({"id": test["id"], TARGET: test_pred_avg})
    out_path = f"{BASE}/submissions/catboost_5fold_tuned.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nIMPROVED - submission written to: {out_path}")
    print(submission.head())
else:
    print("\nTuned params did NOT beat the untuned baseline - "
          "no submission file written.")

with open(f"{BASE}/tmp_results/optuna_final_summary.json", "w") as f:
    json.dump({
        "stage2_oof_auc": oof_auc,
        "stage2_fold_aucs": fold_aucs,
        "baseline_oof": BASELINE_OOF,
        "best_params": best_params,
    }, f, indent=2)
print("\nSaved to tmp_results/optuna_final_summary.json")
