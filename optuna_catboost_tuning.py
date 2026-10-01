"""
Optuna hyperparameter search for CatBoost on the preprocessed feature
set (income digit decomposition + frequency encoding) - Phase 3,
tuning the current leader (OOF 0.945198, Experiment 007) first.

Two stages, because CatBoost is CPU-only here and a full 5-fold run
takes ~10-15 min - too slow to use directly as the Optuna objective:

  Stage 1 (search): each trial trains on a ~150k-row stratified
  subsample with a single 85/15 holdout, ~1-2 min/trial. Fast enough
  for 40 trials in well under an hour.

  Stage 2 (confirm): the best params from Stage 1 are re-validated
  with the real 5-fold CV on the FULL preprocessed data (same setup
  as run_catboost_preprocessed.py) - this is the number that actually
  counts and is comparable to the 0.945198 baseline.

Run directly (not through a remote process - CatBoost CPU training is
slow enough that a dropped connection loses progress):
    cd ~/Hackathons/Predicting_Electric_Vehicle_Purchases
    .venv/bin/python optuna_catboost_tuning.py

Writes, incrementally as it goes (so a crash mid-run loses nothing):
    tmp_results/optuna_trials.csv         - every trial's params + AUC
    tmp_results/optuna_best_params.json   - best params found so far
    tmp_results/cb_tuned_oof.npy          - Stage 2 OOF predictions
    tmp_results/cb_tuned_test.npy         - Stage 2 test predictions (5, n)
Only overwrites submissions/catboost_5fold_tuned.csv if Stage 2 OOF
beats the 0.945198 baseline.
"""
import json
import time

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases"
BASELINE_OOF = 0.945198
N_TRIALS = 40
SEARCH_SUBSAMPLE = 150_000

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Load & prep (identical to run_catboost_preprocessed.py) ---
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

# --- Stage 1 setup: stratified subsample + single holdout ---
sub_idx, _ = train_test_split(
    np.arange(len(X_full)), train_size=SEARCH_SUBSAMPLE,
    stratify=y_full, random_state=42,
)
X_sub, y_sub = X_full.iloc[sub_idx], y_full.iloc[sub_idx]
X_tr, X_va, y_tr, y_va = train_test_split(
    X_sub, y_sub, test_size=0.15, stratify=y_sub, random_state=42,
)
print(f"Stage 1 search subsample: {len(X_sub)} rows "
      f"(train {len(X_tr)}, valid {len(X_va)})")

# --- Stage 1: Optuna search ---
trial_log = []


def objective(trial):
    params = {
        "iterations": 1500,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": False,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_tr, y_tr, cat_features=cat_cols, eval_set=(X_va, y_va),
              early_stopping_rounds=40)
    auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])

    row = {**trial.params, "auc": auc, "best_iter": model.get_best_iteration()}
    trial_log.append(row)
    pd.DataFrame(trial_log).to_csv(f"{BASE}/tmp_results/optuna_trials.csv", index=False)
    return auc

print(f"\nStarting Stage 1: {N_TRIALS} trials on subsample...")
t0 = time.time()
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                callbacks=[lambda st, tr: print(
                    f"  trial {tr.number}: auc={tr.value:.6f}  "
                    f"best_so_far={st.best_value:.6f}  "
                    f"elapsed={time.time()-t0:.0f}s", flush=True)])

best_params = study.best_params
print(f"\nStage 1 done in {time.time()-t0:.0f}s. Best subsample-holdout AUC: "
      f"{study.best_value:.6f}")
print(f"Best params: {best_params}")
with open(f"{BASE}/tmp_results/optuna_best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)

# --- Stage 2: confirm on real 5-fold CV, full data, identical folds ---
print("\nStarting Stage 2: 5-fold CV on full data with tuned params...")
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
          "no submission file written. Params/trial log are still "
          "saved in tmp_results/ for inspection.")

with open(f"{BASE}/tmp_results/optuna_final_summary.json", "w") as f:
    json.dump({
        "stage1_holdout_auc": study.best_value,
        "stage2_oof_auc": oof_auc,
        "stage2_fold_aucs": fold_aucs,
        "baseline_oof": BASELINE_OOF,
        "best_params": best_params,
    }, f, indent=2)
print("\nFull summary saved to tmp_results/optuna_final_summary.json")
