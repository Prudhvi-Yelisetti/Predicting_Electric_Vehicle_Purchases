"""
CatBoost 5-fold CV, Optuna-tuned params, on the NEW feature set:
existing preprocessed columns (income digit decomposition + frequency
encoding) PLUS nested/multi-resolution OOF target encoding of income
and commute, PLUS per-value frequency/lift/novelty vs the original
~10k-row dataset (Experiment 010).

XGBoost confirmed this feature set on 2026-09-20: OOF 0.944147 ->
0.945733 (+0.001586), already above our previous best CatBoost OOF of
0.945243. This script checks whether CatBoost (our stronger model)
gains similarly or more.

Loads Data/train_nested_te.csv / Data/test_nested_te.csv - already
built by nested_te_experiment.py, no feature engineering here.

Run directly (CPU-only CatBoost, ~10-15 min, same as before):
    cd ~/Hackathons/Predicting_Electric_Vehicle_Purchases
    .venv/bin/python run_catboost_nested_te.py
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

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(train, y))

X = train.drop(columns=[TARGET, "id"]).copy()
Xt = test.drop(columns=["id"]).copy()
for c in cat_cols:
    X[c] = X[c].astype(str)
    Xt[c] = Xt[c].astype(str)

print(f"Feature matrix: {X.shape}, test: {Xt.shape}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
new_cols = [c for c in X.columns if c not in cat_cols and c not in
            ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
             "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
             "Environmental_Concern_Level", "Annual_Income_USD_freq", "Daily_Commute_km_freq"]]
print(f"New nested-TE / orig-dataset columns ({len(new_cols)}): {new_cols}")

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
        **BEST_PARAMS,
    )
    model.fit(X_tr, y_tr, cat_features=cat_cols, eval_set=(X_va, y_va))

    va_pred = model.predict_proba(X_va)[:, 1]
    oof_pred[va_idx] = va_pred
    fold_auc = roc_auc_score(y_va, va_pred)
    fold_aucs.append(fold_auc)
    test_preds[fold] = model.predict_proba(Xt)[:, 1]

    np.save(f"{BASE}/tmp_results/cb_nestedte_oof_partial.npy", oof_pred)
    np.save(f"{BASE}/tmp_results/cb_nestedte_test_partial.npy", test_preds)

    print(f"Fold {fold}: AUC={fold_auc:.6f}  best_iter={model.get_best_iteration()}  "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

oof_auc = roc_auc_score(y, oof_pred)
print(f"\nMean fold AUC: {np.mean(fold_aucs):.6f}  Std: {np.std(fold_aucs):.6f}")
print(f"OOF AUC: {oof_auc:.6f}")
print(f"(For comparison: previous best CatBoost-tuned OOF was 0.945243, "
      f"LB 0.94522; XGBoost confirmed this feature set at OOF 0.945733)")

test_pred_avg = test_preds.mean(axis=0)
submission = pd.DataFrame({"id": test["id"], TARGET: test_pred_avg})
out_path = f"{BASE}/submissions/catboost_5fold_nested_te.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission written to: {out_path}")
print(submission.head())
