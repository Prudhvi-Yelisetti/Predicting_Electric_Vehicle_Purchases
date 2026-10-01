"""
CatBoost 5-fold CV on the preprocessed feature set (income digit
decomposition + frequency encoding of income/commute) - the same
feature set that took XGBoost from 0.941720 to 0.944174 OOF, and
LightGBM from 0.941470 to 0.943663 OOF.

Loads Data/train_preprocessed.csv and Data/test_preprocessed.csv
directly (already contain the engineered columns) - no feature
engineering happens in this script anymore.

Run directly:
    cd ~/Hackathons/Predicting_Electric_Vehicle_Purchases
    .venv/bin/python run_catboost_preprocessed.py

Takes ~10-15 minutes on CPU (CatBoost has no GPU build in this env).
Prints progress per fold and writes a submission CSV at the end.
"""
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

BASE = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases"

train = pd.read_csv(f"{BASE}/Data/train_preprocessed.csv")
test = pd.read_csv(f"{BASE}/Data/test_preprocessed.csv")
TARGET = "Will_Buy_EV"
y = train[TARGET].map({"No": 0, "Yes": 1})

digit_cols = [f"inc_d{i}" for i in range(6)]
base_cat_cols = ["Gender", "City_Type", "Current_Car_Type",
                 "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
cat_cols = base_cat_cols + digit_cols

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

oof_pred = np.zeros(len(X))
test_preds = np.zeros((N_SPLITS, len(Xt)))
fold_aucs = []
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(folds):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=7,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
    )
    model.fit(X_tr, y_tr, cat_features=cat_cols, eval_set=(X_va, y_va), early_stopping_rounds=50)

    va_pred = model.predict_proba(X_va)[:, 1]
    oof_pred[va_idx] = va_pred
    fold_auc = roc_auc_score(y_va, va_pred)
    fold_aucs.append(fold_auc)
    test_preds[fold] = model.predict_proba(Xt)[:, 1]

    # Save progress after every fold, so a crash doesn't lose completed folds
    np.save(f"{BASE}/tmp_results/cb_oof_partial.npy", oof_pred)
    np.save(f"{BASE}/tmp_results/cb_test_partial.npy", test_preds)

    print(f"Fold {fold}: AUC={fold_auc:.6f}  best_iter={model.get_best_iteration()}  "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

oof_auc = roc_auc_score(y, oof_pred)
print(f"\nMean fold AUC: {np.mean(fold_aucs):.6f}  Std: {np.std(fold_aucs):.6f}")
print(f"OOF AUC: {oof_auc:.6f}")
print(f"(For comparison: raw CatBoost OOF was 0.941656, current best is XGBoost at 0.944174)")

test_pred_avg = test_preds.mean(axis=0)
submission = pd.DataFrame({"id": test["id"], TARGET: test_pred_avg})
out_path = f"{BASE}/submissions/catboost_5fold_income_digits_freq.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission written to: {out_path}")
print(submission.head())
