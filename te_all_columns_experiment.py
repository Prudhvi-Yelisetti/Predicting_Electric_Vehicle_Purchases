"""
Experiment 012: target encoding on every remaining column (the 6 base
categoricals + 5 other numeric/low-cardinality columns not already
nested-TE'd in Experiment 010) at two smoothing levels each (m=10,
m=100) - the lowest-priority, smallest-expected-gain item from
External-Research-Deep-Dive.md (+0.0003 in the external ablation).
Tested with XGBoost first, same harness/protocol as Experiment 010.
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
test = pd.read_csv(DATA_DIR + "test_nested_te.csv")
y = train[TARGET].map({"Yes": 1, "No": 0}).astype("int8")

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold_idx = list(skf.split(train, y))

def oof_target_encode(train_col, test_col, y, fold_idx, smooth):
    global_mean = y.mean()
    oof = np.full(len(train_col), np.nan)
    for tr_idx, va_idx in fold_idx:
        stats = pd.DataFrame({"k": train_col.iloc[tr_idx].values, "y": y.iloc[tr_idx].values})
        grp = stats.groupby("k")["y"].agg(["mean", "count"])
        enc = (grp["count"] * grp["mean"] + smooth * global_mean) / (grp["count"] + smooth)
        oof[va_idx] = train_col.iloc[va_idx].map(enc).fillna(global_mean).values
    stats_full = pd.DataFrame({"k": train_col.values, "y": y.values})
    grp_full = stats_full.groupby("k")["y"].agg(["mean", "count"])
    enc_full = (grp_full["count"] * grp_full["mean"] + smooth * global_mean) / (grp_full["count"] + smooth)
    test_enc = test_col.map(enc_full).fillna(global_mean).values
    return oof, test_enc

# columns not already covered by Experiment 010's nested income/commute TE
te_cols = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
           "Subsidy_Available", "Range_Anxiety_Level", "Age", "Number_of_Cars_Owned",
           "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
           "Environmental_Concern_Level"]
smooths = {"s10": 10, "s100": 100}

new_train_cols = {}
new_test_cols = {}
for col in te_cols:
    for sname, m in smooths.items():
        oof, te = oof_target_encode(train[col], test[col], y, fold_idx, m)
        new_train_cols[f"{col}_te_{sname}"] = oof
        new_test_cols[f"{col}_te_{sname}"] = te

print(f"New columns ({len(new_train_cols)}): {list(new_train_cols.keys())}")

train_new = train.copy()
test_new = test.copy()
for k, v in new_train_cols.items():
    train_new[k] = v
for k, v in new_test_cols.items():
    test_new[k] = v

train_new.to_csv(DATA_DIR + "train_nested_te_allcols.csv", index=False)
test_new.to_csv(DATA_DIR + "test_nested_te_allcols.csv", index=False)
print("Saved train_nested_te_allcols.csv / test_nested_te_allcols.csv")

cat_cols = train.select_dtypes(include=["object", "category", "str"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]
for df in (train_new, test_new):
    for c in cat_cols:
        df[c] = df[c].astype("category")

base_feature_cols = [c for c in train.columns if c not in ("id", TARGET)]
new_feature_cols = list(new_train_cols.keys())

def run_cv(feature_cols, label):
    oof = np.zeros(len(train_new))
    aucs = []
    t0 = time.time()
    for i, (tr_idx, va_idx) in enumerate(fold_idx):
        X_tr, X_va = train_new.iloc[tr_idx][feature_cols], train_new.iloc[va_idx][feature_cols]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.05, n_estimators=3000,
            early_stopping_rounds=50, device="cuda",
            enable_categorical=True, eval_metric="auc", random_state=42,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = pred
        fold_auc = roc_auc_score(y_va, pred)
        aucs.append(fold_auc)
        print(f"[{label}] fold {i+1}: AUC={fold_auc:.6f}")
    oof_auc = roc_auc_score(y, oof)
    print(f"[{label}] mean={np.mean(aucs):.6f} std={np.std(aucs):.6f} OOF={oof_auc:.6f} time={time.time()-t0:.0f}s")
    return oof_auc

baseline_auc = run_cv(base_feature_cols, "baseline (nested TE, no all-col TE)")
new_auc = run_cv(base_feature_cols + new_feature_cols, "+ all-column TE (2 smoothings)")

print("=" * 60)
print(f"Baseline OOF (nested TE only):     {baseline_auc:.6f}")
print(f"+ all-column TE OOF:               {new_auc:.6f}")
print(f"Delta:                             {new_auc - baseline_auc:+.6f}")
