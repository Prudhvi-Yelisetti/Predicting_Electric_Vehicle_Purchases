"""
Experiment 010: nested/multi-resolution OOF target encoding of income &
commute, plus per-value frequency/lift/novelty vs the original ~10k-row
dataset. Tested with XGBoost (GPU, fast) as harness, identical 5-fold
StratifiedKFold(random_state=42) split used throughout this project, on
top of the existing winning preprocessed feature set (income digit
decomposition + frequency encoding). Baseline to beat: XGBoost OOF
0.944174 (Experiment 005), CatBoost-tuned OOF 0.945243 (current best).
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

DATA_DIR = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases/Data/"
TARGET = "Will_Buy_EV"

train = pd.read_csv(DATA_DIR + "train_preprocessed.csv")
test = pd.read_csv(DATA_DIR + "test_preprocessed.csv")
orig = pd.read_csv(DATA_DIR + "EV_Adoption_and_Range_Anxiety_Dataset.csv")

y = train[TARGET].map({"Yes": 1, "No": 0}).astype("int8")
orig_y = orig[TARGET].map({"Yes": 1, "No": 0}).astype("int8")

FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold_idx = list(skf.split(train, y))

def oof_target_encode(train_col, test_col, y, fold_idx, smooth):
    """Leak-free OOF target encoding of a (rounded) key column."""
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

income_resolutions = {"exact": 1, "r10": 10, "r100": 100, "r1000": 1000}
commute_resolutions = {"exact": 1, "r1": 1, "r5": 5}
income_smooth = {"exact": 20, "r10": 15, "r100": 8, "r1000": 3}
commute_smooth = {"exact": 5, "r1": 5, "r5": 3}

new_train_cols = {}
new_test_cols = {}

for name, width in income_resolutions.items():
    key_tr = (train["Annual_Income_USD"] // width * width) if width > 1 else train["Annual_Income_USD"]
    key_te = (test["Annual_Income_USD"] // width * width) if width > 1 else test["Annual_Income_USD"]
    oof, te = oof_target_encode(key_tr, key_te, y, fold_idx, income_smooth[name])
    new_train_cols[f"inc_te_{name}"] = oof
    new_test_cols[f"inc_te_{name}"] = te

for name, width in commute_resolutions.items():
    key_tr = (train["Daily_Commute_km"] // width * width) if width > 1 else train["Daily_Commute_km"]
    key_te = (test["Daily_Commute_km"] // width * width) if width > 1 else test["Daily_Commute_km"]
    oof, te = oof_target_encode(key_tr, key_te, y, fold_idx, commute_smooth[name])
    new_train_cols[f"com_te_{name}"] = oof
    new_test_cols[f"com_te_{name}"] = te

global_orig_rate = orig_y.mean()

def orig_stats(col_name, smooth=3):
    grp = orig.groupby(col_name)[TARGET].apply(lambda s: (s == "Yes").mean())
    cnt = orig.groupby(col_name)[TARGET].size()
    freq = cnt / len(orig)
    smoothed_rate = (cnt * grp + smooth * global_orig_rate) / (cnt + smooth)
    lift = smoothed_rate - global_orig_rate
    return freq, lift

inc_freq_map, inc_lift_map = orig_stats("Annual_Income_USD")
com_freq_map, com_lift_map = orig_stats("Daily_Commute_km")

new_train_cols["inc_freq_orig"] = train["Annual_Income_USD"].map(inc_freq_map).fillna(0.0).values
new_train_cols["inc_lift_orig"] = train["Annual_Income_USD"].map(inc_lift_map).fillna(0.0).values
new_train_cols["inc_novel_orig"] = (~train["Annual_Income_USD"].isin(orig["Annual_Income_USD"])).astype("int8").values
new_train_cols["com_freq_orig"] = train["Daily_Commute_km"].map(com_freq_map).fillna(0.0).values
new_train_cols["com_lift_orig"] = train["Daily_Commute_km"].map(com_lift_map).fillna(0.0).values
new_train_cols["com_novel_orig"] = (~train["Daily_Commute_km"].isin(orig["Daily_Commute_km"])).astype("int8").values

new_test_cols["inc_freq_orig"] = test["Annual_Income_USD"].map(inc_freq_map).fillna(0.0).values
new_test_cols["inc_lift_orig"] = test["Annual_Income_USD"].map(inc_lift_map).fillna(0.0).values
new_test_cols["inc_novel_orig"] = (~test["Annual_Income_USD"].isin(orig["Annual_Income_USD"])).astype("int8").values
new_test_cols["com_freq_orig"] = test["Daily_Commute_km"].map(com_freq_map).fillna(0.0).values
new_test_cols["com_lift_orig"] = test["Daily_Commute_km"].map(com_lift_map).fillna(0.0).values
new_test_cols["com_novel_orig"] = (~test["Daily_Commute_km"].isin(orig["Daily_Commute_km"])).astype("int8").values

train_new = train.copy()
test_new = test.copy()
for k, v in new_train_cols.items():
    train_new[k] = v
for k, v in new_test_cols.items():
    test_new[k] = v

print("New feature columns:", list(new_train_cols.keys()))
print("train_new shape:", train_new.shape)

train_new.to_csv(DATA_DIR + "train_nested_te.csv", index=False)
test_new.to_csv(DATA_DIR + "test_nested_te.csv", index=False)
print("Saved train_nested_te.csv / test_nested_te.csv")

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
    print(f"[{label}] mean fold AUC={np.mean(aucs):.6f} std={np.std(aucs):.6f} OOF AUC={oof_auc:.6f} time={time.time()-t0:.0f}s")
    return oof_auc, oof

baseline_oof_auc, _ = run_cv(base_feature_cols, "baseline (existing preprocessed set)")
new_oof_auc, new_oof = run_cv(base_feature_cols + new_feature_cols, "baseline + nested TE + orig lift/freq/novel")

print("=" * 60)
print(f"Baseline XGBoost OOF:            {baseline_oof_auc:.6f}")
print(f"+ nested TE + orig features OOF: {new_oof_auc:.6f}")
print(f"Delta:                           {new_oof_auc - baseline_oof_auc:+.6f}")
