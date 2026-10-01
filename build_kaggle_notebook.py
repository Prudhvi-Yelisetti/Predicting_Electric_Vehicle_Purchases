"""Builds kaggle_ev_pipeline.ipynb (self-contained, importable into Kaggle)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Playground S6E9 - EV purchase prediction (current best pipeline)

Self-contained port of the local project pipeline:

1. Rebuild features from raw `train.csv` / `test.csv`: income digits + frequency encoding, nested multi-resolution OOF target encoding of income & commute, and frequency / lift / novelty features from the original ~10k-row dataset.
2. **CatBoost** (Optuna-tuned params), 5-fold, multi-seed bag - the submitted best (OOF ~0.94600, LB 0.94610).
3. **Entity-embedding DNN** on the same features, multi-seed bag (GPU recommended) - for diversity checks / optional blending.

**Before running:** click *Add Input* and attach (a) the competition `playground-series-s6e9` and (b) the original dataset *EV Adoption Behavior & Range Anxiety* (file `EV_Adoption_and_Range_Anxiety_Dataset.csv`). For the DNN section set Accelerator to a GPU (T4/P100). CatBoost runs on CPU by default so results match the local runs.

Use the `RUN_*` flags in the config cell to run pieces separately (Kaggle sessions are limited, so run CatBoost and DNN in separate sessions if needed).""")

code('''import os, time, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
RUN_CATBOOST = True
RUN_DNN = True
CB_SEEDS = [42, 123, 456]          # 3-seed bag (each seed = 5 folds, ~15-20 min on Kaggle CPU)
DNN_SEEDS = [42, 123, 456, 789, 2024]
N_SPLITS = 5
FOLD_SEED = 42                     # fixed across the whole project
TARGET = "Will_Buy_EV"
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
LOCAL_DATA = os.environ.get("LOCAL_DATA", "")   # only used when testing off-Kaggle

def find_file(name):
    for root in ["/kaggle/input", LOCAL_DATA]:
        if root and os.path.isdir(root):
            for dp, _, fs in os.walk(root):
                if name in fs:
                    return os.path.join(dp, name)
    raise FileNotFoundError(f"{name} not found - add the input dataset/competition to the notebook")
''')

code('''train_raw = pd.read_csv(find_file("train.csv"))
test_raw = pd.read_csv(find_file("test.csv"))
orig = pd.read_csv(find_file("EV_Adoption_and_Range_Anxiety_Dataset.csv"))
print(train_raw.shape, test_raw.shape, orig.shape)

y = train_raw[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
orig_y = orig[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED)
fold_idx = list(skf.split(train_raw, y))
''')

md("## Feature engineering")
code('''# ---- 1. income digits + frequency encoding (train counts; unseen test values -> 0) ----
def add_base_features(tr, te):
    tr, te = tr.copy(), te.copy()
    for df in (tr, te):
        inc = df["Annual_Income_USD"].astype("int64")
        for i in range(6):
            df[f"inc_d{i}"] = (inc // 10**i) % 10
    for c in ["Annual_Income_USD", "Daily_Commute_km"]:
        vc = tr[c].value_counts()
        tr[c + "_freq"] = tr[c].map(vc).astype("int64")
        te[c + "_freq"] = te[c].map(vc).fillna(0).astype("int64")
    return tr, te

train_p, test_p = add_base_features(train_raw, test_raw)

# ---- 2. nested / multi-resolution OOF target encoding ----
def oof_target_encode(train_col, test_col, y, fold_idx, smooth):
    gm = y.mean()
    oof = np.full(len(train_col), np.nan)
    for tr_idx, va_idx in fold_idx:
        s = pd.DataFrame({"k": train_col.iloc[tr_idx].values, "y": y.iloc[tr_idx].values})
        g = s.groupby("k")["y"].agg(["mean", "count"])
        enc = (g["count"] * g["mean"] + smooth * gm) / (g["count"] + smooth)
        oof[va_idx] = train_col.iloc[va_idx].map(enc).fillna(gm).values
    s = pd.DataFrame({"k": train_col.values, "y": y.values})
    g = s.groupby("k")["y"].agg(["mean", "count"])
    enc = (g["count"] * g["mean"] + smooth * gm) / (g["count"] + smooth)
    return oof, test_col.map(enc).fillna(gm).values

inc_res = {"exact": 1, "r10": 10, "r100": 100, "r1000": 1000}
com_res = {"exact": 1, "r1": 1, "r5": 5}
inc_smooth = {"exact": 20, "r10": 15, "r100": 8, "r1000": 3}
com_smooth = {"exact": 5, "r1": 5, "r5": 3}

new_tr, new_te = {}, {}
for name, w in inc_res.items():
    ktr = train_p["Annual_Income_USD"] // w * w if w > 1 else train_p["Annual_Income_USD"]
    kte = test_p["Annual_Income_USD"] // w * w if w > 1 else test_p["Annual_Income_USD"]
    new_tr[f"inc_te_{name}"], new_te[f"inc_te_{name}"] = oof_target_encode(ktr, kte, y, fold_idx, inc_smooth[name])
for name, w in com_res.items():
    ktr = train_p["Daily_Commute_km"] // w * w if w > 1 else train_p["Daily_Commute_km"]
    kte = test_p["Daily_Commute_km"] // w * w if w > 1 else test_p["Daily_Commute_km"]
    new_tr[f"com_te_{name}"], new_te[f"com_te_{name}"] = oof_target_encode(ktr, kte, y, fold_idx, com_smooth[name])

# ---- 3. frequency / lift / novelty vs the original dataset ----
g_rate = orig_y.mean()
def orig_stats(col, smooth=3):
    rate = orig.groupby(col)[TARGET].apply(lambda s: (s == "Yes").mean())
    cnt = orig.groupby(col)[TARGET].size()
    freq = cnt / len(orig)
    sm = (cnt * rate + smooth * g_rate) / (cnt + smooth)
    return freq, sm - g_rate

for pre, col in [("inc", "Annual_Income_USD"), ("com", "Daily_Commute_km")]:
    fmap, lmap = orig_stats(col)
    for df, store in ((train_p, new_tr), (test_p, new_te)):
        store[f"{pre}_freq_orig"] = df[col].map(fmap).fillna(0.0).values
        store[f"{pre}_lift_orig"] = df[col].map(lmap).fillna(0.0).values
        store[f"{pre}_novel_orig"] = (~df[col].isin(orig[col])).astype("int8").values

train_fe, test_fe = train_p.copy(), test_p.copy()
for k, v in new_tr.items(): train_fe[k] = v
for k, v in new_te.items(): test_fe[k] = v
# same column order as the local train_nested_te.csv
order = [c for c in train_fe.columns if c != TARGET] + [TARGET]
train_fe = train_fe[order]
print(train_fe.shape, test_fe.shape)
print(list(new_tr.keys()))
''')

code('''# Optional: if you uploaded the local train_nested_te.csv as an input, verify the rebuild matches it exactly
try:
    ref = pd.read_csv(find_file("train_nested_te.csv"))
    diff = max(np.abs(ref[c].astype(float) - train_fe[c].astype(float)).max()
               for c in ref.columns if ref[c].dtype != object and str(ref[c].dtype) != "str")
    print("max abs diff vs reference train_nested_te.csv:", diff)
except FileNotFoundError:
    print("reference file not attached - skipping check")
''')

md("## CatBoost - multi-seed bag (CPU)")
code('''from catboost import CatBoostClassifier

BEST_PARAMS = dict(learning_rate=0.04304867132714463, depth=5, l2_leaf_reg=2.998747815633837,
                   bagging_temperature=1.0470518167824743, random_strength=2.685717108403337,
                   border_count=196)
cb_cat_cols = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
               "Subsidy_Available", "Range_Anxiety_Level"] + [f"inc_d{i}" for i in range(6)]

X = train_fe.drop(columns=[TARGET, "id"]).copy()
Xt = test_fe.drop(columns=["id"]).copy()
for c in cb_cat_cols:
    X[c] = X[c].astype(str); Xt[c] = Xt[c].astype(str)

cb_oof, cb_test = {}, {}
if RUN_CATBOOST:
    t0 = time.time()
    for s in CB_SEEDS:
        oof = np.zeros(len(X)); tp = np.zeros((N_SPLITS, len(Xt)))
        for f, (tr_i, va_i) in enumerate(fold_idx):
            m = CatBoostClassifier(iterations=40000, loss_function="Logloss", eval_metric="AUC",
                                   random_seed=s, verbose=False, early_stopping_rounds=150, **BEST_PARAMS)
            m.fit(X.iloc[tr_i], y.iloc[tr_i], cat_features=cb_cat_cols, eval_set=(X.iloc[va_i], y.iloc[va_i]))
            oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
            tp[f] = m.predict_proba(Xt)[:, 1]
            print(f"seed={s} fold={f} AUC={roc_auc_score(y.iloc[va_i], oof[va_i]):.6f} "
                  f"iters={m.get_best_iteration()} elapsed={time.time()-t0:.0f}s", flush=True)
            np.save(f"{WORK}/cb_oof_seed{s}.npy", oof); np.save(f"{WORK}/cb_test_seed{s}.npy", tp)
        cb_oof[s], cb_test[s] = oof, tp.mean(axis=0)
        print(f"--- seed {s}: OOF {roc_auc_score(y, oof):.6f} ---", flush=True)
    bag_oof = np.mean(list(cb_oof.values()), axis=0)
    bag_test = np.mean(list(cb_test.values()), axis=0)
    print(f"CatBoost {len(CB_SEEDS)}-seed bag OOF AUC: {roc_auc_score(y, bag_oof):.6f}  (local reference: 0.945997)")
    np.save(f"{WORK}/cb_bag_oof.npy", bag_oof); np.save(f"{WORK}/cb_bag_test.npy", bag_test)
    pd.DataFrame({"id": test_fe["id"], TARGET: bag_test}).to_csv(f"{WORK}/submission_catboost_seedbag.csv", index=False)
''')

md("## Entity-embedding DNN - multi-seed bag (use a GPU accelerator)")
code('''import torch, torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DNN device:", DEV)

DNN_CAT = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available",
           "Range_Anxiety_Level", "Environmental_Concern_Level", "Number_of_Cars_Owned"] + [f"inc_d{i}" for i in range(6)]
DNN_CONT = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
            "Annual_Income_USD_freq", "Daily_Commute_km_freq",
            "inc_te_exact", "inc_te_r10", "inc_te_r100", "inc_te_r1000", "com_te_exact", "com_te_r1", "com_te_r5",
            "inc_freq_orig", "inc_lift_orig", "inc_novel_orig", "com_freq_orig", "com_lift_orig", "com_novel_orig"]

class Net(nn.Module):
    def __init__(self, cards, dims, n_cont):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cards, dims)])
        d_in = sum(dims) + n_cont
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(d_in),
            nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1))
    def forward(self, xc, xn):
        return self.mlp(torch.cat([e(xc[:, i]) for i, e in enumerate(self.emb)] + [xn], 1)).squeeze(-1)

def fit_fold(cards, dims, tr, va, te, y_tr, y_va, max_epochs=50, patience=5, bs=4096):
    model = Net(cards, dims, tr[1].shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    T = lambda a: torch.from_numpy(a).to(DEV)
    trc, trn, ytr, vac, van, tec, ten = T(tr[0]), T(tr[1]), T(y_tr), T(va[0]), T(va[1]), T(te[0]), T(te[1])
    best, state, bad = -1, None, 0
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(len(ytr), device=DEV)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); lossf(model(trc[idx], trn[idx]), ytr[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(y_va, torch.sigmoid(model(vac, van)).cpu().numpy())
        if auc > best:
            best, bad = auc, 0; state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        return (torch.sigmoid(model(vac, van)).cpu().numpy(),
                torch.sigmoid(model(tec, ten)).cpu().numpy(), best, ep + 1)

dnn_oof, dnn_test = {}, {}
if RUN_DNN:
    cat_tr = np.zeros((len(train_fe), len(DNN_CAT)), dtype=np.int64)
    cat_te = np.zeros((len(test_fe), len(DNN_CAT)), dtype=np.int64)
    cards = []
    for i, c in enumerate(DNN_CAT):
        le = LabelEncoder().fit(pd.concat([train_fe[c].astype(str), test_fe[c].astype(str)]))
        cat_tr[:, i] = le.transform(train_fe[c].astype(str)); cat_te[:, i] = le.transform(test_fe[c].astype(str))
        cards.append(len(le.classes_))
    dims = [min(50, max(2, (c + 1) // 2)) for c in cards]
    cont_tr = train_fe[DNN_CONT].to_numpy(np.float32); cont_te = test_fe[DNN_CONT].to_numpy(np.float32)
    yf = y.to_numpy().astype(np.float32)

    t0 = time.time()
    for s in DNN_SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        oof = np.zeros(len(train_fe)); tp = np.zeros((N_SPLITS, len(test_fe)))
        for f, (tr_i, va_i) in enumerate(fold_idx):
            sc = StandardScaler().fit(cont_tr[tr_i])
            n = lambda a: sc.transform(a).astype(np.float32)
            va_p, te_p, best, ep = fit_fold(cards, dims, (cat_tr[tr_i], n(cont_tr[tr_i])), (cat_tr[va_i], n(cont_tr[va_i])),
                                            (cat_te, n(cont_te)), yf[tr_i], yf[va_i])
            oof[va_i], tp[f] = va_p, te_p
            print(f"DNN seed={s} fold={f} AUC={best:.6f} epochs={ep} elapsed={time.time()-t0:.0f}s", flush=True)
        dnn_oof[s], dnn_test[s] = oof, tp.mean(axis=0)
        np.save(f"{WORK}/dnn_oof_seed{s}.npy", oof); np.save(f"{WORK}/dnn_test_seed{s}.npy", tp)
        print(f"--- DNN seed {s}: OOF {roc_auc_score(y, oof):.6f} ---", flush=True)
    d_oof = np.mean(list(dnn_oof.values()), axis=0); d_test = np.mean(list(dnn_test.values()), axis=0)
    print(f"DNN {len(DNN_SEEDS)}-seed bag OOF AUC: {roc_auc_score(y, d_oof):.6f}")
    np.save(f"{WORK}/dnn_bag_oof.npy", d_oof); np.save(f"{WORK}/dnn_bag_test.npy", d_test)
    pd.DataFrame({"id": test_fe["id"], TARGET: d_test}).to_csv(f"{WORK}/submission_dnn_seedbag.csv", index=False)
''')

md("""## Blend check (only trust it if the gain is large relative to noise)
Rank-average blend of the two bags; compare OOF against CatBoost alone. Stacking has failed repeatedly on this project, so treat any gain below ~+0.0002 as noise.""")
code('''from scipy.stats import rankdata
if RUN_CATBOOST and RUN_DNN:
    r = lambda a: rankdata(a) / len(a)
    print("corr(CB, DNN):", np.corrcoef(bag_oof, d_oof)[0, 1])
    base = roc_auc_score(y, bag_oof)
    for w in [0.0, 0.1, 0.2, 0.3]:
        print(f"w_dnn={w:.1f}  OOF={roc_auc_score(y, (1-w)*r(bag_oof) + w*r(d_oof)):.6f}  (CatBoost alone {base:.6f})")
''')

code('''# Final submission: CatBoost bag (best validated). Kaggle picks up /kaggle/working/submission.csv
if RUN_CATBOOST:
    pd.DataFrame({"id": test_fe["id"], TARGET: bag_test}).to_csv(f"{WORK}/submission.csv", index=False)
    print(pd.read_csv(f"{WORK}/submission.csv").head())
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases/kaggle_ev_pipeline.ipynb")
print("written", len(cells), "cells")
