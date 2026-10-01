"""
Entity-embedding DNN, REBUILT on the current winning feature set
(train_nested_te.csv - income-digit + frequency + nested/multi-res
target-encoding + original-dataset lift/novelty features, Experiment
010), with 5-seed bagging.

Why: the existing dnn_entity_embedding.py / tmp_results/dnn_*.npy were
trained on the OLD preprocessed feature set, before the nested-TE jump
(+0.0007 to +0.0016 OOF for the tree models). Its OOF (0.941944) and its
correlation with the tree models (0.987-0.989) are both stale numbers
measured against a feature set nobody submits anymore. This script:
  1. Rebuilds the DNN on the actual current best feature set.
  2. Seed-bags it (5 seeds, SAME fixed 5-fold split each time, only
     model init/shuffle seed varies) - the same variance-reduction
     lever that gave CatBoost/XGBoost real (if small) gains.
  3. Recomputes correlation against the current best tree OOFs
     (CatBoost 3-seed bag, XGBoost nested-TE) so a stacking re-attempt
     can be judged on up-to-date numbers, still gated by the z>=3 bar.

Long CPU-only job (668k rows x 5 folds x 5 seeds) - run directly in
your own terminal, NOT backgrounded through the remote connection
(that has died silently mid-run on this project before):

    cd ~/Hackathons/Predicting_Electric_Vehicle_Purchases
    .venv/bin/python -u dnn_nested_te_seedbag.py 2>&1 | tee logs/dnn_seedbag_run.log

Checkpointed per (seed, fold) under tmp_results/ - safe to Ctrl-C and
resume manually by editing SEEDS below if it gets interrupted.
"""
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases"
SEEDS = [42, 123, 456, 789, 2024]
N_SPLITS = 5
FOLD_SPLIT_SEED = 42  # fixed across all seeds/experiments - keeps OOF positions aligned

CAT_COLS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
            "Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level",
            "Number_of_Cars_Owned", "inc_d0", "inc_d1", "inc_d2", "inc_d3", "inc_d4", "inc_d5"]
CONT_COLS = ["Age", "Annual_Income_USD", "Daily_Commute_km",
             "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
             "Annual_Income_USD_freq", "Daily_Commute_km_freq",
             "inc_te_exact", "inc_te_r10", "inc_te_r100", "inc_te_r1000",
             "com_te_exact", "com_te_r1", "com_te_r5",
             "inc_freq_orig", "inc_lift_orig", "inc_novel_orig",
             "com_freq_orig", "com_lift_orig", "com_novel_orig"]

CB_SEEDBAG_OOF = (np.load(f"{BASE}/tmp_results/cb_seedbag_oof_seed42.npy")
                   + np.load(f"{BASE}/tmp_results/cb_seedbag_oof_seed123.npy")
                   + np.load(f"{BASE}/tmp_results/cb_seedbag_oof_seed456.npy")) / 3
XGB_NESTEDTE_OOF = np.load(f"{BASE}/tmp_results/xgb_nestedte_oof.npy")
CB_SEEDBAG_AUC = 0.945997

# --- Load & prep ---
train = pd.read_csv(f"{BASE}/Data/train_nested_te.csv")
test = pd.read_csv(f"{BASE}/Data/test_nested_te.csv")
TARGET = "Will_Buy_EV"
y_full = train[TARGET].map({"No": 0, "Yes": 1}).to_numpy().astype(np.float32)

cat_cardinalities = []
train_cat = np.zeros((len(train), len(CAT_COLS)), dtype=np.int64)
test_cat = np.zeros((len(test), len(CAT_COLS)), dtype=np.int64)
for i, c in enumerate(CAT_COLS):
    combined = pd.concat([train[c].astype(str), test[c].astype(str)], axis=0)
    le = LabelEncoder()
    le.fit(combined)
    train_cat[:, i] = le.transform(train[c].astype(str))
    test_cat[:, i] = le.transform(test[c].astype(str))
    cat_cardinalities.append(len(le.classes_))

emb_dims = [min(50, max(2, (card + 1) // 2)) for card in cat_cardinalities]
print("Categorical cardinalities:", dict(zip(CAT_COLS, cat_cardinalities)))
print("Embedding dims:", dict(zip(CAT_COLS, emb_dims)), flush=True)

train_cont_raw = train[CONT_COLS].to_numpy(dtype=np.float32)
test_cont_raw = test[CONT_COLS].to_numpy(dtype=np.float32)
print(f"\nFull data: {train_cat.shape}, test: {test_cat.shape}", flush=True)


class EntityEmbeddingNet(nn.Module):
    def __init__(self, cardinalities, emb_dims, n_continuous):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cardinalities, emb_dims)
        ])
        input_dim = sum(emb_dims) + n_continuous
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x_cat, x_cont):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_cont], dim=1)
        return self.mlp(x).squeeze(-1)


def train_one_fold(X_cat_tr, X_cont_tr, y_tr, X_cat_va, X_cont_va, y_va,
                    X_cat_test, X_cont_test, max_epochs=50, patience=5, batch_size=4096):
    model = EntityEmbeddingNet(cat_cardinalities, emb_dims, X_cont_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    X_cat_tr_t = torch.from_numpy(X_cat_tr)
    X_cont_tr_t = torch.from_numpy(X_cont_tr)
    y_tr_t = torch.from_numpy(y_tr)
    X_cat_va_t = torch.from_numpy(X_cat_va)
    X_cont_va_t = torch.from_numpy(X_cont_va)
    X_cat_test_t = torch.from_numpy(X_cat_test)
    X_cont_test_t = torch.from_numpy(X_cont_test)

    n = len(y_tr)
    best_auc, best_state, epochs_no_improve = -1.0, None, 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            out = model(X_cat_tr_t[idx], X_cont_tr_t[idx])
            loss = loss_fn(out, y_tr_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            va_pred = torch.sigmoid(model(X_cat_va_t, X_cont_va_t)).numpy()
        va_auc = roc_auc_score(y_va, va_pred)
        print(f"      epoch {epoch}: train_loss={total_loss/n:.5f} valid_auc={va_auc:.6f}",
              flush=True)

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"      early stop at epoch {epoch} (best={best_auc:.6f})", flush=True)
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_pred = torch.sigmoid(model(X_cat_va_t, X_cont_va_t)).numpy()
        test_pred = torch.sigmoid(model(X_cat_test_t, X_cont_test_t)).numpy()
    return va_pred, test_pred, best_auc


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SPLIT_SEED)
folds = list(skf.split(train_cat, y_full))

seed_oofs = []
seed_tests = []
t_start = time.time()

for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    oof_pred = np.zeros(len(train))
    test_preds = np.zeros((N_SPLITS, len(test)))
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n  --- seed {seed} fold {fold} ---", flush=True)
        scaler = StandardScaler()
        X_cont_tr = scaler.fit_transform(train_cont_raw[tr_idx]).astype(np.float32)
        X_cont_va = scaler.transform(train_cont_raw[va_idx]).astype(np.float32)
        X_cont_test = scaler.transform(test_cont_raw).astype(np.float32)

        va_pred, test_pred, best_auc = train_one_fold(
            train_cat[tr_idx], X_cont_tr, y_full[tr_idx],
            train_cat[va_idx], X_cont_va, y_full[va_idx],
            test_cat, X_cont_test,
        )
        oof_pred[va_idx] = va_pred
        test_preds[fold] = test_pred
        fold_aucs.append(best_auc)
        print(f"  seed {seed} fold {fold} best valid AUC: {best_auc:.6f} "
              f"elapsed={time.time()-t_start:.0f}s", flush=True)

        np.save(f"{BASE}/tmp_results/dnn_nestedte_oof_seed{seed}_partial.npy", oof_pred)
        np.save(f"{BASE}/tmp_results/dnn_nestedte_test_seed{seed}_partial.npy", test_preds)

    seed_oof_auc = roc_auc_score(y_full, oof_pred)
    print(f"\nSeed {seed} done: OOF AUC = {seed_oof_auc:.6f} "
          f"(fold mean {np.mean(fold_aucs):.6f} +/- {np.std(fold_aucs):.6f})", flush=True)
    np.save(f"{BASE}/tmp_results/dnn_nestedte_oof_seed{seed}.npy", oof_pred)
    np.save(f"{BASE}/tmp_results/dnn_nestedte_test_seed{seed}.npy", test_preds)
    seed_oofs.append(oof_pred)
    seed_tests.append(test_preds.mean(axis=0))

# --- Bag across seeds ---
bagged_oof = np.mean(seed_oofs, axis=0)
bagged_test = np.mean(seed_tests, axis=0)
bagged_auc = roc_auc_score(y_full, bagged_oof)

print(f"\n{'='*60}\nFINAL: {len(SEEDS)}-seed bagged DNN (nested-TE features)\n{'='*60}")
print(f"Bagged OOF AUC: {bagged_auc:.6f}")
print(f"Old DNN OOF (stale feature set): 0.941944  ->  delta {bagged_auc - 0.941944:+.6f}")

corr_cb = np.corrcoef(bagged_oof, CB_SEEDBAG_OOF)[0, 1]
corr_xgb = np.corrcoef(bagged_oof, XGB_NESTEDTE_OOF)[0, 1]
print(f"\nCorrelation with CatBoost 3-seed bag OOF ({CB_SEEDBAG_AUC:.6f}): {corr_cb:.4f}")
print(f"Correlation with XGBoost nested-TE OOF: {corr_xgb:.4f}")
print("(old DNN was 0.987-0.989 with the pre-nested-TE tree OOFs - compare against that)")

np.save(f"{BASE}/tmp_results/dnn_nestedte_bagged_oof.npy", bagged_oof)
np.save(f"{BASE}/tmp_results/dnn_nestedte_bagged_test.npy", bagged_test)
with open(f"{BASE}/tmp_results/dnn_nestedte_seedbag_summary.json", "w") as f:
    json.dump({
        "seeds": SEEDS,
        "bagged_oof_auc": float(bagged_auc),
        "old_dnn_oof_stale_features": 0.941944,
        "corr_with_catboost_seedbag": float(corr_cb),
        "corr_with_xgboost_nested_te": float(corr_xgb),
        "cb_seedbag_auc_for_reference": CB_SEEDBAG_AUC,
    }, f, indent=2)
print("\nSaved tmp_results/dnn_nestedte_seedbag_summary.json")
print("\nNext step (only if corr dropped meaningfully below the old 0.987-0.989):")
print("  rerun a stack attempt with dnn_nestedte_bagged_oof.npy added, still gated")
print("  by the z>=3 significance bar before trusting any delta.")
