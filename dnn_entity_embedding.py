"""
Entity-embedding DNN for the preprocessed feature set (income digits +
frequency encoding) - Phase 3 backlog item, aimed at real ensemble
DIVERSITY rather than beating CatBoost outright. Two independent tree-
model stacking attempts (Experiment 008) both failed because CatBoost/
XGBoost/LightGBM learn too similar a function on identical features
(OOF correlation 0.994-0.996) - a genuinely different architecture is
the remaining lever for a productive stack.

Architecture: per-column embeddings for the 14 categorical/low-
cardinality columns (6 base categoricals + Environmental_Concern_Level
+ Number_of_Cars_Owned + the 6 income-digit columns), concatenated with
7 standardized continuous columns, through a BatchNorm/Dropout MLP
(256 -> 128 -> 64 -> 1). CPU-only in this environment (only 4GB VRAM on
the laptop GPU, and this model is small enough CPU is fine) - 16 cores
available, torch threads set accordingly.

Same 5-fold StratifiedKFold (random_state=42) as every other experiment
so OOF predictions line up for stacking. Continuous-feature scaling is
fit per-fold on the training split only, to avoid leakage.

Run directly in your own terminal (long job, and remote connections
have died mid-run on this project before):
    cd ~/Hackathons/Predicting_Electric_Vehicle_Purchases
    .venv/bin/python -u dnn_entity_embedding.py
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

torch.set_num_threads(16)

BASE = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases"
CB_OOF_BASELINE = 0.945243
CAT_COLS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
            "Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level",
            "Number_of_Cars_Owned", "inc_d0", "inc_d1", "inc_d2", "inc_d3", "inc_d4", "inc_d5"]
CONT_COLS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Charging_Stations_Near_Home",
             "Charging_Stations_Near_Work", "Annual_Income_USD_freq", "Daily_Commute_km_freq"]

# --- Load & prep ---
train = pd.read_csv(f"{BASE}/Data/train_preprocessed.csv")
test = pd.read_csv(f"{BASE}/Data/test_preprocessed.csv")
TARGET = "Will_Buy_EV"
y_full = train[TARGET].map({"No": 0, "Yes": 1}).to_numpy().astype(np.float32)

# Label-encode categoricals on train+test combined (safe against any
# category only appearing in test, though none are expected here)
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
print("Embedding dims:", dict(zip(CAT_COLS, emb_dims)))

train_cont_raw = train[CONT_COLS].to_numpy(dtype=np.float32)
test_cont_raw = test[CONT_COLS].to_numpy(dtype=np.float32)

print(f"\nFull data: {train_cat.shape}, test: {test_cat.shape}")

# --- Model ---
class EntityEmbeddingNet(nn.Module):
    def __init__(self, cardinalities, emb_dims, n_continuous):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cardinalities, emb_dims)
        ])
        emb_total = sum(emb_dims)
        input_dim = emb_total + n_continuous
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
            va_logits = model(X_cat_va_t, X_cont_va_t)
            va_pred = torch.sigmoid(va_logits).numpy()
        va_auc = roc_auc_score(y_va, va_pred)

        print(f"    epoch {epoch}: train_loss={total_loss/n:.5f}  valid_auc={va_auc:.6f}",
              flush=True)

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    early stopping at epoch {epoch} (best valid_auc={best_auc:.6f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_pred = torch.sigmoid(model(X_cat_va_t, X_cont_va_t)).numpy()
        test_pred = torch.sigmoid(model(X_cat_test_t, X_cont_test_t)).numpy()
    return va_pred, test_pred, best_auc

# --- 5-fold CV, identical folds to every other experiment ---
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(train_cat, y_full))

oof_pred = np.zeros(len(train))
test_preds = np.zeros((N_SPLITS, len(test)))
fold_aucs = []
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(folds):
    print(f"\n=== Fold {fold} ===", flush=True)
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
    print(f"  Fold {fold} best valid AUC: {best_auc:.6f}  elapsed={time.time()-t0:.0f}s")

    np.save(f"{BASE}/tmp_results/dnn_oof.npy", oof_pred)
    np.save(f"{BASE}/tmp_results/dnn_test.npy", test_preds)

oof_auc = roc_auc_score(y_full, oof_pred)
print(f"\nMean fold AUC: {np.mean(fold_aucs):.6f}  Std: {np.std(fold_aucs):.6f}")
print(f"DNN OOF AUC: {oof_auc:.6f}")
print(f"Best tree model (tuned CatBoost) OOF: {CB_OOF_BASELINE:.6f}")
print(f"Delta: {oof_auc - CB_OOF_BASELINE:+.6f}")

# --- Correlation with existing tree-model OOF, for stacking purposes ---
cb_oof = np.load(f"{BASE}/tmp_results/cb_tuned_oof.npy")
xgb_oof = np.load(f"{BASE}/tmp_results/xgb_oof.npy")
lgb_oof = np.load(f"{BASE}/tmp_results/lgb_oof.npy")
corr_cb = np.corrcoef(oof_pred, cb_oof)[0, 1]
corr_xgb = np.corrcoef(oof_pred, xgb_oof)[0, 1]
corr_lgb = np.corrcoef(oof_pred, lgb_oof)[0, 1]
print(f"\nDNN OOF correlation with CatBoost: {corr_cb:.4f}")
print(f"DNN OOF correlation with XGBoost:  {corr_xgb:.4f}")
print(f"DNN OOF correlation with LightGBM: {corr_lgb:.4f}")
print("(tree models were 0.994-0.996 correlated with each other - "
      "lower numbers here would mean real diversity)")

if oof_auc > CB_OOF_BASELINE:
    test_pred_avg = test_preds.mean(axis=0)
    submission = pd.DataFrame({"id": test["id"], TARGET: test_pred_avg})
    out_path = f"{BASE}/submissions/dnn_entity_embedding_5fold.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nDNN alone IMPROVED on best single tree model - "
          f"submission written to: {out_path}")
else:
    print(f"\nDNN alone did not beat {CB_OOF_BASELINE:.6f} - "
          "expected, this is about diversity for stacking, not a standalone win. "
          "No submission written for the DNN alone; check the correlation numbers "
          "above and rerun stack_phase4.py including dnn_oof/test.npy next.")

with open(f"{BASE}/tmp_results/dnn_summary.json", "w") as f:
    json.dump({
        "oof_auc": float(oof_auc),
        "fold_aucs": fold_aucs,
        "best_tree_model_oof": CB_OOF_BASELINE,
        "corr_with_catboost": float(corr_cb),
        "corr_with_xgboost": float(corr_xgb),
        "corr_with_lightgbm": float(corr_lgb),
        "cat_cardinalities": dict(zip(CAT_COLS, cat_cardinalities)),
        "emb_dims": dict(zip(CAT_COLS, emb_dims)),
    }, f, indent=2)
print("\nSaved tmp_results/dnn_summary.json")
