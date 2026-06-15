#!/usr/bin/env python3
"""Run the documented-but-deferred XGBoost hyperparameter scan on the REAL
champion rotation dataset, producing genuine AUC error-surfaces for the paper.

Mirrors the reference trainer predictors/xgboost-classifier/train.py exactly:
XGBClassifier(objective="binary:logistic"), fit on x[train_idx], AUC =
roc_auc_score(y_test, P(class=1)) on the frozen test split. We first reproduce
the champion config (depth 6, lr 0.05, 500 trees) and assert AUC ~ 0.7429 so the
harness is trustworthy, then sweep two 2-D grids holding the other defaults.

Run with the shared predictor venv:
  predictors/.venv/bin/python docs/sweeps/run_xgb_sweep.py
"""
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

SPE = Path("/home/gonik/Documents/git/snappler/lensing-workspace/lensing-instances/spotify-predict-engagement")
DATASET = SPE / "data/datasets/ds-20260609-141622-p1-s42"
OUT = Path(__file__).resolve().parent / "xgb_sweep.json"

DEFAULTS = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0,
                reg_lambda=1.0, seed=42)
CHAMPION_AUC = 0.7429335172599421  # from data/runs/run-20260609-172920-627d6-xgboost-classifier/metrics.json

DEPTHS = [2, 3, 4, 6, 8, 10, 12]
LRS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
NESTS = [50, 100, 200, 500, 1000, 2000]


def load():
    m = json.loads((DATASET / "manifest.json").read_text())
    n, c = m["n_rows"], m["n_cols"]
    x = np.fromfile(DATASET / "features.f32", dtype="<f4").reshape(n, c).astype(np.float64)
    t = np.fromfile(DATASET / "target.f32", dtype="<f4").astype(np.float64)
    tr = np.fromfile(DATASET / "train_idx.u32", dtype="<u4")
    te = np.fromfile(DATASET / "test_idx.u32", dtype="<u4")
    y = np.rint(t).astype(np.int64)  # rotation ∈ {0,1}
    return m, x, y, tr, te


def fit_auc(x, y, tr, te, **hp):
    p = {**DEFAULTS, **hp}
    model = xgb.XGBClassifier(
        objective="binary:logistic", num_class=None,
        n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]),
        learning_rate=float(p["learning_rate"]), subsample=float(p["subsample"]),
        colsample_bytree=float(p["colsample_bytree"]),
        min_child_weight=float(p["min_child_weight"]),
        reg_lambda=float(p["reg_lambda"]), random_state=int(p["seed"]),
        eval_metric="logloss", n_jobs=-1)
    model.fit(x[tr], y[tr])
    p1 = model.predict_proba(x[te])[:, 1]
    return float(roc_auc_score(y[te], p1)), float(log_loss(y[te], p1, labels=[0, 1]))


def main():
    m, x, y, tr, te = load()
    print(f"dataset {m['dataset_id']}: {len(tr)} train / {len(te)} test, {m['n_cols']} cols; "
          f"class balance train={y[tr].mean():.3f} test={y[te].mean():.3f}")

    t0 = time.time()
    champ_auc, champ_ll = fit_auc(x, y, tr, te)
    print(f"champion (depth 6, lr 0.05, 500): AUC {champ_auc:.4f} (published {CHAMPION_AUC:.4f}), "
          f"logloss {champ_ll:.4f}  [{time.time()-t0:.1f}s]")
    assert abs(champ_auc - CHAMPION_AUC) < 0.01, "harness does NOT reproduce champion AUC — aborting"

    grid_dl = {"depths": DEPTHS, "lrs": LRS, "auc": [], "logloss": []}
    for d in DEPTHS:
        row_a, row_l = [], []
        for lr in LRS:
            a, l = fit_auc(x, y, tr, te, max_depth=d, learning_rate=lr)
            row_a.append(a); row_l.append(l)
            print(f"  depth={d:2d} lr={lr:<4} -> AUC {a:.4f}")
        grid_dl["auc"].append(row_a); grid_dl["logloss"].append(row_l)

    grid_dn = {"depths": DEPTHS, "nests": NESTS, "auc": [], "logloss": []}
    for d in DEPTHS:
        row_a, row_l = [], []
        for ne in NESTS:
            a, l = fit_auc(x, y, tr, te, max_depth=d, n_estimators=ne)
            row_a.append(a); row_l.append(l)
            print(f"  depth={d:2d} n_est={ne:<4} -> AUC {a:.4f}")
        grid_dn["auc"].append(row_a); grid_dn["logloss"].append(row_l)

    out = {
        "dataset_id": m["dataset_id"], "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_cols": int(m["n_cols"]),
        "champion": {**DEFAULTS, "auc": champ_auc, "auc_published": CHAMPION_AUC, "logloss": champ_ll},
        "depth_lr": grid_dl, "depth_nest": grid_dn,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}  ({out['elapsed_s']}s total)")
    best = max((grid_dl["auc"][i][j], DEPTHS[i], LRS[j])
               for i in range(len(DEPTHS)) for j in range(len(LRS)))
    print(f"best depth×lr cell: AUC {best[0]:.4f} at depth={best[1]}, lr={best[2]} "
          f"(champion {champ_auc:.4f}; noise 2σ≈0.0126)")


if __name__ == "__main__":
    main()
