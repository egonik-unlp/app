#!/usr/bin/env python3
"""Aggregate the REAL logged runs from spotify-predict-engagement into the
family bake-off, the 3-seed noise band, and the feature-variant (AE-dilution)
comparison used by the paper. All numbers come straight from data/runs/*/
metrics.json + data/best-models.json (no recomputation).

  /home/gonik/anaconda3/bin/python docs/sweeps/load_runs.py
"""
import json
import os
from pathlib import Path

SPE = Path("/home/gonik/Documents/git/snappler/lensing-workspace/lensing-instances/spotify-predict-engagement")
RUNS = SPE / "data/runs"
OUT = Path(__file__).resolve().parent / "runs.json"
CHAMP_DS = "ds-20260609-141622-p1-s42"  # metadata-only, 115 cols, the rotation champion dataset

# SVC run-id -> kernel, from experiments/2026-06-09-rotation-classification.md
# (run dirs don't store the kernel; the experiment table maps AUC->kernel).
SVC_KERNEL = {
    "run-20260609-172933-848b8-svc": "svc (poly)",
    "run-20260609-172933-a704f-svc": "svc (rbf)",
    "run-20260609-172933-45798-svc": "svc (linear)",
    "run-20260609-172933-53bb5-svc": "svc (sigmoid)",
}


def auc(run):
    return json.loads((RUNS / run / "metrics.json").read_text())["auc"]


def ds_of(run):
    pj = RUNS / run / "progress.jsonl"
    if pj.exists():
        for line in pj.read_text().splitlines():
            try:
                msg = json.loads(line).get("msg", "")
            except Exception:
                continue
            if msg.startswith("dataset ds-"):
                return msg.split()[1].rstrip(":")
    return ""


def main():
    runs = sorted(os.path.basename(p) for p in RUNS.glob("run-*"))

    # --- family bake-off on the champion dataset (rotation, 115 cols) ---
    bakeoff = []
    for r in runs:
        if ds_of(r) != CHAMP_DS:
            continue
        if r.endswith("-xgboost-classifier"):
            label = "xgboost-classifier"
        elif r.endswith("-logistic"):
            label = "logistic"
        elif r in SVC_KERNEL:
            label = SVC_KERNEL[r]
        else:
            continue
        bakeoff.append({"label": label, "auc": auc(r), "run": r})
    bakeoff.sort(key=lambda d: -d["auc"])

    # --- 3-seed noise band: xgboost-classifier on metadata-only, seeds 42/17/101 ---
    seed_runs = {
        42: "run-20260609-172920-627d6-xgboost-classifier",
        17: "run-20260609-172933-684f2-xgboost-classifier",
        101: "run-20260609-172933-6df93-xgboost-classifier",
    }
    seeds = {s: auc(r) for s, r in seed_runs.items()}
    vals = list(seeds.values())
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    # --- feature-variant: metadata-only vs +64 AE-latent cols (dilution) ---
    variants = [
        {"label": "metadata-only (115 cols)", "auc": auc(seed_runs[42]),
         "ds": CHAMP_DS},
        {"label": "+ 64 AE-latent (178 cols)",
         "auc": auc("run-20260609-172933-fffa4-xgboost-classifier"),
         "ds": "ds-20260608-204402-p64-s42"},
    ]

    out = {
        "champion_dataset": CHAMP_DS,
        "bakeoff": bakeoff,
        "noise_band": {"seeds": seeds, "mean": mean, "std": std, "two_sigma": 2 * std},
        "feature_variants": variants,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
