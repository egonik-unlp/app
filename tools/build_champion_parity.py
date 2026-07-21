#!/usr/bin/env python3
"""Generate the PYTHON reference for the champion parity gate.

For a handful of sample prefixes (real item indices, lengths 1-4) it runs the
EXACT reference `seq_blend.py predict` subcommand (build_blend_score_fn +
rank_topk over the promoted champion dir) to capture the top-10 item indices,
and separately captures the raw GRU predicted-latent (head output, pre-
normalize) for two prefixes. Written to
worker-core/tests/fixtures/champion_parity.json, which worker-core/tests/
champion.rs asserts the Rust port matches exactly (top-10) / to ~1e-4 (latent).

Run with the next-track predictor venv (torch cpu, canonical):
  ../spotify-next-track/predictors/.venv/bin/python tools/build_champion_parity.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

APP_ROOT = Path(__file__).resolve().parents[1]
NEXTTRACK = APP_ROOT.parent / "spotify-next-track"
MODEL_DIR = NEXTTRACK / "data" / "models" / "blend-gru-markov-content-proj"
OUT = APP_ROOT / "worker-core" / "tests" / "fixtures" / "champion_parity.json"

os.environ.setdefault("LENSING_MUSIC_METRIC_DISABLE", "1")
sys.path.insert(0, str(NEXTTRACK / "predictors"))
import seq_blend as SB                                       # noqa: E402
from seq_common import load_artifact, resolve_prefix        # noqa: E402

# Sample prefixes as ordered item indices (oldest -> newest), lengths 1..4.
PREFIXES = [
    [5271],
    [10246, 15570],
    [6287, 4776, 6740],
    [16892, 18540, 6177, 3295],
    [11233, 6475, 11285, 8844],
]
K = 10


def reference_topk(prefix: list[int]) -> list[int]:
    """Run the real `seq_blend.py predict` for one prefix; return top-K ids."""
    with tempfile.TemporaryDirectory() as td:
        idir = Path(td)
        (idir / "prefix.json").write_text(json.dumps({"prefix": prefix, "k": K}))
        out = idir / "pred.json"
        SB.predict(MODEL_DIR, idir, out)
        preds = json.loads(out.read_text())
        return [int(i) for i in preds[0]["top_k_ids"]]


def raw_pred_latent(prefix: list[int]) -> list[float]:
    """The GRU's raw predicted-latent (head output, pre-normalize) for a prefix,
    over the PROJECTED artifact — exactly what build_blend_score_fn's GRU leg
    normalizes then dots against the vocab."""
    hp = SB.load_hp(str(MODEL_DIR / "hyperparams.json"))
    art = load_artifact(MODEL_DIR)
    W = np.load(MODEL_DIR / "projection.npz")["W"]
    art = SB._projected_artifact(art, W)
    model = SB.load_model(MODEL_DIR, art, hp)
    idx, unknown = resolve_prefix(art, prefix)
    assert not unknown
    x = torch.from_numpy(art.item_latents[idx][None, :, :])   # (1, T, D)
    with torch.no_grad():
        pred = model.predict_next(x)                          # (1, D) raw head out
    return [float(v) for v in pred.squeeze(0).numpy()]


def main() -> None:
    prefixes = []
    for p in PREFIXES:
        top = reference_topk(p)
        prefixes.append({"prefix": p, "k": K, "top_k_ids": top})
        print(f"prefix len {len(p)} {p} -> top10 {top}")

    latent_checks = []
    for p in (PREFIXES[0], PREFIXES[3]):
        lat = raw_pred_latent(p)
        latent_checks.append({"prefix": p, "pred_latent": lat})
        print(f"latent check prefix {p}: dim {len(lat)} "
              f"[{lat[0]:.5f}, {lat[1]:.5f}, ...]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"k": K, "prefixes": prefixes, "latent_checks": latent_checks}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
