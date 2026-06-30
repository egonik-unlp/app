#!/usr/bin/env python3
"""Export the rotation-fit ONNX model to a self-contained ``scorer.bin`` (PFS1)
for the Rust/WASM worker core, and generate a parity fixture for the Rust tests.

The model ``burn-deep-embeddings-highvocab`` is a fixed graph (verified below):

    input[497] = cont(71) ++ onehot_genre(121) ++ onehot_artist(301) ++ onehot_album(4)
    cont_std   = (input[0:71]   - cont_mean) / cont_std          # Sub / Div
    e_genre    =  input[71:192]  @ embed_w0[121,16]              # MatMul
    e_artist   =  input[192:493] @ embed_w1[301,16]              # MatMul
    e_album    =  input[493:497] @ embed_w2[4,4]                 # MatMul
    h          = concat(cont_std, e_genre, e_artist, e_album)    # 107
    h          = relu(h @ trunk_w0 + trunk_b0)                   # Gemm/Relu -> 128
    h          = relu(h @ trunk_w1 + trunk_b1)                   # Gemm/Relu -> 64
    logits     = h @ head_w + head_b                             # Gemm -> 2
    raw        = softmax(logits)[1]                              # Softmax / Gather

The 497-vector itself is assembled per ``featurize.json`` (see OnnxScorer.features
in server/app.py) — this exporter ships that spec verbatim so the Rust featurizer
is a 1:1 port.

scorer.bin layout (cfg-first, all floats little-endian f32):

    "PFS1"  version:u32=1  cfg_len:u32  cfg_json:bytes  blobs...

``cfg_json`` declares pca/columns/network dims and a ``blobs`` list giving the
order + shape of each trailing f32 blob, so the Rust parser is fully data-driven.

Run standalone:
    server/.venv/bin/python tools/export_scorer.py \
        [--out public/data/scorer.bin] [--fixture worker-core/tests/fixtures/scorer_parity.json]
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import server.app as A  # noqa: E402  (reuses OnnxScorer + MODEL_DIR)

# Expected ONNX op sequence — a re-export with a different architecture must fail
# loudly rather than silently mis-port.
EXPECTED_OPS = [
    "Gather", "Sub", "Div",            # continuous standardize
    "Gather", "MatMul",                # genre embedding
    "Gather", "MatMul",                # artist embedding
    "Gather", "MatMul",                # album embedding
    "Concat",
    "Gemm", "Relu", "Gemm", "Relu", "Gemm", "Softmax", "Gather",
]


def load_params(model_dir: Path) -> dict:
    """Read every initializer + featurize spec, asserting the wiring is the one
    this exporter knows how to serialize."""
    import onnx
    from onnx import numpy_helper

    g = onnx.load(str(model_dir / "model.onnx")).graph
    ops = [n.op_type for n in g.node]
    if ops != EXPECTED_OPS:
        raise SystemExit(
            f"unexpected ONNX op sequence:\n  got {ops}\n  want {EXPECTED_OPS}\n"
            "the scorer architecture changed — update export_scorer.py + scorer.rs together."
        )
    w = {init.name: numpy_helper.to_array(init) for init in g.initializer}

    # column-partition indices must be contiguous and match the featurize order
    idx1, idx5, idx8, idx11 = (w[k] for k in ("idx_1", "idx_5", "idx_8", "idx_11"))
    blocks = {"cont": len(idx1), "genre": len(idx5), "artist": len(idx8), "album": len(idx11)}
    expected = np.arange(497)
    if not np.array_equal(np.concatenate([idx1, idx5, idx8, idx11]), expected):
        raise SystemExit("input column partition is not the contiguous 0..496 layout we assume")
    if int(w["idx_21"][0]) != 1:
        raise SystemExit(f"class index is {int(w['idx_21'][0])}, expected 1")

    feat = json.loads((model_dir / "featurize.json").read_text())
    if feat["n_cols"] != 497 or feat["pca"]["components_shape"] != [64, 64]:
        raise SystemExit("featurize.json shape changed — revisit the export")
    components = np.fromfile(
        model_dir / feat["pca"]["components_file"], dtype="<f4"
    ).reshape(tuple(feat["pca"]["components_shape"]))
    mean = np.asarray(feat["pca"]["mean"], dtype=np.float32)

    return {
        "blocks": blocks,
        "embed": {
            "genre": int(w["embed_w0"].shape[1]),
            "artist": int(w["embed_w1"].shape[1]),
            "album": int(w["embed_w2"].shape[1]),
        },
        "trunk": [int(w["trunk_w0"].shape[1]), int(w["trunk_w1"].shape[1])],
        "n_class": int(w["head_w"].shape[1]),
        "columns": feat["columns"],
        "pca_dims": int(feat["pca"]["dims"]),
        # f32 blobs, in the fixed order written below
        "pca_components": components.astype(np.float32),
        "pca_mean": mean,
        "cont_mean": w["cont_mean"].astype(np.float32),
        "cont_std": w["cont_std"].astype(np.float32),
        "embed_genre": w["embed_w0"].astype(np.float32),
        "embed_artist": w["embed_w1"].astype(np.float32),
        "embed_album": w["embed_w2"].astype(np.float32),
        "trunk_w0": w["trunk_w0"].astype(np.float32),
        "trunk_b0": w["trunk_b0"].astype(np.float32),
        "trunk_w1": w["trunk_w1"].astype(np.float32),
        "trunk_b1": w["trunk_b1"].astype(np.float32),
        "head_w": w["head_w"].astype(np.float32),
        "head_b": w["head_b"].astype(np.float32),
    }


# blob name -> attribute, in the exact serialization order the Rust parser reads
BLOB_ORDER = [
    "pca_components", "pca_mean", "cont_mean", "cont_std",
    "embed_genre", "embed_artist", "embed_album",
    "trunk_w0", "trunk_b0", "trunk_w1", "trunk_b1", "head_w", "head_b",
]


def forward_np(p: dict, feats: np.ndarray) -> float:
    """NumPy reference forward over a single assembled 497-vector. This is the
    exact arithmetic the Rust port must reproduce; we assert it matches
    onnxruntime over the fixture below."""
    b = p["blocks"]
    o_cont = 0
    o_gen = o_cont + b["cont"]
    o_art = o_gen + b["genre"]
    o_alb = o_art + b["artist"]
    cont = (feats[o_cont:o_gen] - p["cont_mean"]) / p["cont_std"]
    e_gen = feats[o_gen:o_art] @ p["embed_genre"]
    e_art = feats[o_art:o_alb] @ p["embed_artist"]
    e_alb = feats[o_alb:o_alb + b["album"]] @ p["embed_album"]
    h = np.concatenate([cont, e_gen, e_art, e_alb])
    h = np.maximum(0.0, h @ p["trunk_w0"] + p["trunk_b0"])
    h = np.maximum(0.0, h @ p["trunk_w1"] + p["trunk_b1"])
    logits = h @ p["head_w"] + p["head_b"]
    z = logits - logits.max()
    sm = np.exp(z) / np.exp(z).sum()
    raw = float(sm[1])
    return _nan_clip(raw)


def _nan_clip(x: float) -> float:
    if math.isnan(x):
        return 0.5
    if x == math.inf:
        return 1.0
    if x == -math.inf:
        return 0.0
    return min(1.0, max(0.0, x))


def write_scorer_bin(p: dict, out: Path) -> None:
    cfg = {
        "blocks": p["blocks"],
        "embed": p["embed"],
        "trunk": p["trunk"],
        "n_class": p["n_class"],
        "class_index": 1,
        "pca_dims": p["pca_dims"],
        "columns": p["columns"],
        "blobs": [
            {"name": name, "shape": list(np.asarray(p[name]).shape)} for name in BLOB_ORDER
        ],
    }
    cfg_bytes = json.dumps(cfg, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(b"PFS1")
        f.write(struct.pack("<II", 1, len(cfg_bytes)))
        f.write(cfg_bytes)
        for name in BLOB_ORDER:
            f.write(np.ascontiguousarray(p[name], dtype="<f4").tobytes())
    print(f"  scorer.bin: {len(p['columns'])} cols, trunk={p['trunk']}  ({out.stat().st_size/1e3:.1f} KB)")


# --- fixture (synthetic cold-start-shaped inputs) ---------------------------
def _genre_artist_vocab(p: dict) -> tuple[list[str], list[str], list[str]]:
    g = [c["value"] for c in p["columns"] if c.get("group") == "genre_primary" and c["value"] != "__other__"]
    a = [c["value"] for c in p["columns"] if c.get("group") == "artist" and c["value"] != "__other__"]
    al = [c["value"] for c in p["columns"] if c.get("group") == "album_type" and c["value"] != "__other__"]
    return g, a, al


def make_fixture(model_dir: Path, p: dict, out: Path, n: int, seed: int) -> None:
    scorer = A.OnnxScorer(model_dir)  # reuse the production featurizer + ONNX session
    rng = np.random.default_rng(seed)
    genres, artists, albums = _genre_artist_vocab(p)
    cases = []
    for i in range(n):
        latent = rng.standard_normal(64).astype(np.float32)
        latent /= np.linalg.norm(latent) + 1e-9  # L2-normalized, like real latents
        # vary coverage: known/unknown artist+genre, missing numerics, odd album types
        meta = {
            "artist_popularity": None if i % 5 == 0 else int(rng.integers(0, 100)),
            "track_popularity": int(rng.integers(0, 100)),
            "artist_followers": None if i % 7 == 0 else int(rng.integers(0, 5_000_000)),
            "release_year": None if i % 11 == 0 else int(rng.integers(1960, 2026)),
            "artist_count": int(rng.integers(1, 4)),
            "genre_primary": (genres[i % len(genres)] if i % 3 else "totally-unknown-genre"),
            "artist": (artists[i % len(artists)] if i % 2 else "Some Unseen Artist"),
            "album_type": (albums[i % len(albums)] if i % 4 else "weird_type"),
        }
        feats = scorer.features(meta, latent)
        out_raw = scorer.session.run([scorer.output_name], {scorer.input_name: feats[None, :]})[0].reshape(-1)
        out_raw = float(np.clip(np.nan_to_num(out_raw, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)[0])
        # guard: our numpy reference forward must match onnxruntime
        ref = forward_np(p, feats.astype(np.float32))
        if abs(ref - out_raw) > 1e-5:
            raise SystemExit(f"numpy forward disagrees with onnxruntime at case {i}: {ref} vs {out_raw}")
        cases.append({
            "latent": [float(x) for x in latent],
            "meta": meta,
            "feats": [float(x) for x in feats],
            "raw": out_raw,
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": n, "seed": seed, "cases": cases}, separators=(",", ":")))
    print(f"  fixture: {n} cases -> {out}  (numpy forward matches onnxruntime within 1e-5)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(A.MODEL_DIR))
    ap.add_argument("--out", default=str(ROOT / "public" / "data" / "scorer.bin"))
    ap.add_argument("--fixture", default=str(ROOT / "worker-core" / "tests" / "fixtures" / "scorer_parity.json"))
    ap.add_argument("--fixture-n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    print(f"loading model: {model_dir.name}")
    p = load_params(model_dir)
    print(f"  blocks={p['blocks']} embed={p['embed']} trunk={p['trunk']} n_class={p['n_class']}")
    write_scorer_bin(p, Path(args.out))
    if args.fixture:
        fixture = Path(args.fixture)
        make_fixture(model_dir, p, fixture, args.fixture_n, args.seed)
        # drop a copy of the binary next to the fixture so the Rust parity test
        # is hermetic (doesn't depend on a freshly built public/data/scorer.bin)
        write_scorer_bin(p, fixture.with_name("scorer.bin"))
    print("done")


if __name__ == "__main__":
    main()
