#!/usr/bin/env python3
"""Offline snapshot builder for the Cloudflare-Worker-native pathfinder.

Reuses the proven logic in ``server/app.py`` (Qdrant load, symmetric KNN, ONNX
fit scoring, cold-start feature assembly) and emits the static artifacts the
Rust/WASM core consumes at runtime:

    public/data/corpus.bin        ids + latents + fit + KNN + display metadata
    public/data/transitions.json  copied verbatim from the sibling pathfinder
    public/data/projector.bin     cold-start projector MLP + feature config  (optional)
    public/data/manifest.json     counts / dims / provenance

Run (from app/):  server/.venv/bin/python tools/build_snapshot.py [--projector] [--limit N]

The corpus snapshot (Phase A) needs only Qdrant + onnxruntime. The projector
(Phase B, ``--projector``) additionally needs torch and a bge-m3 embedding
backend: Cloudflare Workers AI (CF_ACCOUNT_ID + CF_AI_TOKEN) — preferred so it
matches the runtime AI binding — or a local ``sentence-transformers`` install.
See tools/SNAPSHOT_FORMAT.md for byte layouts.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import server.app as A  # noqa: E402  (reuses load_graph / OnnxScorer / content_doc / build_input_row)

OUT = ROOT / "public" / "data"
ARTIFACTS = A.SIBLING / "pipeline" / "artifacts"
TEXT_MODEL_CF = "@cf/baai/bge-m3"


# ---------------------------------------------------------------------------
# corpus.bin
# ---------------------------------------------------------------------------
def write_corpus(g: A.TrackGraph, scores: dict[int, float], path: Path) -> None:
    n = len(g.ids)
    dim = int(g.vectors.shape[1])
    k = A.KNN_K
    row = {pid: i for i, pid in enumerate(g.ids)}

    ids = np.asarray(g.ids, dtype="<u8")
    vecs = np.ascontiguousarray(g.vectors, dtype="<f4")
    fit = np.asarray([scores.get(pid, 0.5) for pid in g.ids], dtype="<f4")

    nbr_idx = np.full((n, k), 0xFFFFFFFF, dtype="<u4")
    nbr_dist = np.zeros((n, k), dtype="<f4")
    for i, pid in enumerate(g.ids):
        for j, (other, dist) in enumerate(g.neighbors.get(pid, [])[:k]):
            nbr_idx[i, j] = row[other]
            nbr_dist[i, j] = dist

    meta = [
        {
            "uri": m.get("track_uri", ""),
            "name": m.get("track_name") or "Unknown track",
            "artist": m.get("artist") or "Unknown artist",
            "album": m.get("album"),
            "genre": m.get("genre_primary") or "unknown",
            "release_year": m.get("release_year"),
        }
        for m in (g.meta[pid] for pid in g.ids)
    ]
    meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    with path.open("wb") as f:
        f.write(b"PFC1")
        f.write(struct.pack("<IIII", 1, n, dim, k))
        f.write(ids.tobytes())
        f.write(vecs.tobytes())
        f.write(fit.tobytes())
        f.write(nbr_idx.tobytes())
        f.write(nbr_dist.tobytes())
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
    print(f"  corpus.bin: {n} tracks, dim={dim}, k={k}  ({path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# projector.bin  (cold-start)
# ---------------------------------------------------------------------------
def embed_texts_cf(texts: list[str], account: str, token: str, model: str) -> np.ndarray:
    import requests

    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    out: list[list[float]] = []
    B = 100
    for i in range(0, len(texts), B):
        batch = texts[i : i + B]
        for attempt in range(5):
            r = requests.post(url, headers=headers, json={"text": batch}, timeout=120)
            if r.status_code == 200:
                break
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Workers AI embed failed: {r.status_code} {r.text[:200]}")
        data = r.json()["result"]["data"]
        out.extend(data)
        print(f"    embedded {min(i+B, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    return np.asarray(out, dtype=np.float32)


def embed_texts_local(texts: list[str], model: str = "BAAI/bge-m3") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"    loading local {model} (bge-m3 weights match @cf/baai/bge-m3) ...")
    st = SentenceTransformer(model)
    return st.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True).astype(np.float32)


def build_projector_features(pre: dict, g: A.TrackGraph, text_emb: np.ndarray) -> np.ndarray:
    """Assemble [text ++ z(num) ++ z(ac) ++ flags ++ genre_multihot ++ album_onehot]
    in the SAME order as the runtime worker (matches app.py build_input_row tail),
    for every corpus track."""
    import math

    feats = []

    def z(value, st):
        if value is None:
            v = st["mean"]
        else:
            v = float(value)
            if st.get("log") and v >= 0:
                v = math.log1p(v)
        return (v - st["mean"]) / st["std"]

    for i, pid in enumerate(g.ids):
        m = g.meta[pid]
        num = [z(m.get(kk), pre["num_stats"][kk]) for kk in pre["numerics"]]
        ac, flags = [], []
        for kk in pre["acoustics"]:
            v = m.get(kk)
            st = pre["ac_stats"][kk]
            present = isinstance(v, (int, float))
            ac.append(((float(v) if present else st["mean"]) - st["mean"]) / st["std"])
            flags.append(0.0 if present else 1.0)
        genres = set(m.get("sp_genres") or [])
        if m.get("genre_primary"):
            genres.add(m["genre_primary"])
        gmat = [1.0 if gg in genres else 0.0 for gg in pre["genre_vocab"]]
        amat = [1.0 if m.get("album_type") == aa else 0.0 for aa in pre["album_types"]]
        tail = np.asarray(num + ac + flags + gmat + amat, dtype=np.float32)
        feats.append(np.concatenate([text_emb[i], tail]))
    return np.vstack(feats).astype(np.float32)


def train_projector(g: A.TrackGraph, path: Path, args) -> str | None:
    import torch
    import torch.nn as nn

    pre = json.loads((ARTIFACTS / "song_ae_preprocess.json").read_text())
    account = os.environ.get("CF_ACCOUNT_ID", "").strip()
    token = os.environ.get("CF_AI_TOKEN", "").strip()

    texts = [A.content_doc(g.meta[pid]) for pid in g.ids]
    if account and token:
        print(f"  embedding {len(texts)} docs via Workers AI {TEXT_MODEL_CF} ...")
        text_emb = embed_texts_cf(texts, account, token, TEXT_MODEL_CF)
    else:
        try:
            text_emb = embed_texts_local(texts)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip projector] no bge backend (set CF_ACCOUNT_ID+CF_AI_TOKEN or pip install sentence-transformers): {e}")
            return None
    text_dim = int(text_emb.shape[1])

    X = build_projector_features(pre, g, text_emb)
    Y = np.ascontiguousarray(g.vectors, dtype=np.float32)  # existing L2-normalized latents
    in_dim = X.shape[1]
    hidden = 256
    out_dim = Y.shape[1]
    print(f"  projector: in_dim={in_dim} (text {text_dim}) hidden={hidden} out_dim={out_dim}")

    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)

    net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    n = X.shape[0]
    idx = np.arange(n)
    for epoch in range(args.epochs):
        net.train()
        np.random.shuffle(idx)
        total = 0.0
        for s in range(0, n, 512):
            b = idx[s : s + 512]
            xb = Xt[b]
            yb = Yt[b]
            pred = net(xb)
            predn = pred / (pred.norm(dim=1, keepdim=True) + 1e-9)
            loss = (1.0 - (predn * yb).sum(dim=1)).mean()  # cosine loss vs normalized target
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"    epoch {epoch:3d}  cos-loss {total/n:.4f}")

    # self-recall sanity: projector(track) nearest corpus neighbor should be itself
    net.eval()
    with torch.no_grad():
        P = net(Xt).numpy()
    P /= np.linalg.norm(P, axis=1, keepdims=True).clip(min=1e-9)
    sample = idx[:2000]
    sims = P[sample] @ Y.T
    nn_idx = sims.argmax(axis=1)
    recall = float((nn_idx == sample).mean())
    print(f"  projector self-recall@1 (2k sample): {recall:.3f}")

    w = list(net.parameters())
    W1 = w[0].detach().numpy().astype("<f4")
    b1 = w[1].detach().numpy().astype("<f4")
    W2 = w[2].detach().numpy().astype("<f4")
    b2 = w[3].detach().numpy().astype("<f4")
    cfg = {
        "text_model": TEXT_MODEL_CF,
        "numerics": pre["numerics"],
        "num_stats": pre["num_stats"],
        "acoustics": pre["acoustics"],
        "ac_stats": pre["ac_stats"],
        "genre_vocab": pre["genre_vocab"],
        "album_types": pre["album_types"],
    }
    cfg_bytes = json.dumps(cfg, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(b"PFP1")
        f.write(struct.pack("<IIIII", 1, in_dim, hidden, out_dim, text_dim))
        f.write(W1.tobytes())
        f.write(b1.tobytes())
        f.write(W2.tobytes())
        f.write(b2.tobytes())
        f.write(struct.pack("<I", len(cfg_bytes)))
        f.write(cfg_bytes)
    print(f"  projector.bin written ({path.stat().st_size/1e6:.2f} MB)")
    return TEXT_MODEL_CF


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projector", action="store_true", help="also train + emit the cold-start projector")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0, help="cap track count (debug)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Qdrant: {A.QDRANT_URL}  collection: {A.COLLECTION}")
    print("loading graph (scroll + KNN) ...")
    g = A.load_graph()
    if args.limit and args.limit < len(g.ids):
        keep = set(g.ids[: args.limit])
        g.ids = g.ids[: args.limit]
        g.vectors = g.vectors[: args.limit]
        g.meta = {pid: g.meta[pid] for pid in g.ids}
        g.learned_ids = set(g.ids)
        g._row = {pid: i for i, pid in enumerate(g.ids)}
        A.build_knn(g)
        print(f"  (debug) limited to {len(g.ids)} tracks")
    print(f"loaded {len(g.ids)} tracks")

    print("scoring fit (ONNX) ...")
    scorer = A.OnnxScorer(A.MODEL_DIR)
    scores = scorer.score_graph(g)

    write_corpus(g, scores, OUT / "corpus.bin")

    # scorer.bin: the same fit model + featurize spec, for live (cold-start)
    # scoring in the Worker. Independent of the corpus, but emitted here so a
    # single build keeps corpus fit and live fit in lock-step.
    import tools.export_scorer as E  # noqa: PLC0415

    print("exporting scorer.bin (live fit model) ...")
    E.write_scorer_bin(E.load_params(A.MODEL_DIR), OUT / "scorer.bin")

    src_trans = A.TRANSITIONS_JSON
    (OUT / "transitions.json").write_bytes(src_trans.read_bytes())
    print(f"  transitions.json copied ({(OUT/'transitions.json').stat().st_size/1e6:.1f} MB)")

    text_model = None
    if args.projector:
        print("training cold-start projector ...")
        text_model = train_projector(g, OUT / "projector.bin", args)

    manifest = {
        "version": 1,
        "n_tracks": len(g.ids),
        "dim": int(g.vectors.shape[1]),
        "knn_k": A.KNN_K,
        "has_projector": text_model is not None,
        "text_model": text_model,
        # raw-score bounds used to normalize live cold-start fit onto the baked scale
        "fit_raw_min": scorer.raw_min,
        "fit_raw_max": scorer.raw_max,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {manifest}")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
