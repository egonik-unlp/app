#!/usr/bin/env python3
"""Bake the promoted next-track champion `blend-gru-markov-content-proj`
(R'+M+C' z-blend over a learned-projected PCA-192 item space) into a single
static asset `public/data/champion.bin`, loaded once per isolate by the
Rust/WASM core (worker-core/src/champion.rs). No torch / numpy at runtime.

The champion is a z-blend of three legs over the LEARNED-PROJECTED item space
(projected = L2-normalize(item_latents @ W.T)):
  * R' = single-layer GRU (hidden 256) over projected prefix latents -> next
         latent (192-d) -> cosine retrieval,
  * M  = first-order Markov (track bigram + artist/genre back-off), the exact
         formula from worker-core/src/transit.rs::affinity, rebuilt from the
         champion's TRAIN sessions (leak-free), PLUS the 1e-6*pop tie-break tail
         that seq_baselines.markov_scorer adds,
  * C' = content-kNN: max cosine of each candidate's projected latent to the
         prefix items' projected latents.
Blend = z-normalize each leg over the CANDIDATES (items not in the prefix),
equal-thirds average; rank descending (stable); exclude prefix items; top-k.

Everything reuses the champion's own artifacts + the reference predictor code
(imported from the sibling next-track instance's `predictors/`), so the baked
numbers are byte-identical to how the model was trained/evaluated.

Run with the app's venv (has numpy + torch):
  server/.venv/bin/python tools/build_champion.py

==========================================================================
champion.bin byte layout  (all multi-byte values little-endian)
==========================================================================
  off  type                     field
  0    u8[4]                    magic = "PFCH"
  4    u32                      version = 1
  8    u32                      n_items                 (= 19402)
  12   u32                      dim        (projected/rank = 192)
  16   u32                      hidden                  (= 256, GRU width)
  20   f32[n_items*dim]         proj    projected + L2-normalized latents,
                                        row-major (row = item index). Written
                                        EXACTLY as the reference produces it:
                                        (item_latents.f32 @ W.T) then f32
                                        row-normalize (+1e-12), astype f32.
       --- GRU state_dict (torch row-major; gate order r,z,n) ---
       f32[3*hidden*dim]        gru_w_ih   rnn.weight_ih_l0  (768 x 192)
       f32[3*hidden*hidden]     gru_w_hh   rnn.weight_hh_l0  (768 x 256)
       f32[3*hidden]            gru_b_ih   rnn.bias_ih_l0    (768)
       f32[3*hidden]            gru_b_hh   rnn.bias_hh_l0    (768)
       f32[dim*hidden]          head_w     head.weight       (192 x 256)
       f32[dim]                 head_b     head.bias         (192)
       --- Markov popularity tie-break tail ---
       f32[n_items]             pop_counts  per-item TRAIN occurrence count
                                            (train_play_counts); the leg adds
                                            1e-6 * counts/(max+1e-9).
       --- variable-length JSON sidecars ---
       u32                      trans_len
       u8[trans_len]            transitions_json   transit.rs TransitionModel
                                                   schema: {track_bigram,
                                                   artist_cond, genre_cond,
                                                   ctx_genre_lift:{},
                                                   ctx_track_lift:{}} keyed by
                                                   uri / artist / genre.
       u32                      items_len
       u8[items_len]            items_json   JSON array of n_items objects
                                             {uri,name,artist,genre} in item
                                             index order.
==========================================================================
"""
from __future__ import annotations

import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

APP_ROOT = Path(__file__).resolve().parents[1]
NEXTTRACK = APP_ROOT.parent / "spotify-next-track"
MODEL_DIR = NEXTTRACK / "data" / "models" / "blend-gru-markov-content-proj"
OUT = APP_ROOT / "public" / "data" / "champion.bin"

# Reuse the reference predictor plumbing so the baked stats are byte-identical
# to how the champion was trained/evaluated.
sys.path.insert(0, str(NEXTTRACK / "predictors"))
from seq_common import load_artifact                       # noqa: E402
from seq_baselines import train_play_counts                # noqa: E402

# Affinity constants (mirrors transit.rs / seq_baselines).
ARTIST_DEFAULT = "?"
GENRE_DEFAULT = "unknown"


def build_uri_transitions(art) -> dict:
    """Rebuild the first-order transitions from TRAIN sessions, keyed by
    uri / artist / genre so worker-core/src/transit.rs::TransitionModel can
    consume them and ::affinity reproduces seq_baselines.markov_scorer.

    Mirrors seq_baselines.build_train_transitions exactly (consecutive
    within-train-session pairs, self-loops dropped, unit weight), but emits
    STRING keys: track_bigram holds RAW counts (affinity normalizes by the
    row sum n_u); artist_cond / genre_cond are normalized to probabilities."""
    def uri(i):
        return art.items[str(int(i))]["uri"]

    def artist(i):
        return art.items.get(str(int(i)), {}).get("artist", ARTIST_DEFAULT)

    def genre(i):
        return art.items.get(str(int(i)), {}).get("genre", GENRE_DEFAULT)

    track_bigram: dict[str, Counter] = defaultdict(Counter)
    artist_bigram: dict[str, Counter] = defaultdict(Counter)
    genre_bigram: dict[str, Counter] = defaultdict(Counter)
    n_pairs = 0
    for s in art.train_sessions:
        seq = art.session(int(s))
        for a, b in zip(seq[:-1], seq[1:]):
            a, b = int(a), int(b)
            if a == b:
                continue
            track_bigram[uri(a)][uri(b)] += 1.0
            artist_bigram[artist(a)][artist(b)] += 1.0
            genre_bigram[genre(a)][genre(b)] += 1.0
            n_pairs += 1

    def conditional(bigram):
        cond = {}
        for src, tgts in bigram.items():
            tot = sum(tgts.values())
            if tot > 0:
                cond[src] = {d: w / tot for d, w in tgts.items()}
        return cond

    return {
        "track_bigram": {u: dict(c) for u, c in track_bigram.items()},
        "artist_cond": conditional(artist_bigram),
        "genre_cond": conditional(genre_bigram),
        "ctx_genre_lift": {},
        "ctx_track_lift": {},
        "meta": {"n_pairs": n_pairs, "source": "champion train sessions"},
    }


def main() -> None:
    assert MODEL_DIR.is_dir(), f"champion model dir not found: {MODEL_DIR}"
    art = load_artifact(MODEL_DIR)
    n_items = art.n_items
    D_in = art.latent_dim  # 192 (raw PCA latent dim)

    # ---- projected + L2-normalized latents (EXACT reference recipe) --------
    W = np.load(MODEL_DIR / "projection.npz")["W"]  # (rank, D_in) float32
    rank = int(W.shape[0])
    # Mirror seq_blend._projected_artifact byte-for-byte:
    Zp = art.item_latents.astype(np.float32) @ W.T          # (n_items, rank) f32
    Zp = Zp / (np.linalg.norm(Zp, axis=1, keepdims=True) + 1e-12)
    Zp = Zp.astype(np.float32)
    assert Zp.shape == (n_items, rank)

    # ---- GRU weights ------------------------------------------------------
    sd = torch.load(MODEL_DIR / "model.pt", map_location="cpu")
    hidden = sd["rnn.bias_ih_l0"].shape[0] // 3
    assert hidden == 256, f"unexpected hidden {hidden}"
    assert sd["rnn.weight_ih_l0"].shape == (3 * hidden, rank)
    assert sd["rnn.weight_hh_l0"].shape == (3 * hidden, hidden)
    assert sd["head.weight"].shape == (rank, hidden)

    def f32(t):
        return t.detach().cpu().numpy().astype("<f4").ravel(order="C")

    gru_w_ih = f32(sd["rnn.weight_ih_l0"])
    gru_w_hh = f32(sd["rnn.weight_hh_l0"])
    gru_b_ih = f32(sd["rnn.bias_ih_l0"])
    gru_b_hh = f32(sd["rnn.bias_hh_l0"])
    head_w = f32(sd["head.weight"])
    head_b = f32(sd["head.bias"])

    # ---- Markov popularity tail (train_play_counts) -----------------------
    pop_counts = train_play_counts(art).astype("<f4")       # (n_items,)
    assert pop_counts.shape == (n_items,)

    # ---- Markov transitions (uri/artist/genre keyed) ----------------------
    transitions = build_uri_transitions(art)
    trans_bytes = json.dumps(transitions, separators=(",", ":")).encode("utf-8")

    # ---- items metadata (index order) -------------------------------------
    items_arr = []
    for i in range(n_items):
        m = art.items[str(i)]
        items_arr.append({
            "uri": m["uri"],
            "name": m.get("name", ""),
            "artist": m.get("artist", ARTIST_DEFAULT),
            "genre": m.get("genre", GENRE_DEFAULT),
        })
    items_bytes = json.dumps(items_arr, separators=(",", ":")).encode("utf-8")

    # ---- write champion.bin ----------------------------------------------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as fh:
        fh.write(b"PFCH")
        fh.write(struct.pack("<IIII", 1, n_items, rank, hidden))
        fh.write(Zp.astype("<f4").tobytes(order="C"))
        fh.write(gru_w_ih.tobytes())
        fh.write(gru_w_hh.tobytes())
        fh.write(gru_b_ih.tobytes())
        fh.write(gru_b_hh.tobytes())
        fh.write(head_w.tobytes())
        fh.write(head_b.tobytes())
        fh.write(pop_counts.tobytes())
        fh.write(struct.pack("<I", len(trans_bytes)))
        fh.write(trans_bytes)
        fh.write(struct.pack("<I", len(items_bytes)))
        fh.write(items_bytes)

    # ---- register in public/data/manifest.json (merge, don't clobber) -----
    manifest_path = OUT.parent / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}
    import datetime as _dt
    manifest["champion"] = {
        "name": "blend-gru-markov-content-proj",
        "legs": "R'+M+C'",
        "file": "champion.bin",
        "n_items": int(n_items),
        "dim": int(rank),
        "hidden": int(hidden),
        "bytes": int(OUT.stat().st_size),
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    size = OUT.stat().st_size
    print(f"wrote {OUT}  ({size/1e6:.2f} MB)")
    print(f"  manifest.champion registered in {manifest_path}")
    print(f"  n_items={n_items} dim(rank)={rank} hidden={hidden}")
    print(f"  proj floats={Zp.size}  transitions={len(trans_bytes)}B "
          f"items={len(items_bytes)}B")
    print(f"  track_bigram srcs={len(transitions['track_bigram'])} "
          f"artist_cond={len(transitions['artist_cond'])} "
          f"genre_cond={len(transitions['genre_cond'])} "
          f"n_pairs={transitions['meta']['n_pairs']}")


if __name__ == "__main__":
    main()
