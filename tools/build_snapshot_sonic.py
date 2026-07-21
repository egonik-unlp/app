#!/usr/bin/env python3
"""Offline snapshot builder — MUSICAL-DISTANCE ("sonic") variant (Path A).

Same corpus, same dim (64), SAME per-track fit scores — but the item-latent
DISTANCE space that drives the A* journey geometry is swapped from the raw
64-d song-AE latent to the 64-d ZCA-whitened AE "sonic" ("sounds-alike")
space, so route geometry is musical (genre/timbre coherent) rather than
dominated by loud popularity/era directions.

It reuses ``server/app.py`` (load_graph / build_knn / OnnxScorer) and
``tools/build_snapshot.py`` (write_corpus / train_projector) verbatim — the
ONLY change is the vector matrix and the KNN derived from it.

Provenance:
  distance space = sonic (whitened-AE) musical-distance, dim 64,
  source collection ``spotify_tracks_content_metric`` named vector ``sonic``
  @ http://localhost:6337, joined to the song-AE corpus by track uri.

fit scores stay computed on the AE latent (the rotation-fit model's native
input); fit is a per-track scalar independent of the distance geometry, so it
remains valid — exactly as the projector note in SNAPSHOT_FORMAT.md intends.

Run (from app/):
    server/.venv/bin/python tools/build_snapshot_sonic.py [--no-projector] [--epochs N] [--limit N]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# --- 1. point server.app at THIS instance's Qdrant (6337) BEFORE importing it ---
os.environ["QDRANT_URL"] = "http://localhost:6337"
os.environ["PATHFINDER_COLLECTION"] = "spotify_tracks_song_ae"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- 2. import the proven building blocks ---
import server.app as A  # noqa: E402
import tools.build_snapshot as B  # noqa: E402

OUT = ROOT / "public" / "data"
METRIC_COLLECTION = "spotify_tracks_content_metric"
SONIC_VECTOR = "sonic"
PROVENANCE = (
    "sonic (whitened-AE) musical-distance space, dim 64, "
    "source spotify_tracks_content_metric[sonic] @6337"
)


def to_id(uri: str) -> int:
    """Same id scheme the metric collection is keyed by (proven 19402/19402
    in the sibling next-track predictors/seq_common._load_music_vectors)."""
    return int.from_bytes(hashlib.sha256(uri.encode()).digest()[:8], "little")


def fetch_sonic(g: A.TrackGraph, batch: int = 256) -> dict[int, np.ndarray]:
    """Retrieve the sonic named-vector for every corpus track, keyed by the
    metric-collection id (to_id(uri))."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=A.QDRANT_URL, timeout=120)
    # metric id -> corpus uri (so we can map results back), plus the ordered
    # list of metric ids to retrieve.
    want_ids: list[int] = []
    for pid in g.ids:
        uri = (g.meta[pid] or {}).get("track_uri", "") or ""
        want_ids.append(to_id(uri))

    sonic_by_mid: dict[int, np.ndarray] = {}
    for s in range(0, len(want_ids), batch):
        chunk = want_ids[s : s + batch]
        recs = client.retrieve(
            METRIC_COLLECTION, ids=chunk, with_vectors=[SONIC_VECTOR], with_payload=False
        )
        for rec in recs:
            v = rec.vector
            if isinstance(v, dict):
                v = v.get(SONIC_VECTOR)
            if v is None:
                continue
            sonic_by_mid[int(rec.id)] = np.asarray(v, dtype=np.float32)
        print(f"    sonic fetch {min(s + batch, len(want_ids))}/{len(want_ids)}", end="\r", flush=True)
    print()
    return sonic_by_mid


def swap_to_sonic(g: A.TrackGraph, sonic_by_mid: dict[int, np.ndarray]) -> tuple[np.ndarray, int]:
    """Build the sonic matrix aligned to g.ids (float32, L2-normalized),
    falling back to the existing AE vector for any miss. Returns (matrix, hits)."""
    dim = None
    for v in sonic_by_mid.values():
        dim = int(v.shape[0])
        break
    if dim is None:
        raise RuntimeError("no sonic vectors fetched")

    ae = g.vectors  # already L2-normalized AE latents
    if ae.shape[1] != dim:
        raise RuntimeError(f"dim mismatch: AE {ae.shape[1]} vs sonic {dim}; cannot fall back cleanly")

    rows = np.empty((len(g.ids), dim), dtype=np.float32)
    hits = 0
    for i, pid in enumerate(g.ids):
        uri = (g.meta[pid] or {}).get("track_uri", "") or ""
        mid = to_id(uri)
        v = sonic_by_mid.get(mid)
        if v is not None:
            rows[i] = v
            hits += 1
        else:
            rows[i] = ae[i]  # fall back to the AE vector for this track
    rows /= np.linalg.norm(rows, axis=1, keepdims=True).clip(min=1e-9)
    return rows, hits


def _topk_neighbors(vectors: np.ndarray, i: int, k: int = 5) -> list[tuple[int, float]]:
    sims = vectors @ vectors[i]
    sims[i] = -np.inf
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(j), float(sims[j])) for j in top]


def _label(g: A.TrackGraph, idx: int) -> str:
    m = g.meta[g.ids[idx]] or {}
    return (
        f"{(m.get('track_name') or '?')!r} — {m.get('artist') or '?'} "
        f"[{m.get('genre_primary') or 'unknown'}, {m.get('release_year') or '?'}]"
    )


def report_before_after(g: A.TrackGraph, ae_vectors: np.ndarray, sonic_vectors: np.ndarray, n: int = 3) -> None:
    """Pick n corpus tracks with distinct primary genres and print top-5 KNN
    neighbors in the AE (before) vs sonic (after) space."""
    seen_genres: set[str] = set()
    picks: list[int] = []
    for i, pid in enumerate(g.ids):
        gen = (g.meta[pid] or {}).get("genre_primary") or ""
        if gen and gen not in seen_genres:
            seen_genres.add(gen)
            picks.append(i)
        if len(picks) >= n:
            break
    print("\n==================== BEFORE / AFTER neighbor geometry ====================")
    for i in picks:
        print(f"\nSEED  {_label(g, i)}")
        print("  BEFORE (raw song-AE latent):")
        for j, sim in _topk_neighbors(ae_vectors, i):
            print(f"    cos={sim:.3f}  {_label(g, j)}")
        print("  AFTER  (sonic / whitened-AE musical-distance):")
        for j, sim in _topk_neighbors(sonic_vectors, i):
            print(f"    cos={sim:.3f}  {_label(g, j)}")
    print("==========================================================================\n")


def write_manifest(g: A.TrackGraph, scorer: A.OnnxScorer, text_model: str | None, projector_stale: bool) -> dict:
    manifest = {
        "version": 1,
        "n_tracks": len(g.ids),
        "dim": int(g.vectors.shape[1]),
        "knn_k": A.KNN_K,
        "has_projector": text_model is not None,
        "text_model": text_model,
        "fit_raw_min": scorer.raw_min,
        "fit_raw_max": scorer.raw_max,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # --- sonic-variant provenance (extends the base schema) ---
        "distance_space": PROVENANCE,
        "projector_stale": projector_stale,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-projector", action="store_true", help="skip the cold-start projector retrain entirely")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0, help="cap track count (debug)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Qdrant: {A.QDRANT_URL}  corpus collection: {A.COLLECTION}")
    print(f"sonic source: {METRIC_COLLECTION}[{SONIC_VECTOR}]")

    # 3. load ids/uris/meta + AE latents from 6337, AE-based KNN (throwaway).
    print("loading graph (scroll + AE KNN) ...")
    g = A.load_graph()
    if args.limit and args.limit < len(g.ids):
        keep_ids = g.ids[: args.limit]
        g.ids = keep_ids
        g.vectors = g.vectors[: args.limit]
        g.meta = {pid: g.meta[pid] for pid in g.ids}
        g.learned_ids = set(g.ids)
        g._row = {pid: i for i, pid in enumerate(g.ids)}
        print(f"  (debug) limited to {len(g.ids)} tracks")
    print(f"loaded {len(g.ids)} tracks")

    # keep the AE latents for fit scoring (native input) + before/after report.
    ae_vectors = np.ascontiguousarray(g.vectors, dtype=np.float32).copy()

    # score fit on the AE latent (per-track scalar; distance-space independent).
    print("scoring fit (ONNX, on AE latent) ...")
    scorer = A.OnnxScorer(A.MODEL_DIR)
    scores = scorer.score_graph(g)

    # 4. fetch sonic + swap the distance space.
    print("fetching sonic named-vectors ...")
    sonic_by_mid = fetch_sonic(g)
    if not sonic_by_mid:
        raise RuntimeError(f"no sonic vectors returned from {METRIC_COLLECTION}[{SONIC_VECTOR}] @ {A.QDRANT_URL}")
    sonic_vectors, hits = swap_to_sonic(g, sonic_by_mid)
    misses = len(g.ids) - hits
    print(f"  sonic coverage: {hits}/{len(g.ids)} ({100.0 * hits / len(g.ids):.2f}%)  misses(fallback→AE)={misses}")
    g.vectors = sonic_vectors

    # 5. recompute KNN on the sonic space.
    print("recomputing KNN on the sonic space ...")
    A.build_knn(g)

    # before/after neighbor demonstration.
    report_before_after(g, ae_vectors, sonic_vectors, n=3)

    # 6a. corpus.bin (fit scores are the AE-derived ones).
    B.write_corpus(g, scores, OUT / "corpus.bin")

    # 6b. transitions.json copied verbatim from the sibling file.
    (OUT / "transitions.json").write_bytes(A.TRANSITIONS_JSON.read_bytes())
    print(f"  transitions.json copied ({(OUT / 'transitions.json').stat().st_size / 1e6:.1f} MB)")

    # write a valid manifest NOW (before the slow projector step) so an
    # interrupted projector never leaves a stale/missing manifest.
    write_manifest(g, scorer, text_model=None, projector_stale=True)

    # 7. cold-start projector — retrain to regress the NEW sonic g.vectors,
    #    ONLY if a bge backend is available. Never fatal.
    text_model = None
    projector_stale = True
    if args.no_projector:
        print(
            "[projector] SKIPPED (--no-projector). The existing projector.bin regresses the OLD AE "
            "latent and is now STALE w.r.t. the sonic corpus: in-corpus journeys use sonic correctly, "
            "but off-corpus cold-start snapping would need a projector retrain."
        )
    else:
        cf = bool(os.environ.get("CF_ACCOUNT_ID", "").strip() and os.environ.get("CF_AI_TOKEN", "").strip())
        st_ok = False
        try:
            import importlib.util

            st_ok = importlib.util.find_spec("sentence_transformers") is not None
        except Exception:  # noqa: BLE001
            st_ok = False
        if not (cf or st_ok):
            print(
                "[projector] SKIPPED — no bge backend (set CF_ACCOUNT_ID+CF_AI_TOKEN or pip install "
                "sentence-transformers). The existing projector.bin regresses the OLD AE latent and is "
                "now STALE w.r.t. the sonic corpus: in-corpus journeys use sonic correctly, but "
                "off-corpus cold-start snapping would need a projector retrain later."
            )
        else:
            backend = "Workers AI" if cf else "local sentence-transformers"
            print(f"[projector] retraining to regress the sonic latent via {backend} ...")
            try:
                text_model = B.train_projector(g, OUT / "projector.bin", SimpleNamespace(epochs=args.epochs))
                if text_model is None:
                    print("[projector] backend unavailable at runtime — projector left STALE (AE-regressing).")
                else:
                    projector_stale = False
                    print("[projector] retrained OK — now regresses the sonic latent.")
            except Exception as e:  # noqa: BLE001
                text_model = None
                print(f"[projector] retrain FAILED (non-fatal): {e!r}. projector.bin left STALE (AE-regressing).")

    manifest = write_manifest(g, scorer, text_model=text_model, projector_stale=projector_stale)
    print(f"manifest: {json.dumps(manifest, indent=2)}")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
