from __future__ import annotations

import base64
import datetime
import heapq
import itertools
import json
import math
import os
import re
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT.parent / "spotify-predict-engagement"
MODEL_DIR = ROOT / "best-xgboost-classifier-20260609-172920-627d6-export" / "best-xgboost-classifier-20260609-172920-627d6"
ARTIFACTS = SIBLING / "pipeline" / "artifacts"
TRANSITIONS_JSON = SIBLING / "pathfinder" / "transitions.json"

QDRANT_URL = os.environ.get("QDRANT_URL", os.environ.get("PATHFINDER_QDRANT_URL", "http://localhost:6335"))
COLLECTION = os.environ.get("PATHFINDER_COLLECTION", "spotify_tracks_song_ae")
PORT = int(os.environ.get("PATHFINDER_PORT", "8098"))

SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")

KNN_K = int(os.environ.get("PATHFINDER_KNN_K", "20"))
PATH_LENGTH = int(os.environ.get("PATHFINDER_DEFAULT_LENGTH", "14"))
MAX_EXPANSIONS = int(os.environ.get("PATHFINDER_MAX_EXPANSIONS", "200000"))
W_DIST = 1.0
W_FIT = 0.5
W_DIV = 0.5
W_TRANS = 0.6
W_CTX = 0.4
TRANSITION_BACKOFF_K = 8.0
TRANSITION_ARTIST_WEIGHT = 0.6
TRANSITION_GENRE_WEIGHT = 0.4
MAX_CONSECUTIVE_ARTIST = 2
ARTIST_SHARE_DIVISOR = 4
GENRE_JUMP_PENALTY = 0.15

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API = "https://api.spotify.com/v1"
RECCOBEATS_API = "https://api.reccobeats.com/v1"
UA = {"User-Agent": "Mozilla/5.0 spotify-pathfinder-visualizer/0.1", "Accept": "application/json"}

_lock = threading.Lock()
_graph: TrackGraph | None = None
_encoder: ColdStartEncoder | None = None
_scorer: OnnxScorer | None = None
_scores: dict[int, float] | None = None
_transition_model: TransitionModel | None | bool = False
_spotify_token: tuple[str, float] | None = None


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value)


load_env(ROOT / ".env")
load_env(SIBLING / ".env")
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", SPOTIFY_CLIENT_ID)
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", SPOTIFY_CLIENT_SECRET)


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class TrackGraph:
    ids: list[int]
    vectors: np.ndarray
    meta: dict[int, dict[str, Any]]
    learned_ids: set[int]
    neighbors: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    _row: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._row = {pid: i for i, pid in enumerate(self.ids)}

    def vec(self, pid: int) -> np.ndarray:
        return self.vectors[self._row[pid]]

    def cos_dist(self, a: int, b: int) -> float:
        return float(1.0 - self.vec(a) @ self.vec(b))


@dataclass
class Context:
    genre_tod: dict[str, float]
    genre_shuf: dict[str, float]
    track_tod: dict[str, float]
    label: str


@dataclass
class TransitionModel:
    track_bigram: dict[str, dict[str, float]]
    artist_cond: dict[str, dict[str, float]]
    genre_cond: dict[str, dict[str, float]]
    ctx_genre_lift: dict[str, dict[str, float]]
    ctx_track_lift: dict[str, dict[str, float]]
    tod_buckets: list[str]

    def affinity(self, mu: dict[str, Any], mv: dict[str, Any]) -> float:
        u = mu.get("track_uri", "")
        v = mv.get("track_uri", "")
        tb = self.track_bigram.get(u)
        track_p = 0.0
        n_u = 0.0
        if tb:
            n_u = float(sum(tb.values()))
            if n_u > 0:
                track_p = float(tb.get(v, 0.0)) / n_u
        a_p = self.artist_cond.get(mu.get("artist", "?"), {}).get(mv.get("artist", "?"), 0.0)
        g_p = self.genre_cond.get(mu.get("genre_primary", "unknown"), {}).get(
            mv.get("genre_primary", "unknown"), 0.0
        )
        backoff = TRANSITION_ARTIST_WEIGHT * a_p + TRANSITION_GENRE_WEIGHT * g_p
        trust = n_u / (n_u + TRANSITION_BACKOFF_K)
        return trust * track_p + (1.0 - trust) * backoff

    def context_fit(self, mv: dict[str, Any], ctx: Context) -> float:
        genre = mv.get("genre_primary", "unknown")
        lift = ctx.genre_tod.get(genre, 1.0) * ctx.genre_shuf.get(genre, 1.0)
        track_lift = ctx.track_tod.get(mv.get("track_uri", ""))
        if track_lift is not None:
            lift *= track_lift
        return lift

    def resolve_context(self, tod: str | None = "now", shuffle: bool | None = None) -> Context:
        labels: list[str] = []
        if tod == "now":
            tod = tod_bucket(datetime.datetime.now().hour)
        genre_tod: dict[str, float] = {}
        track_tod: dict[str, float] = {}
        if tod:
            genre_tod = self.ctx_genre_lift.get(f"tod:{tod}", {})
            track_tod = self.ctx_track_lift.get(tod, {})
            labels.append(tod)
        genre_shuf: dict[str, float] = {}
        if shuffle is not None:
            state = "shuffle" if shuffle else "linear"
            genre_shuf = self.ctx_genre_lift.get(f"shuf:{state}", {})
            labels.append(state)
        return Context(genre_tod, genre_shuf, track_tod, " · ".join(labels) if labels else "any")


def tod_bucket(hour: int) -> str:
    if 0 <= hour < 6:
        return "night"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening"


def minmax(raw: dict[int, float]) -> dict[int, float]:
    if not raw:
        return {}
    lo = min(raw.values())
    hi = max(raw.values())
    span = hi - lo
    if span <= 0:
        return {k: 0.5 for k in raw}
    return {k: (v - lo) / span for k, v in raw.items()}


def violates(g: TrackGraph, path: list[int], candidate: int, length_hint: int) -> bool:
    if candidate in path:
        return True
    artist = g.meta[candidate].get("artist")
    run = 1
    for pid in reversed(path):
        if g.meta[pid].get("artist") == artist:
            run += 1
        else:
            break
    if run > MAX_CONSECUTIVE_ARTIST:
        return True
    cap = math.ceil(max(length_hint, len(path) + 1) / ARTIST_SHARE_DIVISOR)
    total = sum(1 for pid in path if g.meta[pid].get("artist") == artist) + 1
    return total > cap


def find_path(
    g: TrackGraph,
    start: int,
    end: int,
    scores: dict[int, float],
    length_hint: int = 12,
    max_expansions: int = 200_000,
    model: TransitionModel | None = None,
    ctx: Context | None = None,
) -> list[int] | None:
    counter = itertools.count()
    open_heap = [(g.cos_dist(start, end), next(counter), 0.0, start, [start])]
    best_g: dict[int, float] = {start: 0.0}
    for _ in range(max_expansions):
        if not open_heap:
            return None
        _, _, g_cost, node, path = heapq.heappop(open_heap)
        if node == end:
            return path
        if g_cost > best_g.get(node, float("inf")):
            continue
        mu = g.meta[node]
        prev_genre = mu.get("genre_primary")
        cands = [
            (nbr, dist)
            for nbr, dist in g.neighbors.get(node, [])
            if nbr == end or not violates(g, path, nbr, length_hint)
        ]
        trans_n: dict[int, float] = {}
        ctx_n: dict[int, float] = {}
        if model is not None and cands:
            trans_n = minmax({nbr: model.affinity(mu, g.meta[nbr]) for nbr, _ in cands})
            if ctx is not None:
                ctx_n = minmax({nbr: model.context_fit(g.meta[nbr], ctx) for nbr, _ in cands})
        for nbr, dist in cands:
            step = W_DIST * dist
            step += W_FIT * (1.0 - scores.get(nbr, 0.5))
            if g.meta[nbr].get("genre_primary") != prev_genre:
                step += W_DIV * GENRE_JUMP_PENALTY
            if trans_n:
                step += W_TRANS * (1.0 - trans_n.get(nbr, 0.5))
            if ctx_n:
                step += W_CTX * (1.0 - ctx_n.get(nbr, 0.5))
            new_g = g_cost + step
            if new_g >= best_g.get(nbr, float("inf")):
                continue
            best_g[nbr] = new_g
            heapq.heappush(open_heap, (new_g + g.cos_dist(nbr, end), next(counter), new_g, nbr, path + [nbr]))
    return None


def violates_insertion(g: TrackGraph, path: list[int], i: int, candidate: int, length_hint: int) -> bool:
    if candidate in path:
        return True
    artist = g.meta[candidate].get("artist")
    run = 1
    for pid in reversed(path[: i + 1]):
        if g.meta[pid].get("artist") == artist:
            run += 1
        else:
            break
    for pid in path[i + 1 :]:
        if g.meta[pid].get("artist") == artist:
            run += 1
        else:
            break
    if run > MAX_CONSECUTIVE_ARTIST:
        return True
    cap = math.ceil(max(length_hint, len(path) + 1) / ARTIST_SHARE_DIVISOR)
    total = sum(1 for pid in path if g.meta[pid].get("artist") == artist) + 1
    return total > cap


def densify(
    g: TrackGraph,
    path: list[int],
    scores: dict[int, float],
    target_length: int,
    model: TransitionModel | None = None,
    ctx: Context | None = None,
) -> list[int]:
    path = list(path)
    while len(path) < target_length:
        gaps = [(g.cos_dist(a, b), i) for i, (a, b) in enumerate(zip(path, path[1:]))]
        gaps.sort(reverse=True)
        inserted = False
        for _, i in gaps:
            a, b = path[i], path[i + 1]
            ma, mb = g.meta[a], g.meta[b]
            candidates = {pid for pid, _ in g.neighbors.get(a, [])} | {pid for pid, _ in g.neighbors.get(b, [])}
            gap = g.cos_dist(a, b)
            valid: list[tuple[int, float]] = []
            for c in candidates:
                if violates_insertion(g, path, i, c, target_length):
                    continue
                if g.cos_dist(a, c) >= gap or g.cos_dist(c, b) >= gap:
                    continue
                valid.append((c, g.cos_dist(a, c) + g.cos_dist(c, b) - gap))
            if not valid:
                continue
            trans_n: dict[int, float] = {}
            ctx_n: dict[int, float] = {}
            if model is not None:
                trans_n = minmax({c: model.affinity(ma, g.meta[c]) + model.affinity(g.meta[c], mb) for c, _ in valid})
                if ctx is not None:
                    ctx_n = minmax({c: model.context_fit(g.meta[c], ctx) for c, _ in valid})
            best = None
            best_cost = float("inf")
            for c, detour in valid:
                cost = W_DIST * detour + W_FIT * (1.0 - scores.get(c, 0.5))
                if trans_n:
                    cost += W_TRANS * (1.0 - trans_n.get(c, 0.5))
                if ctx_n:
                    cost += W_CTX * (1.0 - ctx_n.get(c, 0.5))
                if cost < best_cost:
                    best, best_cost = c, cost
            if best is not None:
                path.insert(i + 1, best)
                inserted = True
                break
        if not inserted:
            break
    return path


def graph() -> TrackGraph:
    global _graph
    with _lock:
        if _graph is None:
            _graph = load_graph()
        return _graph


def load_graph() -> TrackGraph:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=QDRANT_URL, timeout=120)
    ids: list[int] = []
    vectors: list[list[float]] = []
    meta: dict[int, dict[str, Any]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION,
            limit=2048,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in points:
            ids.append(int(p.id))
            vectors.append(p.vector)
            meta[int(p.id)] = (p.payload or {}).get("metadata", {}) or {}
        if offset is None:
            break
    if not ids:
        raise ApiError(503, f"Qdrant collection {COLLECTION!r} has no points")

    vecs = np.asarray(vectors, dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
    g = TrackGraph(ids=ids, vectors=vecs, meta=meta, learned_ids=set(ids))
    build_knn(g)
    return g


def build_knn(g: TrackGraph, block: int = 2048) -> None:
    sub = g.vectors
    m = sub.shape[0]
    k = min(KNN_K, max(1, m - 1))
    edges: dict[int, dict[int, float]] = {pid: {} for pid in g.ids}
    for start in range(0, m, block):
        end = min(start + block, m)
        sims = sub[start:end] @ sub.T
        for r in range(end - start):
            sims[r, start + r] = -np.inf
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        for r in range(end - start):
            pid = g.ids[start + r]
            for j in top[r]:
                other = g.ids[int(j)]
                dist = float(1.0 - sims[r, int(j)])
                edges[pid][other] = dist
                edges[other][pid] = dist
    g.neighbors = {pid: sorted(nbrs.items(), key=lambda item: item[1]) for pid, nbrs in edges.items()}


def transition_model() -> TransitionModel | None:
    global _transition_model
    with _lock:
        if _transition_model is not False:
            return _transition_model
        if not TRANSITIONS_JSON.exists():
            _transition_model = None
            return None
        data = json.loads(TRANSITIONS_JSON.read_text())
        _transition_model = TransitionModel(
            track_bigram=data["track_bigram"],
            artist_cond=data["artist_cond"],
            genre_cond=data["genre_cond"],
            ctx_genre_lift=data["ctx_genre_lift"],
            ctx_track_lift=data["ctx_track_lift"],
            tod_buckets=data.get("meta", {}).get("tod_buckets", ["night", "morning", "afternoon", "evening"]),
        )
        return _transition_model


def fit_scores() -> dict[int, float]:
    global _scores
    if _scores is not None:
        return _scores
    s = scorer().score_graph(graph())
    with _lock:
        if _scores is None:
            _scores = s
        return _scores


def scorer() -> OnnxScorer:
    global _scorer
    with _lock:
        if _scorer is None:
            _scorer = OnnxScorer(MODEL_DIR)
        return _scorer


def encoder() -> ColdStartEncoder:
    global _encoder
    with _lock:
        if _encoder is None:
            _encoder = ColdStartEncoder()
        return _encoder


def spotify_token() -> str:
    global _spotify_token
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        raise ApiError(500, "Spotify credentials are missing. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.")
    if _spotify_token and _spotify_token[1] > time.time() + 30:
        return _spotify_token[0]
    auth = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    resp = requests.post(
        SPOTIFY_TOKEN_URL,
        headers={"Authorization": f"Basic {auth}"},
        data={"grant_type": "client_credentials"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise ApiError(502, f"Spotify token failed ({resp.status_code}): {resp.text[:200]}")
    data = resp.json()
    _spotify_token = (data["access_token"], time.time() + float(data.get("expires_in", 3600)))
    return _spotify_token[0]


def spotify_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    resp = requests.get(
        f"{SPOTIFY_API}/{path}",
        headers={"Authorization": f"Bearer {spotify_token()}"},
        params=params,
        timeout=30,
    )
    if resp.status_code != 200:
        raise ApiError(502, f"Spotify request failed ({resp.status_code}): {resp.text[:200]}")
    return resp.json()


def search_spotify(query: str, limit: int = 10) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    data = spotify_get("search", {"q": query, "type": "track", "limit": limit})
    out = []
    for t in data.get("tracks", {}).get("items", []):
        imgs = ((t.get("album") or {}).get("images") or [])
        out.append(
            {
                "spotify_id": t["id"],
                "uri": t.get("uri") or f"spotify:track:{t['id']}",
                "name": t.get("name") or "Unknown track",
                "artist": ", ".join(a.get("name", "") for a in t.get("artists", []) if a.get("name")),
                "album": (t.get("album") or {}).get("name"),
                "art": imgs[1]["url"] if len(imgs) > 1 else (imgs[0]["url"] if imgs else None),
            }
        )
    return out


def spotify_id(value: str) -> str:
    value = value.strip()
    if value.startswith("spotify:track:"):
        return value.rsplit(":", 1)[-1]
    return re.sub(r".*[/:]", "", value.split("?", 1)[0])


def reccobeats_features(sid: str) -> dict[str, Any]:
    try:
        r = requests.get(f"{RECCOBEATS_API}/track", headers=UA, params={"ids": sid}, timeout=30)
        r.raise_for_status()
        content = r.json().get("content", [])
        if not content:
            return {}
        uuid = content[0]["id"]
        r = requests.get(f"{RECCOBEATS_API}/audio-features", headers=UA, params={"ids": uuid}, timeout=30)
        r.raise_for_status()
        feats = r.json().get("content", [])
        return feats[0] if feats else {}
    except requests.RequestException:
        return {}


def content_doc(meta: dict[str, Any]) -> str:
    parts: list[str] = []
    if meta.get("track_name"):
        parts.append(str(meta["track_name"]) + ".")
    if meta.get("artist"):
        artist = str(meta["artist"])
        if (meta.get("artist_count") or 1) > 1:
            artist += f" (with {int(meta['artist_count']) - 1} other artist(s))"
        parts.append(f"Artist: {artist}.")
    if meta.get("album"):
        bits = [b for b in (meta.get("album_type", ""), str(meta.get("release_year") or "")) if b]
        parts.append(f"Album: {meta['album']}" + (f" ({', '.join(bits)})." if bits else "."))
    if meta.get("genre_primary"):
        parts.append(f"Genre: {meta['genre_primary']}.")
    return " ".join(parts) or "Unknown track."


class ColdStartEncoder:
    def __init__(self) -> None:
        import torch
        import torch.nn as nn
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        self.pre = json.loads((ARTIFACTS / "song_ae_preprocess.json").read_text())
        self.text_model = SentenceTransformer(self.pre["text_model"])

        class EncoderNet(nn.Module):
            def __init__(self, din: int, hidden: int, latent: int):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, latent))

            def forward(self, x):
                return self.enc(x)

        self.net = EncoderNet(self.pre["din"], self.pre["hidden"], self.pre["latent"])
        state = torch.load(ARTIFACTS / "song_ae.pt", map_location="cpu")
        self.net.load_state_dict({k: v for k, v in state.items() if k.startswith("enc.")})
        self.net.eval()

    def encode_spotify(self, uri: str) -> tuple[dict[str, Any], np.ndarray]:
        sid = spotify_id(uri)
        track = spotify_get(f"tracks/{sid}")
        artist_ids = [a["id"] for a in track.get("artists", []) if a.get("id")]
        artists = (
            spotify_get("artists", {"ids": ",".join(artist_ids[:50])}).get("artists", [])
            if artist_ids
            else []
        )
        genres = sorted({g for a in artists for g in (a.get("genres") or [])})
        af = reccobeats_features(sid)
        album = track.get("album") or {}
        release_date = album.get("release_date") or ""
        meta = {
            "track_uri": f"spotify:track:{sid}",
            "track_name": track.get("name"),
            "album": album.get("name"),
            "artist": (track.get("artists") or [{}])[0].get("name"),
            "artist_count": len(track.get("artists", []) or []),
            "track_popularity": track.get("popularity"),
            "artist_popularity": artists[0].get("popularity") if artists else None,
            "artist_followers": (artists[0].get("followers") or {}).get("total") if artists else None,
            "album_type": album.get("album_type"),
            "release_year": int(release_date[:4]) if release_date[:4].isdigit() else None,
            "sp_duration_ms": track.get("duration_ms"),
            "sp_explicit": 1 if track.get("explicit") else 0,
            "sp_track_number": track.get("track_number"),
            "sp_n_markets": len(track.get("available_markets", []) or []),
            "sp_genres": genres,
            "sp_genre_count": len(genres),
            "genre_primary": genres[0] if genres else "unknown",
            **{f"af_{k}": af.get(k) for k in [
                "danceability",
                "energy",
                "valence",
                "tempo",
                "acousticness",
                "instrumentalness",
                "loudness",
                "speechiness",
                "liveness",
                "key",
                "mode",
            ]},
        }
        text_emb = self.text_model.encode([content_doc(meta)], normalize_embeddings=True)[0]
        row = self.build_input_row(meta, text_emb)
        with self.torch.no_grad():
            latent = self.net(self.torch.from_numpy(row).unsqueeze(0)).numpy()[0]
        latent = latent.astype(np.float32)
        latent /= np.linalg.norm(latent).clip(min=1e-9)
        return meta, latent

    def build_input_row(self, meta: dict[str, Any], text_emb: np.ndarray) -> np.ndarray:
        pre = self.pre

        def z(value: Any, stats: dict[str, float]) -> float:
            if value is None:
                v = stats["mean"]
            else:
                v = float(value)
                if stats.get("log") and v >= 0:
                    v = math.log1p(v)
            return (v - stats["mean"]) / stats["std"]

        num = [z(meta.get(k), pre["num_stats"][k]) for k in pre["numerics"]]
        ac: list[float] = []
        flags: list[float] = []
        for k in pre["acoustics"]:
            v = meta.get(k)
            st = pre["ac_stats"][k]
            present = isinstance(v, (int, float))
            ac.append(((float(v) if present else st["mean"]) - st["mean"]) / st["std"])
            flags.append(0.0 if present else 1.0)
        genres = set(meta.get("sp_genres") or [])
        if meta.get("genre_primary"):
            genres.add(meta["genre_primary"])
        gmat = [1.0 if g in genres else 0.0 for g in pre["genre_vocab"]]
        amat = [1.0 if meta.get("album_type") == a else 0.0 for a in pre["album_types"]]
        row = np.concatenate([text_emb, np.asarray(num + ac + flags + gmat + amat, dtype=np.float32)])
        if row.shape[0] != pre["din"]:
            raise ApiError(500, f"AE input row has {row.shape[0]} columns, expected {pre['din']}")
        return row.astype(np.float32)


class OnnxScorer:
    def __init__(self, model_dir: Path) -> None:
        import onnxruntime as ort

        self.feat = json.loads((model_dir / "featurize.json").read_text())
        self.components = np.fromfile(model_dir / self.feat["pca"]["components_file"], dtype="<f4").reshape(
            tuple(self.feat["pca"]["components_shape"])
        )
        self.mean = np.asarray(self.feat["pca"]["mean"], dtype=np.float32)
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def score_graph(self, g: TrackGraph) -> dict[int, float]:
        rows = np.vstack([self.features(g.meta[pid], g.vec(pid)) for pid in g.ids]).astype(np.float32)
        out = self.session.run([self.output_name], {self.input_name: rows})[0].reshape(-1)
        out = np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
        out = np.clip(out, 0.0, 1.0)
        lo = float(out.min())
        hi = float(out.max())
        span = hi - lo
        norm = np.full_like(out, 0.5) if span <= 1e-12 else (out - lo) / span
        return {pid: float(norm[i]) for i, pid in enumerate(g.ids)}

    def features(self, meta: dict[str, Any], embedding: np.ndarray) -> np.ndarray:
        pca = (embedding.astype(np.float32) - self.mean) @ self.components.T
        values: list[float] = []
        for col in self.feat["columns"]:
            op = col["op"]
            if op == "pca":
                values.append(float(pca[int(col["component"])]))
            elif op == "numeric_present_raw":
                values.append(float(meta.get(col["field"]) or 0.0))
            elif op == "numeric_missing_flag":
                values.append(0.0 if meta.get(col["field"]) not in (None, "") else 1.0)
            elif op == "numeric_log1p":
                raw = meta.get(col["field"])
                values.append(math.log1p(float(raw)) if raw not in (None, "") and float(raw) >= 0 else 0.0)
            elif op == "numeric_verbatim":
                values.append(float(meta.get(col["field"]) or 0.0))
            elif op == "onehot":
                value = meta.get(col["group"])
                if col["value"] == "__other__":
                    group_values = {c["value"] for c in self.feat["columns"] if c.get("op") == "onehot" and c.get("group") == col["group"] and c.get("value") != "__other__"}
                    values.append(1.0 if value not in group_values else 0.0)
                else:
                    values.append(1.0 if value == col["value"] else 0.0)
            else:
                values.append(0.0)
        return np.asarray(values, dtype=np.float32)


def track_json(g: TrackGraph, pid: int, fit: float | None = None, pos: list[float] | None = None) -> dict[str, Any]:
    m = g.meta[pid]
    uri = m.get("track_uri", "")
    return {
        "id": str(pid),
        "uri": uri,
        "name": m.get("track_name") or "Unknown track",
        "artist": m.get("artist") or "Unknown artist",
        "album": m.get("album"),
        "genre": m.get("genre_primary") or "unknown",
        "spotify_url": "https://open.spotify.com/track/" + uri.rsplit(":", 1)[-1] if uri else None,
        "fit": fit,
        "position": pos,
        "kind": "path",
    }


def requested_json(meta: dict[str, Any], vector: np.ndarray, pos: list[float], snapped: int | None, g: TrackGraph) -> dict[str, Any]:
    uri = meta.get("track_uri", "")
    return {
        "id": uri,
        "uri": uri,
        "name": meta.get("track_name") or "Unknown track",
        "artist": meta.get("artist") or "Unknown artist",
        "album": meta.get("album"),
        "genre": meta.get("genre_primary") or "unknown",
        "spotify_url": "https://open.spotify.com/track/" + uri.rsplit(":", 1)[-1] if uri else None,
        "fit": None,
        "position": pos,
        "kind": "requested",
        "snapped_to": str(snapped) if snapped is not None else None,
        "snapped_label": g.meta[snapped].get("track_name") if snapped is not None else None,
    }


def exact_or_nearest(g: TrackGraph, meta: dict[str, Any], vector: np.ndarray) -> tuple[int, bool]:
    uri = meta.get("track_uri")
    for pid, m in g.meta.items():
        if m.get("track_uri") == uri:
            return pid, True
    sims = g.vectors @ vector.astype(np.float32)
    return g.ids[int(np.argmax(sims))], False


def local_positions(g: TrackGraph, vectors: dict[str, np.ndarray]) -> dict[str, list[float]]:
    keys = list(vectors.keys())
    mat = np.vstack([vectors[k] for k in keys]).astype(np.float32)
    centered = mat - mat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[: min(3, vt.shape[0])]
    coords = centered @ axes.T
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    scale = np.percentile(np.linalg.norm(coords, axis=1), 90)
    scale = float(scale if scale > 1e-6 else 1.0)
    coords = coords / scale * 8.0
    return {k: [float(x) for x in coords[i, :3]] for i, k in enumerate(keys)}


def build_route(payload: dict[str, Any]) -> dict[str, Any]:
    start_uri = payload.get("start_uri")
    end_uri = payload.get("end_uri")
    if not start_uri or not end_uri:
        raise ApiError(400, "start_uri and end_uri are required")

    g = graph()
    enc = encoder()
    start_meta, start_vec = enc.encode_spotify(start_uri)
    end_meta, end_vec = enc.encode_spotify(end_uri)
    start_anchor, start_exact = exact_or_nearest(g, start_meta, start_vec)
    end_anchor, end_exact = exact_or_nearest(g, end_meta, end_vec)
    length = max(4, min(50, int(payload.get("length") or PATH_LENGTH)))
    model = transition_model()
    ctx = None
    if model is not None:
        context = payload.get("context", "now")
        tod = None if context == "any" else context
        shuffle = {"any": None, "shuffle": True, "linear": False}.get(payload.get("shuffle", "any"))
        ctx = model.resolve_context(tod=tod, shuffle=shuffle)

    scores = fit_scores()
    path = find_path(g, start_anchor, end_anchor, scores, length_hint=length, max_expansions=MAX_EXPANSIONS, model=model, ctx=ctx)
    if path is None:
        raise ApiError(404, "No path found between these tracks")
    if len(path) < length:
        path = densify(g, path, scores, length, model=model, ctx=ctx)

    cloud_ids: list[int] = []
    seen = set(path)
    for pid in path:
        for nbr, _ in g.neighbors.get(pid, [])[:8]:
            if nbr not in seen:
                seen.add(nbr)
                cloud_ids.append(nbr)
            if len(cloud_ids) >= 90:
                break
        if len(cloud_ids) >= 90:
            break

    vectors: dict[str, np.ndarray] = {f"path:{pid}": g.vec(pid) for pid in path}
    vectors.update({f"cloud:{pid}": g.vec(pid) for pid in cloud_ids})
    vectors["requested:start"] = start_vec
    vectors["requested:end"] = end_vec
    positions = local_positions(g, vectors)

    return {
        "context": ctx.label if ctx else None,
        "requested_start": requested_json(
            start_meta,
            start_vec,
            positions["requested:start"],
            None if start_exact else start_anchor,
            g,
        ),
        "requested_end": requested_json(
            end_meta,
            end_vec,
            positions["requested:end"],
            None if end_exact else end_anchor,
            g,
        ),
        "path": [track_json(g, pid, scores.get(pid), positions[f"path:{pid}"]) for pid in path],
        "cloud": [track_json(g, pid, scores.get(pid), positions[f"cloud:{pid}"]) for pid in cloud_ids],
        "edges": [
            {"from": str(a), "to": str(b), "kind": "path"}
            for a, b in zip(path, path[1:])
        ]
        + ([] if start_exact else [{"from": start_meta["track_uri"], "to": str(start_anchor), "kind": "snap"}])
        + ([] if end_exact else [{"from": str(end_anchor), "to": end_meta["track_uri"], "kind": "snap"}]),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._headers("application/json", 0)

    def do_GET(self) -> None:
        try:
            url = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(url.query)
            if url.path == "/api/health":
                g = _graph
                self.json({"ready": g is not None, "tracks": len(g.ids) if g else 0})
            elif url.path == "/api/search":
                self.json(search_spotify(q.get("q", [""])[0]))
            else:
                raise ApiError(404, "not found")
        except ApiError as e:
            self.json({"error": str(e)}, e.status)
        except Exception as e:
            self.json({"error": str(e)}, 500)

    def do_POST(self) -> None:
        try:
            url = urllib.parse.urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if url.path == "/api/route":
                self.json(build_route(payload))
            else:
                raise ApiError(404, "not found")
        except ApiError as e:
            self.json({"error": str(e)}, e.status)
        except Exception as e:
            self.json({"error": str(e)}, 500)

    def json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._headers("application/json", len(body))
        self.wfile.write(body)

    def _headers(self, ctype: str, length: int) -> None:
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"pathfinder visual API: http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
