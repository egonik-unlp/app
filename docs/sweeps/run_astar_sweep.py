#!/usr/bin/env python3
"""A* cost-weight RESPONSE SURFACES over the real corpus snapshot.

Faithful numpy port of worker-core/src/pathfind.rs + transit.rs. We run the
actual search on a few fixed routes across 2-D weight grids and measure how the
*found path* changes — mean consecutive-edge cosine distance ("roughness") and
genre-jump rate. This is a parameter response surface for the search, NOT a
trained-model error surface (A* is deterministic search, not learned).

Run with anaconda python (numpy):
  /home/gonik/anaconda3/bin/python docs/sweeps/run_astar_sweep.py
"""
import heapq
import json
import struct
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent.parent.parent
CORPUS = APP / "public/data/corpus.bin"
TRANS = APP / "public/data/transitions.json"
OUT = Path(__file__).resolve().parent / "astar_sweep.json"

# defaults (pathfind.rs / transit.rs)
W = dict(W_DIST=1.0, W_FIT=0.5, W_DIV=0.5, W_TRANS=0.6, W_CTX=0.4)
GENRE_JUMP_PENALTY = 0.15
MAX_CONSEC_ARTIST = 2
ARTIST_SHARE_DIV = 4.0
BACKOFF_K = 8.0
T_ART_W, T_GEN_W = 0.6, 0.4
MAX_EXP = 15000          # capped below the production 45k for python speed
LENGTH_HINT = 14

GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]   # weight axis values (incl. shipped 0.4/0.5/0.6)


def load_corpus():
    b = CORPUS.read_bytes()
    assert b[:4] == b"PFC1"
    _, n, dim, k = struct.unpack_from("<4I", b, 4)
    o = 20
    ids = np.frombuffer(b, "<u8", n, o); o += 8 * n
    vecs = np.frombuffer(b, "<f4", n * dim, o).reshape(n, dim); o += 4 * n * dim
    fit = np.frombuffer(b, "<f4", n, o); o += 4 * n
    nbr_idx = np.frombuffer(b, "<u4", n * k, o).reshape(n, k); o += 4 * n * k
    nbr_dist = np.frombuffer(b, "<f4", n * k, o).reshape(n, k); o += 4 * n * k
    (mlen,) = struct.unpack_from("<I", b, o); o += 4
    meta = json.loads(b[o:o + mlen])
    artist = [m["artist"] for m in meta]
    genre = [m["genre"] for m in meta]
    uri = [m["uri"] for m in meta]
    return dict(n=n, dim=dim, k=k, vecs=np.ascontiguousarray(vecs, np.float64),
                fit=fit.astype(np.float64), nbr_idx=nbr_idx, nbr_dist=nbr_dist.astype(np.float64),
                artist=artist, genre=genre, uri=uri)


def neighbors(C, row):
    out = []
    for j in range(C["k"]):
        idx = C["nbr_idx"][row, j]
        if idx == 0xFFFFFFFF:
            continue
        out.append((int(idx), float(C["nbr_dist"][row, j])))
    return out


def cos_dist(C, a, b):
    return 1.0 - float(C["vecs"][a] @ C["vecs"][b])


def minmax(d):
    if not d:
        return {}
    lo = min(d.values()); hi = max(d.values()); span = hi - lo
    if span <= 0:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / span for k, v in d.items()}


class Trans:
    def __init__(self, j):
        self.tb = j["track_bigram"]; self.ac = j["artist_cond"]; self.gc = j["genre_cond"]
        self.cgl = j["ctx_genre_lift"]; self.ctl = j["ctx_track_lift"]

    def affinity(self, C, a, b):
        ua, ub = C["uri"][a], C["uri"][b]
        track_p, n_u = 0.0, 0.0
        tb = self.tb.get(ua)
        if tb:
            n_u = float(sum(tb.values()))
            if n_u > 0:
                track_p = tb.get(ub, 0.0) / n_u
        a_p = self.ac.get(C["artist"][a], {}).get(C["artist"][b], 0.0)
        g_p = self.gc.get(C["genre"][a], {}).get(C["genre"][b], 0.0)
        backoff = T_ART_W * a_p + T_GEN_W * g_p
        trust = n_u / (n_u + BACKOFF_K)
        return trust * track_p + (1 - trust) * backoff

    def context(self, tod, shuffle):
        gt = self.cgl.get(f"tod:{tod}", {}) if tod else {}
        tt = self.ctl.get(tod, {}) if tod else {}
        gs = self.cgl.get(f"shuf:{shuffle}", {}) if shuffle in ("shuffle", "linear") else {}
        return (gt, gs, tt)

    def context_fit(self, C, v, ctx):
        gt, gs, tt = ctx
        lift = gt.get(C["genre"][v], 1.0) * gs.get(C["genre"][v], 1.0)
        t = tt.get(C["uri"][v])
        if t is not None:
            lift *= t
        return lift


def violates(C, path, cand, length_hint):
    if cand in path:
        return True
    a = C["artist"][cand]
    run = 1
    for pid in reversed(path):
        if C["artist"][pid] == a:
            run += 1
        else:
            break
    if run > MAX_CONSEC_ARTIST:
        return True
    cap = -(-max(length_hint, len(path) + 1) // int(ARTIST_SHARE_DIV))  # ceil
    total = sum(1 for pid in path if C["artist"][pid] == a) + 1
    return total > cap


def find_path(C, T, start, end, w, ctx, max_exp=MAX_EXP):
    counter = 0
    heap = [(cos_dist(C, start, end), counter, 0.0, start, (start,))]
    best_g = {start: 0.0}
    for _ in range(max_exp):
        if not heap:
            return None
        f, _, g, node, path = heapq.heappop(heap)
        if node == end:
            return list(path)
        if g > best_g.get(node, float("inf")):
            continue
        prev_genre = C["genre"][node]
        cands = [(nb, d) for nb, d in neighbors(C, node)
                 if nb == end or not violates(C, path, nb, LENGTH_HINT)]
        trans_n, ctx_n = {}, {}
        if cands:
            trans_n = minmax({nb: T.affinity(C, node, nb) for nb, _ in cands})
            if ctx is not None:
                ctx_n = minmax({nb: T.context_fit(C, nb, ctx) for nb, _ in cands})
        for nb, dist in cands:
            step = w["W_DIST"] * dist
            step += w["W_FIT"] * (1.0 - C["fit"][nb])
            if C["genre"][nb] != prev_genre:
                step += w["W_DIV"] * GENRE_JUMP_PENALTY
            if trans_n:
                step += w["W_TRANS"] * (1.0 - trans_n.get(nb, 0.5))
            if ctx_n:
                step += w["W_CTX"] * (1.0 - ctx_n.get(nb, 0.5))
            ng = g + step
            if ng >= best_g.get(nb, float("inf")):
                continue
            best_g[nb] = ng
            counter += 1
            heapq.heappush(heap, (ng + cos_dist(C, nb, end), counter, ng, nb, path + (nb,)))
    return None


def path_metrics(C, path):
    edges = [cos_dist(C, path[i], path[i + 1]) for i in range(len(path) - 1)]
    jumps = sum(1 for i in range(len(path) - 1) if C["genre"][path[i]] != C["genre"][path[i + 1]])
    return dict(hops=len(path) - 1, mean_edge=float(np.mean(edges)),
                genre_jump_rate=jumps / max(1, len(path) - 1))


def pick_routes(C, T, ctx, n_routes=3, seed=7):
    rng = np.random.default_rng(seed)
    routes = []
    tries = 0
    while len(routes) < n_routes and tries < 400:
        tries += 1
        a, b = int(rng.integers(C["n"])), int(rng.integers(C["n"]))
        if a == b:
            continue
        d = cos_dist(C, a, b)
        if not (0.45 <= d <= 0.85):   # moderate separation -> interesting but tractable
            continue
        p = find_path(C, T, a, b, W, ctx)
        if p is not None and len(p) >= 6:
            routes.append((a, b, round(d, 3), len(p)))
    return routes


def sweep(C, T, ctx, routes, ax, ay, fixed):
    """ax,ay: weight-key names varied over GRID; fixed: other weights from W."""
    mean_edge = [[0.0] * len(GRID) for _ in GRID]
    jump = [[0.0] * len(GRID) for _ in GRID]
    for i, va in enumerate(GRID):
        for j, vb in enumerate(GRID):
            w = {**W, **fixed, ax: va, ay: vb}
            me, jr, ok = [], [], 0
            for (a, b, _, _) in routes:
                p = find_path(C, T, a, b, w, ctx)
                if p is None:
                    continue
                ok += 1
                m = path_metrics(C, p)
                me.append(m["mean_edge"]); jr.append(m["genre_jump_rate"])
            mean_edge[i][j] = float(np.mean(me)) if me else None
            jump[i][j] = float(np.mean(jr)) if jr else None
        print(f"  {ax} row {i+1}/{len(GRID)} done")
    return dict(axis_x=ax, axis_y=ay, grid=GRID, mean_edge=mean_edge, genre_jump_rate=jump,
                n_routes=len(routes))


def main():
    C = load_corpus()
    T = Trans(json.loads(TRANS.read_text()))
    print(f"corpus n={C['n']} dim={C['dim']} k={C['k']}")
    ctx = T.context("evening", "shuffle")
    routes = pick_routes(C, T, ctx)
    print("routes:", routes)
    assert routes, "no tractable routes found"
    g1 = sweep(C, T, ctx, routes, "W_TRANS", "W_CTX", fixed={})
    print("grid 1 (W_TRANS×W_CTX) done")
    g2 = sweep(C, T, ctx, routes, "W_FIT", "W_DIV", fixed={})
    print("grid 2 (W_FIT×W_DIV) done")
    out = dict(corpus_n=C["n"], context="evening · shuffle", max_expansions=MAX_EXP,
               defaults=W, routes=[dict(start=a, end=b, cos_dist=d, default_len=L) for a, b, d, L in routes],
               trans_ctx=g1, fit_div=g2)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
