#!/usr/bin/env python3
"""Trace ONE real route end-to-end for the dev guide's worked example.

Runs the faithful A* port (same as run_astar_sweep.py) on a recognizable
in-corpus pair, and records every stage with real numbers: resolved rows, the
A* expansion count, the found path, the per-hop cost decomposition into the five
weighted terms (reproducing the expansion-time min-max normalization for the
chosen edge), the per-hop interpretable "why" the app surfaces, and 3-D PCA
positions for the path + neighborhood cloud.

  /home/gonik/anaconda3/bin/python docs/sweeps/run_example.py
"""
import heapq
import json
import struct
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent.parent.parent
CORPUS = APP / "public/data/corpus.bin"
TRANS = APP / "public/data/transitions.json"
OUT = Path(__file__).resolve().parent / "example.json"

W = dict(W_DIST=1.0, W_FIT=0.5, W_DIV=0.5, W_TRANS=0.6, W_CTX=0.4)
GJP = 0.15
MAX_CONSEC, SHARE_DIV, BACKOFF_K = 2, 4.0, 8.0
T_ART, T_GEN = 0.6, 0.4
LEN_HINT = 12
MAX_EXP = 20000


def load():
    b = CORPUS.read_bytes()
    assert b[:4] == b"PFC1"
    _, n, dim, k = struct.unpack_from("<4I", b, 4)
    o = 20 + 8 * n
    vecs = np.frombuffer(b, "<f4", n * dim, o).reshape(n, dim).astype(np.float64); o += 4 * n * dim
    fit = np.frombuffer(b, "<f4", n, o).astype(np.float64); o += 4 * n
    nbr_idx = np.frombuffer(b, "<u4", n * k, o).reshape(n, k); o += 4 * n * k
    nbr_dist = np.frombuffer(b, "<f4", n * k, o).reshape(n, k).astype(np.float64); o += 4 * n * k
    (ml,) = struct.unpack_from("<I", b, o); o += 4
    meta = json.loads(b[o:o + ml])
    return dict(n=n, dim=dim, k=k, vecs=vecs, fit=fit, nbr_idx=nbr_idx, nbr_dist=nbr_dist,
                name=[m["name"] for m in meta], artist=[m["artist"] for m in meta],
                genre=[m["genre"] for m in meta], uri=[m["uri"] for m in meta])


def nbrs(C, r):
    return [(int(i), float(d)) for i, d in zip(C["nbr_idx"][r], C["nbr_dist"][r]) if i != 0xFFFFFFFF]


def cdist(C, a, b):
    return 1.0 - float(C["vecs"][a] @ C["vecs"][b])


def minmax(d):
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    return {k: 0.5 for k in d} if hi - lo <= 0 else {k: (v - lo) / (hi - lo) for k, v in d.items()}


class Trans:
    def __init__(s, j):
        s.tb, s.ac, s.gc = j["track_bigram"], j["artist_cond"], j["genre_cond"]
        s.cgl, s.ctl = j["ctx_genre_lift"], j["ctx_track_lift"]

    def aff(s, C, a, b):
        ua, ub = C["uri"][a], C["uri"][b]
        tp, nu = 0.0, 0.0
        tb = s.tb.get(ua)
        if tb:
            nu = float(sum(tb.values()))
            tp = tb.get(ub, 0.0) / nu if nu else 0.0
        ap = s.ac.get(C["artist"][a], {}).get(C["artist"][b], 0.0)
        gp = s.gc.get(C["genre"][a], {}).get(C["genre"][b], 0.0)
        trust = nu / (nu + BACKOFF_K)
        return trust * tp + (1 - trust) * (T_ART * ap + T_GEN * gp)

    def ctx(s, tod, shuf):
        return (s.cgl.get(f"tod:{tod}", {}) if tod else {},
                s.cgl.get(f"shuf:{shuf}", {}) if shuf in ("shuffle", "linear") else {},
                s.ctl.get(tod, {}) if tod else {})

    def cfit(s, C, v, ctx):
        gt, gs, tt = ctx
        lift = gt.get(C["genre"][v], 1.0) * gs.get(C["genre"][v], 1.0)
        t = tt.get(C["uri"][v])
        return lift * t if t is not None else lift


def violates(C, path, c):
    if c in path:
        return True
    a = C["artist"][c]
    run = 1
    for p in reversed(path):
        if C["artist"][p] == a:
            run += 1
        else:
            break
    if run > MAX_CONSEC:
        return True
    cap = -(-max(LEN_HINT, len(path) + 1) // int(SHARE_DIV))
    return sum(1 for p in path if C["artist"][p] == a) + 1 > cap


def astar(C, T, start, end, ctx):
    cnt = exp = 0
    heap = [(cdist(C, start, end), cnt, 0.0, start, (start,))]
    best = {start: 0.0}
    while heap and exp < MAX_EXP:
        f, _, g, node, path = heapq.heappop(heap)
        exp += 1
        if node == end:
            return list(path), exp
        if g > best.get(node, 1e9):
            continue
        pg = C["genre"][node]
        cands = [(nb, d) for nb, d in nbrs(C, node) if nb == end or not violates(C, path, nb)]
        tn = minmax({nb: T.aff(C, node, nb) for nb, _ in cands}) if cands else {}
        cn = minmax({nb: T.cfit(C, nb, ctx) for nb, _ in cands}) if (cands and ctx) else {}
        for nb, d in cands:
            step = W["W_DIST"] * d + W["W_FIT"] * (1 - C["fit"][nb])
            if C["genre"][nb] != pg:
                step += W["W_DIV"] * GJP
            if tn:
                step += W["W_TRANS"] * (1 - tn.get(nb, 0.5))
            if cn:
                step += W["W_CTX"] * (1 - cn.get(nb, 0.5))
            ng = g + step
            if ng >= best.get(nb, 1e9):
                continue
            best[nb] = ng
            cnt += 1
            heapq.heappush(heap, (ng + cdist(C, nb, end), cnt, ng, nb, path + (nb,)))
    return None, exp


def hop_terms(C, T, path, i, ctx):
    """Reproduce the expansion-time weighted terms for the chosen edge path[i-1]->path[i]."""
    a, b = path[i - 1], path[i]
    prefix = list(path[:i])
    cands = [(nb, d) for nb, d in nbrs(C, a) if nb == path[-1] or not violates(C, prefix, nb)]
    if b not in [c for c, _ in cands]:
        cands.append((b, cdist(C, a, b)))
    tn = minmax({nb: T.aff(C, a, nb) for nb, _ in cands})
    cn = minmax({nb: T.cfit(C, nb, ctx) for nb, _ in cands}) if ctx else {}
    d = cdist(C, a, b)
    terms = {
        "dist": W["W_DIST"] * d,
        "fit": W["W_FIT"] * (1 - C["fit"][b]),
        "div": W["W_DIV"] * GJP if C["genre"][b] != C["genre"][a] else 0.0,
        "trans": W["W_TRANS"] * (1 - tn.get(b, 0.5)),
        "ctx": (W["W_CTX"] * (1 - cn.get(b, 0.5))) if cn else 0.0,
    }
    why = {"dist": d, "fit": C["fit"][b], "trans": T.aff(C, a, b),
           "ctx": T.cfit(C, b, ctx) if ctx else None,
           "genre_jump": C["genre"][a] != C["genre"][b]}
    terms["total"] = sum(v for k, v in terms.items())
    return terms, why


def pca3(C, rows, extra=()):
    V = np.array([C["vecs"][r] for r in rows] + list(extra), dtype=np.float64)
    M = V - V.mean(0)
    cov = M.T @ M / len(M)
    w, vec = np.linalg.eigh(cov)
    ax = vec[:, ::-1][:, :3]
    coords = M @ ax
    norms = np.linalg.norm(coords, axis=1)
    scale = np.percentile(norms, 90) or 1.0
    return coords * (8.0 / scale)


def find_row(C, artist_sub, name_sub=""):
    a, nm = artist_sub.lower(), name_sub.lower()
    for r in range(C["n"]):
        if a in C["artist"][r].lower() and nm in C["name"][r].lower():
            return r
    return -1


def main():
    C = load(); T = Trans(json.loads(TRANS.read_text()))
    ctx = T.ctx("evening", "shuffle"); ctx_label = "evening · shuffle"
    pairs = [("kraftwerk", "arctic monkeys"), ("kraftwerk", "muse"),
             ("the prodigy", "arctic monkeys"), ("daft punk", "radiohead"),
             ("kraftwerk", "radiohead"), ("aphex twin", "arctic monkeys"),
             ("massive attack", "muse"), ("boards of canada", "radiohead")]
    chosen = None
    for sa, sb in pairs:
        ra, rb = find_row(C, sa), find_row(C, sb)
        if ra < 0 or rb < 0:
            continue
        path, exp = astar(C, T, ra, rb, ctx)
        if path and 6 <= len(path) <= 12:
            chosen = (sa, sb, ra, rb, path, exp); break
    if not chosen:
        # fallback: deterministic moderate-distance pair
        rng = np.random.default_rng(3)
        for _ in range(500):
            ra, rb = int(rng.integers(C["n"])), int(rng.integers(C["n"]))
            if ra == rb or not (0.5 <= cdist(C, ra, rb) <= 0.8):
                continue
            path, exp = astar(C, T, ra, rb, ctx)
            if path and 6 <= len(path) <= 12:
                chosen = (C["artist"][ra], C["artist"][rb], ra, rb, path, exp); break
    sa, sb, ra, rb, path, exp = chosen

    # neighborhood cloud (<=8 nbrs/node, cap 60)
    seen = set(path); cloud = []
    for p in path:
        for nb, _ in nbrs(C, p)[:8]:
            if nb not in seen:
                seen.add(nb); cloud.append(nb)
            if len(cloud) >= 60:
                break
    pos = pca3(C, list(path) + cloud)
    ppos, cpos = pos[:len(path)], pos[len(path):]

    steps = []
    for i, r in enumerate(path):
        rec = {"i": i, "name": C["name"][r], "artist": C["artist"][r], "genre": C["genre"][r],
               "fit": round(C["fit"][r], 3), "h": round(cdist(C, r, rb), 3),
               "pos": [round(float(x), 2) for x in ppos[i]]}
        if i > 0:
            terms, why = hop_terms(C, T, path, i, ctx)
            rec["terms"] = {k: round(v, 3) for k, v in terms.items()}
            rec["why"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in why.items()}
        steps.append(rec)
    total_cost = round(sum(s["terms"]["total"] for s in steps if "terms" in s), 3)

    # per-hop decision: the neighbors A* weighed at each node, ranked by local
    # attractiveness (step cost + heuristic-to-goal); flag the one it committed to.
    decisions = []
    for i in range(len(path) - 1):
        node, chosen = path[i], path[i + 1]
        prefix = list(path[:i + 1])
        cands = [(nb, d) for nb, d in nbrs(C, node) if nb == rb or not violates(C, prefix, nb)]
        if chosen not in [c for c, _ in cands]:
            cands.append((chosen, cdist(C, node, chosen)))
        tn = minmax({nb: T.aff(C, node, nb) for nb, _ in cands})
        cn = minmax({nb: T.cfit(C, nb, ctx) for nb, _ in cands}) if ctx else {}
        pg = C["genre"][node]
        rows = []
        for nb, d in cands:
            step = W["W_DIST"] * d + W["W_FIT"] * (1 - C["fit"][nb])
            if C["genre"][nb] != pg:
                step += W["W_DIV"] * GJP
            step += W["W_TRANS"] * (1 - tn.get(nb, 0.5))
            if cn:
                step += W["W_CTX"] * (1 - cn.get(nb, 0.5))
            h = cdist(C, nb, rb)
            rows.append({"artist": C["artist"][nb], "genre": C["genre"][nb],
                         "step": round(step, 3), "h": round(h, 3), "prio": round(step + h, 3),
                         "chosen": nb == chosen})
        rows.sort(key=lambda r: r["prio"])
        crank = next(j for j, r in enumerate(rows) if r["chosen"])
        top = rows[:6]
        if not any(r["chosen"] for r in top):  # always keep the chosen visible
            top = top + [rows[crank]]
        decisions.append({"i": i, "from": C["artist"][node], "n_cands": len(rows),
                          "chosen": C["artist"][chosen], "chosen_rank": crank,
                          "cands": top})

    out = {
        "context": ctx_label, "start_row": ra, "end_row": rb, "expansions": exp,
        "n_path": len(path), "n_cloud": len(cloud), "total_cost": total_cost,
        "start": {"name": C["name"][ra], "artist": C["artist"][ra], "genre": C["genre"][ra],
                  "uri": C["uri"][ra], "fit": round(C["fit"][ra], 3)},
        "end": {"name": C["name"][rb], "artist": C["artist"][rb], "genre": C["genre"][rb],
                "uri": C["uri"][rb], "fit": round(C["fit"][rb], 3)},
        "steps": steps,
        "decisions": decisions,
        "cloud_pos": [[round(float(x), 2) for x in p] for p in cpos],
        "cloud_genre": [C["genre"][r] for r in cloud],
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"route: {sa} -> {sb}  | {exp} expansions, {len(path)} hops, cost {total_cost}, "
          f"cloud {len(cloud)}")
    for s in steps:
        t = s.get("terms", {})
        print(f"  {s['i']:2d} {s['artist'][:18]:18s} | {s['genre'][:16]:16s} fit={s['fit']:.2f}"
              + (f"  step={t.get('total',0):.2f}" if t else "  (start)"))


if __name__ == "__main__":
    main()
