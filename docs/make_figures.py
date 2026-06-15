#!/usr/bin/env python3
"""Render every figure in models.pdf from real artifacts + the sweep outputs.

  /home/gonik/anaconda3/bin/python docs/make_figures.py

Reads: docs/sweeps/{xgb_sweep,astar_sweep,runs}.json, app/public/data/{corpus,projector}.bin,
the champion dataset manifest. Writes docs/figures/*.pdf. Brand-themed (lensing
DESIGN.md): orange #f59e0b, slate #3a3a42, blue #4f7bf6.
"""
import json
import struct
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

DOCS = Path(__file__).resolve().parent
APP = DOCS.parent
SPE = APP.parent / "spotify-predict-engagement"
FIG = DOCS / "figures"
SW = DOCS / "sweeps"

ORANGE, SLATE, BLUE = "#f59e0b", "#3a3a42", "#4f7bf6"
GREY = "#9a98a4"
LENS = LinearSegmentedColormap.from_list("lens", ["#2a2a33", "#4f7bf6", "#f4f4f5", "#f59e0b"])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": SLATE, "axes.labelcolor": "#26262e", "text.color": "#26262e",
    "xtick.color": SLATE, "ytick.color": SLATE, "axes.titlesize": 10,
    "axes.titleweight": "bold", "figure.dpi": 140, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#e6e6ea", "grid.linewidth": 0.6,
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("  wrote", name)


# ---------------- corpus.bin ----------------
def load_corpus():
    b = (APP / "public/data/corpus.bin").read_bytes()
    assert b[:4] == b"PFC1"
    _, n, dim, k = struct.unpack_from("<4I", b, 4)
    o = 20
    o += 8 * n
    vecs = np.frombuffer(b, "<f4", n * dim, o).reshape(n, dim); o += 4 * n * dim
    fit = np.frombuffer(b, "<f4", n, o).copy(); o += 4 * n
    o += 4 * n * k  # nbr_idx
    nbr_dist = np.frombuffer(b, "<f4", n * k, o).reshape(n, k).copy()
    return dict(n=n, dim=dim, k=k, vecs=vecs, fit=fit, nbr_dist=nbr_dist)


def fig_fit_hist(C):
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.hist(C["fit"], bins=60, color=ORANGE, edgecolor=SLATE, linewidth=0.3)
    ax.axvline(np.median(C["fit"]), color=BLUE, ls="--", lw=1.4,
               label=f"median {np.median(C['fit']):.3f}")
    ax.set_xlabel(r"$\mathrm{fit}$  =  min-max-normalized  $P(\mathrm{rotation})$")
    ax.set_ylabel("tracks")
    ax.set_title(f"Engagement / fit score over {C['n']:,} corpus tracks")
    ax.legend(frameon=False)
    save(fig, "fig_fit_hist.pdf")


def fig_knn(C):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.8))
    valid = C["nbr_dist"][C["nbr_dist"] >= 0]
    nn1 = C["nbr_dist"][:, 0]
    a1.hist(nn1, bins=60, color=BLUE, edgecolor=SLATE, linewidth=0.3)
    a1.set_xlabel("cosine distance to nearest neighbor")
    a1.set_ylabel("tracks"); a1.set_title("Nearest-neighbor distance")
    mean_by_rank = C["nbr_dist"].mean(axis=0)
    p10 = np.percentile(C["nbr_dist"], 10, axis=0)
    p90 = np.percentile(C["nbr_dist"], 90, axis=0)
    r = np.arange(1, C["k"] + 1)
    a2.fill_between(r, p10, p90, color=ORANGE, alpha=0.25, label="10–90 pct")
    a2.plot(r, mean_by_rank, color=ORANGE, marker="o", ms=3, label="mean")
    a2.set_xlabel("neighbor rank (1..k)"); a2.set_ylabel("cosine distance")
    a2.set_title(f"Distance vs rank (k={C['k']})"); a2.legend(frameon=False)
    save(fig, "fig_knn.pdf")


def fig_pca_var(C):
    rng = np.random.default_rng(0)
    idx = rng.choice(C["n"], 1500, replace=False)
    X = C["vecs"][idx].astype(np.float64)
    X = X - X.mean(0)
    cov = X.T @ X / len(X)
    ev = np.linalg.eigvalsh(cov)[::-1]
    share = ev / ev.sum()
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    nshow = 12
    ax.bar(np.arange(1, nshow + 1), share[:nshow] * 100, color=SLATE)
    ax.bar([1, 2, 3], share[:3] * 100, color=ORANGE,
           label=f"top-3 = {share[:3].sum()*100:.1f}% (the 3D view)")
    ax.set_xlabel("principal component"); ax.set_ylabel("variance explained (%)")
    ax.set_title("PCA spectrum of the 64-d taste latent (1.5k sample)")
    ax.legend(frameon=False)
    save(fig, "fig_pca_var.pdf")


# ---------------- projector.bin ----------------
def fig_projector_blocks():
    b = (APP / "public/data/projector.bin").read_bytes()
    assert b[:4] == b"PFP1"
    _, in_dim, hidden, out_dim, text_dim = struct.unpack_from("<5I", b, 4)
    o = 24
    w1 = np.frombuffer(b, "<f4", hidden * in_dim, o).reshape(hidden, in_dim)
    o += 4 * hidden * in_dim + 4 * hidden
    o += 4 * out_dim * hidden + 4 * out_dim
    (cfg_len,) = struct.unpack_from("<I", b, o); o += 4
    cfg = json.loads(b[o:o + cfg_len])
    ng, na = len(cfg["genre_vocab"]), len(cfg["album_types"])
    nnum, nac = len(cfg["numerics"]), len(cfg["acoustics"])
    # block layout matches project.rs: text, numerics, acoustic values, acoustic flags, genre, album
    spans = [("text\n(bge-m3)", 0, text_dim),
             ("numeric", text_dim, text_dim + nnum),
             ("acoustic\nvalues", text_dim + nnum, text_dim + nnum + nac),
             ("acoustic\nflags", text_dim + nnum + nac, text_dim + nnum + 2 * nac),
             ("genre\nmulti-hot", text_dim + nnum + 2 * nac, text_dim + nnum + 2 * nac + ng),
             ("album\none-hot", in_dim - na, in_dim)]
    col_absmean = np.abs(w1).mean(axis=0)  # mean over hidden units, per input feature
    labels, vals = [], []
    for lab, a, c in spans:
        labels.append(lab); vals.append(float(col_absmean[a:c].mean()))
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    cols = [BLUE, SLATE, ORANGE, "#d98a00", "#6b6878", GREY]
    ax.bar(labels, vals, color=cols, edgecolor=SLATE, linewidth=0.4)
    ax.set_ylabel(r"mean $|W_1|$ per input feature")
    ax.set_title(f"What the cold-start projector leans on  (in_dim={in_dim}$\\to${hidden}$\\to${out_dim})")
    save(fig, "fig_projector_blocks.pdf")


# ---------------- XGBoost sweep ----------------
def _heat(ax, M, xs, ys, xlabel, ylabel, title, star=None, fmt="{:.3f}"):
    M = np.array(M, dtype=float)
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=LENS)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=0)
    ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(False)
    vmin, vmax = np.nanmin(M), np.nanmax(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                    color="white" if v < (vmin + vmax) / 2 else "#26262e")
    if star is not None:
        ax.add_patch(plt.Rectangle((star[1] - .5, star[0] - .5), 1, 1, fill=False,
                                   edgecolor=ORANGE, lw=2.4))
    return im


def fig_xgb():
    d = json.loads((SW / "xgb_sweep.json").read_text())
    dl, dn = d["depth_lr"], d["depth_nest"]
    champ_i = dl["depths"].index(d["champion"]["max_depth"])
    champ_j = dl["lrs"].index(d["champion"]["learning_rate"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    im1 = _heat(a1, dl["auc"], dl["lrs"], dl["depths"], "learning rate", "max depth",
                "Test AUC vs depth × lr", star=(champ_i, champ_j))
    cn_i = dn["depths"].index(d["champion"]["max_depth"])
    cn_j = dn["nests"].index(d["champion"]["n_estimators"])
    im2 = _heat(a2, dn["auc"], dn["nests"], dn["depths"], "n_estimators", "max depth",
                "Test AUC vs depth × n_estimators", star=(cn_i, cn_j))
    fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.04, label="ROC AUC")
    fig.suptitle("XGBoost rotation classifier — deferred HP scan on the real champion dataset "
                 "(orange = shipped config)", fontsize=8.5, y=1.02)
    save(fig, "fig_xgb_sweep.pdf")


# ---------------- A* response surfaces ----------------
def fig_astar():
    d = json.loads((SW / "astar_sweep.json").read_text())
    tc, fd = d["trans_ctx"], d["fit_div"]
    W = d["defaults"]
    g = tc["grid"]
    sidx = lambda v: g.index(v) if v in g else None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.3), constrained_layout=True)
    im1 = _heat(a1, tc["mean_edge"], g, g, r"$W_{\mathrm{CTX}}$", r"$W_{\mathrm{TRANS}}$",
                "Mean edge cos-distance\n(path roughness)",
                star=(sidx(W["W_TRANS"]), sidx(W["W_CTX"])))
    fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.02)
    im2 = _heat(a2, fd["genre_jump_rate"], g, g, r"$W_{\mathrm{DIV}}$", r"$W_{\mathrm{FIT}}$",
                "Genre-jump rate", star=(sidx(W["W_FIT"]), sidx(W["W_DIV"])))
    fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.02)
    fig.suptitle(f"A* response surfaces over {len(d['routes'])} fixed routes "
                 f"(orange = shipped weights); context: {d['context']}", fontsize=8.5)
    save(fig, "fig_astar_sweep.pdf")


# ---------------- runs: bakeoff / noise band / variants ----------------
def fig_bakeoff():
    d = json.loads((SW / "runs.json").read_text())
    bo = d["bakeoff"]; nb = d["noise_band"]
    labels = [e["label"] for e in bo][::-1]
    vals = [e["auc"] for e in bo][::-1]
    colors = [ORANGE if "xgboost" in l else SLATE for l in labels]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.barh(labels, vals, color=colors, edgecolor=SLATE, linewidth=0.4)
    champ = bo[0]["auc"]
    ax.axvspan(champ - nb["two_sigma"], champ, color=BLUE, alpha=0.12)
    ax.axvline(0.5, color=GREY, ls=":", lw=1, label="chance")
    for y, v in enumerate(vals):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_xlim(0.5, 0.78); ax.set_xlabel("ROC AUC (rotation)")
    ax.set_title("Model-family bake-off — identity one-hots + trees win")
    ax.grid(axis="y"); ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_bakeoff.pdf")


def fig_noiseband():
    d = json.loads((SW / "runs.json").read_text())["noise_band"]
    seeds = {int(k): v for k, v in d["seeds"].items()}
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    xs = list(range(len(seeds)))
    ax.axhspan(d["mean"] - d["two_sigma"], d["mean"] + d["two_sigma"], color=BLUE, alpha=0.15,
               label=f"±2σ = ±{d['two_sigma']:.4f}")
    ax.axhline(d["mean"], color=SLATE, lw=1, ls="--", label=f"mean {d['mean']:.4f}")
    ax.scatter(xs, list(seeds.values()), color=ORANGE, zorder=5, s=55, edgecolor=SLATE)
    ax.set_xticks(xs); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("ROC AUC"); ax.set_title("3-seed noise band (champion config)")
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "fig_noiseband.pdf")


def fig_variants():
    d = json.loads((SW / "runs.json").read_text())
    v = d["feature_variants"]; ts = d["noise_band"]["two_sigma"]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    labels = [e["label"] for e in v]; vals = [e["auc"] for e in v]
    ax.bar(labels, vals, color=[ORANGE, GREY], edgecolor=SLATE, linewidth=0.4, width=0.6)
    for x, val in enumerate(vals):
        ax.text(x, val + 0.002, f"{val:.4f}", ha="center", fontsize=8)
    ax.set_ylim(0.70, 0.75); ax.set_ylabel("ROC AUC")
    ax.set_title(f"Adding the 64-d AE latent DILUTES\n(Δ={vals[0]-vals[1]:+.4f}, within 2σ={ts:.4f})")
    save(fig, "fig_variants.pdf")


# ---------------- analytic / constants ----------------
SES = SPE / "pipeline/corpus/out/sessions.parquet"
FEATS = SPE / "pipeline/corpus/out/track_features.parquet"
LIFT_CMAP = LinearSegmentedColormap.from_list("lift", [BLUE, "#dfe6fb", "#f6f6f7", "#ffe2a8", ORANGE])
COLLAB = ["morning", "afternoon", "evening", "night", "shuffle", "linear"]


def _era_lift_table():
    """Same lift definition as the genre table (pipeline/corpus/transitions.py),
    applied to release era. lift = shrink(P(era|ctx)/P(era), support); clip[0.2,5]."""
    import pandas as pd
    SKIP, MINSUP = 30000, 20
    tod = lambda h: "night" if h < 6 else "morning" if h < 12 else "afternoon" if h < 18 else "evening"

    def shrink(obs, base, sup):
        if base <= 0:
            return 1.0
        lift = 1.0 + (obs / base - 1.0) * min(1.0, sup / MINSUP)
        return float(min(5.0, max(0.2, lift)))

    def era(y):
        if pd.isna(y):
            return None
        y = int(y)
        return "$\\leq$1979" if y < 1980 else ("2020s" if y >= 2020 else f"{(y // 10) * 10}s")

    p = pd.read_parquet(SES); f = pd.read_parquet(FEATS)
    ry = dict(zip(f.track_uri, f.release_year))
    nz = p[p.ms_played >= SKIP].copy()
    nz["ts"] = pd.to_datetime(nz.ts, utc=True)
    nz["tod"] = nz.ts.dt.hour.map(tod)
    nz["shuf"] = nz.shuffle.map(lambda s: "shuffle" if bool(s) else "linear")
    nz["era"] = nz.spotify_track_uri.map(ry).map(era)
    nz = nz[nz.era.notna()]
    total = len(nz); base = (nz.era.value_counts() / total).to_dict()
    order = ["$\\leq$1979", "1980s", "1990s", "2000s", "2010s", "2020s"]
    ctxcols = [("tod", c) for c in ["morning", "afternoon", "evening", "night"]] + \
              [("shuf", c) for c in ["shuffle", "linear"]]
    M = []
    for val in order:
        row = []
        for c, bk in ctxcols:
            g = nz[nz[c] == bk]; cnt = int((g.era == val).sum())
            row.append(shrink(cnt / len(g), base.get(val, 0.0), cnt))
        M.append(row)
    return np.array(M), order


def fig_lifts():
    b = (APP / "public/data/corpus.bin").read_bytes()
    _, n, dim, k = struct.unpack_from("<4I", b, 4)
    o = 20 + 8 * n + 4 * n * dim + 4 * n + 4 * n * k + 4 * n * k
    (mlen,) = struct.unpack_from("<I", b, o); o += 4
    meta = json.loads(b[o:o + mlen])
    from collections import Counter
    gc = Counter(m["genre"] for m in meta)
    cgl = json.loads((APP / "public/data/transitions.json").read_text())["ctx_genre_lift"]
    gcols = ["tod:morning", "tod:afternoon", "tod:evening", "tod:night", "shuf:shuffle", "shuf:linear"]
    # a short, fresh, recognizable genre set with sharp, varied patterns
    genres = ["alternative metal", "dance pop", "indie rock", "art pop", "big beat", "deep house"]
    G = np.array([[cgl[c][g] for c in gcols] for g in genres])
    grows = [f"{g}  ({gc[g]})" for g in genres]
    E, erows = _era_lift_table()

    norm = TwoSlopeNorm(vmin=0.3, vcenter=1.0, vmax=3.0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.0),
                                 gridspec_kw={"width_ratios": [1, 1]}, constrained_layout=True)

    def draw(ax, M, rows, title):
        im = ax.imshow(M, cmap=LIFT_CMAP, norm=norm, aspect="auto")
        ax.set_xticks(range(6)); ax.set_xticklabels(COLLAB, rotation=28, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
        ax.axvline(3.5, color=SLATE, lw=1.5)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                        color="#26262e" if 0.65 < v < 1.7 else "white")
        ax.grid(False); ax.set_title(title, fontsize=9.5)
        return im

    draw(a1, G, grows, "Genre  --  a strong daily rhythm")
    im = draw(a2, E, erows, "Release era  --  same study, barely moves")
    cb = fig.colorbar(im, ax=(a1, a2), fraction=0.026, pad=0.02)
    cb.set_label("lift = P(value | context) / P(value)", fontsize=8)
    cb.set_ticks([0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0])
    fig.suptitle("Lifts in action: $>1$ over-represented (boost), $<1$ under-represented (suppress), "
                 "$1$ neutral", fontsize=9)
    save(fig, "fig_lifts.pdf")


def fig_example_cost():
    d = json.loads((SW / "example.json").read_text())
    steps = [s for s in d["steps"] if "terms" in s]
    labels = [f"{i+1}. {d['steps'][i+1]['artist'][:14]}" for i in range(len(steps))]
    keys = ["dist", "fit", "div", "trans", "ctx"]
    kcol = {"dist": BLUE, "fit": ORANGE, "div": "#c9a24a", "trans": SLATE, "ctx": GREY}
    klab = {"dist": "distance", "fit": "fit", "div": "diversity", "trans": "transition", "ctx": "context"}
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(steps)); left = np.zeros(len(steps))
    for k in keys:
        vals = np.array([s["terms"][k] for s in steps])
        ax.barh(y, vals, left=left, color=kcol[k], edgecolor="white", linewidth=0.4, label=klab[k])
        left += vals
    for i, s in enumerate(steps):
        ax.text(s["terms"]["total"] + 0.02, i, f"{s['terms']['total']:.2f}", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("per-hop A* cost (sum of weighted terms)")
    ax.set_title("Cost of each step in the example route")
    ax.legend(frameon=False, ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y")
    save(fig, "fig_example_cost.pdf")


def fig_example_decisions():
    d = json.loads((SW / "example.json").read_text())
    dec = d["decisions"]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for k, hop in enumerate(dec):
        x = k + 1
        steps_all = [c["step"] for c in hop["cands"]]
        ax.scatter([x] * len(steps_all), steps_all, s=22, color=GREY, alpha=0.55, zorder=2)
        ch = next(c for c in hop["cands"] if c["chosen"])
        ax.scatter([x], [ch["step"]], s=80, color=ORANGE, edgecolor=SLATE, zorder=4)
        ax.annotate(ch["artist"][:12], (x, ch["step"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=6.6, color=SLATE, rotation=0)
        # mark the locally-cheapest option if it isn't the chosen one
        cheapest = min(hop["cands"], key=lambda c: c["step"])
        if not cheapest["chosen"]:
            ax.scatter([x], [cheapest["step"]], s=30, facecolor="none",
                       edgecolor=BLUE, linewidth=1.3, zorder=3)
    chosen_line = [next(c for c in h["cands"] if c["chosen"])["step"] for h in dec]
    ax.plot(range(1, len(dec) + 1), chosen_line, color=ORANGE, lw=1.0, alpha=0.5, zorder=1)
    ax.set_xticks(range(1, len(dec) + 1))
    ax.set_xlabel("hop"); ax.set_ylabel("immediate step cost")
    ax.set_title("The decision at each step: options considered vs. chosen")
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markeredgecolor=SLATE,
                  markersize=9, label="chosen step"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=7,
                  label="other candidates"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=BLUE,
                  markersize=8, label="cheapest option (when not chosen)")]
    ax.legend(handles=leg, frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig_example_decisions.pdf")


def fig_example_path():
    d = json.loads((SW / "example.json").read_text())
    pp = np.array([s["pos"][:2] for s in d["steps"]])
    cp = np.array(d["cloud_pos"])[:, :2]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(cp[:, 0], cp[:, 1], s=16, color=GREY, alpha=0.45, label="neighborhood cloud")
    ax.plot(pp[:, 0], pp[:, 1], "-", color=ORANGE, lw=1.6, zorder=3)
    ax.scatter(pp[:, 0], pp[:, 1], s=70, color=ORANGE, edgecolor=SLATE, zorder=4)
    for i, s in enumerate(d["steps"]):
        ax.annotate(str(i), (pp[i, 0], pp[i, 1]), fontsize=7, ha="center", va="center",
                    color="white", zorder=5, fontweight="bold")
    ax.annotate(f"start: {d['start']['artist']}", pp[0], textcoords="offset points",
                xytext=(8, 8), fontsize=8, color=SLATE)
    ax.annotate(f"end: {d['end']['artist']}", pp[-1], textcoords="offset points",
                xytext=(8, -12), fontsize=8, color=SLATE)
    ax.set_xlabel("PCA axis 1"); ax.set_ylabel("PCA axis 2")
    ax.set_title("The route as drawn: path through the taste cloud")
    ax.legend(frameon=False, loc="best", fontsize=8)
    save(fig, "fig_example_path.pdf")


def fig_backoff():
    n = np.linspace(0, 40, 400)
    trust = n / (n + 8.0)
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.plot(n, trust, color=ORANGE, lw=2)
    ax.axvline(8, color=BLUE, ls="--", lw=1.2, label=r"$n_u=K=8$  (trust $=0.5$)")
    ax.fill_between(n, 0, trust, color=ORANGE, alpha=0.10)
    ax.set_xlabel(r"track-bigram support  $n_u$ (observed next-plays of $u$)")
    ax.set_ylabel(r"trust  $=\,n_u/(n_u+K)$")
    ax.set_title("Transition back-off: trust the bigram only when supported")
    ax.set_ylim(0, 1); ax.legend(frameon=False)
    save(fig, "fig_backoff.pdf")


def fig_cost_weights():
    terms = [("dist\n$W_{DIST}$", 1.0, BLUE), ("fit\n$W_{FIT}$", 0.5, ORANGE),
             ("diversity\n$W_{DIV}$", 0.5, SLATE), ("transition\n$W_{TRANS}$", 0.6, "#6b6878"),
             ("context\n$W_{CTX}$", 0.4, GREY)]
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    ax.bar([t[0] for t in terms], [t[1] for t in terms], color=[t[2] for t in terms],
           edgecolor=SLATE, linewidth=0.4)
    for i, t in enumerate(terms):
        ax.text(i, t[1] + 0.02, f"{t[1]}", ha="center", fontsize=8)
    ax.set_ylabel("weight"); ax.set_ylim(0, 1.15)
    ax.set_title("A* edge-cost term weights")
    save(fig, "fig_cost_weights.pdf")


def fig_feature_blocks():
    # Song-AE input (539) and XGBoost featurizer (115, from champion manifest)
    ae = [("text (bge / MiniLM)", 384, BLUE), ("numeric", 10, SLATE),
          ("acoustic + flags", 22, ORANGE), ("genre multi-hot", 120, "#6b6878"),
          ("album one-hot", 3, GREY)]
    man = json.loads((SPE / "data/datasets/ds-20260609-141622-p1-s42/manifest.json").read_text())
    kinds = {}
    for c in man["columns"]:
        k = c["kind"]["type"]
        if k == "onehot":
            grp = c["kind"].get("group", "onehot")
            kinds[f"one-hot: {grp}"] = kinds.get(f"one-hot: {grp}", 0) + 1
        else:
            kinds[k] = kinds.get(k, 0) + 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    left = 0
    for lab, w, c in ae:
        a1.barh(0, w, left=left, color=c, edgecolor="white", label=f"{lab} ({w})")
        left += w
    a1.set_xlim(0, 539); a1.set_yticks([]); a1.set_xlabel("feature index")
    a1.set_title("Song-AE input = 539-d"); a1.grid(False)
    a1.legend(frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    items = sorted(kinds.items(), key=lambda kv: -kv[1])
    palette = [ORANGE, SLATE, BLUE, "#6b6878", GREY, "#c9a24a"]
    a2.bar([k for k, _ in items], [v for _, v in items],
           color=[palette[i % len(palette)] for i in range(len(items))], edgecolor=SLATE, linewidth=0.4)
    a2.set_ylabel("columns"); a2.set_title(f"XGBoost featurizer = {man['n_cols']} cols")
    a2.tick_params(axis="x", rotation=30, labelsize=6.5)
    for lab in a2.get_xticklabels():
        lab.set_ha("right")
    a2.grid(axis="x")
    save(fig, "fig_feature_blocks.pdf")


def main():
    FIG.mkdir(exist_ok=True)
    C = load_corpus()
    print(f"corpus n={C['n']} dim={C['dim']} k={C['k']}")
    fig_fit_hist(C); fig_knn(C); fig_pca_var(C)
    fig_projector_blocks()
    fig_xgb(); fig_astar()
    fig_bakeoff(); fig_noiseband(); fig_variants()
    fig_lifts(); fig_backoff(); fig_cost_weights(); fig_feature_blocks()
    fig_example_cost(); fig_example_path(); fig_example_decisions()
    print("done.")


if __name__ == "__main__":
    main()
