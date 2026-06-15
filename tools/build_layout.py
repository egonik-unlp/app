"""Bake a single GLOBAL 3D layout of the corpus so the splash cloud and any
traced route live in the SAME coordinate space.

Reads public/data/corpus.bin (no Qdrant), runs 3D t-SNE over the 64-d latent
(neighborhood-preserving — a route then reads as a coherent thread through the
galaxy), normalizes to a fixed world scale, and writes public/data/layout.bin:

    off  type          field
    0    u8[4]         magic = "PFL1"
    4    u32           version = 1
    8    u32           n
    12   f32[n*3]      xyz, row-major, SAME row order as corpus.bin

UMAP/PaCMAP would be nicer but need numba, which has no Python 3.14 wheels;
sklearn t-SNE is pure Cython and gives ~69% top-10 neighbor preservation here.
"""
import struct, time
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

CORPUS = "public/data/corpus.bin"
OUT = "public/data/layout.bin"
WORLD_RADIUS = 60.0  # 99th-pct radius maps here, matching the camera/fog scale


def load_vecs(path):
    buf = open(path, "rb").read()
    assert buf[:4] == b"PFC1"
    ver, n, dim, k = struct.unpack_from("<IIII", buf, 4)
    off = 20 + 8 * n
    V = np.frombuffer(buf, "<f4", n * dim, off).reshape(n, dim).astype(np.float32)
    off += 4 * n * dim + 4 * n
    nbr = np.frombuffer(buf, "<u4", n * k, off).reshape(n, k)
    return n, dim, k, V, nbr


def main():
    n, dim, k, V, nbr = load_vecs(CORPUS)
    print(f"corpus n={n} dim={dim}", flush=True)

    t = time.time()
    emb = TSNE(
        n_components=3, init="pca", perplexity=30, metric="cosine",
        random_state=0, n_jobs=-1,
    ).fit_transform(V)
    print(f"t-SNE 3D done in {time.time()-t:.1f}s", flush=True)

    # center + isotropic scale so the 99th-pct radius == WORLD_RADIUS
    emb = emb - emb.mean(0)
    r = np.linalg.norm(emb, axis=1)
    scale = WORLD_RADIUS / np.percentile(r, 99)
    emb = (emb * scale).astype(np.float32)
    print(f"scaled: bbox=[{emb.min(0).round(1).tolist()} .. {emb.max(0).round(1).tolist()}]", flush=True)

    # quality: do corpus neighbors stay near in the layout
    rng = np.random.default_rng(0)
    samp = rng.choice(n, 800, replace=False)
    keep = tot = 0
    for i in samp:
        nb = nbr[i]; nb = nb[nb != 0xFFFFFFFF][:10]
        if len(nb) == 0:
            continue
        d = np.linalg.norm(emb - emb[i], axis=1)
        near = set(np.argpartition(d, 31)[:31].tolist())
        keep += sum(int(x) in near for x in nb); tot += len(nb)
    print(f"knn preservation (top10 in nearest30): {100*keep/tot:.1f}%", flush=True)

    with open(OUT, "wb") as f:
        f.write(b"PFL1")
        f.write(struct.pack("<II", 1, n))
        f.write(emb.tobytes())
    print(f"wrote {OUT} ({4 + 8 + emb.nbytes} bytes)", flush=True)


if __name__ == "__main__":
    main()
