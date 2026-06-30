# Snapshot artifact format

The offline build (`tools/build_snapshot.py`) snapshots the Qdrant
`spotify_tracks_song_ae` corpus + the precomputed pathfinding inputs into static
files under `public/data/`, which are bundled as Cloudflare Worker assets and
loaded once per isolate by the Rust/WASM core. **No Qdrant / Python at runtime.**

All multi-byte integers and floats are **little-endian**.

## `corpus.bin`  (magic `PFC1`)

```
off  type          field
0    u8[4]         magic = "PFC1"
4    u32           version = 1
8    u32           n              # track count
12   u32           dim            # latent dims = 64
16   u32           k              # KNN fan-out = 20
20   u64[n]        ids            # Qdrant point ids
     f32[n*dim]    vecs           # L2-normalized latent, row-major
     f32[n]        fit            # normalized [0,1] XGBoost rotation fit
     u32[n*k]      nbr_idx        # neighbor ROW indices (not ids); 0xFFFFFFFF = empty pad
     f32[n*k]      nbr_dist       # cosine distance (1 - cos) to each neighbor, ascending
     u32           meta_len
     u8[meta_len]  meta_json      # UTF-8 JSON array of n objects (same order as ids)
```

`meta_json[i]` fields (minimal set the runtime needs):
`{ "uri", "name", "artist", "album", "genre" }`  (`genre` = `genre_primary`).

KNN neighbors are symmetric and sorted by ascending cosine distance, matching
`server/app.py:build_knn`.

## `layout.bin`  (magic `PFL1`) — shared 3D galaxy

A baked global 3D layout (t-SNE over the 64-d latent) giving every corpus track a
fixed home, so the splash sample and any traced route render in the **same**
coordinate space. Built by `tools/build_layout.py` from `corpus.bin` (no Qdrant);
loaded by the Rust core via `set_layout`, and `sample_field` / `route` read these
coords instead of a per-view PCA. Row order matches `corpus.bin`.

```
off  type        field
0    u8[4]       magic = "PFL1"
4    u32         version = 1
8    u32         n
12   f32[n*3]    xyz, row-major (same order as corpus ids)
```

## `transitions.json`

Shipped verbatim from `spotify-predict-engagement/pathfinder/transitions.json`
(parsed by the Rust core with serde). Keys: `track_bigram`, `artist_cond`,
`genre_cond`, `ctx_genre_lift`, `ctx_track_lift`, `meta`.

## `projector.bin`  (magic `PFP1`) — cold-start only

A 2-layer MLP mapping `[bge_text_emb ⊕ numeric ⊕ acoustic ⊕ flags ⊕ genre_multihot ⊕ album_onehot]`
→ the existing 64-dim song-AE latent (then L2-normalized). The corpus latent
space is **unchanged**, so precomputed `fit` scores stay valid; the projector
only places arbitrary tracks near their true neighborhood for snapping.

```
off  type          field
0    u8[4]         magic = "PFP1"
4    u32           version = 1
8    u32           in_dim
12   u32           hidden
16   u32           out_dim = 64
20   u32           text_dim       # bge embedding width (e.g. 1024)
     f32[hidden*in_dim]  W1
     f32[hidden]         b1
     f32[out_dim*hidden] W2
     f32[out_dim]        b2
     u32           cfg_len
     u8[cfg_len]   cfg_json       # feature-assembly config (see below)
```

`cfg_json`:
```
{
  "text_model": "@cf/baai/bge-m3",
  "numerics":  ["artist_popularity", ...],
  "num_stats": { field: {"mean":_, "std":_, "log":bool}, ... },
  "acoustics": ["af_danceability", ...],
  "ac_stats":  { field: {"mean":_, "std":_}, ... },
  "genre_vocab": [ ... 120 genres ... ],
  "album_types": ["album","compilation","single"]
}
```
Feature order (must match training): forward pass input is
`text_emb(text_dim) ++ z(numerics) ++ z(acoustics) ++ missing_flags(acoustics) ++ genre_multihot ++ album_onehot`.

## `scorer.bin`  (magic `PFS1`) — live cold-start fit

A self-contained export of the rotation-fit model
(`burn-deep-embeddings-highvocab`) + its featurize spec, so off-corpus tracks get
a real fit value at request time (the baked `corpus.bin` `fit` only covers
in-corpus tracks). Built by `tools/export_scorer.py` (also invoked from
`build_snapshot.py`); loaded by the Rust core via `set_scorer` and run by
`score_cold`. Independent of the corpus, so it can be regenerated alone.

The network is a fixed graph: continuous block standardized, three categorical
one-hot blocks embedded and concatenated, then a 2-hidden-layer MLP + softmax:

```
input[497] = cont(71) ++ onehot_genre(121) ++ onehot_artist(301) ++ onehot_album(4)
cont       = (input[0:71] - cont_mean) / cont_std
e_*        = onehot_block @ embed_*                 # ONNX MatMul, weights [in,out]
h          = concat(cont, e_genre, e_artist, e_album)              # 107
h          = relu(h @ trunk_w0 + b0); relu(h @ trunk_w1 + b1)
raw        = softmax(h @ head_w + head_b)[class 1]   # then nan_to_num + clip(0,1)
```

`raw` is rescaled to the baked `fit` range in the Worker via `fit_raw_min/max`
(below). The 497-vector is assembled per the `columns` spec, mirroring
`server/app.py` `OnnxScorer.features` (`pca = (latent - pca_mean) @ pca_components.T`,
then numeric/onehot ops; the `__other__` onehot is the unseen-value catch-all).

```
off  type        field
0    u8[4]       magic = "PFS1"
4    u32         version = 1
8    u32         cfg_len
12   u8[cfg_len] cfg_json
     f32 blobs   # concatenated in cfg_json.blobs order
```

`cfg_json`: `{ blocks:{cont,genre,artist,album}, embed:{genre,artist,album},
trunk:[h0,h1], n_class, class_index, pca_dims, columns:[…497 featurize specs…],
blobs:[{name,shape}…] }`. Blob order: `pca_components[64,64]`, `pca_mean[64]`,
`cont_mean[71]`, `cont_std[71]`, `embed_genre[121,16]`, `embed_artist[301,16]`,
`embed_album[4,4]`, `trunk_w0[107,128]`, `trunk_b0[128]`, `trunk_w1[128,64]`,
`trunk_b1[64]`, `head_w[64,2]`, `head_b[2]`. A re-export with a different graph
fails loudly (the exporter asserts the ONNX op sequence + shapes).

Rust↔ONNX parity is gated by `worker-core/tests/scorer.rs` against fixtures from
the exporter (featurize < 1e-5, score < 1e-4).

## `manifest.json`

```
{ "version", "n_tracks", "dim", "knn_k", "has_projector": bool,
  "text_model": "@cf/baai/bge-m3" | null,
  "fit_raw_min": float | null, "fit_raw_max": float | null,  # corpus raw-score bounds
  "built_at": "<iso>" }
```

`fit_raw_min`/`fit_raw_max` are the corpus-wide raw model-output bounds captured
during scoring; the Worker uses them to place a live cold-start `raw` score on the
same `[0,1]` scale as the baked corpus `fit` (the min-max step in
`OnnxScorer.score_graph`). Absent in pre-existing snapshots → the Worker falls
back to the clamped raw score until the next rebuild.
