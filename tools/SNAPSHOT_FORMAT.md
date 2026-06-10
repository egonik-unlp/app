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

## `manifest.json`

```
{ "version", "n_tracks", "dim", "knn_k", "has_projector": bool,
  "text_model": "@cf/baai/bge-m3" | null, "built_at": "<iso>" }
```
