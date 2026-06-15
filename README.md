# Pathfinder — the space between two songs

A fullscreen 3D visualizer for the [spotify-predict-engagement](../spotify-predict-engagement)
pathfinder. Pick two Spotify tracks and watch the A* "musical journey" between
them thread through a 3D map of listening taste.

**It runs entirely on Cloudflare** — a single Worker, no separate Python origin,
no Qdrant at request time. The heavy compute (snap, A*, densify, transition
scoring, 3D projection, the cold-start projector) is a Rust crate compiled to
WebAssembly; the Worker adds Spotify/ReccoBeats fetches and a Workers AI
embedding for cold-start. Python + Qdrant are used **only** by the offline build
that bakes the corpus into static files.

## Architecture

```
Browser ── static UI (React + Three.js) ─────────────┐  served by ASSETS binding
   /api/search  ─┐                                    │
   /api/route   ─┤── Worker (worker/index.ts) ────────┘
   /api/config  ─┤     ├─ AI binding   → bge-m3 embedding (cold-start only)
   /api/preview ─┘     ├─ fetch        → Spotify search/meta + ReccoBeats + iTunes
                │     ├─ ASSETS       → loads public/data/* once per isolate
                │     └─ WASM core    → snap · A* · densify · transitions · PCA
                └─ (no Qdrant, no Python, no second origin)
```

## Make the journey playable

The journey strip has two actions:

- **Play journey** — plays a 30-second sample of each stop back-to-back, the 3D
  tracer following along (a "musical procession"). Samples come from the
  **iTunes Search API** (`/api/preview`), matched by artist+track — Spotify's own
  `preview_url` was deprecated in Nov 2024. No login, works for everyone.
- **Save as playlist** — creates the route as a real Spotify playlist on the
  listener's account. Uses browser-side **OAuth (Authorization Code + PKCE)** with
  the public `SPOTIFY_CLIENT_ID` (no secret, no server session). **Register the
  app origin as a Redirect URI** in the Spotify dashboard — `http://127.0.0.1:8787/`
  for dev and `https://<your-worker>.workers.dev/` for deploy — or sign-in fails.

The pathfinding (`worker-core/`) is a faithful Rust port of
`../spotify-predict-engagement/pathfinder/search.py` (same cost weights,
constraints, densify, transition back-off). A parity test
(`worker-core/tests/parity.rs`) checks it against the original Python on corpus
routes.

## Build & run locally

Prerequisites: Node, Rust + `wasm-pack`, and the snapshot venv
(`server/.venv` with numpy/onnxruntime, plus torch for `--projector`).

```sh
npm install
cp .env.example .env          # fill SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET (+ CF_* for cold-start)

# 1. Bake the corpus snapshot from Qdrant -> public/data/*
npm run build:snapshot
#   add cold-start (needs CF_ACCOUNT_ID + CF_AI_TOKEN):
#   server/.venv/bin/python tools/build_snapshot.py --projector

# 2. Build wasm + UI, then run the Worker locally (serves UI + /api)
npm run cf:dev                # http://127.0.0.1:8787
```

For HMR-driven UI work, run `npm run dev` (Vite, :5173) alongside `wrangler dev`;
Vite proxies `/api` to `:8787`. Local Spotify creds go in `.dev.vars`.

## Deploy to Cloudflare

```sh
npm run build                 # build:wasm -> tsc -> vite (copies public/data into dist)
wrangler secret put SPOTIFY_CLIENT_ID
wrangler secret put SPOTIFY_CLIENT_SECRET
npm run cf:deploy
```

The `[ai]` binding (`wrangler.toml`) is required for cold-start of arbitrary
(non-corpus) tracks. Without the projector snapshot, the app still works fully
for the ~23.5k corpus tracks; off-map picks return a clear "not in corpus"
message.

## Refreshing the corpus

The snapshot is static. When the Qdrant corpus changes, re-run
`npm run build:snapshot` (with `--projector` if cold-start is enabled) and
redeploy. See `tools/SNAPSHOT_FORMAT.md` for the byte layouts.

## Files

- `worker-core/` — Rust→WASM compute core (`snapshot`, `pathfind`, `transit`,
  `pca`, `project`); `tests/` hold the parity + projector checks.
- `worker/index.ts` — the Worker: routing, Spotify/ReccoBeats, AI binding, WASM glue.
- `src/main.tsx`, `src/styles.css` — the React/Three.js visualizer.
- `tools/build_snapshot.py` — offline snapshot builder (reuses `server/app.py`).
- `public/data/` — generated artifacts (`corpus.bin`, `transitions.json`,
  `projector.bin`, `manifest.json`).
- `server/app.py` — the original Python reference, now used only by the builder.
