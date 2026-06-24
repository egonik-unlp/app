# Spotify Pathfinder Visualizer Worklog

## Initial User Prompt

> Create an app (fullscreen animation-like) that shows more visually the pathfinder route betweeen one track and another as created in the project at @../spotify-predict-engagement (pathfinder) using the same algorithm as in there. The prediction algorithm is available at best-xgboost-classifier-20260609-172920-627d6-export. All tracks in spotify should be available, the idea is that the user inputs two tracks, those tracks are looked up in spotify, the api that is called in that project (roccobeats) is called and embeddings are generated.
> The app is fullscreen and it shows an animatiuon of the path, showing representations tracks around the path and how it traverses that space. It should be a 3d animation (so the vectors have to be shown in 3d instead of the full dimensionality) that makes the user sort of understand what is happening, the ui is minimal and really only offeres the input fields for tracks, rest is animation. It should be intuitive

## Planning Decisions

- Build a standalone app in this directory rather than modifying `../spotify-predict-engagement`.
- Keep the pathfinder behavior aligned with the sibling project:
  - Spotify catalog search.
  - ReccoBeats audio features.
  - Cold-start embedding through the sibling song autoencoder artifacts.
  - Snap arbitrary Spotify tracks to nearest in-corpus Qdrant anchors for A* route computation.
  - Render the requested Spotify start/end tracks explicitly, with visual links to their snapped anchors when needed.
- Use the exported `best-xgboost-classifier-20260609-172920-627d6` ONNX bundle locally for fit scoring rather than requiring the sibling Rust server.
- Make Cloudflare compatibility mean: Cloudflare serves the frontend and proxies `/api/*`; the Python/Torch/Qdrant pathfinder compute runs as a separate origin because Workers cannot run that stack directly.

## Implemented Files

- `package.json`, `vite.config.ts`, `tsconfig*.json`, `index.html`
  - Standalone Vite + React + Three.js app scaffold.
- `src/main.tsx`
  - Fullscreen React UI.
  - Minimal two-input Spotify search overlay.
  - Three.js route scene with surrounding cloud nodes, path curve, requested endpoint nodes, snapped-anchor links, and animated tracer.
- `src/styles.css`
  - Fullscreen visual styling, compact search controls, responsive layout, and route labels.
- `server/app.py`
  - Python HTTP API with:
    - `GET /api/health`
    - `GET /api/search?q=...`
    - `POST /api/route`
  - Spotify client-credentials search.
  - ReccoBeats audio-feature retrieval.
  - Cold-start track metadata assembly.
  - Sentence-transformer text embedding and song-AE latent generation using `../spotify-predict-engagement/pipeline/artifacts`.
  - Qdrant graph loading from `spotify_tracks_song_ae`.
  - Local copy of the sibling pathfinder A*/densify logic and transition/context scoring behavior.
  - ONNX model scoring with the exported xgboost classifier bundle.
  - Local 3D projection of route plus nearby tracks using SVD/PCA over returned vectors.
- `server/requirements.txt`
  - Python runtime dependencies.
- `worker/index.ts`, `wrangler.toml`
  - Cloudflare Worker asset/API facade.
  - `/api/*` proxy to `PATHFINDER_API_ORIGIN`.
- `.env.example`
  - Runtime configuration template.
- `.gitignore`
  - Ignores `node_modules`, `dist`, virtualenvs, `.env`, and Worker state.
- `README.md`
  - Local setup, run commands, Cloudflare deployment notes, and file map.

## Verification Completed

- Installed frontend dependencies with `npm install`.
- Built frontend successfully with:

```sh
npm run build
```

- Syntax-checked the Python sidecar:

```sh
python3 -m py_compile server/app.py
```

- Created the backend virtualenv after the missing-path error:

```sh
python3 -m venv server/.venv
```

- Installed backend dependencies:

```sh
server/.venv/bin/pip install -r server/requirements.txt
```

- Verified backend imports:

```sh
server/.venv/bin/python -c "import numpy, requests, torch, sentence_transformers, onnxruntime, qdrant_client; print('server venv ok')"
```

- Started the API with the venv and confirmed:

```sh
curl -s http://127.0.0.1:8098/api/health
```

returned:

```json
{"ready": false, "tracks": 0}
```

- Verified Cloudflare Worker dry-run:

```sh
WRANGLER_LOG_PATH=/tmp/wrangler.log XDG_CONFIG_HOME=/tmp npx wrangler deploy --dry-run
```

## Current Runtime Notes

- Start the backend:

```sh
server/.venv/bin/python -m server.app
```

- Start the frontend:

```sh
npm run dev
```

- The frontend dev server was started during implementation at:

```text
http://127.0.0.1:5173/
```

- Full route generation requires:
  - `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`.
  - Reachable Qdrant at `QDRANT_URL`.
  - A populated `spotify_tracks_song_ae` collection, or `PATHFINDER_COLLECTION` set to the correct collection.
  - Network access to Spotify, ReccoBeats, and Hugging Face model download/cache for `sentence-transformers` if not already cached.

## Caveats

- `server/.venv` is approximately `5.2G` because the default PyTorch wheel for Python 3.14 pulled CUDA packages.
- Cloudflare Workers cannot run the Python ML pathfinder stack directly. Deploy the Python API separately, then set `PATHFINDER_API_ORIGIN` in Cloudflare.
- The first real `/api/route` call will be heavier than `/api/health` because it loads Qdrant graph data, the sentence-transformer, the AE model, transitions, and the ONNX scorer.

---

## 2026-06-10 — Fully Cloudflare-Worker-native rewrite + UI redesign

Reworked the app so the **deployed runtime is 100% Cloudflare** — one Worker, no
external origin, no Python, no Qdrant at request time. Python/Qdrant now run only
in an offline build step that bakes the corpus into static files.

### What changed
- **Rust→WASM compute core (`worker-core/`)**: faithful port of the sibling
  pathfinder (`pathfinder/search.py` + `config.py` weights): snap, A*, densify,
  transition back-off, time-of-day/shuffle context, and the SVD/PCA 3D
  projection. Built with `wasm-pack --target web` (wasm-opt disabled — it capped
  the externref table and workerd failed to grow it). Parity test reproduces the
  Python A* on corpus routes (endpoints/length/constraints exact, cost matched;
  interior diverges only by float-precision in the heuristic, expected for a
  heuristic search). Projector round-trip test guards the binary format.
- **Offline snapshot builder (`tools/build_snapshot.py`)**: reuses
  `server/app.py` (Qdrant scroll, symmetric KNN, ONNX xgboost fit scoring) and
  emits `public/data/{corpus.bin, transitions.json, projector.bin, manifest.json}`.
  Corpus latent space is left untouched, so the precomputed fit scores stay
  valid. Format documented in `tools/SNAPSHOT_FORMAT.md`.
- **Cold-start = Workers AI + projector**: the multilingual MiniLM text encoder
  (~118M params) can't fit a Worker's 128 MB isolate, so cold-start embeds via
  the `@cf/baai/bge-m3` AI binding and a small **projector MLP** (trained offline,
  `--projector`) maps `[bge ⊕ numeric ⊕ acoustic ⊕ categorical] → existing 64-dim
  latent` for snapping. Corpus-only routing needs no AI and ships standalone.
- **Worker (`worker/index.ts`)**: replaced the proxy with the full handler —
  loads the snapshot via the ASSETS binding into the WASM core once per isolate,
  does Spotify search + cold-start fetches, and returns the unchanged
  `/api/route` contract.
- **Frontend redesign (`src/main.tsx`, `src/styles.css`)**: "observatory" look —
  glowing route tube + comet tracer, genre-clustered nebula with halos, cloud→path
  web, endpoint rings with snap badges, a journey timeline strip, a legend,
  first-class empty/loading/error states, route options (stops/time/shuffle), and
  a `prefers-reduced-motion` path. `PRODUCT.md` captures the design intent.
- **Config**: `wrangler.toml` gains `[ai]`, drops `PATHFINDER_API_ORIGIN`;
  `package.json` adds `build:wasm` / `build:snapshot`; Spotify creds move to
  Worker secrets / `.dev.vars`.

### Verified
- `cargo test` (parity + projector) pass.
- `npm run build` (wasm + tsc + vite) and `wrangler deploy --dry-run` clean
  (bundles to ~22 KB worker + 255 KB wasm; AI + ASSETS bindings detected).
- `wrangler dev --local`: `/api/health` → `{ready, tracks: 23529}`, corpus
  `/api/route` returns a full journey in ~0.09 s warm, Spotify `/api/search`
  annotates `in_corpus`.

### Not yet verified (needs Cloudflare credentials)
- Live cold-start: the projector snapshot (`--projector`) and the runtime AI
  binding both need a CF account (`CF_ACCOUNT_ID` + `CF_AI_TOKEN`), which weren't
  available here. All code is in place and unit-tested; enable per the README.

### `.env`
- Copied `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` from
  `../spotify-predict-engagement/.env` into `app/.env` and `app/.dev.vars`.

## Mobile responsive pass (2026-06-16)

Made the app genuinely usable on phones while keeping the 3D map the hero and the
desktop layout untouched. Touched only `src/main.tsx` + `src/styles.css`.

### What changed
- **Bottom-sheet composer (≤760px).** The always-expanded top control bar was
  crushing the map on small screens. The top bar now shows brand + a compact
  trigger that summarises the current pick ("Trace a journey…" → "A → B", or
  "Edit" once a route exists); tapping it slides up a bottom sheet with the mode
  switch, From/To pickers, **inline** route options, and a full-width Trace
  button. `useMediaQuery('(max-width:760px)')` drives both the JSX (inline
  options vs. desktop popover) and the CSS, so the two never disagree.
- **Tablet (761–1024px)** keeps the inline bar but stacks + wraps it.
- **Strip / inspector / legend** become safe-area-padded bottom docks/sheets;
  the legend (previously `display:none` on mobile) is restored as a floating
  panel — color is never the only signal, so it must stay reachable.

### Decisions (per request to document them)
1. **Composer pattern = bottom sheet** (user-approved). Search results in the
   sheet **flip upward** (`.menu { bottom: 100% }`) because the field sits low;
   opening downward would collide with the on-screen keyboard.
2. **Scrim / stacking.** The sheet lives inside `.topbar`, which forms a stacking
   context. With the default order the dismiss scrim (z50, a sibling) painted
   *over* the sheet — the reported "greyed out + tap-hides" bug. Fix: the mobile
   `.topbar` is raised to **z-index 70** (above the scrim), so the whole bar +
   sheet sit above it. Trade-off: the small top trigger isn't dimmed by the scrim
   while the sheet is open — accepted as cosmetic.
3. **Intro copy** is trimmed on mobile (`compact` prop) and repositioned to the
   lower third so it no longer covers the central splash dot-cloud.
4. **Splash dot tap → Inspector.** The idle cloud auto-rotates, which broke
   r3f's raycast `onClick` (different dot under the cursor at press vs. release);
   the old `hoveredRef` + DOM-click workaround only flashed the 3D label on touch.
   `GlowDot` now **captures the pointer on press** and opens on a tap (movement
   < 12px). Tapping a splash track opens the existing **Inspector** — for a
   sample track that's already the lighter, reasons-free card, i.e. the
   "intermediate" view between a name tooltip and a full route-stop breakdown.
   Trade-off: you can no longer *start* a camera-rotate drag from directly on top
   of a dot (the dot captures the press) — negligible given dot size; accepted.

### Verified
- `tsc -b` + `vite build` clean. Drove headless Chrome over CDP at 390 / 834 /
  1440 px: idle, composer sheet (confirmed the From field — not the scrim — is the
  topmost element at its centre), route, inspector, legend; desktop unchanged.
  Splash-dot tap confirmed to open the Inspector.

### Known residual risk
- **iOS soft keyboard.** A bottom-anchored sheet can be partially covered by the
  iOS keyboard while typing in a picker. Mitigated by flipping the results menu
  upward; not reproducible in headless. Worth a real-device check.
