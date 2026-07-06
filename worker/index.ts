/// <reference types="@cloudflare/workers-types" />
//
// Cloudflare Worker — fully self-contained pathfinder. No external origin, no
// Python, no Qdrant. Serves the static UI and answers /api/* by running the
// Rust/WASM compute core over a bundled corpus snapshot, with Spotify +
// ReccoBeats fetches and Workers AI embeddings for cold-start of arbitrary
// tracks. See WORKLOG.md / README.md for the architecture.

import {
  initSync,
  load_corpus,
  set_projector,
  set_scorer,
  set_layout,
  info,
  find_row,
  vec_at,
  meta_at,
  fit_at,
  route as wasmRoute,
  route_open as wasmRouteOpen,
  arrange as wasmArrange,
  project_and_snap,
  score_cold,
  search as corpusSearch,
  sample_field,
  embed_track,
  embed_track_open,
} from "../worker-core/pkg/worker_core.js";
import wasmModule from "../worker-core/pkg/worker_core_bg.wasm";

export interface Env {
  ASSETS: Fetcher;
  AI: Ai;
  SPOTIFY_CLIENT_ID?: string;
  SPOTIFY_CLIENT_SECRET?: string;
  EMBED_MODEL?: string;
}

const SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token";
const SPOTIFY_API = "https://api.spotify.com/v1";
const RECCOBEATS_API = "https://api.reccobeats.com/v1";
const UA = { "User-Agent": "spotify-pathfinder-visualizer/1.0", Accept: "application/json" };
const DEFAULT_EMBED_MODEL = "@cf/baai/bge-m3";

const AF_KEYS = [
  "danceability", "energy", "valence", "tempo", "acousticness",
  "instrumentalness", "loudness", "speechiness", "liveness", "key", "mode",
] as const;

// ---- per-isolate state (warm across requests) -----------------------------
interface CoreState {
  hasProjector: boolean;
  hasScorer: boolean;
  tracks: number;
  // corpus-wide raw-score bounds (from manifest.json) used to rescale a live
  // cold-start fit onto the baked [0,1] fit range; null when unavailable.
  fitMin: number | null;
  fitMax: number | null;
}
let ready: Promise<CoreState> | null = null;
let spotifyToken: { value: string; exp: number } | null = null;

async function ensureReady(env: Env): Promise<CoreState> {
  if (!ready) {
    ready = (async () => {
      initSync({ module: wasmModule });
      const [corpus, transitions] = await Promise.all([
        assetBytes(env, "/data/corpus.bin"),
        assetBytes(env, "/data/transitions.json"),
      ]);
      if (!corpus) throw new Error("corpus.bin asset missing — run tools/build_snapshot.py");
      load_corpus(corpus, transitions ?? new Uint8Array());
      const proj = await assetBytes(env, "/data/projector.bin");
      let hasProjector = false;
      if (proj) {
        try {
          set_projector(proj);
          hasProjector = true;
        } catch (e) {
          console.warn("projector load failed:", e);
        }
      }
      // Live cold-start fit scorer (optional, mirrors the baked corpus fit).
      const scorerBin = await assetBytes(env, "/data/scorer.bin");
      let hasScorer = false;
      if (scorerBin) {
        try {
          set_scorer(scorerBin);
          hasScorer = true;
        } catch (e) {
          console.warn("scorer load failed:", e);
        }
      }
      // Baked global 3D layout — shared coordinate space for splash + route.
      const layout = await assetBytes(env, "/data/layout.bin");
      if (layout) {
        try {
          set_layout(layout);
        } catch (e) {
          console.warn("layout load failed:", e);
        }
      }
      let fitMin: number | null = null;
      let fitMax: number | null = null;
      const manifest = await assetBytes(env, "/data/manifest.json");
      if (manifest) {
        try {
          const m = JSON.parse(new TextDecoder().decode(manifest));
          fitMin = typeof m.fit_raw_min === "number" ? m.fit_raw_min : null;
          fitMax = typeof m.fit_raw_max === "number" ? m.fit_raw_max : null;
        } catch (e) {
          console.warn("manifest parse failed:", e);
        }
      }
      const inf = JSON.parse(info());
      return { hasProjector, hasScorer, tracks: inf.n as number, fitMin, fitMax };
    })();
  }
  return ready;
}

// Rescale a raw cold-start fit (post nan/clip, already in [0,1]) onto the baked
// corpus fit scale via the persisted corpus min/max. Mirrors the min-max step in
// OnnxScorer.score_graph so a cold-start fit is comparable to corpus fit values.
function normalizeFit(raw: number, lo: number | null, hi: number | null): number {
  if (lo === null || hi === null) return Math.min(1, Math.max(0, raw));
  const span = hi - lo;
  if (span <= 1e-12) return 0.5;
  return Math.min(1, Math.max(0, (raw - lo) / span));
}

async function assetBytes(env: Env, path: string): Promise<Uint8Array | null> {
  const res = await env.ASSETS.fetch(new Request(`https://assets${path}`));
  if (!res.ok) return null;
  return new Uint8Array(await res.arrayBuffer());
}

// retry transient Spotify/ReccoBeats failures (429 + 5xx) with backoff,
// honoring Retry-After. workerd's local outbound is also flaky, so this helps dev.
async function fetchRetry(input: RequestInfo | URL, init?: RequestInit, attempts = 3): Promise<Response> {
  let last: Response | null = null
  for (let i = 0; i < attempts; i += 1) {
    try {
      const r = await fetch(input, init)
      if (r.status !== 429 && r.status < 500) return r
      last = r
    } catch (e) {
      if (i === attempts - 1) throw e
    }
    if (i < attempts - 1) {
      const ra = last ? Number(last.headers.get('retry-after')) : NaN
      const wait = Number.isFinite(ra) && ra > 0 ? ra * 1000 : 250 * (i + 1)
      await new Promise((res) => setTimeout(res, Math.min(wait, 2000)))
    }
  }
  return last as Response
}

// ---- Spotify + ReccoBeats --------------------------------------------------
function spotifyIdOf(value: string): string {
  const v = value.trim();
  if (v.startsWith("spotify:track:")) return v.split(":").pop()!;
  return v.split("?")[0].replace(/.*[/:]/, "");
}

// Parse a Spotify PLAYLIST id from a uri (spotify:playlist:ID), an
// open.spotify.com/playlist/ID URL, or a bare id. Distinct from spotifyIdOf
// (which is track-only and always rewraps as spotify:track:). Returns null when
// the input isn't recognizably a playlist reference.
function spotifyPlaylistIdOf(value: string): string | null {
  const v = value.trim();
  const m = /playlist[:/]([A-Za-z0-9]+)/.exec(v);
  if (m) return m[1];
  if (/^[A-Za-z0-9]{16,}$/.test(v)) return v; // bare id
  return null;
}

async function spotifyTokenGet(env: Env): Promise<string> {
  if (!env.SPOTIFY_CLIENT_ID || !env.SPOTIFY_CLIENT_SECRET) {
    throw new ApiError(500, "Spotify credentials missing (SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET).");
  }
  if (spotifyToken && spotifyToken.exp > Date.now() / 1000 + 30) return spotifyToken.value;
  const auth = btoa(`${env.SPOTIFY_CLIENT_ID}:${env.SPOTIFY_CLIENT_SECRET}`);
  const r = await fetchRetry(SPOTIFY_TOKEN_URL, {
    method: "POST",
    headers: { Authorization: `Basic ${auth}`, "Content-Type": "application/x-www-form-urlencoded" },
    body: "grant_type=client_credentials",
  });
  if (!r.ok) throw new ApiError(502, `Spotify token failed (${r.status})`);
  const data = (await r.json()) as { access_token: string; expires_in?: number };
  spotifyToken = { value: data.access_token, exp: Date.now() / 1000 + (data.expires_in ?? 3600) };
  return spotifyToken.value;
}

async function spotifyGet(env: Env, path: string, params?: Record<string, string>): Promise<any> {
  const url = new URL(`${SPOTIFY_API}/${path}`);
  if (params) for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  const r = await fetchRetry(url, { headers: { Authorization: `Bearer ${await spotifyTokenGet(env)}` } });
  if (!r.ok) {
    const msg =
      r.status === 429
        ? "Spotify is rate-limiting requests — give it a moment and try again."
        : `Spotify request failed (${r.status})`;
    throw new ApiError(r.status === 429 ? 429 : 502, msg);
  }
  return r.json();
}

async function searchSpotify(env: Env, query: string, limit = 10): Promise<any[]> {
  if (!query.trim()) return [];
  const data = await spotifyGet(env, "search", { q: query, type: "track", limit: String(limit) });
  return (data.tracks?.items ?? []).map((t: any) => {
    const imgs = t.album?.images ?? [];
    const uri = t.uri ?? `spotify:track:${t.id}`;
    return {
      spotify_id: t.id,
      uri,
      name: t.name ?? "Unknown track",
      artist: (t.artists ?? []).map((a: any) => a.name).filter(Boolean).join(", "),
      album: t.album?.name ?? null,
      art: imgs.length > 1 ? imgs[1].url : imgs[0]?.url ?? null,
      in_corpus: find_row(uri) >= 0,
    };
  });
}

// Free 30s previews for the procession playback. Spotify's own preview_url was
// deprecated in Nov 2024 and returns null for this app, so clips are sourced
// from Deezer (primary) with iTunes as a fallback.
//
// NOTE: Apple's iTunes Search API blocks Cloudflare Workers' egress IPs (returns
// 403/empty), so iTunes is effectively useless from the deployed Worker even
// though it works from a normal machine — hence Deezer leads. Both are proxied
// here (no browser CORS) and matched on the primary artist + a cleaned title.

// "Eminem, Rihanna" → "Eminem"; drop featured-artist noise.
function primaryArtist(artist: string): string {
  const a = artist.split(/,|;|&| feat\.?| ft\.?| with /i)[0]?.trim();
  return a || artist.trim();
}
// "Stan - Live (Remastered)" → "Stan"; strip parentheticals and dash-suffixes.
function cleanTitle(name: string): string {
  const t = name
    .replace(/\([^)]*\)/g, "")
    .replace(/\[[^\]]*\]/g, "")
    .replace(/\s-\s.*$/, "")
    .trim();
  return t || name.trim();
}

// Bypass Cloudflare's subrequest cache for the preview lookups: Deezer's search
// response embeds the short-lived signed preview URL, so a cached search result
// hands back a long-dead token. A unique cache-buster param makes every lookup
// URL distinct, so it always hits the upstream fresh instead of the edge cache.
function noCache(u: URL): RequestInit {
  u.searchParams.set("_cb", `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`);
  return { headers: UA, cf: { cacheTtl: 0 } } as RequestInit;
}

async function deezerPreview(query: string): Promise<string | null> {
  const url = new URL("https://api.deezer.com/search");
  url.searchParams.set("q", query);
  url.searchParams.set("limit", "1");
  try {
    const r = await fetchRetry(url, noCache(url));
    if (!r.ok) return null;
    const data = (await r.json()) as { data?: { preview?: string }[] };
    return data.data?.[0]?.preview || null;
  } catch {
    return null;
  }
}

async function itunesPreview(term: string): Promise<string | null> {
  const url = new URL("https://itunes.apple.com/search");
  url.searchParams.set("term", term);
  url.searchParams.set("entity", "song");
  url.searchParams.set("limit", "1");
  try {
    const r = await fetchRetry(url, noCache(url));
    if (!r.ok) return null;
    const data = (await r.json()) as { results?: { previewUrl?: string }[] };
    return data.results?.[0]?.previewUrl ?? null;
  } catch {
    return null;
  }
}

// Returns { preview_url, source } — source is for diagnostics only.
async function lookupPreview(
  artistRaw: string,
  trackRaw: string,
): Promise<{ preview_url: string | null; source: string }> {
  const artist = primaryArtist(artistRaw);
  const title = cleanTitle(trackRaw);
  if (!artist && !title) return { preview_url: null, source: "none" };
  // Deezer: strict field match first, then a loose query, then iTunes.
  let p = await deezerPreview(`artist:"${artist}" track:"${title}"`);
  if (p) return { preview_url: p, source: "deezer-strict" };
  p = await deezerPreview(`${artist} ${title}`.trim());
  if (p) return { preview_url: p, source: "deezer-loose" };
  p = await itunesPreview(`${artist} ${title}`.trim());
  if (p) return { preview_url: p, source: "itunes" };
  return { preview_url: null, source: "miss" };
}

async function reccobeatsFeatures(sid: string): Promise<Record<string, any>> {
  try {
    const r1 = await fetchRetry(`${RECCOBEATS_API}/track?ids=${sid}`, { headers: UA });
    if (!r1.ok) return {};
    const c1 = ((await r1.json()) as any).content ?? [];
    if (!c1.length) return {};
    const r2 = await fetchRetry(`${RECCOBEATS_API}/audio-features?ids=${c1[0].id}`, { headers: UA });
    if (!r2.ok) return {};
    const c2 = ((await r2.json()) as any).content ?? [];
    return c2[0] ?? {};
  } catch {
    return {};
  }
}

// Cold-start meta assembly — mirrors server/app.py ColdStartEncoder.encode_spotify.
async function fetchColdMeta(env: Env, uri: string): Promise<any> {
  const sid = spotifyIdOf(uri);
  const track = await spotifyGet(env, `tracks/${sid}`);
  const artistIds = (track.artists ?? []).map((a: any) => a.id).filter(Boolean);
  const artists = artistIds.length
    ? (await spotifyGet(env, "artists", { ids: artistIds.slice(0, 50).join(",") })).artists ?? []
    : [];
  const genres = [...new Set(artists.flatMap((a: any) => a.genres ?? []) as string[])].sort();
  const af = await reccobeatsFeatures(sid);
  const album = track.album ?? {};
  const releaseDate: string = album.release_date ?? "";
  const meta: Record<string, any> = {
    track_uri: `spotify:track:${sid}`,
    track_name: track.name ?? null,
    album: album.name ?? null,
    artist: track.artists?.[0]?.name ?? null,
    artist_count: (track.artists ?? []).length,
    track_popularity: track.popularity ?? null,
    artist_popularity: artists[0]?.popularity ?? null,
    artist_followers: artists[0]?.followers?.total ?? null,
    album_type: album.album_type ?? null,
    release_year: /^\d{4}/.test(releaseDate) ? parseInt(releaseDate.slice(0, 4), 10) : null,
    sp_duration_ms: track.duration_ms ?? null,
    sp_explicit: track.explicit ? 1 : 0,
    sp_track_number: track.track_number ?? null,
    sp_n_markets: (track.available_markets ?? []).length,
    sp_genres: genres,
    sp_genre_count: genres.length,
    genre_primary: genres[0] ?? "unknown",
  };
  for (const k of AF_KEYS) meta[`af_${k}`] = af[k] ?? null;
  return meta;
}

function contentDoc(m: any): string {
  const parts: string[] = [];
  if (m.track_name) parts.push(`${m.track_name}.`);
  if (m.artist) {
    let artist = String(m.artist);
    if ((m.artist_count || 1) > 1) artist += ` (with ${(m.artist_count | 0) - 1} other artist(s))`;
    parts.push(`Artist: ${artist}.`);
  }
  if (m.album) {
    const bits = [m.album_type, m.release_year ? String(m.release_year) : ""].filter(Boolean);
    parts.push(`Album: ${m.album}${bits.length ? ` (${bits.join(", ")}).` : "."}`);
  }
  if (m.genre_primary) parts.push(`Genre: ${m.genre_primary}.`);
  return parts.join(" ") || "Unknown track.";
}

async function embedText(env: Env, doc: string): Promise<Float32Array> {
  const model = env.EMBED_MODEL || DEFAULT_EMBED_MODEL;
  const res: any = await env.AI.run(model as any, { text: [doc] });
  const data = res?.data?.[0] ?? res?.data ?? res?.embedding;
  if (!Array.isArray(data)) throw new ApiError(502, "embedding model returned no vector");
  return Float32Array.from(data);
}

// ---- /api/route ------------------------------------------------------------
function todBucket(hour: number): string {
  if (hour < 6) return "night";
  if (hour < 12) return "morning";
  if (hour < 18) return "afternoon";
  return "evening";
}

interface Resolved {
  anchorRow: number;
  vec: Float32Array;
  display: any;
  exact: boolean;
  snappedId: string | null;
  anchorName?: string;
  // rotation-fit [0,1]: baked corpus value for exact tracks, live model score
  // for cold-start tracks, null if no scorer is loaded or scoring failed.
  fit: number | null;
}

async function resolveEndpoint(env: Env, uri: string, state: CoreState): Promise<Resolved> {
  const row = find_row(uri);
  if (row >= 0) {
    const m = JSON.parse(meta_at(row));
    return { anchorRow: row, vec: vec_at(row), display: m, exact: true, snappedId: null, fit: fit_at(row) };
  }
  if (!state.hasProjector) {
    throw new ApiError(
      422,
      "This track isn't in the corpus and cold-start embedding is unavailable. Pick an in-corpus track (badge) or rebuild the snapshot with the projector.",
    );
  }
  const cold = await fetchColdMeta(env, uri);
  const emb = await embedText(env, contentDoc(cold));
  const r = JSON.parse(project_and_snap(emb, JSON.stringify(cold)));
  const anchorMeta = JSON.parse(meta_at(r.snap_row));
  // Live fit: score the cold-start track from its projected latent + metadata,
  // then place it on the baked corpus scale. Non-fatal — a scoring failure just
  // leaves fit null.
  let fit: number | null = null;
  if (state.hasScorer) {
    try {
      const raw = score_cold(Float32Array.from(r.vec), JSON.stringify(cold));
      fit = normalizeFit(raw, state.fitMin, state.fitMax);
    } catch (e) {
      console.warn("cold-start scoring failed:", e);
    }
  }
  return {
    anchorRow: r.snap_row,
    vec: Float32Array.from(r.vec),
    display: {
      uri: cold.track_uri,
      name: cold.track_name ?? "Unknown track",
      artist: cold.artist ?? "Unknown artist",
      album: cold.album ?? null,
      genre: cold.genre_primary ?? "unknown",
    },
    exact: false,
    snappedId: r.snap_id,
    anchorName: anchorMeta.name,
    fit,
  };
}

// `opened` = an off-map endpoint was routed as a real node (not snapped), so it
// carries no snap link/label — it renders like an in-corpus endpoint.
function requestedNode(r: Resolved, pos: number[], opened = false): any {
  const uri = r.display.uri ?? "";
  const showSnap = !r.exact && !opened;
  return {
    id: uri,
    uri,
    name: r.display.name,
    artist: r.display.artist,
    album: r.display.album ?? null,
    genre: r.display.genre ?? "unknown",
    spotify_url: uri ? `https://open.spotify.com/track/${spotifyIdOf(uri)}` : null,
    fit: r.fit,
    position: pos,
    kind: "requested",
    snapped_to: showSnap ? r.snappedId : null,
    snapped_label: showSnap ? r.anchorName ?? null : null,
  };
}

async function buildRoute(env: Env, payload: any, state: CoreState) {
  const startUri = payload.start_uri;
  const endUri = payload.end_uri;
  if (!startUri || !endUri) throw new ApiError(400, "start_uri and end_uri are required");

  const [start, end] = await Promise.all([
    resolveEndpoint(env, startUri, state),
    resolveEndpoint(env, endUri, state),
  ]);

  const length = Math.max(4, Math.min(50, parseInt(payload.length, 10) || 14));
  const ctx = payload.context ?? "now";
  const tod = ctx === "any" ? "any" : ctx === "now" ? todBucket(new Date().getUTCHours()) : ctx;
  const shuffle = payload.shuffle ?? "any";

  // When an endpoint is off-map, route it as a real node (route_open) so the
  // path begins/ends at the track itself — threaded through its own nearest
  // corpus neighbors — rather than snapping to an anchor. Snapping stays the
  // fallback (no_path, or both endpoints already in-corpus).
  const offMap = !start.exact || !end.exact;
  let routeJson: any = null;
  let snapped = false;
  if (offMap) {
    try {
      routeJson = JSON.parse(
        wasmRouteOpen(
          start.exact ? start.anchorRow : -1,
          start.vec,
          start.fit ?? 0.5,
          JSON.stringify(start.display),
          end.exact ? end.anchorRow : -1,
          end.vec,
          end.fit ?? 0.5,
          JSON.stringify(end.display),
          length,
          tod,
          shuffle,
        ),
      );
    } catch (e: any) {
      if (!String(e?.message ?? e).includes("no_path")) throw e;
      // no real path to/from the off-map node — fall back to snapping below.
    }
  }
  if (!routeJson) {
    snapped = offMap; // a snap link is only meaningful when an endpoint was off-map
    try {
      routeJson = JSON.parse(
        wasmRoute(start.anchorRow, end.anchorRow, start.vec, end.vec, length, tod, shuffle),
      );
    } catch (e: any) {
      if (String(e?.message ?? e).includes("no_path")) {
        throw new ApiError(404, "No path found between these tracks");
      }
      throw e;
    }
  }

  const opened = offMap && !snapped;
  const edges = [...routeJson.edges];
  // snap edges only when we actually fell back to snapping an off-map endpoint.
  if (snapped) {
    if (!start.exact) edges.push({ from: start.display.uri, to: start.snappedId, kind: "snap" });
    if (!end.exact) edges.push({ from: end.snappedId, to: end.display.uri, kind: "snap" });
  }

  return {
    context: routeJson.context,
    requested_start: requestedNode(start, routeJson.req_start_pos, opened),
    requested_end: requestedNode(end, routeJson.req_end_pos, opened),
    path: routeJson.path,
    cloud: routeJson.cloud,
    edges,
  };
}

// ---- /api/embed ------------------------------------------------------------
// Single-track explorer: resolve one track (corpus or cold-start) and return it
// plus its neighborhood in the shared layout space.
async function buildEmbed(env: Env, payload: any, state: CoreState) {
  const uri = payload.uri;
  if (!uri) throw new ApiError(400, "uri is required");
  const resolved = await resolveEndpoint(env, uri, state);
  // Off-map: show the track's own neighborhood (its nearest corpus rows) instead
  // of the snapped anchor's. In-corpus: the anchor's neighborhood, as before.
  const emb = resolved.exact
    ? JSON.parse(embed_track(resolved.anchorRow, 40))
    : JSON.parse(embed_track_open(resolved.vec, 40));
  return { track: requestedNode(resolved, emb.pos, !resolved.exact), cloud: emb.cloud };
}

// ---- /api/drift ------------------------------------------------------------
// Explore the corpus by musical *intent* rather than by naming a song. The
// listener turns a few music-forward dials (an "air", breadth, temperament);
// this endpoint quietly translates that into a corpus query — pick a seed inside
// the air's genre family, biased along the rotation-fit axis by temperament, then
// grow a coherent neighbourhood (sample_field, the same primitive the splash
// uses) and filter it to the requested character. The construction stays hidden;
// the intent is musical; the result is a legible constellation. See PRODUCT.md
// ("make the abstract ML artifact legible") and README.
//
// Each air is a family of genre substrings matched against genre_primary. The
// substrings are tuned to the corpus's actual mass (Argentine/Latin-and-
// electronic-heavy): alternative dance, downtempo, big beat, alternative rock,
// idm, ambient, reggaetón/cumbia/tango, etc.
const AIRS: Record<string, { label: string; match: string[] }> = {
  nocturne: {
    label: "Nocturne",
    match: [
      "downtempo", "trip hop", "chillwave", "chillhop", "lo-fi house",
      "jazz house", "deep house", "ambient", "organic house", "dub techno",
      "minimal", "melodic house", "melodic techno", "nu jazz", "balearic",
      "chill", "lounge", "abstract",
    ],
  },
  aurora: {
    label: "Aurora",
    match: [
      "dance pop", "art pop", "dream pop", "synthpop", "indie pop", "chamber pop",
      "baroque pop", "city pop", "bedroom pop", "jangle", "power pop", "twee",
      "ambient pop", "folk-pop", "sunshine", "shibuya-kei", "beatlesque", "electropop",
    ],
  },
  kinetic: {
    label: "Kinetic",
    match: [
      "alternative dance", "big beat", "idm", "techno", "tech house", "acid house",
      "breakbeat", "jungle", "uk garage", "indie dance", "electro", "drum and bass",
      "dance-punk", "dance rock", "edm", "hard house", "rave", "electroclash",
      "hip house", "filter house", "disco house", "nu disco", "big room", "eurodance",
    ],
  },
  hearth: {
    label: "Hearth",
    match: [
      "alternative rock", "argentine rock", "album rock", "classic rock", "indie rock",
      "art rock", "garage rock", "blues", "folk", "singer-songwriter", "americana",
      "britpop", "post-punk", "new wave", "rock en", "mellow gold", "art punk",
      "psych", "soul", "funk", "madchester", "permanent wave",
    ],
  },
  undertow: {
    label: "Undertow",
    match: [
      "metal", "industrial", "darkwave", "dark wave", "cold wave", "ebm",
      "horror synth", "dark ambient", "drone", "gothic", "doom", "sludge",
      "stoner", "hardcore", "post-hardcore", "death", "noise", "witch house",
      "neue deutsche harte", "cyberpunk", "post-punk",
    ],
  },
  meridian: {
    label: "Meridian",
    match: [
      "reggaeton", "cumbia", "tango", "flamenco", "bachata", "salsa", "latin",
      "trap argentino", "neoperreo", "rkt", "cuarteto", "candombe", "tropical",
      "perreo", "dembow", "bolero", "chamamé", "murga", "bandoneon", "neotango",
      "folklore argentino", "techengue", "candombe",
    ],
  },
};

// Lazily built once per isolate: genre (lowercased) + fit for every corpus row,
// so a drift query can pool by air and rank by temperament without re-scanning.
interface DriftIndex {
  genre: string[]; // lowercased genre_primary per row
  fit: Float32Array;
  counts: Map<string, number>; // genre → number of tracks (the tuning catalog)
}
let driftIndex: DriftIndex | null = null;
function ensureDriftIndex(n: number): DriftIndex {
  if (driftIndex) return driftIndex;
  const genre = new Array<string>(n);
  const fit = new Float32Array(n);
  const counts = new Map<string, number>();
  for (let r = 0; r < n; r += 1) {
    let g = "";
    try {
      g = (JSON.parse(meta_at(r)).genre ?? "").toString().toLowerCase();
    } catch {
      /* leave blank — never matches, still routable */
    }
    genre[r] = g;
    fit[r] = fit_at(r);
    if (g) counts.set(g, (counts.get(g) ?? 0) + 1);
  }
  driftIndex = { genre, fit, counts };
  return driftIndex;
}

// The tuning catalog: every genre + its track count, most-populated first. Feeds
// the Drift genre picker so a listener can build the exact blend they want.
function driftCatalog(state: CoreState) {
  const idx = ensureDriftIndex(state.tracks);
  return [...idx.counts.entries()]
    .filter(([name]) => name && name !== "unknown")
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({ name, count }));
}

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
const DRIFT_ARTIST_CAP = 4; // keep one artist from swallowing the constellation

async function buildDrift(_env: Env, payload: any, state: CoreState) {
  const idx = ensureDriftIndex(state.tracks);

  // The tuning surface. The primary input is now an explicit set of genres (the
  // listener builds the exact blend); `air` is kept only as a back-compat preset
  // for older clients / API users. `familiarity`, `focus`, and `count` are the
  // concrete dials — see the frontend for their plain-language framing.
  const chosen = new Set<string>(
    (Array.isArray(payload.genres) ? payload.genres : [])
      .map((s: unknown) => String(s).toLowerCase().trim())
      .filter(Boolean),
  );
  const useGenres = chosen.size > 0;
  const airKey = typeof payload.air === "string" ? payload.air : "nocturne";
  const air = AIRS[airKey] ?? AIRS.nocturne;
  const inBlend = (g: string) => {
    const s = (g ?? "").toLowerCase();
    return useGenres ? chosen.has(s) : air.match.some((m) => s.includes(m));
  };

  // familiarity: 0 = mainstream / well-worn (high rotation-fit) … 1 = deep cuts
  // (low fit). Back-compat: `temper` is the old name.
  const familiarity = clamp01(Number(payload.familiarity ?? payload.temper ?? 0.35));
  // focus: 0 = strictly the chosen genres (tight) … 1 = let it wander (more
  // off-genre bleed + a wider scatter of anchors). Back-compat: `breadth` widens.
  const focus = clamp01(Number(payload.focus ?? payload.wander ?? 0.3));
  // count: how many stars to gather. Back-compat: derive from `breadth`.
  const count = Number.isFinite(payload.count)
    ? Math.max(20, Math.min(250, Math.round(Number(payload.count))))
    : Math.round(70 + clamp01(Number(payload.breadth ?? 0.5)) * 130);

  // Pool = every row in the chosen blend. Fall back to the whole corpus if it
  // lands thin, so a drift never dead-ends.
  let pool: number[] = [];
  for (let r = 0; r < state.tracks; r += 1) {
    if (inBlend(idx.genre[r])) pool.push(r);
  }
  const pooled = pool.length;
  if (pool.length < 30) pool = Array.from({ length: state.tracks }, (_, r) => r);

  // Scatter anchors across a fit-band window centred on the familiarity dial —
  // several tight knots rather than one, so the sky is diverse and stays loyal to
  // the blend (a single-seed walk drifts off it). Focus widens the anchor count.
  const byFit = pool.slice().sort((a, b) => idx.fit[b] - idx.fit[a]); // high → low
  const half = 0.24;
  const lo = Math.max(0, Math.floor((familiarity - half) * byFit.length));
  const hi = Math.max(lo + 1, Math.min(byFit.length, Math.ceil((familiarity + half) * byFit.length)));
  const band = byFit.slice(lo, hi);
  // Scatter a handful of seeds across the fit-band and grow a neighbourhood field
  // from each (sample_field: a connected BFS slice of the manifold, up to a few
  // hundred positioned nodes per call — far higher in-blend yield than the 20-wide
  // KNN fan-out). We then keep the in-blend tracks and let focus decide the bleed.
  const nSeeds = Math.min(band.length, 30, Math.max(8, Math.ceil(count / 8)));
  const perSeed = Math.min(220, Math.max(90, count));
  const seedSet = new Set<number>();
  let guard = 0;
  while (seedSet.size < nSeeds && guard < nSeeds * 12) {
    seedSet.add(band[Math.floor(Math.random() * band.length)]);
    guard += 1;
  }

  // Merge the fields; keep the in-blend tracks first; cap per artist so no one act
  // swallows the sky. Baked global-layout coords, so every field shares one space.
  const seen = new Set<string>();
  const perArtist = new Map<string, number>();
  const inGenre: any[] = [];
  const near: any[] = [];
  for (const s of seedSet) {
    const field = JSON.parse(sample_field(s, perSeed)) as any[];
    for (const nd of field) {
      if (seen.has(nd.id)) continue;
      seen.add(nd.id);
      const artist = (nd.artist ?? "").toLowerCase();
      const used = perArtist.get(artist) ?? 0;
      if (used >= DRIFT_ARTIST_CAP) continue;
      perArtist.set(artist, used + 1);
      (inBlend(nd.genre) ? inGenre : near).push({ ...nd, kind: "drift" });
    }
  }

  // In-blend tracks are the constellation; focus controls how much off-genre
  // bleed (the edge of the blend — still nearest-neighbours) rounds it out. Low
  // focus stays pure; high focus lets in more. A niche blend simply returns fewer
  // stars rather than a padded, incoherent sky (PRODUCT: honest about what's there).
  const nearBudget = Math.max(
    0,
    Math.min(near.length, Math.round(count * focus * 0.6), count - inGenre.length),
  );
  const kept = inGenre.slice(0, count).concat(near.slice(0, nearBudget));

  // Dominant genres actually present — the honest read-back for the legend.
  const gc = new Map<string, number>();
  for (const n of kept) gc.set(n.genre, (gc.get(n.genre) ?? 0) + 1);
  const genres = [...gc.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([name, n]) => ({ name, count: n }));

  return {
    air: useGenres ? null : airKey,
    label: useGenres ? "" : air.label,
    genresIn: [...chosen], // echo the requested blend
    familiarity,
    focus,
    requested: count,
    seed: [...seedSet][0] ?? 0,
    pooled, // how large the chosen blend was (before any fallback)
    count: kept.length,
    genres,
    tracks: kept,
  };
}

// ---- /api/arrange + /api/playlist (Bring Your Own Playlist) ----------------
// Upper bounds that keep an arrange request inside the Worker CPU + subrequest
// budget. Every not-in-corpus track costs a cold-start (Spotify track+artists +
// ReccoBeats + one AI embed ≈ 4 subrequests); in-corpus tracks are free. We keep
// at most BYOP_MAX_TRACKS of the playlist and cold-start at most
// BYOP_MAX_COLDSTART of those — anything beyond is reported, never silently
// dropped (PRODUCT: "no silent caps", "honest about uncertainty").
const BYOP_MAX_TRACKS = 50;
const BYOP_MAX_COLDSTART = 20;
const BYOP_COLDSTART_CONCURRENCY = 4;

// Read a PUBLIC playlist's tracks via the app's client-credentials token (no
// user auth). Private/collaborative playlists — and, since Nov 2024, Spotify's
// own editorial playlists — aren't readable this way; those come in through the
// browser user token instead. Returns display rows + the spotify:track: uris.
async function fetchPublicPlaylist(
  env: Env,
  idOrUrl: string,
): Promise<{ id: string; name: string; uris: string[]; tracks: any[]; total: number }> {
  const id = spotifyPlaylistIdOf(idOrUrl);
  if (!id) throw new ApiError(400, "Couldn't read a Spotify playlist link or id from that input.");
  let meta: any;
  try {
    meta = await spotifyGet(env, `playlists/${id}`, { fields: "name" });
  } catch (e: any) {
    if (e instanceof ApiError && (e.status === 404 || e.status === 403)) {
      throw new ApiError(
        404,
        "Couldn't open that playlist — it must be a public, user-made playlist (private, collaborative, and Spotify's own editorial playlists can't be read).",
      );
    }
    throw e;
  }
  const uris: string[] = [];
  const tracks: any[] = [];
  let offset = 0;
  const LIMIT = 100;
  // fetch a little past the cap so the truncation notice is honest.
  const HARD = BYOP_MAX_TRACKS + 50;
  for (;;) {
    const page = await spotifyGet(env, `playlists/${id}/tracks`, {
      fields: "items(track(uri,name,artists(name),is_local,type)),next",
      limit: String(LIMIT),
      offset: String(offset),
    });
    const items = page.items ?? [];
    for (const it of items) {
      const t = it?.track;
      if (!t || t.is_local || t.type !== "track") continue;
      const uri: string = t.uri ?? "";
      if (!uri.startsWith("spotify:track:")) continue;
      uris.push(uri);
      tracks.push({ uri, name: t.name ?? "Unknown track", artist: (t.artists ?? []).map((a: any) => a.name).filter(Boolean).join(", ") });
    }
    if (!page.next || items.length === 0 || uris.length >= HARD) break;
    offset += LIMIT;
  }
  return { id, name: meta?.name ?? "Playlist", uris, tracks, total: uris.length };
}

// Resolve a batch of track uris to corpus/cold-start endpoints under the
// cold-start budget. In-corpus tracks resolve for free; off-corpus tracks are
// cold-started up to the cap with bounded concurrency. Never throws for one bad
// track — failures and budget overflow are collected in `unresolved`.
async function resolveArrangeBatch(
  env: Env,
  uris: string[],
  state: CoreState,
): Promise<{ resolved: Resolved[]; unresolved: { uri: string; reason: string }[] }> {
  const resolved: Resolved[] = [];
  const unresolved: { uri: string; reason: string }[] = [];
  const cold: string[] = [];
  for (const uri of uris) {
    if (find_row(uri) >= 0) {
      // in-corpus: no network / AI — resolve immediately.
      resolved.push(await resolveEndpoint(env, uri, state));
    } else {
      cold.push(uri);
    }
  }
  const toCold = cold.slice(0, BYOP_MAX_COLDSTART);
  for (const uri of cold.slice(BYOP_MAX_COLDSTART))
    unresolved.push({ uri, reason: "cold-start budget reached" });
  for (let i = 0; i < toCold.length; i += BYOP_COLDSTART_CONCURRENCY) {
    const chunk = toCold.slice(i, i + BYOP_COLDSTART_CONCURRENCY);
    const settled = await Promise.allSettled(chunk.map((u) => resolveEndpoint(env, u, state)));
    settled.forEach((s, j) => {
      if (s.status === "fulfilled") resolved.push(s.value);
      else unresolved.push({ uri: chunk[j], reason: String(s.reason?.message ?? s.reason).slice(0, 140) });
    });
  }
  return { resolved, unresolved };
}

// A `requested`-kind endpoint node built from an arrange path endpoint (which is
// already one of the user's own tracks). No snap link — the track IS the stop.
function requestedFromNode(n: any, pos: number[]): any {
  return {
    id: n.id,
    uri: n.uri,
    name: n.name,
    artist: n.artist,
    album: n.album ?? null,
    genre: n.genre ?? "unknown",
    spotify_url: n.spotify_url ?? null,
    fit: n.fit ?? null,
    position: pos,
    kind: "requested",
    snapped_to: null,
    snapped_label: null,
  };
}

// Re-order a user-supplied playlist into a journey. Body: { uris[], name?,
// context?, shuffle? }. Resolves each uri (corpus or cold-start, budgeted),
// permutes the fixed set via the WASM `arrange`, and returns the same shape as
// /api/route — so the frontend composer + save consume it unchanged — plus a
// `source` block describing the playlist and anything that couldn't be placed.
async function buildArrange(env: Env, payload: any, state: CoreState) {
  const rawUris: unknown = payload.uris;
  if (!Array.isArray(rawUris) || rawUris.length === 0) {
    throw new ApiError(400, "uris (a non-empty array of Spotify track uris/ids) is required");
  }
  // Normalize to spotify:track: uris and de-dupe, preserving first-seen order.
  const seen = new Set<string>();
  const norm: string[] = [];
  for (const u of rawUris) {
    if (typeof u !== "string" || !u.trim()) continue;
    const uri = `spotify:track:${spotifyIdOf(u)}`;
    if (!seen.has(uri)) {
      seen.add(uri);
      norm.push(uri);
    }
  }
  const kept = norm.slice(0, BYOP_MAX_TRACKS);
  const truncated = norm.length - kept.length;

  const { resolved, unresolved } = await resolveArrangeBatch(env, kept, state);
  if (resolved.length < 2) {
    throw new ApiError(
      422,
      "Couldn't place enough of this playlist on the map — need at least 2 tracks that are in the corpus or can be cold-started.",
    );
  }

  const tracks = resolved.map((r) =>
    r.exact
      ? { row: r.anchorRow }
      : {
          row: -1,
          vec: Array.from(r.vec),
          fit: r.fit ?? 0.5,
          meta: {
            uri: r.display.uri,
            name: r.display.name,
            artist: r.display.artist,
            album: r.display.album ?? null,
            genre: r.display.genre ?? "unknown",
          },
        },
  );

  const ctx = payload.context ?? "now";
  const tod = ctx === "any" ? "any" : ctx === "now" ? todBucket(new Date().getUTCHours()) : ctx;
  const shuffle = payload.shuffle ?? "any";

  const out = JSON.parse(wasmArrange(JSON.stringify(tracks), tod, shuffle));
  const path = out.path as any[];

  return {
    context: out.context,
    requested_start: requestedFromNode(path[0], out.req_start_pos),
    requested_end: requestedFromNode(path[path.length - 1], out.req_end_pos),
    path,
    cloud: out.cloud,
    edges: out.edges,
    // BYOP-only metadata: the source playlist + an honest account of what was
    // dropped, so the UI can surface it (the /api/route contract is otherwise
    // untouched, and route/explore callers just ignore this field).
    source: {
      name: typeof payload.name === "string" && payload.name.trim() ? payload.name.trim() : null,
      placed: resolved.length,
      requested: norm.length,
      truncated,
      unresolved,
    },
  };
}

// ---- HTTP plumbing ---------------------------------------------------------
class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "content-type",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
};

function json(obj: unknown, status = 200): Response {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });
}

function htmlResponse(html: string, status = 200): Response {
  return new Response(html, {
    status,
    headers: { "Content-Type": "text/html; charset=utf-8", ...CORS },
  });
}

// ---- API docs --------------------------------------------------------------
// Swagger UI, loaded from a CDN (consistent with the app shell, which already
// pulls web fonts from a CDN). It renders the spec served at /api/openapi.json.
const SWAGGER_HTML = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pathfinder API — Swagger UI</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui.css" />
  <style>body { margin: 0; background: #fafafa; }</style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui-bundle.js" crossorigin></script>
  <script>
    window.ui = SwaggerUIBundle({
      url: "/api/openapi.json",
      dom_id: "#swagger-ui",
      deepLinking: true,
      tryItOutEnabled: true,
    });
  </script>
</body>
</html>`;

// OpenAPI 3.0 description of the public /api surface. `origin` is wired in as the
// server so Swagger UI's "Try it out" hits this same deployment.
function openApiSpec(origin: string) {
  const trackNode = {
    type: "object",
    description: "A track placed in the shared 3D layout space.",
    properties: {
      id: { type: "string" },
      uri: { type: "string", example: "spotify:track:7b4fQNd34RVNFKKziQz6mS" },
      name: { type: "string" },
      artist: { type: "string" },
      album: { type: "string", nullable: true },
      genre: { type: "string" },
      spotify_url: { type: "string", nullable: true },
      fit: { type: "number", nullable: true, description: "Rotation-fit score, 0–1. Baked for in-corpus tracks; computed live (and approximate, via the cold-start latent) for off-corpus tracks; null only if no scorer is loaded." },
      w: { type: "number", description: "Hidden 4th PCA axis (ribbon bank)." },
      position: {
        type: "array",
        items: { type: "number" },
        minItems: 3,
        maxItems: 3,
        description: "[x, y, z] in the shared layout space.",
      },
      kind: { type: "string", enum: ["path", "cloud", "requested"] },
      snapped_to: { type: "string", nullable: true, description: "Anchor id, set only on the snap fallback (an off-corpus endpoint that could not be routed as a real node). Null when the off-corpus track was routed directly." },
      snapped_label: { type: "string", nullable: true },
    },
  };
  const routeResponse = {
    type: "object",
    properties: {
      context: { type: "string", nullable: true, description: "Time-of-day context the route was scored under." },
      requested_start: { $ref: "#/components/schemas/TrackNode" },
      requested_end: { $ref: "#/components/schemas/TrackNode" },
      path: { type: "array", items: { $ref: "#/components/schemas/TrackNode" }, description: "Ordered stops, start → end." },
      cloud: { type: "array", items: { $ref: "#/components/schemas/TrackNode" }, description: "Nearby tracks for context." },
      edges: {
        type: "array",
        items: {
          type: "object",
          properties: {
            from: { type: "string" },
            to: { type: "string" },
            kind: { type: "string", enum: ["path", "snap"] },
          },
        },
      },
    },
  };
  const errorResponse = {
    description: "Error",
    content: {
      "application/json": {
        schema: { type: "object", properties: { error: { type: "string" } } },
      },
    },
  };
  const okRoute = {
    description: "A traced route.",
    content: { "application/json": { schema: { $ref: "#/components/schemas/RouteResponse" } } },
  };
  return {
    openapi: "3.0.3",
    info: {
      title: "Spotify Pathfinder API",
      version: "1.0.0",
      description:
        "Traces an A* route across the taste map between two Spotify tracks and " +
        "returns it as an ordered path in a shared 3D layout space. Arbitrary " +
        "(non-corpus) tracks are cold-start embedded and routed as real endpoints " +
        "of the path; if no path can be found that way, they fall back to snapping " +
        "to the nearest in-corpus anchor.",
    },
    servers: [{ url: origin }],
    paths: {
      "/api/path": {
        get: {
          summary: "Trace a route (GET)",
          description: "Shareable, curl-friendly form of POST /api/route.",
          parameters: [
            { name: "from", in: "query", required: true, schema: { type: "string" }, description: "Start track: Spotify id, spotify:track: uri, or open.spotify.com URL.", example: "7b4fQNd34RVNFKKziQz6mS" },
            { name: "to", in: "query", required: true, schema: { type: "string" }, description: "End track (same formats as from).", example: "3IvodZAm4vD1PM3bIEw9Ik" },
            { name: "len", in: "query", schema: { type: "integer", default: 14, minimum: 4, maximum: 50 }, description: "Target number of stops." },
            { name: "ctx", in: "query", schema: { type: "string", default: "now", enum: ["now", "any", "morning", "afternoon", "evening", "night"] }, description: "Time-of-day context for scoring." },
            { name: "shuf", in: "query", schema: { type: "string", default: "any" }, description: "Shuffle/variation preference." },
          ],
          responses: { "200": okRoute, "400": errorResponse, "404": errorResponse, "422": errorResponse },
        },
      },
      "/api/route": {
        post: {
          summary: "Trace a route (POST)",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  required: ["start_uri", "end_uri"],
                  properties: {
                    start_uri: { type: "string", example: "spotify:track:7b4fQNd34RVNFKKziQz6mS" },
                    end_uri: { type: "string", example: "spotify:track:3IvodZAm4vD1PM3bIEw9Ik" },
                    length: { type: "integer", default: 14, minimum: 4, maximum: 50 },
                    context: { type: "string", default: "now" },
                    shuffle: { type: "string", default: "any" },
                  },
                },
              },
            },
          },
          responses: { "200": okRoute, "400": errorResponse, "404": errorResponse, "422": errorResponse },
        },
      },
      "/api/embed": {
        post: {
          summary: "Embed one track and its neighborhood",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: { type: "object", required: ["uri"], properties: { uri: { type: "string", example: "spotify:track:7b4fQNd34RVNFKKziQz6mS" } } },
              },
            },
          },
          responses: {
            "200": {
              description: "The track plus its nearest neighbors in the layout space.",
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    properties: {
                      track: { $ref: "#/components/schemas/TrackNode" },
                      cloud: { type: "array", items: { $ref: "#/components/schemas/TrackNode" } },
                    },
                  },
                },
              },
            },
            "400": errorResponse,
            "422": errorResponse,
          },
        },
      },
      "/api/arrange": {
        post: {
          summary: "Re-order your own playlist into a journey (BYOP)",
          description:
            "Takes a fixed list of Spotify track uris/ids and re-orders them into " +
            "the lowest-cost journey across the taste map — keeping every track (no " +
            "adds, no drops). Response matches /api/route (path = your tracks in the " +
            "chosen order, cloud = their neighbors) plus a `source` block reporting " +
            "how many tracks were placed and anything that couldn't be. Capped at " +
            `${BYOP_MAX_TRACKS} tracks, of which up to ${BYOP_MAX_COLDSTART} may be cold-started.`,
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  required: ["uris"],
                  properties: {
                    uris: { type: "array", items: { type: "string" }, example: ["spotify:track:7b4fQNd34RVNFKKziQz6mS", "spotify:track:3IvodZAm4vD1PM3bIEw9Ik"] },
                    name: { type: "string", description: "Source playlist name (used for the saved playlist)." },
                    context: { type: "string", default: "now" },
                    shuffle: { type: "string", default: "any" },
                  },
                },
              },
            },
          },
          responses: { "200": okRoute, "400": errorResponse, "422": errorResponse },
        },
      },
      "/api/playlist": {
        get: {
          summary: "Read a public playlist's tracks (BYOP)",
          description:
            "Returns a public, user-made playlist's track uris via the app's " +
            "client-credentials token. Private, collaborative, and Spotify editorial " +
            "playlists are not readable this way (use the account-connect path).",
          parameters: [
            { name: "url", in: "query", schema: { type: "string" }, description: "Playlist link, spotify:playlist: uri, or bare id.", example: "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M" },
            { name: "id", in: "query", schema: { type: "string" }, description: "Alternative to url." },
          ],
          responses: {
            "200": {
              description: "The playlist name and its track uris.",
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    properties: {
                      id: { type: "string" },
                      name: { type: "string" },
                      total: { type: "integer" },
                      uris: { type: "array", items: { type: "string" } },
                      tracks: { type: "array", items: { type: "object", properties: { uri: { type: "string" }, name: { type: "string" }, artist: { type: "string" } } } },
                    },
                  },
                },
              },
            },
            "400": errorResponse,
            "404": errorResponse,
          },
        },
      },
      "/api/search": {
        get: {
          summary: "Search tracks",
          parameters: [{ name: "q", in: "query", required: true, schema: { type: "string" }, example: "alan braxe" }],
          responses: {
            "200": {
              description: "Matching tracks.",
              content: {
                "application/json": {
                  schema: {
                    type: "array",
                    items: {
                      type: "object",
                      properties: {
                        spotify_id: { type: "string" },
                        uri: { type: "string" },
                        name: { type: "string" },
                        artist: { type: "string" },
                        album: { type: "string", nullable: true },
                        art: { type: "string", nullable: true },
                        in_corpus: { type: "boolean" },
                      },
                    },
                  },
                },
              },
            },
          },
        },
      },
      "/api/sample": {
        get: {
          summary: "Sample a region of the corpus (splash field)",
          parameters: [
            { name: "n", in: "query", schema: { type: "integer", default: 130, minimum: 1, maximum: 400 } },
            { name: "seed", in: "query", schema: { type: "integer" }, description: "Omit for a random region." },
          ],
          responses: {
            "200": { description: "A neighborhood of tracks.", content: { "application/json": { schema: { type: "array", items: { $ref: "#/components/schemas/TrackNode" } } } } },
          },
        },
      },
      "/api/genres": {
        get: {
          summary: "Drift tuning catalog — every genre and its track count",
          responses: {
            "200": {
              description: "Genres, most-populated first.",
              content: {
                "application/json": {
                  schema: {
                    type: "array",
                    items: { type: "object", properties: { name: { type: "string" }, count: { type: "integer" } } },
                  },
                },
              },
            },
          },
        },
      },
      "/api/drift": {
        post: {
          summary: "Explore the corpus by musical intent (a constellation)",
          description:
            "Gather a coherent cluster of corpus tracks from an explicit blend of " +
            "genres (see GET /api/genres), tuned by familiarity (0 mainstream … 1 " +
            "deep cuts), focus (0 strictly the chosen genres … 1 let it wander), " +
            "and count. Legacy: an `air` preset (nocturne, aurora, kinetic, hearth, " +
            "undertow, meridian) with breadth/temper still works when no genres given.",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  properties: {
                    genres: { type: "array", items: { type: "string" }, example: ["deep house", "downtempo"] },
                    familiarity: { type: "number", minimum: 0, maximum: 1, default: 0.35 },
                    focus: { type: "number", minimum: 0, maximum: 1, default: 0.3 },
                    count: { type: "integer", minimum: 20, maximum: 250, default: 100 },
                    air: { type: "string", example: "nocturne", description: "Legacy preset used only when genres is empty." },
                  },
                },
              },
            },
          },
          responses: {
            "200": {
              description: "The chosen air and the constellation it gathered.",
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    properties: {
                      air: { type: "string" },
                      label: { type: "string" },
                      breadth: { type: "number" },
                      temper: { type: "number" },
                      seed: { type: "integer" },
                      pooled: { type: "integer" },
                      count: { type: "integer" },
                      genres: {
                        type: "array",
                        items: { type: "object", properties: { name: { type: "string" }, count: { type: "integer" } } },
                      },
                      tracks: { type: "array", items: { $ref: "#/components/schemas/TrackNode" } },
                    },
                  },
                },
              },
            },
          },
        },
      },
      "/api/preview": {
        get: {
          summary: "Find a 30s preview clip for a track",
          parameters: [
            { name: "artist", in: "query", schema: { type: "string" }, example: "Alan Braxe" },
            { name: "track", in: "query", schema: { type: "string" }, example: "Intro" },
          ],
          responses: {
            "200": {
              description: "A preview URL (or null) and its source.",
              content: { "application/json": { schema: { type: "object", properties: { preview_url: { type: "string", nullable: true }, source: { type: "string" } } } } },
            },
          },
        },
      },
      "/api/health": {
        get: {
          summary: "Service health + corpus size",
          responses: { "200": { description: "OK", content: { "application/json": { schema: { type: "object", properties: { ready: { type: "boolean" }, tracks: { type: "integer" }, cold_start: { type: "boolean" } } } } } } },
        },
      },
      "/api/config": {
        get: {
          summary: "Public client config",
          responses: { "200": { description: "OK", content: { "application/json": { schema: { type: "object", properties: { spotify_client_id: { type: "string", nullable: true } } } } } } },
        },
      },
    },
    components: { schemas: { TrackNode: trackNode, RouteResponse: routeResponse } },
  };
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });

    if (url.pathname.startsWith("/api/")) {
      // API docs need no compute core — serve them before warming the corpus so
      // they work even if the snapshot is missing.
      if (url.pathname === "/api/docs") return htmlResponse(SWAGGER_HTML);
      if (url.pathname === "/api/openapi.json") return json(openApiSpec(url.origin));
      try {
        const state = await ensureReady(env);
        if (url.pathname === "/api/health") {
          return json({ ready: true, tracks: state.tracks, cold_start: state.hasProjector });
        }
        if (url.pathname === "/api/config") {
          // client id is public (not secret) — the browser needs it for the
          // PKCE playlist-export flow. null hides the playlist button in the UI.
          return json({ spotify_client_id: env.SPOTIFY_CLIENT_ID ?? null });
        }
        if (url.pathname === "/api/preview" && request.method === "GET") {
          const artist = (url.searchParams.get("artist") ?? "").trim();
          const track = (url.searchParams.get("track") ?? "").trim();
          if (!track && !artist) return json({ preview_url: null, source: "none" });
          // Deezer signs preview URLs with a SHORT-LIVED token (~15 min) in the
          // `exp` query param. The whole point of failure was that the signed URL
          // got cached far longer than the token lives, so dead (403) links were
          // served — "songs don't play". Defenses, in order:
          //   1. key bumped to v4 so every poisoned earlier entry is abandoned;
          //   2. on a cache HIT, re-validate the token and refetch if it's expired
          //      (caches.default is shared across our Workers and Cloudflare can
          //      serve entries past their max-age, so TTL alone isn't enough);
          //   3. the CLIENT response is `no-store`, so neither the browser nor any
          //      edge re-caches the signed URL keyed by request URL;
          //   4. the Deezer/iTunes lookups themselves bypass the subrequest cache
          //      (see noCache) so a refetch is always a genuinely fresh token.
          const cache = caches.default;
          const key = new Request(
            `https://pf.cache/preview-v4?a=${encodeURIComponent(artist.toLowerCase())}&t=${encodeURIComponent(
              track.toLowerCase(),
            )}`,
          );
          const nowS = () => Math.floor(Date.now() / 1000);
          const tokenValid = (u: string | null | undefined): boolean => {
            if (!u) return true; // a cached `null` (miss) is fine to serve
            const m = /[?&]exp=(\d+)/.exec(u);
            if (!m) return true; // no expiry token (e.g. iTunes) → stable
            return parseInt(m[1], 10) - nowS() > 60;
          };

          let found: { preview_url: string | null; source: string } | null = null;
          const hit = await cache.match(key);
          if (hit) {
            try {
              const cached = (await hit.json()) as { preview_url: string | null; source: string };
              if (tokenValid(cached.preview_url)) found = cached;
            } catch {
              /* unparseable hit → refetch */
            }
          }
          if (!found) {
            found = await lookupPreview(artist, track);
            let ttl: number;
            if (!found.preview_url) {
              ttl = 600;
            } else {
              const m = /[?&]exp=(\d+)/.exec(found.preview_url);
              ttl = m ? Math.max(60, parseInt(m[1], 10) - nowS() - 60) : 86400;
            }
            ctx.waitUntil(
              cache.put(
                key,
                new Response(JSON.stringify(found), {
                  headers: { "Content-Type": "application/json", "Cache-Control": `max-age=${ttl}`, ...CORS },
                }),
              ),
            );
          }
          // never let the short-lived signed URL be cached downstream of us
          return new Response(JSON.stringify(found), {
            headers: { "Content-Type": "application/json", "Cache-Control": "no-store", ...CORS },
          });
        }
        if (url.pathname === "/api/sample" && request.method === "GET") {
          // A random subspace of the corpus for the idle/splash field: a coherent
          // kNN neighborhood grown from a random seed, projected to 3D. Picked
          // fresh per request (not cached) so each app load shows a new region.
          const n = Math.max(1, Math.min(400, parseInt(url.searchParams.get("n") ?? "130", 10) || 130));
          const seedParam = url.searchParams.get("seed");
          const seed =
            seedParam != null && /^\d+$/.test(seedParam)
              ? parseInt(seedParam, 10)
              : Math.floor(Math.random() * state.tracks);
          return json(JSON.parse(sample_field(seed, n)));
        }
        if (url.pathname === "/api/search" && request.method === "GET") {
          const q = (url.searchParams.get("q") ?? "").trim();
          if (!q) return json([]);
          // cache identical queries (prefixes repeat heavily while typing) to
          // spare Spotify's rate limit.
          const cache = caches.default;
          const key = new Request(`https://pf.cache/search?q=${encodeURIComponent(q.toLowerCase())}`);
          const hit = await cache.match(key);
          if (hit) return hit;
          const results =
            env.SPOTIFY_CLIENT_ID && env.SPOTIFY_CLIENT_SECRET
              ? await searchSpotify(env, q)
              : JSON.parse(corpusSearch(q, 12)); // fallback when no Spotify creds
          const res = new Response(JSON.stringify(results), {
            headers: { "Content-Type": "application/json", "Cache-Control": "max-age=300", ...CORS },
          });
          ctx.waitUntil(cache.put(key, res.clone()));
          return res;
        }
        if (url.pathname === "/api/route" && request.method === "POST") {
          const payload = await request.json().catch(() => ({}));
          return json(await buildRoute(env, payload, state));
        }
        // GET equivalent of /api/route — shareable, curl-friendly, documented in
        // Swagger. `from`/`to` accept a bare Spotify id, a spotify:track: uri, or
        // an open.spotify.com URL; the rest mirror the permalink query params.
        if (url.pathname === "/api/path" && request.method === "GET") {
          const from = url.searchParams.get("from");
          const to = url.searchParams.get("to");
          if (!from || !to) {
            throw new ApiError(
              400,
              "from and to query params are required (Spotify track id, uri, or URL)",
            );
          }
          const payload = {
            start_uri: `spotify:track:${spotifyIdOf(from)}`,
            end_uri: `spotify:track:${spotifyIdOf(to)}`,
            length: url.searchParams.get("len") ?? undefined,
            context: url.searchParams.get("ctx") ?? undefined,
            shuffle: url.searchParams.get("shuf") ?? undefined,
          };
          return json(await buildRoute(env, payload, state));
        }
        if (url.pathname === "/api/embed" && request.method === "POST") {
          const payload = await request.json().catch(() => ({}));
          return json(await buildEmbed(env, payload, state));
        }
        // Drift: explore by musical intent — an explicit blend of genres tuned by
        // familiarity / focus / count → a coherent constellation of corpus tracks.
        if (url.pathname === "/api/drift" && request.method === "POST") {
          const payload = await request.json().catch(() => ({}));
          return json(await buildDrift(env, payload, state));
        }
        // The Drift tuning catalog: every genre + track count (cached per isolate).
        if (url.pathname === "/api/genres" && request.method === "GET") {
          return json(driftCatalog(state));
        }
        // BYOP: re-order a user's own playlist into a journey (same response
        // shape as /api/route, plus a `source` block).
        if (url.pathname === "/api/arrange" && request.method === "POST") {
          const payload = await request.json().catch(() => ({}));
          return json(await buildArrange(env, payload, state));
        }
        // BYOP: read a PUBLIC playlist's tracks by url/id (client-credentials).
        if (url.pathname === "/api/playlist" && request.method === "GET") {
          const src = url.searchParams.get("url") ?? url.searchParams.get("id");
          if (!src) throw new ApiError(400, "url or id query param is required");
          return json(await fetchPublicPlaylist(env, src));
        }
        return json({ error: "not found" }, 404);
      } catch (e: any) {
        const status = e instanceof ApiError ? e.status : 500;
        return json({ error: String(e?.message ?? e) }, status);
      }
    }

    // Friendly top-level aliases for the API docs — people reach for /docs, not
    // /api/docs. The Swagger UI still loads its spec from /api/openapi.json.
    if (url.pathname === "/docs") return htmlResponse(SWAGGER_HTML);
    if (url.pathname === "/openapi.json") return json(openApiSpec(url.origin));

    // static assets (SPA)
    return env.ASSETS.fetch(request);
  },
};
