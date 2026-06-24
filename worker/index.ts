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
  set_layout,
  info,
  find_row,
  vec_at,
  meta_at,
  route as wasmRoute,
  project_and_snap,
  search as corpusSearch,
  sample_field,
  embed_track,
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
let ready: Promise<{ hasProjector: boolean; tracks: number }> | null = null;
let spotifyToken: { value: string; exp: number } | null = null;

async function ensureReady(env: Env) {
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
      // Baked global 3D layout — shared coordinate space for splash + route.
      const layout = await assetBytes(env, "/data/layout.bin");
      if (layout) {
        try {
          set_layout(layout);
        } catch (e) {
          console.warn("layout load failed:", e);
        }
      }
      const inf = JSON.parse(info());
      return { hasProjector, tracks: inf.n as number };
    })();
  }
  return ready;
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
}

async function resolveEndpoint(env: Env, uri: string, hasProjector: boolean): Promise<Resolved> {
  const row = find_row(uri);
  if (row >= 0) {
    const m = JSON.parse(meta_at(row));
    return { anchorRow: row, vec: vec_at(row), display: m, exact: true, snappedId: null };
  }
  if (!hasProjector) {
    throw new ApiError(
      422,
      "This track isn't in the corpus and cold-start embedding is unavailable. Pick an in-corpus track (badge) or rebuild the snapshot with the projector.",
    );
  }
  const cold = await fetchColdMeta(env, uri);
  const emb = await embedText(env, contentDoc(cold));
  const r = JSON.parse(project_and_snap(emb, JSON.stringify(cold)));
  const anchorMeta = JSON.parse(meta_at(r.snap_row));
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
  };
}

function requestedNode(r: Resolved, pos: number[]): any {
  const uri = r.display.uri ?? "";
  return {
    id: uri,
    uri,
    name: r.display.name,
    artist: r.display.artist,
    album: r.display.album ?? null,
    genre: r.display.genre ?? "unknown",
    spotify_url: uri ? `https://open.spotify.com/track/${spotifyIdOf(uri)}` : null,
    fit: null,
    position: pos,
    kind: "requested",
    snapped_to: r.exact ? null : r.snappedId,
    snapped_label: r.exact ? null : r.anchorName ?? null,
  };
}

async function buildRoute(env: Env, payload: any, hasProjector: boolean) {
  const startUri = payload.start_uri;
  const endUri = payload.end_uri;
  if (!startUri || !endUri) throw new ApiError(400, "start_uri and end_uri are required");

  const [start, end] = await Promise.all([
    resolveEndpoint(env, startUri, hasProjector),
    resolveEndpoint(env, endUri, hasProjector),
  ]);

  const length = Math.max(4, Math.min(50, parseInt(payload.length, 10) || 14));
  const ctx = payload.context ?? "now";
  const tod = ctx === "any" ? "any" : ctx === "now" ? todBucket(new Date().getUTCHours()) : ctx;
  const shuffle = payload.shuffle ?? "any";

  let routeJson: any;
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

  const edges = [...routeJson.edges];
  if (!start.exact) edges.push({ from: start.display.uri, to: start.snappedId, kind: "snap" });
  if (!end.exact) edges.push({ from: end.snappedId, to: end.display.uri, kind: "snap" });

  return {
    context: routeJson.context,
    requested_start: requestedNode(start, routeJson.req_start_pos),
    requested_end: requestedNode(end, routeJson.req_end_pos),
    path: routeJson.path,
    cloud: routeJson.cloud,
    edges,
  };
}

// ---- /api/embed ------------------------------------------------------------
// Single-track explorer: resolve one track (corpus or cold-start) and return it
// plus its neighborhood in the shared layout space.
async function buildEmbed(env: Env, payload: any, hasProjector: boolean) {
  const uri = payload.uri;
  if (!uri) throw new ApiError(400, "uri is required");
  const resolved = await resolveEndpoint(env, uri, hasProjector);
  const emb = JSON.parse(embed_track(resolved.anchorRow, 40));
  return { track: requestedNode(resolved, emb.pos), cloud: emb.cloud };
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

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });

    if (url.pathname.startsWith("/api/")) {
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
          return json(await buildRoute(env, payload, state.hasProjector));
        }
        if (url.pathname === "/api/embed" && request.method === "POST") {
          const payload = await request.json().catch(() => ({}));
          return json(await buildEmbed(env, payload, state.hasProjector));
        }
        return json({ error: "not found" }, 404);
      } catch (e: any) {
        const status = e instanceof ApiError ? e.status : 500;
        return json({ error: String(e?.message ?? e) }, status);
      }
    }

    // static assets (SPA)
    return env.ASSETS.fetch(request);
  },
};
