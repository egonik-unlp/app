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
  info,
  find_row,
  vec_at,
  meta_at,
  route as wasmRoute,
  project_and_snap,
  search as corpusSearch,
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
  const r = await fetch(SPOTIFY_TOKEN_URL, {
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
  const r = await fetch(url, { headers: { Authorization: `Bearer ${await spotifyTokenGet(env)}` } });
  if (!r.ok) throw new ApiError(502, `Spotify request failed (${r.status})`);
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

async function reccobeatsFeatures(sid: string): Promise<Record<string, any>> {
  try {
    const r1 = await fetch(`${RECCOBEATS_API}/track?ids=${sid}`, { headers: UA });
    if (!r1.ok) return {};
    const c1 = ((await r1.json()) as any).content ?? [];
    if (!c1.length) return {};
    const r2 = await fetch(`${RECCOBEATS_API}/audio-features?ids=${c1[0].id}`, { headers: UA });
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
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });

    if (url.pathname.startsWith("/api/")) {
      try {
        const state = await ensureReady(env);
        if (url.pathname === "/api/health") {
          return json({ ready: true, tracks: state.tracks, cold_start: state.hasProjector });
        }
        if (url.pathname === "/api/search" && request.method === "GET") {
          const q = url.searchParams.get("q") ?? "";
          if (env.SPOTIFY_CLIENT_ID && env.SPOTIFY_CLIENT_SECRET) {
            return json(await searchSpotify(env, q));
          }
          return json(JSON.parse(corpusSearch(q, 12))); // fallback when no Spotify creds
        }
        if (url.pathname === "/api/route" && request.method === "POST") {
          const payload = await request.json().catch(() => ({}));
          return json(await buildRoute(env, payload, state.hasProjector));
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
