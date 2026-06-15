import React, { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Billboard, Line, OrbitControls, Stars, Text } from '@react-three/drei'
import {
  ArrowRight,
  Check,
  Compass,
  ExternalLink,
  Link2,
  ListMusic,
  Loader2,
  MapPin,
  Pause,
  Play,
  Search,
  Shuffle,
  Sparkles,
  X,
} from 'lucide-react'
import * as THREE from 'three'
import './styles.css'

// ---------------------------------------------------------------------------
// API contract (unchanged — served by the Worker / WASM core)
// ---------------------------------------------------------------------------
type Candidate = {
  spotify_id: string
  uri: string
  name: string
  artist: string
  album: string | null
  art: string | null
  in_corpus?: boolean
}

type WhyHop = {
  dist: number
  fit: number
  genre_jump: boolean
  prev_genre: string
  transition: number | null
  context: number | null
}

type TrackNode = {
  id: string
  uri: string
  name: string
  artist: string
  album: string | null
  genre: string
  spotify_url: string | null
  fit: number | null
  position: [number, number, number]
  kind: 'path' | 'requested' | 'cloud' | 'sample'
  snapped_to?: string | null
  snapped_label?: string | null
  why?: WhyHop | null
}

type Edge = { from: string; to: string; kind: 'path' | 'snap' }

type RouteResponse = {
  context: string | null
  requested_start: TrackNode
  requested_end: TrackNode
  path: TrackNode[]
  cloud: TrackNode[]
  edges: Edge[]
}

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    let message = res.statusText
    try {
      const body = (await res.json()) as { error?: string }
      message = body.error || message
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, message)
  }
  return res.json() as Promise<T>
}

const post = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
})

// bare Spotify track id from a uri/url (for compact shareable permalinks)
const idOf = (uri: string) => uri.split('?')[0].replace(/.*[:/]/, '')

// ---------------------------------------------------------------------------
// Spotify user auth — Authorization Code + PKCE, run entirely in the browser.
// Lets a listener save the journey to their own account. No client secret and
// no server session: the client id is public (served by /api/config), PKCE
// proves the exchange, and Spotify's token endpoint is CORS-enabled for public
// clients. The app's client-credentials search auth (in the Worker) is separate.
// ---------------------------------------------------------------------------
const SPOTIFY_AUTH_URL = 'https://accounts.spotify.com/authorize'
const SPOTIFY_TOKEN_URL = 'https://accounts.spotify.com/api/token'
const SPOTIFY_SCOPE = 'playlist-modify-public playlist-modify-private'
const PKCE_VERIFIER_KEY = 'pf.pkce_verifier'
const TOKEN_KEY = 'pf.spotify_token'
// Persist the traced route across the OAuth redirect (which reloads the page),
// plus a flag to auto-resume the save the user clicked before signing in.
const ROUTE_KEY = 'pf.route'
const PENDING_SAVE_KEY = 'pf.pending_save'

const redirectUri = () => `${window.location.origin}/`

const base64url = (bytes: ArrayBuffer): string =>
  btoa(String.fromCharCode(...new Uint8Array(bytes)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '')

function randomVerifier(): string {
  const bytes = new Uint8Array(64)
  crypto.getRandomValues(bytes)
  return base64url(bytes.buffer)
}

async function challengeFrom(verifier: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier))
  return base64url(digest)
}

type StoredToken = { value: string; exp: number }

function readToken(): string | null {
  try {
    const raw = sessionStorage.getItem(TOKEN_KEY)
    if (!raw) return null
    const t = JSON.parse(raw) as StoredToken
    return t.exp > Date.now() / 1000 + 30 ? t.value : null
  } catch {
    return null
  }
}

function storeToken(value: string, expiresIn: number) {
  const t: StoredToken = { value, exp: Date.now() / 1000 + expiresIn }
  sessionStorage.setItem(TOKEN_KEY, JSON.stringify(t))
}

async function beginSpotifyLogin(clientId: string) {
  const verifier = randomVerifier()
  sessionStorage.setItem(PKCE_VERIFIER_KEY, verifier)
  const params = new URLSearchParams({
    client_id: clientId,
    response_type: 'code',
    redirect_uri: redirectUri(),
    scope: SPOTIFY_SCOPE,
    code_challenge_method: 'S256',
    code_challenge: await challengeFrom(verifier),
  })
  window.location.assign(`${SPOTIFY_AUTH_URL}?${params}`)
}

// Exchange the ?code= we were redirected back with. Returns the access token,
// or null if there's no pending code. Throws on a real exchange failure.
async function completeSpotifyLogin(clientId: string): Promise<string | null> {
  const url = new URL(window.location.href)
  const code = url.searchParams.get('code')
  const verifier = sessionStorage.getItem(PKCE_VERIFIER_KEY)
  if (!code || !verifier) return null
  sessionStorage.removeItem(PKCE_VERIFIER_KEY)
  // strip ?code/?state so a refresh doesn't re-trigger the exchange
  url.searchParams.delete('code')
  url.searchParams.delete('state')
  window.history.replaceState({}, '', url.pathname + url.search)
  const res = await fetch(SPOTIFY_TOKEN_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: clientId,
      grant_type: 'authorization_code',
      code,
      redirect_uri: redirectUri(),
      code_verifier: verifier,
    }),
  })
  if (!res.ok) throw new ApiError(res.status, 'Spotify sign-in failed — please try again.')
  const data = (await res.json()) as { access_token: string; expires_in?: number }
  storeToken(data.access_token, data.expires_in ?? 3600)
  return data.access_token
}

async function spotifyApi<T>(token: string, path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`https://api.spotify.com/v1${path}`, {
    ...init,
    headers: { Authorization: `Bearer ${token}`, 'content-type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    let message = `Spotify request failed (${res.status})`
    try {
      const body = (await res.json()) as { error?: { message?: string } }
      if (body.error?.message) message = body.error.message
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, message)
  }
  return (res.status === 201 || res.status === 200 ? res.json() : ({} as any)) as Promise<T>
}

// Create a playlist on the signed-in account and fill it with the journey, in
// order. Returns the public Spotify URL of the new playlist.
async function createJourneyPlaylist(
  token: string,
  name: string,
  description: string,
  uris: string[],
): Promise<string> {
  const me = await spotifyApi<{ id: string }>(token, '/me')
  const playlist = await spotifyApi<{ id: string; external_urls: { spotify: string } }>(
    token,
    `/users/${me.id}/playlists`,
    post({ name, description, public: false }),
  )
  // /tracks accepts at most 100 uris per call (paths cap at 50, so usually one)
  for (let i = 0; i < uris.length; i += 100) {
    await spotifyApi(token, `/playlists/${playlist.id}/tracks`, post({ uris: uris.slice(i, i + 100) }))
  }
  return playlist.external_urls.spotify
}

// Loads the public client id, completes any pending OAuth redirect, and exposes
// the current user token. clientId === null means the deploy has no Spotify
// creds, so the UI hides the playlist button.
function useSpotifyAuth() {
  const [clientId, setClientId] = useState<string | null>(null)
  const [token, setToken] = useState<string | null>(() => readToken())

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const cfg = await request<{ spotify_client_id: string | null }>('/api/config')
        if (cancelled) return
        setClientId(cfg.spotify_client_id)
        if (cfg.spotify_client_id) {
          const t = await completeSpotifyLogin(cfg.spotify_client_id)
          if (!cancelled && t) setToken(t)
        }
      } catch {
        /* config/login failures just leave the button signed-out */
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const login = () => {
    if (clientId) void beginSpotifyLogin(clientId)
  }
  return { clientId, token, login }
}

const prefersReducedMotion =
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

// ---------------------------------------------------------------------------
// Genre → color. Curated ramp varying hue AND lightness so clusters stay
// distinguishable for color-vision deficiency; color is always backed by labels.
// ---------------------------------------------------------------------------
const GENRE_PALETTE = [
  '#7cc7ff', '#a78bfa', '#f472b6', '#fb923c', '#facc15', '#4ade80',
  '#34d399', '#22d3ee', '#818cf8', '#e879f9', '#fb7185', '#fcd34d',
  '#86efac', '#5eead4',
]
function genreColor(genre: string): string {
  let hash = 0
  for (let i = 0; i < genre.length; i += 1) hash = (hash * 31 + genre.charCodeAt(i)) >>> 0
  return GENRE_PALETTE[hash % GENRE_PALETTE.length]
}

// smooth accel/decel for the route fly-in
const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

const SIGNAL = '#ffd27a' // the journey
const COOL = '#7ad7ff' // snap / cold-start links

const TIME_OPTIONS = [
  { value: 'now', label: 'Now' },
  { value: 'morning', label: 'Morning' },
  { value: 'afternoon', label: 'Afternoon' },
  { value: 'evening', label: 'Evening' },
  { value: 'night', label: 'Night' },
  { value: 'any', label: 'Anytime' },
]
const SHUFFLE_OPTIONS = [
  { value: 'any', label: 'Any' },
  { value: 'linear', label: 'Linear' },
  { value: 'shuffle', label: 'Shuffle' },
]

const LOADING_STEPS = [
  'Locating both tracks',
  'Embedding the taste map',
  'Charting the route',
  'Projecting into 3D',
]

// ---------------------------------------------------------------------------
// Shared halo texture for additive glow (built once)
// ---------------------------------------------------------------------------
const haloTexture = (() => {
  if (typeof document === 'undefined') return null
  const size = 64
  const c = document.createElement('canvas')
  c.width = c.height = size
  const ctx = c.getContext('2d')!
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
  g.addColorStop(0, 'rgba(255,255,255,1)')
  g.addColorStop(0.4, 'rgba(255,255,255,0.35)')
  g.addColorStop(1, 'rgba(255,255,255,0)')
  ctx.fillStyle = g
  ctx.fillRect(0, 0, size, size)
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
})()

// ===========================================================================
// Playback engine — shared by the journey strip (procession) and the Inspector
// (single track), over one <audio>. The journey we *play* is the whole thing:
// the picked start song (even when it was cold-start-snapped and isn't a stop
// on the map), every path stop, then the picked end song. Previews cache by
// track uri so the procession and single-track play share results.
// ===========================================================================
function usePlayback(
  route: RouteResponse | null,
  onFocus: (i: number | null) => void,
  onError: (msg: string | null) => void,
) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const playingRef = useRef(false) // procession auto-advancing
  const idxRef = useRef(0) // procession position within playQueue
  const skipTimer = useRef<number | null>(null)
  const playedAnyRef = useRef(false)
  const previews = useRef(new Map<string, string | null>())
  const [playing, setPlaying] = useState(false) // procession sounding
  const [paused, setPaused] = useState(false) // procession paused, resumable
  const [playingId, setPlayingId] = useState<string | null>(null) // track sounding NOW

  const keyOf = (n: TrackNode) => n.uri || n.id

  const playQueue = useMemo(() => {
    const q: { node: TrackNode; focus: number | null }[] = []
    if (!route) return q
    const path = route.path
    const s = route.requested_start
    if (s?.uri && s.uri !== path[0]?.uri) q.push({ node: s, focus: null })
    path.forEach((n, i) => q.push({ node: n, focus: i }))
    const e = route.requested_end
    if (e?.uri && e.uri !== path[path.length - 1]?.uri) q.push({ node: e, focus: null })
    return q
  }, [route])

  const focusOf = (node: TrackNode): number | null => {
    if (!route) return null
    const i = route.path.findIndex((p) => keyOf(p) === keyOf(node))
    return i >= 0 ? i : null
  }

  const fetchPreviewFor = async (n: TrackNode | undefined): Promise<string | null> => {
    if (!n) return null
    const key = keyOf(n)
    if (previews.current.has(key)) return previews.current.get(key)!
    try {
      const r = await request<{ preview_url: string | null }>(
        `/api/preview?artist=${encodeURIComponent(n.artist)}&track=${encodeURIComponent(n.name)}`,
      )
      previews.current.set(key, r.preview_url)
      return r.preview_url
    } catch {
      previews.current.set(key, null)
      return null
    }
  }

  const haltProcession = () => {
    playingRef.current = false
    if (skipTimer.current) window.clearTimeout(skipTimer.current)
  }

  // full stop: nothing sounding, procession cleared
  const stop = (msg?: string) => {
    haltProcession()
    setPlaying(false)
    setPaused(false)
    setPlayingId(null)
    const a = audioRef.current
    if (a) {
      a.pause()
      a.removeAttribute('src')
    }
    onFocus(null)
    if (msg) onError(msg)
  }

  const advance = () => {
    if (!playingRef.current) return
    const next = idxRef.current + 1
    if (next >= playQueue.length) {
      // ended — if not a single clip played, the browser/codec is the problem,
      // so say so rather than ending in silence
      stop(playedAnyRef.current ? undefined : 'Couldn’t play any previews in this browser.')
      return
    }
    idxRef.current = next
    playQueueItem(next)
  }

  // synchronous up to .play() so a cached clip starts inside the click gesture
  // (browsers block play() that happens after an awaited fetch resolves)
  const startClip = (key: string, url: string | null, onBlocked: () => void) => {
    const a = audioRef.current
    if (!url || !a) {
      skipTimer.current = window.setTimeout(advance, 350) // no preview — skip on
      return
    }
    a.src = url
    setPlayingId(key)
    const p = a.play()
    if (p)
      p.then(() => {
        playedAnyRef.current = true
      }).catch((e: any) => {
        if (e?.name === 'NotAllowedError') onBlocked()
        // other rejections (src swapped) handled by ended/error
      })
  }

  const playQueueItem = (i: number) => {
    if (!playingRef.current) return
    const item = playQueue[i]
    if (!item) return stop()
    onFocus(item.focus) // drives strip highlight + 3D tracer (null = off-map endpoint)
    void fetchPreviewFor(playQueue[i + 1]?.node) // prefetch next
    const blocked = () => stop('Your browser blocked autoplay — press Play journey again.')
    const key = keyOf(item.node)
    if (previews.current.has(key)) startClip(key, previews.current.get(key)!, blocked)
    else
      void fetchPreviewFor(item.node).then((url) => {
        if (playingRef.current && idxRef.current === i) startClip(key, url, blocked)
      })
  }

  const togglePlay = () => {
    onError(null)
    if (playing) {
      // pause the procession (resumable at its current stop)
      haltProcession()
      setPlaying(false)
      setPaused(true)
      setPlayingId(null)
      audioRef.current?.pause()
      return
    }
    if (paused) {
      // resume the procession where it left off
      playingRef.current = true
      setPlaying(true)
      setPaused(false)
      playQueueItem(idxRef.current)
      return
    }
    // fresh start from the top
    playingRef.current = true
    playedAnyRef.current = false
    idxRef.current = 0
    setPlaying(true)
    setPaused(false)
    playQueueItem(0)
  }

  // Play (or stop) a single track on demand. Does NOT advance the journey and
  // preserves the procession's resume point — a side-quest, not a hijack.
  const toggleTrack = (node: TrackNode) => {
    const a = audioRef.current
    const key = keyOf(node)
    onError(null)
    if (playingId === key && a) {
      // this track is the one sounding → stop it; if the procession was running,
      // leave it paused & resumable at its own stop
      haltProcession()
      a.pause()
      setPlayingId(null)
      if (playing) {
        setPlaying(false)
        setPaused(true)
      }
      return
    }
    // a different track: silence the procession but keep it resumable
    if (playing) {
      haltProcession()
      setPlaying(false)
      setPaused(true)
    }
    onFocus(focusOf(node))
    const go = (url: string | null) => {
      if (!url || !a) {
        onError('No preview available for this track.')
        return
      }
      a.src = url
      setPlayingId(key)
      const p = a.play()
      if (p)
        p.then(() => {
          playedAnyRef.current = true
        }).catch((e: any) => {
          if (e?.name === 'NotAllowedError') onError('Your browser blocked autoplay — tap play again.')
        })
    }
    if (previews.current.has(key)) go(previews.current.get(key)!)
    else void fetchPreviewFor(node).then(go)
  }

  const onEnded = () => {
    if (playingRef.current) advance()
    else setPlayingId(null) // single clip finished
  }
  const onAudioError = () => {
    if (playingRef.current && audioRef.current?.getAttribute('src')) advance()
    else setPlayingId(null)
  }

  // reset + warm previews whenever a new route arrives, so plays start
  // synchronously (reliably) instead of after an awaited fetch
  useEffect(() => {
    previews.current.clear()
    stop()
    let cancelled = false
    const nodes = playQueue.map((q) => q.node)
    const run = async () => {
      let i = 0
      const worker = async () => {
        while (!cancelled && i < nodes.length) await fetchPreviewFor(nodes[i++])
      }
      await Promise.all(Array.from({ length: 4 }, worker)) // 4-way throttle
    }
    void run()
    return () => {
      cancelled = true
      stop()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [route])

  return { playing, paused, playingId, keyOf, togglePlay, toggleTrack, audioRef, onEnded, onAudioError }
}

// ===========================================================================
// App
// ===========================================================================
function App() {
  const [start, setStart] = useState<Candidate | null>(null)
  const [end, setEnd] = useState<Candidate | null>(null)
  const [length, setLength] = useState(14)
  const [context, setContext] = useState('now')
  const [shuffle, setShuffle] = useState('any')

  const [route, setRoute] = useState<RouteResponse | null>(() => {
    try {
      // a ?from=&to= permalink takes over — trace it fresh instead of restoring
      const p = new URLSearchParams(window.location.search)
      if (p.get('from') && p.get('to')) return null
      // Only restore the saved route when we're returning from the Spotify OAuth
      // redirect (?code=...) — that round-trip reloads the page mid-save and must
      // keep the route alive. A plain reload (F5) or fresh visit should land on
      // the splash screen, so drop the stale route instead of rehydrating it.
      if (!p.get('code')) {
        sessionStorage.removeItem(ROUTE_KEY)
        return null
      }
      const raw = sessionStorage.getItem(ROUTE_KEY)
      return raw ? (JSON.parse(raw) as RouteResponse) : null
    } catch {
      return null
    }
  })
  const [status, setStatus] = useState<'idle' | 'routing'>('idle')
  const [step, setStep] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [focus, setFocus] = useState<number | null>(null)
  const [inspect, setInspect] = useState<TrackNode | null>(null)
  const auth = useSpotifyAuth()
  const playback = usePlayback(route, setFocus, setError)

  // keep the traced route alive across the Spotify OAuth redirect / reload
  useEffect(() => {
    if (route) sessionStorage.setItem(ROUTE_KEY, JSON.stringify(route))
  }, [route])

  useEffect(() => {
    if (status !== 'routing') return
    setStep(0)
    const t = window.setInterval(() => setStep((s) => Math.min(LOADING_STEPS.length - 1, s + 1)), 650)
    return () => window.clearInterval(t)
  }, [status])

  const candidateFromNode = (n: TrackNode): Candidate => ({
    spotify_id: idOf(n.uri),
    uri: n.uri,
    name: n.name,
    artist: n.artist,
    album: n.album,
    art: null,
  })

  const runRoute = async (
    startUri: string,
    endUri: string,
    len: number,
    ctx: string,
    shuf: string,
    populatePickers = false,
  ) => {
    if (status === 'routing') return
    setStatus('routing')
    setError(null)
    setFocus(null)
    setInspect(null)
    try {
      const data = await request<RouteResponse>(
        '/api/route',
        post({ start_uri: startUri, end_uri: endUri, length: len, context: ctx, shuffle: shuf }),
      )
      setRoute(data)
      // a permalink has no picked Candidates yet — fill the From/To fields from
      // the resolved endpoints so the controls reflect the shared journey
      if (populatePickers) {
        setStart(candidateFromNode(data.requested_start))
        setEnd(candidateFromNode(data.requested_end))
      }
    } catch (e) {
      const cpuLimited = e instanceof ApiError && (e.status === 503 || e.status === 524)
      setError(
        cpuLimited
          ? 'This pair was too far apart to route within the free compute budget. Try tracks closer in style, or fewer stops — and you can trace again right away.'
          : e instanceof ApiError
            ? e.message
            : e instanceof Error
              ? e.message
              : String(e),
      )
    } finally {
      setStatus('idle')
    }
  }

  const trace = () => {
    if (start && end) void runRoute(start.uri, end.uri, length, context, shuffle)
  }

  // on first load, trace a shared ?from=&to= permalink
  useEffect(() => {
    const p = new URLSearchParams(window.location.search)
    const from = p.get('from')
    const to = p.get('to')
    if (!from || !to) return
    const len = Number(p.get('len')) || 14
    const ctx = p.get('ctx') || 'now'
    const shuf = p.get('shuf') || 'any'
    setLength(len)
    setContext(ctx)
    setShuffle(shuf)
    void runRoute(`spotify:track:${from}`, `spotify:track:${to}`, len, ctx, shuf, true)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // shareable permalink for the current pick + options
  const shareUrl = useMemo(() => {
    if (!start || !end) return null
    const u = new URL(window.location.origin + '/')
    u.searchParams.set('from', start.spotify_id)
    u.searchParams.set('to', end.spotify_id)
    u.searchParams.set('len', String(length))
    u.searchParams.set('ctx', context)
    u.searchParams.set('shuf', shuffle)
    return u.toString()
  }, [start, end, length, context, shuffle])

  const snapped =
    route && (route.requested_start.snapped_to || route.requested_end.snapped_to)

  return (
    <main className="app">
      <Scene route={route} idle={!route} focus={focus} onFocus={setFocus} onInspect={setInspect} />

      <div className="topbar">
        <div className="brand">
          <Compass size={18} aria-hidden />
          <span>Pathfinder</span>
        </div>
        <div className="controls" role="search">
          <TrackPicker label="From" selected={start} onSelect={setStart} />
          <ArrowRight className="flow" size={16} aria-hidden />
          <TrackPicker label="To" selected={end} onSelect={setEnd} />
          <Options
            length={length}
            setLength={setLength}
            context={context}
            setContext={setContext}
            shuffle={shuffle}
            setShuffle={setShuffle}
          />
          <button
            className="trace"
            disabled={!start || !end || status === 'routing'}
            onClick={() => void trace()}
          >
            {status === 'routing' ? (
              <Loader2 size={16} className="spin" aria-hidden />
            ) : (
              <Sparkles size={16} aria-hidden />
            )}
            <span>{status === 'routing' ? 'Tracing' : 'Trace route'}</span>
          </button>
        </div>
      </div>

      {!route && status === 'idle' && <Intro ready={!!start && !!end} />}
      {status === 'routing' && <LoadingPanel step={step} />}
      {error && (
        <div className="toast" role="alert">
          <strong>Couldn’t trace that route</strong>
          <span>{error}</span>
          <button aria-label="Dismiss" onClick={() => setError(null)}>
            <X size={14} />
          </button>
        </div>
      )}

      {route && (
        <>
          <JourneyStrip
            route={route}
            focus={focus}
            onFocus={setFocus}
            onInspect={setInspect}
            snapped={!!snapped}
            token={auth.token}
            canSave={!!auth.clientId}
            onLogin={auth.login}
            onError={setError}
            playing={playback.playing}
            paused={playback.paused}
            onTogglePlay={playback.togglePlay}
            playingId={playback.playingId}
            shareUrl={shareUrl}
          />
          <Legend />
          {inspect && (
            <Inspector
              node={inspect}
              onClose={() => setInspect(null)}
              isPlaying={playback.playingId === playback.keyOf(inspect)}
              onTogglePlay={() => playback.toggleTrack(inspect)}
            />
          )}
        </>
      )}
      <audio
        ref={playback.audioRef}
        onEnded={playback.onEnded}
        onError={playback.onAudioError}
        preload="none"
        hidden
      />
    </main>
  )
}

// ===========================================================================
// Search picker
// ===========================================================================
function TrackPicker({
  label,
  selected,
  onSelect,
}: {
  label: string
  selected: Candidate | null
  onSelect: (c: Candidate | null) => void
}) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<Candidate[]>([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [active, setActive] = useState(0)
  const seq = useRef(0)

  useEffect(() => {
    const q = query.trim()
    if (!q || selected) {
      setResults([])
      setError(null)
      return
    }
    const id = ++seq.current
    const ac = new AbortController()
    const timer = window.setTimeout(() => {
      setLoading(true)
      request<Candidate[]>(`/api/search?q=${encodeURIComponent(q)}`, { signal: ac.signal })
        .then((hits) => {
          if (id !== seq.current) return
          setResults(hits)
          setActive(0)
          setError(null)
          setOpen(true)
        })
        .catch((e) => {
          if (id !== seq.current || e?.name === 'AbortError') return
          setError(e instanceof Error ? e.message : String(e))
          setResults([])
          setOpen(true)
        })
        .finally(() => id === seq.current && setLoading(false))
    }, 240)
    return () => {
      window.clearTimeout(timer)
      ac.abort()
    }
  }, [query, selected])

  const choose = (item: Candidate) => {
    onSelect(item)
    setQuery('')
    setOpen(false)
  }
  const clear = () => {
    onSelect(null)
    setQuery('')
    setOpen(false)
  }
  const onKey = (e: React.KeyboardEvent) => {
    if (!open || !results.length) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActive((a) => Math.min(results.length - 1, a + 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((a) => Math.max(0, a - 1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      choose(results[active])
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  return (
    <div className="picker" onBlur={(e) => !e.currentTarget.contains(e.relatedTarget) && setOpen(false)}>
      <div className={`field${selected ? ' filled' : ''}`}>
        <span className="fieldLabel">{label}</span>
        <Search size={14} className="fieldIcon" aria-hidden />
        <input
          value={selected ? `${selected.name} — ${selected.artist}` : query}
          placeholder="Search a song…"
          aria-label={`${label} track`}
          onChange={(e) => {
            onSelect(null)
            setQuery(e.target.value)
          }}
          onFocus={() => results.length > 0 && setOpen(true)}
          onKeyDown={onKey}
          autoComplete="off"
          spellCheck={false}
        />
        {selected?.in_corpus && <span className="dot" title="In the listening corpus" />}
        {loading && <Loader2 size={14} className="spin" aria-hidden />}
        {selected && (
          <button className="iconbtn" aria-label={`Clear ${label}`} onClick={clear}>
            <X size={13} aria-hidden />
          </button>
        )}
      </div>
      {open && !selected && (
        <div className="menu" role="listbox">
          {error ? (
            <p className="menuMsg">{error}</p>
          ) : results.length === 0 ? (
            <p className="menuMsg">{loading ? 'Searching…' : 'No matches yet'}</p>
          ) : (
            results.map((item, i) => (
              <button
                key={item.spotify_id}
                role="option"
                aria-selected={i === active}
                className={`menuItem${i === active ? ' active' : ''}`}
                onMouseEnter={() => setActive(i)}
                onClick={() => choose(item)}
              >
                {item.art ? <img src={item.art} alt="" /> : <span className="blankArt" />}
                <span className="menuText">
                  <b>{item.name}</b>
                  <small>
                    {item.artist}
                    {item.album ? ` · ${item.album}` : ''}
                  </small>
                </span>
                {item.in_corpus && <span className="chip">in map</span>}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  )
}

function Options({
  length,
  setLength,
  context,
  setContext,
  shuffle,
  setShuffle,
}: {
  length: number
  setLength: (n: number) => void
  context: string
  setContext: (s: string) => void
  shuffle: string
  setShuffle: (s: string) => void
}) {
  const [open, setOpen] = useState(false)
  return (
    <div className="options" onBlur={(e) => !e.currentTarget.contains(e.relatedTarget) && setOpen(false)}>
      <button className="optbtn" aria-label="Route options" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <Shuffle size={14} aria-hidden />
        <span>{length} stops</span>
      </button>
      {open && (
        <div className="popover">
          <label className="optRow">
            <span>Stops</span>
            <input
              type="range"
              min={4}
              max={30}
              value={length}
              onChange={(e) => setLength(Number(e.target.value))}
            />
            <b>{length}</b>
          </label>
          <div className="optRow segmented">
            <span>Time</span>
            <div className="seg">
              {TIME_OPTIONS.map((o) => (
                <button key={o.value} className={context === o.value ? 'on' : ''} onClick={() => setContext(o.value)}>
                  {o.label}
                </button>
              ))}
            </div>
          </div>
          <div className="optRow segmented">
            <span>Order</span>
            <div className="seg">
              {SHUFFLE_OPTIONS.map((o) => (
                <button key={o.value} className={shuffle === o.value ? 'on' : ''} onClick={() => setShuffle(o.value)}>
                  {o.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ===========================================================================
// 2D overlays
// ===========================================================================
function Intro({ ready }: { ready: boolean }) {
  return (
    <div className="intro">
      <h1>The space between two songs</h1>
      <p>
        Every track this engine knows lives somewhere on a map of listening taste. Pick a
        start and a destination and watch the route thread through the cloud — staying close on
        the map, leaning toward songs you’d actually play, and easing between genres rather than
        lurching.
      </p>
      <p className="cue">{ready ? 'Press Trace route to begin.' : 'Choose a From and a To above to begin.'}</p>
    </div>
  )
}

function LoadingPanel({ step }: { step: number }) {
  return (
    <div className="loading">
      {LOADING_STEPS.map((s, i) => (
        <div key={s} className={`lstep${i < step ? ' done' : i === step ? ' now' : ''}`}>
          <span className="ldot" />
          {s}
        </div>
      ))}
    </div>
  )
}

function Legend() {
  const [open, setOpen] = useState(false)
  return (
    <div className={`legend${open ? ' open' : ''}`}>
      <button className="legendToggle" onClick={() => setOpen((o) => !o)}>
        <Compass size={14} aria-hidden /> What am I seeing?
      </button>
      {open && (
        <dl>
          <div>
            <dt><span className="sw" style={{ background: SIGNAL }} /> The route</dt>
            <dd>The chosen path, start to destination. The comet shows its direction.</dd>
          </div>
          <div>
            <dt><span className="sw cloud" /> The cloud</dt>
            <dd>Nearby tracks on the map. Color groups them by genre.</dd>
          </div>
          <div>
            <dt><span className="sw ring" /> From / To</dt>
            <dd>Your two tracks. Depth = distance on the taste map.</dd>
          </div>
          <div>
            <dt><span className="sw" style={{ background: COOL }} /> Snap link</dt>
            <dd>An off-map track pulled to its nearest neighbor in the corpus.</dd>
          </div>
        </dl>
      )}
    </div>
  )
}

function FitMeter({ fit }: { fit: number | null }) {
  if (fit == null) return null
  return (
    <span className="fit" title={`fit ${fit.toFixed(2)}`}>
      <span style={{ width: `${Math.round(fit * 100)}%` }} />
    </span>
  )
}

function JourneyStrip({
  route,
  focus,
  onFocus,
  onInspect,
  snapped,
  token,
  canSave,
  onLogin,
  onError,
  playing,
  paused,
  onTogglePlay,
  playingId,
  shareUrl,
}: {
  route: RouteResponse
  focus: number | null
  onFocus: (i: number | null) => void
  onInspect: (n: TrackNode | null) => void
  snapped: boolean
  token: string | null
  canSave: boolean
  onLogin: () => void
  onError: (msg: string | null) => void
  playing: boolean
  paused: boolean
  onTogglePlay: () => void
  playingId: string | null
  shareUrl: string | null
}) {
  // Playback lives in App's usePlayback hook (shared with the Inspector); this
  // component just renders its controls via the playing/paused/playingId props.

  // ---- share permalink -----------------------------------------------------
  const [copied, setCopied] = useState(false)
  const copyLink = async () => {
    if (!shareUrl) return
    try {
      await navigator.clipboard.writeText(shareUrl)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    } catch {
      onError(`Couldn’t copy automatically — here’s the link: ${shareUrl}`)
    }
  }

  // ---- save as playlist ----------------------------------------------------
  const [saving, setSaving] = useState(false)
  const [playlistUrl, setPlaylistUrl] = useState<string | null>(null)

  // clear the saved-playlist link whenever a new route arrives
  useEffect(() => {
    setPlaylistUrl(null)
  }, [route])

  const save = async () => {
    if (!token) {
      // remember intent, sign in (redirects away), resume on return
      sessionStorage.setItem(PENDING_SAVE_KEY, '1')
      onLogin()
      return
    }
    setSaving(true)
    onError(null)
    try {
      const path = route.path
      const uris: string[] = []
      const startUri = route.requested_start.uri
      if (startUri && startUri !== path[0]?.uri) uris.push(startUri)
      uris.push(...path.map((n) => n.uri))
      const endUri = route.requested_end.uri
      if (endUri && endUri !== path[path.length - 1]?.uri) uris.push(endUri)
      const clean = uris.filter((u) => /^spotify:track:/.test(u))
      const name = `${route.requested_start.name} → ${route.requested_end.name}`
      const description =
        `A Pathfinder journey of ${path.length} tracks.${route.context ? ` Context: ${route.context}.` : ''}`.trim()
      const url = await createJourneyPlaylist(token, name, description, clean)
      setPlaylistUrl(url)
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        // token expired mid-session — re-auth and resume
        sessionStorage.setItem(PENDING_SAVE_KEY, '1')
        onLogin()
        return
      }
      onError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  // auto-resume the save the user requested before being redirected to sign in
  useEffect(() => {
    if (token && sessionStorage.getItem(PENDING_SAVE_KEY)) {
      sessionStorage.removeItem(PENDING_SAVE_KEY)
      void save()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token])

  return (
    <div className="strip">
      <div className="stripHead">
        <span className="count">{route.path.length} stops</span>
        {route.context && <span className="tag">{route.context}</span>}
        {snapped && (
          <span className="tag cool">
            <MapPin size={11} aria-hidden /> snapped to nearby tracks
          </span>
        )}
        <div className="stripActions">
          <button
            className={`action${playing ? ' on' : ''}`}
            onClick={onTogglePlay}
            aria-pressed={playing}
            title="Play 30-second samples along the whole journey"
          >
            {playing ? <Pause size={14} aria-hidden /> : <Play size={14} aria-hidden />}
            <span>{playing ? 'Pause' : paused ? 'Resume' : 'Play journey'}</span>
          </button>
          {shareUrl && (
            <button
              className={`action${copied ? ' saved' : ''}`}
              onClick={() => void copyLink()}
              title="Copy a shareable link to this journey"
            >
              {copied ? <Check size={14} aria-hidden /> : <Link2 size={14} aria-hidden />}
              <span>{copied ? 'Link copied' : 'Copy link'}</span>
            </button>
          )}
          {canSave &&
            (playlistUrl ? (
              <a className="action saved" href={playlistUrl} target="_blank" rel="noreferrer">
                <ExternalLink size={14} aria-hidden />
                <span>Open playlist</span>
              </a>
            ) : (
              <button className="action" onClick={() => void save()} disabled={saving}>
                {saving ? (
                  <Loader2 size={14} className="spin" aria-hidden />
                ) : (
                  <ListMusic size={14} aria-hidden />
                )}
                <span>{saving ? 'Saving…' : token ? 'Save as playlist' : 'Save to Spotify'}</span>
              </button>
            ))}
        </div>
      </div>
      <ol className="stripList" onMouseLeave={() => onFocus(null)}>
        {route.path.map((n, i) => (
          <li key={`${n.id}-${i}`}>
            <button
              className={`stop${focus === i ? ' active' : ''}${
                playingId === (n.uri || n.id) ? ' playing' : ''
              }`}
              onMouseEnter={() => onFocus(i)}
              onFocus={() => onFocus(i)}
              onClick={() => {
                onFocus(i)
                onInspect(n)
              }}
              style={{ '--g': genreColor(n.genre) } as React.CSSProperties}
            >
              <span className="idx">{i + 1}</span>
              <span className="stopText">
                <b>{n.name}</b>
                <small>{n.artist}</small>
              </span>
              <FitMeter fit={n.fit} />
            </button>
          </li>
        ))}
      </ol>
    </div>
  )
}

// ===========================================================================
// 3D scene
// ===========================================================================
function Scene({
  route,
  idle,
  focus,
  onFocus,
  onInspect,
}: {
  route: RouteResponse | null
  idle: boolean
  focus: number | null
  onFocus: (i: number | null) => void
  onInspect: (n: TrackNode | null) => void
}) {
  // The route lives in its own sector (ROUTE_SECTOR) away from the splash cloud.
  // While the camera travels there we keep the splash cloud mounted so it recedes
  // into the fog behind us instead of cutting away — that's the visual continuity.
  // It unmounts only once the camera has arrived.
  const [arrived, setArrived] = useState(false)
  useEffect(() => setArrived(false), [route])

  // The splash cloud is anchored at the origin (centred on the random sample's
  // centroid in the shared global layout). We remember that centroid so a traced
  // route can be placed at its TRUE displacement from it — same space, real
  // direction/distance — and the camera travels there. While it travels we keep
  // the splash mounted so it recedes into the fog instead of cutting away.
  const splashCentroid = useRef<THREE.Vector3 | null>(null)

  return (
    <Canvas camera={{ position: [0, 4, 19], fov: 50 }} dpr={[1, 2]} gl={{ antialias: true }}>
      <color attach="background" args={['#070810']} />
      <fog attach="fog" args={['#070810', 18, 46]} />
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 14, 8]} intensity={1.4} color="#fff0d4" />
      <pointLight position={[-12, -8, -6]} intensity={0.9} color="#5fb8ff" />
      <Suspense fallback={null}>
        {/* Warm the troika text font at load so the first hover label doesn't
            hitch the idle animation while the font parses on the main thread. */}
        <Text position={[0, 0, -9999]} fontSize={0.01} color="#000" fillOpacity={0}>
          .
        </Text>
        <Stars radius={70} depth={40} count={1600} factor={2.6} fade speed={prefersReducedMotion ? 0 : 0.4} />
        {(!route || !arrived) && (
          <IdleField onInspect={onInspect} onCentroid={(c) => (splashCentroid.current = c)} />
        )}
        {route && (
          <RouteField
            route={route}
            focus={focus}
            onFocus={onFocus}
            onInspect={onInspect}
            arrived={arrived}
            onArrive={() => setArrived(true)}
            splashCentroid={splashCentroid.current}
          />
        )}
      </Suspense>
      <OrbitControls
        makeDefault
        enableDamping
        enablePan={false}
        minDistance={7}
        maxDistance={34}
        autoRotate={idle && !prefersReducedMotion}
        autoRotateSpeed={0.3}
      />
    </Canvas>
  )
}

// Fallback route sector used only when the splash centroid isn't known yet (e.g.
// a ?from=&to= permalink that traces before the sample loads). Normally the route
// is placed at its true displacement from the splash centroid in the shared layout.
const ROUTE_SECTOR = new THREE.Vector3(12, 7, -46)

// centroid of a set of nodes in raw (shared-layout) coordinates
function nodesCentroid(nodes: TrackNode[]): THREE.Vector3 {
  const c = new THREE.Vector3()
  nodes.forEach((n) => c.add(new THREE.Vector3(...n.position)))
  return nodes.length ? c.multiplyScalar(1 / nodes.length) : c
}

// Procedural fallback used while the real sample loads (or if it fails) — the
// original splash spiral, so the screen never looks empty and the look is
// preserved.
function proceduralDots() {
  const out: { p: [number, number, number]; c: string; s: number }[] = []
  for (let i = 0; i < 130; i += 1) {
    const t = i * 0.49
    const r = 3 + (i % 19) * 0.22
    out.push({
      p: [Math.cos(t) * r, Math.sin(i * 0.27) * 3.4, Math.sin(t) * r],
      c: GENRE_PALETTE[i % GENRE_PALETTE.length],
      s: i % 13 === 0 ? 0.09 : 0.05,
    })
  }
  return out
}

// Map the sampled subspace's raw PCA coordinates into the splash field's visual
// envelope: recenter on the cloud's centroid, scale uniformly so the spread
// matches the original spiral, then gently flatten the vertical axis for the
// same disc-like silhouette. Uniform horizontal scale preserves the real
// cluster structure (similar tracks stay near each other).
function fitToField(nodes: TrackNode[]): TrackNode[] {
  if (!nodes.length) return nodes
  const c: [number, number, number] = [0, 0, 0]
  for (const n of nodes) for (let k = 0; k < 3; k += 1) c[k] += n.position[k]
  for (let k = 0; k < 3; k += 1) c[k] /= nodes.length
  let maxR = 1e-6
  for (const n of nodes) {
    const dx = n.position[0] - c[0]
    const dy = n.position[1] - c[1]
    const dz = n.position[2] - c[2]
    maxR = Math.max(maxR, Math.hypot(dx, dy, dz))
  }
  const scale = 6.5 / maxR
  return nodes.map((n) => ({
    ...n,
    position: [
      (n.position[0] - c[0]) * scale,
      (n.position[1] - c[1]) * scale * 0.6,
      (n.position[2] - c[2]) * scale,
    ] as [number, number, number],
  }))
}

function IdleField({
  onInspect,
  onCentroid,
}: {
  onInspect: (n: TrackNode | null) => void
  onCentroid: (c: THREE.Vector3) => void
}) {
  const group = useRef<THREE.Group>(null)
  const fallback = useMemo(proceduralDots, [])
  const [tracks, setTracks] = useState<TrackNode[] | null>(null)
  const gl = useThree((s) => s.gl)

  // The cloud rotates, which breaks r3f's raycast-based onClick (it needs the
  // same dot under the cursor at both press and release). Hover events still
  // fire reliably, so we track the dot under the cursor and open the inspector
  // from a plain DOM click on the canvas — no raycast at click time.
  const hoveredRef = useRef<TrackNode | null>(null)
  useEffect(() => {
    const el = gl.domElement
    const onClick = () => {
      if (hoveredRef.current) onInspect(hoveredRef.current)
    }
    el.addEventListener('click', onClick)
    return () => el.removeEventListener('click', onClick)
  }, [gl, onInspect])

  useEffect(() => {
    let alive = true
    request<TrackNode[]>('/api/sample?n=130')
      .then((data) => {
        if (!alive) return
        // remember the sample's real centroid in the shared layout (for placing a
        // future route), then anchor the splash itself at the origin for display.
        if (data.length) onCentroid(nodesCentroid(data))
        setTracks(fitToField(data))
      })
      .catch(() => {
        /* keep the procedural fallback on failure */
      })
    return () => {
      alive = false
    }
  }, [])

  useFrame(({ clock }) => {
    if (group.current && !prefersReducedMotion) group.current.rotation.y = clock.elapsedTime * 0.05
  })

  return (
    <group ref={group}>
      {tracks
        ? tracks.map((n, i) => (
            <GlowDot
              key={n.id}
              position={n.position}
              color={genreColor(n.genre)}
              size={i % 13 === 0 ? 0.09 : 0.05}
              glow={0.5}
              label={{ name: n.name, artist: n.artist }}
              onHover={(h) => {
                if (h) hoveredRef.current = n
                else if (hoveredRef.current === n) hoveredRef.current = null
              }}
            />
          ))
        : fallback.map((d, i) => (
            <GlowDot key={i} position={d.p} color={d.c} size={d.s} glow={0.5} />
          ))}
    </group>
  )
}

function GlowDot({
  position,
  color,
  size,
  glow = 1,
  onPick,
  onHover,
  label,
}: {
  position: [number, number, number]
  color: string
  size: number
  glow?: number
  onPick?: () => void
  onHover?: (hovering: boolean) => void
  label?: { name: string; artist: string }
}) {
  const [hover, setHover] = useState(false)
  return (
    <group position={position}>
      {/* generous invisible hit target for mouse + touch */}
      {(onPick || onHover) && (
        <mesh
          onPointerOver={(e) => {
            e.stopPropagation()
            setHover(true)
            onHover?.(true)
          }}
          onPointerOut={() => {
            setHover(false)
            onHover?.(false)
          }}
          onClick={(e) => {
            e.stopPropagation()
            onPick?.()
          }}
        >
          <sphereGeometry args={[Math.max(size * 3, 0.18), 12, 12]} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
      )}
      <mesh scale={hover ? 1.6 : 1}>
        <sphereGeometry args={[size, 16, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={hover ? 1.4 : 0.8} roughness={0.35} />
      </mesh>
      {label && hover && (
        <Billboard position={[0, Math.max(size * 3, 0.18) + 0.16, 0]}>
          <Text fontSize={0.17} color="#fbf7ee" anchorX="center" anchorY="bottom" maxWidth={3} outlineWidth={0.006} outlineColor="#070810">
            {label.name}
          </Text>
          <Text position={[0, -0.02, 0]} fontSize={0.115} color="#b9c2dd" anchorX="center" anchorY="top" maxWidth={3}>
            {label.artist}
          </Text>
        </Billboard>
      )}
      {haloTexture && (
        <sprite scale={size * 9 * glow}>
          <spriteMaterial
            map={haloTexture}
            color={color}
            blending={THREE.AdditiveBlending}
            transparent
            depthWrite={false}
            opacity={0.5 * glow}
          />
        </sprite>
      )}
    </group>
  )
}

function RouteField({
  route,
  focus,
  onFocus,
  onInspect,
  arrived,
  onArrive,
  splashCentroid,
}: {
  route: RouteResponse
  focus: number | null
  onFocus: (i: number | null) => void
  onInspect: (n: TrackNode | null) => void
  arrived: boolean
  onArrive: () => void
  splashCentroid: THREE.Vector3 | null
}) {
  // Place the route in the SAME space as the splash: take its raw layout
  // centroid, position the cluster at its true displacement from the splash
  // centroid (so direction + distance are real), and scale the cluster's spread
  // to a legible size around that point. `R` is the route with these world
  // positions baked in; `center` is where the cluster sits.
  const { R, center } = useMemo(() => {
    const real = (n: TrackNode) => new THREE.Vector3(...n.position)
    const Cr = nodesCentroid(route.path)
    let radius = 1e-6
    for (const n of [...route.path, ...route.cloud]) radius = Math.max(radius, real(n).distanceTo(Cr))
    const s = 8 / radius // normalize cluster spread to ~±8 for legibility
    const offset = splashCentroid ? Cr.clone().sub(splashCentroid) : ROUTE_SECTOR.clone()
    // keep a minimum separation so the route doesn't sit on top of the splash
    const len = offset.length()
    if (len < 14) offset.copy(len > 1e-3 ? offset.multiplyScalar(14 / len) : ROUTE_SECTOR)
    const tf = (p: [number, number, number]): [number, number, number] => [
      offset.x + (p[0] - Cr.x) * s,
      offset.y + (p[1] - Cr.y) * s,
      offset.z + (p[2] - Cr.z) * s,
    ]
    const map = (n: TrackNode): TrackNode => ({ ...n, position: tf(n.position) })
    return {
      R: {
        ...route,
        path: route.path.map(map),
        cloud: route.cloud.map(map),
        requested_start: map(route.requested_start),
        requested_end: map(route.requested_end),
      },
      center: offset.clone(),
    }
  }, [route, splashCentroid])

  const pathPts = useMemo(() => R.path.map((n) => new THREE.Vector3(...n.position)), [R])
  const curve = useMemo(
    () => new THREE.CatmullRomCurve3(pathPts, false, 'centripetal', 0.4),
    [pathPts],
  )
  const camGoal = useMemo(() => {
    const dir = new THREE.Vector3(0.35, 0.5, 1).normalize()
    return center.clone().add(dir.multiplyScalar(24))
  }, [center])

  const nodeById = useMemo(() => {
    const m = new Map<string, TrackNode>()
    for (const n of [...R.path, ...R.cloud, R.requested_start, R.requested_end]) {
      m.set(n.id, n)
      m.set(n.uri, n)
    }
    return m
  }, [R])

  // faint web: each cloud node → nearest path point (shows the manifold)
  const web = useMemo(() => {
    const segs: number[] = []
    for (const c of R.cloud) {
      const cp = new THREE.Vector3(...c.position)
      let best = pathPts[0]
      let bestD = Infinity
      for (const p of pathPts) {
        const d = cp.distanceToSquared(p)
        if (d < bestD) {
          bestD = d
          best = p
        }
      }
      segs.push(cp.x, cp.y, cp.z, best.x, best.y, best.z)
    }
    return new Float32Array(segs)
  }, [R, pathPts])

  const snapEdges = R.edges.filter((e) => e.kind === 'snap')

  return (
    <group>
      <WebLines positions={web} />

      {/* route: soft tube + bright core line */}
      <mesh>
        <tubeGeometry args={[curve, Math.max(24, R.path.length * 12), 0.045, 8, false]} />
        <meshStandardMaterial
          color={SIGNAL}
          emissive={SIGNAL}
          emissiveIntensity={0.9}
          transparent
          opacity={0.55}
          roughness={0.3}
        />
      </mesh>
      <Line points={pathPts} color={SIGNAL} lineWidth={2.4} transparent opacity={0.95} />

      {/* snap links */}
      {snapEdges.map((e, i) => {
        const a = nodeById.get(e.from)
        const b = nodeById.get(e.to)
        if (!a || !b) return null
        return (
          <Line
            key={i}
            points={[a.position, b.position]}
            color={COOL}
            lineWidth={1.4}
            dashed
            dashScale={6}
            transparent
            opacity={0.7}
          />
        )
      })}

      {/* cloud — tappable for info, but unlabeled by default to avoid clutter */}
      {R.cloud.map((n) => (
        <GlowDot
          key={n.id}
          position={n.position}
          color={genreColor(n.genre)}
          size={0.05}
          glow={0.55}
          onPick={() => onInspect(n)}
        />
      ))}

      {/* path nodes */}
      {R.path.map((n, i) => (
        <PathNode
          key={`${n.id}-${i}`}
          node={n}
          index={i}
          focused={focus === i}
          onFocus={onFocus}
          onInspect={onInspect}
        />
      ))}

      {/* requested endpoints */}
      <Endpoint node={R.requested_start} label="From" />
      <Endpoint node={R.requested_end} label="To" />

      <RouteFlyIn goal={camGoal} center={center} trigger={route} onArrive={onArrive} />
      {arrived && <Tracer curve={curve} count={R.path.length} />}
      <FocusRig curve={curve} pathPts={pathPts} focus={focus} idle={focus == null} active={arrived} />
    </group>
  )
}

// Glide the camera from wherever the splash left it to a framing of the path
// sector, then hand control back so the route can be explored / traced.
function RouteFlyIn({
  goal,
  center,
  trigger,
  onArrive,
}: {
  goal: THREE.Vector3
  center: THREE.Vector3
  trigger: unknown
  onArrive: () => void
}) {
  const camera = useThree((s) => s.camera)
  const controls = useThree((s) => s.controls) as unknown as
    | { target: THREE.Vector3; update: () => void; enabled: boolean }
    | null
  const anim = useRef<{
    from: THREE.Vector3
    target: THREE.Vector3
    cur: THREE.Vector3
    t: number
    done: boolean
  } | null>(null)

  // restart the flight whenever a new route arrives
  useEffect(() => {
    anim.current = null
  }, [trigger])
  // never leave the controls disabled if we unmount mid-flight
  useEffect(() => () => { if (controls) controls.enabled = true }, [controls])

  useFrame((_, delta) => {
    if (anim.current?.done) return
    if (!anim.current) {
      if (prefersReducedMotion) {
        camera.position.copy(goal)
        camera.lookAt(center)
        if (controls) controls.target.copy(center)
        anim.current = { from: goal, target: center, cur: center.clone(), t: 1, done: true }
        onArrive()
        return
      }
      anim.current = {
        from: camera.position.clone(),
        target: controls ? controls.target.clone() : center.clone(),
        cur: new THREE.Vector3(),
        t: 0,
        done: false,
      }
      // take over from OrbitControls for the duration (drei only updates when
      // enabled, so this stops it from overwriting our camera each frame).
      if (controls) controls.enabled = false
    }
    const a = anim.current
    a.t = Math.min(1, a.t + delta / 2.6)
    const e = easeInOutCubic(a.t)
    camera.position.lerpVectors(a.from, goal, e)
    a.cur.lerpVectors(a.target, center, e)
    camera.lookAt(a.cur)
    if (controls) controls.target.copy(a.cur)
    if (a.t >= 1) {
      a.done = true
      if (controls) controls.enabled = true // hand control back, target already at center
      onArrive()
    }
  })
  return null
}

function WebLines({ positions }: { positions: Float32Array }) {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    return g
  }, [positions])
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial color="#3a4670" transparent opacity={0.28} depthWrite={false} />
    </lineSegments>
  )
}

function PathNode({
  node,
  index,
  focused,
  onFocus,
  onInspect,
}: {
  node: TrackNode
  index: number
  focused: boolean
  onFocus: (i: number | null) => void
  onInspect: (n: TrackNode | null) => void
}) {
  const [hover, setHover] = useState(false)
  const show = focused || hover
  return (
    <group position={node.position}>
      <mesh
        onPointerOver={(e) => {
          e.stopPropagation()
          setHover(true)
          onFocus(index)
        }}
        onPointerOut={() => setHover(false)}
        onClick={(e) => {
          e.stopPropagation()
          onInspect(node)
        }}
      >
        <sphereGeometry args={[show ? 0.16 : 0.1, 20, 20]} />
        <meshStandardMaterial color="#fff7e6" emissive={SIGNAL} emissiveIntensity={show ? 1.6 : 0.9} roughness={0.25} />
      </mesh>
      {haloTexture && (
        <sprite scale={show ? 1.7 : 1.05}>
          <spriteMaterial map={haloTexture} color={SIGNAL} blending={THREE.AdditiveBlending} transparent depthWrite={false} opacity={show ? 0.85 : 0.5} />
        </sprite>
      )}
      {show && (
        <Billboard position={[0, 0.34, 0]}>
          <Text fontSize={0.17} color="#fbf7ee" anchorX="center" anchorY="bottom" maxWidth={3} outlineWidth={0.006} outlineColor="#070810">
            {node.name}
          </Text>
          <Text position={[0, -0.02, 0]} fontSize={0.115} color="#b9c2dd" anchorX="center" anchorY="top" maxWidth={3}>
            {node.artist}
          </Text>
        </Billboard>
      )}
    </group>
  )
}

function Endpoint({ node, label }: { node: TrackNode; label: string }) {
  const snapped = !!node.snapped_to
  return (
    <group position={node.position}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.32, 0.022, 12, 40]} />
        <meshStandardMaterial color={COOL} emissive={COOL} emissiveIntensity={1.2} roughness={0.3} />
      </mesh>
      <GlowDot position={[0, 0, 0]} color="#ffffff" size={0.13} glow={1.1} />
      <Billboard position={[0, 0.5, 0]}>
        <Text fontSize={0.135} color={COOL} anchorX="center" anchorY="bottom" letterSpacing={0.08} outlineWidth={0.006} outlineColor="#070810">
          {label.toUpperCase()}
        </Text>
        <Text position={[0, 0.04, 0]} fontSize={0.2} color="#ffffff" anchorX="center" anchorY="top" maxWidth={3.4} outlineWidth={0.006} outlineColor="#070810">
          {node.name}
        </Text>
        {snapped && (
          <Text position={[0, -0.34, 0]} fontSize={0.1} color="#9fb0d6" anchorX="center" anchorY="top" maxWidth={3.4}>
            {`↳ snapped to ${node.snapped_label ?? 'nearest in map'}`}
          </Text>
        )}
      </Billboard>
    </group>
  )
}

function Tracer({ curve, count }: { curve: THREE.CatmullRomCurve3; count: number }) {
  const comet = useRef<THREE.Group>(null)
  useFrame(({ clock }) => {
    if (!comet.current) return
    const t = prefersReducedMotion ? 1 : (clock.elapsedTime * 0.05) % 1
    comet.current.position.copy(curve.getPointAt(Math.min(0.999, t)))
  })
  void count
  return (
    <group ref={comet}>
      <mesh>
        <sphereGeometry args={[0.13, 24, 24]} />
        <meshStandardMaterial color="#ffffff" emissive={SIGNAL} emissiveIntensity={2.2} roughness={0.1} />
      </mesh>
      {haloTexture && (
        <sprite scale={2.4}>
          <spriteMaterial map={haloTexture} color={SIGNAL} blending={THREE.AdditiveBlending} transparent depthWrite={false} opacity={0.95} />
        </sprite>
      )}
    </group>
  )
}

// Smoothly steer the orbit target toward a focused stop (or the route center when idle).
function FocusRig({
  curve,
  pathPts,
  focus,
  idle,
  active,
}: {
  curve: THREE.CatmullRomCurve3
  pathPts: THREE.Vector3[]
  focus: number | null
  idle: boolean
  active: boolean
}) {
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3; update: () => void } | null
  // world-space center of the (already world-positioned) path
  const center = useMemo(() => {
    const c = new THREE.Vector3()
    pathPts.forEach((p) => c.add(p))
    return pathPts.length ? c.multiplyScalar(1 / pathPts.length) : c
  }, [pathPts])
  const goal = useRef(new THREE.Vector3())
  useFrame(() => {
    if (!active || !controls) return
    const target = focus != null && pathPts[focus] ? goal.current.copy(pathPts[focus]) : center
    goal.current.copy(target)
    if (prefersReducedMotion) {
      controls.target.copy(goal.current)
    } else {
      controls.target.lerp(goal.current, idle ? 0.02 : 0.08)
    }
    controls.update()
  })
  void curve
  return null
}

type Reason = { label: string; value?: string; bar?: number; detail: string }

function describeWhy(why: WhyHop, genre: string): Reason[] {
  const out: Reason[] = []
  out.push({
    label: 'Distance on the map',
    value: why.dist.toFixed(2),
    bar: Math.max(0, 1 - why.dist / 0.6),
    detail:
      why.dist < 0.15
        ? 'A short hop — these two sit close in taste space.'
        : why.dist < 0.3
          ? 'A moderate step across the map.'
          : 'A longer leap to reach the next region.',
  })
  out.push({
    label: 'Listening fit',
    value: why.fit.toFixed(2),
    bar: why.fit,
    detail:
      why.fit > 0.6
        ? 'Scores high on the rotation model — a track you’d likely keep playing.'
        : why.fit > 0.35
          ? 'Middling fit, taken to bridge the gap.'
          : 'Lower fit, chosen because it bridges the route well.',
  })
  out.push({
    label: why.genre_jump ? 'Genre shift' : 'Genre',
    detail: why.genre_jump ? `Eases from ${why.prev_genre} into ${genre}.` : `Stays in ${genre}.`,
  })
  if (why.transition != null) {
    out.push({
      label: 'Follows naturally',
      value: why.transition.toFixed(2),
      bar: Math.min(1, why.transition * 4),
      detail:
        why.transition > 0.05
          ? 'Often played right after the previous track in real listening.'
          : 'Little direct listening history between the two.',
    })
  }
  if (why.context != null) {
    out.push({
      label: 'Moment fit',
      value: why.context.toFixed(2),
      detail:
        why.context > 1.05
          ? 'Leans into the chosen time of day.'
          : why.context < 0.95
            ? 'Runs slightly against the chosen time of day.'
            : 'Neutral for the chosen time of day.',
    })
  }
  return out
}

function Inspector({
  node,
  onClose,
  isPlaying,
  onTogglePlay,
}: {
  node: TrackNode
  onClose: () => void
  isPlaying: boolean
  onTogglePlay: () => void
}) {
  const why = node.why
  return (
    <aside className="inspector" role="dialog" aria-label={`About ${node.name}`}>
      <button className="inspClose" onClick={onClose} aria-label="Close">
        <X size={15} aria-hidden />
      </button>
      <div className="inspHead">
        <button
          className={`inspPlay${isPlaying ? ' on' : ''}`}
          onClick={onTogglePlay}
          aria-label={isPlaying ? `Stop ${node.name}` : `Play a sample of ${node.name}`}
          title={isPlaying ? 'Stop' : 'Play a 30-second sample'}
          style={{ '--g': genreColor(node.genre) } as React.CSSProperties}
        >
          {isPlaying ? <Pause size={16} aria-hidden /> : <Play size={16} aria-hidden />}
        </button>
        <div className="inspTitle">
          <b>{node.name}</b>
          <small>{node.artist}{node.album ? ` · ${node.album}` : ''}</small>
        </div>
      </div>
      <div className="inspMeta">
        <span className="metaTag">{node.genre}</span>
        {node.fit != null && <span className="metaTag">fit {node.fit.toFixed(2)}</span>}
        {node.kind === 'cloud' && <span className="metaTag dim">near the route</span>}
        {node.kind === 'sample' && <span className="metaTag dim">in the corpus</span>}
        {node.spotify_url && (
          <a className="metaTag link" href={node.spotify_url} target="_blank" rel="noreferrer">
            Open in Spotify ↗
          </a>
        )}
      </div>
      {why ? (
        <div className="inspWhy">
          <h4>Why this stop</h4>
          {describeWhy(why, node.genre).map((r, i) => (
            <div className="reason" key={i}>
              <div className="reasonHead">
                <span>{r.label}</span>
                {r.value != null && <b>{r.value}</b>}
              </div>
              {r.bar != null && (
                <span className="bar">
                  <span style={{ width: `${Math.round(r.bar * 100)}%` }} />
                </span>
              )}
              <p>{r.detail}</p>
            </div>
          ))}
        </div>
      ) : node.kind === 'cloud' ? (
        <p className="inspNote">
          A neighbor of the route, not chosen for it — close enough on the map to show the
          terrain the path moves through.
        </p>
      ) : node.kind === 'sample' ? (
        <p className="inspNote">
          One track from a random corner of the map. Search a start and a destination above to
          plot a journey through this space.
        </p>
      ) : (
        <p className="inspNote">Where the journey begins.</p>
      )}
    </aside>
  )
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
