import React, { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Billboard, Line, OrbitControls, Stars, Text } from '@react-three/drei'
import { ArrowRight, Compass, Loader2, MapPin, Search, Shuffle, Sparkles, X } from 'lucide-react'
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
  kind: 'path' | 'requested'
  snapped_to?: string | null
  snapped_label?: string | null
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
// App
// ===========================================================================
function App() {
  const [start, setStart] = useState<Candidate | null>(null)
  const [end, setEnd] = useState<Candidate | null>(null)
  const [length, setLength] = useState(14)
  const [context, setContext] = useState('now')
  const [shuffle, setShuffle] = useState('any')

  const [route, setRoute] = useState<RouteResponse | null>(null)
  const [status, setStatus] = useState<'idle' | 'routing'>('idle')
  const [step, setStep] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [focus, setFocus] = useState<number | null>(null)

  useEffect(() => {
    if (status !== 'routing') return
    setStep(0)
    const t = window.setInterval(() => setStep((s) => Math.min(LOADING_STEPS.length - 1, s + 1)), 650)
    return () => window.clearInterval(t)
  }, [status])

  const trace = async () => {
    if (!start || !end || status === 'routing') return
    setStatus('routing')
    setError(null)
    setFocus(null)
    try {
      const data = await request<RouteResponse>(
        '/api/route',
        post({ start_uri: start.uri, end_uri: end.uri, length, context, shuffle }),
      )
      setRoute(data)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e))
    } finally {
      setStatus('idle')
    }
  }

  const snapped =
    route && (route.requested_start.snapped_to || route.requested_end.snapped_to)

  return (
    <main className="app">
      <Scene route={route} idle={!route} focus={focus} onFocus={setFocus} />

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
          <JourneyStrip route={route} focus={focus} onFocus={setFocus} snapped={!!snapped} />
          <Legend />
        </>
      )}
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
    const timer = window.setTimeout(() => {
      setLoading(true)
      request<Candidate[]>(`/api/search?q=${encodeURIComponent(q)}`)
        .then((hits) => {
          if (id !== seq.current) return
          setResults(hits)
          setActive(0)
          setError(null)
          setOpen(true)
        })
        .catch((e) => {
          if (id !== seq.current) return
          setError(e instanceof Error ? e.message : String(e))
          setResults([])
          setOpen(true)
        })
        .finally(() => id === seq.current && setLoading(false))
    }, 240)
    return () => window.clearTimeout(timer)
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
      <button className="iconbtn ghost" aria-label="Route options" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <Shuffle size={15} aria-hidden />
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
  snapped,
}: {
  route: RouteResponse
  focus: number | null
  onFocus: (i: number | null) => void
  snapped: boolean
}) {
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
      </div>
      <ol className="stripList" onMouseLeave={() => onFocus(null)}>
        {route.path.map((n, i) => (
          <li key={`${n.id}-${i}`}>
            <button
              className={`stop${focus === i ? ' active' : ''}`}
              onMouseEnter={() => onFocus(i)}
              onFocus={() => onFocus(i)}
              onClick={() => onFocus(i)}
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
}: {
  route: RouteResponse | null
  idle: boolean
  focus: number | null
  onFocus: (i: number | null) => void
}) {
  return (
    <Canvas camera={{ position: [0, 4, 19], fov: 50 }} dpr={[1, 2]} gl={{ antialias: true }}>
      <color attach="background" args={['#070810']} />
      <fog attach="fog" args={['#070810', 18, 46]} />
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 14, 8]} intensity={1.4} color="#fff0d4" />
      <pointLight position={[-12, -8, -6]} intensity={0.9} color="#5fb8ff" />
      <Suspense fallback={null}>
        <Stars radius={70} depth={40} count={1600} factor={2.6} fade speed={prefersReducedMotion ? 0 : 0.4} />
        {route ? <RouteField route={route} focus={focus} onFocus={onFocus} /> : <IdleField />}
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

function IdleField() {
  const group = useRef<THREE.Group>(null)
  const dots = useMemo(() => {
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
  }, [])
  useFrame(({ clock }) => {
    if (group.current && !prefersReducedMotion) group.current.rotation.y = clock.elapsedTime * 0.05
  })
  return (
    <group ref={group}>
      {dots.map((d, i) => (
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
}: {
  position: [number, number, number]
  color: string
  size: number
  glow?: number
}) {
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[size, 16, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.8} roughness={0.35} />
      </mesh>
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
}: {
  route: RouteResponse
  focus: number | null
  onFocus: (i: number | null) => void
}) {
  const pathPts = useMemo(() => route.path.map((n) => new THREE.Vector3(...n.position)), [route])
  const curve = useMemo(
    () => new THREE.CatmullRomCurve3(pathPts, false, 'centripetal', 0.4),
    [pathPts],
  )

  const nodeById = useMemo(() => {
    const m = new Map<string, TrackNode>()
    for (const n of [...route.path, ...route.cloud, route.requested_start, route.requested_end]) {
      m.set(n.id, n)
      m.set(n.uri, n)
    }
    return m
  }, [route])

  // faint web: each cloud node → nearest path point (shows the manifold)
  const web = useMemo(() => {
    const segs: number[] = []
    for (const c of route.cloud) {
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
  }, [route, pathPts])

  const snapEdges = route.edges.filter((e) => e.kind === 'snap')

  return (
    <group>
      <WebLines positions={web} />

      {/* route: soft tube + bright core line */}
      <mesh>
        <tubeGeometry args={[curve, Math.max(24, route.path.length * 12), 0.045, 8, false]} />
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

      {/* cloud */}
      {route.cloud.map((n) => (
        <GlowDot key={n.id} position={n.position} color={genreColor(n.genre)} size={0.05} glow={0.55} />
      ))}

      {/* path nodes */}
      {route.path.map((n, i) => (
        <PathNode
          key={`${n.id}-${i}`}
          node={n}
          index={i}
          focused={focus === i}
          onFocus={onFocus}
        />
      ))}

      {/* requested endpoints */}
      <Endpoint node={route.requested_start} label="From" />
      <Endpoint node={route.requested_end} label="To" />

      <Tracer curve={curve} count={route.path.length} />
      <FocusRig curve={curve} pathPts={pathPts} focus={focus} idle={focus == null} />
    </group>
  )
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
}: {
  node: TrackNode
  index: number
  focused: boolean
  onFocus: (i: number | null) => void
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
}: {
  curve: THREE.CatmullRomCurve3
  pathPts: THREE.Vector3[]
  focus: number | null
  idle: boolean
}) {
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3; update: () => void } | null
  const center = useMemo(() => {
    const c = new THREE.Vector3()
    pathPts.forEach((p) => c.add(p))
    return pathPts.length ? c.multiplyScalar(1 / pathPts.length) : c
  }, [pathPts])
  const goal = useRef(new THREE.Vector3())
  useFrame(() => {
    if (!controls) return
    const target = focus != null && pathPts[focus] ? pathPts[focus] : center
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

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
