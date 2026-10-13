import { useCallback, useEffect, useRef, useState } from 'react'

export type AudioSource = 'mic' | 'tab' | 'file'

export interface LiveSegment { wall: number; text: string; probs: Record<string, number>; team: { home: number; away: number } }
export interface LiveDetection { wall: number; source: 'text' | 'text-fast' | 'crowd' | 'audio'; event: string; p: number | null; team: string | null; text?: string }
export interface LiveFeedEvent { wall: number; event: string; team: string; score: [number, number]; clock: string }
export interface Lead { feed: LiveFeedEvent; text?: number; crowd?: number; audio?: number }

export interface LiveState {
  status: 'idle' | 'connecting' | 'listening' | 'stopped' | 'error'
  error?: string
  home: string
  away: string
  score: [number, number]
  clock: string
  started?: number
  segments: LiveSegment[]
  detections: LiveDetection[]
  feed: LiveFeedEvent[]
  leads: Lead[]
  crowd: { wall: number; z: number; ex: number; pAudio: number | null }[]
  audioSeconds: number
  asrLatency?: number
  vocabulary: string[]
  hasFeed?: boolean
}

const initial: LiveState = {
  status: 'idle', home: '', away: '', score: [0, 0], clock: '', segments: [], detections: [], feed: [], leads: [],
  crowd: [], audioSeconds: 0, vocabulary: [],
}

// Captures audio frames in the audio thread; the AudioContext runs at 16 kHz so the browser resamples.
const WORKLET = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true }
}
registerProcessor('tap', Tap)`

function reduce(s: LiveState, e: any): LiveState {
  switch (e.kind) {
    case 'session':
      return { ...s, home: e.home, away: e.away, score: e.score ?? s.score, clock: e.clock ?? '', started: e.wall, vocabulary: e.vocabulary ?? [], hasFeed: !!e.match_id }
    case 'segment':
      return { ...s, segments: [...s.segments.slice(-199), e] }
    case 'words':
      return { ...s, asrLatency: e.latency }
    case 'acoustic':
      return { ...s, crowd: [...s.crowd.slice(-299), { wall: e.wall, z: e.crowd_z, ex: e.excitement, pAudio: e.p_goal_audio ?? null }] }
    case 'detection':
      return { ...s, detections: [...s.detections.slice(-49), e] }
    case 'score':
      return { ...s, score: e.score, clock: e.clock }
    case 'status':
      return { ...s, audioSeconds: e.audio_seconds }
    case 'feed': {
      const f: LiveFeedEvent = { wall: e.wall, event: e.event, team: e.team, score: e.score, clock: e.clock }
      const before = (srcs: string[]) => s.detections
        .filter((d) => srcs.includes(d.source) && (d.source === 'crowd' || d.event === f.event) && f.wall - d.wall >= 0 && f.wall - d.wall <= 90)
        .map((d) => f.wall - d.wall)
      const lead = (srcs: string[]) => { const v = before(srcs); return v.length ? Math.max(...v) : undefined }
      return { ...s, score: e.score, feed: [...s.feed, f],
        leads: [...s.leads, { feed: f, text: lead(['text', 'text-fast']), crowd: lead(['crowd']), audio: lead(['audio']) }] }
    }
    case 'error':
      return { ...s, error: e.message }
    default:
      return s
  }
}

export function useLive() {
  const [state, setState] = useState<LiveState>(initial)
  const cleanup = useRef<() => void>(() => {})

  const stop = useCallback(() => {
    cleanup.current()
    cleanup.current = () => {}
    setState((s) => ({ ...s, status: s.status === 'error' ? 'error' : 'stopped' }))
  }, [])

  const start = useCallback(async (opts: { source: AudioSource; file?: File; matchId?: string; home?: string; away?: string; threshold: number }) => {
    cleanup.current()
    setState({ ...initial, status: 'connecting' })
    let media: MediaStream | null = null
    let player: HTMLAudioElement | null = null
    try {
      if (opts.source === 'file') {
        if (!opts.file) throw new Error('Choose an audio file first.')
        player = new Audio(URL.createObjectURL(opts.file))
      } else {
        media = opts.source === 'mic'
          ? await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } })
          : await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true })
        if (!media.getAudioTracks().length) throw new Error('No audio track: when sharing a tab, tick "Also share tab audio".')
      }
    } catch (err: any) {
      setState((s) => ({ ...s, status: 'error', error: err?.message ?? String(err) }))
      return
    }
    const q = new URLSearchParams({ threshold: String(opts.threshold) })
    if (opts.matchId) q.set('match_id', opts.matchId)
    if (opts.home) q.set('home', opts.home)
    if (opts.away) q.set('away', opts.away)
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live?${q}`)
    ws.binaryType = 'arraybuffer'
    ws.onmessage = (m) => setState((s) => reduce(s, JSON.parse(m.data)))
    ws.onerror = () => setState((s) => ({ ...s, status: 'error', error: 'Lost the connection to the local backend (:8000).' }))
    ws.onopen = () => setState((s) => ({ ...s, status: 'listening' })) // attach before any await so the event is never missed

    const ctx = new AudioContext({ sampleRate: 16000 })
    await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' })))
    const src = player ? ctx.createMediaElementSource(player) : ctx.createMediaStreamSource(media!)
    const tap = new AudioWorkletNode(ctx, 'tap')
    const mute = ctx.createGain()
    mute.gain.value = 0
    src.connect(tap).connect(mute).connect(ctx.destination)
    if (player) {
      src.connect(ctx.destination) // a recording is played aloud while it is transcribed
      player.onended = () => setTimeout(() => stop(), 4000) // let the last words get transcribed
      player.play()
    }
    let pending: Float32Array[] = [], n = 0
    tap.port.onmessage = (ev: MessageEvent<Float32Array>) => {
      pending.push(ev.data)
      n += ev.data.length
      if (n >= 4000 && ws.readyState === WebSocket.OPEN) { // ~0.25 s per frame
        const buf = new Float32Array(n)
        let o = 0
        for (const p of pending) { buf.set(p, o); o += p.length }
        ws.send(buf.buffer)
        pending = []
        n = 0
      }
    }
    media?.getTracks().forEach((t) => (t.onended = () => stop()))
    cleanup.current = () => {
      tap.port.onmessage = null
      player?.pause()
      media?.getTracks().forEach((t) => t.stop())
      ctx.close()
      ws.close()
    }
  }, [stop])

  useEffect(() => () => cleanup.current(), [])
  return { state, start, stop }
}
