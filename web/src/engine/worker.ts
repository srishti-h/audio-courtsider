/// <reference lib="webworker" />
/**
 * In-browser live engine: the same event stream as the server's /ws/live (session, words, segment,
 * partial, acoustic, detection, status, feed, score), so the Live UI works with either.
 *
 * Whisper runs through transformers.js (WebGPU when available); the text event model and the audio
 * goal model are this project's own weights exported by scripts/export_web_models.py. Streaming follows
 * the server's LocalAgreement idea at phrase level: a phrase is committed once two consecutive passes
 * over the growing buffer agree on it, and the unconfirmed tail is scored speculatively (the fast path).
 */
import { pipeline, PreTrainedTokenizer, env } from '@huggingface/transformers'
import * as ort from 'onnxruntime-web'
import { calibrate, logMel, normalise, type MelConfig } from './features'
import { diff, snapshot, type Snapshot } from './espn'

env.allowLocalModels = false

const SR = 16000, TICK_MS = 1000, POLL_MS = 5000, MAX_BUFFER_S = 20, AUDIO_WINDOW_S = 40, REFRACTORY_S = 20
const DETECT = ['goal', 'penalty', 'red_card']

export interface StartOptions {
  modelsUrl: string; whisper: 'base' | 'small'; language: 'english' | 'spanish'; threshold: number
  matchId?: string; home?: string; away?: string
}

const post = (events: any[]) => events.length && self.postMessage({ type: 'events', events })
const now = () => Date.now() / 1000

let asr: any, tok: any, textModel: ort.InferenceSession, audioModel: ort.InferenceSession | null = null
let meta: any, fb: Float32Array
let opts: StartOptions
let buf = new Float32Array(0), bufStart = 0, total = 0, recent = new Float32Array(0)
let prevHyp: { text: string; start: number; end: number }[] = []
let history: string[] = [], lastSpec = '', busy = false, running = false
const lastFire: Record<string, number> = {}
let timers: number[] = []

async function load(o: StartOptions) {
  const progress = (what: string) => (p: any) => {
    if (p.status === 'progress_total' || p.status === 'progress') self.postMessage({ type: 'loading', what, progress: p.progress })
  }
  meta = await (await fetch(`${o.modelsUrl}meta.json`)).json()
  const [tj, tc] = await Promise.all(['tokenizer.json', 'tokenizer_config.json'].map(async (f) => (await fetch(`${o.modelsUrl}text/${f}`)).json()))
  tok = new (PreTrainedTokenizer as any)(tj, tc)
  const gpu = 'gpu' in navigator && !!(await (navigator as any).gpu.requestAdapter().catch(() => null))
  self.postMessage({ type: 'loading', what: 'text model' })
  // CPU (wasm), not WebGPU: on WebGPU this fp16 model scored a clear goal call at P=0.03 instead of
  // Python's 0.60; on wasm it matches Python exactly (scripts/check-text.ts)
  textModel = await ort.InferenceSession.create(`${o.modelsUrl}text/model.onnx`, { executionProviders: ['wasm'] })
  self.postMessage({ type: 'loading', what: 'audio model' })
  try { // the audio model is optional: only present when its weights are published
    audioModel = await ort.InferenceSession.create(`${o.modelsUrl}audio/model.onnx`, { executionProviders: ['wasm'] })
    fb = new Float32Array(await (await fetch(`${o.modelsUrl}audio/mel.bin`)).arrayBuffer())
  } catch { audioModel = null }
  self.postMessage({ type: 'loading', what: `Whisper ${o.whisper}` })
  asr = await pipeline('automatic-speech-recognition', `onnx-community/whisper-${o.whisper}`, {
    device: gpu ? 'webgpu' : 'wasm',
    dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' } as any,
    progress_callback: progress(`Whisper ${o.whisper}`),
  })
  return gpu
}

function encode(texts: string[]) {
  // left truncation like the Python tokenizer (truncation_side="left"): keep <s> and the newest tokens
  const max = meta.max_len
  const rows = texts.map((t) => { const ids: number[] = Array.from(tok.encode(t)); return ids.length > max ? [ids[0], ...ids.slice(ids.length - max + 1)] : ids })
  const L = Math.max(...rows.map((r) => r.length))
  const ids = new BigInt64Array(rows.length * L), mask = new BigInt64Array(rows.length * L)
  const pad = BigInt(tok.pad_token_id ?? 1)
  rows.forEach((r, i) => { for (let j = 0; j < L; j++) { ids[i * L + j] = j < r.length ? BigInt(r[j]) : pad; mask[i * L + j] = j < r.length ? 1n : 0n } })
  return { input_ids: new ort.Tensor('int64', ids, [rows.length, L]), attention_mask: new ort.Tensor('int64', mask, [rows.length, L]) }
}

async function score(texts: string[]): Promise<{ probs: number[][]; team: number[][] }> {
  const out = await textModel.run(encode(texts))
  const E = meta.event_types.length, ev = out.events.data as Float32Array, tm = out.team.data as Float32Array
  const probs = texts.map((_, i) => Array.from(ev.slice(i * E, (i + 1) * E), (x) => 1 / (1 + Math.exp(-x))))
  const team = texts.map((_, i) => { const r = Array.from(tm.slice(i * 3, i * 3 + 3)); const m = Math.max(...r); const e = r.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b); return e.map((x) => x / s) })
  return { probs, team }
}

const input = (text: string) => `${opts.home} vs ${opts.away} | ${history.slice(-2).join(' ')} || ${text}`
const asProbs = (p: number[]) => Object.fromEntries(meta.event_types.map((e: string, k: number) => [e, Math.round(p[k] * 1e4) / 1e4]))

function fire(p: number[], team: number[], text: string, wall: number, source: string): any[] {
  const out: any[] = []
  meta.event_types.forEach((e: string, k: number) => {
    if (DETECT.includes(e) && p[k] >= opts.threshold && wall - (lastFire[e] ?? -1e9) > REFRACTORY_S) {
      lastFire[e] = wall // shared across sources: one alert per event
      out.push({ kind: 'detection', source, event: e, p: Math.round(p[k] * 1e4) / 1e4, team: team[0] >= team[1] ? 'home' : 'away', text, wall })
    }
  })
  return out
}

const norm = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N} ]/gu, '').trim()

async function transcribe() {
  const t0 = performance.now()
  const res = await asr(buf, { language: opts.language, task: opts.language === 'english' ? 'transcribe' : 'translate', return_timestamps: true })
  const dur = buf.length / SR
  const hyp = (res.chunks ?? [])
    .map((c: any) => ({ text: c.text.trim(), start: bufStart + (c.timestamp[0] ?? 0), end: bufStart + (c.timestamp[1] ?? dur) }))
    .filter((c: any) => /\p{L}/u.test(c.text))
  // commit leading phrases two passes agree on and that don't touch the buffer's still-growing edge
  let n = 0
  while (n < hyp.length && n < prevHyp.length && norm(hyp[n].text) === norm(prevHyp[n].text) && hyp[n].end < bufStart + dur - 1) n++
  if (dur > MAX_BUFFER_S) n = Math.max(n, hyp.length - 1)
  return { hyp, n, compute: (performance.now() - t0) / 1000 }
}

async function tick() {
  if (busy || !running || buf.length < SR) return
  busy = true
  try {
    const events: any[] = []
    const { hyp, n, compute } = await transcribe()
    const wall = now()
    events.push({ kind: 'words', wall, latency: Math.round(compute * 100) / 100 })
    const done = hyp.slice(0, n)
    if (done.length) {
      const { probs, team } = await score(done.map((c: any) => input(c.text)))
      done.forEach((c: any, i: number) => {
        history.push(c.text)
        events.push({ kind: 'segment', wall, audio_start: c.start, audio_end: c.end, text: c.text, probs: asProbs(probs[i]), team: { home: team[i][0], away: team[i][1] } })
        events.push(...fire(probs[i], team[i], c.text, wall, 'text'))
      })
      const cut = Math.max(0, Math.round((done[done.length - 1].end - bufStart) * SR))
      buf = buf.slice(cut)
      bufStart += cut / SR
    }
    prevHyp = hyp.slice(n)
    const tail = prevHyp.map((c) => c.text).join(' ')
    if (tail && tail !== lastSpec) { // the fast path: score words that aren't confirmed yet
      lastSpec = tail
      const { probs, team } = await score([input(tail)])
      events.push({ kind: 'partial', wall, text: tail, p_goal: probs[0][0] }, ...fire(probs[0], team[0], tail, wall, 'text-fast'))
    }
    if (audioModel && recent.length >= 3 * SR) events.push(...(await acoustic(wall)))
    events.push({ kind: 'status', wall, audio_seconds: total / SR })
    post(events)
  } catch (e: any) {
    post([{ kind: 'error', message: String(e?.message ?? e) }])
  } finally { busy = false }
}

async function acoustic(wall: number): Promise<any[]> {
  const a: MelConfig & any = meta.audio
  const mel = logMel(recent, fb, a)
  const x = normalise(mel, a)
  const out = await audioModel!.run({ mel: new ort.Tensor('float32', x, [1, mel.length, a.n_mels * 2]) })
  const logits = out.logits.data as Float32Array
  const p = calibrate(1 / (1 + Math.exp(-logits[logits.length - 1])), a.calibrator.x, a.calibrator.y)
  const events: any[] = [{ kind: 'acoustic', wall, crowd_z: null, excitement: null, p_goal_audio: Math.round(p * 1e4) / 1e4 }]
  if (p >= a.threshold && wall - (lastFire.audio ?? -1e9) > REFRACTORY_S) {
    lastFire.audio = wall
    events.push({ kind: 'detection', source: 'audio', event: 'goal', p: Math.round(p * 1e4) / 1e4, team: null, wall })
  }
  return events
}

async function watchFeed(prev: Snapshot) {
  const poll = async () => {
    if (!running) return
    try {
      const cur = await snapshot(opts.matchId!)
      post([...diff(prev, cur, now()), { kind: 'score', wall: now(), score: cur.score, clock: cur.clock, state: cur.state }])
      prev = cur
    } catch { /* feed hiccup: try again next poll */ }
  }
  timers.push(self.setInterval(poll, POLL_MS))
}

self.onmessage = async (m: MessageEvent) => {
  const msg = m.data
  if (msg.type === 'start') {
    opts = { home: 'Home', away: 'Away', ...msg.opts }
    try {
      const gpu = await load(opts)
      let snap: Snapshot | null = null
      if (opts.matchId) {
        try { snap = await snapshot(opts.matchId); opts.home = snap.home; opts.away = snap.away } catch (e: any) { post([{ kind: 'error', message: `feed unavailable: ${e?.message ?? e}` }]) }
      }
      running = true
      post([{ kind: 'session', wall: now(), home: opts.home, away: opts.away, match_id: opts.matchId, feed: 'espn', vocabulary: [], score: snap?.score ?? null, clock: snap?.clock ?? null }])
      self.postMessage({ type: 'ready', gpu, audioModel: !!audioModel })
      if (snap) watchFeed(snap)
      timers.push(self.setInterval(tick, TICK_MS))
    } catch (e: any) {
      self.postMessage({ type: 'failed', message: String(e?.message ?? e) })
    }
  } else if (msg.type === 'audio') {
    const chunk: Float32Array = msg.pcm
    const nb = new Float32Array(buf.length + chunk.length); nb.set(buf); nb.set(chunk, buf.length); buf = nb
    total += chunk.length
    const keep = AUDIO_WINDOW_S * SR, nr = new Float32Array(Math.min(keep, recent.length + chunk.length))
    const joined = new Float32Array(recent.length + chunk.length); joined.set(recent); joined.set(chunk, recent.length)
    nr.set(joined.subarray(joined.length - nr.length)); recent = nr
  } else if (msg.type === 'score') { // diagnostics: score raw inputs with the loaded text model
    if (!textModel) await load(msg.opts)
    self.postMessage({ type: 'scores', ...(await score(msg.texts)) })
  } else if (msg.type === 'stop') {
    running = false
    timers.forEach((t) => clearInterval(t)); timers = []
  }
}
