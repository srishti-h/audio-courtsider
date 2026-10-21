import { useEffect, useState } from 'react'
import { Meters, ProbChart, Scoreboard, Transcript } from './components'
import { matches as espnMatches } from './engine/espn'
import { MODELS_URL, STATIC } from './source'
import { type AudioSource, type Engine, useLive } from './useLive'

interface FeedMatch { id: string; name: string; state: string; clock: string; home: string; away: string; score: [number, number]; fair?: { home: number; draw: number; away: number } | null; error?: string }

const ago = (wall: number, start?: number) => {
  if (!start) return ''
  const s = Math.max(0, Math.round(wall - start))
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

export function LiveView() {
  return <LiveApp />
}

const WEBGPU = typeof navigator !== 'undefined' && 'gpu' in navigator

function LiveApp() {
  const { state, start, stop } = useLive()
  const [matches, setMatches] = useState<FeedMatch[]>([])
  const [pick, setPick] = useState<FeedMatch | null>(null)
  const [source, setSource] = useState<AudioSource>('tab')
  const [file, setFile] = useState<File | undefined>()
  const [threshold, setThreshold] = useState(0.5)
  const [manual, setManual] = useState({ home: '', away: '' })
  const [now, setNow] = useState(Date.now() / 1000)
  const [engine, setEngine] = useState<Engine>(STATIC ? 'browser' : 'server')
  const [whisper, setWhisper] = useState<'base' | 'small'>('base')
  const [language, setLanguage] = useState<'english' | 'spanish'>('english')
  const [modelsReady, setModelsReady] = useState<boolean | null>(null)

  useEffect(() => {
    if (engine !== 'browser') return
    fetch(new URL('meta.json', new URL(MODELS_URL, location.href)).href).then((r) => setModelsReady(r.ok)).catch(() => setModelsReady(false))
  }, [engine])

  useEffect(() => {
    const load = () => (STATIC ? espnMatches() : fetch('/api/live/matches').then((r) => r.json()))
      .then((m: FeedMatch[]) => setMatches(m.filter((x) => !x.error)))
      .catch(() => setMatches([]))
    load()
    const id = setInterval(load, 30000)
    return () => clearInterval(id)
  }, [])
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now() / 1000), 500)
    return () => clearInterval(id)
  }, [])

  const listening = state.status === 'listening' || state.status === 'connecting'
  const last = state.segments[state.segments.length - 1]
  const lastCrowd = state.crowd[state.crowd.length - 1]
  const points = state.segments.map((s) => ({ t: s.wall, goal: s.probs.goal, penalty: s.probs.penalty, red: s.probs.red_card }))
  const marks: { t: number; kind: 'truth' | 'feed'; label: string }[] = [
    ...state.feed.map((f) => ({ t: f.wall, kind: 'feed' as const, label: `feed: ${f.event}` })),
    ...state.detections.filter((d) => d.source === 'crowd').map((d) => ({ t: d.wall, kind: 'truth' as const, label: 'roar' })),
  ]

  return (
    <>
      {!listening && (
        <div className="live-setup" style={{ marginTop: 16 }}>
          <div className="card" data-explain="live-match">
            <h2>1 · Pick the match <small>live-score feed (ESPN, polled every 5 s)</small></h2>
            <div className="match-list">
              {matches.length === 0 && <div className="note">No matches from the feed right now. Type the teams instead.</div>}
              {matches.map((m) => (
                <button key={m.id} className={`match-item ${pick?.id === m.id ? 'on' : ''}`} onClick={() => setPick(m)}>
                  <span>
                    <div>{m.home} vs {m.away}</div>
                    <div className={`st ${m.state}`}>{m.state === 'in' ? `LIVE · ${m.clock}` : m.state === 'pre' ? 'upcoming' : 'finished'}</div>
                    {m.fair && <div className="fair-1x2" title="Pre-match fair probabilities from the Dixon–Coles model fit on the latest seasons">model {Math.round(m.fair.home * 100)} · {Math.round(m.fair.draw * 100)} · {Math.round(m.fair.away * 100)}%</div>}
                  </span>
                  <span className="sc">{m.state === 'pre' ? '' : `${m.score[0]}–${m.score[1]}`}</span>
                </button>
              ))}
            </div>
            <div className="controls" style={{ marginTop: 10, background: 'var(--bg-2)' }}>
              <label>or teams</label>
              <input type="text" placeholder="Home" value={manual.home} onChange={(e) => { setManual({ ...manual, home: e.target.value }); setPick(null) }} />
              <input type="text" placeholder="Away" value={manual.away} onChange={(e) => { setManual({ ...manual, away: e.target.value }); setPick(null) }} />
            </div>
          </div>
          <div className="card" data-explain="live-audio">
            <h2>2 · Point it at the commentary</h2>
            <div className="controls" style={{ background: 'var(--bg-2)', marginBottom: 8 }} data-explain="live-engine">
              {!STATIC && (
                <>
                  <label>Run models</label>
                  <div className="seg">
                    <button className={engine === 'server' ? 'on' : ''} onClick={() => setEngine('server')}>Local server</button>
                    <button className={engine === 'browser' ? 'on grana' : ''} onClick={() => setEngine('browser')}>This browser</button>
                  </div>
                </>
              )}
              {engine === 'browser' && (
                <>
                  <label>Whisper</label>
                  <div className="seg">
                    <button className={whisper === 'base' ? 'on' : ''} onClick={() => setWhisper('base')}>base · faster</button>
                    <button className={whisper === 'small' ? 'on grana' : ''} onClick={() => setWhisper('small')}>small · more accurate</button>
                  </div>
                  <label>Commentary</label>
                  <div className="seg">
                    <button className={language === 'english' ? 'on' : ''} onClick={() => setLanguage('english')}>English</button>
                    <button className={language === 'spanish' ? 'on grana' : ''} onClick={() => setLanguage('spanish')}>Spanish → English</button>
                  </div>
                </>
              )}
            </div>
            {engine === 'browser' && modelsReady === false && (
              <p className="hint" style={{ color: 'var(--down)' }}>
                The in-browser models haven't been published yet, so Live mode can't run here. It works on a local copy:
                run <code>python scripts/export_web_models.py</code>, then <code>make web</code>.
              </p>
            )}
            {engine === 'browser' && modelsReady !== false && (
              <p className="hint">
                Everything runs on this device: Whisper, the event model and the audio goal model download once
                (~{whisper === 'base' ? '300' : '650'} MB, then cached) and nothing you play leaves your browser.
                {WEBGPU ? ' Your browser has WebGPU, so it runs on the GPU.' : ' This browser has no WebGPU, so it will run slowly on the CPU: use desktop Chrome or Edge for real time.'}
              </p>
            )}
            <div className="controls" style={{ background: 'var(--bg-2)' }}>
              <label>Audio</label>
              <div className="seg">
                <button className={source === 'tab' ? 'on' : ''} onClick={() => setSource('tab')}>Browser tab</button>
                <button className={source === 'mic' ? 'on grana' : ''} onClick={() => setSource('mic')}>Microphone</button>
                <button className={source === 'file' ? 'on' : ''} onClick={() => setSource('file')}>Audio file</button>
              </div>
              {source === 'file' && <input type="file" accept="audio/*" aria-label="Commentary recording" onChange={(e) => setFile(e.target.files?.[0])} />}
              <label>Fire at P ≥ <input type="number" step={0.05} min={0.1} max={0.95} value={threshold} onChange={(e) => setThreshold(+e.target.value)} /></label>
            </div>
            <p className="hint">
              <b>Browser tab:</b> pick the tab playing the match and tick <i>Also share tab audio</i>.{' '}
              <b>Microphone:</b> put the laptop near the TV or radio. <b>Audio file:</b> a saved commentary recording plays in
              real time through the same pipeline. Nothing is recorded; audio is transcribed locally and only
              the text and detections are logged to <code>data/live/</code> for the lead-time report.
            </p>
            <button
              className="btn"
              disabled={(!pick && !(manual.home && manual.away)) || (engine === 'browser' && modelsReady === false)}
              onClick={() => start({ source, file, threshold, matchId: pick?.id, home: pick?.home ?? manual.home, away: pick?.away ?? manual.away,
                engine, whisper, language, modelsUrl: new URL(MODELS_URL, location.href).href })}
            >
              ● Start listening
            </button>
            {state.status === 'error' && <p className="hint" style={{ color: 'var(--down)' }}>{state.error}</p>}
            {state.status === 'stopped' && state.leads.length > 0 && <LeadSummary leads={state.leads} />}
          </div>
        </div>
      )}

      {listening && (
        <>
          <Scoreboard home={state.home || '…'} away={state.away || '…'} score={state.score}
            clock={state.clock || (state.status === 'connecting' ? 'connecting…' : 'live')}
            note={state.hasFeed ? 'score from the live feed · commentary from your audio' : 'no live feed (teams typed in) · commentary from your audio'} />
          <div className="controls">
            {state.status === 'connecting'
              ? <span className="loading-models">Loading {state.loading?.what ?? 'models'}{state.loading?.progress != null ? ` · ${Math.round(state.loading.progress)}%` : '…'}</span>
              : <span><span className="live-dot" />Listening · {Math.floor(state.audioSeconds / 60)}m {Math.round(state.audioSeconds % 60)}s of audio</span>}
            {state.asrLatency !== undefined && <span className="hint">speech → text lag <b>{state.asrLatency.toFixed(1)}s</b></span>}
            {state.vocabulary.length > 0 && <span className="hint">name-biased with {state.vocabulary.length} lineup names</span>}
            <div className="chips"><button className="btn ghost" onClick={stop}>■ Stop</button></div>
          </div>
          <div className="grid" style={{ marginTop: 16 }}>
            <div className="card span-5" data-explain="commentary">
              <h2>Commentary <small>streaming Whisper</small></h2>
              <Transcript lines={state.segments.map((s, i) => ({ key: `${s.wall}-${i}`, ts: ago(s.wall, state.started), text: s.text, p: Math.max(s.probs.goal, s.probs.penalty, s.probs.red_card) }))} />
            </div>
            <div className="card span-7" data-explain="beliefs">
              <h2>What the bot believes <small>last 2 minutes</small></h2>
              <ProbChart points={points} now={now} marks={marks} />
              <Meters goal={last?.probs.goal ?? 0} penalty={last?.probs.penalty ?? 0} red={last?.probs.red_card ?? 0} team={last?.team} />
              {lastCrowd?.z != null && (
                <>
                  <h2 style={{ marginTop: 14 }}>Crowd <small>high-band energy vs the recent baseline · fires at z ≥ 4</small></h2>
                  <div className="roar" data-explain="crowd"><i style={{ width: `${Math.min(100, Math.max(0, lastCrowd.z / 8) * 100)}%` }} /></div>
                </>
              )}
              {lastCrowd?.pAudio !== null && lastCrowd?.pAudio !== undefined && (
                <>
                  <h2 style={{ marginTop: 14 }}>Audio goal model <small>learned from 500 broadcasts · P(goal in the last 6 s)</small></h2>
                  <div className="roar" data-explain="audio-model"><i style={{ width: `${lastCrowd.pAudio * 100}%` }} /></div>
                </>
              )}
            </div>
            <div className="card span-12" data-explain="detections">
              <h2>Detections vs the feed <small>lead = seconds before the live-score feed reported it</small></h2>
              {state.detections.length === 0 && state.feed.length === 0 && <div className="note">Nothing yet. Goals, penalties, red cards and crowd roars show up here.</div>}
              {[...state.detections.map((d) => ({ ...d, kind: 'det' as const })), ...state.feed.map((f) => ({ ...f, kind: 'feed' as const }))]
                .sort((a, b) => b.wall - a.wall)
                .map((e, i) => {
                  if (e.kind === 'feed') {
                    const lead = state.leads.find((l) => l.feed.wall === e.wall)
                    return (
                      <div key={i} className="det">
                        <span className="ts">{ago(e.wall, state.started)}</span>
                        <span><span className="src feed">feed</span> {e.event} · {e.team} · {e.score[0]}–{e.score[1]} ({e.clock})</span>
                        <span className="lead">
                          {lead?.text !== undefined ? `text +${lead.text.toFixed(1)}s ` : ''}
                          {lead?.audio !== undefined ? `audio +${lead.audio.toFixed(1)}s ` : ''}
                          {lead?.crowd !== undefined ? `crowd +${lead.crowd.toFixed(1)}s` : ''}
                        </span>
                      </div>
                    )
                  }
                  return (
                    <div key={i} className="det">
                      <span className="ts">{ago(e.wall, state.started)}</span>
                      <span><span className={`src ${e.source.replace('-fast', '')}`}>{e.source === 'text-fast' ? 'text ⚡' : e.source}</span> {e.event}{e.team ? ` · ${e.team}` : ''}{e.text ? ` · "${e.text}"` : ''}</span>
                      <span>{e.p !== null ? `${Math.round(e.p * 100)}%` : ''}</span>
                    </div>
                  )
                })}
            </div>
          </div>
        </>
      )}
    </>
  )
}

function LeadSummary({ leads }: { leads: { text?: number; crowd?: number }[] }) {
  const t = leads.filter((l) => l.text !== undefined).map((l) => l.text!)
  const c = leads.filter((l) => l.crowd !== undefined).map((l) => l.crowd!)
  const med = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)]
  return (
    <p className="hint">
      Last session: the text model beat the feed on {t.length}/{leads.length} goals{t.length ? ` (median ${med(t).toFixed(1)} s)` : ''};
      the crowd on {c.length}/{leads.length}{c.length ? ` (median ${med(c).toFixed(1)} s)` : ''}. Run{' '}
      <code>python -m courtsider.live.report</code> to summarise every session.
    </p>
  )
}
