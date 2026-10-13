import { useEffect, useState } from 'react'
import { Meters, ProbChart, Scoreboard, Transcript } from './components'
import { STATIC, REPO_URL } from './source'
import { type AudioSource, useLive } from './useLive'

interface FeedMatch { id: string; name: string; state: string; clock: string; home: string; away: string; score: [number, number]; error?: string }

const ago = (wall: number, start?: number) => {
  if (!start) return ''
  const s = Math.max(0, Math.round(wall - start))
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

export function LiveView() {
  if (STATIC) return <LiveUnavailable />
  return <LiveApp />
}

function LiveUnavailable() {
  return (
    <div className="card span-12" style={{ marginTop: 16 }}>
      <h2>Live mode</h2>
      <p className="hint">
        Live mode listens to a match while you watch it: your microphone, or the audio of a browser tab streaming the game.
        Streaming Whisper turns the commentary into text, the event model and crowd-roar detector fire in real time, and a
        live-score feed timestamps when the "official" world finds out. That way you see how many seconds the commentary
        wins by. Speech recognition runs on your own machine, so this page can't do it. Clone the{' '}
        <a href={REPO_URL}>repo</a> and run <code>make serve</code> and <code>make web</code>, then open the Live tab.
      </p>
    </div>
  )
}

function LiveApp() {
  const { state, start, stop } = useLive()
  const [matches, setMatches] = useState<FeedMatch[]>([])
  const [pick, setPick] = useState<FeedMatch | null>(null)
  const [source, setSource] = useState<AudioSource>('tab')
  const [file, setFile] = useState<File | undefined>()
  const [threshold, setThreshold] = useState(0.5)
  const [manual, setManual] = useState({ home: '', away: '' })
  const [now, setNow] = useState(Date.now() / 1000)

  useEffect(() => {
    const load = () => fetch('/api/live/matches').then((r) => r.json()).then((m: FeedMatch[]) => setMatches(m.filter((x) => !x.error)))
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
          <div className="card">
            <h2>1 · Pick the match <small>live-score feed (ESPN, polled every 5 s)</small></h2>
            <div className="match-list">
              {matches.length === 0 && <div className="note">No matches from the feed right now. Type the teams instead.</div>}
              {matches.map((m) => (
                <button key={m.id} className={`match-item ${pick?.id === m.id ? 'on' : ''}`} onClick={() => setPick(m)}>
                  <span>
                    <div>{m.home} vs {m.away}</div>
                    <div className={`st ${m.state}`}>{m.state === 'in' ? `LIVE · ${m.clock}` : m.state === 'pre' ? 'upcoming' : 'finished'}</div>
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
          <div className="card">
            <h2>2 · Point it at the commentary</h2>
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
              disabled={!pick && !(manual.home && manual.away)}
              onClick={() => start({ source, file, threshold, matchId: pick?.id, home: pick?.home ?? manual.home, away: pick?.away ?? manual.away })}
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
            <span><span className="live-dot" />Listening · {Math.floor(state.audioSeconds / 60)}m {Math.round(state.audioSeconds % 60)}s of audio</span>
            {state.asrLatency !== undefined && <span className="hint">speech → text lag <b>{state.asrLatency.toFixed(1)}s</b></span>}
            {state.vocabulary.length > 0 && <span className="hint">name-biased with {state.vocabulary.length} lineup names</span>}
            <div className="chips"><button className="btn ghost" onClick={stop}>■ Stop</button></div>
          </div>
          <div className="grid" style={{ marginTop: 16 }}>
            <div className="card span-5">
              <h2>Commentary <small>streaming Whisper</small></h2>
              <Transcript lines={state.segments.map((s, i) => ({ key: `${s.wall}-${i}`, ts: ago(s.wall, state.started), text: s.text, p: Math.max(s.probs.goal, s.probs.penalty, s.probs.red_card) }))} />
            </div>
            <div className="card span-7">
              <h2>What the bot believes <small>last 2 minutes</small></h2>
              <ProbChart points={points} now={now} marks={marks} />
              <Meters goal={last?.probs.goal ?? 0} penalty={last?.probs.penalty ?? 0} red={last?.probs.red_card ?? 0} team={last?.team} />
              <h2 style={{ marginTop: 14 }}>Crowd <small>high-band energy vs the last 25 s · fires at z ≥ 4</small></h2>
              <div className="roar"><i style={{ width: `${Math.min(100, Math.max(0, (lastCrowd?.z ?? 0) / 8) * 100)}%` }} /></div>
            </div>
            <div className="card span-12">
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
                        <span className="lead">{lead?.text !== undefined ? `text +${lead.text.toFixed(1)}s` : ''} {lead?.crowd !== undefined ? `crowd +${lead.crowd.toFixed(1)}s` : ''}</span>
                      </div>
                    )
                  }
                  return (
                    <div key={i} className="det">
                      <span className="ts">{ago(e.wall, state.started)}</span>
                      <span><span className={`src ${e.source}`}>{e.source}</span> {e.event}{e.team ? ` · ${e.team}` : ''}{e.text ? ` · "${e.text}"` : ''}</span>
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
