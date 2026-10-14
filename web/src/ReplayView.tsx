import { useEffect, useState } from 'react'
import { Books, EdgeTable, fmtClock, Kpis, Meters, PnlChart, ProbChart, Race, Scoreboard, signalPoints, Transcript, useResults } from './components'
import { BASE, demoFile, REPO_URL, STATIC } from './source'
import type { DemoMatch, Game, ReplayParams } from './types'
import { useReplay } from './useReplay'

export function ReplayView() {
  const [params, setParams] = useState<ReplayParams | null>(null)
  const state = useReplay(params)
  const results = useResults()
  const last = state.signals[state.signals.length - 1]
  const marks = state.markers
    .filter((m) => m.kind === 'truth' || m.kind === 'feed')
    .map((m) => ({ t: m.t, kind: m.kind as 'truth' | 'feed', label: m.kind === 'truth' ? `${m.event} on pitch` : 'feed reports' }))

  return (
    <>
      {STATIC ? <StaticControls onPlay={setParams} /> : <LiveControls onPlay={setParams} />}

      {state.status === 'idle' && <Intro onPlay={setParams} r={results} />}
      {state.status === 'loading' && <div className="empty">{STATIC ? 'Loading the replay…' : 'Running the simulation…'}</div>}
      {state.status === 'error' && <div className="empty">{STATIC ? 'Could not load this replay.' : 'Could not reach the backend on :8000.'}</div>}

      {state.meta && (
        <>
          <div data-explain="scoreboard"><Scoreboard home={state.meta.home} away={state.meta.away} score={state.score}
            clock={fmtClock(state.half, state.clock)}
            note={`score as the official feed reports it · speech pipeline latency ${state.meta.latency.toFixed(1)}s${state.status === 'ended' ? ' · replay finished' : ''}`} /></div>
          <div className="grid">
            <div className="card span-12" data-explain="race">
              <h2>The race · who knew first <small>last 90 seconds</small></h2>
              <Race markers={state.markers} now={state.now} />
              <div className="race-legend">
                <span><i style={{ background: 'var(--gold)' }} />event on the pitch</span>
                <span><i style={{ background: 'var(--grana)' }} />commentary bot trades</span>
                <span><i style={{ background: 'var(--blau)' }} />official feed reports it, market maker reprices</span>
              </div>
            </div>
            <div className="card span-5" data-explain="commentary">
              <h2>Commentary <small>SoccerNet-Echoes</small></h2>
              <Transcript lines={state.signals.map((s, i) => ({ key: `${s.t}-${i}`, ts: fmtClock(s.half, s.clock), text: s.text, p: Math.max(s.probs.goal ?? 0, s.probs.penalty ?? 0, s.probs.red_card ?? 0) }))} />
            </div>
            <div className="card span-7" data-explain="beliefs">
              <h2>What the bot believes <small>P(goal) solid · penalty dashed · red card blue</small></h2>
              <ProbChart points={signalPoints(state.signals)} now={state.now} marks={marks} />
              <div data-explain="meters"><Meters goal={last?.probs.goal ?? 0} penalty={last?.probs.penalty ?? 0} red={last?.probs.red_card ?? 0} team={last?.team as any} /></div>
            </div>
            <div className="card span-8" data-explain="books">
              <h2>Order books <small>market maker's quotes · fair value from the in-play model</small></h2>
              <Books books={state.books} fair={state.fair} trades={state.trades} />
            </div>
            <div className="card span-4" data-explain="pnl">
              <h2>P&amp;L</h2>
              <PnlChart pnl={state.pnl} />
            </div>
          </div>
        </>
      )}
    </>
  )
}

// ---- intro (landing) -------------------------------------------------------------------------

function useDemoIndex() {
  const [matches, setMatches] = useState<DemoMatch[]>([])
  useEffect(() => {
    if (!STATIC) return
    fetch(`${BASE}demo/index.json`).then((r) => r.json()).then((d) => setMatches(d.matches))
  }, [])
  return matches
}

function Intro({ onPlay, r }: { onPlay: (p: ReplayParams) => void; r: any }) {
  const matches = useDemoIndex()
  const showcase = matches[0]
  return (
    <section className="hero">
      <div className="card pitch">
        <h2>When a goal goes in, who finds out first?</h2>
        <div className="timeline-strip" data-explain="timeline">
          <div className="c1"><b>0 s</b><span>the ball crosses the line</span></div>
          <div className="c2"><b>~1–6 s</b><span>the commentator calls it</span></div>
          <div className="c3"><b>2–10 s</b><span>the official data feed</span></div>
          <div className="c4"><b>5–40 s</b><span>TV and stream viewers</span></div>
        </div>
        <p>
          This project measures what that gap is worth. It listens to match commentary with streaming Whisper and a fine-tuned
          DistilRoBERTa event model, turns it into calibrated beliefs about goals, cards and penalties, and trades on them in a
          simulated in-play exchange. The other side is a market maker that only reprices when the official feed arrives.
        </p>
        <p>
          Each replay is a window around a real goal from the SoccerNet test set: real commentary, real event times, simulated
          prices. Switch the market's feed delay to watch the edge appear at 8 s and vanish at 1 s.
        </p>
        <div className="cta-row">
          {STATIC && showcase && <button className="btn" data-explain="watch" onClick={() => onPlay(staticParams(showcase, 2, 8, false, 2))}>▶ Watch a goal against a TV-delayed market</button>}
          {!STATIC && <span className="hint">Pick a match above and press play, or jump straight to a goal.</span>}
          <button className="btn gold" onClick={() => window.dispatchEvent(new Event('courtsider-tour'))}>? Take the tour</button>
          <a className="btn ghost" href={REPO_URL}>Code, methods &amp; full results</a>
        </div>
      </div>
      <div className="card" data-explain="kpis">
        <h2>Headline results <small>held-out test matches</small></h2>
        <Kpis r={r} />
        <div data-explain="edge"><EdgeTable r={r} /></div>
      </div>
    </section>
  )
}

// ---- controls --------------------------------------------------------------------------------

const staticParams = (match: DemoMatch, goal: number, feedDelay: number, guard: boolean, speed: number): ReplayParams => ({
  file: demoFile(match.id, goal, feedDelay, guard), game_id: match.id, speed, feed_delay: feedDelay, cs_threshold: 0.2,
  mm_audio_guard: guard, start: 0,
})

function FeedToggle({ value, onChange, options }: { value: number; onChange: (v: number) => void; options: [number, string][] }) {
  return (
    <div className="seg">
      {options.map(([v, l], i) => (
        <button key={v} className={value === v ? `on ${i ? 'grana' : ''}` : ''} onClick={() => onChange(v)}>{l}</button>
      ))}
    </div>
  )
}

function StaticControls({ onPlay }: { onPlay: (p: ReplayParams) => void }) {
  const matches = useDemoIndex()
  const [id, setId] = useState('')
  const [feedDelay, setFeedDelay] = useState(8)
  const [guard, setGuard] = useState(false)
  const [speed, setSpeed] = useState(2)
  useEffect(() => {
    if (!matches.length) return
    // shareable links: ?match=<id>&goal=<n>&fd=<1|8>&guard=1&speed=
    const q = new URLSearchParams(location.search)
    const m = matches.find((x) => x.id === q.get('match')) ?? matches[0]
    setId(m.id)
    if (q.get('goal')) {
      const fd = Number(q.get('fd') ?? 8), g = q.get('guard') === '1', sp = Number(q.get('speed') ?? 2)
      setFeedDelay(fd); setGuard(g); setSpeed(sp)
      onPlay(staticParams(m, Number(q.get('goal')), fd, g, sp))
    }
  }, [matches]) // eslint-disable-line react-hooks/exhaustive-deps
  const match = matches.find((m) => m.id === id)
  return (
    <div className="controls">
      <select data-explain="match" value={id} onChange={(e) => setId(e.target.value)}>
        {matches.map((m) => <option key={m.id} value={m.id}>{m.date} · {m.label}</option>)}
      </select>
      <span data-explain="feed" className="controls-group"><label>Market's feed</label>
      <FeedToggle value={feedDelay} onChange={setFeedDelay} options={[[1, '1 s · fast data feed'], [8, '8 s · TV-delayed']]} /></span>
      <label className="toggle" data-explain="guard"><input type="checkbox" checked={guard} onChange={(e) => setGuard(e.target.checked)} /> MM listens too</label>
      <label data-explain="speed">Speed <input type="number" min={1} max={20} value={speed} onChange={(e) => setSpeed(+e.target.value)} />×</label>
      <div className="chips" data-explain="goals">
        {match?.goals.map((g) => (
          <button key={g.n} className="btn ghost" onClick={() => onPlay(staticParams(match, g.n, feedDelay, guard, speed))}>
            ▶ Goal {g.n} · {g.minute}' {g.team}
          </button>
        ))}
      </div>
    </div>
  )
}

function LiveControls({ onPlay }: { onPlay: (p: ReplayParams) => void }) {
  const [games, setGames] = useState<Game[]>([])
  const [gameId, setGameId] = useState('')
  const [speed, setSpeed] = useState(5)
  const [feedDelay, setFeedDelay] = useState(8)
  const [threshold, setThreshold] = useState(0.2)
  const [guard, setGuard] = useState(false)
  useEffect(() => {
    fetch('/api/games').then((r) => r.json()).then((g: Game[]) => {
      setGames(g)
      const q = new URLSearchParams(location.search)
      const initial = g.find((x) => x.game_id === q.get('game')) ?? g.find((x) => x.goals.length >= 2) ?? g[0]
      if (!initial) return
      setGameId(initial.game_id)
      if (q.get('autoplay')) {
        onPlay({ game_id: initial.game_id, speed: Number(q.get('speed') ?? 5), feed_delay: Number(q.get('feed_delay') ?? 8),
          cs_threshold: Number(q.get('threshold') ?? 0.2), mm_audio_guard: q.get('guard') === '1', start: Number(q.get('start') ?? 0) })
      }
    })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps
  const game = games.find((g) => g.game_id === gameId)
  const play = (start: number) => onPlay({ game_id: gameId, speed, feed_delay: feedDelay, cs_threshold: threshold, mm_audio_guard: guard, start })
  return (
    <div className="controls">
      <select data-explain="match" value={gameId} onChange={(e) => setGameId(e.target.value)}>
        {games.map((g) => <option key={g.game_id} value={g.game_id}>{g.date} · {g.label}</option>)}
      </select>
      <label data-explain="feed">Feed delay <input type="number" step={0.5} min={0} max={20} value={feedDelay} onChange={(e) => setFeedDelay(+e.target.value)} />s</label>
      <label>Entry P ≥ <input type="number" step={0.05} min={0.1} max={0.99} value={threshold} onChange={(e) => setThreshold(+e.target.value)} /></label>
      <label className="toggle" data-explain="guard"><input type="checkbox" checked={guard} onChange={(e) => setGuard(e.target.checked)} /> MM listens too</label>
      <label data-explain="speed">Speed <input type="number" min={1} max={50} value={speed} onChange={(e) => setSpeed(+e.target.value)} />×</label>
      <div className="chips" data-explain="goals">
        <button className="btn" onClick={() => play(0)}>▶ From kick-off</button>
        {game?.goals.map((g, i) => <button key={g} className="btn ghost" onClick={() => play(Math.max(0, g - 45))}>Goal {i + 1}</button>)}
      </div>
    </div>
  )
}
