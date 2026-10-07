import { useEffect, useState } from 'react'
import { Books, fmtClock, Meters, PnlChart, ResultsPanel, Timeline, Transcript } from './components'
import { BASE, demoFile, REPO_URL, STATIC } from './source'
import type { DemoMatch, Game, ReplayParams } from './types'
import { useReplay } from './useReplay'

export default function App() {
  const [params, setParams] = useState<ReplayParams | null>(null)
  const state = useReplay(params)

  return (
    <div className="app">
      <header>
        <div className="brand">
          <h1>Audio Courtsider</h1>
          <p>Hearing the goal before the data feed does: commentary → beliefs → trades on a simulated in-play exchange.</p>
        </div>
        {STATIC ? <StaticControls onPlay={setParams} /> : <LiveControls onPlay={setParams} />}
      </header>

      {state.status === 'idle' && (STATIC ? <Intro onPlay={setParams} /> : (
        <div className="empty">Pick a match and press play, or jump straight to a goal.</div>
      ))}
      {state.status === 'loading' && <div className="empty">{STATIC ? 'Loading the replay…' : 'Running the simulation…'}</div>}
      {state.status === 'error' && (
        <div className="empty">{STATIC ? 'Could not load this replay.' : 'Could not reach the backend on :8000.'}</div>
      )}

      {state.meta && (
        <>
          <div className="scorebar">
            <span className="team">{state.meta.home}</span>
            <span className="score">{state.score[0]} – {state.score[1]}</span>
            <span className="team">{state.meta.away}</span>
            <span className="clock">{fmtClock(state.half, state.clock)}</span>
            <span className="muted">
              score shown as the official feed reports it · pipeline latency {state.meta.latency.toFixed(1)}s
            </span>
            {state.status === 'ended' && <span className="muted">· replay finished</span>}
          </div>
          <main>
            <section className="col">
              <h2>Commentary</h2>
              <Transcript signals={state.signals} />
            </section>
            <section className="col">
              <h2>Beliefs</h2>
              <Meters signals={state.signals} now={state.now} />
              <h2>P&amp;L</h2>
              <PnlChart pnl={state.pnl} />
            </section>
            <section className="col wide">
              <h2>Order books</h2>
              <Books books={state.books} fair={state.fair} trades={state.trades} />
              <h2>Who knew first (last 90 s)</h2>
              <Timeline markers={state.markers} now={state.now} />
              <h2>Experiment results</h2>
              <ResultsPanel />
            </section>
          </main>
        </>
      )}
      <footer>
        Commentary text: SoccerNet-Echoes (CC BY 4.0). Event labels: SoccerNet. Prices are simulated; play money only. ·{' '}
        <a href={REPO_URL}>source on GitHub</a>
      </footer>
    </div>
  )
}

// ---- static (GitHub Pages) mode -------------------------------------------------------------

function useDemoIndex() {
  const [matches, setMatches] = useState<DemoMatch[]>([])
  useEffect(() => {
    fetch(`${BASE}demo/index.json`).then((r) => r.json()).then((d) => setMatches(d.matches))
  }, [])
  return matches
}

const staticParams = (match: DemoMatch, goal: number, feedDelay: number, guard: boolean, speed: number): ReplayParams => ({
  file: demoFile(match.id, goal, feedDelay, guard),
  game_id: match.id,
  speed,
  feed_delay: feedDelay,
  cs_threshold: 0.2,
  mm_audio_guard: guard,
  start: 0,
})

function StaticControls({ onPlay }: { onPlay: (p: ReplayParams) => void }) {
  const matches = useDemoIndex()
  const [id, setId] = useState('')
  const [feedDelay, setFeedDelay] = useState(8)
  const [guard, setGuard] = useState(false)
  const [speed, setSpeed] = useState(2)
  useEffect(() => {
    if (!matches.length) return
    // shareable links: ?match=<id>&goal=<n>&fd=<1|8>&guard=1
    const q = new URLSearchParams(location.search)
    const m = matches.find((x) => x.id === q.get('match')) ?? matches[0]
    setId(m.id)
    if (q.get('goal')) {
      const fd = Number(q.get('fd') ?? 8)
      setFeedDelay(fd)
      setGuard(q.get('guard') === '1')
      setSpeed(Number(q.get('speed') ?? 2))
      onPlay(staticParams(m, Number(q.get('goal')), fd, q.get('guard') === '1', Number(q.get('speed') ?? 2)))
    }
  }, [matches]) // eslint-disable-line react-hooks/exhaustive-deps
  const match = matches.find((m) => m.id === id)
  return (
    <div className="controls">
      <select value={id} onChange={(e) => setId(e.target.value)}>
        {matches.map((m) => <option key={m.id} value={m.id}>{m.date} · {m.label}</option>)}
      </select>
      <label>
        market's feed delay{' '}
        <select value={feedDelay} onChange={(e) => setFeedDelay(+e.target.value)}>
          <option value={1}>1 s (fast data feed)</option>
          <option value={8}>8 s (TV-delayed book)</option>
        </select>
      </label>
      <label>speed <input type="number" min={1} max={20} value={speed} onChange={(e) => setSpeed(+e.target.value)} />×</label>
      <label className="check">
        <input type="checkbox" checked={guard} onChange={(e) => setGuard(e.target.checked)} /> audio-aware market maker
      </label>
      {match?.goals.map((g) => (
        <button key={g.n} onClick={() => onPlay(staticParams(match, g.n, feedDelay, guard, speed))}>
          ▶ goal {g.n} · {g.minute}' {g.team}
        </button>
      ))}
    </div>
  )
}

function Intro({ onPlay }: { onPlay: (p: ReplayParams) => void }) {
  const matches = useDemoIndex()
  const showcase = matches[0]
  return (
    <section className="intro">
      <div className="pitch">
        <p>
          A live soccer goal reaches people at different times: the crowd in the stadium, the commentator a second later, an
          official data feed a few seconds after that, and TV viewers 5–10 s behind. This project measures what that gap is
          worth. It listens to match commentary (streaming Whisper + a fine-tuned DistilRoBERTa event model), turns it into
          calibrated beliefs about goals, cards and penalties, and trades on them in a simulated in-play exchange against a
          market maker that only reprices when the official feed arrives.
        </p>
        <p>
          Each replay is a window around a real goal from the SoccerNet test set: real commentary, real event times, simulated
          prices. Switch the market's feed delay to see the edge appear (8 s) or vanish (1 s).
        </p>
        {showcase && (
          <button className="cta" onClick={() => onPlay(staticParams(showcase, 2, 8, false, 2))}>
            ▶ Watch {showcase.label}, goal 2, against a TV-delayed market
          </button>
        )}
        <a className="repo" href={REPO_URL}>Read the code, methods and full results on GitHub →</a>
      </div>
      <ResultsPanel />
    </section>
  )
}

// ---- local mode (FastAPI backend) -----------------------------------------------------------

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
      // shareable links: ?game=<id>&start=<s>&speed=&feed_delay=&threshold=&autoplay=1
      const q = new URLSearchParams(location.search)
      const initial = g.find((x) => x.game_id === q.get('game')) ?? g.find((x) => x.goals.length >= 2) ?? g[0]
      if (!initial) return
      setGameId(initial.game_id)
      if (q.get('autoplay')) {
        onPlay({
          game_id: initial.game_id,
          speed: Number(q.get('speed') ?? 5),
          feed_delay: Number(q.get('feed_delay') ?? 8),
          cs_threshold: Number(q.get('threshold') ?? 0.2),
          mm_audio_guard: q.get('guard') === '1',
          start: Number(q.get('start') ?? 0),
        })
      }
    })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const game = games.find((g) => g.game_id === gameId)
  const play = (start: number) =>
    onPlay({ game_id: gameId, speed, feed_delay: feedDelay, cs_threshold: threshold, mm_audio_guard: guard, start })

  return (
    <div className="controls">
      <select value={gameId} onChange={(e) => setGameId(e.target.value)}>
        {games.map((g) => <option key={g.game_id} value={g.game_id}>{g.date} · {g.label}</option>)}
      </select>
      <label>speed <input type="number" min={1} max={50} value={speed} onChange={(e) => setSpeed(+e.target.value)} />×</label>
      <label>
        feed delay <input type="number" step={0.5} min={0} max={20} value={feedDelay} onChange={(e) => setFeedDelay(+e.target.value)} />s
      </label>
      <label>
        entry threshold <input type="number" step={0.05} min={0.1} max={0.99} value={threshold} onChange={(e) => setThreshold(+e.target.value)} />
      </label>
      <label className="check"><input type="checkbox" checked={guard} onChange={(e) => setGuard(e.target.checked)} /> audio-aware market maker</label>
      <button onClick={() => play(0)}>▶ from kick-off</button>
      {game?.goals.map((g, i) => (
        <button key={g} className="ghost" onClick={() => play(Math.max(0, g - 45))}>goal {i + 1}</button>
      ))}
    </div>
  )
}
