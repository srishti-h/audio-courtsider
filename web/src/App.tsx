import { useEffect, useState } from 'react'
import { Books, fmtClock, Meters, PnlChart, Timeline, Transcript } from './components'
import type { Game, ReplayParams } from './types'
import { useReplay } from './useReplay'

export default function App() {
  const [games, setGames] = useState<Game[]>([])
  const [gameId, setGameId] = useState('')
  const [speed, setSpeed] = useState(5)
  const [feedDelay, setFeedDelay] = useState(2)
  const [threshold, setThreshold] = useState(0.5)
  const [guard, setGuard] = useState(false)
  const [params, setParams] = useState<ReplayParams | null>(null)
  const state = useReplay(params)

  useEffect(() => {
    fetch('/api/games').then((r) => r.json()).then((g: Game[]) => {
      setGames(g)
      const withGoals = g.find((x) => x.goals.length >= 2) ?? g[0]
      if (withGoals) setGameId(withGoals.game_id)
    })
  }, [])

  const game = games.find((g) => g.game_id === gameId)
  const play = (start: number) =>
    setParams({ game_id: gameId, speed, feed_delay: feedDelay, cs_threshold: threshold, mm_audio_guard: guard, start })

  return (
    <div className="app">
      <header>
        <div className="brand">
          <h1>Audio Courtsider</h1>
          <p>Hearing the goal before the data feed does: commentary → beliefs → trades on a simulated in-play exchange.</p>
        </div>
        <div className="controls">
          <select value={gameId} onChange={(e) => setGameId(e.target.value)}>
            {games.map((g) => <option key={g.game_id} value={g.game_id}>{g.date} · {g.label}</option>)}
          </select>
          <label>speed <input type="number" min={1} max={50} value={speed} onChange={(e) => setSpeed(+e.target.value)} />×</label>
          <label>feed delay <input type="number" step={0.5} min={0} max={10} value={feedDelay} onChange={(e) => setFeedDelay(+e.target.value)} />s</label>
          <label>entry threshold <input type="number" step={0.05} min={0.1} max={0.99} value={threshold} onChange={(e) => setThreshold(+e.target.value)} /></label>
          <label className="check"><input type="checkbox" checked={guard} onChange={(e) => setGuard(e.target.checked)} /> audio-aware market maker</label>
          <button onClick={() => play(0)}>▶ from kick-off</button>
          {game?.goals.map((g, i) => (
            <button key={g} className="ghost" onClick={() => play(Math.max(0, g - 45))}>goal {i + 1}</button>
          ))}
        </div>
      </header>

      {state.status === 'idle' && <div className="empty">Pick a match and press play, or jump straight to a goal.</div>}
      {state.status === 'loading' && <div className="empty">Running the simulation…</div>}
      {state.status === 'error' && <div className="empty">Could not reach the backend on :8000.</div>}

      {state.meta && (
        <>
          <div className="scorebar">
            <span className="team">{state.meta.home}</span>
            <span className="score">{state.score[0]} – {state.score[1]}</span>
            <span className="team">{state.meta.away}</span>
            <span className="clock">{fmtClock(state.half, state.clock)}</span>
            <span className="muted">score shown as the official feed reports it · pipeline latency {state.meta.latency.toFixed(1)}s</span>
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
              <h2>Who knew first (last 60 s)</h2>
              <Timeline markers={state.markers} now={state.now} />
            </section>
          </main>
        </>
      )}
      <footer>
        Commentary text: SoccerNet-Echoes (CC BY 4.0). Event labels: SoccerNet. Prices are simulated; play money only.
      </footer>
    </div>
  )
}
