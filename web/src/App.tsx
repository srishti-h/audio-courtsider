import { useState } from 'react'
import { LiveView } from './LiveView'
import { ReplayView } from './ReplayView'
import { REPO_URL } from './source'

type Tab = 'replay' | 'live'

export default function App() {
  const [tab, setTab] = useState<Tab>(new URLSearchParams(location.search).get('tab') === 'live' ? 'live' : 'replay')
  return (
    <>
      <div className="stripe-band" />
      <div className="app">
        <nav className="nav">
          <div className="logo">
            <div className="logo-mark" />
            <div>
              <h1>Audio <span>Courtsider</span></h1>
              <p>Hearing the goal before the data feed does</p>
            </div>
          </div>
          <div className="tabs">
            <button className={tab === 'replay' ? 'on' : ''} onClick={() => setTab('replay')}>Replay</button>
            <button className={tab === 'live' ? 'on' : ''} onClick={() => setTab('live')}>Live</button>
          </div>
          <a className="gh" href={REPO_URL}>GitHub ↗</a>
        </nav>
        {tab === 'replay' ? <ReplayView /> : <LiveView />}
        <footer>
          Commentary text: SoccerNet-Echoes (CC BY 4.0). Event labels: SoccerNet. Live scores: ESPN public scoreboard. Prices are
          simulated; play money only. Not affiliated with any club.
        </footer>
      </div>
    </>
  )
}
