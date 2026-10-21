import { useResults } from './components'

interface Goal { clock?: string; team?: string; match?: string; [k: string]: string | number | undefined }
interface Session { match: string; started?: string; audio_minutes: number; feed_goals: number; goals: Goal[]; false_alarms: Record<string, number> }

const SOURCES: [string, string][] = [['text', 'Commentary text'], ['audio', 'Audio goal model'], ['crowd', 'Crowd roar']]
const lead = (g: Goal, src: string) => (g[`${src}_lead_s`] === undefined ? '–' : `+${g[`${src}_lead_s`]}s`)

export function RecorderView() {
  const r = useResults()
  const live = r?.live
  const sessions: Session[] = (live?.sessions ?? []).filter((s: Session) => s.feed_goals > 0).reverse()
  return (
    <>
      <div className="card" style={{ marginTop: 16 }} data-explain="recorder-how">
        <h2>Background recorder <small>Spanish radio vs the live-score feed, recorded unattended</small></h2>
        <ol className="hint recorder-steps">
          <li>Every minute it checks ESPN for live La Liga and Spanish-club European matches.</li>
          <li>When one is on, it streams Cadena SER radio and Whisper translates the Spanish commentary to English.</li>
          <li>The text model, the audio goal model and the crowd-roar rule run on it live, while the ESPN feed is polled every 5 s.</li>
          <li>Each goal on the feed is matched to the closest call heard just before it. No audio is kept: only text, detections and features.</li>
        </ol>
      </div>
      {!live && <div className="empty">No recorder results published yet. They appear here after <code>make refresh</code>.</div>}
      {live && (
        <>
          <div className="card" style={{ marginTop: 16 }} data-explain="recorder-kpis">
            <h2>So far <small>{live.n_sessions} sessions · {Math.round(live.audio_minutes / 60)} h of radio · {live.n_feed_goals} goals on the feed</small></h2>
            <div className="kpis kpis-3">
              {SOURCES.map(([k, label]) => (
                <div className="kpi" key={k}>
                  <b>{live[`${k}_beats_feed`]}</b>
                  <span>{label}: goals heard before the feed{live[`${k}_lead_s_median`] != null ? `, median ${Math.round(live[`${k}_lead_s_median`])} s earlier` : ''}</span>
                </div>
              ))}
            </div>
            <p className="note">ESPN's free feed itself trails the pitch by up to a minute, so leads here are larger than against a professional data feed.</p>
          </div>
          <div className="card" style={{ marginTop: 16 }} data-explain="recorder-goals">
            <h2>Every goal <small>lead = seconds before the live-score feed reported it</small></h2>
            <div className="table-scroll">
              <table className="edge-table">
                <thead><tr><th>Session</th><th>Match</th><th>Min</th>{SOURCES.map(([k, l]) => <th key={k}>{l}</th>)}<th>What the radio said</th></tr></thead>
                <tbody>
                  {sessions.flatMap((s) => s.goals.map((g, i) => (
                    <tr key={`${s.started}-${i}`}>
                      <td>{i === 0 ? s.started?.slice(0, 10) : ''}</td>
                      <td>{g.match ?? s.match}</td>
                      <td>{g.clock}</td>
                      {SOURCES.map(([k]) => <td key={k} className={g[`${k}_lead_s`] !== undefined ? 'lead' : 'faint'}>{lead(g, k)}</td>)}
                      <td className="quote">{g.text_quote ? `“${g.text_quote}…”` : ''}</td>
                    </tr>
                  )))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </>
  )
}
