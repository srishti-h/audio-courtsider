import { useEffect, useMemo, useRef, useState } from 'react'
import { RESULTS_URL } from './source'
import type { Level, MarkerMsg, PnlMsg, SignalMsg, TradeMsg } from './types'
import { CONTRACTS } from './types'

export const fmtClock = (half: number, clock: number) => {
  const m = Math.floor(clock / 60), s = Math.floor(clock % 60)
  return `${half === 1 ? '1H' : '2H'} ${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

const KEYWORDS = /\b(goa+l+|scores?|penalt\w*|red card|sent off|corner|yellow|offside|shot|saves?d?|lead)\b/gi
const escapeHtml = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!))
export const highlight = (s: string) => escapeHtml(s).replace(KEYWORDS, '<mark>$1</mark>')

// ---- scoreboard ------------------------------------------------------------------------------

export function Scoreboard({ home, away, score, clock, note }: {
  home: string; away: string; score: [number, number]; clock?: string; note?: string
}) {
  return (
    <div className="scoreboard">
      <div className="side"><span className="tag">Home</span><span className="name">{home}</span></div>
      <div className="mid">
        <div className="score num">{score[0]}<span className="dash">–</span>{score[1]}</div>
        {clock && <span className="clock">{clock}</span>}
        {note && <div className="subnote">{note}</div>}
      </div>
      <div className="side away"><span className="tag">Away</span><span className="name">{away}</span></div>
    </div>
  )
}

// ---- commentary ------------------------------------------------------------------------------

export function Transcript({ lines }: { lines: { key: string; ts: string; text: string; p?: number }[] }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => { ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: 'smooth' }) }, [lines.length])
  return (
    <div className="transcript" ref={ref}>
      {lines.length === 0 && <div className="note">Waiting for commentary…</div>}
      {lines.slice(-80).map((l) => (
        <div key={l.key} className={`line ${(l.p ?? 0) >= 0.5 ? 'hot' : ''}`}>
          <span className="ts">{l.ts}</span>
          <span dangerouslySetInnerHTML={{ __html: highlight(l.text) }} />
          {(l.p ?? 0) >= 0.2 ? <span className="pill">{Math.round((l.p ?? 0) * 100)}%</span> : <span />}
        </div>
      ))}
    </div>
  )
}

// ---- probability chart + meters --------------------------------------------------------------

export interface ProbPoint { t: number; goal: number; penalty: number; red: number }

export function ProbChart({ points, now, span = 120, marks = [] }: {
  points: ProbPoint[]; now: number; span?: number; marks?: { t: number; kind: 'truth' | 'feed'; label: string }[]
}) {
  const w = 600, h = 190, pad = 26, t0 = now - span
  const x = (t: number) => pad + ((Math.max(t, t0) - t0) / span) * (w - pad - 8)
  const y = (p: number) => h - 22 - p * (h - 40)
  const vis = points.filter((p) => p.t >= t0 - 30)
  // step lines: a belief holds until the next segment arrives
  const path = (k: keyof ProbPoint) =>
    vis.map((p, i) => `${i ? `L${x(p.t)},${y(vis[i - 1][k])}L` : 'M'}${x(p.t)},${y(p[k])}`).join('') +
    (vis.length ? `L${x(now)},${y(vis[vis.length - 1][k])}` : '')
  const area = vis.length ? `${path('goal')}L${x(now)},${y(0)}L${x(vis[0].t)},${y(0)}Z` : ''
  return (
    <div className="prob">
      <svg viewBox={`0 0 ${w} ${h}`}>
        <defs>
          <linearGradient id="gGoal" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="#c4214f" stopOpacity="0.45" />
            <stop offset="1" stopColor="#c4214f" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={pad} x2={w - 8} y1={y(v)} y2={y(v)} className="grid-line" />
            <text x={0} y={y(v) + 4} className="axis">{Math.round(v * 100)}</text>
          </g>
        ))}
        {[0, 30, 60, 90, 120].filter((s) => s <= span).map((s) => (
          <text key={s} x={x(now - s)} y={h - 4} className="axis" textAnchor="middle">{s ? `-${s}s` : 'now'}</text>
        ))}
        {marks.filter((m) => m.t >= t0).map((m, i, all) => {
          const labelled = !all.slice(0, i).some((o) => o.kind === m.kind && Math.abs(o.t - m.t) < 4)
          return (
            <g key={i}>
              <line x1={x(m.t)} x2={x(m.t)} y1={10} y2={y(0)} className={m.kind === 'truth' ? 'mark-truth' : 'mark-feed'} />
              {labelled && <text x={x(m.t) + 4} y={m.kind === 'truth' ? 14 : 28} className="mark-label">{m.label}</text>}
            </g>
          )
        })}
        <path d={area} className="area-goal" />
        <path d={path('red')} className="line-red" />
        <path d={path('penalty')} className="line-pen" />
        <path d={path('goal')} className="line-goal" />
      </svg>
    </div>
  )
}

export function Meters({ goal, penalty, red, team }: {
  goal: number; penalty: number; red: number; team?: { home: number; away: number }
}) {
  const items: [string, number][] = [['Goal', goal], ['Penalty', penalty], ['Red card', red]]
  const th = team ? team.home / Math.max(1e-9, team.home + team.away) : 0.5
  return (
    <>
      <div className="meters">
        {items.map(([k, v]) => (
          <div key={k} className={`meter ${v >= 0.5 ? 'hot' : ''}`}>
            <div className="lbl">P({k.toLowerCase()})</div>
            <b>{Math.round(v * 100)}%</b>
            <div className="bar"><i style={{ width: `${v * 100}%` }} /></div>
          </div>
        ))}
      </div>
      {team && (
        <>
          <div className="teamsplit"><i style={{ width: `${th * 100}%` }} /><i style={{ width: `${(1 - th) * 100}%` }} /></div>
          <div className="teamsplit-label"><span>home {Math.round(th * 100)}%</span><span>which team · text model</span><span>away {Math.round((1 - th) * 100)}%</span></div>
        </>
      )}
    </>
  )
}

export const signalPoints = (signals: SignalMsg[]): ProbPoint[] =>
  signals.map((s) => ({ t: s.t, goal: s.probs.goal ?? 0, penalty: s.probs.penalty ?? 0, red: s.probs.red_card ?? 0 }))

// ---- the race: who knew first ----------------------------------------------------------------

const LANES = [
  ['truth', 'On the pitch'],
  ['cs_entry', 'Commentary bot'],
  ['feed', 'Official feed'],
  ['cs_exit', 'Bot exits'],
] as const

export function Race({ markers, now, span = 90 }: { markers: MarkerMsg[]; now: number; span?: number }) {
  const w = 1240, lw = 170, laneH = 46, h = LANES.length * laneH + 56, t0 = now - span
  const x = (t: number) => lw + ((t - t0) / span) * (w - lw - 70)
  const lane = (k: string) => (k === 'mm_guard' ? 1 : LANES.findIndex(([n]) => n === k))
  const y = (k: string) => 44 + lane(k) * laneH
  const vis = markers.filter((m) => m.t >= t0 && m.t <= now && lane(m.kind) >= 0)
  const truth = vis.filter((m) => m.kind === 'truth')
  const firstOf = truth.filter((m, i) => !truth.slice(0, i).some((o) => m.t - o.t < 3))
  return (
    <div className="race">
      <svg viewBox={`0 0 ${w} ${h}`}>
        {LANES.map(([k, label]) => (
          <g key={k}>
            <text x={0} y={y(k) + 4} className="lane-label">{label}</text>
            <line x1={lw} x2={w - 70} y1={y(k)} y2={y(k)} className="lane" />
          </g>
        ))}
        {[0, 30, 60, 90].map((s) => <text key={s} x={x(now - s)} y={h - 4} className="tick">{s ? `-${s}s` : 'now'}</text>)}
        {truth.map((m, i) => (
          <g key={`t${m.t}`}>
            <line x1={x(m.t)} x2={x(m.t)} y1={i % 2 ? 30 : 14} y2={h - 18} className="truth-line" />
            <text x={x(m.t) + (i % 2 ? 0 : 0)} y={i % 2 ? 28 : 12} className="ev-label">{m.event}{m.team ? ` · ${m.team}` : ''}</text>
          </g>
        ))}
        {vis.map((m, i) => <circle key={i} cx={x(m.t)} cy={y(m.kind)} r={9} className={`dot ${m.kind}`} />)}
        {firstOf.map((m) => {
          const feed = vis.find((f) => f.kind === 'feed' && f.event === m.event && f.t >= m.t)
          const entry = vis.find((f) => f.kind === 'cs_entry' && Math.abs(f.t - m.t) < 30)
          return (
            <g key={`d${m.t}`}>
              {entry && <text x={x(entry.t) + 14} y={y('cs_entry') + 7} className="delta cs">+{(entry.t - m.t).toFixed(1)}s</text>}
              {feed && <text x={x(feed.t) + 14} y={y('feed') + 7} className="delta feed">+{(feed.t - m.t).toFixed(1)}s</text>}
            </g>
          )
        })}
      </svg>
    </div>
  )
}

// ---- order books -----------------------------------------------------------------------------

export function Books({ books, fair, trades }: {
  books: Record<string, { bids: Level[]; asks: Level[] }>; fair: Record<string, number>; trades: TradeMsg[]
}) {
  const maxQ = Math.max(20, ...Object.values(books).flatMap((b) => [...b.bids, ...b.asks].map(([, q]) => q)))
  const label: Record<string, string> = { HOME: 'Home win', DRAW: 'Draw', AWAY: 'Away win', OVER25: 'Over 2.5', BTTS: 'Both score' }
  return (
    <div className="books">
      {CONTRACTS.map((c) => {
        const b = books[c] ?? { bids: [], asks: [] }
        const last = [...trades].reverse().find((t) => t.instrument === c)
        const bot = last && (last.buyer === 'cs' || last.seller === 'cs')
        return (
          <div key={c} className="ladder">
            <div className="ladder-head"><b>{label[c]}</b><span>fair</span></div>
            <div className="fair">{fair[c]?.toFixed(1) ?? '–'}</div>
            <div className={`last ${bot ? 'cs' : ''}`}>{last ? `last ${last.price}${bot ? ' · bot' : ''}` : 'no trades yet'}</div>
            {[...b.asks].slice(0, 3).reverse().map(([p, q]) => (
              <div key={`a${p}`} className="lvl ask">
                <i className="depth" style={{ width: `${(q / maxQ) * 100}%` }} />
                <span className="px">{p}</span><span className="q">{q}</span>
              </div>
            ))}
            <div className="spread">SPREAD</div>
            {[...b.bids].slice(0, 3).map(([p, q]) => (
              <div key={`b${p}`} className="lvl bid">
                <i className="depth" style={{ width: `${(q / maxQ) * 100}%` }} />
                <span className="px">{p}</span><span className="q">{q}</span>
              </div>
            ))}
          </div>
        )
      })}
    </div>
  )
}

// ---- P&L -------------------------------------------------------------------------------------

export function PnlChart({ pnl }: { pnl: PnlMsg[] }) {
  const w = 360, h = 130
  const { pathCs, pathMm, zero } = useMemo(() => {
    if (pnl.length < 2) return { pathCs: '', pathMm: '', zero: h / 2 }
    const vals = pnl.flatMap((p) => [p.cs, p.mm])
    const lo = Math.min(0, ...vals), hi = Math.max(1, ...vals)
    const t0 = pnl[0].t, t1 = pnl[pnl.length - 1].t
    const sx = (t: number) => ((t - t0) / Math.max(1, t1 - t0)) * w
    const sy = (v: number) => h - 4 - ((v - lo) / (hi - lo)) * (h - 8)
    const path = (k: 'cs' | 'mm') => pnl.map((p, i) => `${i ? 'L' : 'M'}${sx(p.t)},${sy(p[k])}`).join(' ')
    return { pathCs: path('cs'), pathMm: path('mm'), zero: sy(0) }
  }, [pnl])
  const last = pnl[pnl.length - 1]
  return (
    <div className="pnl">
      <div className="pnl-head">
        <div className="cs"><span className="k">Commentary bot</span><span className="v">${((last?.cs ?? 0) / 100).toFixed(2)}</span></div>
        <div className="mm" style={{ textAlign: 'right' }}><span className="k">Market maker</span><span className="v">${((last?.mm ?? 0) / 100).toFixed(2)}</span></div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`}>
        <line x1={0} x2={w} y1={zero} y2={zero} className="zero" />
        <path d={pathMm} className="mm" />
        <path d={pathCs} className="cs" />
      </svg>
      <div className="note">Marked to the model's fair value · $1 per contract per 100 points · play money</div>
    </div>
  )
}

// ---- headline results ------------------------------------------------------------------------

export function useResults() {
  const [r, setR] = useState<any>(null)
  useEffect(() => { fetch(RESULTS_URL).then((x) => x.json()).then(setR).catch(() => setR(null)) }, [])
  return r
}

export function Kpis({ r }: { r: any }) {
  if (!r) return null
  const goal = (r.text_eval?.detection_test ?? []).find((d: any) => d.event === 'goal' && d.method === r.text_eval.fusion_selected)
  const gain = r.market?.paired_dollars_per_match?.['cs_gain_feed_12s_vs_1s']
  return (
    <div className="kpis">
      {gain && <div className="kpi"><b>+${gain.mean.toFixed(2)}</b><span>per match vs a TV-delayed market (±{gain.sem.toFixed(2)}, 75 matches)</span></div>}
      {r.real_audio && <div className="kpi"><b>{r.real_audio.roar_goal_delay_p50.toFixed(1)}s</b><span>crowd roar after a goal on real broadcasts, vs {r.real_audio.text_goal_delay_p50.toFixed(1)}s for words</span></div>}
      {goal && <div className="kpi"><b>{Math.round(goal.precision * 100)}%</b><span>precision of goal calls from commentary text ({Math.round(goal.recall * 100)}% recall)</span></div>}
      {r.bench && <div className="kpi"><b>{r.bench.exchange.p50_us.toFixed(1)}µs</b><span>median matching latency of the Python exchange</span></div>}
    </div>
  )
}

export function EdgeTable({ r }: { r: any }) {
  const paired = r?.market?.paired_dollars_per_match ?? {}
  const rows = Object.entries(paired).filter(([k]) => k.startsWith('cs_gain')) as [string, any][]
  if (!rows.length) return null
  const max = Math.max(...rows.map(([, v]) => v.mean), 1)
  return (
    <table className="edge-table">
      <thead><tr><th>Market's feed delay</th><th>Bot's gain vs a 1 s feed ($ / match)</th><th style={{ width: '40%' }} /></tr></thead>
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td>{k.match(/feed_([\d.]+)s/)?.[1]} s</td>
            <td>{v.mean >= 0 ? '+' : ''}{v.mean.toFixed(2)} <span style={{ color: 'var(--faint)' }}>± {v.sem.toFixed(2)}</span></td>
            <td><div className="edge-bar" style={{ width: `${Math.max(0, v.mean / max) * 100}%` }} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
