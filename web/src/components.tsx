import { useEffect, useMemo, useRef } from 'react'
import type { MarkerMsg, PnlMsg, SignalMsg, TradeMsg, Level } from './types'
import { CONTRACTS } from './types'

const fmtClock = (half: number, clock: number) => {
  const m = Math.floor(clock / 60), s = Math.floor(clock % 60)
  return `${half === 1 ? '1st' : '2nd'} ${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}
export { fmtClock }

const KEYWORDS = /\b(goa+l+|scores?|penalt\w*|red card|sent off|corner|yellow|offside|shot|saves?d?)\b/gi

export function Transcript({ signals }: { signals: SignalMsg[] }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => { ref.current?.scrollTo({ top: ref.current.scrollHeight }) }, [signals.length])
  return (
    <div className="panel transcript" ref={ref}>
      {signals.slice(-60).map((s, i) => {
        const pg = s.probs.goal ?? 0
        const hot = Math.max(pg, s.probs.penalty ?? 0, s.probs.red_card ?? 0)
        return (
          <div key={`${s.t}-${i}`} className={`line ${hot >= 0.5 ? 'hot' : ''}`}>
            <span className="ts">{fmtClock(s.half, s.clock)}</span>
            <span className="txt" dangerouslySetInnerHTML={{ __html: escapeHtml(s.text).replace(KEYWORDS, '<mark>$1</mark>') }} />
            {hot >= 0.2 && <span className="badge">{Math.round(hot * 100)}%</span>}
          </div>
        )
      })}
    </div>
  )
}

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!))
}

export function Meters({ signals, now }: { signals: SignalMsg[]; now: number }) {
  const last = signals[signals.length - 1]
  const window = signals.filter((s) => s.t > now - 180)
  const rows: [string, string][] = [['goal', 'P(goal)'], ['penalty', 'P(penalty)'], ['red_card', 'P(red card)']]
  return (
    <div className="panel meters">
      {rows.map(([k, label]) => {
        const v = last?.probs[k] ?? 0
        return (
          <div key={k} className="meter">
            <div className="meter-head"><span>{label}</span><b>{(v * 100).toFixed(0)}%</b></div>
            <div className="bar"><div className={`fill ${v >= 0.5 ? 'alert' : ''}`} style={{ width: `${v * 100}%` }} /></div>
            <Spark points={window.map((s) => [s.t, s.probs[k] ?? 0])} t0={now - 180} t1={now} />
          </div>
        )
      })}
      {last && (
        <div className="team">
          team: home {Math.round((last.team.home ?? 0.5) * 100)}% · away {Math.round((last.team.away ?? 0.5) * 100)}%
        </div>
      )}
    </div>
  )
}

function Spark({ points, t0, t1 }: { points: [number, number][]; t0: number; t1: number }) {
  const w = 260, h = 34
  const d = points.map(([t, v], i) => `${i ? 'L' : 'M'}${((t - t0) / (t1 - t0)) * w},${h - v * h}`).join(' ')
  return (
    <svg width={w} height={h} className="spark">
      <line x1={0} x2={w} y1={h / 2} y2={h / 2} className="mid" />
      <path d={d} />
    </svg>
  )
}

export function Books({ books, fair, trades }: {
  books: Record<string, { bids: Level[]; asks: Level[] }>; fair: Record<string, number>; trades: TradeMsg[]
}) {
  return (
    <div className="panel books">
      {CONTRACTS.map((c) => {
        const b = books[c] ?? { bids: [], asks: [] }
        const last = [...trades].reverse().find((t) => t.instrument === c)
        return (
          <div key={c} className="ladder">
            <div className="ladder-head">
              <b>{c}</b>
              <span>fair {fair[c]?.toFixed(1) ?? '–'}</span>
              {last && <span className={`last ${last.buyer === 'cs' || last.seller === 'cs' ? 'cs' : ''}`}>last {last.price}</span>}
            </div>
            <table>
              <tbody>
                {[...b.asks].slice(0, 3).reverse().map(([p, q]) => (
                  <tr key={`a${p}`} className="ask"><td /><td>{p}</td><td>{q}</td></tr>
                ))}
                {[...b.bids].slice(0, 3).map(([p, q]) => (
                  <tr key={`b${p}`} className="bid"><td>{q}</td><td>{p}</td><td /></tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      })}
    </div>
  )
}

const MARK_ROWS: Record<string, number> = { truth: 0, cs_entry: 1, mm_guard: 1, feed: 2, cs_exit: 3 }
const MARK_LABEL: Record<string, string> = { truth: 'pitch', cs_entry: 'courtsider', feed: 'official feed', cs_exit: 'exit' }

export function Timeline({ markers, now, span = 60 }: { markers: MarkerMsg[]; now: number; span?: number }) {
  const w = 760, h = 120, t0 = now - span
  const x = (t: number) => ((t - t0) / span) * w
  const visible = markers.filter((m) => m.t >= t0 && m.t <= now && m.kind in MARK_ROWS)
  const truth = visible.filter((m) => m.kind === 'truth')
  return (
    <div className="panel timeline">
      <svg width="100%" viewBox={`0 0 ${w + 110} ${h}`}>
        {Object.entries(MARK_LABEL).map(([k, label]) => (
          <text key={k} x={0} y={18 + MARK_ROWS[k] * 26} className="row-label">{label}</text>
        ))}
        <g transform="translate(110,0)">
          {[0, 1, 2, 3].map((r) => <line key={r} x1={0} x2={w} y1={14 + r * 26} y2={14 + r * 26} className="row" />)}
          {truth.map((m) => (
            <line key={`v${m.t}`} x1={x(m.t)} x2={x(m.t)} y1={4} y2={h - 10} className="truth-line" />
          ))}
          {visible.map((m, i) => (
            <g key={i} transform={`translate(${x(m.t)},${14 + MARK_ROWS[m.kind] * 26})`}>
              <circle r={6} className={`mk ${m.kind}`} />
              {m.event && <text y={-9} className="mk-label">{m.event}{m.team ? ` (${m.team})` : ''}</text>}
            </g>
          ))}
          {truth.map((m) => {
            const feed = visible.find((f) => f.kind === 'feed' && f.event === m.event && f.t >= m.t)
            const entry = visible.find((f) => f.kind === 'cs_entry' && Math.abs(f.t - m.t) < 30)
            return (
              <g key={`gap${m.t}`}>
                {feed && <text x={x(feed.t) + 8} y={14 + 2 * 26 + 4} className="gap">+{(feed.t - m.t).toFixed(1)}s</text>}
                {entry && <text x={x(entry.t) + 8} y={14 + 26 + 4} className="gap cs">+{(entry.t - m.t).toFixed(1)}s</text>}
              </g>
            )
          })}
        </g>
      </svg>
    </div>
  )
}

export function PnlChart({ pnl }: { pnl: PnlMsg[] }) {
  const w = 360, h = 140
  const { pathCs, pathMm, lo, hi } = useMemo(() => {
    if (pnl.length < 2) return { pathCs: '', pathMm: '', lo: 0, hi: 1 }
    const vals = pnl.flatMap((p) => [p.cs, p.mm]).map((v) => v / 100)
    const lo = Math.min(0, ...vals), hi = Math.max(1, ...vals)
    const t0 = pnl[0].t, t1 = pnl[pnl.length - 1].t
    const sx = (t: number) => ((t - t0) / Math.max(1, t1 - t0)) * w
    const sy = (v: number) => h - ((v - lo) / (hi - lo)) * h
    const path = (k: 'cs' | 'mm') => pnl.map((p, i) => `${i ? 'L' : 'M'}${sx(p.t)},${sy(p[k] / 100)}`).join(' ')
    return { pathCs: path('cs'), pathMm: path('mm'), lo, hi }
  }, [pnl])
  const last = pnl[pnl.length - 1]
  return (
    <div className="panel pnl">
      <div className="pnl-head">
        <span className="cs">courtsider ${((last?.cs ?? 0) / 100).toFixed(2)}</span>
        <span className="mm">market maker ${((last?.mm ?? 0) / 100).toFixed(2)}</span>
      </div>
      <svg width="100%" viewBox={`0 0 ${w} ${h}`}>
        <line x1={0} x2={w} y1={h - ((0 - lo) / (hi - lo)) * h} y2={h - ((0 - lo) / (hi - lo)) * h} className="zero" />
        <path d={pathMm} className="mm" />
        <path d={pathCs} className="cs" />
      </svg>
      <div className="note">mark-to-model, $1 per contract per 100 points</div>
    </div>
  )
}
