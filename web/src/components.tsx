import { useEffect, useMemo, useRef, useState } from 'react'
import type { MarkerMsg, PnlMsg, SignalMsg, TradeMsg, Level } from './types'
import { CONTRACTS } from './types'
import { RESULTS_URL } from './source'

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
const MARK_LABEL: Record<string, string> = { truth: 'pitch', cs_entry: 'courtsider', feed: 'official feed', cs_exit: 'courtsider exit' }
const ROW_H = 40

export function Timeline({ markers, now, span = 90 }: { markers: MarkerMsg[]; now: number; span?: number }) {
  const w = 600, lw = 130, h = 4 * ROW_H + 40, t0 = now - span
  const x = (t: number) => ((t - t0) / span) * w
  const y = (k: string) => 44 + MARK_ROWS[k] * ROW_H
  const visible = markers.filter((m) => m.t >= t0 && m.t <= now && m.kind in MARK_ROWS)
  const truth = visible.filter((m) => m.kind === 'truth')
  return (
    <div className="panel timeline">
      <svg width="100%" viewBox={`0 0 ${w + lw + 50} ${h}`}>
        {Object.entries(MARK_LABEL).map(([k, label]) => (
          <text key={k} x={0} y={y(k) + 5} className="row-label">{label}</text>
        ))}
        <g transform={`translate(${lw},0)`}>
          {Object.keys(MARK_LABEL).map((k) => <line key={k} x1={0} x2={w} y1={y(k)} y2={y(k)} className="row" />)}
          {[0, 30, 60, 90].map((s) => (
            <text key={s} x={x(now - s)} y={h - 2} className="tick">{s ? `-${s}s` : 'now'}</text>
          ))}
          {truth.map((m, i) => (
            <g key={`v${m.t}`}>
              <line x1={x(m.t)} x2={x(m.t)} y1={i % 2 ? 26 : 12} y2={h - 18} className="truth-line" />
              <text x={x(m.t)} y={i % 2 ? 28 : 13} className="mk-label">{m.event}{m.team ? ` · ${m.team}` : ''}</text>
            </g>
          ))}
          {visible.map((m, i) => (
            <circle key={i} cx={x(m.t)} cy={y(m.kind)} r={7} className={`mk ${m.kind}`} />
          ))}
          {truth.filter((m, i) => !truth.slice(0, i).some((o) => m.t - o.t < 3)).map((m) => {
            const feed = visible.find((f) => f.kind === 'feed' && f.event === m.event && f.t >= m.t)
            const entry = visible.find((f) => f.kind === 'cs_entry' && Math.abs(f.t - m.t) < 30)
            return (
              <g key={`gap${m.t}`}>
                {feed && <text x={x(feed.t) + 11} y={y('feed') + 5} className="gap">+{(feed.t - m.t).toFixed(1)}s</text>}
                {entry && <text x={x(entry.t) + 11} y={y('cs_entry') + 5} className="gap cs">+{(entry.t - m.t).toFixed(1)}s</text>}
              </g>
            )
          })}
        </g>
      </svg>
    </div>
  )
}

export function ResultsPanel() {
  const [r, setR] = useState<any>(null)
  useEffect(() => { fetch(RESULTS_URL).then((x) => x.json()).then(setR).catch(() => setR(null)) }, [])
  if (!r?.market) return null
  const paired = r.market.paired_dollars_per_match ?? {}
  const gains = Object.entries(paired).filter(([k]) => k.startsWith('cs_gain')) as [string, any][]
  const goal = (r.text_eval?.detection_test ?? []).find((d: any) => d.event === 'goal' && d.method === r.text_eval.fusion_selected)
  return (
    <div className="panel results">
      <div className="kpis">
        {goal && <div><b>{Math.round(goal.precision * 100)}% / {Math.round(goal.recall * 100)}%</b><span>goal precision / recall from commentary text (test)</span></div>}
        {goal && <div><b>{goal.delay_p50.toFixed(1)} s</b><span>median detection after the goal, incl. ASR latency</span></div>}
        {r.real_audio ? (
          <div><b>{r.real_audio.roar_goal_delay_p50.toFixed(1)} s vs {r.real_audio.text_goal_delay_p50.toFixed(1)} s</b><span>crowd roar vs commentary after a goal, real broadcast audio</span></div>
        ) : r.asr && <div><b>{r.asr.pipeline_latency_s.toFixed(2)} s</b><span>speech → text segment ({r.asr.pipeline_model}, synthetic audio)</span></div>}
        {r.bench && <div><b>{r.bench.exchange.p50_us.toFixed(1)} µs</b><span>exchange matching latency p50 (Python)</span></div>}
      </div>
      <table>
        <thead><tr><th>market's feed delay</th><th>courtsider gain vs a 1 s feed ($ / match, paired)</th></tr></thead>
        <tbody>
          {gains.map(([k, v]) => (
            <tr key={k}><td>{k.match(/feed_([\d.]+)s/)?.[1]} s</td><td>{v.mean >= 0 ? '+' : ''}{v.mean.toFixed(2)} ± {v.sem.toFixed(2)}</td></tr>
          ))}
        </tbody>
      </table>
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
