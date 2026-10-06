import { useEffect, useRef, useState } from 'react'
import type { BookMsg, MarkerMsg, Meta, PnlMsg, ReplayParams, SignalMsg, TradeMsg, Level } from './types'

export interface ReplayState {
  status: 'idle' | 'loading' | 'playing' | 'ended' | 'error'
  meta?: Meta
  now: number
  half: number
  clock: number
  signals: SignalMsg[]
  books: Record<string, { bids: Level[]; asks: Level[] }>
  fair: Record<string, number>
  trades: TradeMsg[]
  markers: MarkerMsg[]
  pnl: PnlMsg[]
  score: [number, number]
}

const initial: ReplayState = {
  status: 'idle', now: 0, half: 1, clock: 0, signals: [], books: {}, fair: {}, trades: [], markers: [], pnl: [],
  score: [0, 0],
}

function apply(s: ReplayState, msgs: any[]): ReplayState {
  const next = { ...s, books: { ...s.books }, signals: s.signals, trades: s.trades, markers: s.markers, pnl: s.pnl }
  let signals = s.signals, trades = s.trades, markers = s.markers, pnl = s.pnl
  for (const m of msgs) {
    if (m.kind === 'meta') { next.meta = m; next.status = 'playing'; continue }
    if (m.kind === 'end') { next.status = 'ended'; continue }
    if (typeof m.t === 'number') { next.now = m.t; next.half = m.half; next.clock = m.clock }
    switch (m.kind) {
      case 'signal': signals = [...signals.slice(-399), m]; break
      case 'book': next.books[(m as BookMsg).instrument] = { bids: m.bids, asks: m.asks }; break
      case 'fair': next.fair = m.prices; break
      case 'trade': trades = [...trades.slice(-29), m]; break
      case 'pnl': pnl = [...pnl, m]; break
      case 'feed':
        if (m.event === 'goal') next.score = m.team === 'home' ? [next.score[0] + 1, next.score[1]] : [next.score[0], next.score[1] + 1]
        markers = [...markers, m]; break
      case 'truth': case 'cs_entry': case 'cs_exit': case 'mm_guard': markers = [...markers, m]; break
    }
  }
  return { ...next, signals, trades, markers, pnl }
}

export function useReplay(params: ReplayParams | null): ReplayState {
  const [state, setState] = useState<ReplayState>(initial)
  const buf = useRef<any[]>([])
  useEffect(() => {
    if (!params) return
    setState({ ...initial, status: 'loading' })
    const q = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]))
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/replay?${q}`)
    ws.onmessage = (e) => { buf.current.push(JSON.parse(e.data)) }
    ws.onerror = () => setState((s) => ({ ...s, status: 'error' }))
    let raf = 0
    const flush = () => {
      if (buf.current.length) {
        const msgs = buf.current
        buf.current = []
        setState((s) => apply(s, msgs))
      }
      raf = requestAnimationFrame(flush)
    }
    raf = requestAnimationFrame(flush)
    return () => { ws.close(); cancelAnimationFrame(raf); buf.current = [] }
  }, [params])
  return state
}
