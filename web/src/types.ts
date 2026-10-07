export type Level = [number, number]

export interface Game {
  game_id: string
  label: string
  league: string
  date: string
  goals: number[]
}

export interface TapeBase {
  kind: string
  t: number
  half: number
  clock: number
}

export interface SignalMsg extends TapeBase {
  kind: 'signal'
  probs: Record<string, number>
  team: Record<string, number>
  text: string
}

export interface BookMsg extends TapeBase {
  kind: 'book'
  instrument: string
  bids: Level[]
  asks: Level[]
}

export interface TradeMsg extends TapeBase {
  kind: 'trade'
  instrument: string
  price: number
  qty: number
  buyer: string
  seller: string
}

export interface PnlMsg extends TapeBase {
  kind: 'pnl'
  minute: number
  cs: number
  mm: number
}

export interface MarkerMsg extends TapeBase {
  kind: 'truth' | 'feed' | 'cs_entry' | 'cs_exit' | 'mm_guard'
  event?: string
  team?: string
  p?: number
}

export interface Meta {
  kind: 'meta'
  game_id: string
  home: string
  away: string
  latency: number
  summary: { pnl: Record<string, number>; cs_entries: number; cs_volume: number }
}

export interface ReplayParams {
  file?: string // static mode: URL of a pre-computed tape
  game_id: string
  speed: number
  feed_delay: number
  cs_threshold: number
  mm_audio_guard: boolean
  start: number
}

export const CONTRACTS = ['HOME', 'DRAW', 'AWAY', 'OVER25', 'BTTS'] as const

export interface DemoMatch {
  id: string
  label: string
  date: string
  home: string
  away: string
  goals: { n: number; minute: number; team: string }[]
}
