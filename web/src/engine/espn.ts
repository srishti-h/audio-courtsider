/** ESPN's public soccer API from the browser (it allows cross-origin requests): port of live/feeds.py. */

const ESPN = 'https://site.api.espn.com/apis/site/v2/sports/soccer/all'

export interface FeedMatch { id: string; name: string; state: string; clock: string; home: string; away: string; score: [number, number] }
export interface Snapshot { home: string; away: string; score: [number, number]; clock: string; state: string; reds: [number, number] }

export async function matches(): Promise<FeedMatch[]> {
  const d = await (await fetch(`${ESPN}/scoreboard`)).json()
  const order: Record<string, number> = { in: 0, pre: 1, post: 2 }
  return (d.events ?? [])
    .map((e: any) => {
      const t = Object.fromEntries(e.competitions[0].competitors.map((x: any) => [x.homeAway, x]))
      return {
        id: e.id, name: e.name, state: e.status.type.state, clock: e.status.displayClock ?? '', date: e.date,
        home: t.home.team.displayName, away: t.away.team.displayName,
        score: [Number(t.home.score ?? 0), Number(t.away.score ?? 0)] as [number, number],
      }
    })
    .sort((a: any, b: any) => (order[a.state] ?? 3) - (order[b.state] ?? 3) || a.date.localeCompare(b.date))
    .slice(0, 60)
}

export async function snapshot(id: string): Promise<Snapshot> {
  const d = await (await fetch(`${ESPN}/summary?event=${encodeURIComponent(id)}`)).json()
  const comp = d.header.competitions[0]
  const t = Object.fromEntries(comp.competitors.map((x: any) => [x.homeAway, x]))
  const side = Object.fromEntries(comp.competitors.map((x: any) => [x.team.id, x.homeAway]))
  const reds = { home: 0, away: 0 } as Record<string, number>
  for (const k of d.keyEvents ?? []) {
    const s = side[k.team?.id]
    if ((k.type?.text ?? '').includes('Red Card') && s) reds[s] += 1
  }
  return {
    home: t.home.team.displayName, away: t.away.team.displayName,
    score: [Number(t.home.score ?? 0), Number(t.away.score ?? 0)], clock: comp.status?.displayClock ?? '',
    state: comp.status?.type?.state ?? '', reds: [reds.home, reds.away],
  }
}

/** Feed events implied by two consecutive snapshots (goals and red cards), like feeds.diff(). */
export function diff(prev: Snapshot, cur: Snapshot, wall: number): any[] {
  const out: any[] = []
  ;(['home', 'away'] as const).forEach((team, i) => {
    for (let k = 0; k < cur.score[i] - prev.score[i]; k++) out.push({ kind: 'feed', event: 'goal', team, score: cur.score, clock: cur.clock, wall })
    for (let k = 0; k < cur.reds[i] - prev.reds[i]; k++) out.push({ kind: 'feed', event: 'red_card', team, score: cur.score, clock: cur.clock, wall })
  })
  return out
}
