import { useEffect, useLayoutEffect, useRef, useState } from 'react'

/** Explanations keyed by the `data-explain` attribute on page elements. */
export const EXPLAIN: Record<string, { title: string; body: string }> = {
  project: {
    title: 'What is Audio Courtsider?',
    body: 'A "courtsider" sits in the stadium and bets the moment something happens, before the official data feed tells the market. This project does that by listening instead: it turns live commentary and crowd noise into calibrated beliefs about goals, penalties and red cards, then trades on them in a simulated exchange against a market maker that only reprices when the official feed arrives. The question it answers: how many seconds of edge does listening buy, and what is that worth?',
  },
  tabs: {
    title: 'Replay vs Live',
    body: 'Replay re-runs real goals from held-out SoccerNet matches through the whole system: real commentary, real event times, simulated prices. Live listens to a match happening right now (browser tab, microphone or audio file) and times every detection against ESPN\'s live-score feed. Live needs speech recognition on your own machine, so on this public site it only explains itself.',
  },
  github: { title: 'Source code', body: 'Everything is open: data pipeline, models, exchange, simulator, evaluations and this dashboard, with the full methods and results in the README.' },
  timeline: {
    title: 'Who finds out first',
    body: 'The information gap the project measures. The crowd and commentator react within seconds; the official data feed that betting markets use lags a few seconds behind; TV and streaming viewers are 5–40 s behind. Anyone between the first and the third can trade on news the market hasn\'t priced yet.',
  },
  watch: { title: 'Watch a goal', body: 'Starts the showcase replay: a real goal, with the market\'s feed 8 s behind. You will see the bot hear the goal, trade, and the market catch up.' },
  kpis: {
    title: 'Headline results',
    body: 'Measured on held-out test matches the models never trained on. Time to hear a goal from audio alone (learned crowd + commentator model), what the audio bot earns per match against a market 5 s behind the broadcast, how fast the commentary-text path detects goals on real broadcasts, and how fast the exchange matches an order.',
  },
  edge: {
    title: 'How the edge grows with feed delay',
    body: 'Paired comparison over 75 test matches: the same bot, the same matches, only the market\'s feed delay changes. The slower the market learns about a goal, the more the commentary-text bot makes compared to a fast 1 s feed. ± is the standard error.',
  },
  match: { title: 'Pick a match', body: 'Five held-out test matches with 21 goals between them. Each replay is a window around one real goal.' },
  feed: {
    title: 'Market\'s feed delay',
    body: 'How long after the goal the market maker finds out. 1 s is a fast professional data feed: the bot\'s edge mostly disappears. 8 s is roughly a TV-delayed feed: the bot gets several seconds to trade at stale prices. Flip it and replay the same goal to see the difference.',
  },
  guard: {
    title: 'MM listens too',
    body: 'The defence. The market maker runs the same audio detector and pulls its quotes the moment it hears a goal, instead of waiting for the feed. Turn it on to see how cheaply a market can protect itself against courtsiders.',
  },
  speed: { title: 'Playback speed', body: 'Replays stream in simulated time; 2× means one real second plays in half a second.' },
  goals: { title: 'Jump to a goal', body: 'Each button replays the window around one real goal: its minute and which side scored. Links are shareable: the URL remembers match, goal, feed delay and speed.' },
  scoreboard: {
    title: 'Scoreboard',
    body: 'The score and clock as the official feed reports them, so it updates late on purpose. In the gap between the goal and this number changing, the market is trading on old information.',
  },
  race: {
    title: 'The race: who knew first',
    body: 'One lane per participant over the last 90 s. Gold: the event actually happens on the pitch. Red: the commentary bot hears it and trades, with how many seconds after the event. Blue: the official feed reports it and the market maker reprices. The horizontal gap between red and blue is the edge.',
  },
  commentary: {
    title: 'Commentary',
    body: 'The real broadcast commentary (SoccerNet-Echoes, transcribed by Whisper) appearing as it is spoken. Key words like goal and penalty are highlighted, and the badge is the language model\'s confidence that this line describes a goal, penalty or red card.',
  },
  beliefs: {
    title: 'What the bot believes',
    body: 'A fine-tuned DistilRoBERTa model reads each line and outputs calibrated probabilities: P(goal) solid, P(penalty) dashed, P(red card) blue. Calibrated means 30% really happens about 30% of the time, so the bot can size its trades on it. Vertical markers show when the event happened and when the feed reported it.',
  },
  meters: {
    title: 'Belief meters and team split',
    body: 'The latest probabilities as bars, and below them which team the model thinks the event is for (it reads names and team words in the commentary). The bot only trades once a probability crosses its entry threshold, which was chosen on validation matches.',
  },
  books: {
    title: 'Order books',
    body: 'Five contracts on a real price-time-priority exchange: home win, draw, away win, over 2.5 goals, both teams score. Each pays 100 if it happens. "Fair" is the in-play Dixon–Coles model\'s price given the score, clock and red cards. Red rows are asks, blue are bids, with depth bars; "last · bot" marks a trade by the commentary bot.',
  },
  pnl: {
    title: 'Profit and loss',
    body: 'The commentary bot\'s and the market maker\'s P&L through the replay, marked to fair value. When the bot buys before the feed and the price jumps after, its line rises and the market maker\'s falls: that is adverse selection. Play money only.',
  },
  live: {
    title: 'Live mode',
    body: 'Runs the same models on a match happening now. Pick a live match from ESPN\'s scoreboard, point it at the commentary (a browser tab streaming the game, your microphone near the TV, or a saved file), and it shows the live transcript, beliefs, crowd meter, and how many seconds each detection beat the official feed by.',
  },
  'live-match': { title: 'Pick the match', body: 'Live matches from ESPN\'s public scoreboard, polled every 5 s. The lineups become Whisper\'s vocabulary so player names are spelled right. The gold numbers are the pricing model\'s pre-match home · draw · away probabilities, fit on results up to this week. You can also type the teams if the match isn\'t listed (no feed timing then).' },
  'live-audio': { title: 'Point it at the commentary', body: 'Audio is resampled to 16 kHz in the browser and streamed to the local server, where Whisper transcribes it in real time. No audio is stored; only text and detections are logged for the lead-time report. The threshold sets how confident the model must be before a detection fires.' },
  crowd: { title: 'Crowd meter', body: 'A hand-built signal: high-frequency crowd energy compared with the last minute or so. A roar 4 standard deviations above normal fires a detection. Fast, but it can\'t tell which team scored.' },
  'audio-model': { title: 'Audio goal model', body: 'A causal temporal CNN trained on the audio of 500 broadcasts. It outputs P(a goal in the last 6 s) 50 times a second and, on test matches, catches about two thirds of goals a median 2.4 s after they happen.' },
  'recorder-how': { title: 'Background recorder', body: 'A background job on the developer\'s Mac that records Spanish radio during every La Liga and Spanish-club European match, runs the same models live, and logs when each goal was heard versus when ESPN\'s live-score feed reported it. Nobody has to press anything; this page shows everything it has logged.' },
  'recorder-kpis': { title: 'Real-world lead times', body: 'How many goals each detector heard before the live-score feed, and by how much (median). These are real 2026 matches, not replays. The free ESPN feed is itself slow, so leads against a professional data feed would be smaller.' },
  'recorder-goals': { title: 'Every recorded goal', body: 'One row per goal the feed reported. A lead means that detector fired before the feed, counted only if it fired between 10 s before and 30 s after ESPN\'s time for the goal. A dash means it missed or was later than the feed. The quote is a short excerpt of the translated commentary.' },
  detections: { title: 'Detections vs the feed', body: 'Every detection from text (⚡ = fast speculative path), crowd and audio model, interleaved with feed events. When the feed reports a goal, the lead column shows how many seconds each source beat it by.' },
}

const TOUR = ['project', 'timeline', 'kpis', 'edge', 'match', 'feed', 'guard', 'goals', 'scoreboard', 'race', 'commentary', 'beliefs', 'meters', 'books', 'pnl', 'tabs']
const REPLAY_STEPS = new Set(['scoreboard', 'race', 'commentary', 'beliefs', 'meters', 'books', 'pnl'])
const SEEN_KEY = 'courtsider-guide-seen'

const find = (key: string) => document.querySelector<HTMLElement>(`[data-explain="${key}"]`)

function startReplay() {
  const btn = find('watch') ?? [...document.querySelectorAll<HTMLElement>('[data-explain="goals"] button')].pop()
  btn?.click()
}

type Open = { key: string; el: HTMLElement; at?: { x: number; y: number }; step?: number }

export function Guide({ onTour }: { onTour: () => void }) {
  const [open, setOpen] = useState<Open | null>(null)
  const [rect, setRect] = useState<DOMRect | null>(null)
  const [hint, setHint] = useState(false)
  const pop = useRef<HTMLDivElement>(null)
  const [popSize, setPopSize] = useState({ w: 340, h: 180 })

  useEffect(() => {
    try { setHint(!localStorage.getItem(SEEN_KEY)) } catch { setHint(true) }
  }, [])
  const dismissHint = () => {
    setHint(false)
    try { localStorage.setItem(SEEN_KEY, '1') } catch { /* private mode */ }
  }

  // right-click anything with data-explain; shift+right-click keeps the browser menu
  useEffect(() => {
    const onMenu = (e: MouseEvent) => {
      if (e.shiftKey) return
      const el = (e.target as HTMLElement).closest<HTMLElement>('[data-explain]')
      const key = el?.dataset.explain
      if (!el || !key || !EXPLAIN[key]) return
      e.preventDefault()
      dismissHint()
      const r = el.getBoundingClientRect()
      setOpen({ key, el, at: { x: e.clientX - r.left, y: e.clientY - r.top } }) // relative, so it follows scrolling
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(null) }
    document.addEventListener('contextmenu', onMenu)
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('contextmenu', onMenu); document.removeEventListener('keydown', onKey) }
  }, [])

  // follow the element while the page scrolls, resizes or re-renders
  useEffect(() => {
    if (!open) return
    let id = 0
    const tick = () => {
      const r = open.el.isConnected ? open.el.getBoundingClientRect() : null
      setRect((p) => (p && r && p.x === r.x && p.y === r.y && p.width === r.width && p.height === r.height ? p : r))
      id = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(id)
  }, [open])
  useLayoutEffect(() => {
    if (pop.current) setPopSize({ w: pop.current.offsetWidth, h: pop.current.offsetHeight })
  }, [open])

  const goto = async (step: number) => {
    if (step < 0 || step >= TOUR.length) return setOpen(null)
    const key = TOUR[step]
    let el = find(key)
    if (!el && REPLAY_STEPS.has(key)) {
      startReplay()
      for (let i = 0; i < 40 && !el; i++) {
        await new Promise((r) => setTimeout(r, 100))
        el = find(key)
      }
    }
    if (!el) return goto(step + 1) // not on this page (e.g. controls still loading): skip it
    el.scrollIntoView({ block: 'center', behavior: 'smooth' })
    setOpen({ key, el, step })
  }
  const startTour = () => {
    dismissHint()
    onTour()
    setTimeout(() => goto(0), 50)
  }
  useEffect(() => {
    const go = () => startTour()
    window.addEventListener('courtsider-tour', go)
    return () => window.removeEventListener('courtsider-tour', go)
  })

  const tour = open?.step !== undefined
  const pos = (() => {
    const vw = window.innerWidth, vh = window.innerHeight, pad = 12
    if (vw < 640) return { left: pad, top: vh - popSize.h - pad } // phones: pinned to the bottom
    let left: number, top: number
    if (open?.at && rect) {
      left = rect.left + open.at.x + 12; top = rect.top + open.at.y + 12
    } else if (rect) {
      left = rect.left; top = rect.bottom + 12
      if (top + popSize.h > vh - pad) top = rect.top - popSize.h - 12
      if (top < pad) top = Math.min(vh - popSize.h - pad, Math.max(pad, rect.top + 12)) // tall element: overlay its top
    } else { left = pad; top = pad }
    return { left: Math.max(pad, Math.min(left, vw - popSize.w - pad)), top: Math.max(pad, Math.min(top, vh - popSize.h - pad)) }
  })()

  return (
    <>
      <button className="btn ghost guide-btn" onClick={startTour}>? Tour</button>
      {hint && !open && (
        <div className="guide-hint" role="status">
          <b>New here?</b> Take the 60-second tour, or <b>right-click</b> any panel to see what it does.
          <div className="guide-actions">
            <button className="btn ghost" onClick={dismissHint}>Not now</button>
            <button className="btn gold" onClick={startTour}>Start the tour</button>
          </div>
        </div>
      )}
      {open && (
        <>
          <div className={`guide-scrim ${tour ? 'dim' : ''}`} onClick={() => setOpen(null)} onContextMenu={(e) => { e.preventDefault(); setOpen(null) }} />
          {rect && (
            <div className={`guide-ring ${tour ? 'dim' : ''}`}
              style={{ left: rect.left - 6, top: rect.top - 6, width: rect.width + 12, height: rect.height + 12 }} />
          )}
          <div ref={pop} className="guide-pop" role="dialog" aria-label={EXPLAIN[open.key].title} style={pos}>
            <div className="guide-stripe" />
            {tour && <div className="guide-step">Step {open.step! + 1} of {TOUR.length}</div>}
            <h3>{EXPLAIN[open.key].title}</h3>
            <p>{EXPLAIN[open.key].body}</p>
            <div className="guide-actions">
              {tour ? (
                <>
                  <button className="btn ghost" onClick={() => setOpen(null)}>End tour</button>
                  {open.step! > 0 && <button className="btn ghost" onClick={() => goto(open.step! - 1)}>← Back</button>}
                  <button className="btn" autoFocus onClick={() => goto(open.step! + 1)}>{open.step! + 1 < TOUR.length ? 'Next →' : 'Done'}</button>
                </>
              ) : (
                <>
                  <span className="guide-tip">Right-click anything else · Esc to close</span>
                  <button className="btn ghost" onClick={startTour}>Take the tour</button>
                </>
              )}
            </div>
          </div>
        </>
      )}
    </>
  )
}
