/** Where data comes from: the FastAPI backend locally, or pre-computed JSON on GitHub Pages. */
export const STATIC = import.meta.env.VITE_STATIC === '1'
export const BASE = import.meta.env.BASE_URL
export const RESULTS_URL = STATIC ? `${BASE}demo/results.json` : '/api/results'
export const REPO_URL = 'https://github.com/srishti-h/audio-courtsider'

export const demoFile = (match: string, goal: number, feedDelay: number, guard: boolean) =>
  `${BASE}demo/${match}__goal${goal}__fd${feedDelay}__${guard ? 'guard' : 'plain'}.json`
