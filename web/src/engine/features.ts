/** Browser port of acoustic.extract.log_mel + acoustic.goal_model.normalise (must match librosa exactly). */

export interface MelConfig { n_fft: number; hop: number; n_mels: number; baseline_s: number; sr: number }

const FPS = 50

function hann(n: number): Float32Array {
  const w = new Float32Array(n)
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n) // periodic, as librosa
  return w
}

/** In-place iterative radix-2 FFT of (re, im); n must be a power of two. */
function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j], re[i]]
      ;[im[i], im[j]] = [im[j], im[i]]
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(ang * k), wi = Math.sin(ang * k)
        const a = i + k, b = a + len / 2
        const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr
        re[b] = re[a] - tr; im[b] = im[a] - ti
        re[a] += tr; im[a] += ti
      }
    }
  }
}

/** (frames x n_mels) natural-log mel power, centre=False, like log_mel() in Python. */
export function logMel(y: Float32Array, fb: Float32Array, c: MelConfig): Float32Array[] {
  const bins = c.n_fft / 2 + 1, win = hann(c.n_fft)
  const frames = y.length < c.n_fft ? 0 : 1 + Math.floor((y.length - c.n_fft) / c.hop)
  const out: Float32Array[] = []
  const re = new Float64Array(c.n_fft), im = new Float64Array(c.n_fft), pow = new Float64Array(bins)
  for (let f = 0; f < frames; f++) {
    const s = f * c.hop
    for (let i = 0; i < c.n_fft; i++) { re[i] = y[s + i] * win[i]; im[i] = 0 }
    fft(re, im)
    for (let k = 0; k < bins; k++) pow[k] = re[k] * re[k] + im[k] * im[k]
    const m = new Float32Array(c.n_mels)
    for (let b = 0; b < c.n_mels; b++) {
      let acc = 0
      const row = b * bins
      for (let k = 0; k < bins; k++) acc += fb[row + k] * pow[k]
      m[b] = Math.log(acc + 1e-6)
    }
    out.push(m)
  }
  return out
}

/** Causal 30 s EMA normalisation -> flat (frames x 2*n_mels) input for the TCN. */
export function normalise(mel: Float32Array[], c: MelConfig): Float32Array {
  const n = mel.length, d = c.n_mels, alpha = 1 / (c.baseline_s * FPS)
  const out = new Float32Array(n * d * 2)
  const ema = Float32Array.from(mel[0] ?? [])
  for (let i = 0; i < n; i++) {
    const x = mel[i]
    for (let b = 0; b < d; b++) {
      if (i > 0) ema[b] = ema[b] + alpha * (x[b] - ema[b])
      out[i * 2 * d + b] = (x[b] - ema[b]) / 2
      out[i * 2 * d + d + b] = (x[b] + 8) / 6
    }
  }
  return out
}

/** Isotonic calibrator: piecewise-linear interpolation clipped at the ends (sklearn, out_of_bounds="clip"). */
export function calibrate(raw: number, x: number[], y: number[]): number {
  if (raw <= x[0]) return y[0]
  if (raw >= x[x.length - 1]) return y[y.length - 1]
  let lo = 0, hi = x.length - 1
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (x[mid] <= raw) lo = mid; else hi = mid }
  return x[hi] === x[lo] ? y[hi] : y[lo] + ((raw - x[lo]) * (y[hi] - y[lo])) / (x[hi] - x[lo])
}

/** The deterministic test signal from scripts/export_web_models.py. */
export function selftestSignal(sr: number, seconds = 3): Float32Array {
  const y = new Float32Array(Math.floor(seconds * sr))
  for (let i = 0; i < y.length; i++) {
    const t = i / sr
    y[i] = 0.5 * Math.sin(2 * Math.PI * 440 * t) + 0.3 * Math.sin(2 * Math.PI * 3000 * t) * (t % 1)
  }
  return y
}
