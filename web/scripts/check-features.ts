// node scripts/check-features.ts — the browser log-mel must match librosa (reference frames in meta.json)
import { readFileSync } from 'node:fs'
import { logMel, selftestSignal } from '../src/engine/features.ts'

const dir = new URL('../public/models/', import.meta.url)
const meta = JSON.parse(readFileSync(new URL('meta.json', dir), 'utf8')).audio
const buf = readFileSync(new URL('audio/mel.bin', dir))
const fb = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)
const mel = logMel(selftestSignal(meta.sr), fb, meta)
let worst = 0
meta.selftest.frames.forEach((f: number, i: number) => {
  meta.selftest.logmel[i].forEach((v: number, b: number) => { worst = Math.max(worst, Math.abs(v - mel[f][b])) })
})
console.log(`frames ${mel.length}, max |log-mel diff| vs librosa ${worst.toExponential(2)}`)
if (worst > 1e-2) process.exit(1)

// ...and the whole chain (features -> TCN) must reproduce PyTorch's last logit
const ort = await import('onnxruntime-web')
const { normalise } = await import('../src/engine/features.ts')
const x = normalise(mel, meta)
const sess = await ort.InferenceSession.create(readFileSync(new URL('audio/model.onnx', dir)))
const out = await sess.run({ mel: new ort.Tensor('float32', x, [1, mel.length, meta.n_mels * 2]) })
const logits = out.logits.data as Float32Array
const gap = Math.abs(logits[logits.length - 1] - meta.selftest.last_logit)
console.log(`TCN last logit ${logits[logits.length - 1].toFixed(4)} vs PyTorch ${meta.selftest.last_logit.toFixed(4)}`)
if (gap > 0.05) process.exit(1)
