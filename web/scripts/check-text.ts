// node scripts/check-text.ts — the browser tokenizer + exported text model must match Python's probabilities
import { readFileSync } from 'node:fs'
import { PreTrainedTokenizer } from '@huggingface/transformers'
import * as ort from 'onnxruntime-web'

const dir = new URL('../public/models/text/', import.meta.url)
const tok = new (PreTrainedTokenizer as any)(JSON.parse(readFileSync(new URL('tokenizer.json', dir), 'utf8')), JSON.parse(readFileSync(new URL('tokenizer_config.json', dir), 'utf8')))
const x = "Cordoba vs Barcelona | It's with Messi. Here's Rakitay. || It's Ivan Rakitich!"
const ids: number[] = Array.from(tok.encode(x))
console.log('ids', ids.slice(0, 12).join(','))
const sess = await ort.InferenceSession.create(readFileSync(new URL('model.onnx', dir)))
const t = (a: number[]) => new ort.Tensor('int64', BigInt64Array.from(a.map(BigInt)), [1, a.length])
const out = await sess.run({ input_ids: t(ids), attention_mask: t(ids.map(() => 1)) })
console.log('P(goal)', (1 / (1 + Math.exp(-(out.events.data as Float32Array)[0]))).toFixed(3), '(Python 0.597)')

// padded batch (how the worker scores several committed phrases at once) must give the same answer
const xs = [x, 'Cordoba vs Barcelona | a | b || short']
const rows = xs.map((s) => Array.from(tok.encode(s)) as number[])
const L = Math.max(...rows.map((r) => r.length))
const pad = (r: number[], v: number) => [...r, ...Array(L - r.length).fill(v)]
const b = (a: number[][]) => new ort.Tensor('int64', BigInt64Array.from(a.flat().map(BigInt)), [a.length, L])
const ob = await sess.run({ input_ids: b(rows.map((r) => pad(r, 1))), attention_mask: b(rows.map((r) => pad(r.map(() => 1), 0))) })
const ev = ob.events.data as Float32Array
console.log('batched P(goal)', (1 / (1 + Math.exp(-ev[0]))).toFixed(3), 'second row', (1 / (1 + Math.exp(-ev[8]))).toFixed(3))
