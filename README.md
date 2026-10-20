# onnx-fp16-guard

> English · [简体中文](README.zh-CN.md)

**fp16 conversion for ONNX that halves the file without losing quality — and refuses to silently destroy your model.**

[![npm](https://img.shields.io/npm/v/onnx-fp16-guard.svg)](https://www.npmjs.com/package/onnx-fp16-guard)
[![CI](https://github.com/SharksevenRo/onnx-fp16-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/SharksevenRo/onnx-fp16-guard/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![node](https://img.shields.io/badge/node-%E2%89%A518.17-brightgreen.svg)](package.json)

```bash
npx onnx-fp16-guard inspect model.onnx
```

```
✖ 16 tensor(s) exceed the fp16 range (65504)
tensor                                           max|value|  kind                    fix
model.generator.model.5.conv1.bn_l.running_var      4.84e+5  batch-norm-running-var  rescale-batchnorm
model.generator.model.25.running_var                4.41e+5  batch-norm-running-var  rescale-batchnorm
…
  16 of 75 BatchNormalization layer(s) need attention
```

Every mainstream fp16 converter clamps those values to `65504`, prints nothing,
and hands you a model that produces garbage. This one tells you first, and fixes
it.

- **Zero dependencies** beyond the ONNX protobuf definitions — pure TypeScript,
  no Python, no onnxruntime.
- **Runs anywhere** — Node 18+, CI, or directly in the browser (the core has no
  Node built-ins).
- **Guarded by default** — `--overflow=rescale` repairs BatchNorm algebraically;
  `--overflow=error` turns conversion into a CI gate.

---

## The 13.9 dB bug

In October 2026 we were shipping a browser inpainting model (LaMa, 208 MB) and
tried the usual float16 conversion to halve the download. The result looked
broken: masked-region PSNR dropped from **29.74 dB to 15.81 dB**.

The weights were fine. The precision was fine (we reduced the mantissa to 7 bits
with no visible loss). The problem was **range**: 16 of the 606 tensors —
all of them BatchNorm `running_var` — had values up to **484,322**, far above
the largest finite fp16 value, **65,504**.

```
model.5.conv1.bn_l.running_var = 484,322.56   → fp16 clamp → 65,504.0
1 / sqrt(var)                  is off by 2.7× → that layer's output is scaled by 2.7×
```

Nobody warned us. `onnxconverter-common` defaults to an even more aggressive
`max_finite_val=10000` (a 48× error), and its `float16` converter logs nothing.
A bare `astype(float16)` produces `inf`, which is worse: `sqrt(inf)` makes the
layer output a constant.

The fix is an algebraic identity — rescale `var`, `scale` and `epsilon` by a
power of two:

```
y = (x − μ) / √(var + ε) · γ + β
  ≡ (x − μ) / √(var/K² + ε/K²) · (γ/K) + β          K = 2^⌈log₂(max|var| / 40000)⌉

484,322 / 16² = 1,892      → fits in fp16, same function, zero extra nodes
```

| Variant                        | Masked-region PSNR | vs fp32   | Size      |
| ------------------------------ | -----------------: | --------- | --------- |
| fp32 (baseline)                |           29.74 dB | —         | 198 MiB   |
| fp16, naive converter          |       **15.81 dB** | −13.93 dB | 102.8 MiB |
| fp16, 16 BatchNorms frozen     |           29.73 dB | −0.01 dB  | 102.8 MiB |
| **fp16, rescaled (this tool)** |       **29.73 dB** | −0.01 dB  | 102.8 MiB |

Full write-up: [docs/case-study-lama.md](docs/case-study-lama.md) ·
[why fp16 breaks models](docs/why-fp16-breaks.md).

## Install

```bash
npm install onnx-fp16-guard          # library
npx onnx-fp16-guard --help           # CLI, no install
```

Node 18.17+. ESM. Works in browsers and edge runtimes.

## CLI

```bash
# What is in this model, and what would break?
onnx-fp16-guard inspect model.onnx

# Convert (rescales overflowing BatchNorms, keeps I/O fp32)
onnx-fp16-guard convert model.onnx -o model.fp16.onnx

# Convert, then prove the output still matches the original
npm install --no-save onnxruntime-node
onnx-fp16-guard verify model.onnx model.fp16.onnx
```

| Flag                  | Meaning                                                                   |
| --------------------- | ------------------------------------------------------------------------- |
| `--overflow=rescale`  | **default** — fix overflowing BatchNorm layers algebraically              |
| `--overflow=block`    | keep the affected layers in fp32 (`Cast` boundaries, no data loss)        |
| `--overflow=error`    | abort on any tensor above 65504 — use this in CI                          |
| `--fail-on-overflow`  | `inspect` exits 2 when a tensor does not fit                              |
| `--block-op <op>`     | keep an op type in fp32 (repeatable; `Range`/`Resize`/`TopK` are default) |
| `--block-node <glob>` | keep matching nodes in fp32 (repeatable, `*` and `/regex/` supported)     |
| `--no-keep-io`        | also make graph inputs/outputs fp16                                       |

Everything else is in [docs/cli.md](docs/cli.md).

## API

```ts
import { readFileSync, writeFileSync } from 'node:fs';
import { convertBytes, inspectModel, decodeModel } from 'onnx-fp16-guard';

const source = readFileSync('model.onnx');

// 1. Look before you leap (no output written, model not mutated)
const report = inspectModel(decodeModel(source));
if (report.overflow.findings.length > 0) {
  console.log(report.overflow.findings.map((f) => `${f.tensor}: ${f.absMax} (${f.fix})`));
}

// 2. Convert
const { bytes, report: conversion } = convertBytes(source, {
  overflow: 'rescale', // 'rescale' | 'block' | 'error'
  keepIoTypes: true, // graph I/O stays fp32
  blockOps: ['Resize'], // extra ops to keep in fp32
});

writeFileSync('model.fp16.onnx', bytes);
console.log(`${conversion.inputBytes} → ${conversion.outputBytes} bytes`);
```

Browser, same API:

```ts
import { convertBytes } from 'onnx-fp16-guard';

const bytes = new Uint8Array(await file.arrayBuffer());
const { bytes: fp16, report } = convertBytes(bytes);
console.log(`saved ${report.bytesSaved} bytes`);
```

API reference: [docs/api.md](docs/api.md).

## What it does

1. **Inspects** every float tensor — initializers _and_ `Constant` node payloads
   — for values outside `[−65504, 65504]`.
2. **Repairs** overflowing BatchNorm layers with the power-of-two rescale above
   (or freezes them in fp32, or refuses — your call).
3. **Converts** float32 → float16 with round-to-nearest-even, bit-exact against
   NumPy (`astype(np.float16)`), including subnormals and overflow-to-infinity.
4. **Keeps the graph valid**: retargets existing `Cast(to=float32)` nodes,
   inserts fp16 ⇄ fp32 `Cast`s around blocked ops, respects int64/bool tensors,
   topological-sorts, and structurally validates the result.
5. **Reports** exactly what changed so a review can see it — no silent clamps.

## When to use it

**Use it** when the model runs on a runtime with fp16 kernels (WebGPU, CUDA,
CoreML, DirectML) and download size matters. Half the bytes, same output.

**Be careful** with WASM: onnxruntime-web's WASM backend has no native fp16
compute, so an fp16 model is _slower_ there (it inserts casts and runs fp32
math). Keep an fp32 copy for CPU-only devices — or benchmark first.

**Don't use it** on models whose weights legitimately need values above 65504
and are not BatchNorm statistics: the tool will report them and refuse
(`--overflow=rescale`), keep them fp32 (`--overflow=block`), or stop CI
(`--overflow=error`). It never clamps silently.

## Validation

The test suite checks the converter against ground truth, not against itself:

- 267 reference vectors generated by NumPy's `astype(np.float16)` —
  random magnitudes, subnormals, ties, `inf`, `NaN` — plus an exhaustive
  round-trip over **all 65,536 fp16 bit patterns**.
- BatchNorm rescale is verified numerically: the rescaled fp32 layer is
  identical to the original within 1e-5 before rounding; after fp16 storage the
  error stays below 0.01 absolute on O(1) inputs.
- End-to-end on **LaMa 208 MB**: 606/606 initializers converted, 16 BatchNorms
  rescaled, 101.98 MB of fp16 weights (identical to the model we ship), loads and
  runs in ONNX Runtime 1.30, and agrees with the shipped fp16 model at
  **66.3 dB** (masked region) / **76.7 dB** (full frame) on identical input.

```bash
npm test                          # 36 tests, no network, < 1 s
node scripts/bench.mjs model.onnx # sizes + timings (see BENCHMARKS.md)
```

## Docs

| Document                                      | What it covers                                         |
| --------------------------------------------- | ------------------------------------------------------ |
| [why-fp16-breaks.md](docs/why-fp16-breaks.md) | The range-vs-precision diagnosis, with evidence        |
| [how-it-works.md](docs/how-it-works.md)       | Graph surgery, dtype inference, Cast placement         |
| [cli.md](docs/cli.md)                         | Every flag and exit code                               |
| [api.md](docs/api.md)                         | Types, functions, options                              |
| [browser.md](docs/browser.md)                 | Converting in the browser, and running fp16 ONNX there |
| [case-study-lama.md](docs/case-study-lama.md) | The 208 MB inpainting model, end to end                |
| [BENCHMARKS.md](BENCHMARKS.md)                | How to reproduce the numbers                           |
| [faq.md](docs/faq.md)                         | Short answers                                          |

## FAQ

**Does it need Python or onnxruntime?**
No. Conversion is pure TypeScript on the protobuf bytes. Only the optional
`verify` command needs `onnxruntime-node`.

**Does it quantize to int8 too?**
No — fp16 only. Int8 needs calibration data and a different set of trade-offs.

**Will it make my model faster?**
On WebGPU/CUDA/CoreML: usually, because there is half the memory traffic. On
WASM/CPU: no, it will be slower. Measure.

**What if the model is already fp16?**
`inspect` reports the fp16 payload, `convert` is a no-op.

**Why the name?**
It guards the one conversion step every other tool performs silently.

## License

[Apache-2.0](LICENSE). Built and extracted from the
[ClearPix](https://clearpix.org) browser photo editor, where
the same code keeps a 208 MB inpainting model at 102 MB without losing quality.
